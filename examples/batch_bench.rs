//! Measure what a `Workspace` saves when many decks share one big `*INCLUDE`.
//!
//! Generates a mesh (`nodes` `*NODE`s + `elems` `*ELEMENT_SHELL`s over them) and
//! `decks` variant roots that each `*INCLUDE` it, then times two ways of
//! parsing+validating all of them with the same rules:
//!   naive     — `parse_deck` + `validate` per deck (re-reads/re-checks the mesh
//!               every time), and
//!   workspace — `parse_decks` + `validate_decks` (mesh read/parsed/indexed once).
//! It also asserts both produce identical findings, so this doubles as a
//! correctness check at scale.
//!
//! Usage (release matters): cargo run --release --example batch_bench -- [nodes] [elems] [decks]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dynars::Workspace;
use dynars::deck::{Deck, parse_deck};
use dynars::validate::{Finding, Rule};

fn rules() -> [Rule; 2] {
    [
        Rule::references_resolve_with_connectivity(),
        Rule::duplicate_ids(),
    ]
}

/// Sorted, comparable projection of a report's findings.
fn keys(findings: &[Finding]) -> Vec<String> {
    let mut v: Vec<String> = findings
        .iter()
        .map(|f| format!("{}|{}|{}", f.file.display(), f.line, f.message))
        .collect();
    v.sort();
    v
}

/// Write `mesh.k` (a shared mesh) + `decks` variant roots that each include it.
fn generate(nodes: usize, elems: usize, decks: usize) -> (PathBuf, Vec<PathBuf>) {
    let dir = std::env::temp_dir().join("dynars_batch_bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // mesh.k: a material/section/part, `nodes` nodes, `elems` shells over them.
    let mut mesh = String::with_capacity(nodes * 40 + elems * 40);
    mesh.push_str("*KEYWORD\n*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n");
    mesh.push_str("*SECTION_SHELL\n1,2\n*PART\np\n1,1,1\n*NODE\n");
    for i in 1..=nodes {
        let x = (i % 1000) as f64;
        writeln!(mesh, "{i},{x:.1},0.0,0.0").unwrap();
    }
    mesh.push_str("*ELEMENT_SHELL\n");
    // Each shell references 4 in-range nodes, so connectivity fully resolves.
    let span = nodes.max(4) - 3;
    for e in 1..=elems {
        let n1 = (e - 1) % span + 1;
        writeln!(mesh, "{e},1,{n1},{},{},{}", n1 + 1, n1 + 2, n1 + 3).unwrap();
    }
    let mesh_path = dir.join("mesh.k");
    std::fs::write(&mesh_path, mesh).unwrap();
    let mesh_bytes = std::fs::metadata(&mesh_path).unwrap().len();
    println!(
        "mesh: {nodes} nodes, {elems} shells, {:.1} MB; {decks} decks include it\n",
        mesh_bytes as f64 / 1e6
    );

    let roots = (0..decks)
        .map(|k| {
            let sub = dir.join(format!("v{k}"));
            std::fs::create_dir_all(&sub).unwrap();
            let root = sub.join("main.k");
            // A unique keyword per variant keeps the roots distinct files.
            std::fs::write(
                &root,
                format!("*KEYWORD\n*INCLUDE\n../mesh.k\n*PARAMETER\nR run,{k}.0\n*END\n"),
            )
            .unwrap();
            root
        })
        .collect();
    (mesh_path, roots)
}

fn main() {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (nodes, elems, decks) = (arg(1, 500_000), arg(2, 500_000), arg(3, 8));
    let (_mesh, roots) = generate(nodes, elems, decks);

    // ── Naive: parse_deck + validate per deck, from scratch each time ──────────
    let t = Instant::now();
    let naive: Vec<Deck> = roots.iter().map(|r| parse_deck(r).unwrap()).collect();
    let naive_parse = t.elapsed();
    let t = Instant::now();
    let naive_reports: Vec<_> = naive.iter().map(|d| d.validate(rules())).collect();
    let naive_validate = t.elapsed();

    // ── Workspace: shared parse + parallel validate off one cache ─────────────
    let ws = Workspace::new();
    let t = Instant::now();
    let decks_ws: Vec<Deck> = ws
        .parse_decks(&roots)
        .into_iter()
        .map(|(_, d)| d.unwrap())
        .collect();
    let ws_parse = t.elapsed();
    let t = Instant::now();
    let ws_reports = ws.validate_decks(&decks_ws, rules());
    let ws_validate = t.elapsed();

    // Correctness: identical findings, deck for deck.
    for (i, (a, b)) in naive_reports.iter().zip(&ws_reports).enumerate() {
        assert_eq!(
            keys(&a.findings),
            keys(&b.findings),
            "deck {i} findings differ"
        );
    }

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    let row = |name: &str, np: f64, nv: f64| {
        println!(
            "  {name:<10} parse {np:8.1} ms   validate {nv:8.1} ms   total {:8.1} ms",
            np + nv
        );
    };
    println!("parsed+validated {decks} decks; findings identical ✓\n");
    row("naive", ms(naive_parse), ms(naive_validate));
    row("workspace", ms(ws_parse), ms(ws_validate));
    let speed = |a: std::time::Duration, b: std::time::Duration| a.as_secs_f64() / b.as_secs_f64();
    println!(
        "\n  speedup    parse {:.1}x   validate {:.1}x   total {:.1}x",
        speed(naive_parse, ws_parse),
        speed(naive_validate, ws_validate),
        speed(naive_parse + naive_validate, ws_parse + ws_validate),
    );

    let s = ws.stats();
    println!(
        "\n  workspace cache: {} files read ({} reuses), def index built {}x, conn index built {}x\n  (naive would build each ~{}x — once per deck)",
        s.files_parsed, s.files_reused, s.def_indices_built, s.ref_indices_built, decks
    );
    let _ = Path::new("");
}
