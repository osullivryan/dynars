//! Scaling benchmark for the README figures.
//!
//! Sweeps a synthetic (but realistic, cross-referenced) deck across **deck size**
//! and **include-tree shape**, measuring each pipeline stage and writing
//! `assets/bench_scaling.csv`. Plotting is a separate step
//! (`python scripts/plot_bench.py`, matplotlib), so figures re-render from the
//! committed CSV without re-running this sweep.
//!
//! Three include shapes, each swept over a range of sizes (the plot lines):
//! * **monolithic** — one file holding everything.
//! * **flat** — root → 256 leaf files (wide, depth 1; 257 files).
//! * **deep tree** — balanced 6-ary tree, depth 3 → 216 leaves (wide *and* deep;
//!   259 files total).
//!
//! Both include shapes fan out to ~256 leaf files; flat vs deep-tree contrasts a
//! single wide level against a 3-level tree. Every leaf carries `*NODE` +
//! `*ELEMENT_SHELL` (per-line) plus one `*BOUNDARY_SPC_NODE` per node;
//! parts/sections/materials live in the root and are cross-referenced from every
//! leaf.
//!
//!     cargo run --release --example bench_scaling
//!
//! Progress prints to **stderr** (unbuffered) so it streams live; don't pipe it
//! through `tail`, which only emits at EOF. Generated decks are cached in the
//! temp dir and reused across runs.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use dynars::deck::parse_deck;
use dynars::include::build_include_tree;
use dynars::schema::parse_schema_files;
use dynars::validate::{Rule, Value};

const N_PARTS: usize = 100;
const N_SECTIONS: usize = 50;
const N_MATERIALS: usize = 50;

/// One include-tree shape: `branch^depth` leaf files (depth 0 ⇒ a single
/// monolithic file). `label` names the plot line.
#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    branch: usize,
    depth: u32,
}

const SHAPES: [Shape; 3] = [
    Shape {
        label: "monolithic",
        branch: 1,
        depth: 0,
    },
    Shape {
        label: "flat",
        branch: 256,
        depth: 1,
    }, // 256 leaves, depth 1 (wide)
    Shape {
        label: "deep-tree",
        branch: 6,
        depth: 3,
    }, // 216 leaves, depth 3 (wide + deep)
];

impl Shape {
    fn leaves(&self) -> usize {
        self.branch.pow(self.depth).max(1)
    }
    fn files(&self) -> usize {
        (0..=self.depth).map(|l| self.branch.pow(l)).sum()
    }
}

/// Write one leaf's mesh: `m` nodes, `m` shells (each → 4 nodes + a part), and
/// `m` SPC constraints (each → a node). Ids are offset by `base` so leaves don't
/// collide once flattened.
fn write_mesh(w: &mut impl Write, base: usize, m: usize) {
    writeln!(w, "*NODE").unwrap();
    for i in 0..m {
        let nid = base + i + 1;
        let x = i as f64 * 1.5;
        writeln!(
            w,
            "{:>8}{:>16.6}{:>16.6}{:>16.6}{:>8}{:>8}",
            nid, x, x, x, 0, 0
        )
        .unwrap();
    }
    writeln!(w, "*ELEMENT_SHELL").unwrap();
    for i in 0..m {
        let eid = base + i + 1;
        let pid = (i % N_PARTS) + 1;
        let n = |o: usize| base + ((i + o) % m) + 1;
        writeln!(
            w,
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
            eid,
            pid,
            n(0),
            n(1),
            n(2),
            n(3)
        )
        .unwrap();
    }
    for i in 0..m {
        let nid = base + i + 1;
        writeln!(w, "*BOUNDARY_SPC_NODE").unwrap();
        writeln!(
            w,
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
            nid, 0, 1, 1, 1, 0, 0, 0
        )
        .unwrap();
    }
}

fn shared_defs(w: &mut impl Write) {
    for mid in 1..=N_MATERIALS {
        writeln!(w, "*MAT_ELASTIC").unwrap();
        writeln!(
            w,
            "{:>10}{:>10}{:>10}{:>10}",
            mid, "7.85e-9", "210000.0", "0.3"
        )
        .unwrap();
    }
    for sid in 1..=N_SECTIONS {
        writeln!(w, "*SECTION_SHELL").unwrap();
        writeln!(w, "{:>10}{:>10}{:>10}{:>10}", sid, 16, "1.0", 5).unwrap();
    }
    for pid in 1..=N_PARTS {
        writeln!(w, "*PART").unwrap();
        writeln!(w, "part {pid}").unwrap();
        writeln!(
            w,
            "{:>10}{:>10}{:>10}",
            pid,
            (pid % N_SECTIONS) + 1,
            (pid % N_MATERIALS) + 1
        )
        .unwrap();
    }
}

/// Generate a deck for `shape` sized so its total keyword-block count ≈
/// `target_blocks`, laid out as an include tree in one directory. Cached.
fn gen_deck(shape: Shape, target_blocks: usize) -> PathBuf {
    let leaves = shape.leaves();
    let m = (target_blocks / leaves).max(1); // mesh entities per leaf
    let dir = std::env::temp_dir().join(format!("dynars_bigbench_{}_{target_blocks}", shape.label));
    let root = dir.join("root.k");
    // sentinel: the last leaf at the deepest level.
    let sentinel = if shape.depth == 0 {
        root.clone()
    } else {
        dir.join(format!("f{}_{}.k", shape.depth, leaves - 1))
    };
    if sentinel.exists() {
        eprint!("reuse… ");
        return root;
    }
    eprint!("gen({}, m={m})… ", shape.label);
    let t = Instant::now();
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    for level in 0..=shape.depth {
        let count = shape.branch.pow(level);
        for i in 0..count {
            let name = if level == 0 {
                "root.k".to_string()
            } else {
                format!("f{level}_{i}.k")
            };
            let f = File::create(dir.join(&name)).unwrap();
            let mut w = BufWriter::with_capacity(1 << 20, f);
            writeln!(w, "*KEYWORD").unwrap();
            if level == 0 {
                shared_defs(&mut w); // parts/sections/materials referenced by every leaf
            }
            if level < shape.depth {
                for j in 0..shape.branch {
                    writeln!(w, "*INCLUDE").unwrap();
                    writeln!(w, "f{}_{}.k", level + 1, i * shape.branch + j).unwrap();
                }
            } else {
                write_mesh(&mut w, i * m, m); // leaf: `i` is the global leaf index
            }
            writeln!(w, "*END").unwrap();
            w.flush().unwrap();
        }
    }
    eprint!("{:.1}s ", t.elapsed().as_secs_f64());
    root
}

// ── timing ───────────────────────────────────────────────────────────────────

/// Time `f`, print the elapsed ms with a label, return the best of `iters`.
fn stage(label: &str, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    eprint!("{label} {:.0}ms  ", b * 1e3);
    b
}

#[derive(Clone)]
struct Row {
    shape: &'static str,
    files: usize,
    blocks: usize,
    bytes: usize,
    nodes: usize,
    t_include: f64,
    t_parse: f64,
    t_marshal: f64,
    t_index: f64,
    t_dangle: f64,
    t_conn: f64,
    t_field: f64,
}

fn measure(shape: Shape, target_blocks: usize, idx: usize, total: usize) -> Row {
    eprint!("[{idx}/{total}] {:>10}: ", shape.label);
    let root = gen_deck(shape, target_blocks);
    let node_schema = dynars::keywords::schema("NODE").unwrap();

    let (blocks, bytes) = {
        let d = parse_deck(&root).unwrap();
        (
            d.files.iter().map(|f| f.blocks.len()).sum(),
            d.total_bytes(),
        )
    };
    let nodes = shape.leaves() * (target_blocks / shape.leaves()).max(1);
    eprint!(
        "{}k blk {}MB {}f | ",
        blocks / 1000,
        bytes / 1_000_000,
        shape.files()
    );

    let t_include = stage("incl", 3, || {
        build_include_tree(&root).unwrap();
    });
    let t_parse = stage("parse", 3, || {
        parse_deck(&root).unwrap();
    });

    // One parsed deck for marshalling + validation; the definition index is
    // built lazily on the first `validate`, so `cold - warm` isolates it.
    let deck = parse_deck(&root).unwrap();
    let t_marshal = stage("marshal", 3, || {
        let _ = parse_schema_files(&deck.files, &node_schema);
    });
    let cold = {
        let t = Instant::now();
        let _ = deck.validate([Rule::references_resolve_with_connectivity()]);
        t.elapsed().as_secs_f64()
    };
    let t_conn = stage("conn", 3, || {
        let _ = deck.validate([Rule::references_resolve_with_connectivity()]);
    });
    let t_index = (cold - t_conn).max(0.0);
    eprint!("index {:.0}ms  ", t_index * 1e3);
    let t_dangle = stage("dangle", 3, || {
        let _ = deck.validate([Rule::references_resolve()]);
    });
    let t_field = stage("field", 3, || {
        let _ = deck.validate([Rule::field_forbidden_values(
            "BOUNDARY_SPC_NODE",
            "DOFX",
            [Value::Int(7)],
        )]);
    });
    eprintln!();

    Row {
        shape: shape.label,
        files: shape.files(),
        blocks,
        bytes,
        nodes,
        t_include,
        t_parse,
        t_marshal,
        t_index,
        t_dangle,
        t_conn,
        t_field,
    }
}

fn main() {
    // Deck sizes (≈ total keyword blocks) — the x-axis. Geometric, denser at the
    // low end for smooth curves; the top point is a ~5 M-block, ~1.1 GB deck
    // (5 M nodes + 5 M shells + 5 M constraints).
    let sizes = [
        8_000usize, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000, 2_000_000,
        3_500_000, 5_000_000,
    ];
    let total = SHAPES.len() * sizes.len();
    eprintln!(
        "scaling sweep: {total} points ({} shapes × {} sizes)\n",
        SHAPES.len(),
        sizes.len()
    );

    let mut grid: Vec<Row> = Vec::new();
    let mut idx = 0;
    for &shape in &SHAPES {
        for &sz in &sizes {
            idx += 1;
            grid.push(measure(shape, sz, idx, total));
        }
    }

    let mut csv = String::from(
        "shape,files,blocks,bytes,nodes,include_s,parse_s,marshal_s,index_s,dangle_s,conn_s,field_s\n",
    );
    for r in &grid {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            r.shape,
            r.files,
            r.blocks,
            r.bytes,
            r.nodes,
            r.t_include,
            r.t_parse,
            r.t_marshal,
            r.t_index,
            r.t_dangle,
            r.t_conn,
            r.t_field
        ));
    }
    fs::write("assets/bench_scaling.csv", &csv).unwrap();
    println!("\nwrote assets/bench_scaling.csv ({} rows)", grid.len());
    println!("marshalling: cargo run --release --example marshal_bench");
    println!("plot with:   python scripts/plot_bench.py");
}
