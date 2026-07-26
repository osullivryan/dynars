//! How fast can we validate a **GB-scale** multi-file deck? Generates a deck
//! with real cross-references (elements → parts/nodes, parts → sections/
//! materials), then measures rule-evaluation throughput and how the reference
//! check scales with cores.
//!
//!     cargo run --release --example validate_throughput -- [n_files] [per_file]
//!
//! Defaults to ~2 GB (128 files × 128k nodes+elements). The generated deck is
//! reused if it already exists, so re-runs skip regeneration.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

use dynars::deck::parse_deck;
use dynars::keywords::names;
use dynars::validate::{Cmp, Rule, Value, pred};

fn main() {
    let mut args = std::env::args().skip(1);
    let n_files: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let per_file: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128_000);
    let n_parts = 1_000usize;
    let n_sections = 100usize;
    let n_materials = 100usize;

    let dir = std::env::temp_dir().join(format!("dynars_validate_throughput_{n_files}x{per_file}"));
    let root = dir.join("root.k");
    let last = dir.join(format!("mesh_{}.k", n_files - 1));

    // ── generate (skip if a matching deck is already on disk) ───────────────
    if !last.exists() {
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        println!(
            "generating {} files ({} nodes + {} elements each)…",
            n_files + 1,
            per_file,
            per_file
        );
        let t = Instant::now();

        let f = File::create(&root).unwrap();
        let mut w = BufWriter::with_capacity(1 << 20, f);
        writeln!(w, "*KEYWORD").unwrap();
        // One entity per block — LS-DYNA registers *PART / *MAT / *SECTION per
        // block, so a single block with many entities would leave all but the
        // first "undefined" and every reference to them would (falsely) dangle.
        for mid in 1..=n_materials {
            writeln!(w, "*MAT_ELASTIC").unwrap();
            writeln!(
                w,
                "{:>10}{:>10}{:>10}{:>10}",
                mid, "7.85e-9", "210000.0", "0.3"
            )
            .unwrap();
        }
        for sid in 1..=n_sections {
            writeln!(w, "*SECTION_SHELL").unwrap();
            writeln!(w, "{:>10}{:>10}{:>10}{:>10}", sid, 16, "1.0", 5).unwrap();
        }
        for pid in 1..=n_parts {
            let secid = (pid % n_sections) + 1;
            let mid = (pid % n_materials) + 1;
            writeln!(w, "*PART").unwrap();
            writeln!(w, "part {pid}").unwrap();
            writeln!(w, "{:>10}{:>10}{:>10}", pid, secid, mid).unwrap();
        }
        for f in 0..n_files {
            writeln!(w, "*INCLUDE").unwrap();
            writeln!(w, "mesh_{f}.k").unwrap();
        }
        writeln!(w, "*END").unwrap();
        w.flush().unwrap();

        for f in 0..n_files {
            let file = File::create(dir.join(format!("mesh_{f}.k"))).unwrap();
            let mut w = BufWriter::with_capacity(1 << 20, file);
            // Global sequential ids so every id fits LS-DYNA's 8-col field
            // (n_files*per_file must stay ≤ 99,999,999).
            let base = f * per_file;
            writeln!(w, "*KEYWORD").unwrap();
            writeln!(w, "*NODE").unwrap();
            for i in 0..per_file {
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
            for i in 0..per_file {
                let eid = base + i + 1;
                let pid = (i % n_parts) + 1;
                let n1 = base + (i % per_file) + 1;
                let n2 = base + ((i + 1) % per_file) + 1;
                let n3 = base + ((i + 2) % per_file) + 1;
                let n4 = base + ((i + 3) % per_file) + 1;
                writeln!(
                    w,
                    "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
                    eid, pid, n1, n2, n3, n4
                )
                .unwrap();
            }
            writeln!(w, "*END").unwrap();
            w.flush().unwrap();
            if (f + 1) % 16 == 0 {
                println!("  … {}/{} files", f + 1, n_files);
            }
        }
        println!("generated in {:.1}s", t.elapsed().as_secs_f64());
    } else {
        println!("reusing deck at {}", dir.display());
    }

    // ── parse ───────────────────────────────────────────────────────────────
    println!("parsing…");
    let t = Instant::now();
    let deck = parse_deck(&root).unwrap();
    let parse_s = t.elapsed().as_secs_f64();
    let gb = deck.total_bytes() as f64 / 1e9;
    let threads = rayon::current_num_threads();

    let n_elements = n_files * per_file;
    let n_nodes = n_files * per_file;
    // Reference lookups the connectivity pass performs: pid + 4 nodes per
    // element, secid + mid per part.
    let conn_checks = n_elements * 5 + n_parts * 2;
    let plain_checks = n_parts * 2;

    println!("\n=== deck ===");
    println!(
        "  files:    {} (1 root + {} mesh)   size: {:.2} GB",
        n_files + 1,
        n_files,
        gb
    );
    println!("  entities: {n_elements} elements, {n_nodes} nodes, {n_parts} parts");
    println!(
        "  parse_deck: {:.2} s ({:.0} MB/s)",
        parse_s,
        gb * 1e3 / parse_s
    );
    println!("  rayon threads: {threads}");

    // The first validate is "cold": it builds the defined-id index *and* runs
    // the check. Later ones reuse the cached index, so `cold - warm check`
    // isolates the one-time index build.
    let wt = Instant::now();
    let _ = deck.validate([Rule::references_resolve_with_connectivity()]);
    let cold_s = wt.elapsed().as_secs_f64();

    let best = |make: &dyn Fn() -> Vec<Rule>, iters: usize| -> f64 {
        let mut b = f64::MAX;
        for _ in 0..iters {
            let rs = make();
            let t = Instant::now();
            let _ = deck.validate(rs);
            b = b.min(t.elapsed().as_secs_f64());
        }
        b
    };

    let plain = best(&|| vec![Rule::references_resolve()], 5);
    let conn = best(&|| vec![Rule::references_resolve_with_connectivity()], 3);
    let fields = best(
        &|| {
            vec![
                Rule::keyword_forbidden(names::MAT_RIGID),
                Rule::field_forbidden_values(names::SECTION_SHELL, "SECID", [Value::Int(999)]),
                Rule::field_required(
                    names::SECTION_SHELL,
                    Some(pred("NIP", Cmp::Ge, Value::Int(3))),
                    pred("ELFORM", Cmp::Eq, Value::Int(16)),
                ),
                Rule::field_forbidden_values(names::MAT_ELASTIC, "PR", [Value::Float(0.5)]),
            ]
        },
        20,
    );

    println!("\n=== validation throughput (best-of, index cached) ===");
    println!(
        "  index build (defs, one-time)         : {:8.1} ms",
        (cold_s - conn).max(0.0) * 1e3
    );
    println!(
        "  cold validate (index + connectivity) : {:8.1} ms",
        cold_s * 1e3
    );
    println!(
        "  references_resolve (no connectivity) : {:8.1} ms   ({} ref lookups)",
        plain * 1e3,
        plain_checks
    );
    println!(
        "  references_resolve_with_connectivity : {:8.1} ms   ({:.1} M lookups, {:.0} M/s)",
        conn * 1e3,
        conn_checks as f64 / 1e6,
        conn_checks as f64 / conn / 1e6
    );
    println!(
        "  field rules (4 rules)                : {:8.1} ms",
        fields * 1e3
    );

    // ── parallelism: does the reference check scale with cores? ─────────────
    let time_in_pool = |n: usize, iters: usize| -> f64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        pool.install(|| {
            let mut b = f64::MAX;
            for _ in 0..iters {
                let t = Instant::now();
                let _ = deck.validate([Rule::references_resolve_with_connectivity()]);
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        })
    };
    let one = time_in_pool(1, 2);
    let many = time_in_pool(threads, 3);

    println!("\n=== parallelism (connectivity check) ===");
    println!(
        "   1 thread  : {:8.1} ms   ({:.0} M ref/s)",
        one * 1e3,
        conn_checks as f64 / one / 1e6
    );
    println!(
        "  {:>2} threads : {:8.1} ms   ({:.0} M ref/s)",
        threads,
        many * 1e3,
        conn_checks as f64 / many / 1e6
    );
    println!("  speedup    : {:.1}x on {} cores", one / many, threads);
    println!(
        "\n  Reference/connectivity checks fan out over files (par_iter), so they\n  \
         scale with cores; multiple rules also run in parallel. The remaining gap:\n  \
         a single per-keyword field rule scans its occurrences sequentially."
    );
}
