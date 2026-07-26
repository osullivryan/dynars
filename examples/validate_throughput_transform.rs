//! Like `validate_throughput`, but every mesh file is pulled in with
//! **`*INCLUDE_TRANSFORM`** — so this measures the *offset* path: the case where
//! `collect_def_ids` and `is_dangling` shift every id by its file's per-kind
//! offset before building/probing the id sets.
//!
//! The trick that makes it realistic: each mesh file numbers its nodes and
//! elements **locally** (`1..per_file`) and the root pulls it in at a distinct
//! `IDNOFF`/`IDEOFF`, so after transform the whole deck shares one dense, global
//! id space with no collisions — the classic "instance the same mesh N times"
//! idiom. Element→node references inside a file are local too, so they only
//! resolve once the offset is applied to *both* the node defs and the refs;
//! that's exactly the wiring under test, exercised on all ~82 M lookups.
//!
//!     cargo run --release --example validate_throughput_transform -- [n_files] [per_file]
//!
//! Compare its numbers against `validate_throughput` (the transform-free deck)
//! to see the offset path's overhead. The generated deck is reused if present.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

use dynars::deck::parse_deck;
use dynars::validate::{Rule, Severity};

fn main() {
    let mut args = std::env::args().skip(1);
    let n_files: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let per_file: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128_000);
    let n_parts = 1_000usize;
    let n_sections = 100usize;
    let n_materials = 100usize;

    let dir =
        std::env::temp_dir().join(format!("dynars_validate_throughput_xform_{n_files}x{per_file}"));
    let root = dir.join("root.k");
    let last = dir.join(format!("mesh_{}.k", n_files - 1));

    // Per-file offset: file `f` is shifted so its local ids land in a unique,
    // contiguous global band. `(f + 1)` (not `f`) so *every* file — including the
    // first — carries a non-identity transform, keeping all lookups on the
    // offset path.
    let offset_of = |f: usize| (f + 1) * per_file;

    // ── generate (skip if a matching deck is already on disk) ───────────────
    if !last.exists() {
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        println!(
            "generating {} files ({} local nodes + {} elements each, INCLUDE_TRANSFORM)…",
            n_files + 1,
            per_file,
            per_file
        );
        let t = Instant::now();

        // Root: shared parts/sections/materials (not offset — IDPOFF/IDMOFF = 0),
        // then one *INCLUDE_TRANSFORM per mesh file.
        let f = File::create(&root).unwrap();
        let mut w = BufWriter::with_capacity(1 << 20, f);
        writeln!(w, "*KEYWORD").unwrap();
        for mid in 1..=n_materials {
            writeln!(w, "*MAT_ELASTIC").unwrap();
            writeln!(w, "{:>10}{:>10}{:>10}{:>10}", mid, "7.85e-9", "210000.0", "0.3").unwrap();
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
            let off = offset_of(f);
            writeln!(w, "*INCLUDE_TRANSFORM").unwrap();
            writeln!(w, "mesh_{f}.k").unwrap();
            // IDNOFF IDEOFF IDPOFF IDMOFF IDSOFF IDFOFF IDDOFF  (I10). Only nodes
            // and elements are offset; parts stay shared with the root.
            writeln!(
                w,
                "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}",
                off, off, 0, 0, 0, 0, 0
            )
            .unwrap();
            writeln!(w, "{:>10}", 0).unwrap(); // IDROFF
        }
        writeln!(w, "*END").unwrap();
        w.flush().unwrap();

        // Mesh files: purely local ids (1..=per_file), so the file itself is
        // small-numbered and reusable; the transform makes it globally unique.
        for f in 0..n_files {
            let file = File::create(dir.join(format!("mesh_{f}.k"))).unwrap();
            let mut w = BufWriter::with_capacity(1 << 20, file);
            writeln!(w, "*KEYWORD").unwrap();
            writeln!(w, "*NODE").unwrap();
            for i in 0..per_file {
                let nid = i + 1; // local
                let x = i as f64 * 1.5;
                writeln!(w, "{:>8}{:>16.6}{:>16.6}{:>16.6}{:>8}{:>8}", nid, x, x, x, 0, 0).unwrap();
            }
            writeln!(w, "*ELEMENT_SHELL").unwrap();
            for i in 0..per_file {
                let eid = i + 1; // local
                let pid = (i % n_parts) + 1; // resolves to a root part (IDPOFF=0)
                let n1 = (i % per_file) + 1; // local node refs
                let n2 = ((i + 1) % per_file) + 1;
                let n3 = ((i + 2) % per_file) + 1;
                let n4 = ((i + 3) % per_file) + 1;
                writeln!(w, "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}", eid, pid, n1, n2, n3, n4).unwrap();
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
    let conn_checks = n_elements * 5 + n_parts * 2;

    println!("\n=== deck (every include is *INCLUDE_TRANSFORM) ===");
    println!(
        "  files:    {} (1 root + {} mesh)   size: {:.2} GB",
        n_files + 1,
        n_files,
        gb
    );
    println!("  entities: {n_elements} elements, {n_nodes} nodes, {n_parts} parts");
    println!("  parse_deck: {:.2} s ({:.0} MB/s)", parse_s, gb * 1e3 / parse_s);
    println!("  rayon threads: {threads}");

    // The first validate is "cold": it builds the defined-id index (shifting
    // every def by its file's offset) *and* runs the check. It doubles as the
    // correctness proof — with the offsets applied this deck is clean, so every
    // local element→node ref and every part ref resolved through the transform.
    // A non-zero count would mean the offset path failed to line defs up with
    // refs (and would also mean the "fast" numbers were fast only by skipping
    // work).
    let wt = Instant::now();
    let report = deck.validate([Rule::references_resolve_with_connectivity()]);
    let cold_s = wt.elapsed().as_secs_f64();
    let errors = report.count(Severity::Error);
    println!(
        "  dangling refs: {errors}  ({})",
        if errors == 0 {
            "clean — all refs resolved via IDNOFF/IDEOFF"
        } else {
            "!! offset path did not resolve refs"
        }
    );
    assert_eq!(errors, 0, "transformed deck should validate clean");

    // ── throughput (offset path) ────────────────────────────────────────────

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
    // Warm the index (built lazily on first validate above), then time the check.
    let conn = best(&|| vec![Rule::references_resolve_with_connectivity()], 3);

    println!("\n=== validation throughput (offset path, index cached) ===");
    println!(
        "  index build (defs, one-time)         : {:8.1} ms",
        (cold_s - conn).max(0.0) * 1e3
    );
    println!("  cold validate (index + connectivity) : {:8.1} ms", cold_s * 1e3);
    println!(
        "  references_resolve_with_connectivity : {:8.1} ms   ({:.1} M lookups, {:.0} M/s)",
        conn * 1e3,
        conn_checks as f64 / 1e6,
        conn_checks as f64 / conn / 1e6
    );

    // ── parallelism ─────────────────────────────────────────────────────────
    let time_in_pool = |n: usize, iters: usize| -> f64 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
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

    println!("\n=== parallelism (connectivity check, offset path) ===");
    println!("   1 thread  : {:8.1} ms   ({:.0} M ref/s)", one * 1e3, conn_checks as f64 / one / 1e6);
    println!(
        "  {:>2} threads : {:8.1} ms   ({:.0} M ref/s)",
        threads,
        many * 1e3,
        conn_checks as f64 / many / 1e6
    );
    println!("  speedup    : {:.1}x on {} cores", one / many, threads);
}
