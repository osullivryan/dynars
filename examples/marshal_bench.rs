//! `*NODE` marshalling benchmark — the columnar (→ typed arrays) extraction
//! path, swept up to 250 M nodes. Runs in its own process (no other stages
//! competing for cache/RAM) and writes `assets/bench_marshal.csv` for
//! `scripts/plot_bench.py`.
//!
//!     cargo run --release --example marshal_bench
//!
//! Node-only, free-format decks are generated fresh per point and deleted right
//! after measuring, so peak disk stays at a single deck. Node extraction scales
//! with node count, not block count, so it gets its own axis (a 250 M-node deck
//! would be ~18 GB as individual keyword blocks).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use dynars::deck::parse_deck;
use dynars::schema::parse_schema_files;

fn make_deck(label: &str, n_files: usize, total: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dynars_marshalbench_{label}_{total}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = dir.join("root.k");
    let per = (total / n_files).max(1);
    let write_nodes = |w: &mut BufWriter<File>, base: usize, count: usize| {
        writeln!(w, "*KEYWORD\n*NODE").unwrap();
        for i in 0..count {
            writeln!(w, "{},0.0,0.0,0.0", base + i + 1).unwrap();
        }
        writeln!(w, "*END").unwrap();
    };
    if n_files == 1 {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&root).unwrap());
        write_nodes(&mut w, 0, total);
        w.flush().unwrap();
    } else {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&root).unwrap());
        writeln!(w, "*KEYWORD").unwrap();
        for f in 0..n_files {
            writeln!(w, "*INCLUDE\nmesh_{f}.k").unwrap();
        }
        writeln!(w, "*END").unwrap();
        w.flush().unwrap();
        for f in 0..n_files {
            let mut w = BufWriter::with_capacity(
                1 << 20,
                File::create(dir.join(format!("mesh_{f}.k"))).unwrap(),
            );
            write_nodes(&mut w, f * per, per);
            w.flush().unwrap();
        }
    }
    root
}

fn main() {
    let schema = dynars::keywords::schema("NODE").unwrap();
    // Geometric, non-linear at the high end, up to 140 M — the largest that stays
    // clear of the 16 GB RAM ceiling here (~9 GB of columns). Past ~150 M the
    // columns + input start hitting swap and the numbers stop reflecting the
    // algorithm.
    let sizes = [
        2_000_000usize,
        10_000_000,
        30_000_000,
        70_000_000,
        100_000_000,
        140_000_000,
    ];
    let configs: [(&str, usize); 2] = [("monolithic", 1), ("flat", 256)];

    eprintln!(
        "marshalling sweep, {} threads",
        rayon::current_num_threads()
    );
    let mut csv = String::from("shape,files,nodes,marshal_s\n");
    for (label, nf) in configs {
        for total in sizes {
            eprint!("  {label:>10} {:>4} M: gen… ", total / 1_000_000);
            let t = Instant::now();
            let root = make_deck(label, nf, total);
            eprint!("{:.1}s  ", t.elapsed().as_secs_f64());

            // Best of N; fewer iterations for the huge (swap-bound) points.
            let iters = if total <= 100_000_000 { 3 } else { 2 };
            let mut best = f64::MAX;
            {
                let deck = parse_deck(&root).unwrap();
                for _ in 0..iters {
                    let t = Instant::now();
                    let table = parse_schema_files(&deck.files, &schema);
                    best = best.min(t.elapsed().as_secs_f64());
                    std::hint::black_box(&table);
                }
            }
            eprintln!(
                "{:>6.0} ms  →  {:>4.0} M nodes/s",
                best * 1e3,
                total as f64 / best / 1e6
            );
            csv.push_str(&format!("{label},{nf},{total},{best:.6}\n"));
            let _ = fs::remove_dir_all(root.parent().unwrap()); // bound peak disk
        }
    }
    fs::write("assets/bench_marshal.csv", &csv).unwrap();
    println!("wrote assets/bench_marshal.csv");
}
