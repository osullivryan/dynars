//! Pure-Rust marshalling benchmark — no Python, no numpy, no FFI.
//!
//! Run with: cargo run --release --example bench_marshal

use std::time::Instant;

use dynars::bulk::parse_nodes;
use dynars::parser::parse_file_blocks;
use dynars::testgen;

fn best<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let dir = "/tmp/dynars_bench_rs";
    let nodes = 5_000_000;
    testgen::generate_test_files(0, 0, nodes, dir);
    let path = format!("{}/root.k", dir);
    let bytes = std::fs::metadata(&path).unwrap().len();
    let mb = bytes as f64 / 1_048_576.0;
    println!("file: {:.1} MB, {} nodes\n", mb, nodes);

    // Warm the page cache.
    let _ = std::fs::read(&path).unwrap();

    // 1) read + block split
    let t_blocks = best(5, || {
        let p = parse_file_blocks(std::path::Path::new(&path)).unwrap();
        std::hint::black_box(&p);
    });
    println!(
        "parse_file_blocks (read + split): {:7.1} ms  -> {:8.0} MB/s",
        t_blocks * 1000.0,
        mb / t_blocks
    );

    // 2) node parsing only (file already parsed into blocks)
    let parsed = parse_file_blocks(std::path::Path::new(&path)).unwrap();
    let t_nodes = best(5, || {
        let n = parse_nodes(&parsed);
        std::hint::black_box(&n);
    });
    let n = parse_nodes(&parsed);
    println!(
        "parse_nodes (5M nodes -> SoA):    {:7.1} ms  -> {:6.1} M nodes/s  ({} ids, {} coords)",
        t_nodes * 1000.0,
        nodes as f64 / t_nodes / 1e6,
        n.ids.len(),
        n.coords.len(),
    );

    // 3) end-to-end: read + split + node parse
    let t_all = best(5, || {
        let p = parse_file_blocks(std::path::Path::new(&path)).unwrap();
        let n = parse_nodes(&p);
        std::hint::black_box(&n);
    });
    println!(
        "end-to-end (blocks + nodes):      {:7.1} ms  -> {:8.0} MB/s",
        t_all * 1000.0,
        mb / t_all
    );

    let _ = std::fs::remove_dir_all(dir);
}
