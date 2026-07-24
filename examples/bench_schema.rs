//! Profile schema-driven parsing vs the hardcoded columnar parsers.
//!
//!     cargo run --release --example bench_schema
//!
//! The hardcoded `parse_nodes` / `parse_element_shell` are hand-specialized;
//! the schema versions interpret a user-defined layout. This measures the cost
//! of that generality on 5M nodes / elements.

use std::hint::black_box;
use std::time::Instant;

use dynars::bulk::parse_nodes;
use dynars::parser::parse_file_blocks;
use dynars::schema::{parse_schema, Card, Schema};
use dynars::testgen;

fn best<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    let dir = "/tmp/dynars_bench_schema";
    let n = 5_000_000;
    testgen::generate_test_files(0, 0, n, dir);
    let path = format!("{}/root.k", dir);
    let parsed = parse_file_blocks(std::path::Path::new(&path)).unwrap();

    let node_schema = Schema::new("NODE")
        .card(Card::new().int("nid", 8).float("x", 16).float("y", 16).float("z", 16));

    // Warm.
    black_box(parse_nodes(&parsed));
    black_box(parse_schema(&parsed, &node_schema));

    let t_hard = best(5, || {
        black_box(parse_nodes(&parsed));
    });
    let t_schema = best(5, || {
        black_box(parse_schema(&parsed, &node_schema));
    });

    println!("*NODE — {} nodes\n", n);
    println!(
        "  hardcoded parse_nodes:  {:6.1} ms  ({:6.1} M nodes/s)",
        t_hard * 1e3,
        n as f64 / t_hard / 1e6
    );
    println!(
        "  schema    parse_schema: {:6.1} ms  ({:6.1} M nodes/s)   {:.2}x hardcoded",
        t_schema * 1e3,
        n as f64 / t_schema / 1e6,
        t_schema / t_hard
    );

    std::fs::remove_dir_all(dir).ok();
}
