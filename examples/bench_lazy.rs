//! Measure the cost of eager int/float/str conversion vs. leaving fields as
//! bytes at read time.
//!
//!     cargo run --release --example bench_lazy
//!
//! Three timings on a large *NODE deck:
//!   A. block split only            (parse_file_blocks — the unavoidable "read")
//!   B. full schema parse           (split + convert to typed columns, today)
//!   C. tokenize to byte slices     (split fields, NO conversion — the proposal)

use std::hint::black_box;
use std::time::Instant;

use dynars::parser::{parse_file_blocks, Field};
use dynars::schema::{parse_schema, Card, Schema};

fn time<T>(label: &str, iters: u32, mut f: impl FnMut() -> T) -> f64 {
    // warm up
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let per = start.elapsed().as_secs_f64() / iters as f64;
    println!("  {label:<40} {:>9.2} ms/iter", per * 1000.0);
    per
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);

    // Build an N-node deck (fixed-width I8 + three E16, like real *NODE).
    let mut deck = String::from("*KEYWORD\n*NODE\n");
    deck.reserve(n * 56);
    for i in 0..n {
        let x = (i as f64) * 0.5;
        let y = (i as f64) * 0.25 + 1.0;
        let z = (i as f64) * -0.125;
        deck += &format!("{:>8}{:>16.6}{:>16.6}{:>16.6}\n", i + 1, x, y, z);
    }
    deck += "*END\n";

    let path = std::env::temp_dir().join("dynars_bench_lazy.k");
    std::fs::write(&path, &deck).unwrap();
    let bytes = deck.len();
    println!(
        "{} nodes, {:.1} MB deck\n",
        n,
        bytes as f64 / 1_048_576.0
    );

    let iters = 20;

    // A. Block split only — the read we can't avoid (memmap + find *KEYWORD lines).
    let a = time("A. block split (parse_file_blocks)", iters, || {
        parse_file_blocks(&path).unwrap()
    });

    // B. Full schema parse: split fields AND convert to i64/f64 columns (today).
    let schema = Schema::new("NODE")
        .card(Card::new().int("nid", 8).float("x", 16).float("y", 16).float("z", 16));
    let parsed = parse_file_blocks(&path).unwrap();
    let b = time("B. full parse (split + convert)", iters, || {
        parse_schema(&parsed, &schema)
    });

    // --- Single-core micro-bench: isolate slicing vs. numeric conversion. ---
    // Both loops do the SAME fixed-width slicing the real parser does (I8 + 3xE16
    // at offsets 0/8/24/40); the only difference is whether we call lexical to
    // turn each slice into an i64/f64. Single-threaded so parallelism isn't a
    // confound — the ratio is what matters, and it holds per core.
    let widths = [(0usize, 8usize), (8, 16), (24, 16), (40, 16)];

    let s_bytes = time("D. slice only, 1 core (no convert)", iters, || {
        let mut acc = 0usize;
        for block in &parsed.blocks {
            for line in parsed.body(block).split(|&ch| ch == b'\n') {
                if line.len() < 8 {
                    continue;
                }
                for &(off, w) in &widths {
                    let s = if off >= line.len() { &[][..] } else { &line[off..(off + w).min(line.len())] };
                    acc = acc.wrapping_add(black_box(s).len());
                }
            }
        }
        acc
    });

    let s_convert = time("E. slice + convert, 1 core (i64/f64)", iters, || {
        let mut isum = 0i64;
        let mut fsum = 0f64;
        for block in &parsed.blocks {
            for line in parsed.body(block).split(|&ch| ch == b'\n') {
                if line.len() < 8 {
                    continue;
                }
                for (k, &(off, w)) in widths.iter().enumerate() {
                    let s = if off >= line.len() { &[][..] } else { &line[off..(off + w).min(line.len())] };
                    let f = Field { raw: s };
                    if k == 0 {
                        isum = isum.wrapping_add(f.as_i64().unwrap_or(0));
                    } else {
                        fsum += f.as_f64().unwrap_or(0.0);
                    }
                }
            }
        }
        (isum, fsum as i64)
    });

    println!();
    println!("=== read throughput (parallel, real code paths) ===");
    println!("  full parse (B)            : {:>8.1} M nodes/s   ({:.2} ms)", n as f64 / b / 1e6, b * 1000.0);
    println!("  block split only (A)      : {:>8.1} M nodes/s   ({:.2} ms)", n as f64 / a / 1e6, a * 1000.0);
    println!("  => if we stop read at the block index and convert on demand,");
    println!("     read is {:.1}x faster ({:.1} ms -> {:.1} ms).", b / a, b * 1000.0, a * 1000.0);
    println!();
    println!("=== where the per-field work goes (single core) ===");
    println!("  slice + convert (E)       : {:>8.2} ms", s_convert * 1000.0);
    println!("  slice only     (D)        : {:>8.2} ms", s_bytes * 1000.0);
    println!("  => numeric conversion is {:.0}% of the per-field CPU work", (s_convert - s_bytes) / s_convert * 100.0);
    println!("     (leaving fields as bytes is {:.1}x cheaper per field).", s_convert / s_bytes);

    std::fs::remove_file(&path).ok();
}
