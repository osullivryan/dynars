//! Benchmark deck validation speed.
//! Usage: cargo run --release --example validate_bench -- <root.k>
use dynars::deck::parse_deck;
use dynars::keywords::names;
use dynars::validate::{Cmp, Rule, Value, pred};
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: validate_bench <root.k>");
    let threads = rayon::current_num_threads();

    // parse the whole deck once (the dominant cost) — shared core primitive.
    let n = 8;
    let mut best_open = f64::MAX;
    let mut deck = None;
    for _ in 0..n {
        let t = Instant::now();
        let d = parse_deck(std::path::Path::new(&path)).unwrap();
        best_open = best_open.min(t.elapsed().as_secs_f64());
        deck = Some(d);
    }
    let deck = deck.unwrap();
    let mb = deck.total_bytes() as f64 / 1e6;

    // rule evaluation over the parsed deck
    let rules = vec![
        Rule::keyword_forbidden(names::MAT_ADD_EROSION),
        Rule::keyword_forbidden(names::MAT_RIGID),
        Rule::field_forbidden_values(names::SECTION_SHELL, "SECID", [Value::Int(999)]),
        Rule::field_required(
            names::SECTION_SHELL,
            Some(pred("NIP", Cmp::Ge, Value::Int(3))),
            pred("ELFORM", Cmp::Eq, Value::Int(16)),
        ),
        Rule::include_missing(),
    ];
    let mut best_run = f64::MAX;
    for _ in 0..50 {
        let t = Instant::now();
        let _ = deck.validate(rules.clone());
        best_run = best_run.min(t.elapsed().as_secs_f64());
    }

    println!("deck: {:.1} MB   rayon threads: {}", mb, threads);
    println!(
        "  parse deck  : {:8.2} ms   ({:7.0} MB/s)",
        best_open * 1e3,
        mb / best_open
    );
    println!(
        "  rule eval   : {:8.3} ms   (5 rules, over parsed deck)",
        best_run * 1e3
    );
    println!("  end-to-end  : {:8.2} ms", (best_open + best_run) * 1e3);
}
