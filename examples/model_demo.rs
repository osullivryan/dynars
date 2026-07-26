//! Resolve a deck and report definition counts + validation findings.
//! Usage: cargo run --release --example model_demo -- <main.k>
use dynars::deck::parse_deck;
use dynars::validate::Rule;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let deck = parse_deck(std::path::Path::new(&path)).unwrap();
    println!("definitions: {:?}", deck.definition_counts());

    let report = deck.validate([Rule::include_missing(), Rule::references_resolve()]);
    println!("findings: {}", report.findings.len());
    for f in report.findings.iter().take(12) {
        println!("  {}  {}", f.location(), f.message);
    }
}
