use std::time::Instant;
use dynars::deck::parse_deck;
use dynars::validate::Rule;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let p = std::path::Path::new(&path);
    let bytes = std::fs::metadata(p).unwrap().len();

    // warm the page cache
    let _ = std::fs::read(p).unwrap();

    // pure parse, best of 5
    let mut best_parse = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let deck = parse_deck(p).unwrap();
        let dt = t.elapsed().as_secs_f64();
        best_parse = best_parse.min(dt);
        std::hint::black_box(&deck);
    }

    // parse + validate (references_resolve + include_missing), best of 5
    let mut best_pv = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let deck = parse_deck(p).unwrap();
        let rep = deck.validate([Rule::include_missing(), Rule::references_resolve()]);
        let dt = t.elapsed().as_secs_f64();
        best_pv = best_pv.min(dt);
        std::hint::black_box((&deck, rep.findings.len()));
    }

    let mb = bytes as f64 / (1024.0 * 1024.0);
    println!("file: {:.1} MB", mb);
    println!("parse only        : {:.4} s   ({:.0} MB/s)", best_parse, mb / best_parse);
    println!("parse + validate  : {:.4} s   ({:.0} MB/s)", best_pv, mb / best_pv);
}
