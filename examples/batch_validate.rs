//! Validate several decks at once against one shared file cache: common
//! `*INCLUDE`s (mesh, materials, sections) are read and parsed exactly once no
//! matter how many decks pull them in, and the definition-index extraction on
//! those shared files is reused across decks too.
//!
//! Usage: cargo run --example batch_validate -- <a/main.k> <b/main.k> ...

use std::path::PathBuf;

use dynars::Workspace;
use dynars::deck::Deck;
use dynars::validate::{Rule, Severity};

fn main() {
    let roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: batch_validate <main.k> [<main2.k> ...]");
        std::process::exit(2);
    }

    // One workspace shared across every deck. Decks parse sequentially, so each
    // reuses the files (and cached id indices) the earlier ones already read.
    let ws = Workspace::new();

    let mut roots_ok: Vec<PathBuf> = Vec::new();
    let mut decks: Vec<Deck> = Vec::new();
    for (root, result) in ws.parse_decks(&roots) {
        match result {
            Ok(d) => {
                roots_ok.push(root);
                decks.push(d);
            }
            Err(e) => println!("{}: parse failed — {e}", root.display()),
        }
    }

    // Validate every deck in parallel off the shared cache — a shared mesh's
    // parse, id index, and connectivity index are each built once, not per deck.
    let rules = [
        Rule::references_resolve_with_connectivity(),
        Rule::duplicate_ids(),
        // A missing `*INCLUDE` is never parsed, so it adds no file and no cached
        // content — this rule is what surfaces it. Don't rely on
        // `references_resolve` alone: if the missing file was the only source of
        // an entity kind, references to it are left unflagged (conservative).
        Rule::include_missing().with_severity(Severity::Warning),
    ];
    let reports = ws.validate_decks(&decks, rules);

    for ((root, deck), report) in roots_ok.iter().zip(&decks).zip(&reports) {
        println!(
            "{}: {} file(s), {} error(s), {} warning(s)",
            root.display(),
            deck.files.len(),
            report.count(Severity::Error),
            report.count(Severity::Warning),
        );
        for f in report.findings.iter().take(10) {
            println!("  [{:?}] {} — {}", f.severity, f.location(), f.message);
        }
    }

    // What the sharing bought: distinct files read vs. served from cache.
    let s = ws.stats();
    let total = s.files_parsed + s.files_reused;
    println!(
        "\ncache: {} file(s) read from disk, {} reuse(s) across {} decks \
         ({:.0}% of file touches avoided a re-read)",
        s.files_parsed,
        s.files_reused,
        decks.len(),
        if total == 0 {
            0.0
        } else {
            100.0 * s.files_reused as f64 / total as f64
        },
    );
}
