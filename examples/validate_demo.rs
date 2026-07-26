//! Validate a real LS-DYNA deck with typed rules, combinators, scope, and a
//! custom check.
//! Usage: cargo run --example validate_demo -- <main.k>

use std::collections::HashMap;

use dynars::deck::{parse_deck, Deck};
use dynars::keywords::names;
use dynars::validate::{pred, Check, Cmp, Expr, Finding, Rule, Severity, Value};

/// A custom rule (arbitrary Rust logic): SECIDs must be unique across the deck.
/// The built-in `Rule`s are per-occurrence and can't express cross-occurrence
/// aggregation — this is what you drop to a `Check` for. It just iterates
/// `deck.keywords(...)`, the same view the built-in rules use.
struct UniqueSectionIds;
impl Check for UniqueSectionIds {
    fn name(&self) -> String {
        "custom:unique_section_ids".into()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        let mut seen: HashMap<i64, String> = HashMap::new();
        for kw in deck.keywords(names::SECTION_SHELL) {
            let Some(id) = kw.field("SECID").and_then(|f| f.as_i64()) else { continue };
            let here = format!("{}:{}", kw.file().display(), kw.line());
            if let Some(first) = seen.get(&id) {
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: "SECTION_SHELL".into(),
                    file: kw.file().to_path_buf(),
                    line: kw.line(),
                    message: format!("duplicate SECID {id} (first defined at {first})"),
                });
            } else {
                seen.insert(id, here);
            }
        }
        out
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: validate_demo <main.k>");
    let deck = parse_deck(std::path::Path::new(&path)).expect("parse deck");

    // One entry — `deck.validate([rules])` — off the deck we already parsed.
    // Built-ins and the custom check are the same currency: a `Rule`.
    let report = deck.validate([
        // 1. a material *type* that can't be used
        Rule::keyword_forbidden(names::MAT_ADD_EROSION),
        // 2. a specific id that can't be used (SECTION with SECID == 2)
        Rule::field_forbidden_values(names::SECTION_SHELL, "SECID", [Value::Int(2)]),
        // 3. every *INCLUDE must resolve on disk
        Rule::include_missing().with_severity(Severity::Warning),
        // 4. combinator (tier 2): if NIP >= 3 AND PROPT == 1, ELFORM must be 16
        Rule::field_required(
            names::SECTION_SHELL,
            Some(Expr::all([
                pred("NIP", Cmp::Ge, Value::Int(3)),
                pred("PROPT", Cmp::Eq, Value::Int(1)),
            ])),
            pred("ELFORM", Cmp::Eq, Value::Int(16)),
        ),
        // 5. same shape, but demand ELFORM == 2 → violated, proves detection
        Rule::field_required(
            names::SECTION_SHELL,
            Some(pred("NIP", Cmp::Ge, Value::Int(3))),
            pred("ELFORM", Cmp::Eq, Value::Int(2)),
        )
        .with_severity(Severity::Warning),
        // 6. scoped rule: MAT_RIGID is fine only inside geometry includes,
        //    flagged anywhere else (here: the main deck).
        Rule::keyword_forbidden(names::MAT_RIGID)
            .except_in(["00_Includes", "geo_"])
            .with_severity(Severity::Warning),
        // 7. a custom rule (impl Check) doing cross-row logic, lifted to a Rule
        Rule::custom(UniqueSectionIds),
    ]);

    println!(
        "{} findings  ({} errors, {} warnings)  clean={}\n",
        report.findings.len(),
        report.count(Severity::Error),
        report.count(Severity::Warning),
        report.is_clean(),
    );
    for f in &report.findings {
        println!("[{:?}] {}\n    {}\n    {}", f.severity, f.rule, f.message, f.location());
    }
}
