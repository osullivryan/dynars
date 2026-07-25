//! Validate a real LS-DYNA deck with typed rules, combinators, scope, and a
//! custom check.
//! Usage: cargo run --example validate_demo -- <main.k>

use std::collections::HashMap;

use dynars::deck::Deck;
use dynars::keywords::names;
use dynars::validate::{pred, visit_rows, Check, Cmp, Expr, Finding, Rule, Severity, Validator, Value};

/// A custom rule (arbitrary Rust logic): SECIDs must be unique across the deck.
/// The built-in `Rule`s are per-row and can't express cross-row aggregation —
/// this is exactly the kind of thing you drop to a `Check` for. It reuses
/// `visit_rows`, the same primary-card view the built-in field rules use.
struct UniqueSectionIds;
impl Check for UniqueSectionIds {
    fn name(&self) -> String {
        "custom:unique_section_ids".into()
    }
    fn run(&self, deck: &Deck, out: &mut Vec<Finding>) {
        let mut seen: HashMap<i64, String> = HashMap::new();
        visit_rows(deck, names::SECTION_SHELL, |r| {
            let Some(Value::Int(id)) = r.field("SECID") else { return };
            let here = format!("{}:{}", r.file.display(), r.line);
            if let Some(first) = seen.get(&id) {
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: "SECTION_SHELL".into(),
                    file: r.file.to_path_buf(),
                    line: r.line,
                    message: format!("duplicate SECID {id} (first defined at {first})"),
                });
            } else {
                seen.insert(id, here);
            }
        });
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: validate_demo <main.k>");

    let validator = Validator::new()
        // 1. a material *type* that can't be used
        .rule(Rule::keyword_forbidden(names::MAT_ADD_EROSION))
        // 2. a specific id that can't be used (SECTION with SECID == 2)
        .rule(Rule::field_forbidden_values(names::SECTION_SHELL, "SECID", [Value::Int(2)]))
        // 3. every *INCLUDE must resolve on disk
        .rule(Rule::include_missing().with_severity(Severity::Warning))
        // 4. combinator (tier 2): if NIP >= 3 AND PROPT == 1, ELFORM must be 16
        .rule(Rule::field_required(
            names::SECTION_SHELL,
            Some(Expr::all([
                pred("NIP", Cmp::Ge, Value::Int(3)),
                pred("PROPT", Cmp::Eq, Value::Int(1)),
            ])),
            pred("ELFORM", Cmp::Eq, Value::Int(16)),
        ))
        // 5. same shape, but demand ELFORM == 2 → violated, proves detection
        .rule(
            Rule::field_required(
                names::SECTION_SHELL,
                Some(pred("NIP", Cmp::Ge, Value::Int(3))),
                pred("ELFORM", Cmp::Eq, Value::Int(2)),
            )
            .with_severity(Severity::Warning),
        )
        // 6. scoped rule: MAT_RIGID is fine only inside geometry includes,
        //    flagged anywhere else (here: the main deck).
        .rule(
            Rule::keyword_forbidden(names::MAT_RIGID)
                .except_in(["00_Includes", "geo_"])
                .with_severity(Severity::Warning),
        )
        // 7. a custom rule (impl Check) doing cross-row logic
        .check(Box::new(UniqueSectionIds));

    let report = validator.run(&path).expect("parse+validate");

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
