//! Validate a real LS-DYNA deck with typed rules, combinators, scope, and a
//! custom check.
//! Usage: cargo run --example validate_demo -- <main.k>

use dynars::deck::Deck;
use dynars::keywords::names;
use dynars::validate::{pred, Check, Cmp, Expr, Finding, Rule, Severity, Validator, Value};

/// A user-defined check (arbitrary Rust logic): flag rigid materials.
struct NoRigidBodies;
impl Check for NoRigidBodies {
    fn name(&self) -> String {
        "custom:no_rigid_bodies".into()
    }
    fn run(&self, deck: &Deck, out: &mut Vec<Finding>) {
        // Reuse the same typed primitives the built-in rules use.
        Rule::keyword_forbidden(names::MAT_RIGID)
            .with_severity(Severity::Warning)
            .run(deck, out);
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
        // 7. a custom Rust check
        .check(Box::new(NoRigidBodies));

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
