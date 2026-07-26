//! Fast, **typed** rule-based validation of LS-DYNA keyword decks.
//!
//! Checks are expressed against real types, not magic strings: keywords come
//! from the typo-proof [`names`](crate::keywords::names) constants, comparisons
//! use the [`Cmp`] enum, severities use [`Severity`], values use [`Value`].
//!
//! Validation runs off a parsed [`Deck`]: [`Deck::validate`](crate::deck::Deck::validate)
//! takes a set of [`Rule`]s and runs every one over the deck in parallel,
//! reusing the parse. Built-in [`Rule`]s cover the common cases and any custom
//! [`Check`] becomes a `Rule` via [`Rule::custom`]; the [`Expr`] tree composes
//! predicates with `all`/`any`/`not`; [`FileScope`] limits a rule to (or
//! excludes it from) particular include files.
//!
//! This lives in the core crate (not a separate one) because its Python
//! bindings must share the `dynars._dynars` extension, and it carries no heavy
//! dependencies. The heavy Arrow/Iceberg sinks stay in their own crates.

mod check;
mod expr;
mod report;
mod rules;

pub use check::Check;
pub use expr::{pred, Cmp, Expr, FieldPredicate};
pub use report::{FileScope, Finding, Report, Severity};
pub use rules::Rule;

/// The keyword-occurrence handle and its value type live in the core
/// [`model`](crate::model). Re-exported here so callers reaching for them
/// through `validate` still resolve `dynars::validate::{Keyword, Value}`.
pub use crate::model::{Keyword, Value};

use rayon::prelude::*;

pub use crate::deck::Deck;

impl Deck {
    /// Run a set of [`Rule`]s over this deck, reusing the already-parsed blocks
    /// and cached resolution indices. Rules fan out across cores; each returns
    /// its own findings. There is no default rule set — the caller states
    /// exactly which checks to run.
    pub fn validate(&self, rules: impl IntoIterator<Item = Rule>) -> Report {
        let rules: Vec<Rule> = rules.into_iter().collect();
        let findings = rules.par_iter().flat_map_iter(|r| r.run(self)).collect();
        Report { findings }
    }
}
