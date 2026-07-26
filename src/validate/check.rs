//! The open extension point for custom validation.
//!
//! Implement [`Check`] for arbitrary logic — walk the deck with
//! [`Deck::rows`](crate::deck::Deck::rows) and read fields with
//! [`field`](super::field) — then wrap it in [`Rule::custom`](super::Rule::custom)
//! to run it through [`Deck::validate`](crate::deck::Deck::validate) beside the
//! built-ins.

use super::Deck;
use super::report::Finding;

/// A validation check — the open extension point. Implementing this **is** how
/// you write a custom rule; wrap it in [`Rule::custom`](super::Rule::custom) to
/// run it alongside the built-ins. Returns the violations it found (empty =
/// clean).
pub trait Check: Send + Sync {
    fn name(&self) -> String;
    fn run(&self, deck: &Deck) -> Vec<Finding>;
}
