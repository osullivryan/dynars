//! **dynars** — a fast toolkit for LS-DYNA keyword decks and binary results.
//!
//! dynars parses keyword decks (`*KEYWORD` files and everything they
//! `*INCLUDE`), navigates and validates them against a typed model, and reads
//! the binary result files (`d3plot`, `binout`). The same Rust core backs the
//! Python package and the C/Fortran bindings.
//!
//! # Quick start
//!
//! Parse a deck (root file + all its includes) once, then validate and navigate
//! off the one handle — the resolution indices are built lazily and cached:
//!
//! ```no_run
//! use dynars::deck::parse_deck;
//! use dynars::validate::{Rule, Severity};
//!
//! let deck = parse_deck(std::path::Path::new("root.k")).unwrap();
//!
//! // Validate: no default rule set — you pass exactly the checks you want.
//! let report = deck.validate([
//!     Rule::references_resolve(), // every id reference resolves to a definition
//!     Rule::duplicate_ids(),      // no two entities of a kind share an id
//! ]);
//! println!("{} error(s)", report.count(Severity::Error));
//! for f in &report.findings {
//!     println!("{} — {}", f.location(), f.message); // clickable file:line
//! }
//!
//! // Navigate by id and follow references (`*PART.mid` -> `*MAT`):
//! if let Some(part) = deck.part(5) {
//!     if let Some(mat) = part.material() {
//!         println!("part 5 uses *{}", mat.name());
//!     }
//! }
//! ```
//!
//! ## Many decks at once
//!
//! Variants of one model usually `*INCLUDE` the same big files (mesh, materials).
//! A [`Workspace`] reads, parses, and indexes each shared file **once** across the
//! whole batch, then validates the decks in parallel:
//!
//! ```no_run
//! use dynars::Workspace;
//! use dynars::validate::Rule;
//!
//! let ws = Workspace::new();
//! let decks: Vec<_> = ws
//!     .parse_decks(["variant_a/main.k", "variant_b/main.k"])
//!     .into_iter()
//!     .filter_map(|(_root, d)| d.ok())
//!     .collect();
//!
//! let reports = ws.validate_decks(&decks, [Rule::references_resolve()]);
//! println!("{} decks, cache stats {:?}", reports.len(), ws.stats());
//! ```
//!
//! # Where things live
//!
//! - [`deck`] — parse a whole deck ([`parse_deck`](deck::parse_deck)) into a
//!   [`Deck`](deck::Deck); validate and navigate off one handle.
//! - [`batch`] — a [`Workspace`] that parses/validates many decks against a
//!   shared file-and-index cache.
//! - [`validate`] — typed, rule-based checks ([`Rule`](validate::Rule)), with a
//!   custom-[`Check`](validate::Check) escape hatch for arbitrary logic.
//! - [`model`] — the navigation spine ([`Keyword`](model::Keyword) → card →
//!   field), id resolution, and reference following.
//! - [`keywords`] — the built-in keyword library (field layouts + reference
//!   metadata), generated from the Ansys pyDYNA snapshot.
//! - [`schema`] — user-defined schemas for keywords the library doesn't ship.
//! - [`results`] — binary result readers (`d3plot`, `binout`), element
//!   invariants, signal processing, and occupant-injury criteria.
//! - [`include`](mod@include) / [`parser`] / [`file`](mod@file) — the include
//!   graph and the low-level block/file parsing everything above is built on.
//!
//! # Feature flags
//!
//! - `python` — PyO3 bindings (the `dynars` Python package).
//! - `signal` — result-history signal processing (SAE J211 CFC, Butterworth,
//!   integrate/differentiate).
//! - `ffi` — C ABI (and, through it, Fortran) bindings for the parse + validate
//!   path.
//! - `typed-keywords` — a generated typed struct per keyword (~3170; opt-in).
//! - `arrow` — convert `binout`/`d3plot` results into Apache Arrow
//!   `RecordBatch`es (the Parquet/Iceberg seam); pulls in arrow-rs, so opt-in.

// Keep the docs honest: broken or private intra-doc links are warnings (the docs
// CI builds with `-D warnings`, so they fail the build there).
#![warn(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

/// Convert `binout`/`d3plot` results into Apache Arrow `RecordBatch`es — the
/// Parquet/Iceberg seam. Behind the `arrow` feature (pulls in arrow-rs).
#[cfg(feature = "arrow")]
pub mod arrow;
pub mod batch;
pub mod deck;
pub mod file;
pub mod include;
pub mod keywords;
pub mod model;
pub mod parser;
pub mod results;
pub mod schema;
pub mod testgen;
pub mod validate;

/// `#[derive(Keyword)]` / `#[derive(Card)]` for declaring keyword schemas as
/// structs (see [`schema`]).
pub use dynars_derive::{Card, Keyword};
pub use schema::{CardLayout, KeywordSchema};

/// Batch parsing/checking across decks that share `*INCLUDE`s (see [`batch`]).
pub use batch::Workspace;

/// C ABI (and, through it, Fortran) bindings for the deck parse + validate
/// path. Opt-in and self-contained: the `unsafe` FFI layer only compiles under
/// `--features ffi`, so a normal build, the Python extension, and the CLI never
/// pull it in.
#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "python")]
mod python;

/// High-performance LS-DYNA keyword file include tree parser.
#[cfg(feature = "python")]
#[pyo3::pymodule]
pub mod _dynars {
    #[pymodule_export]
    use crate::python::PyIncludeNode;

    #[pymodule_export]
    use crate::python::parse_include_tree;

    #[pymodule_export]
    use crate::python::PyKeywordFile;

    #[pymodule_export]
    use crate::python::parse_keyword_file;

    #[pymodule_export]
    use crate::python::write_keyword;

    #[pymodule_export]
    use crate::python::PyBinout;

    #[pymodule_export]
    use crate::python::parse_binout;

    #[pymodule_export]
    use crate::python::PyD3plot;

    #[pymodule_export]
    use crate::python::open_d3plot;

    #[pymodule_export]
    use crate::python::PyD3plotWriter;

    #[pymodule_export]
    use crate::python::PyD3plotEditor;

    #[pymodule_export]
    use crate::python::PyIntforWriter;

    #[pymodule_export]
    use crate::python::PyBinoutEditor;

    #[pymodule_export]
    use super::results::StateBlock;

    #[pymodule_export]
    use super::results::InterfaceField;

    #[pymodule_export]
    use super::results::FsiforField;

    // Deck validation
    #[pymodule_export]
    use crate::python::PyRule;

    #[pymodule_export]
    use crate::python::PyPredicate;

    #[pymodule_export]
    use crate::python::PyFinding;

    #[pymodule_export]
    use crate::python::PyReport;

    #[pymodule_export]
    use super::validate::Cmp;

    #[pymodule_export]
    use super::validate::Severity;

    // Deck: parse once, validate + navigate
    #[pymodule_export]
    use crate::python::PyDeck;

    #[pymodule_export]
    use crate::python::PyEntity;

    #[pymodule_export]
    use crate::python::parse_deck;

    // Workspace: batch-parse/validate many decks sharing *INCLUDEs
    #[pymodule_export]
    use crate::python::PyWorkspace;

    // Occupant injury criteria (always available).
    #[pymodule_export]
    use crate::python::resultant;

    #[pymodule_export]
    use crate::python::hic;

    #[pymodule_export]
    use crate::python::hic15;

    #[pymodule_export]
    use crate::python::hic36;

    #[pymodule_export]
    use crate::python::clip;

    #[pymodule_export]
    use crate::python::severity_index;

    // Tier 2 injury criteria (neck, brain, chest, tibia).
    #[pymodule_export]
    use crate::python::bric;

    #[pymodule_export]
    use crate::python::ubric;

    #[pymodule_export]
    use crate::python::vc;

    #[pymodule_export]
    use crate::python::nij;

    #[pymodule_export]
    use crate::python::nic;

    #[pymodule_export]
    use crate::python::tibia_index;

    // Signal post-processing (feature `signal`, folded into `python`).
    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::cfc;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::filtfilt;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::butterworth;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::integrate;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::differentiate;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::decimate;

    #[cfg(feature = "signal")]
    #[pymodule_export]
    use crate::python::resample_linear;
}
