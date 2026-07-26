//! PyO3 bindings for dynars, organized by domain. Feature-gated behind
//! `python`; every type is registered in the `_dynars` extension module
//! (see `lib.rs`). These bindings live in-crate (not a sibling crate) so they
//! can call the crate-private resolution internals (`model::site_index`,
//! `entity_field`, …) that the public API deliberately hides.

mod include_tree;
mod keyword;
mod results;
mod validate;
mod deck;

pub use deck::{parse_deck, PyDeck, PyEntity};
pub use include_tree::{parse_include_tree, PyIncludeNode};
pub use keyword::{parse_keyword_file, PyKeywordFile};
pub use results::{
    open_d3plot, parse_binout, PyBinout, PyBinoutEditor, PyD3plot, PyD3plotEditor, PyD3plotWriter,
    PyIntforWriter,
};
pub use validate::{PyFinding, PyPredicate, PyReport, PyRule};
