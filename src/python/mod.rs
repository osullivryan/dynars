//! PyO3 bindings for dynars, organized by domain. Feature-gated behind
//! `python`; every type is registered in the `_dynars` extension module
//! (see `lib.rs`). These bindings live in-crate (not a sibling crate) so they
//! can call the crate-private resolution internals (`model::site_index`,
//! `entity_field`, …) that the public API deliberately hides.

mod batch;
mod deck;
mod include_tree;
mod injury;
mod keyword;
mod results;
#[cfg(feature = "signal")]
mod signal;
mod validate;

pub use batch::PyWorkspace;
pub use deck::{PyDeck, PyEntity, parse_deck};
pub use include_tree::{PyIncludeNode, parse_include_tree};
pub use injury::{
    bric, clip, hic, hic15, hic36, nic, nij, resultant, severity_index, tibia_index, ubric, vc,
};
pub use keyword::{PyKeywordFile, parse_keyword_file};
pub use results::{
    PyBinout, PyBinoutEditor, PyD3plot, PyD3plotEditor, PyD3plotWriter, PyIntforWriter,
    open_d3plot, parse_binout,
};
#[cfg(feature = "signal")]
pub use signal::{butterworth, cfc, decimate, differentiate, filtfilt, integrate, resample_linear};
pub use validate::{PyFinding, PyPredicate, PyReport, PyRule};
