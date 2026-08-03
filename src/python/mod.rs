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
pub use deck::{PyDeck, PyEntity, PyFile, PyKeyword, parse_deck};
pub use include_tree::{PyIncludeNode, parse_include_tree};
pub use injury::{
    bric, clip, hic, hic15, hic36, nic, nij, resultant, severity_index, tibia_index, ubric, vc,
};
pub use keyword::{PyKeywordFile, parse_keyword_file, write_keyword};
pub use results::{
    PyBinout, PyBinoutEditor, PyD3plot, PyD3plotEditor, PyD3plotWriter, PyIntforWriter,
    open_d3plot, parse_binout,
};
#[cfg(feature = "signal")]
pub use signal::{butterworth, cfc, decimate, differentiate, filtfilt, integrate, resample_linear};
pub use validate::{PyFinding, PyPredicate, PyReport, PyRule};

use std::borrow::Cow;

use numpy::PyReadonlyArray1;

/// A contiguous `&[f64]` view of a 1-D numpy array, copying **only** when the
/// array is strided — e.g. a column slice `values[:, i]` out of the `[T, nodes]`
/// matrix `Binout.read_states` returns. Lets the signal / injury kernels accept
/// any 1-D `float64` array, not just C-contiguous ones, so the natural
/// "one entity's history" indexing feeds straight in without an
/// `np.ascontiguousarray` dance.
pub(crate) fn f64_slice<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> Cow<'a, [f64]> {
    match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().to_vec()),
    }
}
