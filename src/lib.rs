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
}
