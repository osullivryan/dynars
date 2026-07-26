//! LS-DYNA binary results readers — Binout (LSDA time-history) and d3plot
//! (state database). Pure Rust, no PyO3; numeric data comes back as typed
//! `Vec`s ready to hand to numpy.
//!
//! - [`Binout`] walks the LSDA symbol tree and reads channels by path.
//! - [`D3plot`] reads the control block, initial geometry, and per-state nodal
//!   coordinates / displacements.

mod binout;
pub mod d3plot;
mod diskfile;
mod edit;
mod lsda;
mod symbol;

pub use binout::Binout;
pub use d3plot::{
    BlockArray, D3plot, D3plotEditor, D3plotError, D3plotWriter, FsiforField, InterfaceField,
    InterfaceFields, IntforWriter, NodeFields, StateBlock,
};
pub use edit::{BinoutEditor, Data};
pub use symbol::ReadResult;

// ── Time-series support ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub time: Vec<f64>,
    pub values: Vec<f64>,
    pub channel: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ScalarExtract {
    Last,
    First,
    Max,
    Min,
    AbsMax,
    Mean,
    /// Trapezoidal integration.
    Integral,
}

pub fn extract_scalar(ts: &TimeSeries, method: ScalarExtract) -> f64 {
    let v = &ts.values;
    if v.is_empty() {
        return 0.0;
    }
    match method {
        ScalarExtract::Last => *v.last().unwrap(),
        ScalarExtract::First => v[0],
        ScalarExtract::Max => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ScalarExtract::Min => v.iter().cloned().fold(f64::INFINITY, f64::min),
        ScalarExtract::AbsMax => v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
        ScalarExtract::Mean => v.iter().sum::<f64>() / v.len() as f64,
        ScalarExtract::Integral => ts
            .time
            .windows(2)
            .zip(v.windows(2))
            .map(|(t, f)| (t[1] - t[0]) * (f[0] + f[1]) / 2.0)
            .sum(),
    }
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LsdaError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid data type size")]
    InvalidDataTypeSize,
    #[error("File not found")]
    FileNotFound,
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("Conversion error: {0}")]
    Conversion(String),
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}
