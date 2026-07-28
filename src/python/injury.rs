//! Python bindings for `results::injury` — occupant injury criteria. Always
//! available in the Python build (pure Rust, no external deps). Acceleration
//! inputs are numpy `float64` arrays in **g**, sampled every `dt` seconds.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::results::injury;

/// Elementwise resultant magnitude √(x²+y²+z²) of three channels.
#[pyfunction]
#[pyo3(name = "resultant", signature = (x, y, z))]
pub fn resultant<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let r = injury::resultant(x.as_slice()?, y.as_slice()?, z.as_slice()?);
    Ok(r.into_pyarray(py).into_any())
}

/// Head Injury Criterion over a `window`-second interval; `a` is resultant head
/// acceleration in g sampled every `dt` seconds.
#[pyfunction]
#[pyo3(name = "hic", signature = (a, dt, window=0.036))]
pub fn hic(a: PyReadonlyArray1<'_, f64>, dt: f64, window: f64) -> PyResult<f64> {
    Ok(injury::hic(a.as_slice()?, dt, window))
}

/// HIC15 — [`hic`] over a 15 ms window.
#[pyfunction]
#[pyo3(name = "hic15", signature = (a, dt))]
pub fn hic15(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::hic15(a.as_slice()?, dt))
}

/// HIC36 — [`hic`] over a 36 ms window.
#[pyfunction]
#[pyo3(name = "hic36", signature = (a, dt))]
pub fn hic36(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::hic36(a.as_slice()?, dt))
}

/// The "3 ms clip": highest acceleration (g) sustained for `window` seconds
/// (default 3 ms).
#[pyfunction]
#[pyo3(name = "clip", signature = (a, dt, window=0.003))]
pub fn clip(a: PyReadonlyArray1<'_, f64>, dt: f64, window: f64) -> PyResult<f64> {
    Ok(injury::clip(a.as_slice()?, dt, window))
}

/// Gadd Severity Index (CSI on a chest resultant): ∫ a^2.5 dt over the pulse.
#[pyfunction]
#[pyo3(name = "severity_index", signature = (a, dt))]
pub fn severity_index(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::severity_index(a.as_slice()?, dt))
}
