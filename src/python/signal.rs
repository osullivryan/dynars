//! Python bindings for `results::signal` — numerical post-processing of result
//! time-histories. Gated on the `signal` feature (folded into `python`). Every
//! function takes and returns numpy `float64` arrays, so it chains straight off
//! the readers' channel arrays.
#![cfg(feature = "signal")]

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::results::signal;

/// Apply an SAE J211 CFC low-pass filter (zero-phase). `cfc` is the class in Hz
/// (60/180/600/1000 or any value); `dt` is the sample interval in seconds.
#[pyfunction]
#[pyo3(name = "cfc", signature = (values, cfc, dt))]
pub fn cfc<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    cfc: f64,
    dt: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let y = signal::cfc(values.as_slice()?, cfc, dt);
    Ok(y.into_pyarray(py).into_any())
}

/// Zero-phase forward-backward filtering of a `(b, a)` filter — the analogue of
/// `scipy.signal.filtfilt` with default odd padding.
#[pyfunction]
#[pyo3(name = "filtfilt", signature = (b, a, values))]
pub fn filtfilt<'py>(
    py: Python<'py>,
    b: Vec<f64>,
    a: Vec<f64>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let y = signal::filtfilt(&b, &a, values.as_slice()?);
    Ok(y.into_pyarray(py).into_any())
}

/// Zero-phase Butterworth filter: `order`-pole, corner `cutoff` Hz at sample
/// rate `fs` Hz, `btype` = "low" or "high".
#[pyfunction]
#[pyo3(name = "butterworth", signature = (values, order, cutoff, fs, btype="low"))]
pub fn butterworth<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    order: u32,
    cutoff: f64,
    fs: f64,
    btype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let band = match btype {
        "low" | "lowpass" => signal::Band::Low,
        "high" | "highpass" => signal::Band::High,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "btype must be 'low' or 'high', got {other:?}"
            )));
        }
    };
    let y = signal::butterworth(values.as_slice()?, order, cutoff, fs, band)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(y.into_pyarray(py).into_any())
}

/// Cumulative trapezoidal integral (e.g. acceleration → velocity). Same length
/// as `values`, starting at 0.
#[pyfunction]
#[pyo3(name = "integrate", signature = (values, dt))]
pub fn integrate<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let y = signal::integrate(values.as_slice()?, dt);
    Ok(y.into_pyarray(py).into_any())
}

/// Central-difference derivative (e.g. velocity → acceleration). Same length as
/// `values`.
#[pyfunction]
#[pyo3(name = "differentiate", signature = (values, dt))]
pub fn differentiate<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let y = signal::differentiate(values.as_slice()?, dt);
    Ok(y.into_pyarray(py).into_any())
}
