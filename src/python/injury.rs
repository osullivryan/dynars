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
    let r = injury::resultant(&crate::python::f64_slice(&x), &crate::python::f64_slice(&y), &crate::python::f64_slice(&z));
    Ok(r.into_pyarray(py).into_any())
}

/// Head Injury Criterion over a `window`-second interval; `a` is resultant head
/// acceleration in g sampled every `dt` seconds.
#[pyfunction]
#[pyo3(name = "hic", signature = (a, dt, window=0.036))]
pub fn hic(a: PyReadonlyArray1<'_, f64>, dt: f64, window: f64) -> PyResult<f64> {
    Ok(injury::hic(&crate::python::f64_slice(&a), dt, window))
}

/// HIC15 — [`hic`] over a 15 ms window.
#[pyfunction]
#[pyo3(name = "hic15", signature = (a, dt))]
pub fn hic15(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::hic15(&crate::python::f64_slice(&a), dt))
}

/// HIC36 — [`hic`] over a 36 ms window.
#[pyfunction]
#[pyo3(name = "hic36", signature = (a, dt))]
pub fn hic36(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::hic36(&crate::python::f64_slice(&a), dt))
}

/// The "3 ms clip": highest acceleration (g) sustained for `window` seconds
/// (default 3 ms).
#[pyfunction]
#[pyo3(name = "clip", signature = (a, dt, window=0.003))]
pub fn clip(a: PyReadonlyArray1<'_, f64>, dt: f64, window: f64) -> PyResult<f64> {
    Ok(injury::clip(&crate::python::f64_slice(&a), dt, window))
}

/// Gadd Severity Index (CSI on a chest resultant): ∫ a^2.5 dt over the pulse.
#[pyfunction]
#[pyo3(name = "severity_index", signature = (a, dt))]
pub fn severity_index(a: PyReadonlyArray1<'_, f64>, dt: f64) -> PyResult<f64> {
    Ok(injury::severity_index(&crate::python::f64_slice(&a), dt))
}

// ── Tier 2 criteria (SI units: N, N·m, rad/s, rad/s², m/s²) ──────────────────

/// Brain Injury Criterion from the three head angular-velocity channels (rad/s)
/// and their critical values.
#[pyfunction]
#[pyo3(name = "bric", signature = (wx, wy, wz, crit_x, crit_y, crit_z))]
pub fn bric(
    wx: PyReadonlyArray1<'_, f64>,
    wy: PyReadonlyArray1<'_, f64>,
    wz: PyReadonlyArray1<'_, f64>,
    crit_x: f64,
    crit_y: f64,
    crit_z: f64,
) -> PyResult<f64> {
    Ok(injury::bric(&crate::python::f64_slice(&wx), &crate::python::f64_slice(&wy), &crate::python::f64_slice(&wz), crit_x, crit_y, crit_z))
}

/// Universal Brain Injury Criterion (uBRIC) from angular velocity + acceleration
/// channels and their critical values.
#[pyfunction]
#[pyo3(
    name = "ubric",
    signature = (wx, wy, wz, ax, ay, az, crit_wx, crit_wy, crit_wz, crit_ax, crit_ay, crit_az)
)]
#[allow(clippy::too_many_arguments)]
pub fn ubric(
    wx: PyReadonlyArray1<'_, f64>,
    wy: PyReadonlyArray1<'_, f64>,
    wz: PyReadonlyArray1<'_, f64>,
    ax: PyReadonlyArray1<'_, f64>,
    ay: PyReadonlyArray1<'_, f64>,
    az: PyReadonlyArray1<'_, f64>,
    crit_wx: f64,
    crit_wy: f64,
    crit_wz: f64,
    crit_ax: f64,
    crit_ay: f64,
    crit_az: f64,
) -> PyResult<f64> {
    Ok(injury::ubric(
        &crate::python::f64_slice(&wx),
        &crate::python::f64_slice(&wy),
        &crate::python::f64_slice(&wz),
        &crate::python::f64_slice(&ax),
        &crate::python::f64_slice(&ay),
        &crate::python::f64_slice(&az),
        crit_wx,
        crit_wy,
        crit_wz,
        crit_ax,
        crit_ay,
        crit_az,
    ))
}

/// Viscous Criterion (VC)max from a chest deflection channel `y` (m).
#[pyfunction]
#[pyo3(name = "vc", signature = (y, dt, scaling_factor, deformation_constant))]
pub fn vc(
    y: PyReadonlyArray1<'_, f64>,
    dt: f64,
    scaling_factor: f64,
    deformation_constant: f64,
) -> PyResult<f64> {
    Ok(injury::vc(&crate::python::f64_slice(&y), dt, scaling_factor, deformation_constant))
}

/// Neck Injury Criterion Nij (max) — see the Rust docs for the signed-critical
/// convention (compression/extension criticals are negative).
#[pyfunction]
#[pyo3(name = "nij", signature = (fx, fz, my, distance, fzc_te, fzc_co, myc_fl, myc_ex))]
#[allow(clippy::too_many_arguments)]
pub fn nij(
    fx: PyReadonlyArray1<'_, f64>,
    fz: PyReadonlyArray1<'_, f64>,
    my: PyReadonlyArray1<'_, f64>,
    distance: f64,
    fzc_te: f64,
    fzc_co: f64,
    myc_fl: f64,
    myc_ex: f64,
) -> PyResult<f64> {
    Ok(injury::nij(
        &crate::python::f64_slice(&fx),
        &crate::python::f64_slice(&fz),
        &crate::python::f64_slice(&my),
        distance,
        fzc_te,
        fzc_co,
        myc_fl,
        myc_ex,
    ))
}

/// Rear-impact Neck Injury Criterion NIC (max) from T1 and head accel (m/s²).
#[pyfunction]
#[pyo3(name = "nic", signature = (a_t1, a_head, dt))]
pub fn nic(
    a_t1: PyReadonlyArray1<'_, f64>,
    a_head: PyReadonlyArray1<'_, f64>,
    dt: f64,
) -> PyResult<f64> {
    Ok(injury::nic(&crate::python::f64_slice(&a_t1), &crate::python::f64_slice(&a_head), dt))
}

/// Tibia Index (max) from bending moments (N·m) and axial force (N).
#[pyfunction]
#[pyo3(name = "tibia_index", signature = (mx, my, fz, critical_bending_moment, critical_compression_force))]
pub fn tibia_index(
    mx: PyReadonlyArray1<'_, f64>,
    my: PyReadonlyArray1<'_, f64>,
    fz: PyReadonlyArray1<'_, f64>,
    critical_bending_moment: f64,
    critical_compression_force: f64,
) -> PyResult<f64> {
    Ok(injury::tibia_index(
        &crate::python::f64_slice(&mx),
        &crate::python::f64_slice(&my),
        &crate::python::f64_slice(&fz),
        critical_bending_moment,
        critical_compression_force,
    ))
}
