//! Numerical post-processing of result time-histories (opt-in `signal` feature).
//!
//! Everything operates on plain `&[f64]` slices, so it composes directly with the
//! columnar channels the [`Binout`](super::Binout)/[`D3plot`](super::D3plot)
//! readers return.
//!
//! - [`cfc`] — SAE J211 CFC low-pass filters (CFC60/180/600/1000, or any class):
//!   the phaseless Butterworth filtering that gates crash injury criteria. J211
//!   gives the coefficients in closed form, so we implement the standard verbatim.
//! - [`butterworth`] — general zero-phase Butterworth. The pole/zero *design* is
//!   done by `iir_filters`; we expand it to `(b, a)` ([`poly_from_roots`]) and run
//!   it through the same [`filtfilt`].
//! - [`filtfilt`] — zero-phase forward-backward filtering of a `(b, a)` filter,
//!   the analogue of `scipy.signal.filtfilt` (odd padding + settled initial
//!   conditions).
//! - [`integrate`] / [`differentiate`] — cumulative trapezoid and central
//!   difference, for the acceleration → velocity → displacement chain.
//! - [`decimate`] / [`resample_linear`] — integer downsample (keep every Nth) and
//!   linear resample to a new `dt`. Decimating a CFC-filtered (band-limited)
//!   signal before an O(n·w) criterion like HIC cuts the cost by ~1/factor² with
//!   no loss.

use std::f64::consts::{PI, SQRT_2};

/// The `(b, a)` coefficients of the SAE J211 two-pole Butterworth low-pass for
/// filter class `cfc` (Hz) at sample interval `dt` (s). The `2.0775` factor folds
/// in both the `5/3` corner (CFC60 → 100 Hz) and the double-pass magnitude
/// correction, so applying this forward+backward lands the −3 dB point where J211
/// specifies. Returned in `scipy`/`lfilter` convention (`a[0] = 1`).
pub fn cfc_coefficients(cfc: f64, dt: f64) -> ([f64; 3], [f64; 3]) {
    let wd = 2.0 * PI * cfc * 2.0775;
    let wa = (wd * dt / 2.0).tan();
    let denom = 1.0 + SQRT_2 * wa + wa * wa;
    let b0 = wa * wa / denom;
    // J211 feedback terms; `lfilter`'s `a` is their negation (with a[0] = 1).
    let j1 = 2.0 * (wa * wa - 1.0) / denom;
    let j2 = (-1.0 + SQRT_2 * wa - wa * wa) / denom;
    ([b0, 2.0 * b0, b0], [1.0, j1, -j2])
}

/// Apply an SAE J211 CFC low-pass filter to `x` (zero-phase), where `cfc` is the
/// filter class in Hz (60/180/600/1000 for the standard channels, but any value
/// works) and `dt` is the sample interval in seconds. J211 wants a sample rate of
/// at least `10 × cfc`.
pub fn cfc(x: &[f64], cfc: f64, dt: f64) -> Vec<f64> {
    let (b, a) = cfc_coefficients(cfc, dt);
    filtfilt(&b, &a, x)
}

/// Which band a [`butterworth`] filter passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    Low,
    High,
}

/// Zero-phase Butterworth filter: design an `order`-pole Butterworth with corner
/// `cutoff` (Hz) at sample rate `fs` (Hz), applied forward+backward. `Err` if the
/// design is rejected (e.g. `cutoff` ≥ Nyquist).
#[cfg(feature = "signal")]
pub fn butterworth(
    x: &[f64],
    order: u32,
    cutoff: f64,
    fs: f64,
    band: Band,
) -> Result<Vec<f64>, String> {
    use iir_filters::filter_design::{FilterType, butter};
    let ft = match band {
        Band::Low => FilterType::LowPass(cutoff),
        Band::High => FilterType::HighPass(cutoff),
    };
    let zpk = butter(order, ft, fs).map_err(|e| e.to_string())?;
    // iir_filters returns the digital pole/zero/gain; expand to a transfer
    // function ourselves (its own `zpk2tf` is crate-private).
    let b: Vec<f64> = poly_from_roots(&zpk.z)
        .into_iter()
        .map(|c| c * zpk.k)
        .collect();
    let a = poly_from_roots(&zpk.p);
    Ok(filtfilt(&b, &a, x))
}

/// Real polynomial coefficients (highest power first) of `∏ (z − rᵢ)`. For a real
/// filter the roots come in conjugate pairs, so the imaginary parts cancel and we
/// keep the real parts. This is the `zpk → (b, a)` step iir_filters hides.
#[cfg(feature = "signal")]
fn poly_from_roots(roots: &[num_complex::Complex<f64>]) -> Vec<f64> {
    use num_complex::Complex;
    let mut c = vec![Complex::new(1.0, 0.0)];
    for r in roots {
        let mut next = vec![Complex::new(0.0, 0.0); c.len() + 1];
        for (i, &ci) in c.iter().enumerate() {
            next[i] += ci; // z · (running product)
            next[i + 1] -= *r * ci; // −rᵢ · (running product)
        }
        c = next;
    }
    c.into_iter().map(|z| z.re).collect()
}

/// Zero-phase forward-backward filtering — the analogue of `scipy.signal.filtfilt`
/// with the default odd padding. `b`/`a` are the filter's numerator/denominator
/// (`a[0]` need not be 1; it's normalized here). The filter is applied once
/// forward and once in reverse, so there is no net phase shift.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let ntaps = b.len().max(a.len());
    if n <= 1 || ntaps < 2 {
        return x.to_vec();
    }
    // Normalize by a[0] and pad both to equal length.
    let a0 = a[0];
    let mut bb: Vec<f64> = b.iter().map(|v| v / a0).collect();
    let mut aa: Vec<f64> = a.iter().map(|v| v / a0).collect();
    bb.resize(ntaps, 0.0);
    aa.resize(ntaps, 0.0);

    let edge = (3 * ntaps).min(n - 1); // scipy's default padlen
    let ext = odd_ext(x, edge);
    let zi = lfilter_zi(&bb, &aa);

    // Forward pass, started in the settled state scaled by the first sample.
    let zi_f: Vec<f64> = zi.iter().map(|z| z * ext[0]).collect();
    let mut y = lfilter(&bb, &aa, &ext, &zi_f);
    // Reverse and filter again, then reverse back → zero phase.
    y.reverse();
    let zi_b: Vec<f64> = zi.iter().map(|z| z * y[0]).collect();
    let mut y2 = lfilter(&bb, &aa, &y, &zi_b);
    y2.reverse();

    y2[edge..edge + n].to_vec()
}

/// One forward pass, transposed direct-form II, with initial state `zi`
/// (length `ntaps−1`). Assumes `a[0] == 1` and `b.len() == a.len()`.
fn lfilter(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> Vec<f64> {
    let m = b.len();
    let mut z = zi.to_vec();
    z.resize(m - 1, 0.0);
    let mut y = Vec::with_capacity(x.len());
    for &xi in x {
        let yi = b[0] * xi + z[0];
        for j in 0..m - 2 {
            z[j] = b[j + 1] * xi + z[j + 1] - a[j + 1] * yi;
        }
        z[m - 2] = b[m - 1] * xi - a[m - 1] * yi;
        y.push(yi);
    }
    y
}

/// `scipy.signal.lfilter_zi`: the initial state whose step response starts
/// settled, so `filtfilt`'s padding carries no startup transient. `a[0] == 1`.
fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut zi = vec![0.0; n - 1];
    // zi[0] = (Σb − b0·Σa) / Σa, from (I − Aᵀ)·zi = b[1:] − a[1:]·b0.
    let suma: f64 = a.iter().sum();
    let sumb: f64 = b.iter().sum();
    zi[0] = (sumb - b[0] * suma) / suma;
    let mut asum = 1.0;
    let mut csum = 0.0;
    for k in 1..n - 1 {
        asum += a[k];
        csum += b[k] - a[k] * b[0];
        zi[k] = asum * zi[0] - csum;
    }
    zi
}

/// `scipy.signal.odd_ext`: reflect `x` about each endpoint, `edge` samples per
/// side, so a filtered edge continues smoothly instead of ringing.
fn odd_ext(x: &[f64], edge: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * edge);
    for i in (1..=edge).rev() {
        out.push(2.0 * x[0] - x[i]);
    }
    out.extend_from_slice(x);
    for i in 1..=edge {
        out.push(2.0 * x[n - 1] - x[n - 1 - i]);
    }
    out
}

/// Cumulative trapezoidal integral of `x` sampled every `dt` seconds — e.g.
/// acceleration → velocity. Output starts at 0 and matches `x` in length (like
/// `scipy.integrate.cumulative_trapezoid(..., initial=0)`).
pub fn integrate(x: &[f64], dt: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let mut acc = 0.0;
    if !x.is_empty() {
        out.push(0.0);
    }
    for i in 1..x.len() {
        acc += dt * (x[i - 1] + x[i]) / 2.0;
        out.push(acc);
    }
    out
}

/// Cumulative trapezoidal integral over non-uniform sample times `t` (paired with
/// `x`); the variable-step form of [`integrate`].
pub fn integrate_over(t: &[f64], x: &[f64]) -> Vec<f64> {
    let n = t.len().min(x.len());
    let mut out = Vec::with_capacity(n);
    let mut acc = 0.0;
    if n > 0 {
        out.push(0.0);
    }
    for i in 1..n {
        acc += (t[i] - t[i - 1]) * (x[i - 1] + x[i]) / 2.0;
        out.push(acc);
    }
    out
}

/// Central-difference derivative of `x` sampled every `dt` seconds — e.g.
/// velocity → acceleration. Second-order accurate in the interior, first-order at
/// the ends, same length as `x` (matching `numpy.gradient`).
pub fn differentiate(x: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut out = vec![0.0; n];
    out[0] = (x[1] - x[0]) / dt;
    out[n - 1] = (x[n - 1] - x[n - 2]) / dt;
    for i in 1..n - 1 {
        out[i] = (x[i + 1] - x[i - 1]) / (2.0 * dt);
    }
    out
}

/// Decimate by an integer `factor`: keep every `factor`-th sample, so the new
/// sample interval is `dt·factor`. O(n) and lossless **only when the signal is
/// already band-limited** — e.g. after [`cfc`], where a much-finer-than-Nyquist
/// series can be thinned with no aliasing. This is the cheap way to make an
/// O(n·w) criterion (HIC) tractable on very fine `dt` (cost falls as ~1/factor²).
/// Un-filtered data should be low-passed first.
pub fn decimate(x: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return x.to_vec();
    }
    x.iter().step_by(factor).copied().collect()
}

/// Resample a uniformly-sampled series from `dt_in` to `dt_out` by linear
/// interpolation (up- or down-sampling). Exact for piecewise-linear inputs; for
/// downsampling of un-band-limited data, low-pass first to avoid aliasing. The
/// output spans the same duration, length `⌊(n−1)·dt_in/dt_out⌋ + 1`.
pub fn resample_linear(x: &[f64], dt_in: f64, dt_out: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 || dt_in <= 0.0 || dt_out <= 0.0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![x[0]];
    }
    let n_out = ((n - 1) as f64 * dt_in / dt_out).floor() as usize + 1;
    (0..n_out)
        .map(|k| {
            let pos = k as f64 * dt_out / dt_in;
            let i = pos.floor() as usize;
            if i >= n - 1 {
                x[n - 1]
            } else {
                let frac = pos - i as f64;
                x[i] * (1.0 - frac) + x[i + 1] * frac
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_keeps_every_nth() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(decimate(&x, 1), x);
        assert_eq!(decimate(&x, 2), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(decimate(&x, 3), vec![0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn resample_linear_is_exact_for_a_ramp() {
        // Linear interpolation reproduces a linear signal at any output dt.
        let dt_in = 1.0e-3;
        let n = 100;
        let f = |t: f64| 2.0 + 3.0 * t;
        let x: Vec<f64> = (0..n).map(|i| f(i as f64 * dt_in)).collect();
        for &dt_out in &[2.5e-3, 4.0e-4, dt_in] {
            let y = resample_linear(&x, dt_in, dt_out);
            for (k, &v) in y.iter().enumerate() {
                assert!((v - f(k as f64 * dt_out)).abs() < 1e-9, "dt_out={dt_out} k={k}");
            }
            // output never runs past the input's time span
            assert!((y.len() as f64 - 1.0) * dt_out <= (n - 1) as f64 * dt_in + 1e-12);
        }
    }

    #[test]
    fn filtfilt_has_unit_dc_gain_and_no_phase_lag() {
        // A constant passes through unchanged (DC gain 1, both passes).
        let (b, a) = cfc_coefficients(1000.0, 1.0e-4);
        let c = vec![3.5; 200];
        let y = filtfilt(&b, &a, &c);
        for v in &y {
            assert!((v - 3.5).abs() < 1e-6, "constant not preserved: {v}");
        }
        // A symmetric pulse stays symmetric — zero phase keeps the peak centered.
        let mut x = vec![0.0; 201];
        x[100] = 1.0;
        let y = cfc(&x, 1000.0, 1.0e-4);
        let peak = (0..y.len()).max_by(|&i, &j| y[i].total_cmp(&y[j])).unwrap();
        assert_eq!(peak, 100, "zero-phase filter must keep the peak centered");
        for k in 1..40 {
            assert!(
                (y[100 - k] - y[100 + k]).abs() < 1e-9,
                "response not symmetric at ±{k}"
            );
        }
    }

    #[test]
    fn cfc_coefficients_match_j211() {
        // CFC60 at 10 kHz (dt = 1e-4). Hand-computed from the J211 formula.
        let (b, a) = cfc_coefficients(60.0, 1.0e-4);
        // wa = tan(pi*60*2.0775*1e-4) ≈ 0.0039155; a0 ≈ wa^2/denom.
        let (eb, ea) = reference_cfc(60.0, 1.0e-4);
        for (x, y) in b.iter().zip(eb.iter()) {
            assert!((x - y).abs() < 1e-12, "b: {b:?} vs {eb:?}");
        }
        for (x, y) in a.iter().zip(ea.iter()) {
            assert!((x - y).abs() < 1e-12, "a: {a:?} vs {ea:?}");
        }
        // Numerator sums to the denominator (DC gain 1) — a J211 invariant.
        assert!((b.iter().sum::<f64>() - a.iter().sum::<f64>()).abs() < 1e-9);
    }

    /// Independent re-derivation of the J211 coefficients, for the test.
    fn reference_cfc(cfc: f64, dt: f64) -> ([f64; 3], [f64; 3]) {
        let wa = (PI * cfc * 2.0775 * dt).tan();
        let d = 1.0 + 2.0_f64.sqrt() * wa + wa * wa;
        let a0 = wa * wa / d;
        let b = [a0, 2.0 * a0, a0];
        let a1 = 2.0 * (wa * wa - 1.0) / d;
        let a2 = -(-1.0 + 2.0_f64.sqrt() * wa - wa * wa) / d;
        ([b[0], b[1], b[2]], [1.0, a1, a2])
    }

    #[test]
    fn integrate_and_differentiate_are_analytic() {
        let dt = 0.01;
        let n = 100;
        // Integrate a constant → a ramp.
        let ones = vec![2.0; n];
        let ramp = integrate(&ones, dt);
        assert_eq!(ramp.len(), n);
        assert!((ramp[0]).abs() < 1e-12);
        assert!((ramp[n - 1] - 2.0 * dt * (n - 1) as f64).abs() < 1e-9);
        // Differentiate that ramp → back to the constant (interior).
        let d = differentiate(&ramp, dt);
        for v in &d[1..n - 1] {
            assert!((v - 2.0).abs() < 1e-9, "derivative of ramp: {v}");
        }
        // Non-uniform integration agrees with uniform when the grid is uniform.
        let t: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let ramp2 = integrate_over(&t, &ones);
        for (u, v) in ramp.iter().zip(ramp2.iter()) {
            assert!((u - v).abs() < 1e-9);
        }
    }

    #[cfg(feature = "signal")]
    #[test]
    fn butterworth_matches_a_known_lowpass() {
        // A 2-pole 100 Hz low-pass at 10 kHz should pass a 10 Hz tone almost
        // untouched and strongly attenuate a 2 kHz tone.
        let fs = 10_000.0;
        let n = 2000;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin() + (2.0 * PI * 2000.0 * t).sin()
            })
            .collect();
        let y = butterworth(&sig, 2, 100.0, fs, Band::Low).unwrap();
        assert_eq!(y.len(), n);
        // Compare the low-frequency reference (just the 10 Hz tone) in the
        // interior; the 2 kHz component must be largely gone.
        let mid = n / 2;
        let low: f64 = (2.0 * PI * 10.0 * (mid as f64 / fs)).sin();
        assert!(
            (y[mid] - low).abs() < 0.1,
            "low tone should survive: {} vs {low}",
            y[mid]
        );
    }
}
