//! Occupant injury criteria (Tier 1: single-channel, no dummy-specific tables).
//!
//! All operate on plain `&[f64]` and assume acceleration in **g** (divide a
//! m/s² channel by 9.81 first) sampled uniformly at interval `dt` seconds — so
//! they chain straight off a [`cfc`](super::signal::cfc)-filtered resultant.
//!
//! - [`resultant`] — √(x²+y²+z²) of three channels.
//! - [`hic`] / [`hic15`] / [`hic36`] — Head Injury Criterion.
//! - [`clip`] — the "3 ms clip": highest level sustained for a window (default 3 ms).
//! - [`severity_index`] — Gadd Severity Index (a.k.a. CSI on the chest resultant).
//!
//! Cross-checked against the Dynasaur reference implementation
//! (VSI-TUGraz/Dynasaur `calc/standard_functions.py`); [`hic`] maximizes over
//! every sub-window `≤ window` (the FMVSS 208 / SAE J1727 definition), which is
//! `≥` Dynasaur's fixed-width-window value.

use super::signal::integrate;
use wide::f64x4;

/// Elementwise resultant magnitude `√(x²+y²+z²)` of three channels (truncated to
/// the shortest).
pub fn resultant(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len()).min(z.len());
    (0..n)
        .map(|i| (x[i] * x[i] + y[i] * y[i] + z[i] * z[i]).sqrt())
        .collect()
}

/// Head Injury Criterion: `max (t2−t1)·[mean a over (t1,t2)]^2.5` over all
/// intervals with `t2−t1 ≤ window` seconds. `a` is the resultant head
/// acceleration in g, sampled every `dt` seconds. Returns 0 for a signal too
/// short to form a window.
pub fn hic(a: &[f64], dt: f64, window: f64) -> f64 {
    let n = a.len();
    if n < 2 || dt <= 0.0 {
        return 0.0;
    }
    // Cumulative ∫a dt, so the mean over (i,j) is (vel[j] − vel[i]) / (t_j − t_i).
    let vel = integrate(a, dt);
    let w = ((window / dt).round() as usize).clamp(1, n - 1);
    let mut hic = 0.0f64;
    for i in 0..n - 1 {
        let jmax = (i + w).min(n - 1);
        for j in i + 1..=jmax {
            let tdiff = (j - i) as f64 * dt;
            let avg = (vel[j] - vel[i]) / tdiff;
            if avg > 0.0 {
                hic = hic.max(tdiff * avg.powf(2.5));
            }
        }
    }
    hic
}

/// HIC15 — [`hic`] over a 15 ms window.
pub fn hic15(a: &[f64], dt: f64) -> f64 {
    hic(a, dt, 0.015)
}

/// HIC36 — [`hic`] over a 36 ms window.
pub fn hic36(a: &[f64], dt: f64) -> f64 {
    hic(a, dt, 0.036)
}

/// The "3 ms clip": the highest acceleration level (in g) sustained continuously
/// for `window` seconds (default 0.003 — pass it explicitly). Computed as the
/// maximum over sliding windows of the per-window minimum (Dynasaur's `a3ms`).
/// If the signal is shorter than the window, returns the whole-signal minimum.
pub fn clip(a: &[f64], dt: f64, window: f64) -> f64 {
    let n = a.len();
    if n == 0 || dt <= 0.0 {
        return 0.0;
    }
    let w = ((window / dt).round() as usize).max(1);
    let win_min = |s: &[f64]| s.iter().copied().fold(f64::INFINITY, f64::min);
    if w >= n {
        return win_min(a);
    }
    let mut best = f64::NEG_INFINITY;
    for i in 0..=n - w {
        best = best.max(win_min(&a[i..i + w]));
    }
    best
}

/// Gadd Severity Index — `∫ a^2.5 dt` over the whole pulse (`a` in g, `dt` in
/// seconds). The same integral applied to the chest resultant is the CSI.
pub fn severity_index(a: &[f64], dt: f64) -> f64 {
    if a.len() < 2 || dt <= 0.0 {
        return 0.0;
    }
    let f: Vec<f64> = a.iter().map(|&v| v.max(0.0).powf(2.5)).collect();
    integrate(&f, dt).pop().unwrap_or(0.0)
}

/// HIC for many channels at once, **SIMD-vectorized across channels** (`wide`
/// f64 lanes). `data` is row-major `n_steps × n_channels` acceleration in g;
/// returns one HIC per channel over the given `window` (seconds). The hot loop
/// exploits `a^2.5 = a²·√a` so it is multiplies + `sqrt` (both vectorize), not a
/// scalar `powf`. Numerically equal to per-channel [`hic`] to ~1e-12, and
/// composes with rayon (split the channel range across threads).
pub fn hic_batch(data: &[f64], n_steps: usize, n_channels: usize, dt: f64, window: f64) -> Vec<f64> {
    let (nt, nc) = (n_steps, n_channels);
    if nt < 2 || nc == 0 || dt <= 0.0 {
        return vec![0.0; nc];
    }
    let w = ((window / dt).round() as usize).clamp(1, nt - 1);
    // Cumulative-trapezoid velocity, same row-major layout.
    let mut vel = vec![0.0f64; nt * nc];
    for i in 1..nt {
        let (prev, cur) = ((i - 1) * nc, i * nc);
        for j in 0..nc {
            vel[cur + j] = vel[prev + j] + 0.5 * (data[cur + j] + data[prev + j]) * dt;
        }
    }
    let mut hic = vec![0.0f64; nc];
    let simd = nc - nc % 4;
    let zero = f64x4::splat(0.0);
    for i in 0..nt - 1 {
        let jmax = (i + w).min(nt - 1);
        let vi = i * nc;
        for jj in i + 1..=jmax {
            let tdiff = (jj - i) as f64 * dt;
            let inv = 1.0 / tdiff;
            let (td, invv) = (f64x4::splat(tdiff), f64x4::splat(inv));
            let vj = jj * nc;
            let mut c = 0;
            while c < simd {
                let a = f64x4::from([vel[vj + c], vel[vj + c + 1], vel[vj + c + 2], vel[vj + c + 3]]);
                let b = f64x4::from([vel[vi + c], vel[vi + c + 1], vel[vi + c + 2], vel[vi + c + 3]]);
                let avgp = ((a - b) * invv).max(zero); // negative means → 0 (dropped)
                let cand = td * avgp * avgp * avgp.sqrt(); // avgp^2.5
                let cur = f64x4::from([hic[c], hic[c + 1], hic[c + 2], hic[c + 3]]);
                hic[c..c + 4].copy_from_slice(&cur.max(cand).to_array());
                c += 4;
            }
            for j in simd..nc {
                let avg = (vel[vj + j] - vel[vi + j]) * inv;
                if avg > 0.0 {
                    hic[j] = hic[j].max(tdiff * avg * avg * avg.sqrt());
                }
            }
        }
    }
    hic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hic_batch_matches_scalar_hic() {
        // 7 channels (not a multiple of 4 → exercises the SIMD tail), each a
        // positive half-sine of a different amplitude.
        let (nt, nc, dt) = (200usize, 7usize, 1.0e-4);
        let mut data = vec![0.0f64; nt * nc];
        for i in 0..nt {
            for j in 0..nc {
                let phase = std::f64::consts::PI * i as f64 / (nt as f64 - 1.0);
                data[i * nc + j] = (10.0 + j as f64) * phase.sin().max(0.0);
            }
        }
        let batch = hic_batch(&data, nt, nc, dt, 0.036);
        for j in 0..nc {
            let col: Vec<f64> = (0..nt).map(|i| data[i * nc + j]).collect();
            let scalar = hic36(&col, dt);
            assert!(
                (batch[j] - scalar).abs() <= 1e-9 * scalar.max(1.0),
                "chan {j}: batch {} vs scalar {scalar}",
                batch[j]
            );
        }
    }

    #[test]
    fn resultant_is_the_vector_norm() {
        let r = resultant(&[3.0, 0.0], &[4.0, 5.0], &[0.0, 12.0]);
        assert!((r[0] - 5.0).abs() < 1e-12);
        assert!((r[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn hic_of_constant_acceleration_is_analytic() {
        // Constant a = 10 g: mean is 10 for every window, so HIC maximizes at the
        // full window width → window · 10^2.5.
        let dt = 1.0e-4;
        let a = vec![10.0; 500];
        let expect = 0.036 * 10.0f64.powf(2.5);
        assert!((hic36(&a, dt) - expect).abs() < 1e-9, "{}", hic36(&a, dt));
        assert!((hic15(&a, dt) - 0.015 * 10.0f64.powf(2.5)).abs() < 1e-9);
    }

    #[test]
    fn clip_finds_the_sustained_level_not_brief_spikes() {
        let dt = 1.0e-4; // 3 ms = 30 samples
        // Baseline 3 g with a 0.5 ms spike to 10 g — too brief to be "sustained".
        let mut a = vec![3.0; 400];
        for v in a.iter_mut().take(105).skip(100) {
            *v = 10.0;
        }
        assert!(
            (clip(&a, dt, 0.003) - 3.0).abs() < 1e-12,
            "{}",
            clip(&a, dt, 0.003)
        );
        // Widen the plateau past 3 ms (40 samples) → now it *is* sustained.
        for v in a.iter_mut().take(140).skip(100) {
            *v = 10.0;
        }
        assert!(
            (clip(&a, dt, 0.003) - 10.0).abs() < 1e-12,
            "{}",
            clip(&a, dt, 0.003)
        );
    }

    #[test]
    fn severity_index_of_constant_is_analytic() {
        let dt = 1.0e-4;
        let a = vec![2.0; 100];
        let expect = 99.0 * dt * 2.0f64.powf(2.5); // (n-1)·dt · a^2.5
        assert!((severity_index(&a, dt) - expect).abs() < 1e-9);
    }
}
