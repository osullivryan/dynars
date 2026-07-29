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
    // HIC = max (t_j−t_i)·avg^2.5, and (that)² = Δv⁵/((j−i)³·dt³). Since √ is
    // monotone the argmax is unchanged, so the hot loop maximizes the pow-free
    // score Δv⁵/(j−i)³ (pure multiplies — no sqrt/powf) and we take ONE √ at the
    // end. SIMD runs four window ends `j` at once (`vel[j..j+4]` is contiguous).
    let inv_cube: Vec<f64> = (0..=w)
        .map(|d| if d == 0 { 0.0 } else { 1.0 / (d as f64).powi(3) })
        .collect();
    let mut best4 = f64x4::splat(0.0);
    let mut best = 0.0f64;
    for i in 0..n - 1 {
        let (vi, vi4) = (vel[i], f64x4::splat(vel[i]));
        let jmax = (i + w).min(n - 1);
        let mut j = i + 1;
        while j + 3 <= jmax {
            let dv = f64x4::from([vel[j], vel[j + 1], vel[j + 2], vel[j + 3]]) - vi4;
            let d = j - i;
            let ic = f64x4::from([inv_cube[d], inv_cube[d + 1], inv_cube[d + 2], inv_cube[d + 3]]);
            best4 = best4.max(dv * dv * dv * dv * dv * ic); // Δv⁵/(j−i)³ (neg for Δv<0 → dropped)
            j += 4;
        }
        while j <= jmax {
            let dv = vel[j] - vi;
            best = best.max(dv * dv * dv * dv * dv * inv_cube[j - i]);
            j += 1;
        }
    }
    let m = best4.to_array().iter().cloned().fold(best, f64::max);
    if m > 0.0 {
        (m / (dt * dt * dt)).sqrt() // the single deferred √
    } else {
        0.0
    }
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
    if w >= n {
        return a.iter().copied().fold(f64::INFINITY, f64::min);
    }
    // Sliding-window minimum via an ascending monotonic deque: each index is
    // pushed and popped at most once, so the whole scan is O(n) — independent of
    // the window width `w` (the naive form is O(n·w)). The clip is the maximum
    // over windows of each window's minimum.
    use std::collections::VecDeque;
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut best = f64::NEG_INFINITY;
    for i in 0..n {
        while dq.back().is_some_and(|&b| a[b] >= a[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
        if dq.front().is_some_and(|&f| f + w <= i) {
            dq.pop_front(); // fell out of the window ending at i
        }
        if i + 1 >= w {
            best = best.max(a[*dq.front().unwrap()]); // front = min of the full window
        }
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
    // Same pow-free `Δv⁵/(j−i)³` score as [`hic`], SIMD across channels — the
    // single deferred `√` happens once per channel at the end.
    let inv_cube: Vec<f64> = (0..=w)
        .map(|d| if d == 0 { 0.0 } else { 1.0 / (d as f64).powi(3) })
        .collect();
    let mut best = vec![0.0f64; nc];
    let simd = nc - nc % 4;
    for i in 0..nt - 1 {
        let jmax = (i + w).min(nt - 1);
        let vi = i * nc;
        for jj in i + 1..=jmax {
            let ic = f64x4::splat(inv_cube[jj - i]);
            let ics = inv_cube[jj - i];
            let vj = jj * nc;
            let mut c = 0;
            while c < simd {
                let a = f64x4::from([vel[vj + c], vel[vj + c + 1], vel[vj + c + 2], vel[vj + c + 3]]);
                let b = f64x4::from([vel[vi + c], vel[vi + c + 1], vel[vi + c + 2], vel[vi + c + 3]]);
                let dv = a - b;
                let score = dv * dv * dv * dv * dv * ic; // Δv⁵/(j−i)³
                let cur = f64x4::from([best[c], best[c + 1], best[c + 2], best[c + 3]]);
                best[c..c + 4].copy_from_slice(&cur.max(score).to_array());
                c += 4;
            }
            for j in simd..nc {
                let dv = vel[vj + j] - vel[vi + j];
                best[j] = best[j].max(dv * dv * dv * dv * dv * ics);
            }
        }
    }
    let inv_dt3 = 1.0 / (dt * dt * dt);
    best.iter()
        .map(|&m| if m > 0.0 { (m * inv_dt3).sqrt() } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hic_matches_a_powf_reference() {
        // The fast a²·√a form must equal the textbook `powf(2.5)` definition.
        let (dt, n) = (1.0e-4, 400usize);
        let a: Vec<f64> = (0..n)
            .map(|i| (30.0 * (2.0 * std::f64::consts::PI * 80.0 * i as f64 * dt).sin()).abs() + 5.0)
            .collect();
        let vel = integrate(&a, dt);
        let w = ((0.036 / dt).round() as usize).clamp(1, n - 1);
        let mut refv = 0.0f64;
        for i in 0..n - 1 {
            for j in i + 1..=(i + w).min(n - 1) {
                let td = (j - i) as f64 * dt;
                let avg = (vel[j] - vel[i]) / td;
                if avg > 0.0 {
                    refv = refv.max(td * avg.powf(2.5));
                }
            }
        }
        let got = hic36(&a, dt);
        assert!((got - refv).abs() <= 1e-9 * refv.max(1.0), "{got} vs {refv}");
    }

    #[test]
    fn hic_cross_validated_reference_pulses() {
        use std::f64::consts::PI;
        // Values pinned from a cross-check (2026-07-28) against an analytic ground
        // truth and Dynasaur's actual HIC code (pint stripped). dynars matches the
        // analytic constant case exactly and is bit-identical to Dynasaur when the
        // HIC-maximizing window is interior (the haversine below: Dynasaur agreed to
        // 2.8e-15). The small diffs Dynasaur shows on boundary-optimum pulses are a
        // one-sample off-by-one in Dynasaur (its window tops out at 35.9ms, not 36ms).

        // Half-sine 80 g / 80 ms @ 0.1 ms.
        let dt = 1.0e-4;
        let a: Vec<f64> = (0..801)
            .map(|i| {
                let t = i as f64 * dt;
                if t <= 0.08 { 80.0 * (PI * t / 0.08).sin() } else { 0.0 }
            })
            .collect();
        assert!((hic36(&a, dt) - 1667.462731).abs() < 1e-4, "half-sine {}", hic36(&a, dt));

        // Haversine 100 g / 30 ms @ 0.05 ms — bit-identical to Dynasaur.
        let dt = 5.0e-5;
        let a: Vec<f64> = (0..1401)
            .map(|i| {
                let t = i as f64 * dt;
                if t <= 0.03 { 50.0 * (1.0 - (2.0 * PI * t / 0.03).cos()) } else { 0.0 }
            })
            .collect();
        assert!((hic36(&a, dt) - 908.805760).abs() < 1e-4, "haversine {}", hic36(&a, dt));

        // Constant 50 g → analytic HIC36 = 0.036·50^2.5, exactly.
        let dt = 1.0e-4;
        let a = vec![50.0f64; 500];
        assert!((hic36(&a, dt) - 0.036 * 50.0f64.powf(2.5)).abs() < 1e-6);
    }

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
    fn clip_deque_matches_naive_reference() {
        // The O(n) monotonic-deque clip must equal the naive O(n·w) sliding min,
        // for several window sizes, on a bumpy signal.
        let dt = 1.0e-4;
        let a: Vec<f64> = (0..300)
            .map(|i| ((i * 7 % 13) as f64 - 5.0) + (i as f64 * 0.1).sin())
            .collect();
        for &win in &[0.0003_f64, 0.001, 0.003, 0.01, 0.05] {
            let w = ((win / dt).round() as usize).max(1);
            let naive = if w >= a.len() {
                a.iter().copied().fold(f64::INFINITY, f64::min)
            } else {
                (0..=a.len() - w)
                    .map(|i| a[i..i + w].iter().copied().fold(f64::INFINITY, f64::min))
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            assert!((clip(&a, dt, win) - naive).abs() < 1e-12, "win={win}");
        }
    }

    #[test]
    fn severity_index_of_constant_is_analytic() {
        let dt = 1.0e-4;
        let a = vec![2.0; 100];
        let expect = 99.0 * dt * 2.0f64.powf(2.5); // (n-1)·dt · a^2.5
        assert!((severity_index(&a, dt) - expect).abs() < 1e-9);
    }
}
