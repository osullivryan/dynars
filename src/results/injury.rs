//! Occupant injury criteria.
//!
//! **Tier 1** — head/chest acceleration criteria. All operate on plain `&[f64]`
//! and assume acceleration in **g** (divide a m/s² channel by 9.81 first) sampled
//! uniformly at interval `dt` seconds — so they chain straight off a
//! [`cfc`](super::signal::cfc)-filtered resultant.
//!
//! - [`resultant`] — √(x²+y²+z²) of three channels.
//! - [`hic`] / [`hic15`] / [`hic36`] — Head Injury Criterion.
//! - [`clip`] — the "3 ms clip": highest level sustained for a window (default 3 ms).
//! - [`severity_index`] — Gadd Severity Index (a.k.a. CSI on the chest resultant).
//!
//! **Tier 2** — force / moment / kinematic criteria (still single-object, no
//! hard-coded tables — dummy-specific critical values are passed in). These take
//! **SI** inputs: force in N, moment in N·m, distance in m, angular velocity in
//! rad/s, angular acceleration in rad/s², linear acceleration (NIC) in m/s². Each
//! returns the scalar criterion — the maximum over the pulse.
//!
//! - [`bric`] / [`ubric`] — Brain Injury Criterion (head angular velocity/accel).
//! - [`vc`] — Viscous Criterion on chest deflection.
//! - [`nij`] — neck injury (tension/compression × flexion/extension).
//! - [`nic`] — rear-impact Neck Injury Criterion.
//! - [`tibia_index`] — lower-leg Tibia Index.
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

// ── Tier 2: force / moment / kinematic criteria (SI units) ───────────────────

/// Peak absolute value of a channel.
fn peak_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// Brain Injury Criterion — `‖(max|ωx|/ωxc, max|ωy|/ωyc, max|ωz|/ωzc)‖` from the
/// three head angular-velocity channels (rad/s) and their critical values
/// (Takhounts et al.; Dynasaur `BrIC`). For the Hybrid III 50th the criticals are
/// about `ωxc=66.25`, `ωyc=56.45`, `ωzc=42.87` rad/s.
pub fn bric(wx: &[f64], wy: &[f64], wz: &[f64], crit_x: f64, crit_y: f64, crit_z: f64) -> f64 {
    let rx = peak_abs(wx) / crit_x;
    let ry = peak_abs(wy) / crit_y;
    let rz = peak_abs(wz) / crit_z;
    (rx * rx + ry * ry + rz * rz).sqrt()
}

/// Universal Brain Injury Criterion (uBRIC) — combines the peak angular velocity
/// and acceleration ratios per axis (Dynasaur `uBRIC`):
/// `√(Σ [wᵢ + (aᵢ−wᵢ)·e^(−aᵢ/wᵢ)]²)`, `wᵢ = max|ωᵢ|/ωᵢc`, `aᵢ = max|αᵢ|/αᵢc`.
#[allow(clippy::too_many_arguments)]
pub fn ubric(
    wx: &[f64],
    wy: &[f64],
    wz: &[f64],
    ax: &[f64],
    ay: &[f64],
    az: &[f64],
    crit_wx: f64,
    crit_wy: f64,
    crit_wz: f64,
    crit_ax: f64,
    crit_ay: f64,
    crit_az: f64,
) -> f64 {
    // Per-axis blend of the velocity ratio `w` and acceleration ratio `a`.
    let term = |w: f64, a: f64| if w == 0.0 { a } else { w + (a - w) * (-a / w).exp() };
    let tx = term(peak_abs(wx) / crit_wx, peak_abs(ax) / crit_ax);
    let ty = term(peak_abs(wy) / crit_wy, peak_abs(ay) / crit_ay);
    let tz = term(peak_abs(wz) / crit_wz, peak_abs(az) / crit_az);
    (tx * tx + ty * ty + tz * tz).sqrt()
}

/// Viscous Criterion `(VC)ₘₐₓ` — `max[ scaling·(y/deformation_constant)·ẏ ]`,
/// where `ẏ` is a 5-point central derivative of the chest deflection `y`
/// (Dynasaur `vc`; Lau & Viano). `y` in m, `dt` in s. `deformation_constant` is
/// the chest depth (≈0.229 m for the Hybrid III 50th), `scaling_factor` ≈ 1.
/// Evaluated over the interior, where the criterion peak lies.
pub fn vc(y: &[f64], dt: f64, scaling_factor: f64, deformation_constant: f64) -> f64 {
    let n = y.len();
    if n < 5 || dt <= 0.0 || deformation_constant == 0.0 {
        return 0.0;
    }
    let mut best = f64::NEG_INFINITY;
    for i in 2..n - 2 {
        let dv = (8.0 * (y[i + 1] - y[i - 1]) - (y[i + 2] - y[i - 2])) / (12.0 * dt);
        best = best.max(scaling_factor * (y[i] / deformation_constant) * dv);
    }
    best
}

/// Neck Injury Criterion `Nij_max` — the largest of the four neck loading modes
/// (axial tension/compression × flexion/extension) over the pulse (Dynasaur
/// `nij`; FMVSS 208). `fx` shear and `fz` axial neck force (N), `my` moment (N·m),
/// `distance` the occipital-condyle offset (m). Critical values: `fzc_te`/`fzc_co`
/// axial tension/compression, `myc_fl`/`myc_ex` flexion/extension moments. The
/// moment is transferred to the occipital condyle as `moc = my − distance·fx`.
///
/// Following Dynasaur/FMVSS 208, the compression and extension criticals are
/// **signed to their loading direction** (i.e. passed negative), e.g. Hybrid III
/// 50th: `fzc_te=6806, fzc_co=−6160, myc_fl=310, myc_ex=−135`.
#[allow(clippy::too_many_arguments)]
pub fn nij(
    fx: &[f64],
    fz: &[f64],
    my: &[f64],
    distance: f64,
    fzc_te: f64,
    fzc_co: f64,
    myc_fl: f64,
    myc_ex: f64,
) -> f64 {
    let n = fx.len().min(fz.len()).min(my.len());
    let mut best = 0.0_f64;
    for i in 0..n {
        let moc = my[i] - distance * fx[i];
        let fzc = if fz[i] <= 0.0 { fzc_co } else { fzc_te };
        let myc = if moc > 0.0 { myc_fl } else { myc_ex };
        best = best.max(fz[i] / fzc + moc / myc);
    }
    best
}

/// Rear-impact Neck Injury Criterion `NIC_max` — `max[ 0.2·a_rel + v_rel² ]` where
/// `a_rel = a_T1 − a_head` (m/s²) and `v_rel = ∫a_rel dt` (Dynasaur `NIC`; Boström
/// et al.). `dt` in s.
pub fn nic(a_t1: &[f64], a_head: &[f64], dt: f64) -> f64 {
    let n = a_t1.len().min(a_head.len());
    if n == 0 || dt <= 0.0 {
        return 0.0;
    }
    let a_rel: Vec<f64> = (0..n).map(|i| a_t1[i] - a_head[i]).collect();
    let v_rel = integrate(&a_rel, dt); // cumulative trapezoid, starts at 0
    (0..n)
        .map(|i| 0.2 * a_rel[i] + v_rel[i] * v_rel[i])
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Tibia Index `TI_max` — `max[ |√(Mx²+My²)/M_c| + |Fz/F_c| ]` over the pulse
/// (Dynasaur `tibia_index`; FMVSS 208). `mx`,`my` bending moments (N·m), `fz`
/// axial force (N); `critical_bending_moment` (N·m), `critical_compression_force`
/// (N).
pub fn tibia_index(
    mx: &[f64],
    my: &[f64],
    fz: &[f64],
    critical_bending_moment: f64,
    critical_compression_force: f64,
) -> f64 {
    let n = mx.len().min(my.len()).min(fz.len());
    (0..n)
        .map(|i| {
            let mr = (mx[i] * mx[i] + my[i] * my[i]).sqrt();
            (mr / critical_bending_moment).abs() + (fz[i] / critical_compression_force).abs()
        })
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tier 2 criteria ──────────────────────────────────────────────────────

    #[test]
    fn bric_is_the_normalized_peak_velocity_norm() {
        // Peaks equal to the critical values → each ratio 1 → norm √3.
        let b = bric(&[0.0, 66.25], &[-56.45, 0.0], &[42.87, 1.0], 66.25, 56.45, 42.87);
        assert!((b - 3.0_f64.sqrt()).abs() < 1e-12, "{b}");
        // One axis at half its critical → 0.5.
        assert!((bric(&[33.125], &[0.0], &[0.0], 66.25, 56.45, 42.87) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ubric_reduces_to_bric_when_accel_ratio_equals_velocity_ratio() {
        // aᵢ = wᵢ ⟹ the blend term is exactly wᵢ, so uBRIC == BrIC.
        let w = [10.0, 20.0, 30.0];
        let u = ubric(&w, &[0.0], &[0.0], &w, &[0.0], &[0.0], 40.0, 1.0, 1.0, 40.0, 1.0, 1.0);
        assert!((u - 0.75).abs() < 1e-12, "{u}"); // 30/40 = 0.75 on x, 0 elsewhere
    }

    #[test]
    fn vc_of_a_linear_deflection_is_analytic() {
        // y = A·t → the 5-point derivative is exactly A; VC = scaling·(A·t/dc)·A,
        // maximized at the last interior sample.
        let dt = 1e-3;
        let n = 200;
        let slope = 2.0;
        let y: Vec<f64> = (0..n).map(|i| slope * i as f64 * dt).collect();
        let dc = 0.229;
        let got = vc(&y, dt, 1.0, dc);
        let i = n - 3;
        let expect = (slope * i as f64 * dt / dc) * slope;
        assert!((got - expect).abs() < 1e-9, "{got} vs {expect}");
    }

    #[test]
    fn nij_selects_the_active_loading_mode() {
        // Signed criticals (compression/extension negative), per FMVSS/Dynasaur.
        let (fzc_te, fzc_co, myc_fl, myc_ex) = (6806.0, -6160.0, 310.0, -135.0);
        // Tension-extension: fz=+Fzc_te, moc=−135 → 1 + (−135)/(−135) = 2.
        let n = nij(&[0.0], &[6806.0], &[-135.0], 0.0, fzc_te, fzc_co, myc_fl, myc_ex);
        assert!((n - 2.0).abs() < 1e-9, "{n}");
        // Occipital-condyle transfer shifts the moment: moc = my − distance·fx = −1.
        let n2 = nij(&[100.0], &[6806.0], &[0.0], 0.01, fzc_te, fzc_co, myc_fl, myc_ex);
        assert!((n2 - (1.0 + 1.0 / 135.0)).abs() < 1e-9, "{n2}"); // extension
        // Compression-flexion: fz=−6160, moc=+310 → (−6160)/(−6160) + 310/310 = 2.
        let n3 = nij(&[0.0], &[-6160.0], &[310.0], 0.0, fzc_te, fzc_co, myc_fl, myc_ex);
        assert!((n3 - 2.0).abs() < 1e-9, "{n3}");
    }

    #[test]
    fn nic_of_constant_relative_accel_is_analytic() {
        // a_rel = c constant → v_rel = c·t; NIC = 0.2c + (c·t)², max at final t.
        let dt = 1e-3;
        let n = 100;
        let c = 5.0;
        let got = nic(&vec![c; n], &vec![0.0; n], dt);
        let t_final = (n - 1) as f64 * dt;
        assert!((got - (0.2 * c + (c * t_final).powi(2))).abs() < 1e-9, "{got}");
    }

    #[test]
    fn tibia_index_combines_bending_and_axial() {
        // Mx=3, My=4 → Mr=5; Fz=−2; crit 10 / 8 → 0.5 + 0.25.
        let ti = tibia_index(&[3.0], &[4.0], &[-2.0], 10.0, 8.0);
        assert!((ti - 0.75).abs() < 1e-12, "{ti}");
    }

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
