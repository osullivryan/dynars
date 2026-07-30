//! Element-level derived quantities and failure criteria, built on the packed
//! per-element result blocks the [`D3plot`](super::D3plot) reader returns
//! (`block_data` → `n_states × n_elem × nv`, where the base per-element layout is
//! 6 stress components `σxx,σyy,σzz,σxy,σyz,σzx` followed by effective plastic
//! strain, then history variables).
//!
//! **Layer 1** — pointwise tensor invariants from the six stress (or strain)
//! components: [`von_mises`], [`principal`], [`mean_stress`] / [`pressure`],
//! [`max_shear`], [`triaxiality`]. Plain `f64` functions, unit-agnostic.
//!
//! **Layer 2** — per-part reductions over a whole element block: max
//! ([`part_max_history`]), percentile ([`part_percentile_history`]), and failure
//! fraction ([`part_failure_fraction_history`]) *time histories* of a per-element
//! `quantity` (ready-made: [`von_mises_stress`], [`effective_plastic_strain`]).

use rayon::prelude::*;

/// Von Mises equivalent stress `√(½[(σxx−σyy)²+(σyy−σzz)²+(σzz−σxx)²] +
/// 3(σxy²+σyz²+σzx²)])` — the standard yield/failure scalar of the stress tensor.
pub fn von_mises(sxx: f64, syy: f64, szz: f64, sxy: f64, syz: f64, szx: f64) -> f64 {
    let dev = (sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2);
    (0.5 * dev + 3.0 * (sxy * sxy + syz * syz + szx * szx)).sqrt()
}

/// Mean (hydrostatic) stress `(σxx+σyy+σzz)/3`.
pub fn mean_stress(sxx: f64, syy: f64, szz: f64) -> f64 {
    (sxx + syy + szz) / 3.0
}

/// Pressure `−(σxx+σyy+σzz)/3` (compression positive, LS-DYNA convention).
pub fn pressure(sxx: f64, syy: f64, szz: f64) -> f64 {
    -mean_stress(sxx, syy, szz)
}

/// Principal values of a symmetric tensor given its six components, sorted
/// descending `[λ₁ ≥ λ₂ ≥ λ₃]`. Works for the stress tensor (principal stresses)
/// or the strain tensor (principal strains — pass the tensor shear components
/// `εxy = γxy/2`). Closed-form symmetric-3×3 eigenvalues (Smith's method).
pub fn principal(xx: f64, yy: f64, zz: f64, xy: f64, yz: f64, zx: f64) -> [f64; 3] {
    let p1 = xy * xy + yz * yz + zx * zx;
    if p1 == 0.0 {
        // already diagonal
        let mut d = [xx, yy, zz];
        d.sort_by(|a, b| b.total_cmp(a));
        return d;
    }
    let q = (xx + yy + zz) / 3.0;
    let p2 = (xx - q).powi(2) + (yy - q).powi(2) + (zz - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    // B = (A − qI)/p; r = det(B)/2, clamped for acos.
    let (bxx, byy, bzz) = ((xx - q) / p, (yy - q) / p, (zz - q) / p);
    let (bxy, byz, bzx) = (xy / p, yz / p, zx / p);
    let det = bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bzx)
        + bzx * (bxy * byz - byy * bzx);
    let r = (det / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
    let l1 = q + 2.0 * p * phi.cos(); // largest
    let l3 = q + 2.0 * p * (phi + two_pi_3).cos(); // smallest
    let l2 = 3.0 * q - l1 - l3; // trace − l1 − l3
    [l1, l2, l3]
}

/// Maximum shear stress `(σ₁ − σ₃)/2` (Tresca), from the principal values.
pub fn max_shear(sxx: f64, syy: f64, szz: f64, sxy: f64, syz: f64, szx: f64) -> f64 {
    let p = principal(sxx, syy, szz, sxy, syz, szx);
    (p[0] - p[2]) / 2.0
}

/// Stress triaxiality `σ_mean / σ_vm` (ratio of hydrostatic to equivalent stress)
/// — the ductile-damage driver. Returns 0 when the von Mises stress is 0.
pub fn triaxiality(sxx: f64, syy: f64, szz: f64, sxy: f64, syz: f64, szx: f64) -> f64 {
    let vm = von_mises(sxx, syy, szz, sxy, syz, szx);
    if vm == 0.0 {
        0.0
    } else {
        mean_stress(sxx, syy, szz) / vm
    }
}

// ── Layer 2: per-part reductions over a packed element block ─────────────────
//
// These take the d3plot block flattened row-major as `n_states × n_elem × nv`
// (from `D3plot::block_data`), the per-element `part_ids` (from the connectivity),
// the target `part`, and a `quantity` closure mapping one element's `nv`-word
// slice for a state to a scalar. Ready-made extractors below.

/// Effective plastic strain of an element (word 6, after the 6 stresses).
pub fn effective_plastic_strain(elem: &[f64]) -> f64 {
    elem.get(6).copied().unwrap_or(0.0)
}

/// Von Mises stress of an element from its first 6 words (`σxx…σzx`).
pub fn von_mises_stress(elem: &[f64]) -> f64 {
    if elem.len() < 6 {
        return 0.0;
    }
    von_mises(elem[0], elem[1], elem[2], elem[3], elem[4], elem[5])
}

/// Column indices of the elements belonging to `part`.
fn part_indices(n_elem: usize, part_ids: &[i64], part: i64) -> Vec<usize> {
    (0..n_elem).filter(|&e| part_ids.get(e) == Some(&part)).collect()
}

/// `nv`-word slice of element `e` at state `s`.
#[inline]
fn elem(data: &[f64], n_elem: usize, nv: usize, s: usize, e: usize) -> &[f64] {
    let base = s * n_elem * nv + e * nv;
    &data[base..base + nv]
}

/// Max of `quantity` over a part's elements, per state (length `n_states`).
/// Parallelized across states (independent); no per-state allocation.
pub fn part_max_history(
    data: &[f64],
    n_states: usize,
    n_elem: usize,
    nv: usize,
    part_ids: &[i64],
    part: i64,
    quantity: impl Fn(&[f64]) -> f64 + Sync,
) -> Vec<f64> {
    let idx = part_indices(n_elem, part_ids, part);
    (0..n_states)
        .into_par_iter()
        .map(|s| {
            idx.iter()
                .fold(0.0_f64, |m, &e| m.max(quantity(elem(data, n_elem, nv, s, e))))
        })
        .collect()
}

/// The `pct`-th percentile (0–100, linear interpolation) of `quantity` over a
/// part's elements, per state — robust to single-element outliers.
#[allow(clippy::too_many_arguments)]
pub fn part_percentile_history(
    data: &[f64],
    n_states: usize,
    n_elem: usize,
    nv: usize,
    part_ids: &[i64],
    part: i64,
    pct: f64,
    quantity: impl Fn(&[f64]) -> f64 + Sync,
) -> Vec<f64> {
    let idx = part_indices(n_elem, part_ids, part);
    (0..n_states)
        .into_par_iter()
        .map(|s| {
            let mut v: Vec<f64> =
                idx.iter().map(|&e| quantity(elem(data, n_elem, nv, s, e))).collect();
            percentile(&mut v, pct)
        })
        .collect()
}

/// Fraction (0–1) of a part's elements whose `quantity` exceeds `threshold`, per
/// state — e.g. an effective-plastic-strain failure indicator for the part.
#[allow(clippy::too_many_arguments)]
pub fn part_failure_fraction_history(
    data: &[f64],
    n_states: usize,
    n_elem: usize,
    nv: usize,
    part_ids: &[i64],
    part: i64,
    threshold: f64,
    quantity: impl Fn(&[f64]) -> f64 + Sync,
) -> Vec<f64> {
    let idx = part_indices(n_elem, part_ids, part);
    if idx.is_empty() {
        return vec![0.0; n_states];
    }
    let inv = 1.0 / idx.len() as f64;
    (0..n_states)
        .into_par_iter()
        .map(|s| {
            idx.iter()
                .filter(|&&e| quantity(elem(data, n_elem, nv, s, e)) > threshold)
                .count() as f64
                * inv
        })
        .collect()
}

/// Linear-interpolation percentile (`pct` in 0–100) of `v`; sorts in place.
fn percentile(v: &mut [f64], pct: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    if v.len() == 1 {
        return v[0];
    }
    let rank = (pct / 100.0).clamp(0.0, 1.0) * (v.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_reductions_over_a_synthetic_block() {
        // nv=7 (6 stress + eff plastic strain), 4 elements, parts [10,10,20,10], 2 states.
        let (nv, n_elem, n_states) = (7usize, 4usize, 2usize);
        let part_ids = [10i64, 10, 20, 10];
        let eps = [[0.1, 0.3, 0.9, 0.2], [0.4, 0.5, 0.1, 0.05]]; // [state][elem], word 6
        let mut data = vec![0.0f64; n_states * n_elem * nv];
        for s in 0..n_states {
            for e in 0..n_elem {
                data[s * n_elem * nv + e * nv + 6] = eps[s][e];
            }
        }
        // part 10 = elements 0,1,3
        let mx = part_max_history(&data, n_states, n_elem, nv, &part_ids, 10, effective_plastic_strain);
        assert_eq!(mx, vec![0.3, 0.5]);
        let ff = part_failure_fraction_history(
            &data, n_states, n_elem, nv, &part_ids, 10, 0.25, effective_plastic_strain,
        );
        assert!((ff[0] - 1.0 / 3.0).abs() < 1e-12 && (ff[1] - 2.0 / 3.0).abs() < 1e-12);
        let p50 = part_percentile_history(
            &data, n_states, n_elem, nv, &part_ids, 10, 50.0, effective_plastic_strain,
        );
        assert!((p50[0] - 0.2).abs() < 1e-12 && (p50[1] - 0.4).abs() < 1e-12);
        // extractors
        assert!((von_mises_stress(&[120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) - 120.0).abs() < 1e-9);
        assert_eq!(effective_plastic_strain(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42]), 0.42);
    }

    #[test]
    fn von_mises_known_states() {
        // Uniaxial σ → |σ|.
        assert!((von_mises(250.0, 0.0, 0.0, 0.0, 0.0, 0.0) - 250.0).abs() < 1e-9);
        // Hydrostatic → 0.
        assert!(von_mises(100.0, 100.0, 100.0, 0.0, 0.0, 0.0).abs() < 1e-9);
        // Pure shear τ → √3·|τ|.
        assert!((von_mises(0.0, 0.0, 0.0, 10.0, 0.0, 0.0) - 3.0f64.sqrt() * 10.0).abs() < 1e-9);
    }

    #[test]
    fn principal_recovers_diagonal_and_uniaxial() {
        // Diagonal → sorted descending.
        assert_eq!(principal(1.0, 5.0, -3.0, 0.0, 0.0, 0.0), [5.0, 1.0, -3.0]);
        // Uniaxial tension → [σ, 0, 0].
        let p = principal(200.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((p[0] - 200.0).abs() < 1e-9 && p[1].abs() < 1e-9 && p[2].abs() < 1e-9);
    }

    #[test]
    fn principal_pure_shear_is_plus_minus_tau() {
        // σxy = τ (else 0): eigenvalues τ, 0, −τ.
        let p = principal(0.0, 0.0, 0.0, 7.0, 0.0, 0.0);
        assert!((p[0] - 7.0).abs() < 1e-9, "{:?}", p);
        assert!(p[1].abs() < 1e-9);
        assert!((p[2] + 7.0).abs() < 1e-9);
    }

    #[test]
    fn principal_invariants_match_components() {
        // For an arbitrary symmetric tensor, the principal values must reproduce
        // the trace (I₁) and the von Mises built from either representation.
        let (xx, yy, zz, xy, yz, zx) = (120.0, -40.0, 30.0, 25.0, -15.0, 10.0);
        let p = principal(xx, yy, zz, xy, yz, zx);
        assert!(p[0] >= p[1] && p[1] >= p[2]);
        assert!((p.iter().sum::<f64>() - (xx + yy + zz)).abs() < 1e-6); // I₁
        let vm_p = ((0.5)
            * ((p[0] - p[1]).powi(2) + (p[1] - p[2]).powi(2) + (p[2] - p[0]).powi(2)))
        .sqrt();
        assert!((vm_p - von_mises(xx, yy, zz, xy, yz, zx)).abs() < 1e-6);
    }

    #[test]
    fn pressure_and_triaxiality() {
        // Hydrostatic tension p: mean = p, pressure = −p, von Mises 0 → triax 0 (guarded).
        assert!((pressure(50.0, 50.0, 50.0) + 50.0).abs() < 1e-9);
        assert!(triaxiality(50.0, 50.0, 50.0, 0.0, 0.0, 0.0).abs() < 1e-9);
        // Uniaxial tension σ: mean = σ/3, vm = σ → triax = 1/3.
        assert!((triaxiality(90.0, 0.0, 0.0, 0.0, 0.0, 0.0) - 1.0 / 3.0).abs() < 1e-9);
        // Tresca of uniaxial σ is σ/2.
        assert!((max_shear(90.0, 0.0, 0.0, 0.0, 0.0, 0.0) - 45.0).abs() < 1e-9);
    }
}
