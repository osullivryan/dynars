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

// ── Beams ───────────────────────────────────────────────────────────────────
//
// A beam result record leads with 6 cross-section resultants, then per
// integration-point stress/strain history. These extract the resultants.

/// Beam cross-section force/moment resultants (the first 6 words of a beam record).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamResultants {
    /// Axial force `N`.
    pub axial_force: f64,
    /// Transverse shear forces `Qs, Qt`.
    pub shear_force: [f64; 2],
    /// Bending moments `Ms, Mt`.
    pub bending_moment: [f64; 2],
    /// Torsional moment `T`.
    pub torsion_moment: f64,
}

/// Cross-section resultants of a beam record, or `None` if it is too short.
pub fn beam_resultants(elem: &[f64]) -> Option<BeamResultants> {
    let r = elem.get(..6)?;
    Some(BeamResultants {
        axial_force: r[0],
        shear_force: [r[1], r[2]],
        bending_moment: [r[3], r[4]],
        torsion_moment: r[5],
    })
}

/// Beam axial force (word 0) — a ready extractor for the reductions.
pub fn beam_axial_force(elem: &[f64]) -> f64 {
    elem.first().copied().unwrap_or(0.0)
}

// ── Shell through-thickness layers ──────────────────────────────────────────
//
// A shell element's result record packs `n_layers` integration points at the
// front, each `stride` words = 6 stress (if `has_stress`) + 1 plastic strain (if
// `has_pstrain`) + `neips` history, followed by element-level resultants. These
// helpers pick a layer (or reduce over layers) so the generic reductions can run
// on shells via a closure: `|rec| element::shell_von_mises(rec, &layout, sel)`.

/// Which through-thickness integration point a shell criterion reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerSelect {
    /// Outermost layer (last integration point, index `n_layers-1`).
    Top,
    /// Innermost layer (first integration point, index 0).
    Bottom,
    /// Middle layer (`n_layers / 2`).
    Mid,
    /// A specific 0-based layer index (clamped to the last layer).
    Index(usize),
    /// The worst layer — reduce the quantity's max across all layers.
    Max,
}

/// Layout of a shell result record's layer block (from the d3plot control words).
/// Build it with `D3plot::shell_layout()`.
#[derive(Debug, Clone, Copy)]
pub struct ShellLayout {
    /// Number of through-thickness integration points.
    pub n_layers: usize,
    /// Words per layer (`6·has_stress + has_pstrain + neips`).
    pub stride: usize,
    /// The 6 stress components lead each layer.
    pub has_stress: bool,
    /// Effective plastic strain follows the stresses in each layer.
    pub has_pstrain: bool,
    /// Element-level force resultants (8: moments/shear/normal) follow the layers.
    pub has_forces: bool,
    /// Element-level "extra" (thickness + 2 energy words) follow the resultants.
    pub has_extra: bool,
}

/// Element-level (nonlayer) shell force resultants — one set per shell, after the
/// through-thickness layer block. Membrane forces, bending moments, transverse
/// shear (per unit width), plus thickness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellResultants {
    /// Bending moments `Mx, My, Mxy`.
    pub bending_moment: [f64; 3],
    /// Transverse shear forces `Qx, Qy`.
    pub shear_force: [f64; 2],
    /// Membrane (normal) forces `Nx, Ny, Nxy`.
    pub normal_force: [f64; 3],
    /// Element thickness (`None` if the "extra" block is absent).
    pub thickness: Option<f64>,
}

impl ShellLayout {
    /// Word offset where the element-level resultant block begins.
    fn nonlayer_off(&self) -> usize {
        self.n_layers * self.stride
    }
    /// Resolve a selector to a concrete layer index; `None` means "reduce over
    /// all layers" (the `Max` selector) or "no layers".
    fn resolve(&self, sel: LayerSelect) -> Option<usize> {
        if self.n_layers == 0 {
            return None;
        }
        Some(match sel {
            LayerSelect::Bottom => 0,
            LayerSelect::Top => self.n_layers - 1,
            LayerSelect::Mid => self.n_layers / 2,
            LayerSelect::Index(k) => k.min(self.n_layers - 1),
            LayerSelect::Max => return None,
        })
    }

    /// The 6 stress components at `layer`, or `None` if stress is absent or the
    /// record is too short.
    pub fn layer_stress(&self, rec: &[f64], layer: usize) -> Option<[f64; 6]> {
        if !self.has_stress || layer >= self.n_layers {
            return None;
        }
        let off = layer * self.stride;
        let s = rec.get(off..off + 6)?;
        Some([s[0], s[1], s[2], s[3], s[4], s[5]])
    }

    /// Effective plastic strain at `layer` (the word after the 6 stresses).
    pub fn layer_pstrain(&self, rec: &[f64], layer: usize) -> Option<f64> {
        if !self.has_pstrain || layer >= self.n_layers {
            return None;
        }
        let off = layer * self.stride + if self.has_stress { 6 } else { 0 };
        rec.get(off).copied()
    }

    /// Element-level force resultants (moments/shear/normal + thickness), or
    /// `None` if the record carries no resultant block (`ioshl3 == 0`).
    pub fn resultants(&self, rec: &[f64]) -> Option<ShellResultants> {
        if !self.has_forces {
            return None;
        }
        let b = self.nonlayer_off();
        let f = rec.get(b..b + 8)?; // 3 moment + 2 shear + 3 normal
        Some(ShellResultants {
            bending_moment: [f[0], f[1], f[2]],
            shear_force: [f[3], f[4]],
            normal_force: [f[5], f[6], f[7]],
            thickness: if self.has_extra { rec.get(b + 8).copied() } else { None },
        })
    }
}

/// Von Mises stress of a shell record at the selected layer (`Max` → worst layer).
pub fn shell_von_mises(rec: &[f64], layout: &ShellLayout, sel: LayerSelect) -> f64 {
    let vm = |s: [f64; 6]| von_mises(s[0], s[1], s[2], s[3], s[4], s[5]);
    match layout.resolve(sel) {
        Some(l) => layout.layer_stress(rec, l).map(vm).unwrap_or(0.0),
        None => (0..layout.n_layers)
            .filter_map(|l| layout.layer_stress(rec, l))
            .map(vm)
            .fold(0.0, f64::max),
    }
}

/// Effective plastic strain of a shell record at the selected layer (`Max` → worst).
pub fn shell_plastic_strain(rec: &[f64], layout: &ShellLayout, sel: LayerSelect) -> f64 {
    match layout.resolve(sel) {
        Some(l) => layout.layer_pstrain(rec, l).unwrap_or(0.0),
        None => (0..layout.n_layers)
            .filter_map(|l| layout.layer_pstrain(rec, l))
            .fold(0.0, f64::max),
    }
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

    #[test]
    fn shell_layer_selection_and_max() {
        // 3 layers, stride = 6 stress + 1 pstrain + 0 history = 7. Each layer
        // uniaxial (sxx only) so von Mises == |sxx|: bottom=100, mid=300, top=200.
        let layout = ShellLayout {
            n_layers: 3,
            stride: 7,
            has_stress: true,
            has_pstrain: true,
            has_forces: false,
            has_extra: false,
        };
        let uni = |sxx: f64, eps: f64| [sxx, 0.0, 0.0, 0.0, 0.0, 0.0, eps];
        let mut rec = Vec::new();
        rec.extend_from_slice(&uni(100.0, 0.01)); // layer 0 (bottom)
        rec.extend_from_slice(&uni(300.0, 0.03)); // layer 1 (mid)
        rec.extend_from_slice(&uni(200.0, 0.02)); // layer 2 (top)

        assert!((shell_von_mises(&rec, &layout, LayerSelect::Bottom) - 100.0).abs() < 1e-9);
        assert!((shell_von_mises(&rec, &layout, LayerSelect::Mid) - 300.0).abs() < 1e-9);
        assert!((shell_von_mises(&rec, &layout, LayerSelect::Top) - 200.0).abs() < 1e-9);
        assert!((shell_von_mises(&rec, &layout, LayerSelect::Index(0)) - 100.0).abs() < 1e-9);
        assert!((shell_von_mises(&rec, &layout, LayerSelect::Index(9)) - 200.0).abs() < 1e-9); // clamps to top
        assert!((shell_von_mises(&rec, &layout, LayerSelect::Max) - 300.0).abs() < 1e-9);

        assert!((shell_plastic_strain(&rec, &layout, LayerSelect::Bottom) - 0.01).abs() < 1e-9);
        assert!((shell_plastic_strain(&rec, &layout, LayerSelect::Top) - 0.02).abs() < 1e-9);
        assert!((shell_plastic_strain(&rec, &layout, LayerSelect::Max) - 0.03).abs() < 1e-9);

        // pstrain sits after the 6 stresses within the selected layer.
        assert_eq!(layout.layer_pstrain(&rec, 1), Some(0.03));

        // Resultant (nonlayer) level: 1 layer (stride 7), then 8 force words + thickness.
        let rlay = ShellLayout {
            n_layers: 1,
            stride: 7,
            has_stress: true,
            has_pstrain: true,
            has_forces: true,
            has_extra: true,
        };
        let mut r2 = uni(50.0, 0.0).to_vec();
        r2.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.9, 0.0, 0.0]);
        let res = rlay.resultants(&r2).unwrap();
        assert_eq!(res.bending_moment, [1.0, 2.0, 3.0]);
        assert_eq!(res.shear_force, [4.0, 5.0]);
        assert_eq!(res.normal_force, [6.0, 7.0, 8.0]);
        assert_eq!(res.thickness, Some(0.9));
        assert!(layout.resultants(&rec).is_none()); // no force block
        // no-stress layout: stress reads return None, von Mises falls to 0.
        let nostress = ShellLayout {
            n_layers: 2,
            stride: 1,
            has_stress: false,
            has_pstrain: true,
            has_forces: false,
            has_extra: false,
        };
        assert_eq!(nostress.layer_stress(&[0.5, 0.7], 0), None);
        assert_eq!(shell_von_mises(&[0.5, 0.7], &nostress, LayerSelect::Max), 0.0);
    }
}
