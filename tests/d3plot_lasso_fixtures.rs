//! Regression tests against real LS-DYNA d3plot families taken from lasso-python's
//! test suite (`test/test_data/{d3plot_beamip,order_d3plot}`). The reference values
//! below were cross-checked bit-exact (f32) against lasso 2.0.4. These lock in two
//! fixes that only reproduce on real files: numeric family-member ordering
//! (non-contiguous `d3plot01,02,10,22,100`) and per-file EOF-marker handling
//! (a geometry-only base with states in a continuation file).
use dynars::results::{element, D3plot, GlobalField, PartField, StateBlock};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/d3plot")
        .join(name)
        .join("d3plot")
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn beamip_family_beams_and_eof_per_file() {
    // Geometry-only base + 2 states in d3plot01 (EOF marker after base geometry).
    let d = D3plot::open(fixture("beamip")).unwrap();
    assert_eq!(d.num_states(), 2, "base geometry-only; states live in d3plot01");
    let c = d.control();
    assert_eq!((c.numnp, c.nel2), (2, 1));

    // Beam cross-section resultants at state 1 (lasso-validated).
    let rec = d.element_result(StateBlock::Beam, 1, 0).unwrap();
    let br = element::beam_resultants(&rec).unwrap();
    assert!(approx(br.bending_moment[0], -0.009_219_318_628_311_157, 1e-9), "{br:?}");
    assert!(approx(br.bending_moment[1], 0.001_209_799_200_296_402, 1e-9), "{br:?}");
    assert!(approx(br.shear_force[0], 2.402_827_703_917_864_7e-6, 1e-12));
    assert!(approx(br.axial_force, 4.797_982_32e-12, 1e-15));
}

#[test]
fn order_family_numeric_member_sort() {
    // Members d3plot, 01, 02, 10, 11, 12, 22, 100 (gaps) must sort numerically → 7 states.
    let d = D3plot::open(fixture("order")).unwrap();
    assert_eq!(d.num_states(), 7, "non-contiguous family must not truncate at a gap");
    let c = d.control();
    assert_eq!((c.numnp, c.nel8, c.nel4), (106, 16, 16));

    // Solid + shell von Mises at the last state (lasso-validated, f32 precision).
    let s = d.num_states() - 1;
    let solid = d.element_result(StateBlock::Solid, s, 0).unwrap();
    assert!(approx(element::von_mises_stress(&solid), 165.891_65, 1e-2));
    let ly = d.shell_layout();
    assert_eq!(ly.n_layers, 5);
    let shell = d.element_result(StateBlock::Shell, s, 0).unwrap();
    assert!(approx(
        element::shell_von_mises(&shell, &ly, element::LayerSelect::Max),
        326.647_36,
        1e-2
    ));
}

#[test]
fn order_family_globals_parts_and_deletion() {
    let d = D3plot::open(fixture("order")).unwrap();

    // Global energy histories, 7 states each.
    let ke = d.global_history(GlobalField::KineticEnergy).unwrap();
    let ie = d.global_history(GlobalField::InternalEnergy).unwrap();
    assert_eq!((ke.len(), ie.len()), (7, 7));
    assert_eq!(ke[0], 0.0);
    assert!(ie[6] > ie[0]); // internal energy grows

    // Per-part energy matrix (n_states, n_parts).
    let mass_block = d.part_field_history(PartField::Mass).unwrap();
    assert_eq!(mass_block.dims(), [7, 4]);
    let mass = mass_block.data();
    // Mass is constant over time.
    for s in 1..7 {
        for p in 0..4 {
            assert!(approx(mass[s * 4 + p], mass[p], 1e-9));
        }
    }

    // Element deletion flags present (mdlopt 2): one per element.
    let alive = d.element_alive(StateBlock::Solid, 6).unwrap();
    assert_eq!(alive.len(), 16);
}
