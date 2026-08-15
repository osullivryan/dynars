//! Cross-tool interop harness for d3plot read/write against open-lasso-python.
//!
//! Usage:
//!   cargo run --example lasso_interop -- write <path>   # our writer  -> file (scheme A)
//!   cargo run --example lasso_interop -- read  <path>   # our reader  <- file (scheme B, asserts)
//!
//! The companion `examples/lasso_interop.py` reads scheme A and writes scheme B,
//! so the two tools validate each other in both directions with matching values.

use dynars::results::{D3plot, D3plotWriter, GlobalField, NodeField, ResultBlock, StateBlock};

// ---- deterministic value schemes shared with the Python side ----
const NUMNP: usize = 8;
const NSTATES: usize = 2;

fn coord(i: usize) -> f64 {
    i as f64 // node n coords = (3n, 3n+1, 3n+2)
}

fn write_scheme_a(path: &str, double: bool) {
    let coords: Vec<f64> = (0..NUMNP * 3).map(coord).collect();
    let mut w = D3plotWriter::new(coords.clone()).unwrap();
    w.set_double_precision(double);
    w.set_title("dynars<->lasso interop A");
    // one hex solid over all 8 nodes, one quad shell over the first 4, one beam.
    w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
    w.add_shell([1, 2, 3, 4], 2);
    w.add_beam([1, 2, 3], 3);

    // per-element results.
    let mut solid = Vec::new();
    let mut shell = Vec::new();
    let mut beam = Vec::new();
    for s in 0..NSTATES {
        // solids/shells: 7 vars = 6 stress + 1 effective plastic strain.
        for v in 0..7 {
            solid.push((1000 * s + v) as f64);
        }
        for v in 0..7 {
            shell.push((2000 * s + v) as f64);
        }
        // beams: nv1d = 6 (axial, shear s/t, moment s/t, torsion).
        for v in 0..6 {
            beam.push((3000 * s + v) as f64);
        }
    }
    w.set_solid_results(ResultBlock::new([NSTATES, 1, 7], solid));
    w.set_shell_results(ResultBlock::new([NSTATES, 1, 7], shell)); // one layer: 6 stress + pstrain
    w.set_beam_results(ResultBlock::new([NSTATES, 1, 6], beam));

    // global energies.
    w.set_global_history(GlobalField::KineticEnergy, vec![10.0, 11.0]);
    w.set_global_history(GlobalField::InternalEnergy, vec![20.0, 21.0]);
    w.set_global_history(GlobalField::TotalEnergy, vec![30.0, 31.0]);

    // nodal temperature (IT=1).
    let temp: Vec<f64> = (0..NSTATES)
        .flat_map(|s| (0..NUMNP).map(move |n| (7000 + 100 * s + n) as f64))
        .collect();
    w.set_node_field(NodeField::Temperature, temp);

    for s in 0..NSTATES {
        let disp: Vec<f64> = coords.iter().map(|&c| c + s as f64).collect();
        let vel: Vec<f64> = vec![0.5; NUMNP * 3];
        w.add_state(s as f64, disp, Some(vel), None).unwrap();
    }
    w.write(path).unwrap();
    println!("wrote scheme A -> {path}");
}

/// Read a lasso-written file (scheme B) with our reader and assert its contents.
/// Scheme B: NUMNP nodes with coords[i]=i, 2 states, disp = coords + state,
/// velocity = 0.25, one solid [1..8] part 1 with stress 100*s+v (v 0..5) and
/// pstrain 100*s+6, global TE = [7,8].
fn read_scheme_b(path: &str) {
    let d = D3plot::open(path).unwrap();
    let approx = |a: f64, b: f64, what: &str| {
        assert!((a - b).abs() < 1e-4, "{what}: {a} != {b}");
    };
    assert_eq!(d.num_nodes(), NUMNP, "numnp");
    assert_eq!(d.num_states(), NSTATES, "nstates");

    // initial coordinates
    let c0 = d.node_coordinates(0).unwrap();
    for (i, &v) in c0.iter().enumerate() {
        approx(v, i as f64, "coord");
    }
    // displacement (deformed coords) at state 1 = coords + 1
    let c1 = d.node_coordinates(1).unwrap();
    for (i, &v) in c1.iter().enumerate() {
        approx(v, i as f64 + 1.0, "disp state1");
    }
    // velocity block = 0.25 everywhere
    let all = d.resolve_states(None).unwrap();
    let (vel, vdims) = d.block_data(StateBlock::Velocity, &all).unwrap();
    assert_eq!(vdims, [NSTATES, NUMNP, 3], "vel dims");
    for x in vel.to_f64() {
        approx(x, 0.25, "vel");
    }
    // solid raw block = [6 stress, pstrain] per state, values 100*s + v
    let (solid, sdims) = d.block_data(StateBlock::Solid, &all).unwrap();
    assert_eq!(sdims, [NSTATES, 1, 7], "solid dims");
    let sd = solid.to_f64();
    for s in 0..NSTATES {
        for v in 0..7 {
            approx(sd[s * 7 + v], (100 * s + v) as f64, "solid var");
        }
    }
    // global total energy = [7, 8]
    let te = d.global_history(GlobalField::TotalEnergy).unwrap();
    approx(te[0], 7.0, "TE0");
    approx(te[1], 8.0, "TE1");

    println!("read scheme B OK: node coords/disp/vel, solid stress+pstrain, global energy all match");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, path) = (args.get(1).map(String::as_str), args.get(2).map(String::as_str));
    match (mode, path) {
        (Some("write"), Some(p)) => write_scheme_a(p, false),
        (Some("write8"), Some(p)) => write_scheme_a(p, true), // double precision
        (Some("read"), Some(p)) => read_scheme_b(p),
        _ => {
            eprintln!("usage: lasso_interop <write|write8|read> <path>");
            std::process::exit(2);
        }
    }
}
