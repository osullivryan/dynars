//! End-to-end d3plot in Rust: write a model, read it back, edit it in place.
//!
//!     cargo run --release --example d3plot_demo
//!
//! Covers: mesh + connectivity, real node/element/part IDs (NARBS), per-state
//! nodal results (displacement/velocity), element results with **custom history
//! variables**, generic block extraction, and in-place editing.

use dynars::results::{BlockArray, D3plot, D3plotEditor, D3plotWriter, StateBlock};

fn main() {
    let path = std::env::temp_dir().join("dynars_demo.d3plot");

    // --- 1. WRITE a model ------------------------------------------------
    // 8 nodes: a unit cube. One hex solid + one quad shell (its bottom face).
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // bottom
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, // top
    ];
    let mut w = D3plotWriter::new(coords.clone()).unwrap();
    w.set_title("dynars demo model");
    w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1); // hex, part index 1
    w.add_shell([1, 2, 3, 4], 2); // quad, part index 2

    // Real IDs, written into the NARBS numbering section.
    w.set_node_ids((101..=108).collect());
    w.set_solid_ids(vec![9001]);
    w.set_shell_ids(vec![7001]);
    w.set_part_ids(vec![10, 20]);

    // Per-solid results, one integration point, solver order:
    // [6 stress, 1 effective plastic strain, 2 CUSTOM history vars] = 9 vars.
    let n_states = 3;
    let mut solid = Vec::new();
    for s in 0..n_states {
        solid.extend_from_slice(&[
            1.0,
            2.0,
            3.0,
            0.5,
            0.5,
            0.5,             // stress
            0.01 * s as f64, // plastic strain
            42.0,            // custom field #1 (constant)
            10.0 * s as f64, // custom field #2 (grows with time)
        ]);
    }
    w.set_solid_results(9, solid);

    // States: time + deformed coordinates (+ optional velocity/acceleration).
    for s in 0..n_states {
        let dz = 0.1 * s as f64;
        let disp: Vec<f64> = coords
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2] + dz])
            .collect();
        let vel = vec![0.0; coords.len()];
        w.add_state(s as f64 * 1e-3, disp, Some(vel), None).unwrap();
    }
    w.write(&path).unwrap();
    println!("wrote {}", path.display());

    // --- 2. READ it back -------------------------------------------------
    let d = D3plot::open(&path).unwrap();
    println!(
        "\nread: {} nodes, {} states, times {:?}",
        d.num_nodes(),
        d.num_states(),
        d.times()
    );

    let (conn, parts) = d.solid_connectivity();
    println!("solid connectivity {conn:?}, parts {parts:?}");
    println!("node_ids {:?}, part_ids {:?}", d.node_ids(), d.part_ids());

    let last = d.node_coordinates(d.num_states() - 1).unwrap();
    println!("node0 z at final state = {}", last[2]);
    println!(
        "peak displacement = {:.4}",
        d.max_displacement_final().unwrap()
    );

    // Generic block extraction: raw (n_states, count, vars). The custom fields
    // are columns 7 and 8 (after 6 stress + 1 plastic strain).
    let all = d.resolve_states(None).unwrap();
    if let Some((BlockArray::F32(v), dims)) = d.block_data(StateBlock::Solid, &all) {
        println!("solid block dims {dims:?}");
        println!("  custom #1 (col 7), state0 = {}", v[7]);
        println!(
            "  custom #2 (col 8), state2 = {}",
            v[2 * dims[1] * dims[2] + 8]
        );
    }

    // --- 3. EDIT in place ------------------------------------------------
    let mut e = D3plotEditor::open(&path).unwrap();
    e.set_node_coordinates(0, &vec![9.0f32; d.num_nodes() * 3])
        .unwrap();
    e.save().unwrap();
    let d2 = D3plot::open(&path).unwrap();
    println!(
        "\nafter edit, state0 node0 = {:?}",
        &d2.node_coordinates(0).unwrap()[..3]
    );

    std::fs::remove_file(&path).ok();
    println!("\nOK");
}
