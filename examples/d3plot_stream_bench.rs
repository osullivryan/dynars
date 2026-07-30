//! Streaming element reductions at scale. Run:
//!   cargo run --release --example d3plot_stream_bench
//!
//! Real models have *dozens of millions* of elements. Materializing the whole
//! result block as f64 is a non-starter there (30M elem × 50 states × 7 vars ×
//! 8 B ≈ 168 GB). This bench compares the two reader paths on a d3plot big
//! enough to be honest:
//!   * `element_block_f64`  — copies the block into an f64 Vec, then reduces.
//!   * `part_max_history`   — streams straight off the mmap, no materialization.
use dynars::results::{element, D3plot, D3plotWriter, StateBlock};
use std::time::Instant;

fn main() {
    // Feasible stand-in for a large model. Element reads in a reduction scale as
    // n_elem × n_states, so this exercises ~N·S element decodes per pass.
    let (n_elem, n_states, nv) = (2_000_000usize, 10usize, 7usize);
    let reads = n_elem * n_states;
    let file_mb = reads * nv * 4 / (1 << 20);
    println!("building d3plot: {n_elem} solids × {n_states} states × {nv} vars  (~{file_mb} MB f32 on disk)");

    // 8 shared nodes; every solid references them (connectivity validity is not
    // what we're measuring). One part so the reduction touches every element.
    let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
    let mut w = D3plotWriter::new(nodes.clone()).unwrap();
    for _ in 0..n_elem {
        w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
    }
    w.set_part_ids(vec![7]);
    // von Mises varies element-to-element so max isn't trivially constant.
    let mut data = vec![0.0f64; reads * nv];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 2_654_435_761usize) % 997) as f64 / 7.0;
    }
    w.set_solid_results(nv, data);
    for s in 0..n_states {
        let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
        w.add_state(s as f64, disp, None, None).unwrap();
    }
    let p = std::env::temp_dir().join("dynars_stream_bench.d3plot");
    w.write(&p).unwrap();

    let d = D3plot::open(&p).unwrap();

    // Warm the page cache, then take the best of a few passes.
    let bench = |f: &dyn Fn() -> Vec<f64>| -> (f64, Vec<f64>) {
        let mut best = f64::INFINITY;
        let mut out = Vec::new();
        for _ in 0..4 {
            let t = Instant::now();
            out = f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        (best, out)
    };

    // Materialize-as-f64, then reduce (the path that explodes at real scale).
    let (mat_s, vm_mat) = bench(&|| {
        let (blk, dims, parts) = d.element_block_f64(StateBlock::Solid).unwrap();
        element::part_max_history(&blk, dims[0], dims[1], dims[2], &parts, 1, element::von_mises_stress)
    });

    // Stream straight off the mmap.
    let (str_s, vm_str) = bench(&|| {
        d.part_max_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap()
    });

    assert_eq!(vm_mat, vm_str, "streaming must equal materialized");
    let mreads = reads as f64 / 1e6;
    println!("\nvon_mises part_max_history over {mreads:.0}M element-reads (warm):");
    println!("  materialize f64 + reduce : {:7.1} ms   ({:6.0} M elem/s)", mat_s * 1e3, mreads / mat_s);
    println!("  stream off mmap          : {:7.1} ms   ({:6.0} M elem/s)   {:.2}× faster, 0 extra bytes",
             str_s * 1e3, mreads / str_s, mat_s / str_s);
    println!(
        "\nextrapolated to a 30M-elem × 50-state model: materialize would need \
         ~{:.0} GB of f64 (infeasible); streaming moves ~{:.0} GB off the mmap \
         and stays bandwidth-bound.",
        30e6 * 50.0 * 7.0 * 8.0 / 1e9,
        30e6 * 50.0 * 7.0 * 4.0 / 1e9,
    );

    // Full per-element history matrix for the whole part (the "every element over
    // time" path). Bounded by part size: here n_states × n_elem × 8 B.
    let (hist_s, mat) = bench(&|| {
        d.part_element_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap().0
    });
    let mat_gb = mat.len() as f64 * 8.0 / 1e9;
    println!(
        "\npart_element_history (every element's vM over time): {:7.1} ms   ({:6.0} M elem/s)   matrix {:.2} GB",
        hist_s * 1e3, mreads / hist_s, mat_gb
    );

    let _ = std::fs::remove_file(&p);
}
