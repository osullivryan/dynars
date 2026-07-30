//! Benchmark element part-reductions on a large block. Run:
//! `cargo run --release --example element_bench`
use dynars::results::element;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    // 100 states × 300k solids × 7 vars (6 stress + eff plastic strain) ≈ 210M f64 (1.7 GB).
    let (n_states, n_elem, nv) = (100usize, 300_000usize, 7usize);
    let part_ids: Vec<i64> = (0..n_elem).map(|e| (e % 4) as i64).collect(); // 4 parts, interleaved (worst-case gather)
    let mut data = vec![0.0f64; n_states * n_elem * nv];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 2654435761usize) % 1000) as f64 / 100.0; // cheap pseudo-fill
    }
    let part = 1;
    let n_in_part = part_ids.iter().filter(|&&p| p == part).count();
    println!("block: {n_states} states × {n_elem} elem × {nv} vars ; part {part} has {n_in_part} elems");

    // serial reference (idx loop, no rayon), von Mises max per state
    let idx: Vec<usize> = (0..n_elem).filter(|&e| part_ids[e] == part).collect();
    let t = Instant::now();
    let serial: Vec<f64> = (0..n_states)
        .map(|s| {
            idx.iter().fold(0.0f64, |m, &e| {
                let b = s * n_elem * nv + e * nv;
                m.max(element::von_mises_stress(&data[b..b + nv]))
            })
        })
        .collect();
    let ser_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let par = element::part_max_history(&data, n_states, n_elem, nv, &part_ids, part, element::von_mises_stress);
    let par_ms = t.elapsed().as_secs_f64() * 1e3;

    assert_eq!(serial, par);
    println!("von_mises part_max_history:  serial {ser_ms:7.1} ms   rayon {par_ms:7.1} ms   {:.1}x", ser_ms / par_ms);
    let _ = par.par_iter().count();
}
