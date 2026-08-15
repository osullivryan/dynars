//! Write a synthetic solid d3plot for cross-reader benchmarking. Args:
//!   cargo run --release --example gen_solid_d3plot -- <n_elem> <n_states> <path>
use dynars::results::{D3plotWriter, ResultBlock};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_elem: usize = a[1].parse().unwrap();
    let n_states: usize = a[2].parse().unwrap();
    let path = &a[3];
    let nv = 7usize;

    let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
    let mut w = D3plotWriter::new(nodes.clone()).unwrap();
    for _ in 0..n_elem {
        w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
    }
    w.set_part_ids(vec![7]);
    let mut data = vec![0.0f64; n_elem * n_states * nv];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i * 2_654_435_761usize) % 997) as f64 / 7.0;
    }
    w.set_solid_results(ResultBlock::new([n_states, n_elem, nv], data));
    for s in 0..n_states {
        let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
        w.add_state(s as f64, disp, None, None).unwrap();
    }
    w.write(path).unwrap();
    println!("wrote {n_elem} solids × {n_states} states → {path}");
}
