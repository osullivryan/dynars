//! Time our d3plot open + streaming von-Mises part-max history on an existing
//! file. Args: cargo run --release --example d3plot_read -- <path> <part>
use dynars::results::{element, D3plot, StateBlock};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let part: i64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut open_best = f64::INFINITY;
    let mut red_best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..4 {
        let t = Instant::now();
        let d = D3plot::open(path).unwrap();
        open_best = open_best.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        out = d.part_max_history(StateBlock::Solid, part, element::von_mises_stress).unwrap();
        red_best = red_best.min(t.elapsed().as_secs_f64());
    }
    println!("dynars  open {:6.1} ms   vm-part-max {:6.1} ms   states={}  vm[0]={:.4}",
             open_best * 1e3, red_best * 1e3, out.len(), out.first().copied().unwrap_or(0.0));
}
