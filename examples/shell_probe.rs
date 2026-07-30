//! Probe a shell d3plot's layer layout and print per-layer von Mises for the
//! first few shells, for cross-checking against lasso.
//! Args: cargo run --release --example shell_probe -- <path> [n_elems]
use dynars::results::{
    element::{self, LayerSelect},
    D3plot, StateBlock,
};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = D3plot::open(&a[1]).unwrap();
    let n = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3usize);
    let c = d.control();
    println!(
        "nel4={} nv2d={} maxint={} neips={} ioshl=[{},{},{},{}]",
        c.nel4, c.nv2d, c.maxint, c.neips, c.ioshl1, c.ioshl2, c.ioshl3, c.ioshl4
    );
    println!(
        "it={} iu={} iv={} ia={} nglbv={} numnp={} | nel8={} nv3d={} nel2={} nv1d={} nelth={} nv3dt={}",
        c.it, c.iu, c.iv, c.ia, c.nglbv, c.numnp, c.nel8, c.nv3d, c.nel2, c.nv1d, c.nelth, c.nv3dt
    );
    let raw = d.element_result(StateBlock::Shell, 0, 0).unwrap();
    println!("elem0 raw[0..8]={:?}", &raw[..8.min(raw.len())]);
    let ly = d.shell_layout();
    println!(
        "shell_layout: n_layers={} stride={} has_stress={} has_pstrain={} has_forces={} has_extra={}",
        ly.n_layers, ly.stride, ly.has_stress, ly.has_pstrain, ly.has_forces, ly.has_extra
    );
    println!("states={}", d.num_states());
    for e in 0..n.min(c.nel4) {
        let rec = d.element_result(StateBlock::Shell, 0, e).unwrap();
        let per_layer: Vec<f64> = (0..ly.n_layers)
            .map(|l| ly.layer_stress(&rec, l).map(|s| element::von_mises_stress(&s)).unwrap_or(0.0))
            .collect();
        println!(
            "elem {e}: vM/layer={:?}  bottom={:.4} mid={:.4} top={:.4} max={:.4}",
            per_layer,
            element::shell_von_mises(&rec, &ly, LayerSelect::Bottom),
            element::shell_von_mises(&rec, &ly, LayerSelect::Mid),
            element::shell_von_mises(&rec, &ly, LayerSelect::Top),
            element::shell_von_mises(&rec, &ly, LayerSelect::Max),
        );
    }
}
