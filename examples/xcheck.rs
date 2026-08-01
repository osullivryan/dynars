//! Dump key d3plot values for cross-checking against lasso.
//! cargo run --release --example xcheck -- <path>
use dynars::results::{element, D3plot, StateBlock};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = D3plot::open(&a[1]).unwrap();
    let c = d.control();
    let ns = d.num_states();
    println!("states={ns} numnp={} nel8={} nel4={} nel2={} nelth={}", c.numnp, c.nel8, c.nel4, c.nel2, c.nelth);
    // solid von Mises, elem 0, states 0 and last
    if c.nel8 > 0 {
        for s in [0, ns - 1] {
            let r = d.element_result(StateBlock::Solid, s, 0).unwrap();
            println!("solid vM e0 s{s} = {:.6}", element::von_mises_stress(&r));
        }
    }
    // shell von Mises (max layer), elem 0, states 0 and last
    if c.nel4 > 0 {
        let ly = d.shell_layout();
        for s in [0, ns - 1] {
            let r = d.element_result(StateBlock::Shell, s, 0).unwrap();
            println!("shell vM(max) e0 s{s} = {:.6}  (layers={})", element::shell_von_mises(&r, &ly, element::LayerSelect::Max), ly.n_layers);
        }
    }
    // node coordinate of node 0 at last state
    let x = d.node_coordinates(ns - 1).unwrap();
    println!("node0 coord s{} = [{:.6},{:.6},{:.6}]", ns - 1, x[0], x[1], x[2]);
}
