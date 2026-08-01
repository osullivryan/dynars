use dynars::results::{D3plot, NodeField};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = D3plot::open(&a[1]).unwrap();
    for (name, f) in [("temperature", NodeField::Temperature), ("heat_flux", NodeField::HeatFlux)] {
        match d.node_field(f, 0) {
            Some(v) => println!("{name}: len={} first6={:?}", v.len(), &v[..6.min(v.len())]),
            None => println!("{name}: absent"),
        }
    }
}
