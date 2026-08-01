use dynars::results::{D3plot, GlobalField};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = D3plot::open(&a[1]).unwrap();
    for (n, f) in [("kinetic", GlobalField::KineticEnergy), ("internal", GlobalField::InternalEnergy), ("total", GlobalField::TotalEnergy)] {
        match d.global_history(f) {
            Some(v) => println!("{n}: [{:.6},{:.6},{:.6}] .. last {:.6}", v[0], v[1], v[2], v[v.len()-1]),
            None => println!("{n}: absent"),
        }
    }
}
