use dynars::results::{D3plot, PartField, StateBlock};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = D3plot::open(&a[1]).unwrap();
    for (n, f) in [("internal", PartField::InternalEnergy), ("kinetic", PartField::KineticEnergy), ("mass", PartField::Mass), ("hourglass", PartField::HourglassEnergy)] {
        match d.part_field_history(f) {
            Some((v, [ns, np])) => println!("{n}: [{ns}x{np}] last-state row = {:?}", &v[(ns-1)*np..ns*np]),
            None => println!("{n}: absent"),
        }
    }
    for (n, b) in [("solid", StateBlock::Solid), ("shell", StateBlock::Shell)] {
        match d.element_alive(b, d.num_states()-1) {
            Some(v) => println!("{n}_is_alive last state first6 = {:?}", &v[..6.min(v.len())]),
            None => println!("{n}_is_alive: absent"),
        }
    }
}
