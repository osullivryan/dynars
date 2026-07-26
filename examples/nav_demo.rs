//! Navigate a deck: part.material(), part.section().
//! Usage: cargo run --release --example nav_demo -- <main.k>
use dynars::deck::parse_deck;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let deck = parse_deck(std::path::Path::new(&path)).unwrap();
    for pid in [1, 2, 3, 4] {
        let Some(part) = deck.part(pid) else { continue };
        let mat = part.material();
        let sec = part.section();
        println!("PART {pid} ({:?})  @ {}:{}", part.field("heading").map(|f| f.value()), part.file().display(), part.line());
        if let Some(m) = &mat {
            println!("   material -> id {} [{}]  ro={:?} e={:?}", m.id().unwrap_or(0), m.name(), m.field("RO").map(|f| f.value()), m.field("E").map(|f| f.value()));
        }
        if let Some(s) = &sec {
            println!("   section  -> id {} [{}]  elform={:?} nip={:?}", s.id().unwrap_or(0), s.name(), s.field("ELFORM").map(|f| f.value()), s.field("NIP").map(|f| f.value()));
        }
    }
}
