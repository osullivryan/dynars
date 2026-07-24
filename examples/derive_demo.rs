//! Marshalling keywords with `#[derive(Keyword)]` — the Rust mirror of the
//! Python `@keyword` classes.
//!
//!     cargo run --release --example derive_demo
//!
//! Declare a keyword as a struct; field *types* imply Int/Float/Str, so you
//! only annotate widths. The derive lowers to the same `Schema` the builder and
//! the Python classes produce.
#![allow(dead_code)] // schema structs are declarations; their fields are never read

use dynars::parser::parse_file_blocks;
use dynars::{Card, Keyword};

#[derive(Keyword)]
#[keyword("NODE")] // repeat defaults to true
struct Node {
    #[field(8)]
    nid: i64,
    #[field(16)]
    x: f64,
    #[field(16)]
    y: f64,
    #[field(16)]
    z: f64,
}

#[derive(Keyword)]
#[keyword("ELEMENT_SHELL")]
struct ElementShell {
    #[field(8)]
    eid: i64,
    #[field(8)]
    pid: i64,
    #[field(8)]
    nodes: [i64; 4], // -> one (N, 4) column
}

// Reusable cards, composed into a multi-card keyword.
#[derive(Card)]
struct Heading {
    #[field(80)]
    title: String,
}

#[derive(Card)]
struct PartData {
    #[field(8)]
    pid: i64,
    #[field(8)]
    secid: i64,
    #[field(8)]
    mid: i64,
}

#[derive(Keyword)]
#[keyword("PART")]
#[cards(Heading, PartData)]
struct Part;

fn main() {
    let mut deck = String::from("*KEYWORD\n*NODE\n");
    for (i, (x, y, z)) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)].iter().enumerate() {
        deck += &format!("{:>8}{:>16.6}{:>16.6}{:>16.6}\n", i + 1, x, y, z);
    }
    deck += "\
*ELEMENT_SHELL
1,10,1,2,3,4
2,10,3,4,1,2
*PART
steel bracket
10,20,1
aluminium panel
11,21,2
*END
";
    let path = std::env::temp_dir().join("dynars_derive_demo.k");
    std::fs::write(&path, &deck).unwrap();
    let parsed = parse_file_blocks(&path).unwrap();

    let nodes = Node::parse(&parsed);
    println!("NODE — {} rows", nodes.rows());
    println!("  nid = {:?}", nodes.column("nid").unwrap().as_int().unwrap());
    println!("  x   = {:?}", nodes.column("x").unwrap().as_float().unwrap());

    let shells = ElementShell::parse(&parsed);
    println!("\nELEMENT_SHELL — {} rows", shells.rows());
    println!("  nodes = {:?}", shells.column("nodes").unwrap().as_int().unwrap());

    let parts = Part::parse(&parsed);
    println!("\nPART — {} rows", parts.rows());
    for (t, pid) in parts
        .column("title")
        .unwrap()
        .as_str()
        .unwrap()
        .iter()
        .zip(parts.column("pid").unwrap().as_int().unwrap())
    {
        println!("  pid {pid}: {t}");
    }

    std::fs::remove_file(&path).ok();
}
