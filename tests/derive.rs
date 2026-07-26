//! Integration test: `#[derive(Keyword)]` / `#[derive(Card)]` produce the same
//! results as the builder / Python classes.
#![allow(dead_code)] // schema structs are declarations; their fields are never read

use dynars::parser::parse_file_blocks;
use dynars::{Card, Keyword, KeywordSchema};

#[derive(Keyword)]
#[keyword("NODE")]
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
    nodes: [i64; 4],
}

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

fn write_deck() -> std::path::PathBuf {
    let node =
        |i: usize, x: f64, y: f64, z: f64| format!("{:>8}{:>16.6}{:>16.6}{:>16.6}\n", i, x, y, z);
    let mut deck = String::from("*KEYWORD\n*NODE\n");
    deck += &node(1, 0.0, 0.0, 0.0);
    deck += &node(2, 1.0, 2.0, 3.0);
    deck += "*ELEMENT_SHELL\n1,10,1,2,3,4\n2,10,5,6,7,8\n";
    deck += "*PART\nsteel\n1,2,3\nalu\n10,20,30\n*END\n";

    let dir = std::env::temp_dir().join(format!("dynars_derive_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("deck.k");
    std::fs::write(&path, deck).unwrap();
    path
}

#[test]
fn derive_reproduces_schema_parsing() {
    let path = write_deck();
    let parsed = parse_file_blocks(&path).unwrap();

    // Single card.
    let nodes = Node::parse(&parsed);
    assert_eq!(nodes.rows(), 2);
    assert_eq!(nodes.column("nid").unwrap().as_int().unwrap(), &[1, 2]);
    assert_eq!(nodes.column("z").unwrap().as_float().unwrap(), &[0.0, 3.0]);

    // Array field -> a 4-wide column.
    let shells = ElementShell::parse(&parsed);
    assert_eq!(shells.rows(), 2);
    let conn = shells.column("nodes").unwrap();
    assert_eq!(conn.rows(), 2);
    assert_eq!(conn.as_int().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);

    // Multi-card via #[cards(...)].
    let parts = Part::parse(&parsed);
    assert_eq!(parts.rows(), 2);
    assert_eq!(
        parts.column("title").unwrap().as_str().unwrap(),
        &["steel", "alu"]
    );
    assert_eq!(parts.column("mid").unwrap().as_int().unwrap(), &[3, 30]);

    // The generated schema equals what the builder would make.
    assert_eq!(Node::schema().keyword, "NODE");
    assert!(Node::schema().repeat);
    assert_eq!(Part::schema().cards.len(), 2);

    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}
