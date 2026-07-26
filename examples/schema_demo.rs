//! Marshalling keywords with user-defined schemas — Rust builder API.
//!
//!     cargo run --release --example schema_demo
//!
//! A `Schema` is a keyword name + an ordered list of `Card`s (lines), each an
//! ordered list of typed fields. It parses into a columnar `Table`. This is the
//! same schema the Python `@keyword` classes lower to — one parser underneath.

use dynars::parser::parse_file_blocks;
use dynars::schema::{Card, Schema, parse_schema};

fn main() {
    // A small deck. *NODE is fixed-width (I8 + three E16); the rest use
    // comma-free format — the parser handles both, per line.
    let mut deck = String::from("*KEYWORD\n*NODE\n");
    for (i, (x, y, z)) in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        .iter()
        .enumerate()
    {
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
*MAT_ELASTIC
1,7.85e-9,210000.0,0.3
*END
";

    let path = std::env::temp_dir().join("dynars_schema_demo.k");
    std::fs::write(&path, &deck).unwrap();
    let parsed = parse_file_blocks(&path).unwrap();

    // --- *NODE: one card, repeats over the block (the default) ---
    let nodes = parse_schema(
        &parsed,
        &Schema::new("NODE").card(
            Card::new()
                .int("nid", 8)
                .float("x", 16)
                .float("y", 16)
                .float("z", 16),
        ),
    );
    println!("NODE — {} rows", nodes.rows());
    println!(
        "  nid = {:?}",
        nodes.column("nid").unwrap().as_int().unwrap()
    );
    println!(
        "  x   = {:?}",
        nodes.column("x").unwrap().as_float().unwrap()
    );

    // --- *ELEMENT_SHELL: connectivity as one 4-wide array column ---
    let shells = parse_schema(
        &parsed,
        &Schema::new("ELEMENT_SHELL").card(
            Card::new()
                .int("eid", 8)
                .int("pid", 8)
                .int_array("nodes", 4, 8),
        ),
    );
    let conn = shells.column("nodes").unwrap().as_int().unwrap();
    println!("\nELEMENT_SHELL — {} rows", shells.rows());
    println!(
        "  eids  = {:?}",
        shells.column("eid").unwrap().as_int().unwrap()
    );
    println!("  nodes = {:?}  (row-major {}x4)", conn, shells.rows());

    // --- *PART: multi-card — a reusable title card + a data card ---
    let title = Card::new().str("title", 80);
    let data = Card::new().int("pid", 8).int("secid", 8).int("mid", 8);
    let parts = parse_schema(&parsed, &Schema::new("PART").card(title).card(data));
    println!("\nPART — {} rows", parts.rows());
    let titles = parts.column("title").unwrap().as_str().unwrap();
    let pids = parts.column("pid").unwrap().as_int().unwrap();
    for (t, pid) in titles.iter().zip(pids) {
        println!("  pid {pid}: {t}");
    }

    // --- *MAT_ELASTIC: one entity per block (still works with the default) ---
    let mats = parse_schema(
        &parsed,
        &Schema::new("MAT_ELASTIC").card(
            Card::new()
                .int("mid", 8)
                .float("ro", 16)
                .float("e", 16)
                .float("pr", 16),
        ),
    );
    println!(
        "\nMAT_ELASTIC — mid={:?}, E={:?}",
        mats.column("mid").unwrap().as_int().unwrap(),
        mats.column("e").unwrap().as_float().unwrap(),
    );

    std::fs::remove_file(&path).ok();
}
