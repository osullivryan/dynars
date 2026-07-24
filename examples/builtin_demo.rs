//! Built-in keyword library — parse thousands of LS-DYNA keywords by name with
//! no schema declaration at all (generated from the pyDYNA field database).
//!
//!     cargo run --release --example builtin_demo
//!     cargo run --release --example builtin_demo --features typed-keywords
//!
//! Keyword names are also typo-proof constants (`keywords::names::*`), and with
//! the `typed-keywords` feature each keyword has a fully typed struct.

use dynars::keywords::{self, names};
use dynars::parser::parse_file_blocks;
use dynars::schema::parse_schema;

fn main() {
    // A small deck. *NODE is fixed-width; the rest use comma-free format.
    let mut deck = String::from("*KEYWORD\n*NODE\n");
    for (i, (x, y, z)) in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)].iter().enumerate() {
        deck += &format!("{:>8}{:>16.6}{:>16.6}{:>16.6}\n", i + 1, x, y, z);
    }
    // Two materials = two separate *MAT_ELASTIC blocks (one material each, as
    // in a real deck). parse() gathers *every* matching block into one table.
    deck += "\
*MAT_ELASTIC
1,7.85e-9,210000.0,0.3
*MAT_ELASTIC
2,2.70e-9,70000.0,0.33
*END
";
    let path = std::env::temp_dir().join("dynars_builtin_demo.k");
    std::fs::write(&path, &deck).unwrap();
    let parsed = parse_file_blocks(&path).unwrap();

    println!("built-in keyword library: {} keywords\n", keywords::count());

    // Parse a keyword straight from the library — no declaration needed. The
    // name is a constant, so it autocompletes and can't be mistyped. Every
    // *MAT_ELASTIC block in the file becomes one row (here: 2 blocks -> 2 rows).
    let n_blocks = parsed
        .blocks
        .iter()
        .filter(|b| parsed.keyword_name(b).eq_ignore_ascii_case(names::MAT_ELASTIC))
        .count();
    let mats = parse_schema(&parsed, &keywords::schema(names::MAT_ELASTIC).unwrap());
    println!("MAT_ELASTIC — {} blocks in file, aggregated into {} rows", n_blocks, mats.rows());
    println!("  MID = {:?}", mats.column("MID").unwrap().as_int().unwrap());
    println!("  E   = {:?}", mats.column("E").unwrap().as_float().unwrap());

    // *NODE comes from the hand-written supplement (pyDYNA omits it).
    let nodes = parse_schema(&parsed, &keywords::schema(names::NODE).unwrap());
    println!("\nNODE (supplement) — {} rows", nodes.rows());
    println!("  nid = {:?}", nodes.column("nid").unwrap().as_int().unwrap());
    println!("  x   = {:?}", nodes.column("x").unwrap().as_float().unwrap());

    // With `--features typed-keywords`, every keyword also has a typed struct.
    #[cfg(feature = "typed-keywords")]
    {
        // Same aggregation, now with a typed struct (fields are Vec<i64>/Vec<f64>).
        let m = dynars::keywords::typed::MAT_ELASTIC::parse(&parsed);
        println!(
            "\ntyped MAT_ELASTIC struct — {} rows: mid={:?} e={:?}",
            m.mid.len(),
            m.mid,
            m.e
        );
    }
    #[cfg(not(feature = "typed-keywords"))]
    println!("\n(build with --features typed-keywords for typed structs: MAT_ELASTIC::parse(&p).e)");

    std::fs::remove_file(&path).ok();
}
