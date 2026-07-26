//! Convert a binout to Arrow tables and print the shape of each branch.
//! Usage: cargo run -p dynars-arrow --example dump -- <binout>
use dynars_arrow::binout_tables;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <binout>");
    let tables = binout_tables(&path, "run_A").expect("convert binout");
    for t in &tables {
        let schema = t.batch.schema();
        let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        println!(
            "{:8} {:5} rows x {:2} cols  {:?}",
            t.branch,
            t.batch.num_rows(),
            t.batch.num_columns(),
            cols
        );
    }
}
