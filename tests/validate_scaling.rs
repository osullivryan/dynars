//! Regression: the dangling-reference check must stay **linear** in the number
//! of keyword blocks. A deck with many small reference-bearing keywords (here
//! 30k `*BOUNDARY_SPC_NODE` cards) once made `check_refs` recompute each block's
//! line number by scanning from the start of the file — O(blocks²), which hung
//! for minutes on ~128k blocks. If that regresses, this test goes from
//! milliseconds to tens of seconds; it also pins the reported line number, so
//! the one-pass line table stays correct for blocks late in the file.

use std::fmt::Write as _;
use std::fs;

use dynars::deck::parse_deck;
use dynars::validate::{Rule, Severity};

#[test]
fn dangling_check_is_linear_in_block_count() {
    const N: usize = 30_000;

    let dir = std::env::temp_dir().join("dynars_it_scaling");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut s = String::from("*KEYWORD\n*NODE\n");
    for nid in 1..=N {
        writeln!(s, "{nid},0.0,0.0,0.0").unwrap();
    }
    // N SPC blocks: all reference a defined node except the last, which points
    // at an undefined id — exactly one dangling reference, near the file's end.
    for j in 0..N {
        let nid = if j == N - 1 { N + 1_000_000 } else { j + 1 };
        writeln!(s, "*BOUNDARY_SPC_NODE\n{nid},0,1,1,1,0,0,0").unwrap();
    }
    s.push_str("*END\n");
    let root = dir.join("root.k");
    fs::write(&root, s).unwrap();

    let deck = parse_deck(&root).unwrap();
    let report = deck.validate([Rule::references_resolve()]);

    let errors: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error && f.keyword == "BOUNDARY_SPC_NODE")
        .collect();
    assert_eq!(errors.len(), 1, "exactly one dangling SPC reference");

    // Layout: line 1 *KEYWORD, line 2 *NODE, lines 3..=N+2 the nodes, then the
    // SPC blocks two lines each. The bad block is index N-1, so its `*KEYWORD`
    // line is (N+2) + (N-1)*2 + 1 = 3N+1.
    assert_eq!(errors[0].line, 3 * N + 1, "line number of the late block is exact");
}
