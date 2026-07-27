//! `*INCLUDE_PATH_RELATIVE` (and `*INCLUDE_PATH`) must widen the search set for
//! `*INCLUDE`s in the *same* file — LS-DYNA applies them file-wide, so an
//! include resolves through the added directory regardless of whether it
//! appears before or after the path directive.

use std::fs;
use std::path::PathBuf;

use dynars::deck::parse_deck;
use dynars::include::build_include_tree;
use dynars::validate::{Rule, Severity};

/// Write a root deck plus `sub/inner.k`, returning the root path. `root_body` is
/// spliced between `*KEYWORD` and `*END` so each test controls directive order.
fn write_deck(tag: &str, root_body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dynars_incpath_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("sub")).unwrap();
    fs::write(dir.join("sub/inner.k"), "*NODE\n1,0.0,0.0,0.0\n").unwrap();
    let root = format!("*KEYWORD\n{root_body}*END\n");
    let root_path = dir.join("root.k");
    fs::write(&root_path, root).unwrap();
    root_path
}

fn missing_includes(root: &std::path::Path) -> usize {
    let deck = parse_deck(root).unwrap();
    deck.validate([Rule::include_missing()])
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .count()
}

#[test]
fn path_relative_resolves_same_file_include() {
    // Directive first, then the include that depends on it.
    let root = write_deck("before", "*INCLUDE_PATH_RELATIVE\nsub\n*INCLUDE\ninner.k\n");
    let deck = parse_deck(&root).unwrap();
    assert_eq!(deck.files.len(), 2, "inner.k should resolve via sub/");
    assert_eq!(missing_includes(&root), 0, "no include should be missing");
}

#[test]
fn path_relative_applies_regardless_of_order() {
    // Include appears *before* the path directive — still resolves, because the
    // directive is applied file-wide, not only to what follows it.
    let root = write_deck("after", "*INCLUDE\ninner.k\n*INCLUDE_PATH_RELATIVE\nsub\n");
    assert_eq!(
        missing_includes(&root),
        0,
        "path directive must apply file-wide"
    );
}

#[test]
fn unresolvable_include_is_still_flagged() {
    // Sanity: without the right search path, the include is genuinely missing.
    let root = write_deck("missing", "*INCLUDE\ninner.k\n");
    assert_eq!(missing_includes(&root), 1);
}

// The include-*tree* builder (streaming scanner) must resolve the same way as
// the deck path — same-file path directives, order-independent.

#[test]
fn include_tree_resolves_same_file_path_relative() {
    let root = write_deck(
        "tree_before",
        "*INCLUDE_PATH_RELATIVE\nsub\n*INCLUDE\ninner.k\n",
    );
    let tree = build_include_tree(&root).unwrap();
    assert_eq!(tree.total_files(), 2, "tree should reach inner.k via sub/");
}

#[test]
fn include_tree_path_relative_order_independent() {
    let root = write_deck(
        "tree_after",
        "*INCLUDE\ninner.k\n*INCLUDE_PATH_RELATIVE\nsub\n",
    );
    let tree = build_include_tree(&root).unwrap();
    assert_eq!(tree.total_files(), 2, "path directive must apply file-wide");
}
