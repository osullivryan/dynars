//! Integration test over the bundled multi-file deck in
//! `tests/data/bolt_a_explicit/`. It exercises real-world include handling on a
//! deck copied from an actual model: nested `*INCLUDE`s, `*INCLUDE_PATH_RELATIVE`,
//! LS-DYNA `+` filename continuation, and an intentionally-missing include.

use std::path::{Path, PathBuf};

use dynars::deck::parse_deck;
use dynars::validate::{Rule, Severity};

fn deck_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/bolt_a_explicit/mainboltaexpl.k")
}

/// The whole include tree resolves — which only happens if the `+` filename
/// continuation is joined (`inclu +`/`des.k` → `includes.k`, and the three-line
/// `mat +`/`eri +`/`al_props.k` → `material_props.k`).
#[test]
fn parses_full_include_tree_with_filename_continuation() {
    let deck = parse_deck(&deck_root()).expect("deck parses");

    // root + includes.k + bolted_connection_a.k + control_explicit.k
    //      + submodels/loading/prescribed_motion.k + submodels/material_props.k
    assert_eq!(deck.files.len(), 6, "whole include tree should resolve");

    let names: Vec<String> = deck
        .files
        .iter()
        .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    for expected in [
        "mainboltaexpl.k",
        "includes.k",           // written across two lines: `inclu +` / `des.k`
        "bolted_connection_a.k",
        "control_explicit.k",
        "prescribed_motion.k",
        "material_props.k",     // written across three lines: `mat +` / `eri +` / `al_props.k`
    ] {
        assert!(names.iter().any(|n| n == expected), "missing {expected}: {names:?}");
    }

    // Guards the continuation fix: a filename must never survive as a literal
    // split with a trailing `+`.
    let raws: Vec<&str> = deck.includes.iter().map(|(_, i)| i.raw_path.as_str()).collect();
    for raw in &raws {
        assert!(!raw.contains('+'), "include filename not joined: {raw:?}");
    }
    assert!(raws.contains(&"includes.k"), "continuation join lost: {raws:?}");
    assert!(raws.contains(&"material_props.k"), "continuation join lost: {raws:?}");
}

/// The deck deliberately `*INCLUDE`s a file that isn't on disk. It must be the
/// *only* missing include: `material_props.k` is pulled in via the root's
/// `*INCLUDE_PATH_RELATIVE submodels`, so it resolves despite living in a
/// subdirectory (guards the same-file path-directive fix).
#[test]
fn validation_flags_only_the_intentionally_missing_include() {
    let deck = parse_deck(&deck_root()).expect("deck parses");
    let report = deck.validate([Rule::include_missing()]);

    let missing: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .map(|f| f.message.as_str())
        .collect();
    assert_eq!(missing.len(), 1, "only missing_geometry.k should be missing: {missing:?}");
    assert!(missing[0].contains("missing_geometry.k"), "unexpected: {missing:?}");
}
