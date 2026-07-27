//! `*SENSOR_DEFINE_*` mints a sensor id (SENSID); references to it — e.g.
//! `*SENSOR_SWITCH`'s SENSID — must be dangling-checked. Before a def rule
//! produced `EntityKind::Sensor`, no keyword defined the kind, so every such
//! reference silently passed (a validation blind spot).

use std::fs;
use std::path::Path;

use dynars::deck::parse_deck;
use dynars::keywords::EntityKind;
use dynars::validate::{Rule, Severity};

fn write_deck(body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("dynars_sensor_refs");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = dir.join("root.k");
    fs::write(&root, body).unwrap();
    root
}

/// Sensor ids the dangling check flags, sorted.
fn dangling_sensor_ids(root: &Path) -> Vec<i64> {
    let deck = parse_deck(root).unwrap();
    let report = deck.validate([Rule::references_resolve()]);
    let mut ids: Vec<i64> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .filter_map(|f| {
            let after = f.message.split("references Sensor ").nth(1)?;
            after.split_whitespace().next()?.parse::<i64>().ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn sensor_reference_is_checked_against_sensor_defs() {
    // Sensor 5 is defined by *SENSOR_DEFINE_NODE (its NODE1/NODE2 are 0 = none).
    // Two switches reference sensors 5 (resolves) and 99 (dangles).
    let root = write_deck(
        "*KEYWORD\n\
         *SENSOR_DEFINE_NODE\n5,0,0\n\
         *SENSOR_SWITCH\n1,cross,5\n\
         *SENSOR_SWITCH\n2,cross,99\n\
         *END\n",
    );

    // Navigation: the defined sensor resolves at its id; the other does not.
    let deck = parse_deck(&root).unwrap();
    assert!(deck.get(EntityKind::Sensor, 5).is_some());
    assert!(deck.get(EntityKind::Sensor, 99).is_none());

    // Validation: only the reference to the undefined sensor is flagged.
    assert_eq!(dangling_sensor_ids(&root), vec![99]);
}
