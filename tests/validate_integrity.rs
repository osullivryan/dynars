//! Deck-integrity checks beyond dangling references: duplicate entity ids and
//! unreferenced ("dead") library definitions — the two flagship reference-graph
//! checks modelled on PRIMER / ANSA / Altair model checkers.

use std::fs;
use std::path::Path;

use dynars::deck::parse_deck;
use dynars::validate::{Rule, Severity};

fn write_deck(tag: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dynars_integrity_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = dir.join("root.k");
    fs::write(&root, body).unwrap();
    root
}

fn messages(root: &Path, rule: Rule, sev: Severity) -> Vec<String> {
    let deck = parse_deck(root).unwrap();
    deck.validate([rule])
        .findings
        .into_iter()
        .filter(|f| f.severity == sev)
        .map(|f| f.message)
        .collect()
}

#[test]
fn duplicate_ids_flags_id_collisions() {
    // Two *MAT_ELASTIC claim mid 1; two *PART claim pid 3. Section 1 is unique.
    let root = write_deck(
        "dup",
        "*KEYWORD\n\
         *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
         *MAT_ELASTIC\n1,7.0e-9,200000.0,0.3\n\
         *SECTION_SHELL\n1,2\n\
         *PART\npart one\n3,1,1\n\
         *PART\npart two\n3,1,1\n\
         *END\n",
    );
    let msgs = messages(&root, Rule::duplicate_ids(), Severity::Error);
    // Every colliding definition is flagged: 2 for Material 1, 2 for Part 3.
    let mat = msgs.iter().filter(|m| m.contains("Material id 1")).count();
    let part = msgs.iter().filter(|m| m.contains("Part id 3")).count();
    assert_eq!(mat, 2, "both *MAT_ELASTIC mid=1 flagged: {msgs:?}");
    assert_eq!(part, 2, "both *PART pid=3 flagged: {msgs:?}");
    assert_eq!(msgs.len(), 4, "nothing else collides: {msgs:?}");
}

#[test]
fn unique_ids_produce_no_duplicate_findings() {
    let root = write_deck(
        "uniq",
        "*KEYWORD\n\
         *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
         *MAT_ELASTIC\n2,7.0e-9,200000.0,0.3\n\
         *SECTION_SHELL\n1,2\n\
         *PART\np\n1,1,1\n\
         *END\n",
    );
    assert!(messages(&root, Rule::duplicate_ids(), Severity::Error).is_empty());
}

#[test]
fn unreferenced_entities_flags_dead_definitions() {
    // Part 1 uses section 1 and material 1. Material 99 and curve 7 are dead.
    let root = write_deck(
        "orphan",
        "*KEYWORD\n\
         *PART\np\n1,1,1\n\
         *SECTION_SHELL\n1,2\n\
         *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
         *MAT_ELASTIC\n99,7.85e-9,210000.0,0.3\n\
         *DEFINE_CURVE\n7\n0.0,0.0\n1.0,1.0\n\
         *END\n",
    );
    let msgs = messages(&root, Rule::unreferenced_entities(), Severity::Warning);
    assert!(
        msgs.iter().any(|m| m.contains("Material 99")),
        "dead material flagged: {msgs:?}"
    );
    assert!(
        msgs.iter().any(|m| m.contains("Curve 7")),
        "dead curve flagged: {msgs:?}"
    );
    // Referenced entities must not be flagged.
    assert!(
        !msgs.iter().any(|m| m.contains("Material 1")),
        "used material not flagged: {msgs:?}"
    );
    assert!(
        !msgs.iter().any(|m| m.contains("Section 1")),
        "used section not flagged: {msgs:?}"
    );
}

#[test]
fn instanced_ids_are_not_false_duplicates() {
    // The same mesh part 1 pulled in at IDPOFF=0 and IDPOFF=1000 becomes parts 1
    // and 1001 — logically distinct, so *not* a duplicate-id collision.
    let dir = std::env::temp_dir().join("dynars_integrity_instance");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("mesh.k"),
        "*KEYWORD\n*SECTION_SHELL\n1,2\n*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n*PART\np\n1,1,1\n*END\n",
    )
    .unwrap();
    let xform = |idpoff: i64| {
        format!(
            "*INCLUDE_TRANSFORM\nmesh.k\n{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n{:>10}\n",
            0, 0, idpoff, 0, 0, 0, 0, 0
        )
    };
    let root = dir.join("root.k");
    fs::write(
        &root,
        format!("*KEYWORD\n{}{}*END\n", xform(0), xform(1000)),
    )
    .unwrap();

    let deck = parse_deck(&root).unwrap();
    let parts: Vec<String> = deck
        .validate([Rule::duplicate_ids()])
        .findings
        .into_iter()
        .filter(|f| f.message.contains("Part"))
        .map(|f| f.message)
        .collect();
    assert!(
        parts.is_empty(),
        "instanced parts are distinct ids: {parts:?}"
    );
}
