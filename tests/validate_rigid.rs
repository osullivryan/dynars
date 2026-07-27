//! Rigid-context consistency: keywords that act on a rigid body (e.g.
//! *LOAD_RIGID_BODY, *CONSTRAINED_RIGID_BODIES) must reference a part whose
//! material is *MAT_RIGID. A deformable target is a modelling error.

use std::fs;
use std::path::Path;

use dynars::deck::parse_deck;
use dynars::validate::{Rule, Severity};

/// A deck with a rigid part (pid 10, *MAT_RIGID) and a deformable part (pid 20,
/// *MAT_ELASTIC); `body` is spliced in after the parts.
fn deck_with(tag: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dynars_rigid_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let root = dir.join("root.k");
    fs::write(
        &root,
        format!(
            "*KEYWORD\n\
             *SECTION_SHELL\n1,2\n\
             *MAT_RIGID\n1,7.85e-9,210000.0,0.3\n\
             *MAT_ELASTIC\n2,7.85e-9,210000.0,0.3\n\
             *PART\nrigid\n10,1,1\n\
             *PART\ndeformable\n20,1,2\n\
             {body}*END\n"
        ),
    )
    .unwrap();
    root
}

fn rigid_findings(root: &Path) -> Vec<String> {
    let deck = parse_deck(root).unwrap();
    deck.validate([Rule::rigid_context()])
        .findings
        .into_iter()
        .filter(|f| f.severity == Severity::Error)
        .map(|f| f.message)
        .collect()
}

#[test]
fn rigid_body_load_on_rigid_part_is_clean() {
    let root = deck_with("ok", "*LOAD_RIGID_BODY\n10,3,0\n");
    assert!(rigid_findings(&root).is_empty());
}

#[test]
fn rigid_body_load_on_deformable_part_is_flagged() {
    let root = deck_with("bad", "*LOAD_RIGID_BODY\n20,3,0\n");
    let msgs = rigid_findings(&root);
    assert_eq!(msgs.len(), 1, "{msgs:?}");
    assert!(msgs[0].contains("part 20"), "{msgs:?}");
    assert!(msgs[0].contains("MAT_ELASTIC"), "{msgs:?}");
}

#[test]
fn constrained_rigid_bodies_flags_only_the_deformable_member() {
    // PIDL 10 is rigid (fine); PIDC 20 is deformable (flagged).
    let root = deck_with("crb", "*CONSTRAINED_RIGID_BODIES\n10,20\n");
    let msgs = rigid_findings(&root);
    assert_eq!(msgs.len(), 1, "{msgs:?}");
    assert!(msgs[0].contains("PIDC"), "{msgs:?}");
    assert!(msgs[0].contains("part 20"), "{msgs:?}");
}

#[test]
fn dangling_rigid_target_is_left_to_references_resolve() {
    // Part 999 doesn't exist: rigid_context stays silent (no material to judge),
    // and references_resolve is the rule that flags the dangling part.
    let root = deck_with("dangling", "*LOAD_RIGID_BODY\n999,3,0\n");
    assert!(
        rigid_findings(&root).is_empty(),
        "unresolved part not flagged here"
    );

    let deck = parse_deck(&root).unwrap();
    let dangling = deck
        .validate([Rule::references_resolve()])
        .findings
        .iter()
        .any(|f| f.message.contains("Part 999"));
    assert!(dangling, "references_resolve flags the missing part");
}
