//! End-to-end: `*INCLUDE_TRANSFORM` id offsets must fold into the id namespace
//! so the dangling-reference check resolves transformed ids — a reference to a
//! node the include defines resolves only at its *offset* (global) id, never at
//! its raw id. The validation layer itself stays oblivious; the model applies
//! the shift when it collects defs and probes refs.

use std::fs;
use std::path::{Path, PathBuf};

use dynars::deck::parse_deck;
use dynars::keywords::EntityKind;
use dynars::validate::{Rule, Severity};

/// Write `root.k` + `mesh.k` into a fresh temp dir and return the root path.
/// `mesh.k` defines nodes 1, 2, 3. `root.k` pins two nodes via
/// `*BOUNDARY_SPC_NODE` (a plain, non-connectivity `Ref::To(Node)`): one at
/// `hi` and one at `lo`. `include_block` is spliced in verbatim so the caller
/// controls whether the mesh is pulled in plain or transformed.
fn write_deck(tag: &str, include_block: &str, hi: i64, lo: i64) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("dynars_it_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    fs::write(
        dir.join("mesh.k"),
        "*NODE\n1,0.0,0.0,0.0\n2,1.0,0.0,0.0\n3,2.0,0.0,0.0\n",
    )
    .unwrap();

    let root = format!(
        "*KEYWORD\n\
         *BOUNDARY_SPC_NODE\n{hi},0,1,1,1,0,0,0\n\
         *BOUNDARY_SPC_NODE\n{lo},0,1,1,1,0,0,0\n\
         {include_block}\
         *END\n"
    );
    let root_path = dir.join("root.k");
    fs::write(&root_path, root).unwrap();
    (dir, root_path)
}

/// The node ids the dangling check flags as undefined, sorted.
fn dangling_node_ids(root: &Path) -> Vec<i64> {
    let deck = parse_deck(root).unwrap();
    let report = deck.validate([Rule::references_resolve()]);
    let mut ids: Vec<i64> = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error && f.keyword == "BOUNDARY_SPC_NODE")
        // message: "BOUNDARY_SPC_NODE.NID references Node <id> — not defined …"
        .filter_map(|f| {
            let after = f.message.split("references Node ").nth(1)?;
            after.split_whitespace().next()?.parse::<i64>().ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn include_transform_shifts_node_id_namespace() {
    // idnoff = 1_000_000: mesh nodes 1,2,3 become global 1_000_001..=1_000_003.
    // Card 2 is the seven IDNOFF..IDDOFF offsets (I10); card 3 is IDROFF.
    let transform = format!(
        "*INCLUDE_TRANSFORM\nmesh.k\n{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n{:>10}\n",
        1_000_000, 0, 0, 0, 0, 0, 0, 0
    );
    let (_dir, root) = write_deck(
        "xform", &transform, 1_000_001, // resolves against the transformed node
        1,         // raw id no longer exists globally → dangles
    );
    assert_eq!(dangling_node_ids(&root), vec![1]);
}

#[test]
fn navigation_resolves_ids_in_the_global_namespace() {
    // A transformed include with IDPOFF=500, IDMOFF=300. Its local part 1 and
    // material 1 become global part 501 / material 301, and the part's mid ref
    // (local 1) must follow to material 301.
    let dir = std::env::temp_dir().join("dynars_it_nav");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("mesh.k"),
        "*KEYWORD\n*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n*PART\npart one\n1,0,1\n*END\n",
    )
    .unwrap();
    let root = dir.join("root.k");
    fs::write(
        &root,
        format!(
            "*KEYWORD\n*INCLUDE_TRANSFORM\nmesh.k\n\
             {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n{:>10}\n*END\n",
            0, 0, 500, 300, 0, 0, 0, 0
        ),
    )
    .unwrap();

    let deck = parse_deck(&root).unwrap();

    // Sites are keyed by the global id: the part is found at 501, not its raw 1.
    assert!(
        deck.get(EntityKind::Part, 501).is_some(),
        "part at offset id"
    );
    assert!(
        deck.get(EntityKind::Part, 1).is_none(),
        "raw id must not resolve"
    );

    // Following the part's material reference shifts the local mid (1) by IDMOFF.
    let part = deck.part(501).unwrap();
    let material = part
        .material()
        .expect("material resolves across the transform");
    assert_eq!(material.id(), Some(301));

    // Introspection: the effective offsets applied to this entity's file.
    let t = part
        .transform()
        .expect("part sits under an *INCLUDE_TRANSFORM");
    assert_eq!((t.idpoff, t.idmoff), (500, 300));
    assert_eq!(t.idnoff, 0);
    // The root file itself carries no transform.
    assert!(
        deck.keywords("INCLUDE_TRANSFORM")
            .next()
            .unwrap()
            .transform()
            .is_none()
    );
}

#[test]
fn plain_include_keeps_raw_ids() {
    // Control: a plain *INCLUDE applies no shift, so the raw id 1 resolves and
    // the offset id 1_000_001 is the one that dangles — the exact opposite.
    let (_dir, root) = write_deck("plain", "*INCLUDE\nmesh.k\n", 1_000_001, 1);
    assert_eq!(dangling_node_ids(&root), vec![1_000_001]);
}
