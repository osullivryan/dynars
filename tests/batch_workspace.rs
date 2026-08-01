//! `Workspace` batch parsing: decks that share an `*INCLUDE` read it once, and a
//! workspace-parsed deck is byte- and finding-identical to a standalone
//! `parse_deck`.

use std::fs;
use std::path::{Path, PathBuf};

use dynars::Workspace;
use dynars::deck::{Deck, parse_deck};
use dynars::validate::{Finding, Rule};

/// Two variant decks (`a/main.k`, `b/main.k`) that both `*INCLUDE ../shared.k`.
/// `shared.k` defines material 1 and section 1; variant A's part resolves cleanly,
/// variant B's part references an undefined material to exercise a per-deck finding.
fn write_variants(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("dynars_batch_ws_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("a")).unwrap();
    fs::create_dir_all(dir.join("b")).unwrap();

    fs::write(
        dir.join("shared.k"),
        "*KEYWORD\n\
         *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
         *SECTION_SHELL\n1,2\n\
         *NODE\n1,0.0,0.0,0.0\n2,1.0,0.0,0.0\n3,0.0,1.0,0.0\n",
    )
    .unwrap();

    let a = dir.join("a/main.k");
    fs::write(
        &a,
        "*KEYWORD\n\
         *INCLUDE\n../shared.k\n\
         *PART\npart a\n10,1,1\n\
         *END\n",
    )
    .unwrap();

    let b = dir.join("b/main.k");
    fs::write(
        &b,
        "*KEYWORD\n\
         *INCLUDE\n../shared.k\n\
         *PART\npart b\n20,1,99\n\
         *END\n",
    )
    .unwrap();

    (a, b)
}

fn rules() -> Vec<Rule> {
    vec![
        Rule::references_resolve(),
        Rule::duplicate_ids(),
        Rule::keyword_forbidden("MAT_ADD_EROSION"),
    ]
}

/// A stable, comparable projection of a finding (severity/location/message).
fn finding_keys(mut findings: Vec<Finding>) -> Vec<String> {
    findings.sort_by(|x, y| x.line.cmp(&y.line).then(x.message.cmp(&y.message)));
    findings
        .into_iter()
        .map(|f| {
            format!(
                "{:?}|{}|{}|{}",
                f.severity,
                f.file.display(),
                f.line,
                f.message
            )
        })
        .collect()
}

/// Every file's reconstructed bytes and the validation report, for equivalence.
fn deck_fingerprint(deck: &Deck) -> (Vec<(PathBuf, Vec<u8>)>, Vec<String>) {
    let files = deck
        .files
        .iter()
        .map(|f| (f.path.clone(), f.to_bytes()))
        .collect();
    let report = deck.validate(rules());
    (files, finding_keys(report.findings))
}

#[test]
fn shared_include_is_parsed_once() {
    let (a, b) = write_variants("once");
    let ws = Workspace::new();

    let results = ws.parse_decks([&a, &b]);
    assert!(results.iter().all(|(_, d)| d.is_ok()), "both decks parse");

    // a/main.k, shared.k, b/main.k are read once; shared.k is a cache hit for b.
    let stats = ws.stats();
    assert_eq!(
        stats.files_parsed, 3,
        "distinct files read from disk: {stats:?}"
    );
    assert_eq!(
        stats.files_reused, 1,
        "shared.k reused for deck b: {stats:?}"
    );
}

#[test]
fn workspace_deck_matches_parse_deck() {
    let (a, b) = write_variants("match");
    let ws = Workspace::new();

    for root in [&a, &b] {
        let plain = parse_deck(root).unwrap();
        let shared = ws.parse_deck(root).unwrap();

        assert_eq!(
            plain.files.len(),
            shared.files.len(),
            "same file count for {}",
            root.display()
        );
        assert_eq!(
            deck_fingerprint(&plain),
            deck_fingerprint(&shared),
            "identical bytes + findings for {}",
            root.display()
        );
    }
}

/// One `*INCLUDE_TRANSFORM <target>` block shifting node ids by `idnoff`.
fn xform_block(target: &str, idnoff: i64) -> String {
    format!(
        "*INCLUDE_TRANSFORM\n{target}\n\
         {:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n{:>10}\n",
        idnoff, 0, 0, 0, 0, 0, 0, 0
    )
}

/// Node ids the dangling check flags on `*BOUNDARY_SPC_NODE`, sorted.
fn dangling_nodes(deck: &Deck) -> Vec<i64> {
    let mut ids: Vec<i64> = deck
        .validate([Rule::references_resolve()])
        .findings
        .iter()
        .filter(|f| f.keyword == "BOUNDARY_SPC_NODE")
        .filter_map(|f| {
            f.message
                .split("references Node ")
                .nth(1)?
                .split_whitespace()
                .next()?
                .parse::<i64>()
                .ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn shared_mesh_shifts_per_deck_across_offsets() {
    // The instancing idiom across *decks*: one cached `mesh.k` (nodes 1,2,3) is
    // pulled into deck A at idnoff=1000 and deck B at idnoff=2000. The physical
    // extraction runs once; each deck applies its own shift, so A defines
    // 1001..1003 and B defines 2001..2003 — the foreign-namespace ref dangles.
    let dir = std::env::temp_dir().join("dynars_batch_xform");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("a")).unwrap();
    fs::create_dir_all(dir.join("b")).unwrap();
    fs::write(
        dir.join("mesh.k"),
        "*NODE\n1,0.0,0.0,0.0\n2,1.0,0.0,0.0\n3,2.0,0.0,0.0\n",
    )
    .unwrap();

    let a = dir.join("a/root.k");
    fs::write(
        &a,
        format!(
            "*KEYWORD\n\
             *BOUNDARY_SPC_NODE\n1001,0,1,1,1,0,0,0\n\
             *BOUNDARY_SPC_NODE\n2002,0,1,1,1,0,0,0\n\
             {}*END\n",
            xform_block("../mesh.k", 1000)
        ),
    )
    .unwrap();

    let b = dir.join("b/root.k");
    fs::write(
        &b,
        format!(
            "*KEYWORD\n\
             *BOUNDARY_SPC_NODE\n2001,0,1,1,1,0,0,0\n\
             *BOUNDARY_SPC_NODE\n1002,0,1,1,1,0,0,0\n\
             {}*END\n",
            xform_block("../mesh.k", 2000)
        ),
    )
    .unwrap();

    let ws = Workspace::new();
    let deck_a = ws.parse_deck(&a).unwrap();
    let deck_b = ws.parse_deck(&b).unwrap();

    // Shared cache, per-deck shift: A resolves 1001 (dangles 2002); B resolves
    // 2001 (dangles 1002). No cross-contamination through the shared physical ids.
    assert_eq!(dangling_nodes(&deck_a), vec![2002], "deck A shifts by 1000");
    assert_eq!(dangling_nodes(&deck_b), vec![1002], "deck B shifts by 2000");

    // mesh.k read once despite two decks pulling it in at two offsets.
    let s = ws.stats();
    assert_eq!(s.files_parsed, 3, "a/root, mesh, b/root: {s:?}");
    assert_eq!(s.files_reused, 1, "mesh reused for deck b: {s:?}");

    // And each matches the standalone parse — the cache changes cost, not result.
    assert_eq!(
        dangling_nodes(&deck_a),
        dangling_nodes(&parse_deck(&a).unwrap())
    );
    assert_eq!(
        dangling_nodes(&deck_b),
        dangling_nodes(&parse_deck(&b).unwrap())
    );
}

/// A shared clean mesh (nodes 1-4; part 1 → section 1 / mat 1; one shell over
/// those nodes) plus two decks: A includes it as-is; B adds a shell whose 4th
/// node (999) is undefined — a connectivity dangler in B only.
fn write_connectivity_variants(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("dynars_batch_conn_{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("a")).unwrap();
    fs::create_dir_all(dir.join("b")).unwrap();
    fs::write(
        dir.join("mesh.k"),
        "*KEYWORD\n\
         *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
         *SECTION_SHELL\n1,2\n\
         *PART\np\n1,1,1\n\
         *NODE\n1,0.0,0.0,0.0\n2,1.0,0.0,0.0\n3,1.0,1.0,0.0\n4,0.0,1.0,0.0\n\
         *ELEMENT_SHELL\n1,1,1,2,3,4\n",
    )
    .unwrap();
    let a = dir.join("a/main.k");
    fs::write(&a, "*KEYWORD\n*INCLUDE\n../mesh.k\n*END\n").unwrap();
    let b = dir.join("b/main.k");
    fs::write(
        &b,
        "*KEYWORD\n*INCLUDE\n../mesh.k\n*ELEMENT_SHELL\n2,1,1,2,3,999\n*END\n",
    )
    .unwrap();
    (a, b)
}

fn conn_keys(deck: &Deck) -> Vec<String> {
    finding_keys(
        deck.validate([Rule::references_resolve_with_connectivity()])
            .findings,
    )
}

#[test]
fn connectivity_cache_matches_uncached() {
    let (a, b) = write_connectivity_variants("cache");
    let ws = Workspace::new();
    let da = ws.parse_deck(&a).unwrap();
    let db = ws.parse_deck(&b).unwrap();

    // Deck A: the shared mesh's connectivity is clean, so the cached referenced-id
    // set proves it without walking the element rows — no findings, same as plain.
    assert_eq!(conn_keys(&da), conn_keys(&parse_deck(&a).unwrap()));
    assert!(conn_keys(&da).is_empty(), "deck A connectivity clean");

    // Deck B: the fast probe flags node 999, so the walk still runs to locate the
    // offending element — byte-identical result to the standalone parse.
    assert_eq!(conn_keys(&db), conn_keys(&parse_deck(&b).unwrap()));
    assert!(
        db.validate([Rule::references_resolve_with_connectivity()])
            .findings
            .iter()
            .any(|f| f.message.contains("999")),
        "deck B locates the dangling node 999"
    );
}

#[test]
fn validate_decks_parallel_matches_sequential() {
    let (a, b) = write_connectivity_variants("par");
    let ws = Workspace::new();
    let decks: Vec<Deck> = ws
        .parse_decks([&a, &b])
        .into_iter()
        .map(|(_, d)| d.unwrap())
        .collect();

    let rules = || {
        [
            Rule::references_resolve_with_connectivity(),
            Rule::duplicate_ids(),
        ]
    };
    let parallel = ws.validate_decks(&decks, rules());
    assert_eq!(parallel.len(), 2, "one report per deck");
    for (deck, report) in decks.iter().zip(&parallel) {
        let sequential = deck.validate(rules());
        assert_eq!(
            finding_keys(report.findings.clone()),
            finding_keys(sequential.findings),
            "parallel report equals sequential validate"
        );
    }
}

#[test]
fn shared_indices_built_once_across_decks() {
    // The check-work-sharing claim, made deterministic: the shared mesh's
    // definition and connectivity indices are each built ONCE across both decks,
    // not once per deck. Three distinct files (mesh, a/main, b/main) → a count of
    // 3, never 4 (which would mean the mesh was re-extracted per deck).
    let (a, b) = write_connectivity_variants("built");
    let ws = Workspace::new();
    let decks: Vec<Deck> = ws
        .parse_decks([&a, &b])
        .into_iter()
        .map(|(_, d)| d.unwrap())
        .collect();

    // No connectivity rule: the connectivity index stays lazy (unbuilt); the
    // definition index is built per distinct file — the shared mesh just once.
    ws.validate_decks(&decks, [Rule::references_resolve(), Rule::duplicate_ids()]);
    let s = ws.stats();
    assert_eq!(s.files_parsed, 3, "mesh + two roots read once: {s:?}");
    assert_eq!(s.files_reused, 1, "mesh reused for the 2nd deck: {s:?}");
    assert_eq!(
        s.def_indices_built, 3,
        "def index built per distinct file: {s:?}"
    );
    assert_eq!(
        s.ref_indices_built, 0,
        "no connectivity check → ref index never built: {s:?}"
    );

    // Run a connectivity check: the connectivity index now builds — once per
    // distinct file, the shared mesh included, and def stays cached (no rebuild).
    ws.validate_decks(&decks, [Rule::references_resolve_with_connectivity()]);
    let s = ws.stats();
    assert_eq!(
        s.ref_indices_built, 3,
        "conn index built per distinct file: {s:?}"
    );
    assert_eq!(
        s.def_indices_built, 3,
        "def index unchanged — still cached: {s:?}"
    );
}

#[test]
fn findings_are_independent_per_deck() {
    // Variant B's part references undefined material 99; variant A is clean.
    // A shared include must not leak one deck's findings into the other.
    // (unique fixture dir per test to avoid cross-test temp-dir races)
    let (a, b) = write_variants("indep");
    let ws = Workspace::new();

    let deck_a = ws.parse_deck(&a).unwrap();
    let deck_b = ws.parse_deck(&b).unwrap();

    let dangling = |deck: &Deck| {
        deck.validate([Rule::references_resolve()])
            .findings
            .iter()
            .any(|f| f.message.contains("99"))
    };

    assert!(!dangling(&deck_a), "variant A resolves cleanly");
    assert!(dangling(&deck_b), "variant B flags undefined material 99");
    let _ = Path::new("");
}
