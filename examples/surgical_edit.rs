//! Surgical single-field editing on a real deck (the AWG `orion.k` model).
//!
//! Loads a deck, tweaks a few individual field values, writes the touched files
//! back out, and prints a line-level diff — demonstrating that only the edited
//! fields change and everything else (comments, rulers, mesh, unrelated cards)
//! round-trips byte-for-byte.
//!
//! Run against the AWG test case (or pass your own root deck):
//!
//! ```text
//! cargo run --release --example surgical_edit -- \
//!     /private/tmp/awg_tc8/AWG_ERIF_TEST_CASE_8/orion.k
//! ```

use std::path::Path;

use dynars::deck::parse_deck;
use dynars::model::Value;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/private/tmp/awg_tc8/AWG_ERIF_TEST_CASE_8/orion.k".to_string()
    });
    let mut deck = parse_deck(Path::new(&path)).expect("parse deck");

    // Keep the pristine bytes of every file so we can diff after editing.
    let original: Vec<Vec<u8>> = deck.files.iter().map(|f| f.src().to_vec()).collect();

    // ── 1) schema-aware, in place: *CONTROL_TIMESTEP TSSFAC (0.9 -> 0.8 note) ──
    // Navigate to the keyword, `locate` the named field, then apply. `locate`
    // borrows the deck immutably and returns an owned FieldLoc (carrying the
    // schema's column widths); binding it in a `let` ends that borrow before the
    // `&mut` call to `set_field`. The same shape works for any navigation entry
    // — `deck.part(id)`, `deck.get(kind, id)`, a followed reference, ….
    let loc = deck.keywords("CONTROL_TIMESTEP").next().and_then(|k| k.locate("tssfac"));
    if let Some(loc) = loc {
        println!("set TSSFAC to 0.850000 -> {:?}", deck.set_field(&loc, "0.850000"));
    }

    // ── 2) schema-aware, in place: *CONTROL_TERMINATION ENDTIM ────────────────
    let loc = deck.keywords("CONTROL_TERMINATION").next().and_then(|k| k.locate("endtim"));
    if let Some(loc) = loc {
        println!("set ENDTIM to 0.03 -> {:?}", deck.set_field(&loc, "0.03"));
    }

    // ── 3) pick a keyword in a SPECIFIC include (file-first navigation) ────────
    // `deck.file(suffix)` selects one parsed include; `keywords_named` scopes the
    // occurrences to just that file (vs `deck.keywords`, which spans the whole
    // deck). Here: the first tied contact *in modcontacts.k*, and its FS field.
    let loc = deck
        .file("modcontacts.k")
        .and_then(|f| f.keywords_named("CONTACT_TIED_SHELL_EDGE_TO_SURFACE_BEAM_OFFSET").next())
        .and_then(|k| k.locate("FS"));
    if let Some(loc) = loc {
        println!("modcontacts.k CONTACT/FS -> {:?}", deck.set_field(&loc, "0.150"));
    }

    // Enumerate includes and their keyword makeup (root is files().next()):
    for f in deck.files() {
        let n = f.keywords().count();
        println!("  file {:>2}: {:<20} {n} keywords", f.index(), f.path().file_name().unwrap().to_string_lossy());
    }

    // ── write the touched files, show the diff, and re-parse to verify ────────
    // Mirror the whole deck into a temp dir (by file name, so the `*INCLUDE`s
    // still resolve) and re-parse it — proving the edits produced a valid deck
    // that reads back the new values.
    let out_dir = std::env::temp_dir().join("dynars_surgical_edit");
    std::fs::create_dir_all(&out_dir).expect("temp dir");

    let mut total_changed = 0usize;
    for (i, file) in deck.files.iter().enumerate() {
        let name = file.path.file_name().unwrap().to_string_lossy();
        let after = file.to_bytes();
        if file.is_dirty() {
            println!("\n=== {name} ===");
            total_changed += print_line_diff(&original[i], &after);
        }
        std::fs::write(out_dir.join(&*name), &after).expect("write file");
    }
    println!("\ntotal changed lines across the deck: {total_changed}");
    assert!(total_changed <= 3, "only the edited lines should differ");

    let root = Path::new(&path).file_name().unwrap();
    let reparsed = parse_deck(&out_dir.join(root)).expect("re-parse edited deck");
    let read = |kw: &str, field: &str| {
        reparsed
            .keywords(kw)
            .next()
            .and_then(|k| k.field(field))
            .map(|f| f.value())
    };
    println!("\nre-parsed {}:", out_dir.join(root).display());
    println!("  CONTROL_TIMESTEP/tssfac   = {:?}", read("CONTROL_TIMESTEP", "tssfac"));
    println!("  CONTROL_TERMINATION/endtim= {:?}", read("CONTROL_TERMINATION", "endtim"));
    assert_eq!(read("CONTROL_TIMESTEP", "tssfac"), Some(Value::Float(0.85)));
    assert_eq!(read("CONTROL_TERMINATION", "endtim"), Some(Value::Float(0.03)));
    println!("\nverified: edits round-tripped and re-parse to the new values.");
}

/// Print the lines that differ between `before` and `after`, and return how many
/// changed. Same line count on both sides (surgical edits never add/remove
/// lines), so a positional zip is exact.
fn print_line_diff(before: &[u8], after: &[u8]) -> usize {
    let bl: Vec<&[u8]> = before.split(|&c| c == b'\n').collect();
    let al: Vec<&[u8]> = after.split(|&c| c == b'\n').collect();
    assert_eq!(bl.len(), al.len(), "line count must be preserved");
    let mut n = 0;
    for (i, (b, a)) in bl.iter().zip(&al).enumerate() {
        if b != a {
            n += 1;
            println!("  L{:<6} - {}", i + 1, String::from_utf8_lossy(b));
            println!("  {:<7} + {}", "", String::from_utf8_lossy(a));
        }
    }
    n
}
