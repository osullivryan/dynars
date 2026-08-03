# Decks & navigation

A `Deck` is the root keyword file plus everything it `*INCLUDE`s, parsed once and
presented as a single model. This page covers the four things you do with one:
inspect the include graph, navigate by id and reference, bulk-read columns, and
edit files back to disk.

## Parsing a deck

`parse_deck` takes the **root** keyword file and follows its `*INCLUDE` graph —
`*INCLUDE`, `*INCLUDE_PATH`, `*INCLUDE_TRANSFORM`, and friends — parsing every
reachable file in one parallel pass. The result is a single `Deck` handle that
owns all the parsed files.

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("main.k")
    print(deck)  # Deck(<n> files)
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;

    let deck = parse_deck(std::path::Path::new("main.k")).unwrap();
    println!("{} files, {} bytes", deck.files.len(), deck.total_bytes());
    ```

## The include tree

To see the file structure *without* a full parse — how many files, how big, how
deep — walk the include tree. It's cheap: the scanner only reads each file's
`*INCLUDE` lines, not its contents.

=== "Python"

    ```python
    root = dynars.parse_include_tree("main.k")
    print(root.total_files(), "files,", root.total_bytes(), "bytes")

    def walk(node, depth=0):
        print("  " * depth, node.path, f"({node.kind or 'root'}, {node.byte_count} B)")
        for child in node.children:
            walk(child, depth + 1)

    walk(root)
    ```

=== "Rust"

    ```rust
    use dynars::include::build_include_tree;

    let root = build_include_tree(std::path::Path::new("main.k")).unwrap();
    println!("{} files, {} bytes", root.total_files(), root.total_bytes());

    fn walk(node: &dynars::include::IncludeNode, depth: usize) {
        println!("{:indent$}{}", "", node.path.display(), indent = depth * 2);
        for child in &node.children {
            walk(child, depth + 1);
        }
    }
    walk(&root, 0);
    ```

Each node carries its `path`, its `kind` (`"INCLUDE"`, `"INCLUDE_TRANSFORM"`, …,
or `None` for the root), its own `byte_count`, and its `children`. `total_files()`
and `total_bytes()` sum the whole subtree.

## Navigate by id, follow references

Look an entity up by its id, then follow the references in its fields — a
`*PART`'s `mid` to its `*MAT`, its `secid` to its `*SECTION`, a load's `lcid` to
its `*DEFINE_CURVE`, and so on. Ids are resolved in the deck's **global**
namespace, so references that cross an `*INCLUDE_TRANSFORM` are followed correctly
(and the id's sign is ignored — `deck.curve(5)` matches a reference to `-5`).

=== "Python"

    ```python
    part = deck.part(5)
    if part is not None:
        mat = part.material()       # follow *PART.mid -> *MAT
        sec = part.section()        # follow *PART.secid -> *SECTION
        print(part.id, mat.field("RO") if mat else None)

        # Any reference field by name:
        curve = part.reference("lcid")
    ```

=== "Rust"

    ```rust
    if let Some(part) = deck.part(5) {
        let mat = part.material();  // follow *PART.mid -> *MAT
        let sec = part.section();   // follow *PART.secid -> *SECTION
        if let Some(m) = mat {
            println!("part 5 uses *{} (RO = {:?})", m.name(), m.field("RO").and_then(|f| f.as_f64()));
        }
        // Any reference field by name:
        let curve = part.reference("lcid");
        let _ = (sec, curve);
    }
    ```

`part`, `material`, `section`, and `curve` are the common id lookups;
`.material()`, `.section()`, `.eos()`, `.hourglass()`, and the generic
`.reference(name)` are the reference-following moves. Every other kind is
reachable too (see the [API reference](reference.md)).

!!! tip "Where did this come from?"
    Every entity knows its source. `entity.file` and `entity.line` (a clickable
    `file:line`) point at the exact `*KEYWORD` line — useful when a navigation
    surprises you and you want to open the deck at that spot.

### Entities behind a transform

When an entity sits behind an `*INCLUDE_TRANSFORM`, the id you navigate by is the
**global** (post-offset) id. The offsets that produced it are visible in Python:

=== "Python"

    ```python
    part = deck.part(1000005)          # a global id in a transformed submodel
    if part and part.offsets:
        print(part.offsets)            # {'idnoff': 1000000, 'ideoff': 1000000, ...}
    ```

## Enumerate entities

Don't guess ids — iterate what's actually defined. The definition census tells you
what's there; the per-kind iterators hand you the entities.

=== "Python"

    ```python
    for mat in deck.materials():
        print(mat.id, mat.keyword, f"{mat.file}:{mat.line}")
    # parts(), sections(), curves() likewise.

    # By keyword name — every occurrence of ANY keyword, not just definitions:
    for kw in deck.keywords("CONTACT_TIED_SHELL_EDGE_TO_SURFACE"):
        print(kw.name, f"{kw.file}:{kw.line}")

    # File-first: enumerate files, or scope to one *INCLUDE.
    for f in deck.files():                 # root first, then includes
        print(f.index, f.path)
    contacts = deck.file("modcontacts.k").keywords("CONTACT_TIED_SHELL_EDGE_TO_SURFACE")

    # A quick census of the whole deck:
    for kind, count in deck.definition_counts():
        print(f"{count:>8}  {kind}")
    ```

=== "Rust"

    ```rust
    for mat in deck.materials() {
        println!("{:?} {} at {}:{}", mat.id(), mat.name(), mat.file().display(), mat.line());
    }

    // Or by keyword name, for any keyword:
    for kw in deck.keywords("MAT_PIECEWISE_LINEAR_PLASTICITY") {
        println!("{:?} at {}:{}", kw.id(), kw.file().display(), kw.line());
    }

    for (kind, count) in deck.definition_counts() {
        println!("{count:>8}  {kind:?}");
    }
    ```

!!! note "By kind, by name, or by file"
    Definition entities: `parts()`, `materials()`, `sections()`, `curves()` (both
    languages). Any keyword by name: `deck.keywords(name)` yields every
    occurrence. Scope to one include: `deck.file(suffix)` / `deck.files()` hand
    back a `File` whose `keywords()` are that file's only. For whole *columns* of
    an arbitrary keyword, `deck.table(name)` (below).

## Bulk-read a keyword as columns

For the high-volume keywords (`*NODE`, `*ELEMENT_*`) you rarely want per-entity
handles — you want columns. `Deck.table` reads every occurrence across the whole
deck (root + includes) into a dict of NumPy arrays (Python) / a `Table` (Rust):

=== "Python"

    ```python
    nodes = deck.table("NODE")            # {"nid": int64[N], "x": ..., "y": ..., "z": ...}
    xyz = nodes["x"], nodes["y"], nodes["z"]

    shells = deck.table("ELEMENT_SHELL")  # {"eid", "pid", "nodes": int64[N, 4]}
    conn = shells["nodes"]                # (N, 4) — one column, not four
    ```

=== "Rust"

    ```rust
    let nodes = deck.table("NODE").unwrap();
    let nid = nodes.column("nid").unwrap().as_int().unwrap();     // &[i64]
    let x   = nodes.column("x").unwrap().as_float().unwrap();     // &[f64]
    println!("{} nodes", nid.len());
    ```

For a **low-volume** keyword where you'd rather have rows than columns, Python's
`rows()` helper turns a column dict into per-row dicts lazily:

=== "Python"

    ```python
    import dynars

    kf = dynars.parse_keyword_file("materials.k")
    for m in dynars.rows(dynars.parse_keyword(kf, "MAT_ELASTIC")):
        print(m["MID"], m["RO"], m["E"], m["PR"])
    ```

Keywords the built-in library doesn't ship are still reachable — register a
[schema](schemas.md) and they get the same typed, columnar access via `table_with`
(Python) / a `Schema` (Rust).

## Editing a deck (round-trip)

Navigation and columns are read-only views, but a `Deck` can also **edit** one
field and write the deck back **byte-identical everywhere else** — no whole-deck
rewrite, and nothing but the field you change is touched. Comments (including
`$#` header rulers) and every other card are preserved verbatim; a fixed-format
value is right-justified back into its own column, so the columns don't move.
Find the field with the same navigation you already use, then `set_field`:

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("root.k")

    # By name (schema-aware): retard the termination time.
    deck.keywords("CONTROL_TERMINATION")[0].set_field("endtim", 0.02)

    # By entity id: change a material's Young's modulus.
    deck.material(72).set_field("e", 2.1e11)        # -> "in_place"

    # Scope to one *INCLUDE: a contact's static friction, in modcontacts.k only.
    mc = deck.file("modcontacts.k")
    contact = mc.keywords("CONTACT_TIED_SHELL_EDGE_TO_SURFACE_BEAM_OFFSET")[0]
    contact.set_field("fs", 0.1)

    # Edits are a write-time overlay — realise them by writing the touched files.
    for f in deck.files():
        if f.dirty:
            f.write(f.path)                          # in place, or a new path
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;

    let mut deck = parse_deck(std::path::Path::new("root.k")).unwrap();

    // Navigate (immutable) → snapshot the field's address → apply (&mut deck).
    // `FieldLoc` is borrow-free, so it crosses the read/write borrow split.
    if let Some(loc) = deck.keywords("CONTROL_TERMINATION").next()
        .and_then(|k| k.locate("endtim"))
    {
        deck.set_field(&loc, "0.02");     // Some(FieldEdit::InPlace | Reflowed)
    }

    // File-first: pick one include, then its keyword.
    if let Some(loc) = deck.file("modcontacts.k")
        .and_then(|f| f.keywords_named("CONTACT_TIED_SHELL_EDGE_TO_SURFACE_BEAM_OFFSET").next())
        .and_then(|k| k.locate("fs"))
    {
        deck.set_field(&loc, "0.1");
    }

    // Write the touched files (untouched ones round-trip byte-for-byte).
    for f in &deck.files {
        if f.is_dirty() { f.write(&f.path).unwrap(); }
    }
    ```

`set_field` returns `"in_place"` (only that field's bytes changed) or
`"reflowed"` (the value was wider than its fixed column, so that one card was
re-emitted in free format — no other line moves). Without a schema for the
keyword, either [register one](schemas.md) or use the low-level, explicit-width
form: `deck.file("sub.k").set_field(block, row, col, widths, value)`.

!!! tip "Standalone single files"
    To edit one file with no `*INCLUDE` graph around it, `parse_keyword_file`
    returns a `KeywordFile` with the same lossless round-trip and block-level
    editing (`set_keyword`), plus `to_bytes` / `write`.

## Next steps

- [Schemas](schemas.md) — teach dynars a keyword it doesn't ship, then read it
  like any built-in.
- [Validation](validation.md) — check the deck you just parsed.
- [Recipes](recipes.md) — short snippets for common navigation and extraction
  tasks.
