# Decks & navigation

## Parsing a deck

`parse_deck` takes the **root** keyword file and follows its `*INCLUDE` graph —
`*INCLUDE`, `*INCLUDE_PATH`, `*INCLUDE_TRANSFORM`, and friends — parsing every
reachable file in one parallel pass. The result is a single `Deck` handle that
owns all the parsed files.

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("main.k")
    print(len(dynars.parse_include_tree("main.k").children), "top-level includes")
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;
    use dynars::include::build_include_tree;

    let deck = parse_deck(std::path::Path::new("main.k")).unwrap();
    let tree = build_include_tree(std::path::Path::new("main.k")).unwrap();
    println!("{} files, {} bytes", deck.files.len(), tree.total_bytes());
    ```

## Navigate by id, follow references

Look an entity up by its id, then follow the references in its fields — a
`*PART`'s `mid` to its `*MAT`, its `secid` to its `*SECTION`, a load's `lcid` to
its `*DEFINE_CURVE`, and so on. Ids are resolved in the deck's **global**
namespace, so references that cross an `*INCLUDE_TRANSFORM` are followed
correctly.

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
    }
    ```

`part`, `material`, `section`, and `curve` are the common lookups; every other
kind is reachable too (see the [API reference](reference.md)).

## Enumerate entities

Don't guess ids — iterate what's actually defined:

=== "Python"

    ```python
    for mat in deck.materials():
        print(mat.id, mat.keyword, f"{mat.file}:{mat.line}")

    # A quick census of what the deck defines:
    for kind, count in deck.definition_counts():
        print(f"{count:>8}  {kind}")
    ```

=== "Rust"

    ```rust
    for kw in deck.keywords("MAT_ELASTIC") {
        println!("{:?} at {}:{}", kw.id(), kw.file().display(), kw.line());
    }

    for (kind, count) in deck.definition_counts() {
        println!("{count:>8}  {kind:?}");
    }
    ```

## Bulk-read a keyword as columns

For the high-volume keywords (`*NODE`, `*ELEMENT_*`) you rarely want per-entity
handles — you want columns. In Python, `Deck.table` reads every occurrence across
the whole deck (root + includes) into a dict of NumPy arrays:

=== "Python"

    ```python
    nodes = deck.table("NODE")          # {'nid': int64[...], 'x': float64[...], ...}
    xyz = nodes["x"], nodes["y"], nodes["z"]

    shells = deck.table("ELEMENT_SHELL")  # {'eid':..., 'pid':..., 'n1':..., ...}
    ```

=== "Rust"

    ```rust
    // Iterate occurrences and read typed fields off the navigation spine:
    for node in deck.keywords("NODE") {
        for card in node.cards() {
            let nid = card.field("nid").and_then(|f| f.as_i64());
            let x = card.field("x").and_then(|f| f.as_f64());
            // ...
            let _ = (nid, x);
        }
    }
    ```

Keywords the built-in library doesn't ship are still reachable — register a
[user schema](reference.md) (`register_schema` / `table_with` in Python,
`#[derive(Keyword)]` in Rust) and they get the same typed, columnar access.
