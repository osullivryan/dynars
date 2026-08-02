# Schemas: extending dynars

dynars ships schemas for **~3,170 LS-DYNA keywords**, generated from the Ansys
pyDYNA field database, so the common keywords parse — as columns and as typed
`field(...)` access — with no declaration at all. For a vendor, rare, or
newer-than-our-snapshot keyword, you describe its card layout **once** and it
becomes first-class: same columnar reads, same typed navigation, and (in Rust)
its references get dangling-checked like any built-in.

A schema is just data: each card is a list of fields, each field a `(name, type,
width, count)`. The Rust hot loop executes it — it never calls back into Python
per card, so a declared keyword parses at the same speed as a built-in.

## The built-in library first

Before declaring anything, check whether the keyword is already covered — most
are. Pass a name and it resolves from the library.

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("root.k")
    mats = deck.table("MAT_PIECEWISE_LINEAR_PLASTICITY")   # already known
    ```

=== "Rust"

    ```rust
    // A built-in schema by name:
    let schema = dynars::keywords::schema("MAT_ELASTIC").unwrap();
    ```

To avoid magic strings, every keyword name is also a generated, autocompletable
constant — `dynars.kw.MAT_ELASTIC` in Python,
`dynars::keywords::names::MAT_ELASTIC` in Rust.

## Declaring a keyword

The two front ends — a Python class and a Rust struct — lower to the same schema.

=== "Python"

    ```python
    from dynars import keyword, Card, Int, Float, Str, IntArray, parse_keyword

    @keyword("NODE")                     # one card, repeats over the block
    class Node(Card):
        nid = Int(8)
        x = Float(16); y = Float(16); z = Float(16)

    @keyword("ELEMENT_SHELL")
    class ElementShell(Card):
        eid = Int(8); pid = Int(8)
        nodes = IntArray(4, width=8)     # -> one (N, 4) column

    kf = dynars.parse_keyword_file("deck.k")
    cols = parse_keyword(kf, Node)       # {"nid": int64[N], "x": float64[N], ...}
    conn = parse_keyword(kf, "ElementShell")["nodes"]   # by registered name
    ```

=== "Rust"

    ```rust
    use dynars::{Card, Keyword};

    #[derive(Keyword)]
    #[keyword("NODE")]                   // repeat defaults to true
    struct Node {
        #[field(8)]  nid: i64,           // i64 -> Int, f64 -> Float, String -> Str
        #[field(16)] x: f64,
        #[field(16)] y: f64,
        #[field(16)] z: f64,
    }

    #[derive(Keyword)]
    #[keyword("ELEMENT_SHELL")]
    struct ElementShell {
        #[field(8)] eid: i64,
        #[field(8)] pid: i64,
        #[field(8)] nodes: [i64; 4],     // -> one (N, 4) column
    }

    let nodes = Node::parse(&parsed);    // columnar Table
    let ids = nodes.column("nid").unwrap().as_int().unwrap();
    let _ = ids;
    ```

In Rust the field *type* implies Int/Float/Str, so you only annotate widths. A
`@keyword` class / `#[derive(Keyword)]` with the same name as a built-in
**overrides** it.

## Multi-card keywords

A keyword whose entity spans several lines (a `*PART`'s heading + data, say) is a
list of cards. Reusable card classes/structs compose them.

=== "Python"

    ```python
    from dynars import keyword, Card, Int, Str

    class Heading(Card):
        title = Str(80)

    class PartData(Card):
        pid = Int(8); secid = Int(8); mid = Int(8)

    @keyword("PART")
    class Part:
        cards = [Heading, PartData]      # multi-card
    ```

=== "Rust"

    ```rust
    use dynars::{Card, Keyword};

    #[derive(Card)] struct Heading  { #[field(80)] title: String }
    #[derive(Card)] struct PartData { #[field(8)] pid: i64, #[field(8)] secid: i64, #[field(8)] mid: i64 }

    #[derive(Keyword)]
    #[keyword("PART")]
    #[cards(Heading, PartData)]          // multi-card by composition
    struct Part;
    ```

## Registering on a deck (columns + navigation)

The examples above parse a single `KeywordFile`. To make a custom keyword
first-class on a whole **`Deck`** — so `table_with` reads it across all includes,
and (Rust) navigation and reference checks understand it — register it on the
deck.

=== "Python"

    ```python
    deck = dynars.parse_deck("root.k")
    cards = [[("wid", "int", 8, 1), ("mass", "float", 8, 1)]]  # (name, type, width, count)

    deck.register_schema("VENDOR_WIDGET", cards)               # for navigation + rules
    cols = deck.table_with("VENDOR_WIDGET", cards)             # {"wid": int64[N], "mass": float64[N]}

    # Rules can now target it:
    from dynars import Rule, Predicate, Cmp
    deck.validate([Rule.field_required("VENDOR_WIDGET",
                                       require=Predicate.field("mass", Cmp.Gt, 0.0))])
    ```

=== "Rust"

    ```rust
    use dynars::schema::{Schema, Card};
    use dynars::keywords::EntityKind;
    use dynars::validate::Rule;

    // `deck` must be `let mut` to register. A ref_to field declares a reference,
    // so references_resolve() dangling-checks it like any built-in.
    deck.register_schema(Schema::new("VENDOR_WIDGET").card(
        Card::new()
            .int("wid", 8)
            .float("mass", 8)
            .ref_to("mat", 8, EntityKind::Material),  // id references a *MAT
    ));

    let w = deck.keywords("VENDOR_WIDGET").next().unwrap();
    let mass = w.card(0).and_then(|c| c.field("mass")).and_then(|f| f.as_f64());
    let mat  = w.card(0).and_then(|c| c.field("mat")).and_then(|f| f.reference());
    let report = deck.validate([Rule::references_resolve()]);
    let _ = (mass, mat, report);
    ```

## Fully typed Rust access

For a keyword you know at compile time, the opt-in `typed-keywords` feature
generates a struct per built-in keyword with named, typed column fields:

```rust
// Cargo.toml: dynars = { version = "0.1", features = ["typed-keywords"] }
let m = dynars::keywords::typed::MAT_ELASTIC::parse(&parsed);
let (mid, e) = (m.mid, m.e);   // Vec<i64>, Vec<f64>
```

It's off by default — enabling it compiles ~3,170 structs (a one-time, cached
cost), so the base build stays fast.

## Scope

Schemas cover fixed `K`-cards-per-entity layouts (repeating or single-entity),
`int`/`float`/`str` and array fields, in fixed / long / free formats.
*Conditional* or *count-driven* cards (e.g. `*DEFINE_CURVE`, whose card count
depends on a field) are out of scope — those keywords stay in the generic, lazy
field model. Runnable examples: `examples/schema_demo.{rs,py}`,
`examples/derive_demo.rs`, `examples/builtin_demo.{rs,py}`.
