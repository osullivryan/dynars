# Getting started

This page takes you from an empty environment to a program that parses a real
LS-DYNA deck, navigates it, validates it, and reads a result file — in about ten
minutes. Every snippet is shown in **both languages**; pick the tab for the one
you use and the whole page follows.

## Install

=== "Python"

    ```bash
    pip install dynars
    ```

    Requires Python 3.9+. Numeric data (node coordinates, element connectivity,
    result channels) comes back as NumPy arrays, so `numpy` is pulled in as a
    dependency. The published wheels bundle the `signal` feature, so filtering and
    injury criteria work out of the box — nothing else to enable.

    Verify the install:

    ```python
    import dynars
    print(dynars.__version__ if hasattr(dynars, "__version__") else "ok")
    print([n for n in dir(dynars) if not n.startswith("_")][:12])
    ```

=== "Rust"

    ```bash
    cargo add dynars
    ```

    Or add it to `Cargo.toml`:

    ```toml
    [dependencies]
    dynars = "1.0"
    ```

    Optional features — off by default so a plain build stays lean:

    ```toml
    [dependencies]
    dynars = { version = "1.0", features = ["signal", "ffi"] }
    ```

    | Feature | Enables |
    |---------|---------|
    | `signal` | result-history signal processing (SAE J211 CFC, Butterworth, integrate/differentiate) and the injury criteria |
    | `ffi` | a C ABI (and, through it, Fortran) for the parse + validate path |
    | `typed-keywords` | a generated typed struct per keyword (~3,170; opt-in) |

    See [feature flags](reference.md#rust) for the full matrix.

Prefer the command line? `cargo install dynars` installs a `dynars` binary that
parses a deck and prints its include tree — see [CLI](cli.md).

## Your first program

`parse_deck` reads the root file **and everything it `*INCLUDE`s** in one parallel
pass, and hands back a single `Deck`. You validate and navigate off that one
handle — the id and reference indices are built lazily and cached on first use, so
a parse that only reads columns never pays for them.

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("main.k")
    print(deck)  # Deck(<n> files)

    report = deck.validate([
        dynars.Rule.references_resolve(),   # every id reference resolves
        dynars.Rule.duplicate_ids(),        # no two entities share an id
        dynars.Rule.include_missing(),      # every *INCLUDE exists on disk
    ])

    if report.is_clean():
        print("no errors")
    else:
        for f in report.findings:
            print(f"[{f.severity}] {f.location()} — {f.message}")
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;
    use dynars::validate::Rule;

    fn main() {
        let deck = parse_deck(std::path::Path::new("main.k")).unwrap();
        println!("{} files", deck.files.len());

        let report = deck.validate([
            Rule::references_resolve(), // every id reference resolves
            Rule::duplicate_ids(),      // no two entities share an id
            Rule::include_missing(),    // every *INCLUDE exists on disk
        ]);

        if report.is_clean() {
            println!("no errors");
        } else {
            for f in &report.findings {
                println!("[{:?}] {} — {}", f.severity, f.location(), f.message);
            }
        }
    }
    ```

A **finding** carries a `severity`, a human-readable `message`, and a clickable
`file:line` `location()`. A report `is_clean()` when it has no `Error`-severity
findings (warnings are allowed). There is **no default rule set** — you pass
exactly the checks you want.

## A five-minute tour

The same `Deck` is your entry point for four different jobs. Here they are back
to back so you can see how they fit together.

### 1. Inspect what the deck contains

Get a census before diving in — the definition counts tell you what kinds of
entity are defined and how many of each.

=== "Python"

    ```python
    deck = dynars.parse_deck("main.k")
    for kind, count in deck.definition_counts():
        print(f"{count:>8}  {kind}")
    # e.g.  500000  Node / 480000  Element / 312  Part / 45  Material ...
    ```

=== "Rust"

    ```rust
    let deck = parse_deck(std::path::Path::new("main.k")).unwrap();
    for (kind, count) in deck.definition_counts() {
        println!("{count:>8}  {kind:?}");
    }
    ```

### 2. Navigate by id and follow references

Look an entity up by id, then walk the references in its fields — a `*PART`'s
material and section, a load's curve, and so on. Ids resolve in the deck's
**global** namespace, so references that cross an `*INCLUDE_TRANSFORM` are followed
correctly.

=== "Python"

    ```python
    part = deck.part(1)
    if part is not None:
        mat = part.material()           # follow *PART.mid -> *MAT
        sec = part.section()            # follow *PART.secid -> *SECTION
        print(part.id, part.keyword, "at", f"{part.file}:{part.line}")
        print("  density:", mat.field("RO") if mat else None)
    ```

=== "Rust"

    ```rust
    if let Some(part) = deck.part(1) {
        let mat = part.material();      // follow *PART.mid -> *MAT
        let sec = part.section();       // follow *PART.secid -> *SECTION
        println!("part {:?} ({})", part.id(), part.name());
        if let Some(m) = mat {
            println!("  density: {:?}", m.field("RO").and_then(|f| f.as_f64()));
        }
        let _ = sec;
    }
    ```

### 3. Bulk-read the high-volume keywords as columns

For `*NODE` and `*ELEMENT_*` you rarely want per-entity handles — you want
columns. `table` reads every occurrence across the **whole deck** (root +
includes) at once.

=== "Python"

    ```python
    nodes = deck.table("NODE")          # {"nid": int64[N], "x": float64[N], ...}
    print(nodes["nid"].shape, nodes["x"].mean())

    shells = deck.table("ELEMENT_SHELL")  # {"eid", "pid", "nodes": int64[N, 4]}
    ```

=== "Rust"

    ```rust
    let nodes = deck.table("NODE").unwrap();
    let ids = nodes.column("nid").unwrap().as_int().unwrap();
    let xs = nodes.column("x").unwrap().as_float().unwrap();
    println!("{} nodes, mean x = {:.3}", ids.len(), xs.iter().sum::<f64>() / xs.len() as f64);
    ```

### 4. Read the result files

Once the run has finished, the same package reads the binary output — `d3plot`
(geometry + per-state fields) and `binout` (time histories). Numeric data comes
back as NumPy arrays in Python, typed `Vec`s in Rust.

=== "Python"

    ```python
    d = dynars.open_d3plot("d3plot")           # opens the whole family
    print(d.num_nodes, d.num_states)
    print("peak displacement:", d.max_displacement_final())
    ```

=== "Rust"

    ```rust
    use dynars::results::D3plot;

    let d = D3plot::open("d3plot").unwrap();
    println!("{} nodes, {} states", d.num_nodes(), d.num_states());
    ```

That is the whole surface in miniature: **parse → inspect → navigate → validate →
read results**. The rest of the guides go deep on each.

## What you get back

A few types show up everywhere; knowing them makes the rest of the docs read
easily.

| Type | What it is |
|------|------------|
| `Deck` | the parsed root + all includes; the single handle for navigation, columns, and validation |
| `Entity` (Py) / `Keyword` (Rust) | one entity — typed `field(...)` access, source `file`/`line`, and reference-following (`material()`, `section()`, `reference(name)`) |
| `Report` | the result of `validate(...)` — `is_clean()`, `count(severity)`, and a list of `findings` |
| `Finding` | one violation — `severity`, `rule`, `message`, and a clickable `location()` |
| `D3plot` / `Binout` | the two result readers |

If those distinctions feel fuzzy, the [Concepts](concepts.md) page draws the
mental model — deck vs. keyword file, global ids, includes and transforms.

## Troubleshooting

- **`parse_deck` succeeds but an entity is missing.** A missing `*INCLUDE` is
  never parsed, so its entities simply aren't there. Add
  [`Rule.include_missing()`](validation.md) to surface the missing file
  explicitly.
- **A reference "doesn't resolve" but the target is clearly present.** Check
  whether it lives behind an `*INCLUDE_TRANSFORM` — ids are matched in the global
  namespace after offsets are applied; see [Concepts → includes &
  transforms](concepts.md#includes-and-transforms).
- **`deck.table("FOO")` raises / returns nothing.** `FOO` isn't in the built-in
  library. Register a [schema](schemas.md) and read it with `table_with`.
- **Signal / injury functions missing in Rust.** They live behind the `signal`
  feature: `dynars = { version = "1.0", features = ["signal"] }`. The Python
  wheels already include it.

## Next steps

- [Concepts](concepts.md) — the mental model behind the API.
- [Decks & navigation](decks.md) — navigate by id, follow references, bulk-read,
  and edit decks.
- [Validation](validation.md) — the full rule set and how to write your own
  checks.
- [Workspace (batch)](workspace.md) — do all of this across many decks at once
  without re-reading shared files.
- [Results](results.md) — `d3plot` / `binout`, signal processing, injury criteria.
- [Recipes](recipes.md) — short, task-oriented "how do I…" snippets.
