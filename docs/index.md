# dynars

A fast toolkit for **LS-DYNA keyword decks and binary results**, written in Rust
with first-class Python bindings. The same core parses keyword decks (`*KEYWORD`
files and everything they `*INCLUDE`), navigates and validates them against a
typed model, and reads the binary result files (`d3plot`, `binout`).

Every example in these guides is shown in **both languages** — pick the tab for
the one you use and the rest of the page follows:

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("root.k")

    report = deck.validate([
        dynars.Rule.references_resolve(),   # every id reference resolves
        dynars.Rule.duplicate_ids(),        # no two entities share an id
    ])
    print(report.is_clean(), report.count(dynars.Severity.Error))
    for f in report.findings:
        print(f.location(), "—", f.message)   # clickable file:line
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;
    use dynars::validate::{Rule, Severity};

    let deck = parse_deck(std::path::Path::new("root.k")).unwrap();

    let report = deck.validate([
        Rule::references_resolve(), // every id reference resolves
        Rule::duplicate_ids(),      // no two entities share an id
    ]);
    println!("{} {}", report.is_clean(), report.count(Severity::Error));
    for f in &report.findings {
        println!("{} — {}", f.location(), f.message); // clickable file:line
    }
    ```

## What's here

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, a five-minute tour, and
  your first parse + validate.
- **[Concepts](concepts.md)** — the mental model: deck vs. keyword file, global
  ids, includes & transforms, columns vs. handles.
- **[Decks & navigation](decks.md)** — parse a deck, walk includes, navigate by
  id and follow references, bulk-read, and edit files.
- **[Validation](validation.md)** — the typed rule model, built-in checks, file
  scope, house rules, and custom checks.
- **[Workspace (batch)](workspace.md)** — parse and validate many decks that
  share `*INCLUDE`s against one cache, in parallel.
- **[Results](results.md)** — read and write `d3plot` / `binout`, element
  invariants, signal processing, and occupant-injury criteria.
- **[Schemas](schemas.md)** — teach dynars a keyword it doesn't ship, then read
  it like any built-in.
- **[Recipes](recipes.md)** — short, task-oriented "how do I…" snippets.
- **[Command line](cli.md)** — the `dynars` CLI: parse a deck, scan its include
  tree, generate test decks.
- **[API reference](reference.md)** — the complete Rust and Python API.

</div>

## Why dynars

- **Fast.** The `*INCLUDE` scanner runs at ~15 GB/s per core (SIMD `memchr`) and
  spreads cross-file work over every core; node marshalling parses tens of
  millions of nodes per second.
- **One core, three languages.** Rust, Python (PyO3), and C/Fortran (a C ABI)
  all drive the same engine.
- **Handles the awkward formats.** Fixed-width (8-col), long (`*KEYWORD LONG`),
  and free (comma-separated) cards; Fortran float quirks (`1.5D+3`, `1.234-5`).
