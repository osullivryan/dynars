# Getting started

## Install

=== "Python"

    ```bash
    pip install dynars
    ```

    Requires Python 3.9+. Numeric data (node coordinates, element connectivity,
    result channels) comes back as NumPy arrays, so `numpy` is pulled in as a
    dependency.

=== "Rust"

    ```bash
    cargo add dynars
    ```

    Or add it to `Cargo.toml`:

    ```toml
    [dependencies]
    dynars = "0.1"
    ```

    Optional features: `signal` (result-history filtering), `ffi` (C/Fortran
    bindings). See [feature flags](reference.md#rust).

## Your first program

Parse a deck, validate it, and print any problems. `parse_deck` reads the root
file **and everything it `*INCLUDE`s** in one pass; you then validate and
navigate off the one handle — the resolution indices are built lazily and cached,
so a plain parse pays for neither.

=== "Python"

    ```python
    import dynars

    deck = dynars.parse_deck("main.k")
    print(deck)  # Deck(<n> files)

    report = deck.validate([
        dynars.Rule.references_resolve(),
        dynars.Rule.duplicate_ids(),
        dynars.Rule.include_missing(),   # flag *INCLUDEs that don't exist on disk
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
            Rule::references_resolve(),
            Rule::duplicate_ids(),
            Rule::include_missing(), // flag *INCLUDEs that don't exist on disk
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
findings (warnings are allowed).

## Next steps

- [Decks & navigation](decks.md) — navigate by id, follow references, and
  bulk-read keywords as arrays.
- [Validation](validation.md) — the full rule set and how to write your own
  checks.
- [Workspace (batch)](workspace.md) — do this across many decks at once without
  re-reading shared files.
