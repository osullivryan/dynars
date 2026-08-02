# API reference

The guides cover the common paths; the complete, generated API for every public
item lives here:

<div class="grid cards" markdown>

- **[Rust API →](https://osullivryan.github.io/dynars/rust/dynars/)**
  <br>rustdoc for the `dynars` crate.
- **[Python API →](https://osullivryan.github.io/dynars/python/dynars.html)**
  <br>pdoc for the `dynars` package.

</div>

## Rust

Add the crate and pick the features you need:

```toml
[dependencies]
dynars = "1.0"
# dynars = { version = "1.0", features = ["signal", "ffi"] }
```

| Feature | Enables |
|---------|---------|
| *(default)* | parsing, navigation, validation, `d3plot` / `binout` readers |
| `signal` | result-history signal processing (SAE J211 CFC, Butterworth, integrate/differentiate) and injury criteria |
| `ffi` | C ABI (and, through it, Fortran) bindings for the parse + validate path |
| `typed-keywords` | a generated typed struct per keyword (~3170; opt-in) |

## Python

```bash
pip install dynars
```

The published wheels bundle the `signal` feature, so filtering and injury
criteria are available out of the box. `import dynars` exposes everything —
`parse_deck`, `Workspace`, `Rule`, `D3plot`, `Binout`, the signal/injury
functions, and the keyword-name constants under `dynars.kw`.

## Keywords dynars doesn't ship

The built-in library covers thousands of LS-DYNA keywords. For a vendor, newer,
or otherwise unlisted keyword, describe its card layout once and get the same
typed, columnar access:

=== "Python"

    ```python
    deck.register_schema(
        "MY_KEYWORD",
        cards=[[("id", "int", 8, 1), ("value", "float", 16, 1)]],
    )
    rows = deck.table("MY_KEYWORD")   # now columnar, like any built-in
    ```

=== "Rust"

    ```rust
    use dynars::Keyword;

    #[derive(Keyword)]
    #[keyword("MY_KEYWORD")]
    struct MyKeyword {
        id: i64,
        value: f64,
    }
    ```
