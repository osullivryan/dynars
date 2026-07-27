# Using dynars from C and Fortran

The deck **parse + validate** path is exposed over a C ABI (`src/ffi.rs`, behind
the `ffi` cargo feature). C calls it directly; Fortran calls it through
`iso_c_binding` — there is no direct Rust↔Fortran bridge, both meet at the C ABI.

Only parse + validate is exported. Navigation, columnar tables, and the
d3plot/binout readers are not part of this surface.

## Build

```sh
# from this directory (examples/ffi/)
make            # builds the Rust lib with --features ffi, then both examples

# run against the bundled multi-file deck (root + nested *INCLUDEs)
DECK=../../tests/data/bolt_a_explicit/mainboltaexpl.k
./c_example  "$DECK"
./f_example  "$DECK"
```

`example.c` and `example.f90` produce byte-identical output and the same exit
code (non-zero if any error-severity finding is reported).

`make` runs `cargo build --release --features ffi`, which produces the
C-linkable libraries in `../../target/release/`:

- `libdynars.dylib` / `libdynars.so` — shared (what the examples link against)
- `libdynars.a` — static, if you prefer to link statically

The `ffi` feature is off by default, so a normal `cargo build` compiles none of
the `unsafe` boundary code.

## What the examples do

Both parse a deck, run `references_resolve` + `include_missing`, print each
finding as `severity file:line message`, and exit non-zero if any error-severity
finding was reported. `example.c` and `example.f90` are line-for-line
equivalents so you can compare the two bindings.

## The API (see `dynars.h`)

- **Handles** — `DynarsDeck`, `DynarsRuleSet`, `DynarsReport` are opaque
  pointers you own; release each with the matching `*_free`.
- **Errors** — fallible calls return `NULL`/`-1` and set a thread-local message
  you read with `dynars_last_error()`.
- **Strings** — finding accessors return `const char*` that live inside the
  `DynarsReport`; copy them before `dynars_report_free`.

Available rules: `references_resolve`,
`references_resolve_with_connectivity`, `include_missing`, `keyword_forbidden`.

## Regenerating the header (optional)

`dynars.h` is hand-maintained and authoritative. To regenerate it from the Rust
source with [cbindgen](https://github.com/mozilla/cbindgen):

```sh
cbindgen --config cbindgen.toml --crate dynars --output dynars.h -- --features ffi
```
