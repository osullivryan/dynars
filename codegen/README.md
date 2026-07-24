# codegen — built-in keyword library

`src/keywords/data.rs` (the built-in keyword registry) is generated from the
[Ansys pyDYNA](https://github.com/ansys/pydyna) field database `kwd.json`, which
maps each LS-DYNA keyword to its cards and per-field `{name, type, width}`.

## Regenerate

```bash
# kwd.json is ~19 MB and not vendored (gitignored):
curl -sSL -o codegen/kwd.json \
  https://raw.githubusercontent.com/ansys/pydyna/main/codegen/kwd.json

python codegen/gen_keywords.py            # -> src/keywords/data.rs
cargo test keywords                       # sanity-check the registry
```

## What's covered

- **3,168** keywords generated from `kwd.json`, plus a small hand-written
  **supplement** in `src/keywords/mod.rs` for fundamentals pyDYNA omits
  (`*NODE`, `*PART` — it handles those via dedicated mesh APIs, so they aren't in
  its keyword database).
- Field types map `integer → Int`, `real → Float`, `string → Str`,
  `real-integer → Float`. Unused/placeholder fields are kept (their widths set
  later fields' offsets) and given unique names.

## Limitations

`kwd.json` describes each keyword's **static** card layout. A minority of
keywords have **conditional** cards (present only if a flag is set) or
**count-driven** cards (repeated `N` times). Their generated schema is the base
layout and may under-read the variable tail — those need the schema's
conditional/variable-card support (see `manifest.json` in pyDYNA for the rules).
