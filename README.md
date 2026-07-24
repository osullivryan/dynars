# dynars

High-performance LS-DYNA keyword file parser, written in Rust with first-class
Python bindings.

`dynars` does two things, both built for very large decks:

1. **Include-tree scanning** — walk a deck's `*INCLUDE` graph across many files,
   in parallel, at memory-bandwidth speed.
2. **Keyword marshalling** — index a file into keyword blocks, read the
   high-volume keywords (`*NODE`, `*ELEMENT_*`) as columnar numpy arrays, edit
   individual keywords, and write the deck back with **byte-for-byte lossless
   round-trip** of everything you didn't touch.

## Highlights

- **Fast.** The `*INCLUDE` scanner runs at ~15 GB/s per core (SIMD `memchr`);
  cross-file work is spread over all cores. Node marshalling parses ~73 M
  nodes/s across 10 cores.
- **Lossless.** A parse → write cycle of an unedited deck is a no-op diff.
  Edits are tracked per keyword block; untouched blocks are re-emitted verbatim.
- **Zero-copy to numpy.** `*NODE` coordinates and element connectivity cross the
  FFI boundary as numpy arrays without a copy.
- **Handles the awkward formats.** Fixed-width (8-col), long (`*KEYWORD LONG`),
  and free (comma-separated) cards; Fortran float quirks (`1.5D+3`, `1.234-5`).

## Installation

### Python

Requires a Rust toolchain and [maturin](https://www.maturin.rs/).

```bash
# Editable dev install into the active virtualenv
maturin develop --release

# Or build a distributable wheel
maturin build --release
pip install target/wheels/dynars-*.whl
```

`numpy` is a runtime dependency of the Python package.

### Rust / CLI

```bash
cargo build --release
# binary at target/release/dynars
```

## Command-line usage

```bash
# Parse a deck and print the include tree + throughput
dynars parse root.k

# Generate a synthetic deck for benchmarking
dynars generate --depth 6 --breadth 4 --nodes 100000 --output test_output
```

## Python API

### Include tree

```python
import dynars

root = dynars.parse_include_tree("root.k")
print(root.total_files(), root.total_bytes())
for child in root.children:
    print(child.path, child.kind, child.byte_count)
```

### Keyword marshalling

```python
import dynars

kf = dynars.parse_keyword_file("deck.k")
print(kf.num_blocks, kf.block_names())

# Bulk geometry as numpy arrays (zero-copy)
ids, coords = kf.nodes()               # int64[N], float64[N, 3]
eids, pids, conn = kf.elements_shell() # int64[N], int64[N], int64[N, 4]

# Edit node coordinates and write back; untouched blocks stay byte-identical
import numpy as np
new = coords.copy()
new[:, 2] += 5.0
kf.set_node_coords(new)
kf.write("deck_shifted.k")

# Generic access + edit for any of the ~2000 keywords
i = kf.block_names().index("MAT_ELASTIC")
kw = kf.keyword(i)                      # {"name", "options", "cards"}
kw_cards = kw["cards"]
kw_cards[0][2] = "70000.0"              # change Young's modulus
kf.set_keyword(i, "MAT_ELASTIC", kw_cards)

raw = kf.to_bytes()                     # the (edited) deck as bytes
```

## Rust API

```rust
use dynars::include_tree::build_include_tree;
use dynars::parser::parse_file_blocks;
use dynars::bulk::{parse_nodes, parse_element_shell};

// Include tree
let tree = build_include_tree(std::path::Path::new("root.k")).unwrap();

// Marshalling
let parsed = parse_file_blocks(std::path::Path::new("deck.k")).unwrap();
let nodes = parse_nodes(&parsed);          // NodeArrays { ids, coords }
let shells = parse_element_shell(&parsed);  // ElementArrays { eids, pids, nodes, .. }

// Lossless round-trip: no edits -> identical bytes
assert_eq!(parsed.to_bytes(), std::fs::read("deck.k").unwrap());
```

## Design

The two capabilities are deliberately separate, and marshalling is additive —
the include-tree path is unchanged and pays nothing for the new features.

- **Streaming scanner** (`parser::parse_file_from_path`): reads a file in 4 MB
  chunks (not `mmap` — macOS page faults are single-threaded), scans for `*` at
  line starts with SIMD `memchr`, and finds includes anywhere in the file.
  Parallelism is across files via a work-stealing pool (`include_tree`).
- **Block index** (`parser::parse_file_blocks`): owns the file buffer and splits
  it into keyword blocks that *tile the source exactly*. This is the lossless
  round-trip guarantee — re-emitting every block reproduces the input. Edits are
  an overlay keyed by block index.
- **Tokenizer** (`Field`, `split_fields`, `CardIter`): lazy, format-aware field
  splitting for the long tail of keywords. Nothing is parsed until read.
- **Columnar parsers** (`bulk`): struct-of-arrays parsers for `*NODE` and
  `*ELEMENT_*`, parallelized with rayon over line-aligned chunks, using
  `lexical` for fast float/int conversion. These map straight onto numpy.
- **Owned model + typed structs** (`parser::Keyword`, `typed`): an editable,
  allocation-backed view for round-trip editing, plus example typed structs
  (`Part`, `MatElastic`) that any keyword can follow.

### Card formats

Fixed-width is the default (8-col; `*NODE` uses its exact I8/E16 widths). Long
format is detected from `*KEYWORD LONG=Y|S`. Free format is decided per line —
a line switches to comma-splitting the moment it contains a comma, matching
LS-DYNA's own rule.

## Performance

Measured on a 10-core Apple Silicon machine, 386 MB single-file deck (5 M
nodes), warm page cache:

| Operation | Throughput |
|-----------|-----------|
| `*INCLUDE` scan (streaming, per core) | ~15 GB/s |
| Block index (read + split) | ~12 GB/s |
| Node parse → arrays (Rust) | ~73 M nodes/s (68 ms) |
| Node parse → numpy (Python) | ~64 M nodes/s (78 ms) |

Cold decks larger than RAM are limited by disk bandwidth (~2 GB/s sustained
NVMe), not CPU — the scanner is ~7× faster than the disk can deliver bytes.

Reproduce the marshalling numbers:

```bash
cargo run --release --example bench_marshal
```

## Development

```bash
cargo test                    # Rust unit + integration tests
cargo check --features python # type-check the pyo3 bindings

# Regenerate the Python type stub after changing the API
maturin generate-stubs --features python --out python/dynars
```

Note: `generate-stubs` is a maturin **CLI** step, not a `pyproject.toml` key —
run the command above whenever the Python API changes.

## Status

Speed is mature. The open frontier is correctness coverage on real decks:
long-format field widths, multi-line element cards, and per-keyword field
schemas for the generic splitter. Validation against representative customer
`.k` files is the highest-value next step.
