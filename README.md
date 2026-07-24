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

## User-defined keyword schemas

Beyond the built-in `*NODE`/`*ELEMENT_*` parsers, you can declare how to marshal
*any* keyword — no recompile — and get the same columnar output. The declaration
is data (field layout), executed by the Rust hot loop; it never calls back into
Python per card, so it stays fast.

**Python** — a keyword is a class; fields on the class are one card, or a
`cards` list composes several. Reusable card classes and array fields included:

```python
from dynars import keyword, Card, Int, Float, Str, IntArray, parse_keyword

@keyword("NODE")                      # one card, repeats over the block
class Node(Card):
    nid = Int(8)
    x = Float(16); y = Float(16); z = Float(16)

@keyword("ELEMENT_SHELL")
class ElementShell(Card):
    eid = Int(8); pid = Int(8)
    nodes = IntArray(4, width=8)      # -> one (N, 4) column

class Heading(Card):                  # reusable cards
    title = Str(80)
class PartData(Card):
    pid = Int(8); secid = Int(8); mid = Int(8)

@keyword("PART")
class Part:
    cards = [Heading, PartData]       # multi-card

cols = parse_keyword(kf, Node)        # {"nid": int64[N], "x": float64[N], ...}
conn = parse_keyword(kf, "ElementShell")["nodes"]   # int64[N, 4]
```

**Rust** — the mirror of the Python class is `#[derive(Keyword)]` on a struct;
field *types* imply Int/Float/Str, so you only annotate widths:

```rust
use dynars::{Card, Keyword, KeywordSchema};

#[derive(Keyword)]
#[keyword("NODE")]                        // repeat defaults to true
struct Node {
    #[field(8)]  nid: i64,                // i64 -> Int, f64 -> Float, String -> Str
    #[field(16)] x: f64,
    #[field(16)] y: f64,
    #[field(16)] z: f64,
}

#[derive(Keyword)]
#[keyword("ELEMENT_SHELL")]
struct ElementShell {
    #[field(8)] eid: i64,
    #[field(8)] pid: i64,
    #[field(8)] nodes: [i64; 4],          // -> one (N, 4) column
}

#[derive(Card)] struct Heading  { #[field(80)] title: String }
#[derive(Card)] struct PartData { #[field(8)] pid: i64, #[field(8)] secid: i64, #[field(8)] mid: i64 }

#[derive(Keyword)]
#[keyword("PART")]
#[cards(Heading, PartData)]               // multi-card by composition
struct Part;

let nodes = Node::parse(&parsed);         // columnar Table
let ids = nodes.column("nid").unwrap().as_int().unwrap();
```

Or, for dynamic construction, the underlying builder that the derive and the
Python classes both lower to:

```rust
use dynars::schema::{parse_schema, Card, Schema};
let node = Schema::new("NODE")
    .card(Card::new().int("nid", 8).float("x", 16).float("y", 16).float("z", 16));
let t = parse_schema(&parsed, &node);
```

Runnable examples: `examples/derive_demo.rs`, `examples/schema_demo.rs`,
`examples/schema_demo.py`.

Scope: fixed `K`-cards-per-entity layouts (repeating or single-entity),
`int`/`float`/`str` and array fields, fixed/long/free formats. *Conditional* or
*count-driven* cards (e.g. `*DEFINE_CURVE`) are out of scope and stay in Rust or
the generic `Keyword` model.

## Design

The two capabilities are deliberately separate, and marshalling is additive —
the include-tree path is unchanged and pays nothing for the new features.

- **Scanner** (`parser::parse_file_from_path`): memory-maps the file (no read()
  copy) and scans for `*` at line starts with SIMD `memchr`, finding includes
  anywhere in the file. Files ≥ 8 MB are scanned in parallel over line-aligned
  chunks; because the mapping is contiguous, a chunk that finds a keyword near
  its end reads forward for the filename, so straddling lines need no overlap
  buffer. Across many files, a work-stealing pool (`include_tree`) parallelizes
  by file. (mmap parallel scanning scales on Linux, where page faults resolve
  concurrently; on macOS minor faults serialize, so single-file scans there are
  bounded near single-thread speed — but the copy-elimination still helps.)
- **Block index** (`parser::parse_file_blocks`): memory-maps the file and splits
  it into keyword blocks that *tile the source exactly* (no read() copy). This is
  the lossless round-trip guarantee — re-emitting every block reproduces the
  input. Edits are an overlay keyed by block index; the backing bytes are a
  `Source` (mapped or owned) so both parse and construct-from-bytes work.
- **Tokenizer** (`Field`, `split_fields`, `CardIter`): lazy, format-aware field
  splitting for the long tail of keywords. Nothing is parsed until read.
- **Columnar parsers** (`bulk`): struct-of-arrays parsers for `*NODE` and
  `*ELEMENT_*`, parallelized with rayon over line-aligned chunks, using
  `lexical` for fast float/int conversion. These map straight onto numpy.
- **Owned model + typed structs** (`parser::Keyword`, `typed`): an editable,
  allocation-backed view for round-trip editing, plus example typed structs
  (`Part`, `MatElastic`) that any keyword can follow.
- **Schemas** (`schema`, `dynars-derive`): a declarative keyword layout (cards →
  typed fields) parsed into columnar `Table`s. Single-card repeating keywords
  parse in parallel like the built-ins; multi-card ones parse sequentially. Three
  front ends lower to one `Schema`: the Rust builder, `#[derive(Keyword)]`
  structs (proc-macro in the `dynars-derive` workspace crate), and the Python
  `@keyword` classes.

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
| `*INCLUDE` scan, single large file (macOS, fault-bound) | ~15 GB/s |
| `*INCLUDE` scan, many warm files (mmap, no copy) | ~45 GB/s |
| Block index (mmap + split) | ~15 GB/s |
| Node parse → arrays (Rust) | ~97 M nodes/s (51 ms) |
| Node parse → numpy (Python) | ~82 M nodes/s (61 ms) |
| Node parse, `#[derive(Keyword)]` (specialized) | ~70 M nodes/s |
| Node parse, builder / Python (interpreted) | ~57–64 M nodes/s |

`#[derive(Keyword)]` emits monomorphized code (offsets known at compile time,
no per-field enum dispatch), so its `parse()` runs ~20% faster than the
interpreted builder/Python path. It doesn't fully match the hand-specialized
`parse_nodes` (~97 M/s), which packs `xyz` into one array; the schema path emits
one column per field. All three are tens of millions of entities per second.
Reproduce with `cargo run --release --example bench_schema`.

The multi-file number roughly doubled after switching the scanner from `read()`
to `mmap` (eliminating a copy of every file). Single-file parallel scanning is
bounded by macOS's serialized page faults here; on Linux it should scale toward
memory bandwidth.

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
