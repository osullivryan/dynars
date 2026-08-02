# Command line

`cargo install dynars` installs a `dynars` binary — a thin front end over the
include-tree scanner. It's handy for a quick "what does this deck include, and how
fast does it scan?" without writing any code, and its `--json` output drops
straight into a shell pipeline.

```bash
cargo install dynars
dynars --help
```

## `dynars parse`

Parse a root keyword file, walk its `*INCLUDE` graph, and print the tree plus
scan throughput.

```bash
dynars parse root.k
```

```text
Parsing: root.k
Threads: 10

=== Include Tree ===
root.k
  mesh/nodes.k
  mesh/shells.k
  materials.k
  ...

=== Performance ===
Total files:  128
Total bytes:  402653184 (384.00 MB)
Parse time:   0.026s (26.4ms)
Throughput:   14550.1 MB/s
              4848 files/s
```

Trees larger than 200 files print a summary line instead of the full listing.

### JSON output

`--json` emits the whole tree plus timing as structured JSON — pipe it to `jq`,
feed a dashboard, or diff two decks' file lists.

```bash
dynars parse root.k --json | jq '.total_files, .total_bytes'
```

```json
{
  "root": "root.k",
  "total_files": 128,
  "total_bytes": 402653184,
  "parse_seconds": 0.0264,
  "throughput_mb_s": 14550.1,
  "tree": { "path": "root.k", "byte_count": 4096, "kind": null, "children": [ ... ] }
}
```

On a parse error, `parse` exits non-zero and writes the error (as `{"error": …}`
under `--json`, or a plain `Error:` line otherwise) to stderr.

## `dynars generate`

Generate a synthetic deck — a tree of `*INCLUDE`s full of `*NODE` lines — for
benchmarking the scanner and the columnar path on decks of a chosen size and
shape.

```bash
dynars generate --depth 6 --breadth 4 --nodes 100000 --output test_output
```

| Flag | Default | Meaning |
|------|--------:|---------|
| `--depth` | 6 | include nesting depth |
| `--breadth` | 4 | includes per file at each level |
| `--nodes` | 100000 | `*NODE` lines per file |
| `--output` | `test_output` | output directory |

The resulting `test_output/root.k` is a normal deck — parse it with
`dynars parse`, or point `parse_deck` at it from Python/Rust.

## Beyond the CLI

The binary intentionally covers only the include-tree scan. For navigation,
validation, columnar reads, and result files, use the library — start with
[Getting started](getting-started.md). Validation makes an easy pre-submit gate;
see [gating a pipeline](validation.md#gate-a-pipeline-on-the-result) for a
ready-to-adapt script.
