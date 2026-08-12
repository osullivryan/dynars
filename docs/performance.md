# Performance

Deck parsing and columnar marshalling throughput are on the
[README](https://github.com/osullivryan/dynars#performance). This page covers the
**results readers** — how dynars' `binout` / `d3plot` compare to
[lasso-python](https://github.com/open-lasso-python/lasso-python), the usual
Python reader.

!!! note "Methodology"
    Apple M4 (10 cores), release builds, warm page cache, best-of-N. dynars 1.1,
    lasso-python 2.0.4. Numbers are **machine- and shape-specific** — reproduce
    on your hardware with `examples/bench_vs_lasso.py` (it generates the data, so
    it needs no crash model) and the `examples/*_bench.rs` benches. **Release
    only**: a debug extension reads several× slower — build with
    `maturin develop --release`. Values are cross-checked identical to lasso.

## The one rule that explains every number

A reader's speedup over an eager reader is, roughly:

```text
speedup  ≈  (bytes the eager reader moves / bytes you actually need)  ×  per-byte efficiency
```

dynars is memory-mapped and lazy: `block(...)` is a **zero-copy view** into the
map, and it decodes only the columns/states you ask for. lasso reads the **whole
file** into NumPy arrays when you open it. So the more of the file you *don't*
need, the larger the gap — and when you genuinely need **all** of it as a private
copy, the byte ratio is 1 and only the per-byte term (~3×) is left. That is why a
"copy everything" bulk read is the *smallest* speedup, not the largest.

## binout

74 MB `nodout`, 300 states × 20 000 nodes (`examples/bench_vs_lasso.py`):

| operation | dynars | lasso-python | speedup |
|-----------|-------:|-------------:|:-------:|
| open + read a channel `[T, nodes]` | 5.4 ms | 174 ms | **~30×** |
| read a channel (warm handle) | 5.4 ms | 164 ms | **~30×** |
| one node's history `[T]` (`id=`) | 0.6 ms | 163 ms | **~270×** |

The one-node case is the payoff of the targeted read: dynars decodes just that
column out of each state record (≈ one page per state), while lasso reads the
full `[states, nodes]` matrix and slices a column out of it.

## d3plot

280 MB of solid results, 50 states × 200 000 solids × 7 vars:

| operation | dynars | lasso-python | speedup |
|-----------|-------:|-------------:|:-------:|
| open + read **one state**'s solids | 0.2 ms | 116 ms | **~500×** |
| open + **materialize all** solids (private copy) | 33 ms (8.6 GB/s) | 108 ms (2.6 GB/s) | ~3× |
| peak memory, full materialize | 310 MB | 683 MB | 2.2× less |

- **Selective reads win big.** Want one state, one part, or one field? dynars
  reads ≈ that much off the map; lasso reads the entire file on open — hence
  ~500× for a single state.
- **A full private copy is only ~3×** — that's the memory-bandwidth floor. Both
  physically move 280 MB; dynars does it at ~8.6 GB/s vs lasso's ~2.6 GB/s
  (Python/dtype overhead), and no reader beats memory bandwidth once every byte
  must move. You rarely need this: `block()` already hands back a **zero-copy
  view**, so unless you explicitly `.copy()` the whole model you never pay it.

### Streaming reductions (Rust)

The largest structural win isn't in the table: computing an engineering result
straight off the map without materializing the block. `examples/d3plot_stream_bench.rs`
(2 M solids × 10 states × 7 vars):

```text
von Mises part-max history over 20M element-reads:
  materialize f64 + reduce :  56.7 ms   (353 M elem/s)
  stream off mmap          :  33.1 ms   (605 M elem/s)   1.7× faster, 0 extra bytes
```

For a 30 M-element × 50-state model, materializing to f64 would need ~84 GB;
streaming stays bandwidth-bound at ~0 extra memory. lasso must materialize first,
so this workload is simply out of reach at scale.

## Reproducing

```bash
# Python, dynars vs lasso (skips lasso gracefully if it isn't installed):
.venv/bin/python examples/bench_vs_lasso.py

# Rust reader benches (self-contained):
cargo run --release --example d3plot_stream_bench     # streaming vs materialize
cargo run --release --example hic_bench -- <binout>   # single-dummy HIC
cargo run --release --example orion_read_bench -- <root.k>   # deck parse + validate
```
