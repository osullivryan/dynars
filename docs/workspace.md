# Workspace (batch)

Checking many decks — load-case or run variants of one model — usually means they
all `*INCLUDE` the same big files (mesh, materials, sections). A **`Workspace`**
reads, parses, and indexes each shared file **once** across the whole batch, then
validates the decks in parallel. The decks it hands back are ordinary `Deck`s —
navigate or validate them individually too; either way they reuse the shared
cache.

## Parse and validate a batch

=== "Python"

    ```python
    import dynars

    ws = dynars.Workspace()
    decks = ws.parse_decks([
        "variant_a/main.k",
        "variant_b/main.k",
        "variant_c/main.k",
    ])

    reports = ws.validate_decks(decks, [
        dynars.Rule.references_resolve_with_connectivity(),
        dynars.Rule.duplicate_ids(),
        dynars.Rule.include_missing(),
    ])
    for report in reports:
        print(report.is_clean(), report.count(dynars.Severity.Error))

    print(ws.stats())
    # {'files_parsed': 4, 'files_reused': 2, 'def_indices_built': 4, 'ref_indices_built': 4}
    ```

=== "Rust"

    ```rust
    use dynars::Workspace;
    use dynars::validate::Rule;

    let ws = Workspace::new();
    let decks: Vec<_> = ws
        .parse_decks(["variant_a/main.k", "variant_b/main.k", "variant_c/main.k"])
        .into_iter()
        .filter_map(|(_root, d)| d.ok())   // keep the decks that parsed
        .collect();

    let reports = ws.validate_decks(&decks, [
        Rule::references_resolve_with_connectivity(),
        Rule::duplicate_ids(),
        Rule::include_missing(),
    ]);
    println!("{} decks, cache {:?}", reports.len(), ws.stats());
    ```

`stats()` reports what the sharing bought: `files_parsed` vs `files_reused` (disk
reads avoided) and `def_indices_built` / `ref_indices_built` (a shared mesh's id
and connectivity indices are built **once**, not once per deck).

## The mental model

A `Workspace` is a **cache**, not a collection of decks — created empty, with work
submitted against it (like a connection pool). It holds neither the roots nor the
decks, so you can keep parsing more decks against a warm cache over time:

=== "Python"

    ```python
    ws = dynars.Workspace()
    baseline = ws.parse_decks(glob("baseline/*/main.k"))   # reads + indexes the mesh
    # ...later, after generating new variants that *INCLUDE the same mesh...
    sweep = ws.parse_decks(glob("sweep_2/*/main.k"))       # mesh already cached
    ```

=== "Rust"

    ```rust
    let ws = Workspace::new();
    let baseline = ws.parse_decks(baseline_roots);   // reads + indexes the mesh
    // ...later...
    let sweep = ws.parse_decks(sweep_roots);         // mesh already cached
    ```

## Missing includes

A missing `*INCLUDE` is **never parsed** — it adds no file and no cached content
(`files_parsed` doesn't count it), so nothing phantom leaks into the deck. It *is*
recorded as a directive, so add **`include_missing()`** to catch it. Don't rely on
`references_resolve` alone: if the missing file was the *only* source of an entity
kind, references to that kind are left unflagged (the dangling check is
conservative by design).

## What it costs — and saves

The shared work is paid once and amortizes over the batch, so the workspace total
stays roughly flat as decks are added, while the naive per-deck approach grows
linearly. Over a 28 MB shared mesh (500k nodes / 500k shells), naive `parse_deck`
+ `validate` per deck vs. `Workspace`:

| decks | naive total | workspace total | speedup |
|------:|------------:|----------------:|--------:|
| 4  | 240 ms  | 122 ms | 2.0× |
| 8  | 481 ms  | 126 ms | 3.8× |
| 16 | 969 ms  | 150 ms | 6.5× |
| 32 | 1944 ms | 157 ms | 12.4× |

Reproduce with `examples/batch_bench.rs` (Rust) or `examples/batch_bench.py`
(Python — build the extension `--release` first). Runnable walk-throughs live in
`examples/batch_validate.rs` and `examples/batch_demo.py`.
