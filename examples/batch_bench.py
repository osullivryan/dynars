"""Measure what a `Workspace` saves when many decks share one big `*INCLUDE`.

Python mirror of `examples/batch_bench.rs`. Generates a mesh (`nodes` `*NODE`s +
`elems` `*ELEMENT_SHELL`s over them) and `decks` variant roots that each
`*INCLUDE` it, then times two ways of parsing+validating all of them with the
same rules:
  naive     — `parse_deck` + `validate` per deck (re-reads/re-checks the mesh
              every time), and
  workspace — `parse_decks` + `validate_decks` (mesh read/parsed/indexed once).
It also asserts both produce identical findings — a correctness check at scale.

Build the extension in RELEASE first, or the numbers are meaningless:
    maturin develop --release
    python examples/batch_bench.py [nodes] [elems] [decks]
"""

import os
import sys
import tempfile
import time

import dynars


def generate(nodes: int, elems: int, decks: int) -> list[str]:
    d = tempfile.mkdtemp(prefix="dynars_batch_bench_py_")
    parts = [
        "*KEYWORD\n*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n"
        "*SECTION_SHELL\n1,2\n*PART\np\n1,1,1\n*NODE\n"
    ]
    for i in range(1, nodes + 1):
        parts.append(f"{i},{float(i % 1000):.1f},0.0,0.0\n")
    parts.append("*ELEMENT_SHELL\n")
    span = max(nodes, 4) - 3
    for e in range(1, elems + 1):
        n1 = (e - 1) % span + 1
        parts.append(f"{e},1,{n1},{n1 + 1},{n1 + 2},{n1 + 3}\n")
    mesh = os.path.join(d, "mesh.k")
    with open(mesh, "w") as fh:
        fh.write("".join(parts))
    mb = os.path.getsize(mesh) / 1e6
    print(f"mesh: {nodes} nodes, {elems} shells, {mb:.1f} MB; {decks} decks include it\n")

    roots = []
    for k in range(decks):
        sub = os.path.join(d, f"v{k}")
        os.makedirs(sub)
        root = os.path.join(sub, "main.k")
        with open(root, "w") as fh:
            fh.write(f"*KEYWORD\n*INCLUDE\n../mesh.k\n*PARAMETER\nR run,{k}.0\n*END\n")
        roots.append(root)
    return roots


def keys(report) -> list[str]:
    return sorted(f"{f.file}|{f.line}|{f.message}" for f in report.findings)


def rules():
    return [
        dynars.Rule.references_resolve_with_connectivity(),
        dynars.Rule.duplicate_ids(),
    ]


def main() -> None:
    argv = sys.argv
    nodes = int(argv[1]) if len(argv) > 1 else 500_000
    elems = int(argv[2]) if len(argv) > 2 else 500_000
    decks = int(argv[3]) if len(argv) > 3 else 8
    roots = generate(nodes, elems, decks)

    # Naive: parse_deck + validate per deck, from scratch each time.
    t = time.perf_counter()
    naive = [dynars.parse_deck(r) for r in roots]
    naive_parse = time.perf_counter() - t
    t = time.perf_counter()
    naive_reports = [d.validate(rules()) for d in naive]
    naive_validate = time.perf_counter() - t

    # Workspace: shared parse + parallel validate off one cache.
    ws = dynars.Workspace()
    t = time.perf_counter()
    wdecks = ws.parse_decks(roots)
    ws_parse = time.perf_counter() - t
    t = time.perf_counter()
    ws_reports = ws.validate_decks(wdecks, rules())
    ws_validate = time.perf_counter() - t

    for i, (a, b) in enumerate(zip(naive_reports, ws_reports)):
        assert keys(a) == keys(b), f"deck {i} findings differ"
    print(f"parsed+validated {decks} decks; findings identical ✓\n")

    def row(name, p, v):
        print(f"  {name:<10} parse {p * 1e3:8.1f} ms   validate {v * 1e3:8.1f} ms   total {(p + v) * 1e3:8.1f} ms")

    row("naive", naive_parse, naive_validate)
    row("workspace", ws_parse, ws_validate)
    print(
        f"\n  speedup    parse {naive_parse / ws_parse:.1f}x   "
        f"validate {naive_validate / ws_validate:.1f}x   "
        f"total {(naive_parse + naive_validate) / (ws_parse + ws_validate):.1f}x"
    )
    print(f"\n  workspace cache: {ws.stats()}")


if __name__ == "__main__":
    main()
