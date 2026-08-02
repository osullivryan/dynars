"""Head-to-head: dynars vs pyDYNA (ansys-dyna-core) on *reading* LS-DYNA
keyword data — parse a deck and pull the data into arrays.

pyDYNA is primarily a deck-*authoring* API (pure Python + pandas); this measures
the one overlapping path, reading. Three tasks, identical inputs:

  1. *NODE       -> (N, 3) coordinate array
  2. *ELEMENT_SHELL -> (M, 4) connectivity array
  3. include tree -> all node coordinates across a root + K *INCLUDE files
     (dynars follows *INCLUDE natively; pyDYNA does not, so we hand-roll the
      recursion it would otherwise need — see pyd_nodes_including)

Usage:
    pip install ansys-dyna-core
    python examples/compare_pydyna.py

Numbers are machine-specific. Writes assets/bench_pydyna.csv.
"""

import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd

import dynars
from ansys.dyna.core import Deck
from ansys.dyna.core.keywords import keywords as _kw

PydNode = _kw.Node
PYDYNA_CUTOFF_S = 60.0   # once a pyDYNA task exceeds this, skip it at larger sizes


# ---------- deck generators (standard fixed-width cards both parsers accept) ----------
def write_nodes(path, n):
    rng = np.random.default_rng(0)
    xyz = rng.random((n, 3))
    with open(path, "w") as f:
        f.write("*KEYWORD\n*NODE\n")
        f.writelines(
            f"{i + 1:8d}{xyz[i, 0]:16.6f}{xyz[i, 1]:16.6f}{xyz[i, 2]:16.6f}\n"
            for i in range(n)
        )
        f.write("*END\n")


def write_shells(path, m):
    with open(path, "w") as f:
        f.write("*KEYWORD\n*ELEMENT_SHELL\n")
        f.writelines(
            f"{i + 1:8d}{1:8d}{i % 997 + 1:8d}{i % 997 + 2:8d}{i % 997 + 3:8d}{i % 997 + 4:8d}\n"
            for i in range(m)
        )
        f.write("*END\n")


def write_include_deck(dirpath, n, k):
    per = n // k
    root = os.path.join(dirpath, "root.k")
    with open(root, "w") as f:
        f.write("*KEYWORD\n")
        for j in range(k):
            f.write(f"*INCLUDE\ninc_{j}.k\n")
        f.write("*END\n")
    for j in range(k):
        write_nodes(os.path.join(dirpath, f"inc_{j}.k"), per)
    return root


# ---------- timing ----------
def timed(fn, repeat):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


# ---------- dynars readers (parse_deck is include-aware; one call for all three tasks) ----------
def dyn_nodes(path):
    c = dynars.parse_deck(path).table("NODE")
    return np.column_stack([c["x"], c["y"], c["z"]])


def dyn_shells(path):
    t = dynars.parse_deck(path).table("ELEMENT_SHELL")
    return np.column_stack([t["N1"], t["N2"], t["N3"], t["N4"]])


# ---------- pyDYNA readers ----------
def pyd_nodes(path):
    deck = Deck()
    deck.import_file(path)
    arrs = [k.nodes[["x", "y", "z"]].to_numpy()
            for k in deck.keywords if type(k).__name__ == "Node"]
    return np.vstack(arrs) if arrs else np.empty((0, 3))


def pyd_shells(path):
    deck = Deck()
    deck.import_file(path)
    arrs = [k.elements[["n1", "n2", "n3", "n4"]].to_numpy()
            for k in deck.keywords if type(k).__name__ == "ElementShell"]
    return np.vstack(arrs) if arrs else np.empty((0, 4))


def pyd_nodes_including(root):
    """pyDYNA does not follow *INCLUDE — walk it ourselves, the way a user must."""
    arrs, stack = [], [root]
    while stack:
        p = stack.pop()
        base = os.path.dirname(p)
        deck = Deck()
        deck.import_file(p)
        for k in deck.keywords:
            tn = type(k).__name__
            if tn == "Node":
                arrs.append(k.nodes[["x", "y", "z"]].to_numpy())
            elif tn == "Include":
                stack.append(os.path.join(base, k.filename))
    return np.vstack(arrs) if arrs else np.empty((0, 3))


# ---------- authoring: build a *NODE deck from coordinate arrays, write a .k ----------
def dyn_author(coords, outpath):
    """dynars authoring: the columnar writer takes numpy arrays straight into
    Rust (no per-row Python objects) — the inverse of the columnar read path."""
    n = len(coords)
    dynars.write_keyword(outpath, "NODE", {
        "nid": np.arange(1, n + 1, dtype=np.int64),
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
    })
    return coords


def pyd_author(coords, outpath):
    """pyDYNA authoring: its home turf — a Node keyword from a DataFrame, exported."""
    n = len(coords)
    node = PydNode()
    node.nodes = pd.DataFrame({"nid": np.arange(1, n + 1),
                               "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]})
    deck = Deck()
    deck.append(node)
    deck.export_file(outpath)
    return coords


def bench_task(name, size, gen, dyn_fn, pyd_fn, pyd_enabled, rows, dyn_repeat=3):
    """Time dynars (best of `dyn_repeat`) and pyDYNA (1 run) on one task at one size."""
    dyn_out, dyn_t = timed(dyn_fn, repeat=dyn_repeat)
    n_out = len(dyn_out)
    pyd_t = None
    if pyd_enabled:
        pyd_out, pyd_t = timed(pyd_fn, repeat=1)
        assert len(pyd_out) == n_out, f"{name}: row mismatch {len(pyd_out)} vs {n_out}"
    speed = (pyd_t / dyn_t) if pyd_t else None
    rows.append((name, size, dyn_t, pyd_t, speed))
    dcol = f"{dyn_t * 1e3:8.1f} ms"
    pcol = f"{pyd_t * 1e3:9.1f} ms" if pyd_t else "  (skipped)"
    scol = f"{speed:6.0f}x" if speed else "     -"
    print(f"  {name:14} {size:>10,}  dynars {dcol}   pyDYNA {pcol}   speedup {scol}")
    return pyd_t is not None and pyd_t <= PYDYNA_CUTOFF_S


def main():
    sizes = [10_000, 100_000, 1_000_000, 5_000_000]
    tmp = tempfile.mkdtemp(prefix="cmp_pydyna_")
    rows = []
    pyd_ok = {"nodes": True, "shells": True, "include": True, "author": True}

    print("== Task 1: parse *NODE -> (N,3) array ==")
    for n in sizes:
        p = os.path.join(tmp, f"nodes_{n}.k")
        write_nodes(p, n)
        pyd_ok["nodes"] = bench_task("nodes", n, p, lambda: dyn_nodes(p),
                                     lambda: pyd_nodes(p), pyd_ok["nodes"], rows)

    print("\n== Task 2: parse *ELEMENT_SHELL -> (M,4) connectivity ==")
    for n in sizes:
        p = os.path.join(tmp, f"shells_{n}.k")
        write_shells(p, n)
        pyd_ok["shells"] = bench_task("connectivity", n, p, lambda: dyn_shells(p),
                                      lambda: pyd_shells(p), pyd_ok["shells"], rows)

    print("\n== Task 3: resolve a root + 8 *INCLUDE files -> all node coords ==")
    for n in sizes:
        d = tempfile.mkdtemp(prefix=f"inc_{n}_", dir=tmp)
        root = write_include_deck(d, n, k=8)
        pyd_ok["include"] = bench_task("include-tree", n, root, lambda: dyn_nodes(root),
                                       lambda: pyd_nodes_including(root),
                                       pyd_ok["include"], rows)

    print("\n== Task 4: author a *NODE deck from arrays -> write .k (pyDYNA's home turf) ==")
    arng = np.random.default_rng(1)
    for n in [10_000, 100_000, 1_000_000, 5_000_000]:
        coords = arng.random((n, 3))
        outp = os.path.join(tmp, f"authored_{n}")
        pyd_ok["author"] = bench_task("author", n, None,
                                      lambda: dyn_author(coords, outp + ".dyn.k"),
                                      lambda: pyd_author(coords, outp + ".pyd.k"),
                                      pyd_ok["author"], rows, dyn_repeat=3)

    out = os.path.join(os.path.dirname(__file__), "..", "assets", "bench_pydyna.csv")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        f.write("task,size,dynars_s,pydyna_s,speedup\n")
        for name, size, dt, pt, sp in rows:
            f.write(f"{name},{size},{dt:.6f},{'' if pt is None else f'{pt:.6f}'},"
                    f"{'' if sp is None else f'{sp:.1f}'}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
