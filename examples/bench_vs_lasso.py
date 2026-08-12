"""Results-reader benchmark: dynars vs lasso-python, on synthetic binout + d3plot.

    .venv/bin/python examples/bench_vs_lasso.py

Self-contained — it *generates* the data (so it's reproducible anywhere) and
skips the lasso column if lasso-python isn't installed. Numbers are
machine-specific; run it on your hardware. Release build only
(`maturin develop --release`) — a debug extension reads ~5x slower.

The takeaway the numbers teach: a reader's speedup over an eager reader is
roughly `(bytes the eager reader moves / bytes you actually need) x per-byte
efficiency`. A full private copy of every byte is memory-bandwidth-bound, so the
gap is only the ~3x efficiency term; selective / zero-copy / streaming reads move
a fraction of the bytes, so the gap explodes.
"""

import os
import tempfile
import time

import numpy as np

import dynars

try:
    from lasso.dyna import ArrayType
    from lasso.dyna import Binout as LBinout
    from lasso.dyna import D3plot as LD3plot

    HAVE_LASSO = True
except Exception:
    HAVE_LASSO = False


def best(fn, n=6):
    t = float("inf")
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def row(label, td, tl, unit="ms", scale=1e3):
    d = f"{td * scale:8.2f} {unit}"
    if tl is None:
        print(f"  {label:34s} dynars {d}   lasso    n/a")
    else:
        print(f"  {label:34s} dynars {d}   lasso {tl * scale:8.1f} {unit}   ({tl / td:5.1f}x)")


def bench_binout(tmp):
    n_states, n_nodes = 300, 20_000
    ids = np.arange(1, n_nodes + 1)
    ch = {
        c: (np.random.default_rng(0).standard_normal((n_states, n_nodes)) * 100).astype(np.float32)
        for c in ("x_acceleration", "y_acceleration", "z_acceleration")
    }
    w = dynars.build_series("nodout", ids=ids, channels=ch, times=np.linspace(0, 0.15, n_states))
    path = os.path.join(tmp, "binout")
    w.write(path)
    mb = os.path.getsize(path) / 1e6
    print(f"\nbinout — {mb:.0f} MB, {n_states} states x {n_nodes} nodes")

    mid = int(ids[len(ids) // 2])
    db = dynars.parse_binout(path)
    lb = LBinout(path) if HAVE_LASSO else None
    col = int(np.nonzero(db.ids("nodout") == mid)[0][0])

    row("open + read a channel [T,nodes]",
        best(lambda: dynars.parse_binout(path).read("nodout", "x_acceleration")),
        best(lambda: LBinout(path).read("nodout", "x_acceleration")) if HAVE_LASSO else None)
    row("read a channel (warm handle)",
        best(lambda: db.read("nodout", "x_acceleration")),
        best(lambda: lb.read("nodout", "x_acceleration")) if HAVE_LASSO else None)
    row("one node's history [T] (id=)",
        best(lambda: db.read("nodout", "x_acceleration", id=mid)),
        best(lambda: lb.read("nodout", "x_acceleration")[:, col]) if HAVE_LASSO else None)


def bench_d3plot(tmp):
    n_elem, n_states, nv = 200_000, 50, 7
    nodes = np.arange(8 * 3, dtype=np.float64)
    w = dynars.D3plotWriter(nodes)
    w.add_solids(np.tile(np.arange(1, 9), (n_elem, 1)).astype(np.int64), np.full(n_elem, 1))
    w.set_ids(np.array([7]))
    res = (np.arange(n_states * n_elem * nv) % 997 / 7.0).astype(np.float32)
    w.set_solid_results(res.reshape(n_states, n_elem, nv))   # (n_states, n_solids, vars)
    for s in range(n_states):
        w.add_state(float(s), nodes + s, None, None)
    path = os.path.join(tmp, "d3plot")
    w.write(path)
    mb = n_elem * n_states * nv * 4 / 1e6
    print(f"\nd3plot — {mb:.0f} MB solid results, {n_states} states x {n_elem} solids x {nv} vars")

    from dynars import StateBlock

    ld = LD3plot(path) if HAVE_LASSO else None  # noqa: F841 (warm the class)
    row("open + read ONE state's solids",
        best(lambda: np.asarray(dynars.open_d3plot(path).block(StateBlock.Solid, 0))),
        best(lambda: np.asarray(LD3plot(path).arrays[ArrayType.element_solid_stress])[0])
        if HAVE_LASSO else None)

    def dyn_full():
        d = dynars.open_d3plot(path)
        ns = d.num_states
        out = np.empty((ns, n_elem, nv), np.float32)
        for s in range(ns):
            out[s] = np.asarray(d.block(StateBlock.Solid, s)).reshape(n_elem, nv)
        return out

    row("open + materialize ALL solids",
        best(dyn_full),
        best(lambda: np.asarray(LD3plot(path).arrays[ArrayType.element_solid_stress]))
        if HAVE_LASSO else None)
    print("  note: dynars block() is a zero-copy mmap view; the 'materialize' row forces a full")
    print("        private copy (memory-bandwidth-bound). lasso always materializes on open.")


def main():
    print(f"dynars {getattr(dynars, '__version__', '?')} vs lasso "
          f"({'installed' if HAVE_LASSO else 'NOT installed — dynars only'})")
    with tempfile.TemporaryDirectory() as tmp:
        bench_binout(tmp)
        bench_d3plot(tmp)


if __name__ == "__main__":
    main()
