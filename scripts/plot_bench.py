#!/usr/bin/env python3
"""Render the README scaling figures from the benchmark CSV.

The Rust benchmark (`cargo run --release --example bench_scaling`) does the
*measurement* — it drives the actual library and writes `assets/bench_scaling.csv`.
This script does the *plotting*: one figure per pipeline operation, x = number of
keyword blocks (log), y = wall-clock time in ms (log), a line per include-tree
shape (monolithic / flat / deep-tree). Every figure shares one y-axis range so
operations are directly comparable across plots. The CSV is committed, so figures
re-render in a second without re-running the sweep:

    python scripts/plot_bench.py
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "assets", "bench_scaling.csv")

# (csv column, title, filename stem) — all scale with the number of keywords in
# the deck and share one set of axes. Titles are in LS-DYNA analyst terms, not
# internal jargon. (*NODE extraction scales with node count and runs to 250 M —
# it gets its own figure, `plot_marshal`.)
OPS = [
    ("include_s", "Locating *INCLUDE files", "perf_include"),
    ("parse_s", "Reading the keyword deck", "perf_parse"),
    ("index_s", "Cataloguing all entity IDs", "perf_index"),
    ("dangle_s", "Checking every ID reference exists", "perf_dangle"),
    ("conn_s", "ID references + element connectivity", "perf_connectivity"),
    ("field_s", "Checking a field value against a rule", "perf_field"),
]


def human(v):
    """1200000 → '1M', 30000 → '30K'."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            s = v / div
            return f"{s:.0f}{suf}" if s == int(s) else f"{s:g}{suf}"
    return f"{v:g}"


def fmt_ms(v):
    return f"{v/1000:g}s" if v >= 1000 else f"{v:g}ms"


# clean, sparse major ticks (1-3 per decade); minor ticks off
XTICKS = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
YTICKS = [1, 10, 100, 1000]

# plot-line order + human labels for the include-tree shapes
SHAPE_ORDER = ["monolithic", "flat", "deep-tree"]

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def load():
    by_shape = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_shape[r["shape"]].append(r)
    for rows in by_shape.values():
        rows.sort(key=lambda r: int(r["blocks"]))
    return by_shape


def label_for(shape, rows):
    files = int(rows[0]["files"])
    pretty = {"monolithic": "monolithic", "flat": "flat (wide)", "deep-tree": "deep tree"}.get(shape, shape)
    return f"{pretty} — {files} file" + ("s" if files != 1 else "")


def ms(row, col):
    return max(float(row[col]) * 1e3, 1e-3)  # seconds → ms, floored for log axis


def global_limits(by_shape):
    """Shared axis ranges: x = keyword-block count, y = every stage's time — so
    all six panels are directly comparable."""
    xs, ys = [], []
    for rows in by_shape.values():
        for r in rows:
            xs.append(int(r["blocks"]))
            for col, *_ in OPS:
                ys.append(ms(r, col))
    return (min(xs) * 0.75, max(xs) * 1.4), (min(ys) * 0.6, max(ys) * 1.7)


def style_axis(ax, xlim, ylim):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)  # shared across every panel — comparable at a glance
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: human(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt_ms(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("keywords in deck")
    ax.set_ylabel("time")
    ax.grid(True, which="major", ls=":", alpha=0.45)


def plot_op(ax, by_shape, col, title, xlim, ylim):
    for shape in SHAPE_ORDER:
        rows = by_shape.get(shape)
        if not rows:
            continue
        xs = [int(r["blocks"]) for r in rows]
        ys = [ms(r, col) for r in rows]
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.7, label=label_for(shape, rows))
    style_axis(ax, xlim, ylim)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, title="include tree")


def plot_marshal():
    """Standalone: *NODE extraction up to 250 M nodes (log-log, own axes — it runs
    from ms into seconds). Two include layouts; a dashed 1-second reference line."""
    path = os.path.join(ROOT, "assets", "bench_marshal.csv")
    by_shape = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_shape[r["shape"]].append(r)
    for rows in by_shape.values():
        rows.sort(key=lambda r: int(r["nodes"]))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for shape in ("monolithic", "flat"):
        rows = by_shape.get(shape)
        if not rows:
            continue
        xs = [int(r["nodes"]) for r in rows]
        ys = [max(float(r["marshal_s"]) * 1e3, 1e-3) for r in rows]
        files = int(rows[0]["files"])
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0,
                label=f"{shape} — {files} file" + ("s" if files != 1 else ""))
    ax.axhline(1000, color="#888", ls="--", lw=1.0)
    ax.text(0.015, 1000 * 1.08, "1 second", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#666", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1e6, 1e7, 1e8]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: human(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt_ms(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("nodes in deck")
    ax.set_ylabel("time")
    ax.set_title("Extracting *NODE data into arrays (columnar → typed)")
    ax.grid(True, which="major", ls=":", alpha=0.45)
    ax.legend(fontsize=9, title="include layout")
    fig.tight_layout()
    out = os.path.join(ROOT, "assets", "perf_marshal.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(out, ROOT))


def plot_pydyna():
    """dynars vs pyDYNA (assets/bench_pydyna.csv from
    examples/compare_pydyna.py). Four panels — read *NODE, read connectivity,
    resolve an include tree, author a *NODE deck — log-log, shared axes, with the
    top speedup annotated. Skipped if the CSV isn't present (needs ansys-dyna-core)."""
    path = os.path.join(ROOT, "assets", "bench_pydyna.csv")
    if not os.path.exists(path):
        return
    by_task = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_task[r["task"]].append(r)
    for rows in by_task.values():
        rows.sort(key=lambda r: int(r["size"]))

    tasks = [
        ("nodes", "Read *NODE → coordinates"),
        ("connectivity", "Read *ELEMENT_SHELL → connectivity"),
        ("include-tree", "Resolve root + 8 *INCLUDE → nodes"),
        ("author", "Author *NODE deck → write .k"),
    ]
    xs_all, ys_all = [], []
    for rows in by_task.values():
        for r in rows:
            xs_all.append(int(r["size"]))
            ys_all.append(max(float(r["dynars_s"]) * 1e3, 1e-3))
            if r["pydyna_s"]:
                ys_all.append(float(r["pydyna_s"]) * 1e3)
    xlim = (min(xs_all) * 0.7, max(xs_all) * 1.5)
    ylim = (min(ys_all) * 0.55, max(ys_all) * 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.flat, tasks):
        rows = by_task.get(key, [])
        xs = [int(r["size"]) for r in rows]
        ax.plot(xs, [max(float(r["dynars_s"]) * 1e3, 1e-3) for r in rows],
                marker="o", markersize=4.5, linewidth=2.0, color="#1f77b4", label="dynars")
        px = [int(r["size"]) for r in rows if r["pydyna_s"]]
        py = [float(r["pydyna_s"]) * 1e3 for r in rows if r["pydyna_s"]]
        ax.plot(px, py, marker="s", markersize=4.5, linewidth=2.0, color="#d62728", label="pyDYNA")
        if rows and rows[-1]["speedup"]:
            ax.text(0.96, 0.06, f"up to {float(rows[-1]['speedup']):.0f}×",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=12, fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#e0c080"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(FixedLocator([1e4, 1e5, 1e6, 1e7]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: human(v)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt_ms(v)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("entities")
        ax.set_ylabel("time")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="major", ls=":", alpha=0.45)
        ax.legend(fontsize=9)
    fig.suptitle("dynars vs pyDYNA — reading & authoring LS-DYNA keyword data (log-log)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(ROOT, "assets", "perf_pydyna.png")
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print("wrote", os.path.relpath(out, ROOT))


def main():
    by_shape = load()
    xlim, ylim = global_limits(by_shape)

    for col, title, stem in OPS:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        plot_op(ax, by_shape, col, title, xlim, ylim)
        fig.tight_layout()
        out = os.path.join(ROOT, "assets", f"{stem}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", os.path.relpath(out, ROOT))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, _stem) in zip(axes.flat, OPS):
        plot_op(ax, by_shape, col, title, xlim, ylim)
    fig.suptitle(
        "dynars — time per stage vs deck size (shared axes; one line per include layout)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(ROOT, "assets", "perf_overview.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", os.path.relpath(out, ROOT))

    plot_marshal()
    plot_pydyna()


if __name__ == "__main__":
    main()
