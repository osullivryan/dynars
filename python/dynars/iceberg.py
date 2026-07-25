"""Land LS-DYNA binout results into Apache Iceberg tables.

This is an *optional* feature — it pulls in ``pyarrow`` and ``pyiceberg``, which
are deliberately kept out of the fast core. Install with::

    pip install "dynars[iceberg]"

The model is run-oriented: every ingest tags rows with a ``run_id`` so many
simulations accumulate into the same tables and stay queryable together (the
whole point of using Iceberg here). One Iceberg table per binout *branch*
(``rcforc``, ``glstat``, ``matsum`` …), each in long/tidy form::

    run_id | time | id | <var1> | <var2> | ...

``id`` is the per-entity index within a state (interface / part / rigid body);
for scalar branches like ``glstat`` there is a single entity so ``id`` is 0.

PyIceberg assigns stable Iceberg field-ids when it creates the table from the
Arrow schema, so the resulting Parquet data files are proper Iceberg files
(no manual ``PARQUET:field_id`` handling needed on this path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import parse_binout

if TYPE_CHECKING:  # avoid importing the heavy optional deps at module import
    import pyarrow as pa
    from pyiceberg.catalog import Catalog


def _require(mod: str):
    try:
        return __import__(mod)
    except ModuleNotFoundError as e:  # pragma: no cover - trivial
        raise ModuleNotFoundError(
            f"dynars.iceberg needs '{mod}'. Install with: pip install \"dynars[iceberg]\""
        ) from e


# ── binout → Arrow ──────────────────────────────────────────────────────────

def _stack_states(b, branch: str, states: list[str], var: str):
    """Stack one variable across all ``dNNNNNN`` state dirs into (T, k).

    Solver binouts store each output state as a subdirectory; ``read_many``
    reads them all in parallel. Returns ``None`` if the per-state widths are
    ragged (can't form a rectangular column).
    """
    parts = b.read_many([[branch, s, var] for s in states])
    arrs = [np.asarray(p, dtype=np.float64).ravel() for p in parts]
    widths = {a.size for a in arrs}
    if len(widths) != 1:
        return None
    return np.vstack(arrs) if arrs else None


def _branch_table(b, branch: str, run_id: str):
    """Build one long-form pyarrow Table for a single binout branch.

    Returns ``None`` if the branch has no usable time-varying variables.
    """
    pa = _require("pyarrow")

    kids = b.channels([branch])
    states = sorted(k for k in kids if k.startswith("d") and k[1:].isdigit())
    if not states:
        return None
    var_names = [v for v in b.channels([branch, states[0]]) if v != "time"]

    # Stack each variable across the state dirs -> (T, k).
    T = len(states)
    series: dict[str, np.ndarray] = {}
    for v in var_names:
        a = _stack_states(b, branch, states, v)
        if a is not None and a.shape[0] == T and a.shape[1] > 0:
            series[v] = a
    if not series:
        return None

    # Shared time vector: one scalar per state.
    time_stack = _stack_states(b, branch, states, "time")
    time_vec = time_stack[:, 0] if time_stack is not None else np.arange(T, dtype=float)

    # Entity count = the dominant non-scalar width; scalars (k==1) broadcast.
    widths = [a.shape[1] for a in series.values()]
    n_ent = max(widths)

    cols: dict[str, "pa.Array"] = {
        "run_id": pa.array([run_id] * (T * n_ent), type=pa.large_string()),
        "time": pa.array(np.repeat(time_vec, n_ent)),
        "id": pa.array(np.tile(np.arange(n_ent, dtype=np.int64), T)),
    }
    for v, a in series.items():
        if a.shape[1] == n_ent:
            col = a.reshape(-1)
        elif a.shape[1] == 1:
            col = np.repeat(a[:, 0], n_ent)
        else:
            # Ragged within the branch (rare) — skip rather than misalign.
            continue
        cols[_safe(v)] = pa.array(np.ascontiguousarray(col, dtype=np.float64))
    return pa.table(cols)


def _safe(name: str) -> str:
    """Column-name-safe: iceberg/SQL dislike spaces and '+'."""
    return name.replace(" ", "_").replace("+", "_plus_")


def binout_to_arrow(binout_path: str, run_id: str) -> dict:
    """Convert a binout into ``{branch: pyarrow.Table}`` in long/tidy form."""
    b = parse_binout(binout_path)
    out = {}
    for branch in b.channels():
        tbl = _branch_table(b, branch, run_id)
        if tbl is not None:
            out[branch] = tbl
    return out


# ── Iceberg catalog + ingest ────────────────────────────────────────────────

def local_catalog(warehouse: str, name: str = "dynars") -> "Catalog":
    """A local, dependency-free SQLite-backed Iceberg catalog for dev/testing."""
    _require("pyiceberg")
    import os
    from pyiceberg.catalog.sql import SqlCatalog

    os.makedirs(warehouse, exist_ok=True)
    return SqlCatalog(
        name,
        uri=f"sqlite:///{warehouse}/catalog.db",
        warehouse=f"file://{warehouse}",
    )


def ingest_binout(
    binout_path: str,
    run_id: str,
    catalog: "Catalog",
    namespace: str = "binout",
) -> list[str]:
    """Ingest one binout as a run: create-if-absent + append per branch table.

    Returns the list of fully-qualified table identifiers written.
    """
    catalog.create_namespace_if_not_exists(namespace)
    written = []
    for branch, tbl in binout_to_arrow(binout_path, run_id).items():
        ident = f"{namespace}.{branch}"
        try:
            table = catalog.load_table(ident)
        except Exception:
            table = catalog.create_table(ident, schema=tbl.schema)
        table.append(tbl)
        written.append(ident)
    return written
