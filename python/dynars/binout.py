"""Helpers for constructing binout files that follow LS-DYNA's conventions.

The Rust :class:`~dynars.BinoutEditor` is the low-level engine: it writes an
arbitrary tree of typed datasets. This module adds the *time-series convention*
LS-DYNA actually uses so that files you build read back as proper time-histories
(in dynars, and — validate on your decks — in lasso / LS-PrePost):

    <branch>/metadata/{ids, legend, title, date, revision, version}
    <branch>/d000001/{time, cycle, <channel> ...}
    <branch>/d000002/{...}
    ...

Each state dir ``dNNNNNN`` (1-based, 6 digits) holds a scalar ``time`` (float64)
and ``cycle`` (int32) plus one array per channel, length == number of ``ids``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from dynars._dynars import BinoutEditor

__all__ = ["build_series"]


def _fixed_int8(text: str, width: int) -> np.ndarray:
    """`text` as exactly `width` ASCII bytes (space-padded / truncated), int8."""
    raw = text.encode("ascii", "replace")[:width]
    raw = raw + b" " * (width - len(raw))
    return np.frombuffer(raw, dtype=np.int8).copy()


def _legend_int8(labels: Sequence[str] | None, n: int, width: int = 80) -> np.ndarray:
    """`n` labels of `width` chars each, concatenated — the binout legend block."""
    out = bytearray()
    for i in range(n):
        text = (labels[i] if labels is not None and i < len(labels) else "")
        raw = text.encode("ascii", "replace")[:width]
        out += raw + b" " * (width - len(raw))
    return np.frombuffer(bytes(out), dtype=np.int8).copy()


def build_series(
    branch: str,
    ids: Sequence[int] | np.ndarray,
    channels: Mapping[str, np.ndarray],
    *,
    times: Sequence[float] | np.ndarray | None = None,
    cycles: Sequence[int] | np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    editor: BinoutEditor | None = None,
) -> BinoutEditor:
    """Build a binout time-series branch and return the :class:`BinoutEditor`.

    Parameters
    ----------
    branch:
        Top-level group name, e.g. ``"nodout"``, ``"elout"``, ``"rcforc"``.
    ids:
        Entity ids (nodes/elements/…), 1-D, length ``nent``.
    channels:
        Mapping of channel name -> array. A 2-D array is ``[nstate, nent]`` (one
        row per state); a 1-D array is treated as a per-state scalar ``[nstate]``.
        Numeric arrays keep their dtype; pass float32 to match LS-DYNA output.
    times, cycles:
        Optional per-state ``time`` (float64) and ``cycle`` (int32), length
        ``nstate``. ``time`` is strongly recommended — post-processors key on it.
    labels:
        Optional per-entity text labels (written as the 80-char ``legend`` block).
    title:
        Optional run title (80-char metadata field).
    editor:
        Add the branch to an existing editor instead of a fresh one (so several
        branches can share one file).

    Returns
    -------
    BinoutEditor
        Ready to ``.write(path)``.
    """
    e = editor if editor is not None else BinoutEditor()
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError("ids must be 1-D")
    nent = int(ids.shape[0])

    # Infer the number of states from the first 2-D channel, else from times.
    nstate = None
    for arr in channels.values():
        a = np.asarray(arr)
        if a.ndim == 2:
            nstate = int(a.shape[0])
            break
    if nstate is None:
        if times is not None:
            nstate = int(np.asarray(times).shape[0])
        elif cycles is not None:
            nstate = int(np.asarray(cycles).shape[0])
        else:
            raise ValueError("cannot infer number of states: pass a 2-D channel, times, or cycles")

    # Validate shapes up front so we fail before writing anything.
    for name, arr in channels.items():
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape != (nstate, nent):
            raise ValueError(f"channel {name!r} has shape {a.shape}, expected {(nstate, nent)}")
        if a.ndim == 1 and a.shape[0] != nstate:
            raise ValueError(f"scalar channel {name!r} has length {a.shape[0]}, expected {nstate}")
        if a.ndim not in (1, 2):
            raise ValueError(f"channel {name!r} must be 1-D (scalar/state) or 2-D (state x entity)")

    # metadata
    e.set([branch, "metadata", "ids"], ids.astype(np.int64))
    e.set([branch, "metadata", "legend"], _legend_int8(labels, nent))
    if title is not None:
        e.set([branch, "metadata", "title"], _fixed_int8(title, 80))

    # per-state dirs
    times = None if times is None else np.asarray(times, dtype=np.float64)
    cycles = None if cycles is None else np.asarray(cycles, dtype=np.int32)
    for s in range(nstate):
        d = f"d{s + 1:06d}"
        if times is not None:
            e.set([branch, d, "time"], np.array([times[s]], dtype=np.float64))
        if cycles is not None:
            e.set([branch, d, "cycle"], np.array([cycles[s]], dtype=np.int32))
        for name, arr in channels.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                e.set([branch, d, name], np.ascontiguousarray(a[s]))
            else:
                e.set([branch, d, name], np.asarray(a[s]).reshape(1))
    return e
