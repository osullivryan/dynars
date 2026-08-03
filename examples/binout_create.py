"""Construct a binout from scratch, two ways, and read it back.

    python examples/binout_create.py

1. Low-level: dynars.BinoutEditor writes an arbitrary tree of typed datasets.
2. Convention: dynars.build_series lays out a proper time-series branch
   (metadata + dNNNNNN states) that reads back as a time-history.
"""

import os
import tempfile

import numpy as np

import dynars

path = os.path.join(tempfile.gettempdir(), "dynars_created_binout")

# --- 1. Arbitrary low-level construction -------------------------------------
e = dynars.BinoutEditor()
e.set(["glstat", "metadata", "title"], "made by dynars")     # str -> text dataset
e.set(["glstat", "d000001", "time"], np.array([0.0], np.float64))
e.set(["glstat", "d000001", "internal_energy"], np.array([1.5], np.float32))
e.set(["userdata", "matrix"], np.arange(12, dtype=np.int32).reshape(3, 4))  # N-D -> flattened

# --- 2. A real time-series branch from 2-D arrays ----------------------------
nstate, nnodes = 5, 3
ids = np.array([101, 102, 103], dtype=np.int64)
t = np.linspace(0.0, 1.0, nstate)
x_disp = np.outer(t, np.array([1.0, 2.0, 3.0])).astype(np.float32)   # [nstate, nnodes]
x_vel = np.gradient(x_disp, axis=0).astype(np.float32)

dynars.build_series(
    "nodout",
    ids=ids,
    channels={"x_displacement": x_disp, "x_velocity": x_vel},
    times=t,
    cycles=np.arange(1, nstate + 1),
    labels=[f"node {i}" for i in ids],
    title="dynars synthetic nodout",
    editor=e,          # add into the same file
)

e.write(path)
print(f"wrote {os.path.getsize(path)} bytes -> {path}\n")

# --- Read it back -------------------------------------------------------------
b = dynars.parse_binout(path)
print("top-level:", b.read())
print("userdata/matrix (flattened):", b.read(["userdata", "matrix"]))
print("nodout variables:", b.read("nodout"))

# Aggregate the [nstate, nnodes] block straight from the reader.
got = b.read("nodout", "x_displacement")           # [nstate, nnodes]
print("nodout/x_displacement reassembled shape:", got.shape)
assert np.allclose(got, x_disp), "round-trip mismatch"
times_back = b.read("nodout", "time")              # [nstate]
assert np.allclose(times_back, t)
print("time:", times_back)
print("\nOK — round-trips through the reader")

os.remove(path)
