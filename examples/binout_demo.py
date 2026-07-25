"""End-to-end binout in Python: create arbitrary curves, read them, build a
series, and edit an existing file.

    python examples/binout_demo.py
"""

import os
import tempfile

import numpy as np

import dynars

path = os.path.join(tempfile.gettempdir(), "dynars_demo_binout")

# --- 1. CREATE arbitrary time-history curves ---------------------------------
# A curve = a value per state (dNNNNNN dirs) + a sibling `time` array.
t = np.linspace(0, 1, 12)
e = dynars.BinoutEditor()
for i, ti in enumerate(t):
    d = f"d{i + 1:06d}"
    e.set(["mycurve", d, "time"], np.float64([ti]))
    e.set(["mycurve", d, "energy"], np.float32([np.sin(6 * ti)]))  # your custom quantity
e.set(["mycurve", "metadata", "title"], "custom curve")
e.write(path)
print("wrote", path)

# --- 2. READ it back ---------------------------------------------------------
b = dynars.parse_binout(path)
print("top-level:", b.read())
states = sorted(k for k in b.read(["mycurve"]) if k.startswith("d"))
energy = np.array([b.read(["mycurve", s, "energy"])[0] for s in states])
print("energy curve:", energy.round(3))

# Read many channels in parallel (lock-free, GIL released).
vals = b.read_many([["mycurve", s, "energy"] for s in states])
print("read_many:", len(vals), "channels")

# --- 3. build_series: the LS-DYNA time-series convention ---------------------
# One value per entity per state (metadata `ids` + dNNNNNN state dirs).
ids = np.array([101, 102, 103])
x_disp = np.outer(np.linspace(0, 1, 5), [1.0, 2.0, 3.0]).astype(np.float32)  # (n_state, n_node)
w = dynars.build_series(
    "nodout",
    ids=ids,
    channels={"x_displacement": x_disp},
    times=np.linspace(0, 1, 5),
    labels=[f"node {i}" for i in ids],
)
series_path = path + "_series"
w.write(series_path)
b2 = dynars.parse_binout(series_path)
print("series nodout states:", [k for k in b2.read(["nodout"]) if k.startswith("d")])

# --- 4. EDIT an existing binout ----------------------------------------------
ed = dynars.BinoutEditor(path)
ed.set(["mycurve", "d000001", "energy"], np.float32([999.0]))
ed.write(path)
print("after edit, d000001/energy:", dynars.parse_binout(path).read(["mycurve", "d000001", "energy"]))

for f in (path, series_path):
    if os.path.exists(f):
        os.remove(f)
print("\nOK")
