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
print("top-level:", b.read())               # branches
print("mycurve variables:", b.read("mycurve"))          # a branch lists its vars
energy = b.read("mycurve", "energy")         # aggregated across all states -> [T]
print("energy curve:", energy.round(3))

# The raw tree is still there: channels() lists a directory's children (the
# dNNNNNN state records), and an explicit state path reads one state.
states = sorted(s for s in b.channels(["mycurve"]) if s.startswith("d"))
vals = b.read_many([["mycurve", s, "energy"] for s in states])   # parallel, lock-free
print("read_many:", len(vals), "state records")

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
disp = b2.read("nodout", "x_displacement")          # [n_states, n_ids]
print("series x_displacement:", disp.shape, "| node 101:", b2.read("nodout", "x_displacement", id=101).round(3))

# --- 4. EDIT an existing binout ----------------------------------------------
ed = dynars.BinoutEditor(path)
ed.set(["mycurve", "d000001", "energy"], np.float32([999.0]))
ed.write(path)
print("after edit, d000001/energy:", dynars.parse_binout(path).read(["mycurve", "d000001", "energy"]))

for f in (path, series_path):
    if os.path.exists(f):
        os.remove(f)
print("\nOK")
