"""End-to-end d3plot in Python: write a model, read it, edit it, resample it.

    python examples/d3plot_demo.py

Covers mesh + connectivity, real IDs (NARBS), per-state nodal results, element
results with custom history variables, generic block extraction (StateBlock
enum, no magic strings), in-place editing, and time-axis resampling.
"""

import os
import tempfile

import numpy as np

import dynars
from dynars import StateBlock

path = os.path.join(tempfile.gettempdir(), "dynars_demo.d3plot")

# --- 1. WRITE a model --------------------------------------------------------
# 8 nodes (unit cube): one hex solid + one quad shell (its bottom face).
coords = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)
w = dynars.D3plotWriter(coords, title="dynars demo")
w.add_solids(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), parts=np.array([1]))
w.add_shells(np.array([[1, 2, 3, 4]]), parts=np.array([2]))
w.set_ids(node_ids=list(range(101, 109)), solid_ids=[9001], shell_ids=[7001], part_ids=[10, 20])

n_states = 4
# Per-solid results: 6 stress + 1 plastic strain + 2 CUSTOM history vars = 9.
solid = np.zeros((n_states, 1, 9))
solid[..., :6] = np.random.default_rng(0).random((n_states, 1, 6))  # stress
solid[..., 6] = 0.01                                                # plastic strain
solid[..., 7] = 42.0                                                # custom field #1
solid[..., 8] = np.arange(n_states)[:, None]                        # custom field #2
w.set_solid_results(solid)

for s in range(n_states):
    w.add_state(s * 1e-3, coords + [0, 0, 0.1 * s], vel=np.zeros_like(coords))
w.write(path)
print("wrote", path)

# --- 2. READ it back ---------------------------------------------------------
d = dynars.open_d3plot(path)
print("\nnodes", d.num_nodes, "states", d.num_states, "times", d.times())
print("blocks present:", d.available_blocks())
conn, parts = d.solid_connectivity()
print("solid conn", conn.tolist(), "parts", parts.tolist(), "node_ids", d.node_ids().tolist())
print("final-state node0 z:", d.node_coordinates(d.num_states - 1)[0, 2])
sol = d.block(StateBlock.Solid)  # (n_states, 1, 9) raw, native f32
print("custom field #1 (col 7):", sol[:, 0, 7], "| #2 (col 8):", sol[:, 0, 8])

# --- 3. EDIT in place --------------------------------------------------------
e = dynars.D3plotEditor(path)
e.set_node_coordinates(0, np.full((d.num_nodes, 3), 9.0))
e.save()
print("\nafter edit, state0 node0:", dynars.open_d3plot(path).node_coordinates(0)[0])


# --- 4. RESAMPLE the time axis (create new states, preserve the mesh) --------
def interp_axis0(x_new, x, y):
    """Linear-interpolate y (shape (T, ...)) onto x_new along axis 0 — numpy only."""
    flat = y.reshape(len(x), -1)
    out = np.empty((len(x_new), flat.shape[1]), dtype=y.dtype)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(x_new, x, flat[:, k])
    return out.reshape((len(x_new),) + y.shape[1:])


d = dynars.open_d3plot(path)
disp, old_t = d.block(StateBlock.Displacement), d.times()
new_t = np.linspace(old_t[0], old_t[-1], 9)          # your d-vs-t remap goes here
new_disp = interp_axis0(new_t, old_t, disp)
out = path + ".resampled"
w2 = dynars.D3plotWriter(new_disp[0])
w2.add_solids(conn, parts=parts)                      # carry the mesh forward
w2.set_ids(node_ids=d.node_ids().tolist(), part_ids=d.part_ids().tolist())
for t, c in zip(new_t, new_disp):
    w2.add_state(float(t), c)
w2.write(out)
print("resampled:", d.num_states, "->", dynars.open_d3plot(out).num_states, "states")

for f in (path, out):
    if os.path.exists(f):
        os.remove(f)
print("\nOK")
