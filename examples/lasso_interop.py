#!/usr/bin/env python3
"""Cross-tool interop harness (lasso side). Companion to examples/lasso_interop.rs.

  python examples/lasso_interop.py read  <path>   # lasso reads scheme A, asserts
  python examples/lasso_interop.py write <path>    # lasso writes scheme B
"""
import sys
import numpy as np
from lasso.dyna import D3plot, ArrayType as AT

NUMNP = 8
NSTATES = 2


def approx(a, b, what):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert np.allclose(a, b, atol=1e-4), f"{what}: {a} != {b}"


def read_scheme_a(path):
    d = D3plot(path)
    a = d.arrays
    coords = np.arange(NUMNP * 3, dtype=np.float64).reshape(NUMNP, 3)
    approx(a[AT.node_coordinates], coords, "node_coordinates")

    disp = np.stack([coords + s for s in range(NSTATES)])
    approx(a[AT.node_displacement], disp, "node_displacement")
    approx(a[AT.node_velocity], np.full((NSTATES, NUMNP, 3), 0.5), "node_velocity")

    approx(a[AT.global_kinetic_energy], [10.0, 11.0], "global_kinetic_energy")
    approx(a[AT.global_internal_energy], [20.0, 21.0], "global_internal_energy")
    approx(a[AT.global_total_energy], [30.0, 31.0], "global_total_energy")
    approx(a[AT.global_timesteps], [0.0, 1.0], "global_timesteps")

    # solid: 6 stress + 1 pstrain per state
    ss = np.array([[[1000 * s + v for v in range(6)]] for s in range(NSTATES)], dtype=np.float64)
    approx(a[AT.element_solid_stress].reshape(NSTATES, 1, 6), ss, "element_solid_stress")
    ps = np.array([[1000 * s + 6] for s in range(NSTATES)], dtype=np.float64)
    approx(a[AT.element_solid_effective_plastic_strain].reshape(NSTATES, 1), ps, "solid_pstrain")

    # shell: one layer of 6 stress + 1 pstrain
    sh = np.array([[[2000 * s + v for v in range(6)]] for s in range(NSTATES)], dtype=np.float64)
    approx(a[AT.element_shell_stress].reshape(NSTATES, 1, 6), sh, "element_shell_stress")
    shp = np.array([[2000 * s + 6] for s in range(NSTATES)], dtype=np.float64)
    approx(a[AT.element_shell_effective_plastic_strain].reshape(NSTATES, 1), shp, "shell_pstrain")

    # beams: nv1d = 6 (axial, shear s/t, moment s/t, torsion)
    approx(a[AT.element_beam_axial_force].reshape(NSTATES, 1),
           [[3000 * s + 0] for s in range(NSTATES)], "beam_axial_force")
    approx(a[AT.element_beam_shear_force].reshape(NSTATES, 1, 2),
           [[[3000 * s + 1, 3000 * s + 2]] for s in range(NSTATES)], "beam_shear_force")
    approx(a[AT.element_beam_bending_moment].reshape(NSTATES, 1, 2),
           [[[3000 * s + 3, 3000 * s + 4]] for s in range(NSTATES)], "beam_bending_moment")
    approx(a[AT.element_beam_torsion_moment].reshape(NSTATES, 1),
           [[3000 * s + 5] for s in range(NSTATES)], "beam_torsion_moment")

    # nodal temperature (IT=1)
    temp = np.array([[7000 + 100 * s + n for n in range(NUMNP)] for s in range(NSTATES)], dtype=np.float64)
    approx(a[AT.node_temperature].reshape(NSTATES, NUMNP), temp, "node_temperature")

    print("lasso read scheme A OK: coords/disp/vel, solid+shell stress+pstrain,")
    print("  beam resultants (axial/shear/bending/torsion), node temperature, globals, timesteps")


def write_scheme_b(path):
    d = D3plot()
    a = d.arrays
    coords = np.arange(NUMNP * 3, dtype=np.float32).reshape(NUMNP, 3)
    a[AT.node_coordinates] = coords
    a[AT.node_displacement] = np.stack([coords + s for s in range(NSTATES)]).astype(np.float32)
    a[AT.node_velocity] = np.full((NSTATES, NUMNP, 3), 0.25, dtype=np.float32)
    a[AT.global_timesteps] = np.array([0.0, 1.0], dtype=np.float32)

    # one hex solid over all 8 nodes (lasso node indexes are 0-based)
    a[AT.element_solid_node_indexes] = np.arange(8, dtype=np.int64).reshape(1, 8)
    a[AT.element_solid_part_indexes] = np.array([0], dtype=np.int64)
    a[AT.element_solid_ids] = np.array([1], dtype=np.int64)
    # lasso wants a n_solid_layers axis (index 2): stress (states,solids,layers,6),
    # pstrain (states,solids,layers). One layer -> nv3d = 6 stress + 1 pstrain.
    a[AT.element_solid_stress] = np.array(
        [[[[100 * s + v for v in range(6)]]] for s in range(NSTATES)], dtype=np.float32
    )
    a[AT.element_solid_effective_plastic_strain] = np.array(
        [[[100 * s + 6]] for s in range(NSTATES)], dtype=np.float32
    )

    a[AT.global_kinetic_energy] = np.array([1.0, 2.0], dtype=np.float32)
    a[AT.global_internal_energy] = np.array([3.0, 4.0], dtype=np.float32)
    a[AT.global_total_energy] = np.array([7.0, 8.0], dtype=np.float32)

    d.write_d3plot(path)
    print(f"lasso wrote scheme B -> {path}")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("read", "write"):
        print("usage: lasso_interop.py <read|write> <path>", file=sys.stderr)
        sys.exit(2)
    mode, path = sys.argv[1], sys.argv[2]
    (read_scheme_a if mode == "read" else write_scheme_b)(path)


if __name__ == "__main__":
    main()
