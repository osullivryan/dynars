# Results

dynars reads the two LS-DYNA binary result formats — **`d3plot`** (the state
database: geometry + per-state fields) and **`binout`** (LSDA time histories) —
with numeric data coming back as NumPy arrays in Python and typed `Vec`s in Rust.
It also **writes** both formats, so you can synthesize or edit result files.

## Reading a d3plot

Open the family by its base name; `d3plot`, `d3plot01`, `d3plot02`, … are picked
up automatically.

=== "Python"

    ```python
    import dynars

    d = dynars.open_d3plot("d3plot")       # or dynars.D3plot("d3plot")
    print(d.num_nodes, d.num_states)       # properties
    print(d.times())                       # simulation time of each state
    print(d.available_blocks())            # e.g. [StateBlock.Solid, StateBlock.Shell]
    ```

=== "Rust"

    ```rust
    use dynars::results::D3plot;

    let d = D3plot::open("d3plot").unwrap();
    println!("{} nodes, {} states", d.num_nodes(), d.num_states()); // methods
    println!("times: {:?}", d.times());
    ```

### Geometry and connectivity

Geometry is constant across states. Node ids, element connectivity, and part ids
come back as arrays; connectivity is `(conn, parts)` — the node numbers per
element and the part id each element belongs to.

=== "Python"

    ```python
    ids  = d.node_ids()                    # int64[N] user node IDs
    xyz0 = d.initial_coordinates()         # (N, 3) reference coordinates
    parts = d.part_ids()                   # int64 user part/material IDs

    shell_conn, shell_parts = d.shell_connectivity()  # (n_shells, 4), (n_shells,)
    solid_conn, solid_parts = d.solid_connectivity()  # (n_solids, 8), (n_solids,)
    print(shell_conn[0], "belongs to part", shell_parts[0])
    ```

=== "Rust"

    ```rust
    let ids = d.node_ids();                          // Option<Vec<i64>>
    let (shell_conn, shell_parts) = d.shell_connectivity();  // (Vec<[i64;4]>, Vec<i64>)
    let (solid_conn, solid_parts) = d.solid_connectivity();  // (Vec<[i64;8]>, Vec<i64>)
    let _ = (ids, shell_conn, shell_parts, solid_conn, solid_parts);
    ```

### Per-state nodal data

Deformed coordinates and displacement are per state. Read one state, all states at
once, or the peak displacement directly.

=== "Python"

    ```python
    last = d.num_states - 1
    xyz  = d.node_coordinates(last)        # (N, 3) deformed coords at one state
    allc = d.node_coordinates_all()        # (num_states, N, 3) — one allocation
    disp = d.displacement_magnitudes(last) # (N,) magnitude at one state
    peak = d.max_displacement_final()      # scalar: peak magnitude at final state
    ```

=== "Rust"

    ```rust
    let last = d.num_states() - 1;
    let xyz = d.node_coordinates(last).unwrap();      // flat [x0,y0,z0, x1,y1,z1, …]
    let all = d.node_coordinates_all().unwrap();      // every state, flat
    let peak = d.max_displacement_final().unwrap();
    let _ = (xyz, all, peak);
    ```

### Generic result blocks

Element results (stress, plastic strain, history variables) live in per-entity
blocks. `block(StateBlock.X)` returns the block across selected states as an
`(n_states, count, vars)` array in the solver's native precision — the raw packed
layout, which you reshape by integration points/layers as needed.

=== "Python"

    ```python
    from dynars import StateBlock

    solid = d.block(StateBlock.Solid)        # (n_states, n_solids, vars)
    print(d.block_layout(StateBlock.Solid))  # (count, vars_per_entity)

    # `states` selects which to return: None = all, an int = one, a list = those.
    last_only = d.block(StateBlock.Shell, states=d.num_states - 1)

    # A single-precision contiguous read is zero-copy — a view over the mmap.
    disp = d.block(StateBlock.Displacement)  # (n_states, N, 3)
    ```

The column meanings are the solver's standard packing (6 stress components, then
effective plastic strain, then any history variables you requested). See the
[API reference](reference.md) for the layout per block.

## Element invariants

Derived per-element quantities — von Mises, principal stress/strain, pressure,
triaxiality, effective plastic strain — are computed on the d3plot element blocks
(`results::element` in Rust). Per-part histories (max / percentile /
failure-fraction over states) build on top. See the [API reference](reference.md)
for the full set.

## Reading a binout

`binout` is a tree of channels addressed by path — descend from the root, then
read a variable. Continuation files (`binout0000`, `binout0001`, …) are picked up
from a glob pattern automatically.

=== "Python"

    ```python
    b = dynars.parse_binout("binout*")     # or dynars.Binout("binout*")
    print(b.read())                        # top-level branches, e.g. ['glstat', 'matsum', …]
    print(b.channels(["glstat"]))          # variables under a branch

    ke   = b.read(["glstat", "kinetic_energy"])   # NumPy array over states
    time = b.read(["glstat", "time"])
    ```

=== "Rust"

    ```rust
    use dynars::results::Binout;

    let b = Binout::new("binout*").unwrap();
    let top = b.read(&[]).unwrap();                 // top-level branches
    let ke = b.read(&["glstat", "kinetic_energy"]); // then descend
    let _ = (top, ke);
    ```

### Time histories and per-state matrices

Two convenience readers save you the manual assembly:

- **`read_time_series(path)`** pairs a channel with its sibling `time` array —
  `{"time", "values", "channel"}`.
- **`read_states(branch, var)`** aggregates a per-entity variable across every
  state directory into a dense `(T, C)` matrix, plus the entity `ids` and
  `time` — one node's history is a column.

=== "Python"

    ```python
    # A single global channel over time:
    ts = b.read_time_series(["glstat", "kinetic_energy"])
    plt_x, plt_y = ts["time"], ts["values"]

    # Every node's x-acceleration as a (T, C) matrix:
    st = b.read_states("nodout", "x_acceleration")
    time, values, ids = st["time"], st["values"], st["ids"]

    # Pull one node's history by its LS-DYNA id:
    import numpy as np
    col = np.nonzero(ids == 101)[0][0]
    node_101 = values[:, col]

    # Entity metadata for a branch:
    print(b.ids("nodout")[:5], b.legend("nodout")[:2], b.title("nodout"))
    ```

=== "Rust"

    ```rust
    let ts = b.read_time_series(&["glstat", "kinetic_energy"]).unwrap();
    let st = b.read_states("nodout", "x_acceleration").unwrap();
    let _ = (ts, st);
    ```

### Reading many channels at once

Pulling a lot of channels? `read_many` runs the reads in parallel across cores
(GIL released), faster than a Python loop.

=== "Python"

    ```python
    paths = [["nodout", f"d{ i+1 :06d}", "x_acceleration"] for i in range(200)]
    arrays = b.read_many(paths)            # list aligned with `paths`
    ```

## Signal processing & injury criteria

Result channels are plain arrays, so they chain straight into SAE J211 filtering,
Butterworth filters, integration/differentiation, and occupant-injury criteria —
implemented in the Rust core and verified bit-exact against SciPy.

=== "Python"

    ```python
    import numpy as np, dynars

    b = dynars.parse_binout("binout*")
    dt = 1e-4  # s

    # A 3-axis acceleration from three channels, filtered and integrated:
    ax = b.read(["nodout", "d000001", "x_acceleration"])
    ay = b.read(["nodout", "d000001", "y_acceleration"])
    az = b.read(["nodout", "d000001", "z_acceleration"])

    a    = dynars.resultant(ax, ay, az)  # sqrt(x^2 + y^2 + z^2)
    a60  = dynars.cfc(a, 60, dt)         # SAE J211 CFC-60 (zero-phase)
    low  = dynars.butterworth(a, 4, 300.0, 1/dt, "low")  # 4-pole, 300 Hz
    v    = dynars.integrate(a, dt)       # cumulative integral (accel -> velocity)

    # Occupant injury criteria (acceleration in g):
    hic36 = dynars.hic36(a, dt)          # HIC over a 36 ms window (also hic15, hic)
    a3ms  = dynars.clip(a, dt)           # 3 ms clip
    csi   = dynars.severity_index(a, dt) # Gadd severity index
    ```

    *(Filtering and injury criteria ship in the published wheels.)*

=== "Rust"

    ```rust
    // requires: dynars = { version = "1.0", features = ["signal"] }
    use dynars::results::{signal, injury};

    let a = injury::resultant(&ax, &ay, &az);
    let a60 = signal::cfc(&a, 60.0, dt);
    let v = signal::integrate(&a, dt);
    let hic36 = injury::hic(&a, dt, 0.036);
    let _ = (a60, v, hic36);
    ```

The Rust `injury` module also carries the tier-2 criteria (`bric`, `ubric`, `vc`,
`nij`, `nic`, `tibia_index`); see the [API reference](reference.md).

## Writing result files

You can build a `d3plot` or `binout` from scratch — for synthetic test data, for
round-tripping an edited field, or for emitting a derived result LS-PrePost can
open.

### Build a d3plot

=== "Python"

    ```python
    import numpy as np, dynars
    from dynars import StateBlock

    # A unit cube: 8 nodes, one hex solid + one quad shell.
    coords = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)

    w = dynars.D3plotWriter(coords, title="demo")
    w.add_solids(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), parts=np.array([1]))  # 1-based nodes
    w.add_shells(np.array([[1, 2, 3, 4]]), parts=np.array([2]))
    w.set_ids(node_ids=list(range(101, 109)), solid_ids=[9001], part_ids=[10, 20])

    for s in range(4):                       # append states
        w.add_state(s * 1e-3, coords + [0, 0, 0.1 * s], vel=np.zeros_like(coords))
    w.write("demo.d3plot")

    # Read it straight back:
    d = dynars.open_d3plot("demo.d3plot")
    print(d.num_states, d.solid_connectivity()[0].tolist())
    ```

Editing an existing family in place is `D3plotEditor` — overwrite node
coordinates or a result block at chosen states and `save()`; everything else is
preserved byte-for-byte. A full end-to-end walk-through (write → read → edit →
resample) is `examples/d3plot_demo.py` / `examples/d3plot_demo.rs`.

### Build a binout

A binout curve is a value per state (`dNNNNNN` directories) plus a sibling `time`
array. `BinoutEditor` writes an arbitrary tree; the `build_series` helper writes
the LS-DYNA time-series convention (metadata + per-state dirs) so files read back
as proper histories in dynars and LS-PrePost.

=== "Python"

    ```python
    import numpy as np, dynars

    # Low-level: an arbitrary curve.
    e = dynars.BinoutEditor()
    t = np.linspace(0, 1, 12)
    for i, ti in enumerate(t):
        d = f"d{i + 1:06d}"
        e.set(["mycurve", d, "time"], np.float64([ti]))
        e.set(["mycurve", d, "energy"], np.float32([np.sin(6 * ti)]))
    e.write("out.binout")

    # High-level: a proper nodout-style series (one value per entity per state).
    ids = np.array([101, 102, 103])
    x_disp = np.outer(np.linspace(0, 1, 5), [1.0, 2.0, 3.0]).astype(np.float32)  # (T, C)
    w = dynars.build_series("nodout", ids=ids, channels={"x_displacement": x_disp},
                            times=np.linspace(0, 1, 5), labels=[f"node {i}" for i in ids])
    w.write("series.binout")
    ```

`examples/binout_demo.py` / `examples/binout_demo.rs` are the runnable versions
(create → read → build series → edit).
