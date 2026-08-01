# Results

dynars reads the two LS-DYNA binary result formats — **`d3plot`** (the state
database: geometry + per-state fields) and **`binout`** (LSDA time histories) —
with numeric data coming back as NumPy arrays in Python and typed `Vec`s in Rust.

## Reading a d3plot

=== "Python"

    ```python
    import dynars

    d = dynars.D3plot("d3plot")            # opens the whole family (d3plot, d3plot01, …)
    print(d.num_nodes, d.num_states)       # properties

    xyz = d.node_coordinates_all()         # flat [x0,y0,z0, x1,y1,z1, …]
    ids = d.node_ids()
    disp = d.displacement_magnitudes(state=d.num_states - 1)

    print(d.available_blocks())            # e.g. [StateBlock.Solid, StateBlock.Shell]
    ```

=== "Rust"

    ```rust
    use dynars::results::D3plot;

    let d = D3plot::open("d3plot").unwrap();
    println!("{} nodes, {} states", d.num_nodes(), d.num_states()); // methods

    let xyz = d.node_coordinates_all().unwrap(); // flat [x0,y0,z0, …]
    ```

## Reading a binout

`binout` is a tree of channels addressed by path — descend from the root, then
read a variable.

=== "Python"

    ```python
    b = dynars.Binout("binout*")
    print(b.read())                # top-level branches, e.g. ['glstat', 'matsum', …]
    print(b.channels(["glstat"]))  # variables under a branch

    ke = b.read(["glstat", "kinetic_energy"])   # NumPy array over states
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

## Element invariants

Derived per-element quantities — von Mises, principal stress/strain, pressure,
triaxiality, effective plastic strain — are computed on the d3plot element blocks
(`results::element` in Rust). Per-part histories (max / percentile /
failure-fraction over states) build on top. See the
[API reference](reference.md) for the full set.

## Signal processing & injury criteria

Result channels are plain arrays, so they chain straight into SAE J211 filtering,
Butterworth filters, integration/differentiation, and occupant-injury criteria —
implemented in the Rust core and verified bit-exact against SciPy.

=== "Python"

    ```python
    import numpy as np, dynars

    a = dynars.resultant(ax, ay, az)     # magnitude of a 3-axis acceleration
    a60 = dynars.cfc(a, 60, dt)          # SAE J211 CFC-60 (zero-phase)
    hic36 = dynars.hic(a, dt, 0.036)     # HIC over a 36 ms window (g-based)
    v = dynars.integrate(a, dt)          # cumulative integral
    ```

    *(These require the extension built with the `signal` feature, which the
    published wheels include.)*

=== "Rust"

    ```rust
    // requires: dynars = { version = "0.1", features = ["signal"] }
    use dynars::results::{signal, injury};

    let a = injury::resultant(&ax, &ay, &az);
    let a60 = signal::cfc(&a, 60.0, dt);
    let hic36 = injury::hic(&a, dt, 0.036);
    ```
