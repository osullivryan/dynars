# Recipes

Short, copy-paste answers to "how do I…" questions. Each assumes
`import dynars` (Python) or `use dynars::...` (Rust) and a parsed deck. For the
concepts behind them, see [Concepts](concepts.md); for full APIs, the
[reference](reference.md).

## Census: what does this deck contain?

=== "Python"

    ```python
    deck = dynars.parse_deck("main.k")
    for kind, count in deck.definition_counts():
        print(f"{count:>8}  {kind}")
    ```

=== "Rust"

    ```rust
    let deck = dynars::deck::parse_deck(std::path::Path::new("main.k")).unwrap();
    for (kind, count) in deck.definition_counts() {
        println!("{count:>8}  {kind:?}");
    }
    ```

## How big is the model, and what does it include?

=== "Python"

    ```python
    root = dynars.parse_include_tree("main.k")
    mb = root.total_bytes() / 1e6
    print(f"{root.total_files()} files, {mb:.1f} MB")
    ```

=== "Rust"

    ```rust
    let root = dynars::include::build_include_tree(std::path::Path::new("main.k")).unwrap();
    println!("{} files, {:.1} MB", root.total_files(), root.total_bytes() as f64 / 1e6);
    ```

## Find every part that uses a given material

=== "Python"

    ```python
    target = 5
    for part in deck.parts():
        mat = part.material()
        if mat is not None and mat.id == target:
            print(part.id, part.field("secid"))
    ```

=== "Rust"

    ```rust
    let target = 5;
    for part in deck.parts() {
        if let Some(mat) = part.material() {
            if mat.id() == Some(target) {
                println!("{:?}", part.id());
            }
        }
    }
    ```

## List every material with its density

=== "Python"

    ```python
    for m in deck.materials():
        print(m.id, m.keyword, "RO =", m.field("RO"))
    ```

=== "Rust"

    ```rust
    for m in deck.materials() {
        let ro = m.field("RO").and_then(|f| f.as_f64());
        println!("{:?} {} RO = {:?}", m.id(), m.name(), ro);
    }
    ```

## Dump all node coordinates to a NumPy array (or CSV)

=== "Python"

    ```python
    nodes = deck.table("NODE")           # {"nid", "x", "y", "z"} as arrays
    import numpy as np
    xyz = np.column_stack([nodes["x"], nodes["y"], nodes["z"]])
    np.savetxt("nodes.csv", np.column_stack([nodes["nid"], xyz]),
               delimiter=",", header="nid,x,y,z", comments="")
    ```

=== "Rust"

    ```rust
    let nodes = deck.table("NODE").unwrap();
    let (nid, x) = (nodes.column("nid").unwrap().as_int().unwrap(),
                    nodes.column("x").unwrap().as_float().unwrap());
    println!("{} nodes, first at x = {}", nid.len(), x[0]);
    ```

## Get shell connectivity as one array

=== "Python"

    ```python
    shells = deck.table("ELEMENT_SHELL")
    eid, pid, conn = shells["eid"], shells["pid"], shells["nodes"]   # conn is (N, 4)
    print(conn[0], "-> part", pid[0])
    ```

## Find dangling references and print clickable locations

=== "Python"

    ```python
    report = deck.validate([
        dynars.Rule.references_resolve(),
        dynars.Rule.include_missing(),
    ])
    for f in report.findings:
        print(f"{f.location()}: {f.message}")   # file:line, clickable in a terminal
    ```

=== "Rust"

    ```rust
    use dynars::validate::Rule;
    let report = deck.validate([Rule::references_resolve(), Rule::include_missing()]);
    for f in &report.findings {
        println!("{}: {}", f.location(), f.message);
    }
    ```

## Follow a boundary condition to its load curve

=== "Python"

    ```python
    for bc in deck.parts():          # any entity works; a *BOUNDARY_* here
        curve = bc.reference("lcid") # follow the named field to *DEFINE_CURVE
        if curve is not None:
            print(bc.id, "-> curve", curve.id)
    ```

## Check every deck in a folder at once

=== "Python"

    ```python
    from glob import glob
    ws = dynars.Workspace()
    roots = glob("runs/*/main.k")
    decks = ws.parse_decks(roots)                 # shared *INCLUDEs read once
    reports = ws.validate_decks(decks, [
        dynars.Rule.references_resolve(),
        dynars.Rule.duplicate_ids(),
    ])
    for root, r in zip(roots, reports):
        print("OK " if r.is_clean() else "FAIL", root, r.count(dynars.Severity.Error))
    print(ws.stats())
    ```

=== "Rust"

    ```rust
    use dynars::Workspace;
    use dynars::validate::Rule;

    let ws = Workspace::new();
    let roots = ["runs/a/main.k", "runs/b/main.k"];
    let decks: Vec<_> = ws.parse_decks(roots).into_iter()
        .filter_map(|(_root, d)| d.ok()).collect();
    let reports = ws.validate_decks(&decks, [Rule::references_resolve(), Rule::duplicate_ids()]);
    println!("{} decks checked", reports.len());
    ```

## Plot a global energy history from a binout

=== "Python"

    ```python
    import matplotlib.pyplot as plt
    b = dynars.parse_binout("binout*")
    ts = b.read_time_series(["glstat", "kinetic_energy"])
    plt.plot(ts["time"], ts["values"]); plt.xlabel("time"); plt.ylabel("KE")
    plt.savefig("ke.png")
    ```

## Filter an acceleration channel and compute HIC

=== "Python"

    ```python
    b = dynars.parse_binout("binout*")
    dt = 1e-4
    ax = b.read(["nodout", "d000001", "x_acceleration"])
    ay = b.read(["nodout", "d000001", "y_acceleration"])
    az = b.read(["nodout", "d000001", "z_acceleration"])
    a = dynars.cfc(dynars.resultant(ax, ay, az), 1000, dt)   # CFC1000
    print("HIC36 =", dynars.hic36(a, dt), " 3ms clip =", dynars.clip(a, dt))
    ```

## Read one node's displacement history across all states (d3plot)

=== "Python"

    ```python
    d = dynars.open_d3plot("d3plot")
    ids = d.node_ids()
    import numpy as np
    i = int(np.nonzero(ids == 101)[0][0])         # node id -> row index
    allc = d.node_coordinates_all()               # (num_states, N, 3)
    history = allc[:, i, :] - allc[0, i, :]        # displacement vs. state 0
    ```

## Change a material property and write the deck back

Surgical, include-aware, byte-minimal: only the one field changes; comments,
rulers, and the rest of the deck are preserved verbatim.

=== "Python"

    ```python
    deck = dynars.parse_deck("root.k")

    deck.material(72).set_field("e", 2.1e11)      # by id; "in_place" / "reflowed"
    # or by name: deck.keywords("MAT_ELASTIC")[0].set_field("e", 2.1e11)

    for f in deck.files():                         # realise the write-time overlay
        if f.dirty:
            f.write(f.path)
    ```

=== "Rust"

    ```rust
    let mut deck = dynars::deck::parse_deck("root.k".as_ref()).unwrap();
    if let Some(loc) = deck.material(72).and_then(|m| m.locate("e")) {
        deck.set_field(&loc, "2.1e11");
    }
    for f in &deck.files {
        if f.is_dirty() { f.write(&f.path).unwrap(); }
    }
    ```

## Read a keyword dynars doesn't ship

=== "Python"

    ```python
    deck = dynars.parse_deck("root.k")
    cards = [[("wid", "int", 8, 1), ("mass", "float", 8, 1)]]
    cols = deck.table_with("VENDOR_WIDGET", cards)   # columnar, like any built-in
    ```

See [Schemas](schemas.md) for the class-based (`@keyword` / `#[derive(Keyword)]`)
front ends and reference declarations.

## Compare two decks' contents

=== "Python"

    ```python
    a = dict(dynars.parse_deck("base/main.k").definition_counts())
    b = dict(dynars.parse_deck("variant/main.k").definition_counts())
    for kind in sorted(set(a) | set(b)):
        if a.get(kind, 0) != b.get(kind, 0):
            print(f"{kind}: {a.get(kind, 0)} -> {b.get(kind, 0)}")
    ```
