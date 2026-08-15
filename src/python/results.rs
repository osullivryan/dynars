//! PyO3 bindings: binary results (binout / d3plot).

use numpy::IntoPyArray;
use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// -- Phase 5: binary results (binout / d3plot) -------------------------

use crate::results::{
    Binout as RustBinout, BinoutEditor as RustBinoutEditor, BlockArray, D3plot as RustD3plot,
    D3plotEditor as RustD3plotEditor, D3plotError, D3plotWriter as RustD3plotWriter, Data,
    FsiforField, GlobalField, InterfaceField, IntforWriter as RustIntforWriter, LsdaError,
    NodeField, PartField, ReadResult, ResultBlock, StateBlock,
};
use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn};

fn lsda_err(e: LsdaError) -> PyErr {
    let msg = e.to_string();
    match e {
        LsdaError::FileNotFound => pyo3::exceptions::PyFileNotFoundError::new_err(msg),
        LsdaError::SymbolNotFound(_) => pyo3::exceptions::PyKeyError::new_err(msg),
        _ => pyo3::exceptions::PyRuntimeError::new_err(msg),
    }
}

fn d3_err(e: D3plotError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Flatten a float32/float64 numpy array to a `Vec<f64>` (row-major).
fn f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<f32>>() {
        return Ok(a.as_array().iter().map(|&x| x as f64).collect());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a float32 or float64 numpy array",
    ))
}

/// Build a [`ResultBlock`] from a numpy array, preserving its full shape as the
/// block dims (row-major) — e.g. `(n_states, n_elem, vars)`.
fn result_block(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<ResultBlock> {
    let dims: Vec<usize> = if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
        a.as_array().shape().to_vec()
    } else if let Ok(a) = obj.extract::<PyReadonlyArrayDyn<f32>>() {
        a.as_array().shape().to_vec()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a float32 or float64 numpy array",
        ));
    };
    Ok(ResultBlock::new(dims, f64_vec(obj)?))
}

/// Reshape flat connectivity into `(rows, cols)` + a 1-D parts array.
fn conn_to_py<'py>(
    py: Python<'py>,
    nodes: Vec<i64>,
    parts: Vec<i64>,
    cols: usize,
) -> PyResult<(Bound<'py, pyo3::PyAny>, Bound<'py, pyo3::PyAny>)> {
    let rows = nodes.len().checked_div(cols).unwrap_or(0);
    let conn = numpy::ndarray::Array2::from_shape_vec((rows, cols), nodes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py)
        .into_any();
    Ok((conn, parts.into_pyarray(py).into_any()))
}

/// A data leaf becomes the natural numpy dtype (zero-copy from the read
/// buffer); a directory becomes a `list[str]` of child names.
fn readresult_to_py<'py>(py: Python<'py>, r: ReadResult) -> PyResult<Bound<'py, pyo3::PyAny>> {
    Ok(match r {
        ReadResult::Directory(keys) => {
            let list = PyList::empty(py);
            for k in keys {
                list.append(String::from_utf8_lossy(&k).into_owned())?;
            }
            list.into_any()
        }
        ReadResult::I8(v) => v.into_pyarray(py).into_any(),
        ReadResult::I16(v) => v.into_pyarray(py).into_any(),
        ReadResult::I32(v) => v.into_pyarray(py).into_any(),
        ReadResult::I64(v) => v.into_pyarray(py).into_any(),
        ReadResult::U8(v) => v.into_pyarray(py).into_any(),
        ReadResult::U16(v) => v.into_pyarray(py).into_any(),
        ReadResult::U32(v) => v.into_pyarray(py).into_any(),
        ReadResult::U64(v) => v.into_pyarray(py).into_any(),
        ReadResult::F32(v) => v.into_pyarray(py).into_any(),
        ReadResult::F64(v) => v.into_pyarray(py).into_any(),
        ReadResult::Link(v) => v.into_pyarray(py).into_any(),
    })
}

/// Normalize `read`'s positional args into path segments: a single list/tuple
/// arg is expanded to its items, otherwise each arg is one segment — so both
/// `read("a", "b")` and `read(["a", "b"])` work.
fn norm_segs(path: &[Bound<'_, pyo3::PyAny>]) -> PyResult<Vec<String>> {
    if path.len() == 1 {
        if let Ok(list) = path[0].cast::<PyList>() {
            return list.iter().map(|x| x.extract::<String>()).collect();
        }
        if let Ok(tup) = path[0].cast::<pyo3::types::PyTuple>() {
            return tup.iter().map(|x| x.extract::<String>()).collect();
        }
    }
    path.iter().map(|x| x.extract::<String>()).collect()
}

/// Whether `name` is a `dNNNNNN` per-state record directory.
fn is_state_dir(name: &str) -> bool {
    name.len() > 1 && name.starts_with('d') && name[1..].bytes().all(|b| b.is_ascii_digit())
}

/// A resolved column selection: which columns, and whether it came from a single
/// (`id`/`name`) selector — a single one yields a 1-D `[T]` result, a plural
/// (`ids`/`names`) selector a 2-D `[T, k]`.
struct Selection {
    cols: Vec<usize>,
    single: bool,
}

/// Resolve at most one of `{id, ids, name, names}` to column indices: `id`/`ids`
/// look up `<branch>/metadata/ids`, `name`/`names` the `<branch>/metadata/legend`
/// entity names. `Ok(None)` when no selector is given; `ValueError` if more than
/// one is; `KeyError` if an id/name is absent.
fn resolve_cols(
    inner: &RustBinout,
    py: Python<'_>,
    branch: &str,
    id: Option<i64>,
    ids: Option<Vec<i64>>,
    name: Option<String>,
    names: Option<Vec<String>>,
) -> PyResult<Option<Selection>> {
    let n = id.is_some() as u8 + ids.is_some() as u8 + name.is_some() as u8 + names.is_some() as u8;
    if n == 0 {
        return Ok(None);
    }
    if n > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pass only one of id, ids, name, names",
        ));
    }
    // `id`/`ids` index metadata/ids; `name`/`names` index metadata/legend.
    let by_ids = id.is_some() || ids.is_some();
    let table = if by_ids {
        py.detach(|| inner.ids(branch))
            .map_err(lsda_err)?
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
    } else {
        py.detach(|| inner.legend(branch)).map_err(lsda_err)?
    };
    let find = |want: &str| -> PyResult<usize> {
        table.iter().position(|s| s == want).ok_or_else(|| {
            let what = if by_ids { "id" } else { "name" };
            pyo3::exceptions::PyKeyError::new_err(format!("{what} {want:?} not found in '{branch}'"))
        })
    };
    let (wanted, single): (Vec<String>, bool) = match (id, ids, name, names) {
        (Some(id), ..) => (vec![id.to_string()], true),
        (_, Some(ids), ..) => (ids.iter().map(|v| v.to_string()).collect(), false),
        (_, _, Some(name), _) => (vec![name], true),
        (_, _, _, Some(names)) => (names, false),
        _ => unreachable!(),
    };
    let cols = wanted.iter().map(|w| find(w)).collect::<PyResult<Vec<usize>>>()?;
    Ok(Some(Selection { cols, single }))
}

/// Turn a resolved [`Selection`] plus decoded columns into a numpy array: `[T]`
/// for a single selector, `[T, k]` otherwise.
fn columns_to_py<'py>(
    py: Python<'py>,
    sc: &crate::results::StateColumns,
    single: bool,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    if single {
        Ok(sc.values.clone().into_pyarray(py).into_any())
    } else {
        let values =
            numpy::ndarray::Array2::from_shape_vec((sc.n_steps, sc.n_cols), sc.values.clone())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(values.into_pyarray(py).into_any())
    }
}

/// Aggregate `branch/dNNNNNN/var` across all states into a numpy array — the
/// lasso-style data path behind `read(branch, var[, id/ids/name/names])`. Full:
/// `[T, C]` (or `[T]` for a scalar-per-state channel like `time`); a selector
/// decodes only the chosen column(s) (`[T]` or `[T, k]`), never building the
/// full matrix.
#[allow(clippy::too_many_arguments)]
fn aggregate<'py>(
    inner: &RustBinout,
    py: Python<'py>,
    branch: &str,
    var: &str,
    id: Option<i64>,
    ids: Option<Vec<i64>>,
    name: Option<String>,
    names: Option<Vec<String>>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    if let Some(sel) = resolve_cols(inner, py, branch, id, ids, name, names)? {
        let sc = py
            .detach(|| inner.read_columns(branch, var, &sel.cols))
            .map_err(lsda_err)?;
        return columns_to_py(py, &sc, sel.single);
    }
    let m = py.detach(|| inner.read_states(branch, var)).map_err(lsda_err)?;
    if m.n_channels <= 1 {
        return Ok(m.values.into_pyarray(py).into_any());
    }
    let values = numpy::ndarray::Array2::from_shape_vec((m.n_steps, m.n_channels), m.values)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py).into_any())
}

/// LS-DYNA binout reader: walk the LSDA tree by path, read channels as numpy.
#[pyclass(name = "Binout")]
pub struct PyBinout {
    inner: RustBinout,
}

#[pymethods]
impl PyBinout {
    /// Open a binout (glob pattern; continuation files `binout%NNN` are
    /// picked up automatically). Releases the GIL while indexing.
    #[new]
    fn new(py: Python<'_>, pattern: String) -> PyResult<Self> {
        let inner = py.detach(|| RustBinout::new(&pattern)).map_err(lsda_err)?;
        Ok(Self { inner })
    }

    /// The binout files backing this reader, in order.
    #[getter]
    fn files(&self) -> Vec<String> {
        self.inner.filelist.clone()
    }

    /// Read from the binout (lasso-style). Segments may be separate args or one
    /// list: `read("nodout", …)` or `read(["nodout", …])`.
    ///
    /// - `read()` / `read("nodout")` → `list[str]` of children (a branch lists
    ///   its variable names).
    /// - `read("nodout", "x_acceleration")` → the variable aggregated across all
    ///   output states: `float64[T, nodes]` (or `[T]` for a scalar-per-state
    ///   channel such as `time`).
    /// - `read("nodout", "x_acceleration", id=1000001)` → one entity's history
    ///   `float64[T]`; `ids=[…]` → `float64[T, k]`. Select by entity name (from
    ///   the branch `legend`) instead with `name=` / `names=[…]`. Selectors decode
    ///   only the requested column(s) — no full matrix. `KeyError` if absent.
    /// - A literal leaf path — `read("nodout", "d000001", "x_acceleration")` —
    ///   returns that single state's raw array.
    ///
    /// For the structured form (time + ids together) use `read_states`; for the
    /// raw child listing of any directory use `channels`.
    #[pyo3(signature = (*path, id=None, ids=None, name=None, names=None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        path: Vec<Bound<'py, pyo3::PyAny>>,
        id: Option<i64>,
        ids: Option<Vec<i64>>,
        name: Option<String>,
        names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let segs = norm_segs(&path)?;
        let has_selector =
            id.is_some() || ids.is_some() || name.is_some() || names.is_some();
        if has_selector {
            if segs.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "read(branch, var, id=/name=...) needs a branch and a variable",
                ));
            }
            return aggregate(&self.inner, py, &segs[0], &segs[1], id, ids, name, names);
        }
        let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
        match py.detach(|| self.inner.read(&seg_refs)) {
            Ok(ReadResult::Directory(keys)) => {
                let names: Vec<String> = keys
                    .iter()
                    .map(|k| String::from_utf8_lossy(k).into_owned())
                    .collect();
                // A state-bearing branch lists its variable names (peek a state
                // dir), not the raw dNNNNNN records — the lasso view. Use
                // `channels` for the raw children.
                if let Some(state) = names.iter().find(|k| is_state_dir(k)) {
                    let mut vpath: Vec<&str> = seg_refs.clone();
                    vpath.push(state);
                    let vars = py
                        .detach(|| self.inner.channels(&vpath))
                        .map_err(lsda_err)?;
                    return Ok(vars.into_pyobject(py)?.into_any());
                }
                Ok(names.into_pyobject(py)?.into_any())
            }
            Ok(leaf) => readresult_to_py(py, leaf),
            // Not a literal node: a branch+var pair aggregates across states.
            Err(e) => {
                if segs.len() == 2 {
                    aggregate(&self.inner, py, &segs[0], &segs[1], None, None, None, None)
                } else {
                    Err(lsda_err(e))
                }
            }
        }
    }

    /// Read many paths concurrently (lock-free, GIL released), returning a
    /// list aligned with `paths`. Faster than a Python loop when pulling
    /// many channels: the reads run in parallel across cores.
    #[pyo3(signature = (paths))]
    fn read_many<'py>(
        &self,
        py: Python<'py>,
        paths: Vec<Vec<String>>,
    ) -> PyResult<Vec<Bound<'py, pyo3::PyAny>>> {
        let refs: Vec<Vec<&str>> = paths
            .iter()
            .map(|p| p.iter().map(String::as_str).collect())
            .collect();
        let results = py.detach(|| self.inner.read_many(&refs));
        results
            .into_iter()
            .map(|r| readresult_to_py(py, r.map_err(lsda_err)?))
            .collect()
    }

    /// Read a leaf and coerce to float64 (any numeric dtype).
    #[pyo3(signature = (path))]
    fn read_f64<'py>(
        &self,
        py: Python<'py>,
        path: Vec<String>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        let v = py.detach(|| self.inner.read_f64(&segs)).map_err(lsda_err)?;
        Ok(v.into_pyarray(py).into_any())
    }

    /// Read a time-history: `{"time": float64[T], "values": float64[T], "channel": str}`.
    /// `time` is read from the sibling `time` array, or synthesized as 0..T.
    #[pyo3(signature = (path))]
    fn read_time_series<'py>(
        &self,
        py: Python<'py>,
        path: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        let ts = py
            .detach(|| self.inner.read_time_series(&segs))
            .map_err(lsda_err)?;
        let d = PyDict::new(py);
        d.set_item("time", ts.time.into_pyarray(py))?;
        d.set_item("values", ts.values.into_pyarray(py))?;
        d.set_item("channel", ts.channel)?;
        Ok(d)
    }

    /// Child names at a directory path (empty path = top level).
    #[pyo3(signature = (path=Vec::new()))]
    fn channels(&self, py: Python<'_>, path: Vec<String>) -> PyResult<Vec<String>> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        py.detach(|| self.inner.channels(&segs)).map_err(lsda_err)
    }

    /// Aggregate a per-state variable across all state dirs, as a dict.
    ///
    /// Full matrix (default): `{"time": float64[T], "values": float64[T, C],
    /// "ids": int64[C], "n_steps": int, "n_channels": int}`.
    ///
    /// With a selector — `id`/`ids` (by entity id) or `name`/`names` (by the
    /// branch `legend`) — only those columns are decoded (no full matrix):
    /// `{"time": float64[T], "values": float64[T] or [T, k], "ids": int64[k]}`,
    /// where `values` is 1-D for a single `id`/`name`. `KeyError` if absent,
    /// `ValueError` if more than one selector is given. The bare-array
    /// counterpart is `read(branch, var, …)`.
    #[pyo3(signature = (branch, var, id=None, ids=None, name=None, names=None))]
    #[allow(clippy::too_many_arguments)]
    fn read_states<'py>(
        &self,
        py: Python<'py>,
        branch: String,
        var: String,
        id: Option<i64>,
        ids: Option<Vec<i64>>,
        name: Option<String>,
        names: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);

        // Selector: decode only the chosen columns from each state (no full matrix).
        if let Some(sel) = resolve_cols(&self.inner, py, &branch, id, ids, name, names)? {
            let sc = py
                .detach(|| self.inner.read_columns(&branch, &var, &sel.cols))
                .map_err(lsda_err)?;
            let all_ids = py.detach(|| self.inner.ids(&branch)).unwrap_or_default();
            let sel_ids: Vec<i64> = sel
                .cols
                .iter()
                .map(|&c| all_ids.get(c).copied().unwrap_or(0))
                .collect();
            let values = columns_to_py(py, &sc, sel.single)?;
            d.set_item("time", sc.time.into_pyarray(py))?;
            d.set_item("values", values)?;
            d.set_item("ids", sel_ids.into_pyarray(py))?;
            return Ok(d);
        }

        // Full matrix.
        let m = py
            .detach(|| self.inner.read_states(&branch, &var))
            .map_err(lsda_err)?;
        let (nt, nc) = (m.n_steps, m.n_channels);
        let values = numpy::ndarray::Array2::from_shape_vec((nt, nc), m.values)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        d.set_item("time", m.time.into_pyarray(py))?;
        d.set_item("values", values.into_pyarray(py))?;
        d.set_item("ids", m.ids.into_pyarray(py))?;
        d.set_item("n_steps", nt)?;
        d.set_item("n_channels", nc)?;
        Ok(d)
    }

    /// LS-DYNA entity IDs for a state branch (e.g. `nodout` node IDs), as int64.
    #[pyo3(signature = (branch))]
    fn ids<'py>(&self, py: Python<'py>, branch: String) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let v = py.detach(|| self.inner.ids(&branch)).map_err(lsda_err)?;
        Ok(v.into_pyarray(py).into_any())
    }

    /// Per-entity legend/name strings for a state branch (trimmed).
    #[pyo3(signature = (branch))]
    fn legend(&self, branch: String) -> PyResult<Vec<String>> {
        self.inner.legend(&branch).map_err(lsda_err)
    }

    /// Dataset title for a state branch.
    #[pyo3(signature = (branch))]
    fn title(&self, branch: String) -> PyResult<String> {
        self.inner.title(&branch).map_err(lsda_err)
    }

    fn __repr__(&self) -> String {
        format!("Binout({} file(s))", self.inner.filelist.len())
    }
}

/// LS-DYNA d3plot reader: control block, geometry, per-state nodal results.
#[pyclass(name = "D3plot")]
pub struct PyD3plot {
    inner: RustD3plot,
}

#[pymethods]
impl PyD3plot {
    /// Open a d3plot file (single-file, structural layout — see the Rust
    /// `d3plot` module docs for scope).
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let inner = py.detach(|| RustD3plot::open(&path)).map_err(d3_err)?;
        Ok(Self { inner })
    }

    /// Number of nodes (NUMNP) in the mesh.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of output states (time steps) in the file.
    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    /// Simulation time of each state, as a float64 array.
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
        self.inner.times().to_vec().into_pyarray(py).into_any()
    }

    /// Deformed node coordinates at `state` (0-based) as an `(NUMNP, 3)` array.
    fn node_coordinates<'py>(
        &self,
        py: Python<'py>,
        state: usize,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let v = py
            .detach(|| self.inner.node_coordinates(state))
            .map_err(d3_err)?;
        let rows = v.len() / 3;
        let a = numpy::ndarray::Array2::from_shape_vec((rows, 3), v)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any())
    }

    /// Deformed node coordinates for every state as a `(num_states, NUMNP, 3)`
    /// array — one call, one allocation, instead of a Python loop over
    /// `node_coordinates`.
    fn node_coordinates_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let v = py
            .detach(|| self.inner.node_coordinates_all())
            .map_err(d3_err)?;
        let ns = self.inner.num_states();
        let nn = self.inner.num_nodes();
        let a = numpy::ndarray::Array3::from_shape_vec((ns, nn, 3), v)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any())
    }

    /// Per-node displacement magnitude at `state` as a `(NUMNP,)` array.
    fn displacement_magnitudes<'py>(
        &self,
        py: Python<'py>,
        state: usize,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let v = py
            .detach(|| self.inner.displacement_magnitudes(state))
            .map_err(d3_err)?;
        Ok(v.into_pyarray(py).into_any())
    }

    /// Peak nodal displacement magnitude at the final state.
    fn max_displacement_final(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.max_displacement_final())
            .map_err(d3_err)
    }

    /// Initial (reference) node coordinates as an `(N, 3)` array.
    fn initial_coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let v = self.inner.initial_coordinates().to_vec();
        let rows = v.len() / 3;
        let a = numpy::ndarray::Array2::from_shape_vec((rows, 3), v)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any())
    }

    /// Shell connectivity: `(conn, parts)` where `conn` is `(n_shells, 4)`
    /// one-based node numbers and `parts` is `(n_shells,)`.
    fn shell_connectivity<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, pyo3::PyAny>, Bound<'py, pyo3::PyAny>)> {
        let (nodes, parts) = self.inner.shell_connectivity();
        conn_to_py(py, nodes, parts, 4)
    }

    /// Solid connectivity: `(conn, parts)` where `conn` is `(n_solids, 8)`.
    fn solid_connectivity<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, pyo3::PyAny>, Bound<'py, pyo3::PyAny>)> {
        let (nodes, parts) = self.inner.solid_connectivity();
        conn_to_py(py, nodes, parts, 8)
    }

    /// User node IDs (`N`), default `1..=N`.
    fn node_ids<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
        self.inner.node_ids().into_pyarray(py).into_any()
    }
    /// User shell element IDs.
    fn shell_ids<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
        self.inner.shell_ids().into_pyarray(py).into_any()
    }
    /// User solid element IDs.
    fn solid_ids<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
        self.inner.solid_ids().into_pyarray(py).into_any()
    }
    /// User part/material IDs.
    fn part_ids<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
        self.inner.part_ids().into_pyarray(py).into_any()
    }

    /// Control-block file type (1 = d3plot, 4 = intfor, …).
    #[getter]
    fn filetype(&self) -> i64 {
        self.inner.filetype()
    }

    /// Whether this is an interface-force (`intfor`) database. In an intfor
    /// file the contact **segments** are in the shell slot:
    /// `block(StateBlock.Shell)` gives `(n_states, n_segments, nv2d)` and
    /// `shell_connectivity()` the segment nodes; split the per-segment values
    /// with `interface_fields`.
    #[getter]
    fn is_interface_force(&self) -> bool {
        self.inner.is_interface_force()
    }

    /// Whether this is an FSIFOR (ALE) interface-force file — use
    /// `FsiforField` values with `segment_field`.
    #[getter]
    fn is_fsifor(&self) -> bool {
        self.inner.is_fsifor()
    }

    /// Extract one interface-force field's values from the per-segment block
    /// as `(n_states, n_segments, k)`. `field` is an `InterfaceField` (intfor)
    /// or `FsiforField` (FSIFOR) — no magic strings. `states` selects states
    /// like `block`. Raises if the field isn't present in this file.
    #[pyo3(signature = (field, states=None))]
    fn segment_field<'py>(
        &self,
        py: Python<'py>,
        field: Bound<'py, pyo3::PyAny>,
        states: Option<Bound<'py, pyo3::PyAny>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let (off, count) = if let Ok(f) = field.extract::<InterfaceField>() {
            self.inner.interface_field_span(f)
        } else if let Ok(f) = field.extract::<FsiforField>() {
            self.inner.fsifor_field_span(f)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "field must be an InterfaceField or FsiforField",
            ));
        };
        if count == 0 {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                "that interface field is not present in this file",
            ));
        }
        let sel: Option<Vec<i64>> = match &states {
            None => None,
            Some(o) => Some(match o.extract::<i64>() {
                Ok(i) => vec![i],
                Err(_) => o.extract::<Vec<i64>>()?,
            }),
        };
        let idx = self.inner.resolve_states(sel.as_deref()).map_err(d3_err)?;
        let (data, [ns, seg, nv2d]) =
            self.inner
                .block_data(StateBlock::Shell, &idx)
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("no interface segment data in this file")
                })?;
        // Slice columns [off, off+count) out of each segment's nv2d values.
        let slice = |flat: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(ns * seg * count);
            for e in 0..ns * seg {
                let base = e * nv2d + off;
                out.extend_from_slice(&flat[base..base + count]);
            }
            out
        };
        let arr = match data {
            BlockArray::F32(v) => {
                numpy::ndarray::Array3::from_shape_vec((ns, seg, count), slice(&v))
            }
            BlockArray::F64(v) => {
                let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                numpy::ndarray::Array3::from_shape_vec((ns, seg, count), slice(&v32))
            }
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py).into_any())
    }

    /// The result blocks present in this d3plot, as `StateBlock` values.
    fn available_blocks(&self) -> Vec<StateBlock> {
        ALL_BLOCKS
            .into_iter()
            .filter(|b| self.inner.block_layout(*b).is_some())
            .collect()
    }

    /// Generic result extraction: any result block across all states as an
    /// `(n_states, count, vars)` numpy array in native precision. `block` is
    /// a `StateBlock` (or its lowercase name string). Node blocks are
    /// `(…, 3)`; element blocks return the solver's raw packed per-entity
    /// layout — reshape by integration points/layers as needed. Raises if
    /// the block is absent.
    ///
    /// `states` selects which states to return: `None` = all; an int (or
    /// negative int, from the end) = one state; a sequence of ints = those
    /// states. Selecting fewer states reads/copies only those.
    ///
    /// When the selected states are single-precision and contiguous within
    /// one family file, the result is a **zero-copy** read-only view straight
    /// over the memory map (no allocation, no copy). Otherwise the selection
    /// is copied into a fresh array (in parallel for large blocks).
    #[pyo3(signature = (block, states=None))]
    fn block<'py>(
        slf: Bound<'py, Self>,
        block: StateBlock,
        states: Option<Bound<'py, pyo3::PyAny>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let py = slf.py();
        let b = block;

        let sel: Option<Vec<i64>> = match &states {
            None => None,
            Some(o) => Some(match o.extract::<i64>() {
                Ok(i) => vec![i],
                Err(_) => o.extract::<Vec<i64>>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "states must be None, an int, or a sequence of ints",
                    )
                })?,
            }),
        };
        let idx = slf
            .borrow()
            .inner
            .resolve_states(sel.as_deref())
            .map_err(d3_err)?;

        // Zero-copy fast path: strided view over the mmap, kept alive by
        // tying the array's base to this D3plot object.
        let view_info = slf.borrow().inner.block_view(b, &idx);
        if let Some((fi, byte_off, [ns, count, vars], stride)) = view_info {
            use numpy::PyUntypedArrayMethods;
            use numpy::ndarray::ShapeBuilder;
            let borrow = slf.borrow();
            let bytes = borrow.inner.file_bytes(fi);
            // SAFETY: byte_off + block extents were validated on open; the
            // region is 4-aligned; the mapping stays alive via `container`.
            let ptr = unsafe { bytes.as_ptr().add(byte_off) } as *const f32;
            let shape = (ns, count, vars).strides((stride / 4, vars, 1));
            let view = unsafe { numpy::ndarray::ArrayView3::<f32>::from_shape_ptr(shape, ptr) };
            let arr = unsafe { numpy::PyArray3::borrow_from_array(&view, slf.clone().into_any()) };
            // The map is read-only — forbid writes (would fault the mapping).
            unsafe { (*arr.as_array_ptr()).flags &= !numpy::npyffi::NPY_ARRAY_WRITEABLE };
            drop(borrow);
            return Ok(arr.into_any());
        }

        // Fallback: copy into a fresh array (multi-file family or double precision).
        let (data, [ns, count, vars]) =
            slf.borrow().inner.block_data(b, &idx).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "block {b:?} is not present in this d3plot"
                ))
            })?;
        let shape_err =
            |e: numpy::ndarray::ShapeError| pyo3::exceptions::PyValueError::new_err(e.to_string());
        match data {
            BlockArray::F32(v) => Ok(numpy::ndarray::Array3::from_shape_vec((ns, count, vars), v)
                .map_err(shape_err)?
                .into_pyarray(py)
                .into_any()),
            BlockArray::F64(v) => Ok(numpy::ndarray::Array3::from_shape_vec((ns, count, vars), v)
                .map_err(shape_err)?
                .into_pyarray(py)
                .into_any()),
        }
    }

    /// The `(count, vars_per_entity)` layout of a result block, or None.
    #[pyo3(signature = (block))]
    fn block_layout(&self, block: StateBlock) -> PyResult<Option<(usize, usize)>> {
        Ok(self.inner.block_layout(block))
    }

    fn __repr__(&self) -> String {
        format!(
            "D3plot({} nodes, {} states)",
            self.inner.num_nodes(),
            self.inner.num_states()
        )
    }
}

const ALL_BLOCKS: [StateBlock; 7] = [
    StateBlock::Displacement,
    StateBlock::Velocity,
    StateBlock::Acceleration,
    StateBlock::Solid,
    StateBlock::ThickShell,
    StateBlock::Beam,
    StateBlock::Shell,
];

/// Open an LS-DYNA binout for reading (mirrors [`PyBinout::new`]).
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn parse_binout(py: Python<'_>, pattern: String) -> PyResult<PyBinout> {
    PyBinout::new(py, pattern)
}

/// Open an LS-DYNA d3plot for reading (mirrors [`PyD3plot::new`]).
#[pyfunction]
#[pyo3(signature = (path))]
pub fn open_d3plot(py: Python<'_>, path: String) -> PyResult<PyD3plot> {
    PyD3plot::new(py, path)
}

/// Build a single-precision d3plot from a mesh + per-state nodal results.
#[pyclass(name = "D3plotWriter")]
pub struct PyD3plotWriter {
    inner: RustD3plotWriter,
}

#[pymethods]
impl PyD3plotWriter {
    /// `node_coords` is `(N, 3)` (or flat `3N`) initial coordinates.
    #[new]
    #[pyo3(signature = (node_coords, title=None))]
    fn new(node_coords: Bound<'_, pyo3::PyAny>, title: Option<String>) -> PyResult<Self> {
        let mut inner = RustD3plotWriter::new(f64_vec(&node_coords)?).map_err(d3_err)?;
        if let Some(t) = title {
            inner.set_title(&t);
        }
        Ok(Self { inner })
    }

    /// Add shell elements: `conn` is `(M, 4)` one-based node ids; `parts` is
    /// an optional `(M,)` part id per shell (default 1).
    #[pyo3(signature = (conn, parts=None))]
    fn add_shells(
        &mut self,
        conn: PyReadonlyArray2<'_, i64>,
        parts: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let a = conn.as_array();
        if a.ncols() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shell conn must have shape (M, 4)",
            ));
        }
        for (i, row) in a.rows().into_iter().enumerate() {
            let part = parts.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(1) as i32;
            self.inner.add_shell(
                [row[0] as i32, row[1] as i32, row[2] as i32, row[3] as i32],
                part,
            );
        }
        Ok(())
    }

    /// Add solid elements: `conn` is `(M, 8)` one-based node ids; `parts` is
    /// an optional `(M,)` part id per solid (default 1).
    #[pyo3(signature = (conn, parts=None))]
    fn add_solids(
        &mut self,
        conn: PyReadonlyArray2<'_, i64>,
        parts: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let a = conn.as_array();
        if a.ncols() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "solid conn must have shape (M, 8)",
            ));
        }
        for (i, row) in a.rows().into_iter().enumerate() {
            let part = parts.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(1) as i32;
            let mut nodes = [0i32; 8];
            for (d, &v) in nodes.iter_mut().zip(row.iter()) {
                *d = v as i32;
            }
            self.inner.add_solid(nodes, part);
        }
        Ok(())
    }

    /// Set user IDs written into the NARBS numbering section (default 1..N):
    /// node IDs (length N), shell/solid element IDs, and part IDs.
    #[pyo3(signature = (node_ids=None, shell_ids=None, solid_ids=None, part_ids=None))]
    fn set_ids(
        &mut self,
        node_ids: Option<Vec<i64>>,
        shell_ids: Option<Vec<i64>>,
        solid_ids: Option<Vec<i64>>,
        part_ids: Option<Vec<i64>>,
    ) {
        if let Some(v) = node_ids {
            self.inner.set_node_ids(v);
        }
        if let Some(v) = shell_ids {
            self.inner.set_shell_ids(v);
        }
        if let Some(v) = solid_ids {
            self.inner.set_solid_ids(v);
        }
        if let Some(v) = part_ids {
            self.inner.set_part_ids(v);
        }
    }

    /// Append a state: `time`, deformed coords `disp` `(N,3)`, and optional
    /// `vel`/`acc` `(N,3)`. Velocity/acceleration presence is fixed by the
    /// first state added.
    #[pyo3(signature = (time, disp, vel=None, acc=None))]
    fn add_state(
        &mut self,
        time: f64,
        disp: Bound<'_, pyo3::PyAny>,
        vel: Option<Bound<'_, pyo3::PyAny>>,
        acc: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let vel = vel.map(|v| f64_vec(&v)).transpose()?;
        let acc = acc.map(|a| f64_vec(&a)).transpose()?;
        self.inner
            .add_state(time, f64_vec(&disp)?, vel, acc)
            .map_err(d3_err)
    }

    /// Emit double-precision (8-byte word) output when `double` is true (default
    /// single precision). Values are stored as f64, so this is lossless.
    #[pyo3(signature = (double))]
    fn set_double_precision(&mut self, double: bool) {
        self.inner.set_double_precision(double);
    }

    /// Number of through-thickness integration points packed into each shell
    /// result record (MAXINT). `set_shell_results`' innermost dim must be
    /// `n_layers * per_layer`.
    #[pyo3(signature = (n_layers))]
    fn set_shell_layers(&mut self, n_layers: usize) {
        self.inner.set_shell_layers(n_layers);
    }

    /// Add beam elements: `conn` is `(M, 3)` one-based node ids (end, end,
    /// orientation); `parts` optional `(M,)` part id (default 1).
    #[pyo3(signature = (conn, parts=None))]
    fn add_beams(&mut self, conn: PyReadonlyArray2<'_, i64>, parts: Option<Vec<i64>>) -> PyResult<()> {
        let a = conn.as_array();
        if a.ncols() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beam conn must have shape (M, 3)",
            ));
        }
        for (i, row) in a.rows().into_iter().enumerate() {
            let part = parts.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(1) as i32;
            self.inner
                .add_beam([row[0] as i32, row[1] as i32, row[2] as i32], part);
        }
        Ok(())
    }

    /// Add thick-shell elements: `conn` is `(M, 8)` one-based node ids; `parts`
    /// optional `(M,)` part id (default 1).
    #[pyo3(signature = (conn, parts=None))]
    fn add_tshells(&mut self, conn: PyReadonlyArray2<'_, i64>, parts: Option<Vec<i64>>) -> PyResult<()> {
        let a = conn.as_array();
        if a.ncols() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tshell conn must have shape (M, 8)",
            ));
        }
        for (i, row) in a.rows().into_iter().enumerate() {
            let part = parts.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(1) as i32;
            let mut nodes = [0i32; 8];
            for (d, &v) in nodes.iter_mut().zip(row.iter()) {
                *d = v as i32;
            }
            self.inner.add_tshell(nodes, part);
        }
        Ok(())
    }

    /// Per-solid result block, `(n_states, n_solids, vars)` — the same raw
    /// layout `D3plot.solid_results()` returns. Sets NV3D.
    #[pyo3(signature = (results))]
    fn set_solid_results(&mut self, results: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_solid_results(result_block(&results)?);
        Ok(())
    }

    /// Per-shell result block, `(n_states, n_shells, vars)`. Sets NV2D.
    #[pyo3(signature = (results))]
    fn set_shell_results(&mut self, results: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_shell_results(result_block(&results)?);
        Ok(())
    }

    /// Per-beam result block, `(n_states, n_beams, vars)`. Sets NV1D.
    #[pyo3(signature = (results))]
    fn set_beam_results(&mut self, results: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_beam_results(result_block(&results)?);
        Ok(())
    }

    /// Per-thick-shell result block, `(n_states, n_tshells, vars)`. Sets NV3DT.
    #[pyo3(signature = (results))]
    fn set_tshell_results(&mut self, results: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_tshell_results(result_block(&results)?);
        Ok(())
    }

    /// A whole-model global scalar history (one value per state) at `field`'s slot.
    #[pyo3(signature = (field, data))]
    fn set_global_history(&mut self, field: GlobalField, data: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_global_history(field, f64_vec(&data)?);
        Ok(())
    }

    /// A per-part scalar history `(n_states, n_parts)` at `field`.
    #[pyo3(signature = (field, data))]
    fn set_part_field(&mut self, field: PartField, data: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_part_field(field, f64_vec(&data)?);
        Ok(())
    }

    /// Per-part velocity history `(n_states, n_parts, 3)`.
    #[pyo3(signature = (data))]
    fn set_part_velocity(&mut self, data: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_part_velocity(f64_vec(&data)?);
        Ok(())
    }

    /// A per-node thermal/auxiliary field history at `field`. See
    /// `D3plotWriter.set_node_field` (Rust) for the per-node widths.
    #[pyo3(signature = (field, data))]
    fn set_node_field(&mut self, field: NodeField, data: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_node_field(field, f64_vec(&data)?);
        Ok(())
    }

    /// Per-element deletion flags for one family (`block`): `(n_states, n_elem)`,
    /// 1 = alive, 0 = deleted (mdlopt 2).
    #[pyo3(signature = (block, alive))]
    fn set_element_deletion(&mut self, block: StateBlock, alive: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_element_deletion(block, f64_vec(&alive)?);
        Ok(())
    }

    /// Per-node deletion flags: `(n_states, numnp)`, 1 = alive (mdlopt 1).
    #[pyo3(signature = (alive))]
    fn set_node_deletion(&mut self, alive: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner.set_node_deletion(f64_vec(&alive)?);
        Ok(())
    }

    /// User beam / thick-shell element IDs for the NARBS numbering section.
    #[pyo3(signature = (beam_ids=None, tshell_ids=None))]
    fn set_element_ids(&mut self, beam_ids: Option<Vec<i64>>, tshell_ids: Option<Vec<i64>>) {
        if let Some(v) = beam_ids {
            self.inner.set_beam_ids(v);
        }
        if let Some(v) = tshell_ids {
            self.inner.set_tshell_ids(v);
        }
    }

    /// SPH particles: `materials` `(P,)`, `n_vars` per particle, `results`
    /// `(n_states, P, n_vars)`.
    #[pyo3(signature = (materials, n_vars, results))]
    fn set_sph(
        &mut self,
        materials: Vec<i64>,
        n_vars: usize,
        results: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner.set_sph(materials, n_vars, f64_vec(&results)?);
        Ok(())
    }

    /// Airbag / CPM: geometry `(n_airbags, n_geom_vars)`, chamber state
    /// `(n_states, n_airbags, n_airbag_vars)`, particle state `(n_states,
    /// n_particles, n_particle_vars)`.
    #[pyo3(signature = (n_airbags, n_particles, n_geom_vars, n_airbag_vars, n_particle_vars, geom, airbag_state, particle_state))]
    #[allow(clippy::too_many_arguments)]
    fn set_airbag(
        &mut self,
        n_airbags: usize,
        n_particles: usize,
        n_geom_vars: usize,
        n_airbag_vars: usize,
        n_particle_vars: usize,
        geom: Bound<'_, pyo3::PyAny>,
        airbag_state: Bound<'_, pyo3::PyAny>,
        particle_state: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner.set_airbag(
            n_airbags,
            n_particles,
            n_geom_vars,
            n_airbag_vars,
            n_particle_vars,
            f64_vec(&geom)?,
            f64_vec(&airbag_state)?,
            f64_vec(&particle_state)?,
        );
        Ok(())
    }

    /// Rigid bodies: `bodies` is a list of `(part_id, node_ids, active_node_ids)`;
    /// `motion` is `(n_states, n_bodies, k)` (k = 12 with a rigid road, else 24).
    #[pyo3(signature = (bodies, motion))]
    fn set_rigid_bodies(
        &mut self,
        bodies: Vec<(i64, Vec<i64>, Vec<i64>)>,
        motion: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner.set_rigid_bodies(bodies, f64_vec(&motion)?);
        Ok(())
    }

    /// Rigid road: node ids `(P,)`, node coords `(P, 3)`, `segments` a list of
    /// `(road_id, [4 node ids per segment])`, motion `(n_states, n_roads, 6)`.
    #[pyo3(signature = (node_ids, node_coords, segments, motion))]
    fn set_rigid_road(
        &mut self,
        node_ids: Vec<i64>,
        node_coords: Bound<'_, pyo3::PyAny>,
        segments: Vec<(i64, Vec<i64>)>,
        motion: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner
            .set_rigid_road(node_ids, f64_vec(&node_coords)?, segments, f64_vec(&motion)?);
        Ok(())
    }

    /// Rigid walls: `force` `(n_states, n_walls)`, optional `position`
    /// `(n_states, n_walls, 3)`.
    #[pyo3(signature = (n_walls, force, position=None))]
    fn set_rigid_walls(
        &mut self,
        n_walls: usize,
        force: Bound<'_, pyo3::PyAny>,
        position: Option<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        let position = position.map(|p| f64_vec(&p)).transpose()?;
        self.inner.set_rigid_walls(n_walls, f64_vec(&force)?, position);
        Ok(())
    }

    /// The d3plot as bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
    }

    /// Write the d3plot to `path`.
    #[pyo3(signature = (path))]
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let bytes = self.inner.to_bytes();
        py.detach(|| std::fs::write(&path, bytes))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    }
}

/// Build an interface-force (`intfor`) file: contact segments + per-state
/// nodal motion + per-segment interface values.
#[pyclass(name = "IntforWriter")]
pub struct PyIntforWriter {
    inner: RustIntforWriter,
}

#[pymethods]
impl PyIntforWriter {
    /// `node_coords` is `(N, 3)`; `n_interfaces` sliding interfaces.
    #[new]
    #[pyo3(signature = (node_coords, n_interfaces=1, title=None))]
    fn new(
        node_coords: Bound<'_, pyo3::PyAny>,
        n_interfaces: usize,
        title: Option<String>,
    ) -> PyResult<Self> {
        let mut inner =
            RustIntforWriter::new(f64_vec(&node_coords)?, n_interfaces).map_err(d3_err)?;
        if let Some(t) = title {
            inner.set_title(&t);
        }
        Ok(Self { inner })
    }

    /// Add contact segments: `conn` is `(M, 4)` one-based node ids; `ids` is
    /// an optional `(M,)` segment id per segment (default 1..M).
    #[pyo3(signature = (conn, ids=None))]
    fn add_segments(
        &mut self,
        conn: PyReadonlyArray2<'_, i64>,
        ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let a = conn.as_array();
        if a.ncols() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "segment conn must have shape (M, 4)",
            ));
        }
        for (i, row) in a.rows().into_iter().enumerate() {
            let id = ids
                .as_ref()
                .and_then(|v| v.get(i))
                .copied()
                .unwrap_or(i as i64 + 1) as i32;
            self.inner.add_segment(
                [row[0] as i32, row[1] as i32, row[2] as i32, row[3] as i32],
                id,
            );
        }
        Ok(())
    }

    /// User node IDs (length N) for the NARBS numbering section.
    #[pyo3(signature = (node_ids))]
    fn set_node_ids(&mut self, node_ids: Vec<i64>) {
        self.inner.set_node_ids(node_ids);
    }

    /// Declare the intfor per-segment field layout (nv2d = their sum).
    #[pyo3(signature = (wear=0, pressure=0, shear=0, force=0, gap=0))]
    fn set_fields(&mut self, wear: usize, pressure: usize, shear: usize, force: usize, gap: usize) {
        self.inner.set_fields(wear, pressure, shear, force, gap);
    }

    /// Mark this an FSIFOR (ALE) file with `n` fixed per-segment values.
    #[pyo3(signature = (n))]
    fn set_fsifor(&mut self, n: usize) {
        self.inner.set_fsifor(n);
    }

    /// Values per segment in each state.
    #[getter]
    fn nv2d(&self) -> usize {
        self.inner.nv2d()
    }

    /// Append a state: `time`, deformed `disp` `(N,3)`, `vel` `(N,3)`, and
    /// `segment_values` `(n_segments, nv2d)`.
    #[pyo3(signature = (time, disp, vel, segment_values))]
    fn add_state(
        &mut self,
        time: f64,
        disp: Bound<'_, pyo3::PyAny>,
        vel: Bound<'_, pyo3::PyAny>,
        segment_values: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner
            .add_state(
                time,
                f64_vec(&disp)?,
                f64_vec(&vel)?,
                f64_vec(&segment_values)?,
            )
            .map_err(d3_err)
    }

    /// The intfor file as bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
    }

    /// Write the intfor file to `path`.
    #[pyo3(signature = (path))]
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let bytes = self.inner.to_bytes();
        py.detach(|| std::fs::write(&path, bytes))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    }
}

/// Edit an existing d3plot family in place: overwrite node coordinates or a
/// result block at chosen states; everything else is preserved byte-for-byte.
#[pyclass(name = "D3plotEditor")]
pub struct PyD3plotEditor {
    inner: RustD3plotEditor,
}

#[pymethods]
impl PyD3plotEditor {
    /// Load a d3plot family (base + `d3plot01`, …) for editing.
    #[new]
    #[pyo3(signature = (path))]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let inner = py
            .detach(|| RustD3plotEditor::open(&path))
            .map_err(d3_err)?;
        Ok(Self { inner })
    }

    /// Number of nodes (NUMNP) in the mesh.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of output states (time steps) in the family.
    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    /// Overwrite a result `block` (a `StateBlock`) at `state` with `data`
    /// `(count, vars)` — the same layout `D3plot.block(...)` returns.
    #[pyo3(signature = (block, state, data))]
    fn set_block(
        &mut self,
        block: StateBlock,
        state: usize,
        data: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        let v: Vec<f32> = f64_vec(&data)?.into_iter().map(|x| x as f32).collect();
        self.inner.set_block(block, state, &v).map_err(d3_err)
    }

    /// Overwrite deformed node coordinates `(N, 3)` at `state`.
    #[pyo3(signature = (state, coords))]
    fn set_node_coordinates(
        &mut self,
        state: usize,
        coords: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        let v: Vec<f32> = f64_vec(&coords)?.into_iter().map(|x| x as f32).collect();
        self.inner.set_node_coordinates(state, &v).map_err(d3_err)
    }

    /// Overwrite the original files in place.
    fn save(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.save()).map_err(d3_err)
    }

    /// Write the edited family to a new base path (`path`, `path01`, …).
    #[pyo3(signature = (path))]
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.inner.write(&path)).map_err(d3_err)
    }
}

// -- Phase 5b: binout writing (construct / edit via full rewrite) -------

fn data_to_py<'py>(py: Python<'py>, d: &Data) -> Bound<'py, pyo3::PyAny> {
    match d {
        Data::I8(v) => v.clone().into_pyarray(py).into_any(),
        Data::I16(v) => v.clone().into_pyarray(py).into_any(),
        Data::I32(v) => v.clone().into_pyarray(py).into_any(),
        Data::I64(v) => v.clone().into_pyarray(py).into_any(),
        Data::U8(v) => v.clone().into_pyarray(py).into_any(),
        Data::U16(v) => v.clone().into_pyarray(py).into_any(),
        Data::U32(v) => v.clone().into_pyarray(py).into_any(),
        Data::U64(v) => v.clone().into_pyarray(py).into_any(),
        Data::F32(v) => v.clone().into_pyarray(py).into_any(),
        Data::F64(v) => v.clone().into_pyarray(py).into_any(),
        Data::Str(s) => pyo3::types::PyString::new(py, s).into_any(),
    }
}

/// Coerce a Python value (str, numpy array of any shape, or list of
/// numbers) into a typed [`Data`]. The numpy dtype selects the LSDA type
/// verbatim; arrays of rank > 1 are flattened row-major (LSDA leaves are
/// flat, so shape is not preserved).
fn pyany_to_data(v: &Bound<'_, pyo3::PyAny>) -> PyResult<Data> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(Data::Str(s));
    }
    macro_rules! try_arr {
        ($ty:ty, $variant:ident) => {
            if let Ok(a) = v.extract::<PyReadonlyArrayDyn<$ty>>() {
                return Ok(Data::$variant(a.as_array().iter().copied().collect()));
            }
        };
    }
    try_arr!(f64, F64);
    try_arr!(f32, F32);
    try_arr!(i64, I64);
    try_arr!(i32, I32);
    try_arr!(i16, I16);
    try_arr!(i8, I8);
    try_arr!(u64, U64);
    try_arr!(u32, U32);
    try_arr!(u16, U16);
    try_arr!(u8, U8);
    if let Ok(list) = v.extract::<Vec<f64>>() {
        return Ok(Data::F64(list));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a 1-D numpy array, a list of numbers, or a str",
    ))
}

/// Editable binout: a directory tree of typed datasets that writes back a
/// complete LSDA file. Construct new, or open an existing file and mutate it
/// (save re-emits the whole file).
#[pyclass(name = "BinoutEditor")]
pub struct PyBinoutEditor {
    inner: RustBinoutEditor,
}

#[pymethods]
impl PyBinoutEditor {
    /// `BinoutEditor()` starts empty; `BinoutEditor(path)` loads an existing
    /// binout (glob pattern) fully into memory.
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(py: Python<'_>, path: Option<String>) -> PyResult<Self> {
        let inner = match path {
            Some(p) => py.detach(|| RustBinoutEditor::open(&p)).map_err(lsda_err)?,
            None => RustBinoutEditor::new(),
        };
        Ok(Self { inner })
    }

    /// Child names at a directory path (empty path = top level); None if the
    /// path is a dataset.
    #[pyo3(signature = (path=Vec::new()))]
    fn list(&self, path: Vec<String>) -> Option<Vec<String>> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        self.inner.list(&segs)
    }

    /// The dataset at `path` as a numpy array / str, or None.
    #[pyo3(signature = (path))]
    fn get<'py>(&self, py: Python<'py>, path: Vec<String>) -> Option<Bound<'py, pyo3::PyAny>> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        self.inner.get(&segs).map(|d| data_to_py(py, d))
    }

    /// Create or overwrite the dataset at `path` (parent dirs autocreated).
    #[pyo3(signature = (path, values))]
    fn set(&mut self, path: Vec<String>, values: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        let data = pyany_to_data(values)?;
        self.inner.set(&segs, data).map_err(lsda_err)
    }

    /// Remove the dataset/directory at `path`; returns whether it existed.
    #[pyo3(signature = (path))]
    fn remove(&mut self, path: Vec<String>) -> bool {
        let segs: Vec<&str> = path.iter().map(String::as_str).collect();
        self.inner.remove(&segs)
    }

    /// The whole tree serialized as LSDA bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
    }

    /// Write the whole tree to `path` as an LSDA (binout) file.
    #[pyo3(signature = (path))]
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let bytes = self.inner.to_bytes();
        py.detach(|| std::fs::write(&path, bytes))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    }
}
