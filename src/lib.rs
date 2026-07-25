pub mod include_tree;
pub mod keyword;
pub mod keywords;
pub mod parser;
pub mod results;
pub mod schema;
pub mod testgen;
pub mod typed;

/// `#[derive(Keyword)]` / `#[derive(Card)]` for declaring keyword schemas as
/// structs (see [`schema`]).
pub use dynars_derive::{Card, Keyword};
pub use schema::{CardLayout, KeywordSchema};

#[cfg(feature = "python")]
mod python_bindings {
    use std::path::Path;

    use pyo3::prelude::*;
    use pyo3::PyResult;

    use crate::keyword::IncludeNode as RustIncludeNode;

    #[pyclass(name = "IncludeNode")]
    #[derive(Debug, Clone)]
    pub struct PyIncludeNode {
        #[pyo3(get)]
        path: String,
        #[pyo3(get)]
        byte_count: usize,
        #[pyo3(get)]
        kind: Option<String>,
        #[pyo3(get)]
        children: Vec<PyIncludeNode>,
    }

    #[pymethods]
    impl PyIncludeNode {
        /// Total number of files in this subtree (including self).
        fn total_files(&self) -> usize {
            1 + self.children.iter().map(|c| c.total_files()).sum::<usize>()
        }

        /// Total bytes across all files in this subtree.
        fn total_bytes(&self) -> usize {
            self.byte_count + self.children.iter().map(|c| c.total_bytes()).sum::<usize>()
        }

        fn __repr__(&self) -> String {
            let kind_str = match &self.kind {
                Some(k) => format!(" [{}]", k),
                None => String::new(),
            };
            format!(
                "IncludeNode('{}'{}, {} bytes, {} children)",
                self.path, kind_str, self.byte_count, self.children.len(),
            )
        }
    }

    pub(crate) fn rust_to_py(node: &RustIncludeNode) -> PyIncludeNode {
        PyIncludeNode {
            path: node.path.display().to_string(),
            byte_count: node.byte_count,
            kind: node.kind.as_ref().map(|k| format!("{:?}", k)),
            children: node.children.iter().map(rust_to_py).collect(),
        }
    }

    /// Parse an LS-DYNA keyword file and return the include tree.
    ///
    /// Releases the GIL during parsing so other Python threads can run.
    #[pyfunction]
    #[pyo3(signature = (path))]
    pub fn parse_include_tree(path: String) -> PyResult<PyIncludeNode> {
        let file_path = Path::new(&path);

        let result = crate::include_tree::build_include_tree(file_path);

        match result {
            Ok(root) => Ok(rust_to_py(&root)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    // -- Phase 4: keyword-file marshalling ---------------------------------

    use numpy::IntoPyArray;
    use pyo3::types::{PyDict, PyList};
    use pyo3::Bound;

    use crate::keyword::ParsedFile;
    use crate::parser::Keyword;
    use crate::schema::{Card, Column, FieldSpec, FieldType, Schema};

    /// A parsed LS-DYNA keyword file: keyword blocks with lossless round-trip,
    /// columnar bulk access as numpy arrays, and block-level editing.
    #[pyclass(name = "KeywordFile")]
    pub struct PyKeywordFile {
        inner: ParsedFile,
    }

    #[pymethods]
    impl PyKeywordFile {
        /// Number of keyword blocks in the file.
        #[getter]
        fn num_blocks(&self) -> usize {
            self.inner.blocks.len()
        }

        /// The keyword name of every block, in file order.
        fn block_names(&self) -> Vec<String> {
            self.inner
                .blocks
                .iter()
                .map(|b| self.inner.keyword_name(b).to_string())
                .collect()
        }

        /// A block as a dict: `{"name": str, "options": [str], "cards": [[str]]}`.
        fn keyword<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
            let block = self.inner.blocks.get(index).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("block index {} out of range", index))
            })?;
            let kw = self.inner.keyword(block);
            let d = PyDict::new(py);
            d.set_item("name", kw.name)?;
            d.set_item("options", kw.options)?;
            let cards = PyList::empty(py);
            for card in kw.cards {
                cards.append(card)?;
            }
            d.set_item("cards", cards)?;
            Ok(d)
        }

        /// Replace a block's keyword. Cards are re-emitted in free format; the
        /// rest of the file stays byte-for-byte intact.
        #[pyo3(signature = (index, name, cards, options=None))]
        fn set_keyword(
            &mut self,
            index: usize,
            name: String,
            cards: Vec<Vec<String>>,
            options: Option<Vec<String>>,
        ) -> PyResult<()> {
            if index >= self.inner.blocks.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "block index {} out of range",
                    index
                )));
            }
            let kw = Keyword {
                name,
                options: options.unwrap_or_default(),
                cards,
            };
            self.inner.set_keyword(index, &kw);
            Ok(())
        }

        /// Parse a keyword against a user-defined schema, returning a dict of
        /// columns (numpy arrays for numeric fields, lists for strings).
        ///
        /// Low-level: the Python `@keyword` class layer lowers to this. `cards`
        /// is a list of cards, each a list of `(name, type, width, count)` field
        /// tuples where `type` is "int" | "float" | "str".
        #[pyo3(signature = (keyword, cards, repeat=false))]
        fn parse_schema<'py>(
            &self,
            py: Python<'py>,
            keyword: String,
            cards: Vec<Vec<(String, String, usize, usize)>>,
            repeat: bool,
        ) -> PyResult<Bound<'py, PyDict>> {
            let mut schema = Schema::new(&keyword);
            schema.repeat = repeat;
            for card in cards {
                let mut c = Card::new();
                for (name, ty, width, count) in card {
                    let ty = match ty.as_str() {
                        "int" => FieldType::Int,
                        "float" => FieldType::Float,
                        "str" => FieldType::Str,
                        other => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "unknown field type '{}' (expected int/float/str)",
                                other
                            )));
                        }
                    };
                    c.fields.push(FieldSpec { name, ty, width, count: count.max(1) });
                }
                schema.cards.push(c);
            }

            let table = py.detach(|| crate::schema::parse_schema(&self.inner, &schema));
            table_to_pydict(py, table)
        }

        /// Parse a keyword using dynars' built-in library (generated from the
        /// pyDYNA field database), returning the same column dict. Errors if the
        /// keyword is not in the library.
        fn parse_builtin<'py>(
            &self,
            py: Python<'py>,
            keyword: String,
        ) -> PyResult<Bound<'py, PyDict>> {
            let schema = crate::keywords::schema(&keyword).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "'{}' is not in the built-in keyword library",
                    keyword
                ))
            })?;
            let table = py.detach(|| crate::schema::parse_schema(&self.inner, &schema));
            table_to_pydict(py, table)
        }

        /// Whether any block has a pending edit.
        #[getter]
        fn dirty(&self) -> bool {
            self.inner.is_dirty()
        }

        /// The (possibly edited) file contents as bytes.
        fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
            pyo3::types::PyBytes::new(py, &self.inner.to_bytes())
        }

        /// Write the (possibly edited) file to disk.
        fn write(&self, path: String) -> PyResult<()> {
            self.inner
                .write(Path::new(&path))
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        fn __repr__(&self) -> String {
            format!(
                "KeywordFile('{}', {} blocks{})",
                self.inner.path.display(),
                self.inner.blocks.len(),
                if self.inner.is_dirty() { ", edited" } else { "" },
            )
        }
    }

    /// Convert a columnar [`Table`](crate::schema::Table) into a Python dict of
    /// numpy arrays (2-D for array fields) and string lists.
    fn table_to_pydict<'py>(
        py: Python<'py>,
        table: crate::schema::Table,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in table.columns {
            match col {
                Column::Int { data, ncols } => {
                    if ncols <= 1 {
                        d.set_item(name, data.into_pyarray(py))?;
                    } else {
                        let rows = data.len() / ncols;
                        let a = numpy::ndarray::Array2::from_shape_vec((rows, ncols), data)
                            .expect("int column shape");
                        d.set_item(name, a.into_pyarray(py))?;
                    }
                }
                Column::Float { data, ncols } => {
                    if ncols <= 1 {
                        d.set_item(name, data.into_pyarray(py))?;
                    } else {
                        let rows = data.len() / ncols;
                        let a = numpy::ndarray::Array2::from_shape_vec((rows, ncols), data)
                            .expect("float column shape");
                        d.set_item(name, a.into_pyarray(py))?;
                    }
                }
                Column::Str { data, ncols } => {
                    if ncols <= 1 {
                        d.set_item(name, data)?;
                    } else {
                        let rows: Vec<Vec<String>> =
                            data.chunks(ncols).map(|c| c.to_vec()).collect();
                        d.set_item(name, rows)?;
                    }
                }
            }
        }
        Ok(d)
    }

    /// Parse an LS-DYNA keyword file into an editable [`PyKeywordFile`].
    ///
    /// Releases the GIL during the file read and block split.
    #[pyfunction]
    #[pyo3(signature = (path))]
    pub fn parse_keyword_file(py: Python<'_>, path: String) -> PyResult<PyKeywordFile> {
        let file_path = Path::new(&path);
        let inner = py
            .detach(|| crate::parser::parse_file_blocks(file_path))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        Ok(PyKeywordFile { inner })
    }

    // -- Phase 5: binary results (binout / d3plot) -------------------------

    use crate::results::{
        BlockArray, Binout as RustBinout, BinoutEditor as RustBinoutEditor, D3plot as RustD3plot,
        D3plotError, Data, LsdaError, ReadResult, StateBlock,
    };
    use numpy::PyReadonlyArrayDyn;

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
            ReadResult::String(s) => pyo3::types::PyString::new(py, &s).into_any(),
        })
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

        /// Read at `path` (list of segments). A leaf returns a numpy array of
        /// the channel's native dtype; a directory returns `list[str]` of child
        /// names. Empty path returns the top-level datasets.
        #[pyo3(signature = (path=Vec::new()))]
        fn read<'py>(&self, py: Python<'py>, path: Vec<String>) -> PyResult<Bound<'py, pyo3::PyAny>> {
            let segs: Vec<&str> = path.iter().map(String::as_str).collect();
            let r = py.detach(|| self.inner.read(&segs)).map_err(lsda_err)?;
            readresult_to_py(py, r)
        }

        /// Read a leaf and coerce to float64 (any numeric dtype).
        #[pyo3(signature = (path))]
        fn read_f64<'py>(&self, py: Python<'py>, path: Vec<String>) -> PyResult<Bound<'py, pyo3::PyAny>> {
            let segs: Vec<&str> = path.iter().map(String::as_str).collect();
            let v = py.detach(|| self.inner.read_f64(&segs)).map_err(lsda_err)?;
            Ok(v.into_pyarray(py).into_any())
        }

        /// Read a time-history: `{"time": float64[T], "values": float64[T], "channel": str}`.
        /// `time` is read from the sibling `time` array, or synthesized as 0..T.
        #[pyo3(signature = (path))]
        fn read_time_series<'py>(&self, py: Python<'py>, path: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
            let segs: Vec<&str> = path.iter().map(String::as_str).collect();
            let ts = py.detach(|| self.inner.read_time_series(&segs)).map_err(lsda_err)?;
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

        #[getter]
        fn num_nodes(&self) -> usize {
            self.inner.num_nodes()
        }

        #[getter]
        fn num_states(&self) -> usize {
            self.inner.num_states()
        }

        /// Simulation time of each state, as a float64 array.
        fn times<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::PyAny> {
            self.inner.times().to_vec().into_pyarray(py).into_any()
        }

        /// Deformed node coordinates at `state` (0-based) as an `(NUMNP, 3)` array.
        fn node_coordinates<'py>(&self, py: Python<'py>, state: usize) -> PyResult<Bound<'py, pyo3::PyAny>> {
            let v = py.detach(|| self.inner.node_coordinates(state)).map_err(d3_err)?;
            let rows = v.len() / 3;
            let a = numpy::ndarray::Array2::from_shape_vec((rows, 3), v)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(a.into_pyarray(py).into_any())
        }

        /// Per-node displacement magnitude at `state` as a `(NUMNP,)` array.
        fn displacement_magnitudes<'py>(&self, py: Python<'py>, state: usize) -> PyResult<Bound<'py, pyo3::PyAny>> {
            let v = py.detach(|| self.inner.displacement_magnitudes(state)).map_err(d3_err)?;
            Ok(v.into_pyarray(py).into_any())
        }

        /// Peak nodal displacement magnitude at the final state.
        fn max_displacement_final(&self, py: Python<'_>) -> PyResult<f64> {
            py.detach(|| self.inner.max_displacement_final()).map_err(d3_err)
        }

        /// Names of the result blocks present in this d3plot (any of
        /// `displacement`, `velocity`, `acceleration`, `solid`, `tshell`,
        /// `beam`, `shell`).
        fn available_blocks(&self) -> Vec<String> {
            BLOCK_NAMES
                .iter()
                .filter(|(_, b)| self.inner.block_layout(*b).is_some())
                .map(|(n, _)| (*n).to_string())
                .collect()
        }

        /// Generic result extraction: any result block across all states as an
        /// `(n_states, count, vars)` numpy array. `name` is one of the block
        /// names from [`available_blocks`]. Node blocks are `(…, 3)`; element
        /// blocks return the solver's raw packed per-entity layout — reshape by
        /// integration points/layers as needed. Raises if the block is absent.
        #[pyo3(signature = (name))]
        fn block<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, pyo3::PyAny>> {
            let b = block_from_name(name)?;
            let (data, [ns, count, vars]) = self
                .inner
                .block_data(b)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("block '{name}' is not present in this d3plot")))?;
            let shape_err = |e: numpy::ndarray::ShapeError| pyo3::exceptions::PyValueError::new_err(e.to_string());
            // Native precision: f32 for single-precision d3plots, f64 for double.
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
        #[pyo3(signature = (name))]
        fn block_layout(&self, name: &str) -> PyResult<Option<(usize, usize)>> {
            Ok(self.inner.block_layout(block_from_name(name)?))
        }

        fn __repr__(&self) -> String {
            format!("D3plot({} nodes, {} states)", self.inner.num_nodes(), self.inner.num_states())
        }
    }

    const BLOCK_NAMES: &[(&str, StateBlock)] = &[
        ("displacement", StateBlock::Displacement),
        ("velocity", StateBlock::Velocity),
        ("acceleration", StateBlock::Acceleration),
        ("solid", StateBlock::Solid),
        ("tshell", StateBlock::ThickShell),
        ("beam", StateBlock::Beam),
        ("shell", StateBlock::Shell),
    ];

    fn block_from_name(name: &str) -> PyResult<StateBlock> {
        BLOCK_NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, b)| *b)
            .ok_or_else(|| {
                let known: Vec<&str> = BLOCK_NAMES.iter().map(|(n, _)| *n).collect();
                pyo3::exceptions::PyValueError::new_err(format!("unknown block '{name}'; expected one of {known:?}"))
            })
    }

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
}

/// High-performance LS-DYNA keyword file include tree parser.
#[cfg(feature = "python")]
#[pyo3::pymodule]
pub mod _dynars {
    #[pymodule_export]
    use super::python_bindings::PyIncludeNode;

    #[pymodule_export]
    use super::python_bindings::parse_include_tree;

    #[pymodule_export]
    use super::python_bindings::PyKeywordFile;

    #[pymodule_export]
    use super::python_bindings::parse_keyword_file;

    #[pymodule_export]
    use super::python_bindings::PyBinout;

    #[pymodule_export]
    use super::python_bindings::parse_binout;

    #[pymodule_export]
    use super::python_bindings::PyD3plot;

    #[pymodule_export]
    use super::python_bindings::open_d3plot;

    #[pymodule_export]
    use super::python_bindings::PyBinoutEditor;
}
