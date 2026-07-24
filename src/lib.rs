pub mod bulk;
pub mod include_tree;
pub mod keyword;
pub mod parser;
pub mod testgen;
pub mod typed;

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

        if !file_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("File not found: {}", path),
            ));
        }

        let result = crate::include_tree::build_include_tree(file_path);

        match result {
            Ok(root) => Ok(rust_to_py(&root)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    // -- Phase 4: keyword-file marshalling ---------------------------------

    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
    use pyo3::types::{PyDict, PyList};
    use pyo3::Bound;

    use crate::keyword::ParsedFile;
    use crate::parser::Keyword;

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

        /// `*NODE` data as `(ids: int64[N], coords: float64[N, 3])`, zero-copy.
        fn nodes<'py>(
            &self,
            py: Python<'py>,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>) {
            let n = py.detach(|| crate::bulk::parse_nodes(&self.inner));
            let count = n.ids.len();
            let coords = numpy::ndarray::Array2::from_shape_vec((count, 3), n.coords)
                .expect("coords length is 3 * node count");
            (n.ids.into_pyarray(py), coords.into_pyarray(py))
        }

        /// `*ELEMENT_SHELL` as `(eids, pids, nodes: int64[N, 4])`.
        fn elements_shell<'py>(
            &self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray2<i64>>,
        ) {
            let e = py.detach(|| crate::bulk::parse_element_shell(&self.inner));
            elements_to_py(py, e)
        }

        /// `*ELEMENT_SOLID` as `(eids, pids, nodes: int64[N, 8])`.
        fn elements_solid<'py>(
            &self,
            py: Python<'py>,
        ) -> (
            Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray1<i64>>,
            Bound<'py, PyArray2<i64>>,
        ) {
            let e = py.detach(|| crate::bulk::parse_element_solid(&self.inner));
            elements_to_py(py, e)
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

        /// Rewrite all `*NODE` blocks from a new `(N, 3)` coordinate array.
        fn set_node_coords(&mut self, coords: PyReadonlyArray2<f64>) -> PyResult<()> {
            let arr = coords.as_array();
            let flat: Vec<f64> = arr.iter().copied().collect();
            crate::bulk::update_node_coords(&mut self.inner, &flat)
                .map_err(pyo3::exceptions::PyValueError::new_err)
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

    fn elements_to_py<'py>(
        py: Python<'py>,
        e: crate::bulk::ElementArrays,
    ) -> (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray2<i64>>,
    ) {
        let count = e.eids.len();
        let npe = e.nodes_per_elem;
        let conn = numpy::ndarray::Array2::from_shape_vec((count, npe), e.nodes)
            .expect("connectivity length is count * nodes_per_elem");
        (
            e.eids.into_pyarray(py),
            e.pids.into_pyarray(py),
            conn.into_pyarray(py),
        )
    }

    /// Parse an LS-DYNA keyword file into an editable [`PyKeywordFile`].
    ///
    /// Releases the GIL during the file read and block split.
    #[pyfunction]
    #[pyo3(signature = (path))]
    pub fn parse_keyword_file(py: Python<'_>, path: String) -> PyResult<PyKeywordFile> {
        let file_path = Path::new(&path);
        if !file_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                path
            )));
        }
        let inner = py
            .detach(|| crate::parser::parse_file_blocks(file_path))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        Ok(PyKeywordFile { inner })
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
}
