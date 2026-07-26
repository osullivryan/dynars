//! PyO3 bindings: `*INCLUDE` tree.

use std::path::Path;

use pyo3::PyResult;
use pyo3::prelude::*;

use crate::include::IncludeNode as RustIncludeNode;

#[pyclass(name = "IncludeNode", skip_from_py_object)]
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
            self.path,
            kind_str,
            self.byte_count,
            self.children.len(),
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

    let result = crate::include::build_include_tree(file_path);

    match result {
        Ok(root) => Ok(rust_to_py(&root)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
}
