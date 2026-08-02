//! PyO3 bindings: keyword-file marshalling.

use std::path::Path;

use pyo3::PyResult;
use pyo3::prelude::*;

// -- Phase 4: keyword-file marshalling ---------------------------------

use std::fmt::Write as _;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::Bound;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::file::ParsedFile;
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
        let schema = build_schema(&keyword, cards, repeat)?;
        let table = py.detach(|| crate::schema::parse_schema(&self.inner, &schema));
        table_to_pydict(py, table)
    }

    /// Parse a keyword using dynars' built-in library (generated from the
    /// pyDYNA field database), returning the same column dict. Errors if the
    /// keyword is not in the library.
    fn parse_builtin<'py>(&self, py: Python<'py>, keyword: String) -> PyResult<Bound<'py, PyDict>> {
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
            if self.inner.is_dirty() {
                ", edited"
            } else {
                ""
            },
        )
    }
}

/// Build a runtime [`Schema`] from the `(name, type, width, count)` card tuples
/// the Python `@keyword` layer lowers to. Shared by the per-file
/// [`PyKeywordFile`] and the deck-wide [`PyDeck`](super::deck::PyDeck) columnar
/// entry points, so there is one lowering.
pub(crate) fn build_schema(
    keyword: &str,
    cards: Vec<Vec<(String, String, usize, usize)>>,
    repeat: bool,
) -> PyResult<Schema> {
    let mut schema = Schema::new(keyword);
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
            // Python card tuples don't carry reference metadata yet; a
            // registered keyword's references stay unchecked from Python.
            c.fields.push(FieldSpec {
                name,
                ty,
                width,
                count: count.max(1),
                reference: crate::keywords::Ref::None,
            });
        }
        schema.cards.push(c);
    }
    Ok(schema)
}

/// Convert a columnar [`Table`](crate::schema::Table) into a Python dict of
/// numpy arrays (2-D for array fields) and string lists.
pub(crate) fn table_to_pydict<'py>(
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
                    let rows: Vec<Vec<String>> = data.chunks(ncols).map(|c| c.to_vec()).collect();
                    d.set_item(name, rows)?;
                }
            }
        }
    }
    Ok(d)
}

/// A column handed to [`write_keyword`], already copied out of numpy so it can
/// be formatted off the GIL.
enum WCol {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

impl WCol {
    fn len(&self) -> usize {
        match self {
            WCol::Int(v) => v.len(),
            WCol::Float(v) => v.len(),
            WCol::Str(v) => v.len(),
        }
    }
}

/// Author a single-keyword deck from columnar arrays and write it to `path` —
/// the inverse of the columnar read path (`Deck.table` / `parse_keyword`).
///
/// `columns` maps field name to a numpy `int64`/`float64` array (or a `list[str]`),
/// all the same length N; the cards are emitted in dict order, in free (comma)
/// format, straight from Rust with no per-row Python objects. Writes
/// `*KEYWORD` / `*<name>` / N card lines / `*END`. Rows are formatted in
/// parallel with the GIL released.
#[pyfunction]
#[pyo3(signature = (path, name, columns))]
pub fn write_keyword(
    py: Python<'_>,
    path: String,
    name: String,
    columns: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let mut cols: Vec<WCol> = Vec::with_capacity(columns.len());
    for (_key, val) in columns.iter() {
        let col = if let Ok(a) = val.extract::<PyReadonlyArray1<'_, i64>>() {
            WCol::Int(a.as_array().to_vec())
        } else if let Ok(a) = val.extract::<PyReadonlyArray1<'_, f64>>() {
            WCol::Float(a.as_array().to_vec())
        } else if let Ok(a) = val.extract::<PyReadonlyArray1<'_, i32>>() {
            WCol::Int(a.as_array().iter().map(|&x| x as i64).collect())
        } else {
            WCol::Str(val.extract::<Vec<String>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "each column must be an int64/float64 numpy array or a list[str]",
                )
            })?)
        };
        cols.push(col);
    }
    if cols.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no columns given"));
    }
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all columns must have the same length",
        ));
    }

    let bytes = py.detach(|| emit_free_deck(&name, &cols, n));
    std::fs::write(Path::new(&path), bytes)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
}

/// Format `*KEYWORD` / `*<name>` / N free-format card lines / `*END`, building
/// the body in parallel over row chunks.
fn emit_free_deck(name: &str, cols: &[WCol], n: usize) -> Vec<u8> {
    const CHUNK: usize = 65_536;
    let parts: Vec<String> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let (lo, hi) = (c * CHUNK, ((c + 1) * CHUNK).min(n));
            let mut s = String::with_capacity((hi - lo) * 40);
            for i in lo..hi {
                for (k, col) in cols.iter().enumerate() {
                    if k > 0 {
                        s.push(',');
                    }
                    match col {
                        WCol::Int(v) => {
                            let _ = write!(s, "{}", v[i]);
                        }
                        WCol::Float(v) => {
                            let _ = write!(s, "{}", v[i]);
                        }
                        WCol::Str(v) => s.push_str(&v[i]),
                    }
                }
                s.push('\n');
            }
            s
        })
        .collect();

    let body_len: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = String::with_capacity(name.len() + body_len + 16);
    out.push_str("*KEYWORD\n*");
    out.push_str(name);
    out.push('\n');
    for p in &parts {
        out.push_str(p);
    }
    out.push_str("*END\n");
    out.into_bytes()
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
