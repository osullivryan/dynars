//! PyO3 bindings: the `Deck` handle — parse once, validate + navigate.

use std::collections::HashMap;
use std::path::Path;

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use super::validate::{PyReport, PyRule, report_to_py};
use crate::validate;

// ── Deck: parse once, validate + navigate off one handle ─────────────────
use crate::keywords::{EntityKind, canonical_base};
use crate::model;

/// Convert a typed [`model::Value`] into the matching Python scalar.
fn value_to_py(py: Python<'_>, v: model::Value) -> Bound<'_, pyo3::PyAny> {
    match v {
        model::Value::Int(i) => i.into_pyobject(py).unwrap().into_any(),
        model::Value::Float(f) => f.into_pyobject(py).unwrap().into_any(),
        model::Value::Str(s) => s.into_pyobject(py).unwrap().into_any(),
    }
}

/// Coerce a Python `str` / `int` / `float` into the text written into a field.
/// A `str` is written verbatim (full control over the exact column text); a
/// `float` uses the shorter of plain and scientific notation, so a large/small
/// magnitude like `2.1e11` stays compact (`2.1e11`) and fits a fixed column
/// instead of expanding to `210000000000`.
fn coerce_value(v: &Bound<'_, pyo3::PyAny>) -> PyResult<String> {
    if let Ok(s) = v.extract::<String>() {
        Ok(s)
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(i.to_string())
    } else if let Ok(f) = v.extract::<f64>() {
        let plain = format!("{f}");
        let sci = format!("{f:e}");
        Ok(if sci.len() < plain.len() { sci } else { plain })
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "value must be str, int, or float",
        ))
    }
}

fn edit_name(e: crate::parser::FieldEdit) -> String {
    match e {
        crate::parser::FieldEdit::InPlace => "in_place".to_string(),
        crate::parser::FieldEdit::Reflowed => "reflowed".to_string(),
    }
}

/// Locate `name` on `(file, block)` and write `value` in place. Returns
/// `"in_place"` / `"reflowed"`, or `None` if the field isn't found (or the
/// keyword has no schema). One call — the read/`&mut` borrow split the Rust API
/// exposes is handled here.
fn apply_set_field(
    deck: &Py<PyDeck>,
    py: Python<'_>,
    file: usize,
    block: usize,
    name: &str,
    value: &Bound<'_, pyo3::PyAny>,
) -> PyResult<Option<String>> {
    let s = coerce_value(value)?;
    let mut d = deck.borrow_mut(py);
    let Some(loc) = model::locate_field(&d.deck, file, block, name) else {
        return Ok(None);
    };
    Ok(d.deck.set_field(&loc, &s).map(edit_name))
}

/// A parsed LS-DYNA deck (root + all includes). Parse once with
/// [`parse_deck`], then validate (`validate`) and navigate
/// (`part`, `material`, …) off the same object — no second parse. The
/// resolution indices are built lazily on first use.
#[pyclass(name = "Deck")]
pub struct PyDeck {
    deck: crate::deck::Deck,
}

impl PyDeck {
    /// Wrap an already-parsed core [`Deck`] — the seam the batch
    /// [`Workspace`](crate::batch::Workspace) bindings use to hand back decks
    /// that carry its shared cache.
    pub(crate) fn from_deck(deck: crate::deck::Deck) -> Self {
        Self { deck }
    }

    /// Borrow the underlying core [`Deck`] (for batch validation over borrowed
    /// deck handles).
    pub(crate) fn inner(&self) -> &crate::deck::Deck {
        &self.deck
    }
}

#[pymethods]
impl PyDeck {
    #[new]
    #[pyo3(signature = (path))]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let deck = py
            .detach(|| crate::deck::parse_deck(std::path::Path::new(&path)))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(Self { deck })
    }

    /// Run a set of rules over this deck, reusing the parse. No default
    /// rule set — pass the rules you want (e.g. `Rule.references_resolve()`).
    fn validate(&self, py: Python<'_>, rules: Vec<PyRule>) -> PyReport {
        let rs: Vec<validate::Rule> = rules.into_iter().map(|r| r.inner).collect();
        report_to_py(py.detach(move || self.deck.validate(rs)))
    }

    /// The *PART with this id, or `None` if none is defined. Ids are global
    /// (post-`*INCLUDE_TRANSFORM`); the sign is ignored, so `|id|` also matches.
    fn part(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Part, id)
    }
    /// The *MAT with this id, or `None` if none is defined. Ids are global
    /// (post-`*INCLUDE_TRANSFORM`); the sign is ignored, so `|id|` also matches.
    fn material(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Material, id)
    }
    /// The *SECTION with this id, or `None` if none is defined. Ids are global
    /// (post-`*INCLUDE_TRANSFORM`); the sign is ignored, so `|id|` also matches.
    fn section(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Section, id)
    }
    /// The *DEFINE_CURVE with this id, or `None` if none is defined. Ids are
    /// global (post-`*INCLUDE_TRANSFORM`); the sign is ignored, so `|id|` also
    /// matches.
    fn curve(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Curve, id)
    }

    /// Every part in the deck (enumerate, don't guess ids).
    fn parts(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Part)
    }
    /// Every *MAT in the deck (enumerate, don't guess ids).
    fn materials(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Material)
    }
    /// Every *SECTION in the deck (enumerate, don't guess ids).
    fn sections(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Section)
    }
    /// Every *DEFINE_CURVE in the deck (enumerate, don't guess ids).
    fn curves(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Curve)
    }

    /// `(kind, count)` of defined ids, most-numerous first.
    fn definition_counts(&self) -> Vec<(String, usize)> {
        self.deck
            .definition_counts()
            .into_iter()
            .map(|(k, n)| (format!("{k:?}"), n))
            .collect()
    }

    /// Bulk **columnar** read of every occurrence of `keyword` across the whole
    /// deck (root + includes) using the built-in library, as a dict of numpy
    /// arrays (numeric fields) and string lists. The fast path alongside
    /// `part`/`material`/… navigation — the deck is the one columnar entry,
    /// include-aware (unlike the per-file `KeywordFile`). Raises `KeyError` if
    /// the keyword isn't in the built-in library (use `table_with`).
    fn table<'py>(&self, py: Python<'py>, keyword: String) -> PyResult<Bound<'py, PyDict>> {
        let schema = crate::keywords::schema(&keyword).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "'{}' is not in the built-in keyword library",
                keyword
            ))
        })?;
        let table = py.detach(|| crate::schema::parse_schema_files(&self.deck.files, &schema));
        super::keyword::table_to_pydict(py, table)
    }

    /// Bulk columnar read across the whole deck against a user-defined schema —
    /// the escape hatch for a keyword not in the built-in library. `cards` is a
    /// list of cards, each a list of `(name, type, width, count)` field tuples;
    /// `type` is "int" | "float" | "str".
    #[pyo3(signature = (keyword, cards, repeat=false))]
    fn table_with<'py>(
        &self,
        py: Python<'py>,
        keyword: String,
        cards: Vec<Vec<(String, String, usize, usize)>>,
        repeat: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let schema = super::keyword::build_schema(&keyword, cards, repeat)?;
        let table = py.detach(|| crate::schema::parse_schema_files(&self.deck.files, &schema));
        super::keyword::table_to_pydict(py, table)
    }

    /// Register a user schema for a keyword the built-in library doesn't cover,
    /// so navigation (`keywords`, `part`, …) gets named, typed field access for
    /// it. `cards` is a list of cards, each a list of `(name, type, width,
    /// count)` field tuples; `type` is "int" | "float" | "str". Keyed by
    /// canonical base — registering the same base twice replaces it.
    #[pyo3(signature = (keyword, cards, repeat=false))]
    fn register_schema(
        &mut self,
        keyword: String,
        cards: Vec<Vec<(String, String, usize, usize)>>,
        repeat: bool,
    ) -> PyResult<()> {
        let schema = super::keyword::build_schema(&keyword, cards, repeat)?;
        self.deck.register_schema(schema);
        Ok(())
    }

    /// Every occurrence of `keyword` across the whole deck (root + includes), as
    /// `Keyword` handles — matched on the canonical base, so `SECTION_SHELL` also
    /// matches `SECTION_SHELL_TITLE`. The occurrence-navigation counterpart to
    /// the columnar `table`; unlike `part`/`material`/… it isn't limited to
    /// definition entities.
    fn keywords(slf: Py<Self>, py: Python<'_>, keyword: String) -> Vec<PyKeyword> {
        let base = canonical_base(&keyword);
        let sites: Vec<(usize, usize)> = {
            let d = slf.borrow(py);
            d.deck
                .files
                .iter()
                .enumerate()
                .flat_map(|(fi, f)| {
                    let base = base.clone();
                    (0..f.blocks.len())
                        .filter(move |&bi| canonical_base(f.keyword_name(&f.blocks[bi])) == base)
                        .map(move |bi| (fi, bi))
                })
                .collect()
        };
        sites
            .into_iter()
            .map(|(file, block)| PyKeyword {
                deck: slf.clone_ref(py),
                file,
                block,
            })
            .collect()
    }

    /// The deck's parsed files as `File` handles — the root first, then each
    /// `*INCLUDE`d file in include order. File-first navigation: pick a file,
    /// then read/edit its keywords.
    fn files(slf: Py<Self>, py: Python<'_>) -> Vec<PyFile> {
        let n = slf.borrow(py).deck.files.len();
        (0..n)
            .map(|file| PyFile {
                deck: slf.clone_ref(py),
                file,
            })
            .collect()
    }

    /// The first parsed file whose path ends with `suffix` (e.g. `"sub.k"` or
    /// `"mesh/part.k"`), as a `File` — the way into a specific include. `None`
    /// if nothing matches.
    fn file(slf: Py<Self>, py: Python<'_>, suffix: String) -> Option<PyFile> {
        let idx = {
            let d = slf.borrow(py);
            (0..d.deck.files.len()).find(|&i| d.deck.files[i].path.ends_with(&suffix))
        };
        idx.map(|file| PyFile {
            deck: slf.clone_ref(py),
            file,
        })
    }

    fn __repr__(&self) -> String {
        format!("Deck({} files)", self.deck.files.len())
    }
}

/// Parse a deck (root + all includes) once and return a navigable [`PyDeck`].
#[pyfunction]
#[pyo3(signature = (path))]
pub fn parse_deck(py: Python<'_>, path: String) -> PyResult<PyDeck> {
    PyDeck::new(py, path)
}

/// A handle to one entity: typed field access, source location, and
/// reference-following. Keeps its [`PyDeck`] alive.
#[pyclass(name = "Entity")]
pub struct PyEntity {
    deck: Py<PyDeck>,
    kind: EntityKind,
    #[pyo3(get)]
    id: i64,
    file: usize,
    block: usize,
}

impl PyEntity {
    fn make(deck: Py<PyDeck>, py: Python<'_>, kind: EntityKind, id: i64) -> Option<PyEntity> {
        let (file, block) = {
            let d = deck.borrow(py);
            model::site_of(d.deck.site_index(), kind, id)?
        };
        Some(PyEntity {
            deck: deck.clone_ref(py),
            kind,
            id,
            file,
            block,
        })
    }
    fn all(deck: Py<PyDeck>, py: Python<'_>, kind: EntityKind) -> Vec<PyEntity> {
        let sites: Vec<(i64, usize, usize)> = {
            let d = deck.borrow(py);
            d.deck
                .site_index()
                .iter()
                .filter(|((k, _), _)| *k == kind)
                .map(|(&(_, id), &(f, b))| (id, f, b))
                .collect()
        };
        sites
            .into_iter()
            .map(|(id, file, block)| PyEntity {
                deck: deck.clone_ref(py),
                kind,
                id,
                file,
                block,
            })
            .collect()
    }
    fn ref_to(slf: PyRef<'_, Self>, py: Python<'_>, kind: EntityKind) -> Option<PyEntity> {
        let id = {
            let d = slf.deck.borrow(py);
            let id = model::first_ref_to(&d.deck, slf.file, slf.block, kind)?;
            // The ref is written in the file's local ids; resolve it globally
            // (a no-op outside an *INCLUDE_TRANSFORM) — parity with Rust nav.
            d.deck
                .transform_of(slf.file)
                .map_or(id, |t| t.apply(id, kind))
        };
        PyEntity::make(slf.deck.clone_ref(py), py, kind, id)
    }
}

#[pymethods]
impl PyEntity {
    /// The entity kind (e.g. `"Part"`, `"Material"`, `"Section"`, `"Curve"`).
    #[getter]
    fn kind(&self) -> String {
        format!("{:?}", self.kind)
    }
    /// The full `*KEYWORD` name of the block that defines this entity.
    #[getter]
    fn keyword(&self, py: Python<'_>) -> String {
        let d = self.deck.borrow(py);
        let f = &d.deck.files[self.file];
        f.keyword_name(&f.blocks[self.block]).to_string()
    }
    /// The include file this entity is defined in.
    #[getter]
    fn file(&self, py: Python<'_>) -> String {
        let d = self.deck.borrow(py);
        d.deck.files[self.file].path.display().to_string()
    }
    /// 1-based line of the entity's `*KEYWORD` line (jump-to location).
    #[getter]
    fn line(&self, py: Python<'_>) -> usize {
        let d = self.deck.borrow(py);
        let f = &d.deck.files[self.file];
        let b = &f.blocks[self.block];
        1 + f.src()[..b.name_start]
            .iter()
            .filter(|&&c| c == b'\n')
            .count()
    }
    /// The effective `*INCLUDE_TRANSFORM` offsets applied to this entity's file
    /// (composed down the include chain) as a dict `{"idnoff": …, "ideoff": …}`,
    /// or `None` if it sits in the root or a plain `*INCLUDE`. These are the
    /// shifts that turn the file-local ids into the global ones `id` reports.
    #[getter]
    fn offsets(&self, py: Python<'_>) -> Option<HashMap<&'static str, i64>> {
        let d = self.deck.borrow(py);
        let t = d.deck.transform_of(self.file).copied()?;
        Some(HashMap::from([
            ("idnoff", t.idnoff),
            ("ideoff", t.ideoff),
            ("idpoff", t.idpoff),
            ("idmoff", t.idmoff),
            ("idsoff", t.idsoff),
            ("idfoff", t.idfoff),
            ("iddoff", t.iddoff),
            ("idroff", t.idroff),
        ]))
    }

    /// Read a field by name (case-insensitive) → int / float / str.
    fn field<'py>(&self, py: Python<'py>, name: String) -> Option<Bound<'py, pyo3::PyAny>> {
        let d = self.deck.borrow(py);
        let v = model::entity_field(&d.deck, self.file, self.block, &name)?;
        Some(value_to_py(py, v))
    }

    /// Overwrite a named field in place, preserving every other byte of the
    /// deck. Returns `"in_place"`, or `"reflowed"` if the value overflowed its
    /// fixed column (that one card re-emitted in free format), or `None` if the
    /// field isn't found. Realise the change with the owning file's `write` /
    /// `to_bytes` (`deck.file(...)` / `deck.files()`).
    fn set_field(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: String,
        value: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Option<String>> {
        apply_set_field(&slf.deck, py, slf.file, slf.block, &name, &value)
    }
    /// Follow the reference in field `name` to the entity it points at.
    fn reference(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> Option<PyEntity> {
        let (r, id, transform) = {
            let d = slf.deck.borrow(py);
            let (r, id) = model::ref_field(&d.deck, slf.file, slf.block, &name)?;
            (r, id, d.deck.transform_of(slf.file).copied())
        };
        // Shift the local ref id to global per candidate kind before lookup.
        let logical = |k: EntityKind| transform.map_or(id, |t| t.apply(id, k));
        let deck = slf.deck.clone_ref(py);
        match r {
            crate::keywords::Ref::None => None,
            crate::keywords::Ref::To(k) => PyEntity::make(deck, py, k, logical(k)),
            crate::keywords::Ref::AnyOf(ks) => ks
                .iter()
                .find_map(|k| PyEntity::make(deck.clone_ref(py), py, *k, logical(*k))),
        }
    }
    /// Follow this entity's first field that references a *MAT to that
    /// material, or `None` if there is no such field or it doesn't resolve.
    fn material(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Material)
    }
    /// Follow this entity's first field that references a *SECTION to that
    /// section, or `None` if there is no such field or it doesn't resolve.
    fn section(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Section)
    }
    /// Follow this entity's first field that references an *EOS to that equation
    /// of state, or `None` if there is no such field or it doesn't resolve.
    fn eos(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Eos)
    }
    /// Follow this entity's first field that references a *HOURGLASS to that
    /// hourglass definition, or `None` if there is no such field or it doesn't
    /// resolve.
    fn hourglass(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Hourglass)
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Entity({} {} [{}])", self.kind(), self.id, self.keyword(py))
    }
}

/// A keyword occurrence — one `*KEYWORD` block — reached by name
/// (`Deck.keywords`) or through a file (`File.keywords`). Read fields, and edit
/// one in place with `set_field`. Keeps its [`PyDeck`] alive.
#[pyclass(name = "Keyword")]
pub struct PyKeyword {
    deck: Py<PyDeck>,
    file: usize,
    block: usize,
}

#[pymethods]
impl PyKeyword {
    /// The full `*KEYWORD` name of this occurrence (e.g. `SECTION_SHELL_TITLE`).
    #[getter]
    fn name(&self, py: Python<'_>) -> String {
        let d = self.deck.borrow(py);
        let f = &d.deck.files[self.file];
        f.keyword_name(&f.blocks[self.block]).to_string()
    }
    /// The include file this occurrence lives in.
    #[getter]
    fn file(&self, py: Python<'_>) -> String {
        let d = self.deck.borrow(py);
        d.deck.files[self.file].path.display().to_string()
    }
    /// 1-based line of this occurrence's `*KEYWORD` line (jump-to location).
    #[getter]
    fn line(&self, py: Python<'_>) -> usize {
        let d = self.deck.borrow(py);
        let f = &d.deck.files[self.file];
        crate::schema::block_line(f, &f.blocks[self.block])
    }
    /// Read a field by name (case-insensitive) → int / float / str. Honours a
    /// user schema registered with `register_schema`.
    fn field<'py>(&self, py: Python<'py>, name: String) -> Option<Bound<'py, pyo3::PyAny>> {
        let d = self.deck.borrow(py);
        let v = model::read_field(&d.deck, self.file, self.block, &name)?;
        Some(value_to_py(py, v))
    }
    /// Overwrite a named field in place, preserving every other byte of the
    /// deck. Returns `"in_place"` / `"reflowed"`, or `None` if the field isn't
    /// found. Persist via the owning file's `write` / `to_bytes`.
    fn set_field(
        &self,
        py: Python<'_>,
        name: String,
        value: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Option<String>> {
        apply_set_field(&self.deck, py, self.file, self.block, &name, &value)
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Keyword({} [{}])", self.name(py), self.file(py))
    }
}

/// One parsed file in a deck — the root or one `*INCLUDE` instance. Lists its
/// keywords (file-first navigation) and reads/writes its (possibly edited)
/// bytes. Keeps its [`PyDeck`] alive.
#[pyclass(name = "File")]
pub struct PyFile {
    deck: Py<PyDeck>,
    file: usize,
}

#[pymethods]
impl PyFile {
    /// This file's path — the resolved `*INCLUDE` path, or the root deck path.
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        self.deck.borrow(py).deck.files[self.file].path.display().to_string()
    }
    /// This file's index in the deck (`0` is the root).
    #[getter]
    fn index(&self) -> usize {
        self.file
    }
    /// The keyword occurrences in this file as `Keyword` handles. With `name`,
    /// only occurrences of that keyword (canonical-base match); without it,
    /// every block in file order.
    #[pyo3(signature = (name=None))]
    fn keywords(slf: PyRef<'_, Self>, py: Python<'_>, name: Option<String>) -> Vec<PyKeyword> {
        let file = slf.file;
        let want = name.as_deref().map(canonical_base);
        let blocks: Vec<usize> = {
            let d = slf.deck.borrow(py);
            let f = &d.deck.files[file];
            (0..f.blocks.len())
                .filter(|&bi| match &want {
                    Some(base) => canonical_base(f.keyword_name(&f.blocks[bi])) == *base,
                    None => true,
                })
                .collect()
        };
        blocks
            .into_iter()
            .map(|block| PyKeyword {
                deck: slf.deck.clone_ref(py),
                file,
                block,
            })
            .collect()
    }
    /// Whether this file has a pending edit.
    #[getter]
    fn dirty(&self, py: Python<'_>) -> bool {
        self.deck.borrow(py).deck.files[self.file].is_dirty()
    }
    /// The (possibly edited) file contents as bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.deck.borrow(py).deck.files[self.file].to_bytes())
    }
    /// Write the (possibly edited) file to `path`.
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        self.deck.borrow(py).deck.files[self.file]
            .write(Path::new(&path))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
    }
    /// Low-level, schema-free field write: overwrite `(block, row, col)` passing
    /// the fields' fixed column `widths` (e.g. `[10]*8`). For keywords dynars
    /// ships no schema for; otherwise prefer `Keyword.set_field`. Returns
    /// `"in_place"` / `"reflowed"`, or `None` if the card/field is out of range.
    #[pyo3(signature = (block, row, col, widths, value))]
    fn set_field(
        &self,
        py: Python<'_>,
        block: usize,
        row: usize,
        col: usize,
        widths: Vec<usize>,
        value: Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Option<String>> {
        let s = coerce_value(&value)?;
        let mut d = self.deck.borrow_mut(py);
        Ok(d.deck.files[self.file]
            .set_field(block, row, col, &widths, &s)
            .map(edit_name))
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("File('{}'{})", self.path(py), if self.dirty(py) { ", edited" } else { "" })
    }
}
