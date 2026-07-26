//! PyO3 bindings: the `Deck` handle — parse once, validate + navigate.

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::validate::{PyReport, PyRule, report_to_py};
use crate::validate;

// ── Deck: parse once, validate + navigate off one handle ─────────────────
use crate::keywords::EntityKind;
use crate::model;

/// A parsed LS-DYNA deck (root + all includes). Parse once with
/// [`parse_deck`], then validate (`validate`) and navigate
/// (`part`, `material`, …) off the same object — no second parse. The
/// resolution indices are built lazily on first use.
#[pyclass(name = "Deck")]
pub struct PyDeck {
    deck: crate::deck::Deck,
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

    fn part(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Part, id)
    }
    fn material(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Material, id)
    }
    fn section(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Section, id)
    }
    fn curve(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<PyEntity> {
        PyEntity::make(slf, py, EntityKind::Curve, id)
    }

    /// Every part in the deck (enumerate, don't guess ids).
    fn parts(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Part)
    }
    fn materials(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Material)
    }
    fn sections(slf: Py<Self>, py: Python<'_>) -> Vec<PyEntity> {
        PyEntity::all(slf, py, EntityKind::Section)
    }
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
            model::first_ref_to(&d.deck, slf.file, slf.block, kind)?
        };
        PyEntity::make(slf.deck.clone_ref(py), py, kind, id)
    }
}

#[pymethods]
impl PyEntity {
    #[getter]
    fn kind(&self) -> String {
        format!("{:?}", self.kind)
    }
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
    /// Read a field by name (case-insensitive) → int / float / str.
    fn field<'py>(&self, py: Python<'py>, name: String) -> Option<Bound<'py, pyo3::PyAny>> {
        let d = self.deck.borrow(py);
        let v = model::entity_field(&d.deck, self.file, self.block, &name)?;
        Some(match v {
            model::Value::Int(i) => i.into_pyobject(py).unwrap().into_any(),
            model::Value::Float(f) => f.into_pyobject(py).unwrap().into_any(),
            model::Value::Str(s) => s.into_pyobject(py).unwrap().into_any(),
        })
    }
    /// Follow the reference in field `name` to the entity it points at.
    fn reference(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> Option<PyEntity> {
        let (r, id) = {
            let d = slf.deck.borrow(py);
            model::ref_field(&d.deck, slf.file, slf.block, &name)?
        };
        let deck = slf.deck.clone_ref(py);
        match r {
            crate::keywords::Ref::None => None,
            crate::keywords::Ref::To(k) => PyEntity::make(deck, py, k, id),
            crate::keywords::Ref::AnyOf(ks) => ks
                .iter()
                .find_map(|k| PyEntity::make(deck.clone_ref(py), py, *k, id)),
        }
    }
    fn material(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Material)
    }
    fn section(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Section)
    }
    fn eos(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Eos)
    }
    fn hourglass(slf: PyRef<'_, Self>, py: Python<'_>) -> Option<PyEntity> {
        PyEntity::ref_to(slf, py, EntityKind::Hourglass)
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Entity({} {} [{}])", self.kind(), self.id, self.keyword(py))
    }
}
