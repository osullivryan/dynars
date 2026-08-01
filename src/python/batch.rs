//! PyO3 bindings: `Workspace` — batch-parse/validate many decks that share
//! `*INCLUDE`s against one cache.

use std::collections::HashMap;

use pyo3::PyResult;
use pyo3::prelude::*;

use super::deck::PyDeck;
use super::validate::{PyReport, PyRule, report_to_py};
use crate::batch::Workspace;

/// An in-process batch context: parse and validate many decks that share
/// `*INCLUDE`s against one shared cache, so common files (mesh, materials) are
/// read, parsed, and indexed **once** no matter how many decks include them.
///
/// ```python
/// import dynars
/// ws = dynars.Workspace()
/// decks = ws.parse_decks(["variant_a/main.k", "variant_b/main.k"])
/// reports = ws.validate_decks(decks, [
///     dynars.Rule.references_resolve(),
///     dynars.Rule.duplicate_ids(),
/// ])
/// print(ws.stats())  # {'files_parsed': ..., 'files_reused': ..., ...}
/// ```
///
/// The decks handed back are ordinary `Deck`s — validate or navigate them
/// individually too; a deck from a workspace reuses the shared indices whether
/// you call `validate_decks` or its own `.validate(...)`.
#[pyclass(name = "Workspace")]
pub struct PyWorkspace {
    ws: Workspace,
}

#[pymethods]
impl PyWorkspace {
    #[new]
    fn new() -> Self {
        Self {
            ws: Workspace::new(),
        }
    }

    /// Parse one deck (root + all includes), reusing any file this workspace has
    /// already read. Returns a navigable `Deck`.
    fn parse_deck(&self, py: Python<'_>, path: String) -> PyResult<PyDeck> {
        let deck = py
            .detach(|| self.ws.parse_deck(std::path::Path::new(&path)))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(PyDeck::from_deck(deck))
    }

    /// Parse several decks in one batch, sharing all file work across them.
    /// Returns a list of `Deck`s in input order; raises `RuntimeError` naming the
    /// first root that fails to parse.
    fn parse_decks(&self, py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<PyDeck>> {
        let results = py.detach(|| self.ws.parse_decks(&paths));
        let mut decks = Vec::with_capacity(results.len());
        for (root, res) in results {
            match res {
                Ok(d) => decks.push(PyDeck::from_deck(d)),
                Err(e) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "{}: {e}",
                        root.display()
                    )));
                }
            }
        }
        Ok(decks)
    }

    /// Validate several decks in parallel against the shared cache. Returns one
    /// `Report` per deck, in order. Warms the shared definition index first, then
    /// runs `rules` over every deck concurrently — a shared file's id and
    /// connectivity indices are built once, not per deck.
    fn validate_decks(
        &self,
        py: Python<'_>,
        decks: Vec<Py<PyDeck>>,
        rules: Vec<PyRule>,
    ) -> Vec<PyReport> {
        let rs: Vec<crate::validate::Rule> = rules.into_iter().map(|r| r.inner).collect();
        // Borrow each deck handle, then validate over the borrowed cores. The
        // work is pure-Rust and internally parallel (rayon); we keep the GIL
        // because the deck handles are GIL-bound.
        let borrows: Vec<PyRef<'_, PyDeck>> = decks.iter().map(|d| d.borrow(py)).collect();
        let refs: Vec<&crate::deck::Deck> = borrows.iter().map(|b| b.inner()).collect();
        self.ws
            .validate_refs(&refs, rs)
            .into_iter()
            .map(report_to_py)
            .collect()
    }

    /// Cache stats as a dict: `files_parsed` / `files_reused` (disk reads vs.
    /// cache hits) and `def_indices_built` / `ref_indices_built` (distinct files
    /// whose definition / connectivity index was extracted — a shared file counts
    /// once).
    fn stats(&self) -> HashMap<&'static str, usize> {
        let s = self.ws.stats();
        HashMap::from([
            ("files_parsed", s.files_parsed),
            ("files_reused", s.files_reused),
            ("def_indices_built", s.def_indices_built),
            ("ref_indices_built", s.ref_indices_built),
        ])
    }

    fn __repr__(&self) -> String {
        let s = self.ws.stats();
        format!(
            "Workspace({} files read, {} reuses)",
            s.files_parsed, s.files_reused
        )
    }
}
