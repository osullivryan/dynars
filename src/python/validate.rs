//! PyO3 bindings: typed validation rules, predicates, findings.

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::prelude::*;

// ── Deck validation: typed rules, combinators, file scope ────────────────
use crate::validate;

fn py_to_value(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<validate::Value> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(validate::Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(validate::Value::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(validate::Value::Str(s));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "rule value must be int, float, or str",
    ))
}

/// Normalize a keyword name for a rule. Accepts any non-empty name so rules can
/// target keywords the built-in library doesn't cover — vendor / newer / ones
/// you've described via `Deck.register_schema`. (Matching is case-insensitive on
/// the canonical base, so the original text is fine; only an empty name is an
/// error.)
fn check_keyword(kw: &str) -> PyResult<String> {
    let trimmed = kw.trim();
    if trimmed.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keyword name must not be empty",
        ));
    }
    Ok(trimmed.to_string())
}

/// A boolean predicate tree over card fields (tier 2). Evaluated in Rust.
#[pyclass(name = "Predicate", from_py_object)]
#[derive(Clone)]
pub struct PyPredicate {
    inner: validate::Expr,
}

#[pymethods]
impl PyPredicate {
    /// `field <cmp> value`.
    #[staticmethod]
    fn field(field: String, cmp: validate::Cmp, value: Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: validate::Expr::field(field, cmp, py_to_value(&value)?),
        })
    }
    /// All sub-predicates must hold (logical AND).
    #[staticmethod]
    fn all_(preds: Vec<PyPredicate>) -> Self {
        Self {
            inner: validate::Expr::all(preds.into_iter().map(|p| p.inner)),
        }
    }
    /// Any sub-predicate holds (logical OR).
    #[staticmethod]
    fn any_(preds: Vec<PyPredicate>) -> Self {
        Self {
            inner: validate::Expr::any(preds.into_iter().map(|p| p.inner)),
        }
    }
    /// Negation.
    #[staticmethod]
    fn not_(pred: PyPredicate) -> Self {
        Self {
            inner: validate::Expr::not(pred.inner),
        }
    }
}

/// A built-in declarative rule. Constructed in Python, executed in Rust.
#[pyclass(name = "Rule", from_py_object)]
#[derive(Clone)]
pub struct PyRule {
    pub(super) inner: validate::Rule,
}

#[pymethods]
impl PyRule {
    #[staticmethod]
    fn keyword_forbidden(keyword: String) -> PyResult<Self> {
        Ok(Self {
            inner: validate::Rule::keyword_forbidden(check_keyword(&keyword)?),
        })
    }
    #[staticmethod]
    fn field_forbidden_values(
        keyword: String,
        field: String,
        values: Vec<Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<Self> {
        let vals: PyResult<Vec<_>> = values.iter().map(py_to_value).collect();
        Ok(Self {
            inner: validate::Rule::field_forbidden_values(check_keyword(&keyword)?, field, vals?),
        })
    }
    #[staticmethod]
    #[pyo3(signature = (keyword, require, when=None))]
    fn field_required(
        keyword: String,
        require: PyPredicate,
        when: Option<PyPredicate>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: validate::Rule::field_required(
                check_keyword(&keyword)?,
                when.map(|w| w.inner),
                require.inner,
            ),
        })
    }
    #[staticmethod]
    fn include_missing() -> Self {
        Self {
            inner: validate::Rule::include_missing(),
        }
    }
    /// Cross-keyword referential integrity: every id reference resolves
    /// (PART.mid → *MAT, *LOAD.lcid → *DEFINE_CURVE, …). Does not check
    /// element connectivity.
    #[staticmethod]
    fn references_resolve() -> Self {
        Self {
            inner: validate::Rule::references_resolve(),
        }
    }
    /// As `references_resolve`, and additionally checks that every element's
    /// nodes are defined. Heavy on large meshes.
    #[staticmethod]
    fn references_resolve_with_connectivity() -> Self {
        Self {
            inner: validate::Rule::references_resolve_with_connectivity(),
        }
    }
    /// No two labelled definition entities of the same kind share an id (two
    /// *PART pid=5, duplicate *MAT/*SET/*SECTION/*DEFINE_CURVE ids, …). Compared
    /// on logical ids, so *INCLUDE_TRANSFORM instances don't collide.
    #[staticmethod]
    fn duplicate_ids() -> Self {
        Self {
            inner: validate::Rule::duplicate_ids(),
        }
    }
    /// Library definition entities nothing references — dead *MAT, *SECTION,
    /// *DEFINE_CURVE, *SET, *DEFINE_COORDINATE, … Reports at Warning severity.
    #[staticmethod]
    fn unreferenced_entities() -> Self {
        Self {
            inner: validate::Rule::unreferenced_entities(),
        }
    }
    /// Rigid-body keywords (*LOAD_RIGID_BODY, *CONSTRAINED_RIGID_BODIES,
    /// *CONSTRAINED_EXTRA_NODES, *BOUNDARY_PRESCRIBED_MOTION_RIGID, …) must
    /// target a *MAT_RIGID part; flags a reference to a deformable part.
    #[staticmethod]
    fn rigid_context() -> Self {
        Self {
            inner: validate::Rule::rigid_context(),
        }
    }
    /// Set severity (default Error).
    fn with_severity(&self, severity: validate::Severity) -> Self {
        Self {
            inner: self.inner.clone().with_severity(severity),
        }
    }
    /// Apply only within files whose path contains one of `patterns`.
    fn only_in(&self, patterns: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().only_in(patterns),
        }
    }
    /// Apply everywhere except files whose path contains one of `patterns`.
    fn except_in(&self, patterns: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().except_in(patterns),
        }
    }
}

/// One rule violation with a clickable `file:line`.
#[pyclass(name = "Finding", skip_from_py_object)]
#[derive(Clone)]
pub struct PyFinding {
    #[pyo3(get)]
    rule: String,
    #[pyo3(get)]
    severity: validate::Severity,
    #[pyo3(get)]
    keyword: String,
    #[pyo3(get)]
    file: String,
    #[pyo3(get)]
    line: usize,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl PyFinding {
    fn location(&self) -> String {
        format!("{}:{}", self.file, self.line)
    }
    fn __repr__(&self) -> String {
        format!(
            "Finding({:?}, {}, {}:{}, {:?})",
            self.severity, self.rule, self.file, self.line, self.message
        )
    }
}

/// The result of a validation run.
#[pyclass(name = "Report")]
pub struct PyReport {
    #[pyo3(get)]
    findings: Vec<PyFinding>,
}

#[pymethods]
impl PyReport {
    fn is_clean(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == validate::Severity::Error)
    }
    fn count(&self, severity: validate::Severity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == severity)
            .count()
    }
    fn __len__(&self) -> usize {
        self.findings.len()
    }
    fn __repr__(&self) -> String {
        format!("Report({} findings)", self.findings.len())
    }
}

/// Convert a core [`validate::Report`] into its Python mirror. Shared by
/// [`PyDeck::validate`](super::deck).
pub(super) fn report_to_py(report: validate::Report) -> PyReport {
    PyReport {
        findings: report
            .findings
            .into_iter()
            .map(|f| PyFinding {
                rule: f.rule,
                severity: f.severity,
                keyword: f.keyword,
                file: f.file.display().to_string(),
                line: f.line,
                message: f.message,
            })
            .collect(),
    }
}
