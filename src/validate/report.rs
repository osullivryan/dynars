//! Validation output: severity, file scope, and the findings a run produces.

use std::path::{Path, PathBuf};

/// How serious a violation is.
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "Severity")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Which include files a rule applies to.
#[derive(Debug, Clone)]
pub enum FileScope {
    /// Every file in the deck (default).
    Anywhere,
    /// Only files whose path contains one of these substrings (case-insensitive).
    OnlyIn(Vec<String>),
    /// Every file *except* those whose path contains one of these substrings.
    ExceptIn(Vec<String>),
}

impl FileScope {
    pub(crate) fn allows(&self, path: &Path) -> bool {
        let p = path.to_string_lossy().to_ascii_lowercase();
        let hit = |pats: &[String]| pats.iter().any(|s| p.contains(&s.to_ascii_lowercase()));
        match self {
            FileScope::Anywhere => true,
            FileScope::OnlyIn(pats) => hit(pats),
            FileScope::ExceptIn(pats) => !hit(pats),
        }
    }
}

/// One rule violation, with a clickable `file:line` location.
#[derive(Debug, Clone)]
pub struct Finding {
    pub rule: String,
    pub severity: Severity,
    pub keyword: String,
    pub file: PathBuf,
    pub line: usize,
    pub message: String,
}

impl Finding {
    pub fn location(&self) -> String {
        format!("{}:{}", self.file.display(), self.line)
    }
}

/// The result of a validation run.
#[derive(Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
}

impl Report {
    pub fn is_clean(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == Severity::Error)
    }
    pub fn count(&self, sev: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity == sev).count()
    }
}
