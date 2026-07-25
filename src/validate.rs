//! Fast, **typed** rule-based validation of LS-DYNA keyword decks.
//!
//! Checks are expressed against real types, not magic strings: keywords come
//! from the typo-proof [`names`](crate::keywords::names) constants, comparisons
//! use the [`Cmp`] enum, severities use [`Severity`], values use [`Value`].
//!
//! A deck is parsed once into a [`DeckIndex`] (mmap + block index, includes
//! resolved). A [`Validator`] runs every [`Check`] over it in parallel.
//! Built-in [`Rule`]s cover the common cases; the [`Expr`] tree composes
//! predicates with `all`/`any`/`not`; [`FileScope`] limits a rule to (or
//! excludes it from) particular include files.
//!
//! This lives in the core crate (not a separate one) because its Python
//! bindings must share the `dynars._dynars` extension, and it carries no heavy
//! dependencies. The heavy Arrow/Iceberg sinks stay in their own crates.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::keyword::{Block, ParsedFile};
use crate::keywords;
use crate::schema::{parse_schema, Card, Column, FieldSpec, FieldType, Schema, Table};

// ── Typed vocabulary ────────────────────────────────────────────────────────

/// How serious a violation is.
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int, from_py_object, name = "Severity"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A comparison operator — used instead of a stringly `"eq"`/`"ne"`.
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int, from_py_object, name = "Cmp"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn test_num(self, a: f64, b: f64) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
    fn symbol(self) -> &'static str {
        match self {
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        }
    }
}

/// A typed field value, mirroring [`FieldType`].
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    fn as_num(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(_) => None,
        }
    }
    fn cmp_with(&self, cmp: Cmp, other: &Value) -> bool {
        match (self.as_num(), other.as_num()) {
            (Some(a), Some(b)) => cmp.test_num(a, b),
            _ => match cmp {
                Cmp::Eq => self == other,
                Cmp::Ne => self != other,
                _ => false,
            },
        }
    }
    fn display(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Str(s) => s.clone(),
        }
    }
}

/// A predicate on one card field: `field <cmp> value`.
#[derive(Debug, Clone)]
pub struct FieldPredicate {
    pub field: String,
    pub cmp: Cmp,
    pub value: Value,
}

/// A boolean expression tree over field predicates — the composable
/// ("tier 2") layer. Evaluated entirely in Rust.
#[derive(Debug, Clone)]
pub enum Expr {
    Field(FieldPredicate),
    All(Vec<Expr>),
    Any(Vec<Expr>),
    Not(Box<Expr>),
}

impl Expr {
    /// `field <cmp> value`.
    pub fn field(field: impl Into<String>, cmp: Cmp, value: Value) -> Expr {
        Expr::Field(FieldPredicate { field: field.into(), cmp, value })
    }
    pub fn all(exprs: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::All(exprs.into_iter().collect())
    }
    pub fn any(exprs: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::Any(exprs.into_iter().collect())
    }
    pub fn not(expr: Expr) -> Expr {
        Expr::Not(Box::new(expr))
    }

    fn eval(&self, table: &Table, row: usize) -> bool {
        match self {
            Expr::Field(p) => cell(table, &p.field, row).map(|v| v.cmp_with(p.cmp, &p.value)).unwrap_or(false),
            Expr::All(v) => v.iter().all(|e| e.eval(table, row)),
            Expr::Any(v) => v.iter().any(|e| e.eval(table, row)),
            Expr::Not(e) => !e.eval(table, row),
        }
    }

    fn describe(&self) -> String {
        match self {
            Expr::Field(p) => format!("{} {} {}", p.field, p.cmp.symbol(), p.value.display()),
            Expr::All(v) => format!("all({})", v.iter().map(Expr::describe).collect::<Vec<_>>().join(", ")),
            Expr::Any(v) => format!("any({})", v.iter().map(Expr::describe).collect::<Vec<_>>().join(", ")),
            Expr::Not(e) => format!("not({})", e.describe()),
        }
    }
}

/// Convenience: a single-field predicate as an [`Expr`].
pub fn pred(field: impl Into<String>, cmp: Cmp, value: Value) -> Expr {
    Expr::field(field, cmp, value)
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
    fn allows(&self, path: &Path) -> bool {
        let p = path.to_string_lossy().to_ascii_lowercase();
        let hit = |pats: &[String]| pats.iter().any(|s| p.contains(&s.to_ascii_lowercase()));
        match self {
            FileScope::Anywhere => true,
            FileScope::OnlyIn(pats) => hit(pats),
            FileScope::ExceptIn(pats) => !hit(pats),
        }
    }
}

// ── Findings ────────────────────────────────────────────────────────────────

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

// ── Deck access ─────────────────────────────────────────────────────────────
//
// Validation reads the core [`Deck`](crate::deck::Deck) directly — the same
// `ParsedFile` blocks the rest of dynars produces. There is no separate keyword
// index: keywords are located by scanning blocks (as `parse_schema` already
// does), which is negligible next to the one-time parse.

pub use crate::deck::Deck;

/// 1-based line of a block's `*KEYWORD` line, for clickable locations.
fn block_line(file: &ParsedFile, block: &Block) -> usize {
    1 + file.src()[..block.name_start].iter().filter(|&&b| b == b'\n').count()
}

/// Strip trailing pure-annotation options (`_TITLE`, `_ID`) to get the base
/// keyword, uppercased — so a rule on `SECTION_SHELL` matches
/// `SECTION_SHELL_TITLE` blocks.
fn canonical_base(name: &str) -> String {
    let mut s = name.to_ascii_uppercase();
    for opt in ["_TITLE", "_ID"] {
        if let Some(stripped) = s.strip_suffix(opt) {
            s = stripped.to_string();
        }
    }
    s
}

/// A schema that reliably marshals a block's **primary card** (card 1),
/// consuming the `_TITLE` heading line. Primary cards carry the identifying and
/// control fields most checks target (`SECID`, `ELFORM`, `NIP`, `MID`, …).
fn primary_schema(block_kw: &str) -> Option<Schema> {
    let base = canonical_base(block_kw);
    let first = keywords::schema(&base)?.cards.first()?.clone();
    let mut schema = Schema::new(&block_kw.to_ascii_uppercase()); // parse_schema matches exact name
    if block_kw.to_ascii_uppercase().ends_with("_TITLE") {
        schema = schema.card(Card {
            fields: vec![FieldSpec { name: "TITLE".into(), ty: FieldType::Str, width: 80, count: 1 }],
        });
    }
    Some(schema.card(first).once())
}

/// Case-insensitive cell read from a parsed table.
fn cell(table: &Table, field: &str, row: usize) -> Option<Value> {
    let (_, col) = table.columns.iter().find(|(n, _)| n.eq_ignore_ascii_case(field))?;
    match col {
        Column::Int { data, ncols } if *ncols == 1 => data.get(row).map(|v| Value::Int(*v)),
        Column::Float { data, ncols } if *ncols == 1 => data.get(row).map(|v| Value::Float(*v)),
        Column::Str { data, ncols } if *ncols == 1 => data.get(row).map(|v| Value::Str(v.clone())),
        _ => None,
    }
}

/// Iterate every (row, file, line) for a keyword's primary card across the
/// deck, honouring the file scope. Blocks are found by scanning each file (the
/// same way `parse_schema` locates a keyword) and grouped by exact name so
/// `_TITLE`/plain variants each get the right schema.
fn for_each_row(deck: &Deck, base_kw: &str, scope: &FileScope, mut f: impl FnMut(&Table, usize, PathBuf, usize)) {
    let base = canonical_base(base_kw);
    for file in &deck.files {
        if !scope.allows(&file.path) {
            continue;
        }
        let mut groups: HashMap<String, Vec<&Block>> = HashMap::new();
        for block in &file.blocks {
            if canonical_base(file.keyword_name(block)) == base {
                groups.entry(file.keyword_name(block).to_string()).or_default().push(block);
            }
        }
        for (exact, blocks) in groups {
            let Some(schema) = primary_schema(&exact) else { continue };
            let table = parse_schema(file, &schema);
            for (row, block) in blocks.iter().enumerate().take(table.rows()) {
                f(&table, row, file.path.clone(), block_line(file, block));
            }
        }
    }
}

// ── Checks ──────────────────────────────────────────────────────────────────

/// A validation check. Implement for arbitrary logic; built-in [`Rule`]s
/// already implement it. Receives the core [`Deck`].
pub trait Check: Send + Sync {
    fn name(&self) -> String;
    fn run(&self, deck: &Deck, out: &mut Vec<Finding>);
}

/// What a [`Rule`] checks.
#[derive(Debug, Clone)]
pub enum RuleKind {
    /// A keyword must not appear.
    KeywordForbidden { keyword: String },
    /// No occurrence of `keyword` may have `field` equal to any of `values`.
    FieldForbiddenValues { keyword: String, field: String, values: Vec<Value> },
    /// For every `keyword` occurrence, if `when` holds (or is `None`),
    /// `require` must also hold.
    FieldRequired { keyword: String, when: Option<Expr>, require: Expr },
    /// Every `*INCLUDE` must resolve to a file that exists on disk.
    IncludeMissing,
}

/// A built-in declarative check: a [`RuleKind`] plus a severity and file scope.
#[derive(Debug, Clone)]
pub struct Rule {
    pub kind: RuleKind,
    pub severity: Severity,
    pub scope: FileScope,
}

impl Rule {
    fn wrap(kind: RuleKind) -> Rule {
        Rule { kind, severity: Severity::Error, scope: FileScope::Anywhere }
    }
    pub fn keyword_forbidden(keyword: impl Into<String>) -> Rule {
        Rule::wrap(RuleKind::KeywordForbidden { keyword: keyword.into() })
    }
    pub fn field_forbidden_values(keyword: impl Into<String>, field: impl Into<String>, values: impl IntoIterator<Item = Value>) -> Rule {
        Rule::wrap(RuleKind::FieldForbiddenValues { keyword: keyword.into(), field: field.into(), values: values.into_iter().collect() })
    }
    pub fn field_required(keyword: impl Into<String>, when: Option<Expr>, require: Expr) -> Rule {
        Rule::wrap(RuleKind::FieldRequired { keyword: keyword.into(), when, require })
    }
    pub fn include_missing() -> Rule {
        Rule::wrap(RuleKind::IncludeMissing)
    }

    /// Set the severity (default `Error`).
    pub fn with_severity(mut self, sev: Severity) -> Self {
        self.severity = sev;
        self
    }
    /// Apply this rule only within files whose path contains one of `pats`.
    pub fn only_in(mut self, pats: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.scope = FileScope::OnlyIn(pats.into_iter().map(Into::into).collect());
        self
    }
    /// Apply this rule everywhere except files whose path contains one of `pats`.
    pub fn except_in(mut self, pats: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.scope = FileScope::ExceptIn(pats.into_iter().map(Into::into).collect());
        self
    }

    fn keyword(&self) -> &str {
        match &self.kind {
            RuleKind::KeywordForbidden { keyword }
            | RuleKind::FieldForbiddenValues { keyword, .. }
            | RuleKind::FieldRequired { keyword, .. } => keyword,
            RuleKind::IncludeMissing => "INCLUDE",
        }
    }
}

impl Check for Rule {
    fn name(&self) -> String {
        match &self.kind {
            RuleKind::KeywordForbidden { keyword } => format!("keyword_forbidden({keyword})"),
            RuleKind::FieldForbiddenValues { keyword, field, .. } => format!("field_forbidden_values({keyword}.{field})"),
            RuleKind::FieldRequired { keyword, .. } => format!("field_required({keyword})"),
            RuleKind::IncludeMissing => "include_missing".to_string(),
        }
    }

    fn run(&self, deck: &Deck, out: &mut Vec<Finding>) {
        let sev = self.severity;
        match &self.kind {
            RuleKind::KeywordForbidden { keyword } => {
                let base = canonical_base(keyword);
                for file in &deck.files {
                    if !self.scope.allows(&file.path) {
                        continue;
                    }
                    for block in &file.blocks {
                        if canonical_base(file.keyword_name(block)) != base {
                            continue;
                        }
                        out.push(Finding {
                            rule: self.name(),
                            severity: sev,
                            keyword: base.clone(),
                            file: file.path.clone(),
                            line: block_line(file, block),
                            message: format!("keyword *{base} is not allowed here"),
                        });
                    }
                }
            }
            RuleKind::FieldForbiddenValues { keyword, field, values } => {
                for_each_row(deck, keyword, &self.scope, |table, row, file, line| {
                    if let Some(v) = cell(table, field, row) {
                        if values.iter().any(|bad| bad.cmp_with(Cmp::Eq, &v)) {
                            out.push(Finding {
                                rule: self.name(),
                                severity: sev,
                                keyword: canonical_base(keyword),
                                file,
                                line,
                                message: format!("{field} = {} is forbidden", v.display()),
                            });
                        }
                    }
                });
            }
            RuleKind::FieldRequired { keyword, when, require } => {
                for_each_row(deck, keyword, &self.scope, |table, row, file, line| {
                    let applies = when.as_ref().map(|w| w.eval(table, row)).unwrap_or(true);
                    if !applies || require.eval(table, row) {
                        return;
                    }
                    let cond = when.as_ref().map(|w| format!(" when {}", w.describe())).unwrap_or_default();
                    let got = if let Expr::Field(p) = require {
                        cell(table, &p.field, row).map(|v| format!(", got {}", v.display())).unwrap_or_default()
                    } else {
                        String::new()
                    };
                    out.push(Finding {
                        rule: self.name(),
                        severity: sev,
                        keyword: canonical_base(keyword),
                        file,
                        line,
                        message: format!("requires {}{cond}{got}", require.describe()),
                    });
                });
            }
            RuleKind::IncludeMissing => {
                for (fi, inc) in &deck.includes {
                    let including = &deck.files[*fi].path;
                    if !self.scope.allows(including) {
                        continue;
                    }
                    if !inc.resolved_path.exists() {
                        out.push(Finding {
                            rule: self.name(),
                            severity: sev,
                            keyword: "INCLUDE".to_string(),
                            file: including.clone(),
                            line: 0,
                            message: format!("include '{}' resolves to a missing file: {}", inc.raw_path, inc.resolved_path.display()),
                        });
                    }
                }
            }
        }
        let _ = self.keyword(); // keyword() kept for external callers/bindings
    }
}

// ── The engine ──────────────────────────────────────────────────────────────

/// Collects checks and runs them over a deck, in parallel.
#[derive(Default)]
pub struct Validator {
    checks: Vec<Box<dyn Check>>,
}

impl Validator {
    pub fn new() -> Self {
        Validator { checks: Vec::new() }
    }
    pub fn rule(mut self, rule: Rule) -> Self {
        self.checks.push(Box::new(rule));
        self
    }
    pub fn check(mut self, check: Box<dyn Check>) -> Self {
        self.checks.push(check);
        self
    }

    /// Parse `root` (following includes, once) and run every check in parallel.
    pub fn run(&self, root: impl AsRef<Path>) -> Result<Report, String> {
        let deck = crate::deck::parse_deck(root.as_ref())?;
        Ok(self.run_on(&deck))
    }

    /// Run against an already-parsed [`Deck`] (reuse across rule sets).
    pub fn run_on(&self, deck: &Deck) -> Report {
        let findings = self
            .checks
            .par_iter()
            .flat_map(|c| {
                let mut local = Vec::new();
                c.run(deck, &mut local);
                local
            })
            .collect();
        Report { findings }
    }
}
