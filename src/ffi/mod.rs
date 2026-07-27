//! C ABI bindings for the deck **parse + validate** path — and, through the C
//! ABI, Fortran (`iso_c_binding`), which has no direct Rust bridge.
//!
//! Opt-in via the `ffi` cargo feature. Build a `cdylib`/`staticlib` and link
//! against it from C or Fortran; the matching header is `examples/ffi/dynars.h`.
//!
//! Design:
//! - **Opaque handles.** [`DynarsDeck`], [`DynarsRuleSet`], and [`DynarsReport`]
//!   are heap-boxed Rust values the caller owns; C sees only pointers and must
//!   free each with the matching `*_free`. No other Rust type crosses the
//!   boundary.
//! - **Errors.** Fallible calls return NULL (or `-1`) and stash a message in a
//!   thread-local retrievable via [`dynars_last_error`] — there is no `Result`
//!   or panic across FFI.
//! - **Strings.** Every `*const c_char` handed out is NUL-terminated and owned
//!   by the handle it came from (findings live inside their [`DynarsReport`]);
//!   copy before freeing that handle. `dynars_last_error`'s string is valid
//!   until the next `dynars_*` call on the same thread.
//!
//! Nothing here can unwind into the caller: no method used below panics on
//! valid input, and pointer arguments are null-checked before use.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::path::Path;

use crate::deck::{Deck, parse_deck};
use crate::validate::{Finding, Rule, Severity};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Build a [`CString`] that can never fail: interior NUL bytes (which can't
/// appear in a C string) are dropped rather than erroring.
fn cstring(s: &str) -> CString {
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != 0).collect();
    CString::new(bytes).expect("NULs filtered out above")
}

fn set_error(msg: &str) {
    let c = cstring(msg);
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Message describing the most recent failing `dynars_*` call **on this
/// thread**, or NULL if the last such call succeeded. The pointer is owned by
/// the library and stays valid only until the next `dynars_*` call on this
/// thread — copy it if you need it longer.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()))
}

// ── Deck: parse + basic queries ──────────────────────────────────────────

/// An opaque parsed deck (root file + every file it `*INCLUDE`s).
pub struct DynarsDeck {
    inner: Deck,
}

/// Parse `path` (a NUL-terminated UTF-8 path) and every file it includes.
/// Returns an owned handle, or NULL on error (see [`dynars_last_error`]).
/// Free with [`dynars_deck_free`].
#[unsafe(no_mangle)]
pub extern "C" fn dynars_parse_deck(path: *const c_char) -> *mut DynarsDeck {
    clear_error();
    if path.is_null() {
        set_error("dynars_parse_deck: path is NULL");
        return std::ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(p) => p,
        Err(_) => {
            set_error("dynars_parse_deck: path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    match parse_deck(Path::new(path)) {
        Ok(inner) => Box::into_raw(Box::new(DynarsDeck { inner })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Free a deck handle. NULL is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_deck_free(deck: *mut DynarsDeck) {
    if !deck.is_null() {
        drop(unsafe { Box::from_raw(deck) });
    }
}

/// Number of files in the deck (root + includes). 0 if `deck` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_deck_file_count(deck: *const DynarsDeck) -> usize {
    unsafe { deck.as_ref() }.map_or(0, |d| d.inner.files.len())
}

/// Total source bytes across all files. 0 if `deck` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_deck_total_bytes(deck: *const DynarsDeck) -> usize {
    unsafe { deck.as_ref() }.map_or(0, |d| d.inner.total_bytes())
}

// ── Rule set: a builder the caller fills then passes to validate ──────────

/// An opaque, growable set of validation rules.
pub struct DynarsRuleSet {
    rules: Vec<Rule>,
}

/// Create an empty rule set. Free with [`dynars_ruleset_free`].
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_new() -> *mut DynarsRuleSet {
    Box::into_raw(Box::new(DynarsRuleSet { rules: Vec::new() }))
}

/// Free a rule set. NULL is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_free(rules: *mut DynarsRuleSet) {
    if !rules.is_null() {
        drop(unsafe { Box::from_raw(rules) });
    }
}

/// Cross-keyword referential integrity: every id reference resolves
/// (PART.mid → *MAT, *LOAD.lcid → *DEFINE_CURVE, …). Does not check element
/// connectivity. No-op if `rules` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_add_references_resolve(rules: *mut DynarsRuleSet) {
    if let Some(r) = unsafe { rules.as_mut() } {
        r.rules.push(Rule::references_resolve());
    }
}

/// As [`dynars_ruleset_add_references_resolve`], and additionally checks that
/// every element's nodes are defined. Heavy on large meshes. No-op if `rules`
/// is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_add_references_resolve_with_connectivity(rules: *mut DynarsRuleSet) {
    if let Some(r) = unsafe { rules.as_mut() } {
        r.rules.push(Rule::references_resolve_with_connectivity());
    }
}

/// Flag every `*INCLUDE` whose target file is missing on disk. No-op if
/// `rules` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_add_include_missing(rules: *mut DynarsRuleSet) {
    if let Some(r) = unsafe { rules.as_mut() } {
        r.rules.push(Rule::include_missing());
    }
}

/// Flag any occurrence of `keyword` (case-insensitive, matched on canonical
/// base). Returns 0 on success, -1 on error (see [`dynars_last_error`]).
#[unsafe(no_mangle)]
pub extern "C" fn dynars_ruleset_add_keyword_forbidden(
    rules: *mut DynarsRuleSet,
    keyword: *const c_char,
) -> i32 {
    let Some(r) = (unsafe { rules.as_mut() }) else {
        set_error("dynars_ruleset_add_keyword_forbidden: rules is NULL");
        return -1;
    };
    if keyword.is_null() {
        set_error("dynars_ruleset_add_keyword_forbidden: keyword is NULL");
        return -1;
    }
    let kw = match unsafe { CStr::from_ptr(keyword) }.to_str() {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        Ok(_) => {
            set_error("dynars_ruleset_add_keyword_forbidden: keyword must not be empty");
            return -1;
        }
        Err(_) => {
            set_error("dynars_ruleset_add_keyword_forbidden: keyword is not valid UTF-8");
            return -1;
        }
    };
    r.rules.push(Rule::keyword_forbidden(kw));
    0
}

// ── Report: run rules, then read findings back ───────────────────────────

/// Severity of a finding. ABI-stable integer values matching the C enum.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DynarsSeverity {
    Error = 0,
    Warning = 1,
    Info = 2,
}

impl From<Severity> for DynarsSeverity {
    fn from(s: Severity) -> Self {
        match s {
            Severity::Error => DynarsSeverity::Error,
            Severity::Warning => DynarsSeverity::Warning,
            Severity::Info => DynarsSeverity::Info,
        }
    }
}

/// A finding with its strings pre-encoded as owned C strings, so the accessors
/// can hand out stable `*const c_char` for the report's lifetime.
struct FfiFinding {
    rule: CString,
    keyword: CString,
    file: CString,
    message: CString,
    severity: Severity,
    line: usize,
}

impl From<&Finding> for FfiFinding {
    fn from(f: &Finding) -> Self {
        FfiFinding {
            rule: cstring(&f.rule),
            keyword: cstring(&f.keyword),
            file: cstring(&f.file.display().to_string()),
            message: cstring(&f.message),
            severity: f.severity,
            line: f.line,
        }
    }
}

/// An opaque validation report: an ordered list of findings.
pub struct DynarsReport {
    findings: Vec<FfiFinding>,
}

/// Run `rules` over `deck`, reusing the existing parse. Returns an owned report
/// handle, or NULL if either argument is NULL (see [`dynars_last_error`]). Free
/// with [`dynars_report_free`].
#[unsafe(no_mangle)]
pub extern "C" fn dynars_deck_validate(
    deck: *const DynarsDeck,
    rules: *const DynarsRuleSet,
) -> *mut DynarsReport {
    clear_error();
    let Some(d) = (unsafe { deck.as_ref() }) else {
        set_error("dynars_deck_validate: deck is NULL");
        return std::ptr::null_mut();
    };
    let Some(rs) = (unsafe { rules.as_ref() }) else {
        set_error("dynars_deck_validate: rules is NULL");
        return std::ptr::null_mut();
    };
    let report = d.inner.validate(rs.rules.iter().cloned());
    let findings = report.findings.iter().map(FfiFinding::from).collect();
    Box::into_raw(Box::new(DynarsReport { findings }))
}

/// Free a report handle. NULL is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_free(report: *mut DynarsReport) {
    if !report.is_null() {
        drop(unsafe { Box::from_raw(report) });
    }
}

/// Number of findings. 0 if `report` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_len(report: *const DynarsReport) -> usize {
    unsafe { report.as_ref() }.map_or(0, |r| r.findings.len())
}

/// Number of findings at `severity`. 0 if `report` is NULL.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_count(
    report: *const DynarsReport,
    severity: DynarsSeverity,
) -> usize {
    unsafe { report.as_ref() }.map_or(0, |r| {
        r.findings
            .iter()
            .filter(|f| DynarsSeverity::from(f.severity) == severity)
            .count()
    })
}

/// 1 if the report has no `Error`-severity findings (warnings/info are still
/// "clean"), else 0. Also 1 for a NULL report.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_is_clean(report: *const DynarsReport) -> i32 {
    let clean = unsafe { report.as_ref() }
        .is_none_or(|r| !r.findings.iter().any(|f| f.severity == Severity::Error));
    clean as i32
}

fn finding_at<'a>(report: *const DynarsReport, i: usize) -> Option<&'a FfiFinding> {
    unsafe { report.as_ref() }.and_then(|r| r.findings.get(i))
}

/// Severity of finding `i`. Returns `Error` for an out-of-range index or NULL
/// report — check [`dynars_report_len`] first.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_severity(
    report: *const DynarsReport,
    i: usize,
) -> DynarsSeverity {
    finding_at(report, i).map_or(DynarsSeverity::Error, |f| f.severity.into())
}

/// 1-based source line of finding `i`. 0 if out of range.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_line(report: *const DynarsReport, i: usize) -> usize {
    finding_at(report, i).map_or(0, |f| f.line)
}

/// Name of the rule that produced finding `i`, or NULL if out of range. Valid
/// until the report is freed.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_rule(
    report: *const DynarsReport,
    i: usize,
) -> *const c_char {
    finding_at(report, i).map_or(std::ptr::null(), |f| f.rule.as_ptr())
}

/// Keyword the finding is about, or NULL if out of range. Valid until the
/// report is freed.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_keyword(
    report: *const DynarsReport,
    i: usize,
) -> *const c_char {
    finding_at(report, i).map_or(std::ptr::null(), |f| f.keyword.as_ptr())
}

/// File the finding is in, or NULL if out of range. Valid until the report is
/// freed.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_file(
    report: *const DynarsReport,
    i: usize,
) -> *const c_char {
    finding_at(report, i).map_or(std::ptr::null(), |f| f.file.as_ptr())
}

/// Human-readable message for finding `i`, or NULL if out of range. Valid until
/// the report is freed.
#[unsafe(no_mangle)]
pub extern "C" fn dynars_report_finding_message(
    report: *const DynarsReport,
    i: usize,
) -> *const c_char {
    finding_at(report, i).map_or(std::ptr::null(), |f| f.message.as_ptr())
}
