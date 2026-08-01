//! The built-in rules — one [`Check`] each — and the public [`Rule`] wrapper
//! that layers severity and file scope onto a check's findings.

use std::sync::Arc;

use crate::keywords::{self, EntityKind, canonical_base};

use super::expr::{Cmp, Expr};
use super::report::{FileScope, Finding, Severity};
use super::{Check, Deck};
use crate::model::Value;

// ── Built-in rules: one Check per rule ───────────────────────────────────────
//
// Each built-in is its own small `Check`, read in isolation. They report at
// `Severity::Error` over the whole deck; the `Rule` wrapper below applies the
// caller's severity and file scope, so neither concern is repeated per rule.

/// A keyword must not appear.
struct KeywordForbidden {
    keyword: String,
}
impl Check for KeywordForbidden {
    fn name(&self) -> String {
        format!("keyword_forbidden({})", self.keyword)
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let base = canonical_base(&self.keyword);
        let mut out = Vec::new();
        for file in &deck.files {
            for block in &file.blocks {
                if canonical_base(file.keyword_name(block)) != base {
                    continue;
                }
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: base.clone(),
                    file: file.path.clone(),
                    line: crate::schema::block_line(file, block),
                    message: format!("keyword *{base} is not allowed here"),
                });
            }
        }
        out
    }
}

/// No occurrence of `keyword` may have `field` equal to any of `values`.
struct FieldForbiddenValues {
    keyword: String,
    field: String,
    values: Vec<Value>,
}
impl Check for FieldForbiddenValues {
    fn name(&self) -> String {
        format!("field_forbidden_values({}.{})", self.keyword, self.field)
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        for kw in deck.keywords(&self.keyword) {
            if let Some(v) = kw.field(&self.field).map(|f| f.value())
                && self.values.iter().any(|bad| Cmp::Eq.test(bad, &v))
            {
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: canonical_base(&self.keyword),
                    file: kw.file().to_path_buf(),
                    line: kw.line(),
                    message: format!("{} = {} is forbidden", self.field, v.display()),
                });
            }
        }
        out
    }
}

/// For every `keyword` occurrence, if `when` holds (or is `None`), `require`
/// must also hold.
struct FieldRequired {
    keyword: String,
    when: Option<Expr>,
    require: Expr,
}
impl Check for FieldRequired {
    fn name(&self) -> String {
        format!("field_required({})", self.keyword)
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        for kw in deck.keywords(&self.keyword) {
            let applies = self.when.as_ref().map(|w| w.eval(&kw)).unwrap_or(true);
            if !applies || self.require.eval(&kw) {
                continue;
            }
            let cond = self
                .when
                .as_ref()
                .map(|w| format!(" when {}", w.describe()))
                .unwrap_or_default();
            let got = if let Expr::Field(p) = &self.require {
                kw.field(&p.field)
                    .map(|f| format!(", got {}", f.value().display()))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            out.push(Finding {
                rule: self.name(),
                severity: Severity::Error,
                keyword: canonical_base(&self.keyword),
                file: kw.file().to_path_buf(),
                line: kw.line(),
                message: format!("requires {}{cond}{got}", self.require.describe()),
            });
        }
        out
    }
}

/// Every `*INCLUDE` must resolve to a file that exists on disk.
struct IncludeMissing;
impl Check for IncludeMissing {
    fn name(&self) -> String {
        "include_missing".to_string()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        for (fi, inc) in &deck.includes {
            if !inc.resolved_path.exists() {
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: "INCLUDE".to_string(),
                    file: deck.files[*fi].path.clone(),
                    line: 0,
                    message: format!(
                        "include '{}' resolves to a missing file: {}",
                        inc.raw_path,
                        inc.resolved_path.display()
                    ),
                });
            }
        }
        out
    }
}

/// Every cross-keyword id reference must resolve to a defined entity.
/// `connectivity` additionally checks element→node references (heavy on big
/// meshes).
struct ReferencesResolve {
    connectivity: bool,
}
impl Check for ReferencesResolve {
    fn name(&self) -> String {
        "references_resolve".to_string()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        deck.dangling(self.connectivity)
            .into_iter()
            .map(|d| Finding {
                rule: self.name(),
                severity: Severity::Error,
                keyword: d.from_keyword.clone(),
                file: d.file.clone(),
                line: d.line,
                message: format!(
                    "{}.{} references {} {} — not defined in the deck",
                    d.from_keyword,
                    d.field,
                    target_name(&d.target),
                    d.id
                ),
            })
            .collect()
    }
}

/// No two per-block definition entities of the same kind may share an id.
struct DuplicateIds;
impl Check for DuplicateIds {
    fn name(&self) -> String {
        "duplicate_ids".to_string()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        for dup in deck.duplicate_definitions() {
            let n = dup.sites.len();
            // Flag every colliding definition, so each is a clickable location.
            for &(fi, bi) in &dup.sites {
                let file = &deck.files[fi];
                let block = &file.blocks[bi];
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Error,
                    keyword: canonical_base(file.keyword_name(block)),
                    file: file.path.clone(),
                    line: crate::schema::block_line(file, block),
                    message: format!(
                        "duplicate {:?} id {} — defined by {n} keywords in the deck",
                        dup.kind, dup.id
                    ),
                });
            }
        }
        out
    }
}

/// The "library" definition kinds — entities other cards point at — for which
/// "defined but unused" is a meaningful warning. Excludes nodes/elements (a mesh
/// concern) and parts (referenced by element connectivity, which this check does
/// not scan).
const UNREFERENCED_KINDS: &[EntityKind] = &[
    EntityKind::Material,
    EntityKind::ThermalMaterial,
    EntityKind::Section,
    EntityKind::Eos,
    EntityKind::Hourglass,
    EntityKind::Curve,
    EntityKind::Coord,
    EntityKind::Vector,
    EntityKind::Box,
    EntityKind::Transform,
    EntityKind::NodeSet,
    EntityKind::PartSet,
    EntityKind::SegmentSet,
    EntityKind::ShellSet,
    EntityKind::SolidSet,
    EntityKind::BeamSet,
    EntityKind::DiscreteSet,
    EntityKind::Sensor,
    EntityKind::Define,
];

/// A library definition entity that nothing in the deck references — a dead
/// definition. Reported per kind in `kinds`.
struct UnreferencedEntities {
    kinds: Vec<EntityKind>,
}
impl Check for UnreferencedEntities {
    fn name(&self) -> String {
        "unreferenced_entities".to_string()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let referenced = deck.referenced_ids();
        let mut out = Vec::new();
        for &kind in &self.kinds {
            let used = referenced.get(&kind);
            for kw in deck.entities(kind) {
                let Some(id) = kw.id() else {
                    continue;
                };
                if used.is_some_and(|s| s.contains(&id)) {
                    continue;
                }
                out.push(Finding {
                    rule: self.name(),
                    severity: Severity::Warning,
                    keyword: canonical_base(kw.name()),
                    file: kw.file().to_path_buf(),
                    line: kw.line(),
                    message: format!("{kind:?} {id} is defined but never referenced"),
                });
            }
        }
        out
    }
}

/// Keywords that act on a *rigid body*, paired with the part-id field(s) that
/// must therefore reference a `*MAT_RIGID` part. Chosen to be unambiguous — each
/// entry's field is a part the solver treats as rigid, so a deformable target is
/// a genuine modelling error, not a style choice.
const RIGID_CONTEXT: &[(&str, &[&str])] = &[
    ("LOAD_RIGID_BODY", &["PID"]),
    ("CONSTRAINED_RIGID_BODIES", &["PIDL", "PIDC"]),
    ("CONSTRAINED_EXTRA_NODES_NODE", &["PID"]),
    ("CONSTRAINED_EXTRA_NODES_SET", &["PID"]),
    ("CONSTRAINED_RIGID_BODY_STOPPERS", &["PID"]),
    ("BOUNDARY_PRESCRIBED_MOTION_RIGID", &["PID"]),
];

/// Whether a material keyword's canonical base denotes `*MAT_RIGID` — LS-DYNA
/// material 20, writable as `*MAT_RIGID` or `*MAT_020` (some decks drop the pad).
fn is_rigid_material(base: &str) -> bool {
    matches!(base, "MAT_RIGID" | "MAT_020" | "MAT_20")
}

/// Keywords that act on a rigid body must reference a part whose material is
/// `*MAT_RIGID` (e.g. `*LOAD_RIGID_BODY`, `*CONSTRAINED_RIGID_BODIES`). Flags a
/// reference to a part that resolves to a *deformable* material.
struct RigidContext;
impl Check for RigidContext {
    fn name(&self) -> String {
        "rigid_context".to_string()
    }
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        let mut out = Vec::new();
        for &(keyword, fields) in RIGID_CONTEXT {
            for kw in deck.keywords(keyword) {
                let transform = kw.transform();
                let base = canonical_base(kw.name());
                for &field in fields {
                    let Some(pid_raw) = kw.field(field).and_then(|f| f.as_i64()) else {
                        continue;
                    };
                    if pid_raw == 0 {
                        continue;
                    }
                    // The part id is written in this occurrence's file namespace;
                    // shift it into the global one before resolving.
                    let pid = transform.map_or(pid_raw, |t| t.apply(pid_raw, EntityKind::Part));
                    // Only flag a *resolved* part whose material is positively
                    // non-rigid. An unresolved part or material is a dangling
                    // reference, reported by `references_resolve`, not here.
                    let Some(part) = deck.part(pid) else {
                        continue;
                    };
                    let Some(mat) = part.material() else {
                        continue;
                    };
                    let mat_base = canonical_base(mat.name());
                    if !is_rigid_material(&mat_base) {
                        out.push(Finding {
                            rule: self.name(),
                            severity: Severity::Error,
                            keyword: base.clone(),
                            file: kw.file().to_path_buf(),
                            line: kw.line(),
                            message: format!(
                                "{base}.{field} references part {pid}, whose material *{mat_base} is not *MAT_RIGID"
                            ),
                        });
                    }
                }
            }
        }
        out
    }
}

/// A [`Check`] plus a severity and file scope layered over its findings. This
/// is the one currency validation deals in: every built-in is constructed here,
/// and any custom [`Check`] becomes one via [`Rule::custom`]. Cheap to clone —
/// the check is shared.
///
/// `severity` is optional: `None` keeps whatever severity the inner check
/// assigned each finding (built-ins report `Error`); [`with_severity`](Rule::with_severity) sets an
/// override that re-stamps them all.
#[derive(Clone)]
pub struct Rule {
    check: Arc<dyn Check>,
    severity: Option<Severity>,
    scope: FileScope,
}

impl Rule {
    fn wrap(check: impl Check + 'static) -> Rule {
        Rule {
            check: Arc::new(check),
            severity: None,
            scope: FileScope::Anywhere,
        }
    }
    /// Lift any custom [`Check`] into a `Rule`, so it composes with file scope
    /// and severity overrides and runs through [`Deck::validate`](crate::deck::Deck::validate)
    /// alongside the built-ins. The check keeps its own per-finding severities
    /// unless you call [`with_severity`](Rule::with_severity).
    pub fn custom(check: impl Check + 'static) -> Rule {
        Rule::wrap(check)
    }
    pub fn keyword_forbidden(keyword: impl Into<String>) -> Rule {
        Rule::wrap(KeywordForbidden {
            keyword: keyword.into(),
        })
    }
    pub fn field_forbidden_values(
        keyword: impl Into<String>,
        field: impl Into<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> Rule {
        Rule::wrap(FieldForbiddenValues {
            keyword: keyword.into(),
            field: field.into(),
            values: values.into_iter().collect(),
        })
    }
    pub fn field_required(keyword: impl Into<String>, when: Option<Expr>, require: Expr) -> Rule {
        Rule::wrap(FieldRequired {
            keyword: keyword.into(),
            when,
            require,
        })
    }
    pub fn include_missing() -> Rule {
        Rule::wrap(IncludeMissing)
    }
    /// Cross-keyword referential integrity: every id reference resolves
    /// (`PART.mid → *MAT`, `*LOAD.lcid → *DEFINE_CURVE`, `*PART.secid →
    /// *SECTION`, …). Does *not* check element connectivity — use
    /// [`references_resolve_with_connectivity`](Rule::references_resolve_with_connectivity)
    /// for that.
    pub fn references_resolve() -> Rule {
        Rule::wrap(ReferencesResolve {
            connectivity: false,
        })
    }
    /// As [`references_resolve`](Rule::references_resolve), and additionally
    /// checks element connectivity — that every element's nodes are defined.
    /// Heavy on large meshes (millions of element→node references).
    pub fn references_resolve_with_connectivity() -> Rule {
        Rule::wrap(ReferencesResolve { connectivity: true })
    }

    /// No two labelled definition entities of the same kind share an id (two
    /// `*PART` pid=5, duplicate `*MAT`/`*SET`/`*SECTION`/`*DEFINE_CURVE` ids, …)
    /// — an id collision LS-DYNA rejects for several entity types. Compared on
    /// *logical* ids, so the same id reused across two `*INCLUDE_TRANSFORM`
    /// instances is not a collision. Nodes/elements are out of scope (a mesh
    /// concern).
    pub fn duplicate_ids() -> Rule {
        Rule::wrap(DuplicateIds)
    }

    /// Library definition entities that nothing references — dead `*MAT`,
    /// `*SECTION`, `*DEFINE_CURVE`, `*SET`, `*DEFINE_COORDINATE`, … (the "unused
    /// X" warnings other checkers report). Reports at [`Severity::Warning`].
    /// Reference scanning uses the built-in schema and skips element/node
    /// connectivity, so an entity reached only through a user-schema keyword or
    /// a globally-applied control card may show as unreferenced.
    pub fn unreferenced_entities() -> Rule {
        Rule::wrap(UnreferencedEntities {
            kinds: UNREFERENCED_KINDS.to_vec(),
        })
    }

    /// As [`unreferenced_entities`](Rule::unreferenced_entities), but only for the
    /// entity kinds you name — e.g. just unused `*DEFINE_CURVE`s.
    pub fn unreferenced_entities_of(kinds: impl IntoIterator<Item = EntityKind>) -> Rule {
        Rule::wrap(UnreferencedEntities {
            kinds: kinds.into_iter().collect(),
        })
    }

    /// Rigid-body keywords must target a rigid part: `*LOAD_RIGID_BODY`,
    /// `*CONSTRAINED_RIGID_BODIES`, `*CONSTRAINED_EXTRA_NODES`,
    /// `*CONSTRAINED_RIGID_BODY_STOPPERS`, and `*BOUNDARY_PRESCRIBED_MOTION_RIGID`
    /// each name a part that the solver treats as rigid — this flags one that
    /// resolves to a *deformable* (`*MAT`-other-than-020) material. Parts that
    /// don't resolve at all are left to
    /// [`references_resolve`](Rule::references_resolve).
    pub fn rigid_context() -> Rule {
        Rule::wrap(RigidContext)
    }

    /// Override the severity of every finding this rule produces (built-ins
    /// otherwise report `Error`; a custom check keeps its own severities).
    pub fn with_severity(mut self, sev: Severity) -> Self {
        self.severity = Some(sev);
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
}

impl Check for Rule {
    fn name(&self) -> String {
        self.check.name()
    }

    /// Run the inner check over the whole deck, then apply this rule's file
    /// scope and (optional) severity override to what it found — the concerns
    /// the checks themselves don't have to know about.
    fn run(&self, deck: &Deck) -> Vec<Finding> {
        self.check
            .run(deck)
            .into_iter()
            .filter(|f| self.scope.allows(&f.file))
            .map(|mut f| {
                if let Some(sev) = self.severity {
                    f.severity = sev;
                }
                f
            })
            .collect()
    }
}

/// Human name for a reference target (for finding messages).
fn target_name(r: &keywords::Ref) -> String {
    match r {
        keywords::Ref::To(k) => format!("{k:?}"),
        keywords::Ref::AnyOf(ks) => format!("{ks:?}"),
        keywords::Ref::None => String::new(),
    }
}
