//! Built-in keyword library, generated from the Ansys pyDYNA `kwd.json` field
//! database (see `codegen/gen_keywords.py`).
//!
//! Each keyword's card/field layout is stored as compact `&'static` data
//! (`data.rs`, generated) and converted on demand into a runtime
//! [`Schema`](crate::schema::Schema). This gives thousands of LS-DYNA keywords
//! out of the box — no hand-written structs, no user schema required:
//!
//! ```
//! if let Some(schema) = dynars::keywords::schema("MAT_ELASTIC") {
//!     // parse_schema(&parsed, &schema) -> columnar Table
//!     assert_eq!(schema.keyword, "MAT_ELASTIC");
//! }
//! ```
//!
//! Scope note: `kwd.json` describes each keyword's *static* card layout, which
//! covers the great majority of keywords exactly. A minority have conditional
//! or count-driven cards (present only if a flag is set, or repeated `N` times);
//! for those the generated schema is the base layout and may under-read the
//! variable tail. See `codegen/README.md`.

use crate::schema::{Card, FieldSpec, FieldType, Schema};

mod data;

/// The kind of entity a field's id refers to. Derived from the pyDYNA field
/// database's `link` codes (see `codegen/gen_keywords.py`). This is what makes
/// referential-integrity checks possible without hand-coding each keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Node,
    Element,
    Part,
    Material,
    ThermalMaterial,
    Section,
    Eos,
    Hourglass,
    Curve,
    Box,
    Coord,
    Vector,
    NodeSet,
    PartSet,
    SegmentSet,
    ShellSet,
    SolidSet,
    BeamSet,
    DiscreteSet,
    Sensor,
    Transform,
    Define,
}

/// The id offsets carried by an `*INCLUDE_TRANSFORM` directive (card 2/3 of the
/// keyword: `IDNOFF … IDROFF`). Every id *written in the included file* — both
/// the ids it **defines** and the ids it **references** — is shifted by the
/// offset for its entity kind, giving that file its own id namespace. This is
/// how the same mesh can be included several times at different offsets without
/// collision.
///
/// Offsets **compose** down a chain of nested transforms ([`compose`]), so a
/// file's *effective* offset is the sum along its include path from the root.
/// The value is a pure function of the deck text; applying it ([`apply`]) is how
/// the resolution core turns a *physical* id (as written) into the *logical*
/// (global) id every cross-reference is resolved against — see
/// [`crate::model`]. The dangling check itself never sees this type; it only
/// ever compares logical ids.
///
/// [`compose`]: TransformOffsets::compose
/// [`apply`]: TransformOffsets::apply
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransformOffsets {
    pub idnoff: i64,
    pub ideoff: i64,
    pub idpoff: i64,
    pub idmoff: i64,
    pub idsoff: i64,
    pub idfoff: i64,
    pub iddoff: i64,
    pub idroff: i64,
}

impl TransformOffsets {
    /// The no-op transform (a plain `*INCLUDE`, or the root file).
    pub const IDENTITY: TransformOffsets = TransformOffsets {
        idnoff: 0,
        ideoff: 0,
        idpoff: 0,
        idmoff: 0,
        idsoff: 0,
        idfoff: 0,
        iddoff: 0,
        idroff: 0,
    };

    /// Whether this shifts nothing — the overwhelmingly common case, which the
    /// resolution core keeps on its existing zero-overhead path (represented as
    /// `None` rather than an identity `TransformOffsets`).
    #[inline]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Compose a parent's effective offsets with a child directive's own offsets.
    /// Nested `*INCLUDE_TRANSFORM`s accumulate additively — a file two transforms
    /// deep is shifted by the sum of both.
    #[inline]
    pub fn compose(self, child: TransformOffsets) -> TransformOffsets {
        TransformOffsets {
            idnoff: self.idnoff + child.idnoff,
            ideoff: self.ideoff + child.ideoff,
            idpoff: self.idpoff + child.idpoff,
            idmoff: self.idmoff + child.idmoff,
            idsoff: self.idsoff + child.idsoff,
            idfoff: self.idfoff + child.idfoff,
            iddoff: self.iddoff + child.iddoff,
            idroff: self.idroff + child.idroff,
        }
    }

    /// The offset that applies to ids of `kind` (`0` if none).
    ///
    /// The high-traffic buckets — nodes, elements, parts, sets, curves — follow
    /// the LS-DYNA manual directly and are what referential validation leans on.
    /// The `IDMOFF`/`IDDOFF` groupings (section/eos/hourglass, and the `*DEFINE`
    /// family) are best-effort and worth confirming against the manual for a
    /// given solver version — but note that *within* a file, defs and refs of
    /// the same kind always share a bucket, so internal reference resolution is
    /// correct regardless; only a cross-file reference *into* a transformed
    /// include depends on the exact bucket.
    #[inline]
    pub fn for_kind(&self, kind: EntityKind) -> i64 {
        use EntityKind::*;
        match kind {
            Node => self.idnoff,
            Element => self.ideoff,
            // Manual: IDPOFF offsets part IDs *and* part-set IDs.
            Part | PartSet => self.idpoff,
            NodeSet | SegmentSet | ShellSet | SolidSet | BeamSet | DiscreteSet => self.idsoff,
            // IDFOFF: function, table, and curve ids (dynars models these as Curve).
            Curve => self.idfoff,
            // Material family. Section/Eos/Hourglass grouping is best-effort.
            Material | ThermalMaterial | Section | Eos | Hourglass => self.idmoff,
            // *DEFINE family (coord, vector, box, transformation, …).
            Box | Coord | Vector | Transform | Define | Sensor => self.iddoff,
        }
    }

    /// Turn a *physical* id of `kind` (as written in this file) into its
    /// *logical* (global) id. A `0`/blank id means "none" and is left untouched;
    /// a negative id (a signed reference — LS-DYNA denotes entity `|id|`) keeps
    /// its sign while its magnitude is shifted.
    #[inline]
    pub fn apply(&self, raw: i64, kind: EntityKind) -> i64 {
        let off = self.for_kind(kind);
        if off == 0 || raw == 0 {
            return raw;
        }
        raw.signum() * (raw.abs() + off)
    }
}

/// What a field references, if anything. Stored inline on each [`Fld`] in the
/// one keyword table — there is no separate reference file to keep in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    /// Not a reference.
    None,
    /// A fixed target entity kind (positive `link` codes).
    To(EntityKind),
    /// A polymorphic reference whose exact target is chosen by a companion
    /// `*TYP` field (negative `link` codes). Resolves against any listed kind.
    AnyOf(&'static [EntityKind]),
}

/// The referencing fields of a keyword: `(field_name, Ref)` for each field that
/// points at another entity. Read straight from the keyword's own [`Fld`]s (the
/// single source of truth) — no parallel table.
pub fn refs_for(keyword: &str) -> Vec<(&'static str, Ref)> {
    match find(keyword) {
        Some(kw) => kw
            .cards
            .iter()
            .flat_map(|card| card.iter())
            .filter(|f| !matches!(f.r, Ref::None))
            .map(|f| (f.n, f.r))
            .collect(),
        None => Vec::new(),
    }
}

// ── Def-side metadata: what a keyword defines ────────────────────────────────
//
// The mirror of `refs_for` (ref-side): which entity a keyword *defines*, where
// its id lives, and whether it defines one entity per line or per block. Lives
// here, next to the table, so identity (`Keyword::id`/`kind`), the defined-id
// sets, and the navigable site index all read one authority. Hand-maintained
// for now; a later codegen pass can populate it from `kwd.json` link/define
// codes (see `docs/keyword-api-plan.md`, phase 4).

/// What entity a keyword defines and where to find its id. Returned by
/// [`definition_of`]; absent for keywords that define nothing (control cards)
/// or that only *modify* an existing entity (`MAT_ADD_*`, `*_ADD_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefSpec {
    /// The entity kind this keyword defines.
    pub kind: EntityKind,
    /// One entity per data line (`*NODE`, `*ELEMENT_*`) vs one per block.
    pub per_line: bool,
    /// Which data card (0-based, past any `_TITLE`) holds the id in field 0.
    pub id_card: usize,
}

enum NameMatch {
    Exact(&'static str),
    Prefix(&'static str),
}

struct DefRule {
    kind: EntityKind,
    m: NameMatch,
    per_line: bool,
    id_card: usize,
}

// Order matters: more specific prefixes first (MAT_THERMAL before MAT_). `base`
// is assumed canonical (uppercased, `_TITLE`/`_ID` stripped) — see
// [`canonical_base`].
static DEF_RULES: &[DefRule] = &[
    DefRule {
        kind: EntityKind::Node,
        m: NameMatch::Exact("NODE"),
        per_line: true,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Element,
        m: NameMatch::Prefix("ELEMENT_"),
        per_line: true,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Part,
        m: NameMatch::Exact("PART"),
        per_line: false,
        id_card: 1,
    },
    DefRule {
        kind: EntityKind::ThermalMaterial,
        m: NameMatch::Prefix("MAT_THERMAL"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Material,
        m: NameMatch::Prefix("MAT_"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Section,
        m: NameMatch::Prefix("SECTION_"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Eos,
        m: NameMatch::Prefix("EOS_"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Hourglass,
        m: NameMatch::Exact("HOURGLASS"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Curve,
        m: NameMatch::Prefix("DEFINE_CURVE"),
        per_line: false,
        id_card: 0,
    },
    // Tables and functions live in the same id space as load curves (an LCID
    // field commonly accepts a table/function id).
    DefRule {
        kind: EntityKind::Curve,
        m: NameMatch::Prefix("DEFINE_TABLE"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Curve,
        m: NameMatch::Prefix("DEFINE_FUNCTION"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Coord,
        m: NameMatch::Prefix("DEFINE_COORDINATE"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Vector,
        m: NameMatch::Prefix("DEFINE_VECTOR"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Box,
        m: NameMatch::Prefix("DEFINE_BOX"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::Transform,
        m: NameMatch::Prefix("DEFINE_TRANSFORM"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::NodeSet,
        m: NameMatch::Prefix("SET_NODE"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::PartSet,
        m: NameMatch::Prefix("SET_PART"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::SegmentSet,
        m: NameMatch::Prefix("SET_SEGMENT"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::ShellSet,
        m: NameMatch::Prefix("SET_SHELL"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::SolidSet,
        m: NameMatch::Prefix("SET_SOLID"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::BeamSet,
        m: NameMatch::Prefix("SET_BEAM"),
        per_line: false,
        id_card: 0,
    },
    DefRule {
        kind: EntityKind::DiscreteSet,
        m: NameMatch::Prefix("SET_DISCRETE"),
        per_line: false,
        id_card: 0,
    },
];

/// A keyword that modifies an existing entity instead of defining a new one, so
/// its id field is a *reference*, not a definition.
fn is_modifier(base: &str) -> bool {
    base.starts_with("MAT_ADD") || base.starts_with("MAT_CHANGE") || base.contains("_ADD_")
}

/// Def-side metadata for a canonical `base` keyword, or `None` if it defines no
/// trackable entity — including modifier keywords, which reference rather than
/// define. The single authority for entity identity and the resolution indices.
pub fn definition_of(base: &str) -> Option<DefSpec> {
    if is_modifier(base) {
        return None;
    }
    DEF_RULES
        .iter()
        .find(|r| match r.m {
            NameMatch::Exact(k) => base == k,
            NameMatch::Prefix(p) => base.starts_with(p),
        })
        .map(|r| DefSpec {
            kind: r.kind,
            per_line: r.per_line,
            id_card: r.id_card,
        })
}

/// Typo-proof `&str` constants for every built-in keyword name, e.g.
/// `dynars::keywords::names::MAT_ELASTIC`.
pub mod names;

/// Generated typed struct per keyword (opt-in — `typed-keywords` feature).
/// E.g. `dynars::keywords::typed::MAT_ELASTIC::parse(&parsed).mid`.
#[cfg(feature = "typed-keywords")]
pub mod typed;

/// Shared row behaviour for the generated typed keyword structs. A keyword's
/// "columns" struct implements just `len` + `row`; `is_empty` and `iter` (and
/// any future row behaviour) are provided here, once, instead of being
/// generated into every struct.
pub trait Columns {
    /// The per-row struct for this keyword.
    type Row;
    fn len(&self) -> usize;
    fn row(&self, i: usize) -> Self::Row;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Iterate rows as owned row structs (array-of-structs view).
    fn iter(&self) -> RowIter<'_, Self>
    where
        Self: Sized,
    {
        RowIter { cols: self, i: 0 }
    }
}

/// Iterator returned by [`Columns::iter`].
pub struct RowIter<'a, C: Columns> {
    cols: &'a C,
    i: usize,
}

impl<C: Columns> Iterator for RowIter<'_, C> {
    type Item = C::Row;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.cols.len() {
            let r = self.cols.row(self.i);
            self.i += 1;
            Some(r)
        } else {
            None
        }
    }
}

/// Field type in the compact static table.
#[derive(Debug, Clone, Copy)]
pub enum T {
    /// integer
    I,
    /// real (float)
    F,
    /// string
    S,
}

impl From<T> for FieldType {
    fn from(t: T) -> Self {
        match t {
            T::I => FieldType::Int,
            T::F => FieldType::Float,
            T::S => FieldType::Str,
        }
    }
}

/// One field: name, type, fixed-format column width, and what it references.
#[derive(Debug, Clone, Copy)]
pub struct Fld {
    pub n: &'static str,
    pub t: T,
    pub w: usize,
    /// The entity this field's id points at, if any (from kwd.json `link`).
    pub r: Ref,
}

/// One card is a slice of fields.
pub type CardDef = &'static [Fld];

/// One keyword: its name and its ordered cards.
#[derive(Debug, Clone, Copy)]
pub struct Kw {
    pub name: &'static str,
    pub cards: &'static [CardDef],
}

impl Kw {
    /// Convert this static definition into a runtime [`Schema`].
    pub fn to_schema(&self) -> Schema {
        let mut s = Schema::new(self.name);
        for &card in self.cards {
            let mut c = Card::new();
            for f in card {
                c.fields.push(FieldSpec {
                    name: f.n.to_string(),
                    ty: f.t.into(),
                    width: f.w,
                    count: 1,
                    reference: f.r,
                });
            }
            s = s.card(c);
        }
        s
    }

    /// The static card governing data row `i` (0-based, past any `_TITLE`) of a
    /// block of this keyword — the one place that knows how a keyword's cards
    /// tile over its rows. A `per_line` definition (`*NODE`, `*ELEMENT_*`) has a
    /// single card that **repeats** over every data row; every other keyword
    /// maps its cards 1:1 (row `i` → card `i`), returning `None` past the last
    /// card. (Repeating tails of list keywords — `*SET_*`, `*DEFINE_CURVE`
    /// points — are still mapped 1:1 for now; those rows fall back to raw
    /// positional access. See `docs/keyword-api-plan.md`.)
    pub fn card_for_row(&self, i: usize) -> Option<CardDef> {
        if definition_of(self.name).is_some_and(|d| d.per_line) {
            self.cards.first().copied()
        } else {
            self.cards.get(i).copied()
        }
    }
}

/// Hand-written definitions for fundamental keywords that pyDYNA's `kwd.json`
/// omits (it handles these through dedicated mesh/geometry APIs, so they never
/// made it into the generated field database). These take precedence over the
/// generated table.
static SUPPLEMENT: &[Kw] = &[
    // *NODE: NID (I8), X Y Z (E16), TC RC (I8) — LS-DYNA standard widths.
    Kw {
        name: "NODE",
        cards: &[&[
            Fld {
                n: "nid",
                t: T::I,
                w: 8,
                r: Ref::None,
            },
            Fld {
                n: "x",
                t: T::F,
                w: 16,
                r: Ref::None,
            },
            Fld {
                n: "y",
                t: T::F,
                w: 16,
                r: Ref::None,
            },
            Fld {
                n: "z",
                t: T::F,
                w: 16,
                r: Ref::None,
            },
            Fld {
                n: "tc",
                t: T::I,
                w: 8,
                r: Ref::None,
            },
            Fld {
                n: "rc",
                t: T::I,
                w: 8,
                r: Ref::None,
            },
        ]],
    },
    // *PART: an A80 heading card, then the part data card (I10 fields).
    Kw {
        name: "PART",
        cards: &[
            &[Fld {
                n: "heading",
                t: T::S,
                w: 80,
                r: Ref::None,
            }],
            &[
                Fld {
                    n: "pid",
                    t: T::I,
                    w: 10,
                    r: Ref::None,
                },
                Fld {
                    n: "secid",
                    t: T::I,
                    w: 10,
                    r: Ref::To(EntityKind::Section),
                },
                Fld {
                    n: "mid",
                    t: T::I,
                    w: 10,
                    r: Ref::To(EntityKind::Material),
                },
                Fld {
                    n: "eosid",
                    t: T::I,
                    w: 10,
                    r: Ref::To(EntityKind::Eos),
                },
                Fld {
                    n: "hgid",
                    t: T::I,
                    w: 10,
                    r: Ref::To(EntityKind::Hourglass),
                },
                Fld {
                    n: "grav",
                    t: T::I,
                    w: 10,
                    r: Ref::None,
                },
                Fld {
                    n: "adpopt",
                    t: T::I,
                    w: 10,
                    r: Ref::None,
                },
                Fld {
                    n: "tmid",
                    t: T::I,
                    w: 10,
                    r: Ref::To(EntityKind::ThermalMaterial),
                },
            ],
        ],
    },
];

/// All generated built-in keyword definitions, sorted by name. (Does not include
/// the hand-written [`SUPPLEMENT`]; use [`find`]/[`schema`] to resolve either.)
pub fn all() -> &'static [Kw] {
    data::KEYWORDS
}

/// The number of built-in keywords (generated + supplement).
pub fn count() -> usize {
    data::KEYWORDS.len() + SUPPLEMENT.len()
}

/// Look up a keyword definition by name (case-insensitive), if built in.
pub fn find(name: &str) -> Option<&'static Kw> {
    let upper = name.to_ascii_uppercase();
    // Hand-written supplements (fundamentals pyDYNA omits) win.
    if let Some(k) = SUPPLEMENT.iter().find(|k| k.name == upper) {
        return Some(k);
    }
    // KEYWORDS is sorted by uppercase name — binary search, then a
    // case-insensitive scan as a fallback.
    if let Ok(i) = data::KEYWORDS.binary_search_by(|k| k.name.cmp(upper.as_str())) {
        return Some(&data::KEYWORDS[i]);
    }
    data::KEYWORDS
        .iter()
        .find(|k| k.name.eq_ignore_ascii_case(name))
}

/// The runtime [`Schema`] for a built-in keyword, if any.
pub fn schema(name: &str) -> Option<Schema> {
    find(name).map(Kw::to_schema)
}

/// Strip trailing pure-annotation options (`_TITLE`, `_ID`) and uppercase, to
/// get the base keyword — so a lookup/rule on `SECTION_SHELL` matches a
/// `SECTION_SHELL_TITLE` block. The single source of truth for name folding,
/// shared by the resolver, validation, and the schema row iterator.
pub fn canonical_base(name: &str) -> String {
    let mut s = name.to_ascii_uppercase();
    for opt in ["_TITLE", "_ID"] {
        if let Some(stripped) = s.strip_suffix(opt) {
            s = stripped.to_string();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_offsets_route_ids_to_the_right_bucket() {
        let t = TransformOffsets {
            idnoff: 100,
            ideoff: 200,
            idpoff: 300,
            idsoff: 500,
            idfoff: 600,
            ..TransformOffsets::IDENTITY
        };
        // Each kind draws from its own offset; a zero/blank id is "none".
        assert_eq!(t.apply(7, EntityKind::Node), 107);
        assert_eq!(t.apply(7, EntityKind::Element), 207);
        assert_eq!(t.apply(7, EntityKind::Part), 307);
        assert_eq!(t.apply(7, EntityKind::PartSet), 307); // part set shares IDPOFF
        assert_eq!(t.apply(7, EntityKind::NodeSet), 507);
        assert_eq!(t.apply(7, EntityKind::Curve), 607);
        assert_eq!(t.apply(0, EntityKind::Node), 0); // 0 = unset, untouched
        // A kind with no configured offset is left alone.
        assert_eq!(t.apply(7, EntityKind::Material), 7);
    }

    #[test]
    fn transform_offsets_preserve_sign_of_references() {
        let t = TransformOffsets {
            idfoff: 1000,
            ..TransformOffsets::IDENTITY
        };
        // A signed reference (e.g. -curve) denotes entity |id|: shift the
        // magnitude, keep the sign, so it still resolves to the shifted def.
        assert_eq!(t.apply(-5, EntityKind::Curve), -1005);
        assert_eq!(t.apply(5, EntityKind::Curve), 1005);
    }

    #[test]
    fn transform_offsets_compose_additively() {
        let outer = TransformOffsets {
            idnoff: 1000,
            ..TransformOffsets::IDENTITY
        };
        let inner = TransformOffsets {
            idnoff: 40,
            ideoff: 7,
            ..TransformOffsets::IDENTITY
        };
        // Nested transforms accumulate: a node two levels deep is shifted by both.
        let eff = outer.compose(inner);
        assert_eq!(eff.apply(2, EntityKind::Node), 1042);
        assert_eq!(eff.apply(2, EntityKind::Element), 9);
        assert!(!eff.is_identity());
        assert!(TransformOffsets::IDENTITY.is_identity());
    }

    #[test]
    fn registry_is_populated_and_sorted() {
        assert!(
            count() > 1000,
            "expected thousands of keywords, got {}",
            count()
        );
        let names: Vec<_> = all().iter().map(|k| k.name).collect();
        assert!(
            names.windows(2).all(|w| w[0] <= w[1]),
            "KEYWORDS must be sorted"
        );
    }

    #[test]
    fn mat_elastic_has_expected_fields() {
        let s = schema("MAT_ELASTIC").expect("MAT_ELASTIC is built in");
        assert_eq!(s.keyword, "MAT_ELASTIC");
        let f0 = &s.cards[0].fields;
        assert_eq!(f0[0].name, "MID");
        assert!(matches!(f0[0].ty, FieldType::Int));
        assert_eq!(f0[0].width, 10);
        assert_eq!(f0[1].name, "RO");
        assert!(matches!(f0[1].ty, FieldType::Float));
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert!(schema("mat_elastic").is_some());
        assert!(schema("Element_Shell").is_some());
        assert!(schema("NOT_A_REAL_KEYWORD_XYZ").is_none());
    }

    #[test]
    fn supplement_covers_node_and_part() {
        // pyDYNA omits these; the hand-written supplement fills them in.
        let node = schema("NODE").expect("NODE supplement");
        assert_eq!(node.cards[0].fields[0].name, "nid");
        assert_eq!(node.cards[0].fields[1].width, 16); // x is E16
        let part = schema("PART").expect("PART supplement");
        assert_eq!(part.cards.len(), 2); // heading + data
        assert_eq!(part.cards[0].fields[0].name, "heading");
    }

    #[test]
    fn built_in_schema_parses_a_real_deck() {
        use crate::file::ParsedFile;
        use crate::parser::split_blocks;
        use crate::schema::parse_schema;

        let src = b"*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n";
        let p = ParsedFile::new("d.k".into(), src.to_vec(), split_blocks(src));
        let t = parse_schema(&p, &schema("MAT_ELASTIC").unwrap());
        assert_eq!(t.rows(), 1);
        assert_eq!(t.column("MID").unwrap().as_int().unwrap(), &[1]);
        assert_eq!(t.column("E").unwrap().as_float().unwrap(), &[210000.0]);
        assert_eq!(t.column("PR").unwrap().as_float().unwrap(), &[0.3]);
    }

    #[test]
    fn definition_of_classifies_defs_modifiers_and_controls() {
        // per-line definitions
        assert_eq!(
            definition_of("NODE"),
            Some(DefSpec {
                kind: EntityKind::Node,
                per_line: true,
                id_card: 0
            })
        );
        let el = definition_of("ELEMENT_SHELL").expect("ELEMENT_SHELL defines elements");
        assert!(el.per_line && el.kind == EntityKind::Element);

        // per-block definitions
        let mat = definition_of("MAT_ELASTIC").expect("MAT_ELASTIC defines a material");
        assert_eq!(mat.kind, EntityKind::Material);
        assert!(!mat.per_line);
        assert_eq!(definition_of("PART").unwrap().id_card, 1); // id on the 2nd card

        // rule specificity: MAT_THERMAL_* resolves before the generic MAT_*
        assert_eq!(
            definition_of("MAT_THERMAL_ISOTROPIC").unwrap().kind,
            EntityKind::ThermalMaterial
        );

        // modifiers reference rather than define → no DefSpec
        assert!(definition_of("MAT_ADD_EROSION").is_none());
        assert!(definition_of("MAT_CHANGE_SOLID_TYPE").is_none());
        // control cards define no trackable entity
        assert!(definition_of("CONTROL_TERMINATION").is_none());
    }

    #[test]
    fn card_for_row_repeats_per_line_and_maps_others_one_to_one() {
        // A per-line keyword's single card governs every data row.
        let node = find("NODE").unwrap();
        assert_eq!(node.card_for_row(0).unwrap()[0].n, "nid");
        assert_eq!(node.card_for_row(5).unwrap()[0].n, "nid");

        // A fixed multi-card keyword maps 1:1, with nothing past the last card.
        let part = find("PART").unwrap();
        assert_eq!(part.card_for_row(0).unwrap()[0].n, "heading");
        assert_eq!(part.card_for_row(1).unwrap()[0].n, "pid");
        assert!(part.card_for_row(2).is_none());
    }
}
