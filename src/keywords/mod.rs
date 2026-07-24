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

/// One field: name, type, fixed-format column width.
#[derive(Debug, Clone, Copy)]
pub struct Fld {
    pub n: &'static str,
    pub t: T,
    pub w: usize,
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
                });
            }
            s = s.card(c);
        }
        s
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
            Fld { n: "nid", t: T::I, w: 8 },
            Fld { n: "x", t: T::F, w: 16 },
            Fld { n: "y", t: T::F, w: 16 },
            Fld { n: "z", t: T::F, w: 16 },
            Fld { n: "tc", t: T::I, w: 8 },
            Fld { n: "rc", t: T::I, w: 8 },
        ]],
    },
    // *PART: an A80 heading card, then the part data card (I10 fields).
    Kw {
        name: "PART",
        cards: &[
            &[Fld { n: "heading", t: T::S, w: 80 }],
            &[
                Fld { n: "pid", t: T::I, w: 10 },
                Fld { n: "secid", t: T::I, w: 10 },
                Fld { n: "mid", t: T::I, w: 10 },
                Fld { n: "eosid", t: T::I, w: 10 },
                Fld { n: "hgid", t: T::I, w: 10 },
                Fld { n: "grav", t: T::I, w: 10 },
                Fld { n: "adpopt", t: T::I, w: 10 },
                Fld { n: "tmid", t: T::I, w: 10 },
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
    if let Ok(i) = data::KEYWORDS.binary_search_by(|k| k.name.cmp(&upper.as_str())) {
        return Some(&data::KEYWORDS[i]);
    }
    data::KEYWORDS.iter().find(|k| k.name.eq_ignore_ascii_case(name))
}

/// The runtime [`Schema`] for a built-in keyword, if any.
pub fn schema(name: &str) -> Option<Schema> {
    find(name).map(Kw::to_schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated_and_sorted() {
        assert!(count() > 1000, "expected thousands of keywords, got {}", count());
        let names: Vec<_> = all().iter().map(|k| k.name).collect();
        assert!(names.windows(2).all(|w| w[0] <= w[1]), "KEYWORDS must be sorted");
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
        use crate::keyword::ParsedFile;
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
}
