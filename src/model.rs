//! A resolved cross-keyword model of a deck: which ids each keyword **defines**
//! (materials, sections, parts, curves, nodes, sets, …) and which fields
//! **reference** them (via the `Ref` metadata on each [`Fld`](crate::keywords::Fld)).
//!
//! This is the shared resolution core, exposed as methods on [`Deck`]: the
//! dangling-reference check (`Deck::dangling`, behind [`Deck::validate`]) and
//! navigation ([`Deck::part`], [`Entity::material`], …) are two queries over the
//! same cached indices. Building them is lazy — a plain parse pays for neither;
//! validation builds the defined-id sets, navigation builds the site map, each
//! on first use.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rayon::prelude::*;

use crate::deck::Deck;
use crate::file::{Block, CardFormat, ParsedFile};
use crate::keywords::{self, EntityKind, Ref, TransformOffsets, canonical_base};
use crate::parser::Field as RawField;
use crate::schema::{FieldSpec, FieldType, Schema, Table, parse_schema_files};

// Which keywords define which entity ids (and where the id sits) now lives with
// the table, as `keywords::definition_of` — see that module. The resolution core
// below reads it so identity, defined-id sets, and the site index share one
// authority.

// ── Low-level tolerant field reading (any card, missing cards OK) ─────────────
//
// The byte-level plumbing — blank/comment detection, EOL stripping, free-format
// detection, fixed-width slicing — is shared with the schema marshaller through
// its `__` helpers, so there is one implementation. This layer adds only the
// "read field N of a card by index" addressing and the float-tolerant id parse.

/// A block's data-card lines (comments/blanks removed, `\r` trimmed).
fn data_lines<'a>(parsed: &'a ParsedFile, block: &Block) -> Vec<&'a [u8]> {
    parsed
        .body(block)
        .split(|&b| b == b'\n')
        .map(crate::schema::__strip_eol)
        .filter(|l| !crate::schema::__is_skippable(l))
        .collect()
}

/// Parse an id field as an integer, tolerating a float-formatted id (`7.0` → 7)
/// as some decks emit for load-curve / entity references. Uses the same
/// [`Field`] parser as the schema marshaller.
fn parse_i64(s: &[u8]) -> Option<i64> {
    let f = RawField { raw: s };
    f.as_i64().or_else(|| f.as_f64().map(|v| v as i64))
}

/// The raw (untrimmed) byte slice for field `idx` of a card, width-aware.
fn card_field_slice<'a>(
    line: &'a [u8],
    card: &[keywords::Fld],
    idx: usize,
    fmt: CardFormat,
) -> Option<&'a [u8]> {
    if idx >= card.len() {
        return None;
    }
    if crate::schema::__is_free(line, fmt) {
        return line.split(|&c| c == b',').nth(idx);
    }
    let width = |w: usize| if fmt == CardFormat::Long { w * 2 } else { w };
    let off: usize = card[..idx].iter().map(|f| width(f.w)).sum();
    if off >= line.len() {
        return None;
    }
    Some(crate::schema::__slice(line, off, width(card[idx].w)))
}

/// Read field `idx` of a card as an integer, using the schema's per-field
/// widths (the generic `split_fields` assumes a uniform 8-col width).
fn card_field_i64(line: &[u8], card: &[keywords::Fld], idx: usize, fmt: CardFormat) -> Option<i64> {
    card_field_slice(line, card, idx, fmt).and_then(parse_i64)
}

/// Options + title handling.
fn title_offset(exact_kw: &str) -> usize {
    usize::from(exact_kw.to_ascii_uppercase().ends_with("_TITLE"))
}

/// The 1-based line number of every block's `*KEYWORD` line, computed in a
/// single pass over the file. Blocks tile the source in order, so one cursor
/// walk counts each newline exactly once — replacing a per-block
/// scan-from-file-start that was O(blocks²) on decks with many small
/// reference-bearing keywords (tens of thousands of `*BOUNDARY_*` / `*LOAD_*` /
/// set cards). Built lazily by the reference check, only when a file actually
/// has a finding-eligible block.
fn block_start_lines(file: &ParsedFile) -> Vec<usize> {
    let src = file.src();
    let mut out = Vec::with_capacity(file.blocks.len());
    let mut pos = 0usize;
    let mut line = 1usize;
    for b in &file.blocks {
        line += memchr::memchr_iter(b'\n', &src[pos..b.name_start]).count();
        out.push(line);
        pos = b.name_start;
    }
    out
}

/// Read the id offsets off an `*INCLUDE_TRANSFORM` block, using the keyword
/// table's own field widths (`IDNOFF … IDDOFF` on card 1, `IDROFF` on card 2 —
/// all `I10`). Called from the parser as it records each directive; a malformed
/// or truncated card just yields `0` for the missing fields (identity).
///
/// The keyword's cards map one-to-one onto the block's data lines (no `_TITLE`),
/// so card index == line index: line 0 is the filename, line 1 the first offset
/// card, line 2 the second.
pub(crate) fn read_transform_offsets(file: &ParsedFile, block: &Block) -> TransformOffsets {
    let Some(kw) = keywords::find("INCLUDE_TRANSFORM") else {
        return TransformOffsets::IDENTITY;
    };
    let lines = data_lines(file, block);
    let fmt = block.format;
    let read = |card_idx: usize, field_idx: usize| -> i64 {
        match (kw.cards.get(card_idx), lines.get(card_idx)) {
            (Some(card), Some(line)) => card_field_i64(line, card, field_idx, fmt).unwrap_or(0),
            _ => 0,
        }
    };
    TransformOffsets {
        idnoff: read(1, 0),
        ideoff: read(1, 1),
        idpoff: read(1, 2),
        idmoff: read(1, 3),
        idsoff: read(1, 4),
        idfoff: read(1, 5),
        iddoff: read(1, 6),
        idroff: read(2, 0),
    }
}

// ── The resolution core ──────────────────────────────────────────────────────

/// A reference that resolves to nothing defined in the deck.
#[derive(Debug, Clone)]
pub struct Dangling {
    pub from_keyword: String,
    pub field: String,
    pub target: Ref,
    pub id: i64,
    pub file: PathBuf,
    pub line: usize,
}

/// An FxHash-style hasher for the defined-id sets. They hold tens of millions of
/// `i64`s and are probed once per element→node/part reference during validation,
/// where the default SipHash dominated the cost; a single-multiply integer hash
/// spreads LS-DYNA's dense id ranges well and is several times faster.
#[derive(Default, Clone, Copy)]
pub(crate) struct BuildIntHasher;

impl std::hash::BuildHasher for BuildIntHasher {
    type Hasher = IntHasher;
    #[inline]
    fn build_hasher(&self) -> IntHasher {
        IntHasher(0)
    }
}

pub(crate) struct IntHasher(u64);

impl IntHasher {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
}

impl std::hash::Hasher for IntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(Self::K);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for non-integer keys (unused by the `i64` id sets).
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(Self::K);
        }
    }
}

/// A set of defined entity ids. LS-DYNA ids are non-negative and, in practice,
/// densely packed — `1..N`, or a few contiguous ranges — so a **bitset** keyed
/// by id builds and probes far faster than a hash set (the connectivity check
/// does tens of millions of membership tests). The bitset is offset by the
/// minimum id, so a *high but compact* id range (e.g. ids around `10^9`) still
/// uses it — absolute magnitude doesn't matter, only the span. Genuinely sparse
/// ids (a small count spread over a huge span) fall back to a hash set.
pub(crate) enum IdSet {
    /// Dense bitset covering `min..=max`: bit `i` set ⇔ id `min + i` is defined.
    Bits {
        words: Box<[u64]>,
        min: i64,
        max: i64,
    },
    /// Sparse fallback.
    Hash(HashSet<i64, BuildIntHasher>),
}

impl IdSet {
    /// Cap on the id *span* (`max - min`), independent of id magnitude — a bitset
    /// this wide is 128 MiB. The span must also stay within 128× the id count, so
    /// a lone far-away id can't blow the allocation up.
    const SPAN_CAP: u64 = 1 << 30;

    #[inline]
    pub(crate) fn contains(&self, id: i64) -> bool {
        match self {
            IdSet::Bits { words, min, max } => {
                if id < *min || id > *max {
                    return false;
                }
                let bit = (id - *min) as usize;
                (words[bit >> 6] >> (bit & 63)) & 1 != 0
            }
            IdSet::Hash(s) => s.contains(&id),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            IdSet::Bits { words, .. } => words.iter().map(|w| w.count_ones() as usize).sum(),
            IdSet::Hash(s) => s.len(),
        }
    }

    /// Build from per-file id chunks — an offset bitset when ids are dense
    /// enough, else a hash set.
    fn build(chunks: Vec<Vec<i64>>) -> Self {
        let (mut min, mut max, mut total) = (i64::MAX, -1i64, 0u64);
        for &id in chunks.iter().flatten() {
            if id >= 0 {
                min = min.min(id);
                max = max.max(id);
                total += 1;
            }
        }
        let span = if max >= 0 { (max - min) as u64 + 1 } else { 0 };
        if max >= 0 && span <= Self::SPAN_CAP && span <= 128 * total {
            let mut words = vec![0u64; span.div_ceil(64) as usize];
            for &id in chunks.iter().flatten() {
                if id >= 0 {
                    let bit = (id - min) as usize;
                    words[bit >> 6] |= 1u64 << (bit & 63);
                }
            }
            IdSet::Bits {
                words: words.into_boxed_slice(),
                min,
                max,
            }
        } else {
            let n: usize = chunks.iter().map(Vec::len).sum();
            let mut s = HashSet::with_capacity_and_hasher(n, BuildIntHasher);
            for c in chunks {
                s.extend(c);
            }
            IdSet::Hash(s)
        }
    }
}

/// Defined ids grouped by entity kind — the shared resolution core, cached on
/// the [`Deck`] and reused by validation and navigation alike.
pub(crate) type Defs = HashMap<EntityKind, IdSet>;

/// Every defined id in the deck, per kind.
///
/// Extract ids into plain `Vec`s per file (parallel, no hashing), then build
/// each kind's [`IdSet`] exactly once — in parallel across kinds. The previous
/// `map(collect)+reduce(merge)` re-hashed every id through a merge tree
/// (≈`log(files)` times); this hashes each id once, which dominates the index
/// build on a large mesh.
pub(crate) fn build_defs(deck: &Deck) -> Defs {
    let transforms = deck.file_transforms();
    let per_file: Vec<HashMap<EntityKind, Vec<i64>>> = deck
        .files
        .par_iter()
        .zip(transforms.par_iter())
        .map(|(f, transform)| collect_def_ids(f, transform.as_ref()))
        .collect();

    // Gather each kind's id chunks (moves `Vec`s, no element copy).
    let mut by_kind: HashMap<EntityKind, Vec<Vec<i64>>> = HashMap::new();
    for m in per_file {
        for (k, v) in m {
            by_kind.entry(k).or_default().push(v);
        }
    }

    // Each kind's set is built once, sharded across cores (see `IdSet::build`).
    by_kind
        .into_iter()
        .map(|(k, chunks)| (k, IdSet::build(chunks)))
        .collect()
}

impl Deck {
    /// The defined-id sets, built once on first use and cached.
    pub(crate) fn definitions(&self) -> &Defs {
        self.defs.get_or_init(|| build_defs(self))
    }

    /// Effective `*INCLUDE_TRANSFORM` offsets for each file, parallel to
    /// [`Deck::files`] — `None` where a file has no transform (the common case,
    /// kept off the offset path entirely). Built once on first use and cached.
    pub(crate) fn file_transforms(&self) -> &[Option<TransformOffsets>] {
        self.file_transforms
            .get_or_init(|| compute_file_transforms(self))
    }

    /// The effective transform for file `file`, or `None` when it applies no
    /// shift. The one place navigation and resolution turn a *physical* id (as
    /// written in that file) into its *logical* (global) id.
    pub(crate) fn transform_of(&self, file: usize) -> Option<&TransformOffsets> {
        self.file_transforms().get(file).and_then(Option::as_ref)
    }

    /// References that point at an id nothing defines. `connectivity` includes
    /// element→node references (millions on a big mesh) — off keeps it cheap.
    /// Reuses the cached definition sets, so a prior `check`/navigation call is
    /// not re-done here.
    pub(crate) fn dangling(&self, connectivity: bool) -> Vec<Dangling> {
        let defs = self.definitions();
        let transforms = self.file_transforms();
        // `flat_map_iter`, not `flat_map`: each file yields a small (usually
        // empty) `Vec<Dangling>`, so we want the *files* parallelised and the
        // per-file results flattened sequentially — `flat_map` treats each
        // result as its own parallel iterator and schedules far worse here.
        self.files
            .par_iter()
            .zip(transforms.par_iter())
            .flat_map_iter(|(f, transform)| {
                check_refs(
                    f,
                    defs,
                    &self.user_schemas,
                    connectivity,
                    transform.as_ref(),
                )
            })
            .collect()
    }

    /// Number of defined ids of each kind (for reporting), most-numerous first.
    pub fn definition_counts(&self) -> Vec<(EntityKind, usize)> {
        let mut v: Vec<_> = self
            .definitions()
            .iter()
            .map(|(k, s)| (*k, s.len()))
            .collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        v
    }
}

/// [`Deck::transforms`] with identity collapsed to `None`, parallel to
/// [`Deck::files`].
///
/// `None` means "no shift" — the root, plain `*INCLUDE`s, and any file whose
/// composed offsets happen to cancel to identity. This keeps the resolution
/// core's hot paths on their existing zero-offset branch for the common
/// (transform-free) deck.
///
/// The effective offsets themselves are composed by the walker as it traverses
/// (each node carries the composition down its include path), so a file
/// instanced at two different offsets has two `files` entries with two distinct
/// transforms here — this just drops the identity ones. See
/// [`walk_includes`](crate::include::walk_includes).
fn compute_file_transforms(deck: &Deck) -> Vec<Option<TransformOffsets>> {
    deck.transforms
        .iter()
        .map(|t| (!t.is_identity()).then_some(*t))
        .collect()
}

/// Extract a file's defined ids grouped by kind, as plain `Vec`s (no hashing —
/// dedup/hashing happens once in [`build_defs`]).
///
/// Ids are stored *logical* (post-transform): under `*INCLUDE_TRANSFORM`,
/// `transform` shifts each id by its kind's offset, so the whole deck shares one
/// global id namespace and the dangling check never has to know a transform was
/// involved. `transform` is `None` for the transform-free common case, which
/// stays on the plain push path.
fn collect_def_ids(
    file: &ParsedFile,
    transform: Option<&TransformOffsets>,
) -> HashMap<EntityKind, Vec<i64>> {
    let mut out: HashMap<EntityKind, Vec<i64>> = HashMap::new();
    for block in &file.blocks {
        let exact = file.keyword_name(block);
        let base = canonical_base(exact);
        // `definition_of` returns `None` for control cards and for modifier
        // keywords (MAT_ADD_*, *_ADD_*, …), which reference rather than define.
        let Some(def) = keywords::definition_of(&base) else {
            continue;
        };
        let Some(kw) = keywords::find(&base) else {
            continue;
        };

        let id_card = kw.cards.get(def.id_card).copied().unwrap_or(&[]);

        // Offset for this kind, hoisted out of the per-id loop (0 = no shift).
        let off = transform.map_or(0, |t| t.for_kind(def.kind));
        let lines = data_lines(file, block);
        let title = title_offset(exact);
        let ids = out.entry(def.kind).or_default();
        if def.per_line {
            let card0 = kw.cards.first().copied().unwrap_or(&[]);
            for line in lines.iter().skip(title) {
                if let Some(id) = card_field_i64(line, card0, 0, block.format)
                    && id != 0
                {
                    // Def ids are non-negative, so a plain add is the shift.
                    ids.push(if off == 0 { id } else { id + off });
                }
            }
        } else if let Some(line) = lines.get(title + def.id_card)
            && let Some(id) = card_field_i64(line, id_card, 0, block.format)
            && id != 0
        {
            ids.push(if off == 0 { id } else { id + off });
        }
    }
    out
}

/// Does reference `id` (as written in the file) resolve to nothing defined?
///
/// `transform` is the file's transform (or `None`). The membership test is done
/// on the *logical* id — the physical `id` shifted by the offset for the kind
/// actually being probed — so an `*INCLUDE_TRANSFORM`'d reference is matched
/// against the same shifted defs. For a polymorphic [`Ref::AnyOf`], each
/// candidate kind is shifted by *its own* bucket before probing that kind's set.
/// The caller still reports the physical `id`; only resolution is
/// transform-aware.
fn is_dangling(defs: &Defs, r: &Ref, id: i64, transform: Option<&TransformOffsets>) -> bool {
    // Conservative: only flag when we actually track the target kind (else the
    // entity type is externally defined / untracked — don't raise noise).
    // A negative id references the entity |id| (LS-DYNA convention, esp. curves).
    let probe = |s: &IdSet, logical: i64| s.contains(logical) || s.contains(logical.abs());
    match transform {
        // Fast path: no transform on this file — the original zero-offset logic.
        None => match r {
            Ref::None => false,
            Ref::To(k) => defs.get(k).is_some_and(|s| !probe(s, id)),
            Ref::AnyOf(ks) => {
                let tracked: Vec<&IdSet> = ks.iter().filter_map(|k| defs.get(k)).collect();
                !tracked.is_empty() && !tracked.iter().any(|s| probe(s, id))
            }
        },
        Some(transform) => match r {
            Ref::None => false,
            Ref::To(k) => defs
                .get(k)
                .is_some_and(|s| !probe(s, transform.apply(id, *k))),
            Ref::AnyOf(ks) => {
                let tracked: Vec<(&IdSet, i64)> = ks
                    .iter()
                    .filter_map(|k| defs.get(k).map(|s| (s, transform.apply(id, *k))))
                    .collect();
                !tracked.is_empty() && !tracked.iter().any(|(s, l)| probe(s, *l))
            }
        },
    }
}

fn check_refs(
    file: &ParsedFile,
    defs: &Defs,
    user_schemas: &HashMap<String, Schema>,
    connectivity: bool,
    transform: Option<&TransformOffsets>,
) -> Vec<Dangling> {
    let mut out = Vec::new();
    // 1-based `*KEYWORD` line for each block, built once on first use (O(bytes))
    // and indexed O(1) thereafter — see `block_start_lines`.
    let mut block_lines: Option<Vec<usize>> = None;
    for (bi, block) in file.blocks.iter().enumerate() {
        let exact = file.keyword_name(block);
        let base = canonical_base(exact);
        // A registered user schema wins — check the references it declares.
        if let Some(schema) = user_schemas.get(&base) {
            let line0 = block_lines.get_or_insert_with(|| block_start_lines(file))[bi];
            check_refs_user(file, block, schema, defs, transform, line0, &mut out);
            continue;
        }
        let Some(kw) = keywords::find(&base) else {
            continue;
        };
        if kw
            .cards
            .iter()
            .flat_map(|c| c.iter())
            .all(|f| matches!(f.r, Ref::None))
        {
            continue; // no references on this keyword
        }

        // Element connectivity is per-line and high-cardinality — gated.
        let per_line = base.starts_with("ELEMENT_");
        if per_line && !connectivity {
            continue;
        }

        let lines = data_lines(file, block);
        let title = title_offset(exact);
        // Line of the block's `*KEYWORD` — from the file's precomputed table, so
        // it's O(1) here rather than a fresh scan-from-start per block.
        let block_line0 = block_lines.get_or_insert_with(|| block_start_lines(file))[bi];
        let line_no = |ln: usize| block_line0 + ln;

        if per_line {
            // Element cards: all ref fields on card 0, one element per line. This
            // is the hot path (millions of rows), so walk each line once: decide
            // free/fixed a single time, and advance the fixed-format offset
            // incrementally rather than re-summing field widths per field.
            let card0 = kw.cards.first().copied().unwrap_or(&[]);
            let long = block.format == CardFormat::Long;
            for (row, line) in lines.iter().enumerate().skip(title) {
                if crate::schema::__is_free(line, block.format) {
                    let mut toks = line.split(|&c| c == b',');
                    for f in card0 {
                        let Some(tok) = toks.next() else { break };
                        if !matches!(f.r, Ref::None)
                            && let Some(v) = parse_i64(tok)
                            && v != 0
                            && is_dangling(defs, &f.r, v, transform)
                        {
                            out.push(Dangling {
                                from_keyword: base.clone(),
                                field: f.n.to_string(),
                                target: f.r,
                                id: v,
                                file: file.path.clone(),
                                line: line_no(row),
                            });
                        }
                    }
                } else {
                    let mut off = 0usize;
                    for f in card0 {
                        if off >= line.len() {
                            break;
                        }
                        let w = if long { f.w * 2 } else { f.w };
                        if !matches!(f.r, Ref::None)
                            && let Some(v) = parse_i64(crate::schema::__slice(line, off, w))
                            && v != 0
                            && is_dangling(defs, &f.r, v, transform)
                        {
                            out.push(Dangling {
                                from_keyword: base.clone(),
                                field: f.n.to_string(),
                                target: f.r,
                                id: v,
                                file: file.path.clone(),
                                line: line_no(row),
                            });
                        }
                        off += w;
                    }
                }
            }
        } else {
            for (ci, card) in kw.cards.iter().enumerate() {
                let Some(line) = lines.get(title + ci) else {
                    break;
                };
                for (fi, f) in card.iter().enumerate() {
                    if matches!(f.r, Ref::None) {
                        continue;
                    }
                    let Some(v) = card_field_i64(line, card, fi, block.format) else {
                        continue;
                    };
                    if v != 0 && is_dangling(defs, &f.r, v, transform) {
                        out.push(Dangling {
                            from_keyword: base.clone(),
                            field: f.n.to_string(),
                            target: f.r,
                            id: v,
                            file: file.path.clone(),
                            line: line_no(title + ci),
                        });
                    }
                }
            }
        }
    }
    out
}

/// Dangling-reference check for a block governed by a **user schema** (not the
/// built-in table). Reads the `Ref` fields the schema declares
/// ([`Card::ref_to`](crate::schema::Card::ref_to)) at each data row and flags
/// ids that resolve to nothing defined. Row→card tiling matches the navigation
/// spine (a single repeating card governs every row). User keywords reference
/// built-in entities but do not themselves define any, so `defs` is unchanged.
fn check_refs_user(
    file: &ParsedFile,
    block: &Block,
    schema: &Schema,
    defs: &Defs,
    transform: Option<&TransformOffsets>,
    block_line0: usize,
    out: &mut Vec<Dangling>,
) {
    if schema
        .cards
        .iter()
        .flat_map(|c| &c.fields)
        .all(|f| matches!(f.reference, Ref::None))
    {
        return; // no references declared
    }
    let exact = file.keyword_name(block);
    let base = canonical_base(exact);
    let title = title_offset(exact);
    let lines = data_lines(file, block);
    // `block_line0` (the `*KEYWORD` line) is supplied by the caller from its
    // one-pass table — no per-block scan-from-start here either.
    let line_no = |ln: usize| block_line0 + ln;

    for (row, line) in lines.iter().enumerate().skip(title) {
        let Some(fields) = schema.card_for_row(row - title) else {
            continue;
        };
        let card = CardRef::User(fields);
        for col in 0..card.len() {
            let r = card.ref_of(col);
            if matches!(r, Ref::None) {
                continue;
            }
            let Some(v) = card
                .field_slice(line, col, block.format)
                .and_then(parse_i64)
            else {
                continue;
            };
            if v != 0 && is_dangling(defs, &r, v, transform) {
                out.push(Dangling {
                    from_keyword: base.to_string(),
                    field: card.name(col).unwrap_or("").to_string(),
                    target: r,
                    id: v,
                    file: file.path.clone(),
                    line: line_no(row),
                });
            }
        }
    }
}

// ── Navigation graph: entity handles that follow references ──────────────────

use crate::keywords::T;

/// A typed field value read from an entity's card.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Str(_) => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(_) => None,
        }
    }
    /// The string payload, for `Str` values only (not a stringification — use
    /// [`display`](Value::display) for that).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    /// Render any value as a string (for messages / describe output).
    pub fn display(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Str(s) => s.clone(),
        }
    }
}

/// `(kind, id) -> (file index, block index)` for every definition entity
/// (parts, materials, sections, curves, sets, …). High-cardinality per-line
/// entities (nodes, elements) are intentionally excluded — navigating those is
/// a scan, not a lookup. Cached on the [`Deck`]; backs [`Deck::part`] & friends.
pub(crate) type Sites = HashMap<(EntityKind, i64), (usize, usize)>;

/// Index every definition entity (parts, materials, sections, curves, sets, …)
/// by `(kind, id)` → its defining block. Skips per-line entities (nodes,
/// elements) and modifier keywords.
///
/// Ids are keyed *logical* (post-`*INCLUDE_TRANSFORM`), matching the defined-id
/// sets, so `Deck::get` and reference-following both work in the deck's global
/// id namespace — a node/part in a transformed include is found at its offset id.
pub(crate) fn build_sites(deck: &Deck) -> Sites {
    let mut sites = Sites::new();
    let transforms = deck.file_transforms();
    for (fi, file) in deck.files.iter().enumerate() {
        let transform = transforms[fi].as_ref();
        for (bi, block) in file.blocks.iter().enumerate() {
            let exact = file.keyword_name(block);
            let base = canonical_base(exact);
            // Only per-block definitions are navigable by id; per-line entities
            // (nodes, elements) and modifiers (`None` from `definition_of`) are
            // scanned, not indexed.
            let Some(def) = keywords::definition_of(&base) else {
                continue;
            };
            if def.per_line {
                continue;
            }
            let Some(kw) = keywords::find(&base) else {
                continue;
            };
            let id_card = kw.cards.get(def.id_card).copied().unwrap_or(&[]);
            let title = title_offset(exact);
            let lines = data_lines(file, block);
            if let Some(line) = lines.get(title + def.id_card)
                && let Some(id) = card_field_i64(line, id_card, 0, block.format)
                && id != 0
            {
                let id = transform.map_or(id, |t| t.apply(id, def.kind));
                sites.entry((def.kind, id)).or_insert((fi, bi));
            }
        }
    }
    sites
}

/// Resolve `(kind, id)` to a site, honouring the negative-`id` (`|id|`) convention.
pub(crate) fn site_of(sites: &Sites, kind: EntityKind, id: i64) -> Option<(usize, usize)> {
    sites
        .get(&(kind, id))
        .or_else(|| sites.get(&(kind, id.abs())))
        .copied()
}

/// A definition entity id claimed by more than one block — an id collision.
pub(crate) struct DuplicateDef {
    pub kind: EntityKind,
    /// The logical (post-`*INCLUDE_TRANSFORM`) id defined more than once.
    pub id: i64,
    /// Every block that defines it, `(file, block)`, in deck order.
    pub sites: Vec<(usize, usize)>,
}

impl Deck {
    /// Per-block definition entities (parts, materials, sections, sets, curves,
    /// …) whose `(kind, logical id)` is claimed by more than one block. Ids are
    /// compared *logical* (post-transform), so the same id reused across two
    /// `*INCLUDE_TRANSFORM` instances at different offsets is **not** a collision.
    ///
    /// Per-line entities (nodes, elements) are out of scope here — a duplicate
    /// node/element id is better surfaced by a mesh-oriented check that also
    /// catches coincident geometry; this walks the labelled, block-level id space
    /// where numbering collisions actually bite (LS-DYNA rejects some outright).
    pub(crate) fn duplicate_definitions(&self) -> Vec<DuplicateDef> {
        let transforms = self.file_transforms();
        let mut by_id: HashMap<(EntityKind, i64), Vec<(usize, usize)>> = HashMap::new();
        for (fi, file) in self.files.iter().enumerate() {
            let transform = transforms[fi].as_ref();
            for (bi, block) in file.blocks.iter().enumerate() {
                let exact = file.keyword_name(block);
                let base = canonical_base(exact);
                let Some(def) = keywords::definition_of(&base) else {
                    continue;
                };
                if def.per_line {
                    continue;
                }
                let Some(kw) = keywords::find(&base) else {
                    continue;
                };
                let id_card = kw.cards.get(def.id_card).copied().unwrap_or(&[]);
                let title = title_offset(exact);
                let lines = data_lines(file, block);
                if let Some(line) = lines.get(title + def.id_card)
                    && let Some(id) = card_field_i64(line, id_card, 0, block.format)
                    && id != 0
                {
                    let id = transform.map_or(id, |t| t.apply(id, def.kind));
                    by_id.entry((def.kind, id)).or_default().push((fi, bi));
                }
            }
        }
        let mut dups: Vec<DuplicateDef> = by_id
            .into_iter()
            .filter(|(_, sites)| sites.len() > 1)
            .map(|((kind, id), sites)| DuplicateDef { kind, id, sites })
            .collect();
        // Deterministic: order by first defining site, then kind, then id.
        dups.sort_by(|a, b| {
            (a.sites[0], a.kind as usize, a.id).cmp(&(b.sites[0], b.kind as usize, b.id))
        });
        dups
    }

    /// The logical ids referenced by any built-in schema `Ref` field, grouped by
    /// the kind they point at. Ids are shifted into the deck's global namespace
    /// (the same `apply` the dangling check uses) and stored by magnitude (the
    /// `|id|` convention), so a reference resolves regardless of sign or offset.
    ///
    /// The mesh-scale per-line blocks (`*ELEMENT_*`, `*NODE`) are skipped: their
    /// only references are node/part connectivity, which doesn't bear on whether
    /// a *library* entity (material, curve, set, …) is used — and walking them
    /// would be O(mesh). User-schema references are not consulted here.
    pub(crate) fn referenced_ids(&self) -> HashMap<EntityKind, HashSet<i64>> {
        let transforms = self.file_transforms();
        let mut refs: HashMap<EntityKind, HashSet<i64>> = HashMap::new();
        for (fi, file) in self.files.iter().enumerate() {
            let transform = transforms[fi].as_ref();
            for block in &file.blocks {
                let exact = file.keyword_name(block);
                let base = canonical_base(exact);
                if keywords::definition_of(&base).is_some_and(|d| d.per_line) {
                    continue;
                }
                let Some(kw) = keywords::find(&base) else {
                    continue;
                };
                let title = title_offset(exact);
                let lines = data_lines(file, block);
                for (ci, card) in kw.cards.iter().enumerate() {
                    let Some(line) = lines.get(title + ci) else {
                        break;
                    };
                    for (idx, f) in card.iter().enumerate() {
                        let kinds: &[EntityKind] = match &f.r {
                            Ref::None => continue,
                            Ref::To(k) => std::slice::from_ref(k),
                            Ref::AnyOf(ks) => ks,
                        };
                        let Some(v) = card_field_i64(line, card, idx, block.format) else {
                            continue;
                        };
                        if v == 0 {
                            continue;
                        }
                        for k in kinds {
                            let logical = transform.map_or(v, |t| t.apply(v, *k)).abs();
                            refs.entry(*k).or_default().insert(logical);
                        }
                    }
                }
            }
        }
        refs
    }
}

/// Read a field by name (case-insensitive) from a specific block, typed.
pub(crate) fn entity_field(deck: &Deck, file: usize, block: usize, name: &str) -> Option<Value> {
    let f = &deck.files[file];
    let b = &f.blocks[block];
    let base = canonical_base(f.keyword_name(b));
    let kw = keywords::find(&base)?;
    let title = title_offset(f.keyword_name(b));
    let lines = data_lines(f, b);
    for (ci, card) in kw.cards.iter().enumerate() {
        for (fi, fld) in card.iter().enumerate() {
            if fld.n.eq_ignore_ascii_case(name) {
                let line = lines.get(title + ci)?;
                let raw = card_field_slice(line, card, fi, b.format)?;
                return Some(match fld.t {
                    T::I => Value::Int(parse_i64(raw)?),
                    T::F => Value::Float(std::str::from_utf8(raw).ok()?.trim().parse().ok()?),
                    T::S => Value::Str(std::str::from_utf8(raw).ok()?.trim().to_string()),
                });
            }
        }
    }
    None
}

/// The `(Ref, id)` of a named reference field on a block. (Used by the Python
/// `Entity` binding; the Rust `Field::reference` path doesn't go through it.)
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn ref_field(deck: &Deck, file: usize, block: usize, name: &str) -> Option<(Ref, i64)> {
    let f = &deck.files[file];
    let b = &f.blocks[block];
    let base = canonical_base(f.keyword_name(b));
    let kw = keywords::find(&base)?;
    let fld = kw
        .cards
        .iter()
        .flat_map(|c| c.iter())
        .find(|x| x.n.eq_ignore_ascii_case(name))?;
    let id = entity_field(deck, file, block, name)?.as_i64()?;
    Some((fld.r, id))
}

/// The id referenced by this block's first field that targets `kind`.
pub(crate) fn first_ref_to(
    deck: &Deck,
    file: usize,
    block: usize,
    kind: EntityKind,
) -> Option<i64> {
    let f = &deck.files[file];
    let b = &f.blocks[block];
    let base = canonical_base(f.keyword_name(b));
    let kw = keywords::find(&base)?;
    for card in kw.cards {
        for fld in *card {
            let targets = matches!(fld.r, Ref::To(k) if k == kind)
                || matches!(fld.r, Ref::AnyOf(ks) if ks.contains(&kind));
            if targets
                && let Some(id) = entity_field(deck, file, block, fld.n).and_then(|v| v.as_i64())
            {
                return Some(id);
            }
        }
    }
    None
}

// ── The occurrence handle: Keyword → Card → Field ────────────────────────────

impl Deck {
    /// The navigable definition-entity index, built once and cached.
    pub(crate) fn site_index(&self) -> &Sites {
        self.sites.get_or_init(|| build_sites(self))
    }

    /// Iterate every occurrence of `keyword` across the deck — one [`Keyword`]
    /// per matching `*KEYWORD` block. Matches on the canonical base, so
    /// `SECTION_SHELL` also matches a `SECTION_SHELL_TITLE` block. Needs no
    /// schema: a keyword we ship no layout for still yields its occurrences.
    pub fn keywords<'d>(&'d self, keyword: &str) -> impl Iterator<Item = Keyword<'d>> + 'd {
        let base = canonical_base(keyword);
        self.files.iter().enumerate().flat_map(move |(fi, file)| {
            let base = base.clone();
            (0..file.blocks.len())
                .filter(move |&bi| canonical_base(file.keyword_name(&file.blocks[bi])) == base)
                .map(move |bi| Keyword {
                    deck: self,
                    file: fi,
                    block: bi,
                    identity: None,
                })
        })
    }

    /// Look up a definition entity by kind and id (honours the `|id|` convention).
    pub fn get(&self, kind: EntityKind, id: i64) -> Option<Keyword<'_>> {
        site_of(self.site_index(), kind, id).map(|(file, block)| Keyword {
            deck: self,
            file,
            block,
            identity: Some((kind, id)),
        })
    }
    pub fn part(&self, id: i64) -> Option<Keyword<'_>> {
        self.get(EntityKind::Part, id)
    }
    pub fn material(&self, id: i64) -> Option<Keyword<'_>> {
        self.get(EntityKind::Material, id)
    }
    pub fn section(&self, id: i64) -> Option<Keyword<'_>> {
        self.get(EntityKind::Section, id)
    }
    pub fn curve(&self, id: i64) -> Option<Keyword<'_>> {
        self.get(EntityKind::Curve, id)
    }

    /// Every definition entity of a kind (unordered) — e.g. iterate all parts.
    pub fn entities(&self, kind: EntityKind) -> impl Iterator<Item = Keyword<'_>> {
        self.site_index()
            .iter()
            .filter(move |((k, _), _)| *k == kind)
            .map(move |(&(kind, id), &(file, block))| Keyword {
                deck: self,
                file,
                block,
                identity: Some((kind, id)),
            })
    }
    pub fn parts(&self) -> impl Iterator<Item = Keyword<'_>> {
        self.entities(EntityKind::Part)
    }
    pub fn materials(&self) -> impl Iterator<Item = Keyword<'_>> {
        self.entities(EntityKind::Material)
    }
    pub fn sections(&self) -> impl Iterator<Item = Keyword<'_>> {
        self.entities(EntityKind::Section)
    }
    pub fn curves(&self) -> impl Iterator<Item = Keyword<'_>> {
        self.entities(EntityKind::Curve)
    }

    /// Bulk **columnar** read of every occurrence of `keyword` across the whole
    /// deck (root + includes), using dynars' built-in schema. This is the fast
    /// path underneath [`keywords`](Deck::keywords) navigation — same keyword
    /// names, same field names — for when you want whole columns (ids,
    /// coordinates, connectivity) instead of walking occurrences one at a time.
    ///
    /// `None` if `keyword` isn't in the built-in library; describe it yourself
    /// and use [`table_with`](Deck::table_with) instead.
    ///
    /// ```no_run
    /// # fn demo(deck: &dynars::deck::Deck) {
    /// let nodes = deck.table("NODE").unwrap();
    /// let ids = nodes.column("nid").unwrap().as_int().unwrap();
    /// # let _ = ids;
    /// # }
    /// ```
    pub fn table(&self, keyword: &str) -> Option<Table> {
        let schema = keywords::schema(keyword)?;
        Some(parse_schema_files(&self.files, &schema))
    }

    /// Bulk columnar read using a caller-supplied [`Schema`] — the escape hatch
    /// for a keyword not in the built-in library (rare, vendor-specific, or
    /// newer than our snapshot). Spans the whole deck, like [`table`](Deck::table).
    pub fn table_with(&self, schema: &Schema) -> Table {
        parse_schema_files(&self.files, schema)
    }

    /// The user schema registered for a canonical `base`, if any. Consulted
    /// ahead of the built-in table when resolving field layout.
    fn user_schema(&self, base: &str) -> Option<&Schema> {
        self.user_schemas.get(base)
    }

    /// Every card of `base`'s field layout as [`CardRef`]s — the user overlay if
    /// one is registered, else the built-in table, else `None`. Backs the
    /// across-cards name lookup in [`Keyword::field`].
    fn layout_cards(&self, base: &str) -> Option<Vec<CardRef<'_>>> {
        if let Some(s) = self.user_schema(base) {
            return Some(
                s.cards
                    .iter()
                    .map(|c| CardRef::User(c.fields.as_slice()))
                    .collect(),
            );
        }
        let kw = keywords::find(base)?;
        Some(kw.cards.iter().map(|&c| CardRef::Static(c)).collect())
    }
}

/// A card's field layout, from either the built-in static table or a user
/// [`Schema`] registered on the [`Deck`]. Borrowed for the deck's lifetime so
/// one `Field`/`Card` implementation reads both — the built-in `&'static [Fld]`
/// coerces into the same `'d`. Both carry names, types, widths, and (when
/// declared via [`Card::ref_to`](crate::schema::Card::ref_to)) references.
#[derive(Clone, Copy)]
enum CardRef<'d> {
    Static(&'static [keywords::Fld]),
    User(&'d [FieldSpec]),
}

impl<'d> CardRef<'d> {
    fn len(&self) -> usize {
        match self {
            CardRef::Static(c) => c.len(),
            CardRef::User(c) => c.len(),
        }
    }
    fn position_by_name(&self, name: &str) -> Option<usize> {
        match self {
            CardRef::Static(c) => c.iter().position(|f| f.n.eq_ignore_ascii_case(name)),
            CardRef::User(c) => c.iter().position(|f| f.name.eq_ignore_ascii_case(name)),
        }
    }
    fn name(&self, col: usize) -> Option<&'d str> {
        match self {
            CardRef::Static(c) => c.get(col).map(|f| f.n),
            CardRef::User(c) => c.get(col).map(|f| f.name.as_str()),
        }
    }
    fn ty(&self, col: usize) -> Option<T> {
        match self {
            CardRef::Static(c) => c.get(col).map(|f| f.t),
            CardRef::User(c) => c.get(col).map(|f| match f.ty {
                FieldType::Int => T::I,
                FieldType::Float => T::F,
                FieldType::Str => T::S,
            }),
        }
    }
    fn ref_of(&self, col: usize) -> Ref {
        match self {
            CardRef::Static(c) => c.get(col).map_or(Ref::None, |f| f.r),
            CardRef::User(c) => c.get(col).map_or(Ref::None, |f| f.reference),
        }
    }
    /// The raw (untrimmed) byte slice for field `col`, width-aware. Fixed format
    /// sums per-field widths; a user array field (`count > 1`) advances the
    /// offset by `count * width` and exposes its first element.
    fn field_slice(&self, line: &'d [u8], col: usize, fmt: CardFormat) -> Option<&'d [u8]> {
        if col >= self.len() {
            return None;
        }
        if crate::schema::__is_free(line, fmt) {
            return line.split(|&c| c == b',').nth(self.free_index(col));
        }
        let scaled = |w: usize| if fmt == CardFormat::Long { w * 2 } else { w };
        let (off, width) = self.fixed_offset(col, scaled);
        if off >= line.len() {
            return None;
        }
        Some(crate::schema::__slice(line, off, width))
    }
    /// Comma-token index of field `col` in free format (arrays consume `count`).
    fn free_index(&self, col: usize) -> usize {
        match self {
            CardRef::Static(_) => col,
            CardRef::User(c) => c[..col].iter().map(|f| f.count.max(1)).sum(),
        }
    }
    /// `(byte offset, slot width)` of field `col` in fixed format.
    fn fixed_offset(&self, col: usize, scaled: impl Fn(usize) -> usize) -> (usize, usize) {
        match self {
            CardRef::Static(c) => (c[..col].iter().map(|f| scaled(f.w)).sum(), scaled(c[col].w)),
            CardRef::User(c) => (
                c[..col]
                    .iter()
                    .map(|f| scaled(f.width) * f.count.max(1))
                    .sum(),
                scaled(c[col].width),
            ),
        }
    }
}

/// One keyword occurrence in a deck — a single `*KEYWORD` block. The one handle
/// for reading a deck: reached by name ([`Deck::keywords`]), by identity
/// ([`Deck::part`] / [`Deck::get`]), by kind ([`Deck::parts`] / [`Deck::entities`]),
/// or by following a reference ([`reference`](Keyword::reference)).
///
/// Two layers of access:
/// - **document** — [`name`](Keyword::name), [`file`](Keyword::file),
///   [`line`](Keyword::line), [`cards`](Keyword::cards): always available, even
///   for a keyword we ship no schema for.
/// - **schema** — [`field`](Keyword::field) by name, typed values, references,
///   and identity ([`id`](Keyword::id) / [`kind`](Keyword::kind)): present when a
///   schema resolves, `None`/raw when it doesn't.
pub struct Keyword<'d> {
    deck: &'d Deck,
    file: usize,
    block: usize,
    /// Known when reached by id-lookup; otherwise derived on demand.
    identity: Option<(EntityKind, i64)>,
}

impl<'d> Keyword<'d> {
    // ── document layer (no schema needed) ──
    /// The exact keyword name of this occurrence (e.g. `SECTION_SHELL_TITLE`).
    pub fn name(&self) -> &'d str {
        let f = &self.deck.files[self.file];
        f.keyword_name(&f.blocks[self.block])
    }
    /// The canonical base name (`SECTION_SHELL` for a `SECTION_SHELL_TITLE`).
    pub fn base(&self) -> String {
        canonical_base(self.name())
    }
    /// The include file this occurrence is defined in.
    pub fn file(&self) -> &'d std::path::Path {
        &self.deck.files[self.file].path
    }
    /// 1-based line of the occurrence's `*KEYWORD` line, for clickable locations.
    pub fn line(&self) -> usize {
        let f = &self.deck.files[self.file];
        crate::schema::block_line(f, &f.blocks[self.block])
    }
    /// Whether a schema is known for this keyword — a user schema registered on
    /// the deck ([`Deck::register_schema`]), or a built-in/supplement layout.
    /// When `false`, named/typed/reference access degrades to raw positional reads.
    pub fn has_schema(&self) -> bool {
        let base = self.base();
        self.deck.user_schema(&base).is_some() || keywords::find(&base).is_some()
    }

    fn block_format(&self) -> CardFormat {
        self.deck.files[self.file].blocks[self.block].format
    }
    fn rows(&self) -> Vec<&'d [u8]> {
        let f = &self.deck.files[self.file];
        data_lines(f, &f.blocks[self.block])
    }

    /// Iterate this occurrence's data rows as [`Card`]s (one per data line).
    pub fn cards(&self) -> impl Iterator<Item = Card<'d>> + 'd {
        let n = self.rows().len();
        let (deck, file, block) = (self.deck, self.file, self.block);
        (0..n).map(move |row| Card {
            deck,
            file,
            block,
            row,
        })
    }
    /// The `i`-th data row as a [`Card`], if present.
    pub fn card(&self, i: usize) -> Option<Card<'d>> {
        (i < self.rows().len()).then_some(Card {
            deck: self.deck,
            file: self.file,
            block: self.block,
            row: i,
        })
    }

    // ── schema layer ──
    /// Read a field by name (case-insensitive) from **any** of this keyword's
    /// cards — control card, thickness card, … `None` without a schema, or if
    /// the field (or its card) is absent.
    pub fn field(&self, name: &str) -> Option<Field<'d>> {
        let cards = self.deck.layout_cards(&self.base())?;
        let title = title_offset(self.name());
        let rows = self.rows();
        for (ci, card) in cards.iter().enumerate() {
            if let Some(col) = card.position_by_name(name) {
                let row = title + ci;
                return (row < rows.len()).then_some(Field {
                    deck: self.deck,
                    file: self.file,
                    block: self.block,
                    row,
                    col,
                    card: Some(*card),
                });
            }
        }
        None
    }

    /// This occurrence's own id, when it defines an entity (`None` for a
    /// non-definition keyword like `*CONTROL_TERMINATION`, or with no schema).
    pub fn id(&self) -> Option<i64> {
        // Reached by lookup/kind: identity already carries the logical id.
        if let Some((_, id)) = self.identity {
            return Some(id);
        }
        let base = self.base();
        let def = keywords::definition_of(&base)?;
        if def.per_line {
            return None;
        }
        let kw = keywords::find(&base)?;
        let id_card = kw.cards.get(def.id_card).copied().unwrap_or(&[]);
        let rows = self.rows();
        let line = rows.get(title_offset(self.name()) + def.id_card)?;
        let id = card_field_i64(line, id_card, 0, self.block_format())?;
        if id == 0 {
            return None;
        }
        // Reached by name: the card holds the file-local id — report the global
        // one, so it agrees with `Deck::get`/`entities` on a transformed include.
        Some(
            self.deck
                .transform_of(self.file)
                .map_or(id, |t| t.apply(id, def.kind)),
        )
    }
    /// The entity kind this occurrence defines, if any.
    pub fn kind(&self) -> Option<EntityKind> {
        if let Some((k, _)) = self.identity {
            return Some(k);
        }
        keywords::definition_of(&self.base()).map(|d| d.kind)
    }

    /// The effective `*INCLUDE_TRANSFORM` offsets applied to this occurrence's
    /// file — composed down the include chain — or `None` if it sits in the root
    /// or a plain `*INCLUDE`. This is why [`id`](Keyword::id) and the reference
    /// followers report global ids: `raw_id.apply(offsets)` is the id you see.
    pub fn transform(&self) -> Option<TransformOffsets> {
        self.deck.transform_of(self.file).copied()
    }

    // ── reference following ──
    /// Follow the reference in field `name` to the entity it points at.
    pub fn reference(&self, name: &str) -> Option<Keyword<'d>> {
        self.field(name)?.reference()
    }
    /// Follow this occurrence's (first) reference to an entity of `kind`.
    pub fn reference_to(&self, kind: EntityKind) -> Option<Keyword<'d>> {
        let id = first_ref_to(self.deck, self.file, self.block, kind)?;
        // The ref is written in this file's local ids; resolve it in the deck's
        // global namespace before the lookup.
        let id = self
            .deck
            .transform_of(self.file)
            .map_or(id, |t| t.apply(id, kind));
        self.deck.get(kind, id)
    }
    pub fn material(&self) -> Option<Keyword<'d>> {
        self.reference_to(EntityKind::Material)
    }
    pub fn section(&self) -> Option<Keyword<'d>> {
        self.reference_to(EntityKind::Section)
    }
    pub fn eos(&self) -> Option<Keyword<'d>> {
        self.reference_to(EntityKind::Eos)
    }
    pub fn hourglass(&self) -> Option<Keyword<'d>> {
        self.reference_to(EntityKind::Hourglass)
    }
}

/// One data row of a [`Keyword`] occurrence, plus the schema for that row when
/// available. Fields are addressed by name ([`field`](Card::field)) or position
/// ([`at`](Card::at) / [`raw`](Card::raw), which work with or without a schema).
pub struct Card<'d> {
    deck: &'d Deck,
    file: usize,
    block: usize,
    row: usize,
}

impl<'d> Card<'d> {
    /// The schema card governing this row: the deck's user overlay for this
    /// keyword if one is registered ([`Deck::register_schema`]), else the
    /// built-in table. Row→card tiling mirrors on both sides — a single
    /// repeating card governs every row (so `*NODE`/`*ELEMENT_*` and repeating
    /// user schemas all type), other keywords map 1:1. Title rows and unmapped
    /// tails return `None` → raw-only positional access still works.
    fn schema_card(&self) -> Option<CardRef<'d>> {
        let f = &self.deck.files[self.file];
        let exact = f.keyword_name(&f.blocks[self.block]);
        let base = canonical_base(exact);
        let i = self.row.checked_sub(title_offset(exact))?;
        if let Some(s) = self.deck.user_schema(&base) {
            return s.card_for_row(i).map(CardRef::User);
        }
        keywords::find(&base)?.card_for_row(i).map(CardRef::Static)
    }
    /// Read a field by name (case-insensitive) within this row. Needs a schema.
    pub fn field(&self, name: &str) -> Option<Field<'d>> {
        let card = self.schema_card()?;
        let col = card.position_by_name(name)?;
        Some(Field {
            deck: self.deck,
            file: self.file,
            block: self.block,
            row: self.row,
            col,
            card: Some(card),
        })
    }
    /// The `col`-th field of this row (positional): typed via the schema when
    /// present, raw otherwise. `None` if the row has no such column.
    pub fn at(&self, col: usize) -> Option<Field<'d>> {
        let card = self.schema_card();
        let f = Field {
            deck: self.deck,
            file: self.file,
            block: self.block,
            row: self.row,
            col,
            card,
        };
        f.raw_bytes().is_some().then_some(f)
    }
    /// The trimmed raw token at column `col` — never needs a schema.
    pub fn raw(&self, col: usize) -> Option<&'d str> {
        self.at(col)?.as_str()
    }
    /// Iterate this row's fields (schema-driven; empty without a schema).
    pub fn fields(&self) -> impl Iterator<Item = Field<'d>> + 'd {
        let card = self.schema_card();
        let (deck, file, block, row) = (self.deck, self.file, self.block, self.row);
        let n = card.map_or(0, |c| c.len());
        (0..n).map(move |col| Field {
            deck,
            file,
            block,
            row,
            col,
            card,
        })
    }
}

/// One field slot in a [`Card`]: its raw bytes plus, when a schema is present,
/// its name, type, and reference target. Read the datum with
/// [`value`](Field::value) / [`as_i64`](Field::as_i64) / …, or follow it with
/// [`reference`](Field::reference).
pub struct Field<'d> {
    deck: &'d Deck,
    file: usize,
    block: usize,
    row: usize,
    col: usize,
    card: Option<CardRef<'d>>,
}

impl<'d> Field<'d> {
    fn block_format(&self) -> CardFormat {
        self.deck.files[self.file].blocks[self.block].format
    }
    fn line(&self) -> &'d [u8] {
        let f = &self.deck.files[self.file];
        data_lines(f, &f.blocks[self.block])
            .get(self.row)
            .copied()
            .unwrap_or(&[])
    }
    /// The raw (untrimmed) bytes of this slot.
    fn raw_bytes(&self) -> Option<&'d [u8]> {
        let line = self.line();
        match self.card {
            Some(card) => card.field_slice(line, self.col, self.block_format()),
            None => {
                // No schema: comma-split in free format, else whitespace tokens.
                if crate::schema::__is_free(line, self.block_format()) {
                    line.split(|&c| c == b',').nth(self.col)
                } else {
                    line.split(|&c| c == b' ' || c == b'\t')
                        .filter(|t| !t.is_empty())
                        .nth(self.col)
                }
            }
        }
    }

    /// The field's name, when a schema names this position.
    pub fn name(&self) -> Option<&'d str> {
        self.card.and_then(|c| c.name(self.col))
    }
    /// The untrimmed source text of this slot.
    pub fn raw(&self) -> &'d str {
        self.raw_bytes()
            .and_then(|b| std::str::from_utf8(b).ok())
            .unwrap_or("")
    }
    /// The typed value: `Int`/`Float`/`Str` per the schema, or `Str(raw)`
    /// without one (or if a numeric parse fails).
    pub fn value(&self) -> Value {
        let raw = match self.raw_bytes() {
            Some(r) => r,
            None => return Value::Str(String::new()),
        };
        let trimmed = || std::str::from_utf8(raw).unwrap_or("").trim().to_string();
        match self.card.and_then(|c| c.ty(self.col)) {
            Some(T::I) => parse_i64(raw)
                .map(Value::Int)
                .unwrap_or_else(|| Value::Str(trimmed())),
            Some(T::F) => std::str::from_utf8(raw)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .map(Value::Float)
                .unwrap_or_else(|| Value::Str(trimmed())),
            Some(T::S) | None => Value::Str(trimmed()),
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        parse_i64(self.raw_bytes()?)
    }
    pub fn as_f64(&self) -> Option<f64> {
        std::str::from_utf8(self.raw_bytes()?)
            .ok()?
            .trim()
            .parse()
            .ok()
    }
    /// The trimmed text of this slot (any type).
    pub fn as_str(&self) -> Option<&'d str> {
        Some(std::str::from_utf8(self.raw_bytes()?).ok()?.trim())
    }
    /// Follow this field's reference to the entity it points at, if it is one —
    /// for a built-in field, or a user-schema field declared with
    /// [`Card::ref_to`](crate::schema::Card::ref_to).
    pub fn reference(&self) -> Option<Keyword<'d>> {
        let id = self.as_i64()?;
        // Shift the local ref id into the deck's global namespace per candidate
        // kind (a no-op for a file with no `*INCLUDE_TRANSFORM`).
        let transform = self.deck.transform_of(self.file);
        let logical = |k: EntityKind| transform.map_or(id, |t| t.apply(id, k));
        match self.card?.ref_of(self.col) {
            Ref::None => None,
            Ref::To(k) => self.deck.get(k, logical(k)),
            Ref::AnyOf(ks) => ks.iter().find_map(|k| self.deck.get(*k, logical(*k))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::ParsedFile;
    use crate::parser::split_blocks;
    use std::collections::HashMap;
    use std::sync::OnceLock;

    fn deck(src: &[u8]) -> Deck {
        deck_multi(&[src])
    }

    fn deck_multi(srcs: &[&[u8]]) -> Deck {
        let files: Vec<ParsedFile> = srcs
            .iter()
            .enumerate()
            .map(|(i, s)| ParsedFile::new(format!("f{i}.k").into(), s.to_vec(), split_blocks(s)))
            .collect();
        let transforms = vec![crate::keywords::TransformOffsets::IDENTITY; files.len()];
        Deck {
            files,
            includes: vec![],
            transforms,
            defs: OnceLock::new(),
            file_transforms: OnceLock::new(),
            sites: OnceLock::new(),
            user_schemas: HashMap::new(),
        }
    }

    #[test]
    fn tabular_keyword_rows_all_type_through_the_schema() {
        // *NODE is a per-line keyword: one repeating card. Every row — not just
        // the first — must resolve its named, typed fields (the Phase 2 fix).
        let d = deck(b"*NODE\n1,0.0,0.0,0.0\n2,1.0,2.0,3.0\n3,4.0,5.0,6.0\n");
        let node = d.keywords("NODE").next().expect("one *NODE block");

        assert_eq!(node.cards().count(), 3);
        // third row (previously unreachable: card index 2 had no schema card)
        let c2 = node.card(2).unwrap();
        assert_eq!(c2.field("nid").unwrap().as_i64(), Some(3));
        assert_eq!(c2.field("x").unwrap().as_f64(), Some(4.0));
        // and the value is typed, not a raw string fallback
        assert_eq!(
            node.card(1).unwrap().field("nid").unwrap().value(),
            Value::Int(2)
        );
    }

    #[test]
    fn fixed_multi_card_keyword_maps_cards_one_to_one() {
        // *PART: heading card then data card, mapped 1:1 (no regression).
        let d = deck(b"*PART\nsteel bracket\n7,2,3\n");
        let part = d.keywords("PART").next().expect("one *PART block");

        assert_eq!(
            part.card(0).unwrap().field("heading").unwrap().as_str(),
            Some("steel bracket")
        );
        assert_eq!(
            part.card(1).unwrap().field("pid").unwrap().as_i64(),
            Some(7)
        );
        assert_eq!(
            part.card(1).unwrap().field("secid").unwrap().as_i64(),
            Some(2)
        );
        // identity still resolves off the consolidated def metadata
        assert_eq!(part.id(), Some(7));
        assert_eq!(part.kind(), Some(EntityKind::Part));
    }

    #[test]
    fn table_reads_columns_across_the_whole_deck() {
        // Root + an include, each with *NODE — the bulk table must merge them
        // (the unified columnar path is deck-wide, not per-file).
        let d = deck_multi(&[
            b"*NODE\n1,0.0,0.0,0.0\n2,1.0,1.0,1.0\n",
            b"*NODE\n3,2.0,2.0,2.0\n",
        ]);
        let nodes = d.table("NODE").expect("NODE is built in");
        assert_eq!(nodes.rows(), 3);
        assert_eq!(nodes.column("nid").unwrap().as_int().unwrap(), &[1, 2, 3]);
        assert_eq!(
            nodes.column("x").unwrap().as_float().unwrap(),
            &[0.0, 1.0, 2.0]
        );
        // a keyword we ship no schema for → None (use table_with)
        assert!(d.table("NOT_A_REAL_KEYWORD_XYZ").is_none());
    }

    #[test]
    fn table_with_uses_a_caller_supplied_schema() {
        let d = deck_multi(&[b"*FOO\n1,2\n3,4\n"]);
        let schema = Schema::new("FOO").card(crate::schema::Card::new().int("a", 8).int("b", 8));
        let t = d.table_with(&schema);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.column("a").unwrap().as_int().unwrap(), &[1, 3]);
        assert_eq!(t.column("b").unwrap().as_int().unwrap(), &[2, 4]);
    }

    #[test]
    fn register_schema_gives_named_typed_access_to_unknown_keyword() {
        let mut d = deck_multi(&[b"*VENDOR_WIDGET\n42,3.5,hello\n7,1.0,world\n"]);

        // Before registering: no schema → named access degrades to None.
        let kw = d
            .keywords("VENDOR_WIDGET")
            .next()
            .expect("occurrence exists schema-or-not");
        assert!(!kw.has_schema());
        assert!(kw.field("wid").is_none());
        // ...but the document layer already works: positional, typed-on-parse.
        assert_eq!(kw.card(1).unwrap().raw(0), Some("7"));

        // Describe it once (single repeating card).
        d.register_schema(
            Schema::new("VENDOR_WIDGET").card(
                crate::schema::Card::new()
                    .int("wid", 8)
                    .float("mass", 8)
                    .str("tag", 8),
            ),
        );

        let kw = d.keywords("VENDOR_WIDGET").next().unwrap();
        assert!(kw.has_schema());
        // flatten shortcut → first row
        assert_eq!(kw.field("wid").unwrap().as_i64(), Some(42));
        // per-row named + typed access, both rows (repeating card)
        assert_eq!(
            kw.card(0).unwrap().field("mass").unwrap().as_f64(),
            Some(3.5)
        );
        assert_eq!(kw.card(1).unwrap().field("wid").unwrap().as_i64(), Some(7));
        assert_eq!(
            kw.card(1).unwrap().field("tag").unwrap().as_str(),
            Some("world")
        );
        assert_eq!(
            kw.card(0).unwrap().field("tag").unwrap().value(),
            Value::Str("hello".into())
        );
        // the field carries its schema name
        assert_eq!(kw.card(0).unwrap().at(1).unwrap().name(), Some("mass"));
    }

    #[test]
    fn registered_schema_references_are_validated_and_followed() {
        use crate::validate::Rule;
        // *MAT_ELASTIC defines Material 5; the custom keyword references a
        // material on each row — one valid (5), one dangling (99).
        let mut d =
            deck_multi(&[b"*MAT_ELASTIC\n5,7.85e-9,210000.0,0.3\n*VENDOR_WIDGET\n1,5\n2,99\n"]);
        d.register_schema(
            Schema::new("VENDOR_WIDGET").card(crate::schema::Card::new().int("wid", 8).ref_to(
                "mat",
                8,
                EntityKind::Material,
            )),
        );

        // references_resolve now covers the user schema's declared reference.
        let report = d.validate([Rule::references_resolve()]);
        let widget: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.keyword == "VENDOR_WIDGET")
            .collect();
        assert_eq!(widget.len(), 1, "only mat=99 dangles");
        assert!(widget[0].message.contains("mat") && widget[0].message.contains("99"));

        // and Field::reference() follows a valid one to the defining entity.
        let w = d.keywords("VENDOR_WIDGET").next().unwrap();
        let mat = w.card(0).unwrap().field("mat").unwrap().reference();
        assert_eq!(mat.and_then(|m| m.id()), Some(5));
        // the dangling row resolves to nothing.
        assert!(
            w.card(1)
                .unwrap()
                .field("mat")
                .unwrap()
                .reference()
                .is_none()
        );
    }

    #[test]
    fn dangling_check_handles_high_offset_ids() {
        use crate::validate::Rule;
        // Node ids around 3e9 — far above any absolute cap, but a compact range,
        // so the offset bitset still applies. Free format so the wide ids fit.
        let b: i64 = 3_000_000_000;
        let src = format!(
            "*PART\npart\n1,1,1\n\
             *NODE\n{n1},0,0,0\n{n2},0,0,0\n\
             *ELEMENT_SHELL\n1,1,{n1},{n2},{n1},{bad}\n",
            n1 = b + 1,
            n2 = b + 2,
            bad = b + 999,
        );
        let d = deck_multi(&[src.as_bytes()]);
        let report = d.validate([Rule::references_resolve_with_connectivity()]);

        // The two defined high ids resolve; only the undefined one dangles.
        let node_findings: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.keyword == "ELEMENT_SHELL")
            .collect();
        assert_eq!(node_findings.len(), 1, "only {bad} dangles", bad = b + 999);
        assert!(node_findings[0].message.contains(&(b + 999).to_string()));
        assert!(
            !report
                .findings
                .iter()
                .any(|f| f.message.contains(&(b + 1).to_string()))
        );
    }
}
