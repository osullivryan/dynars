//! Phase 5: user-defined keyword schemas.
//!
//! A [`Schema`] declares how to marshal a keyword: an ordered list of cards
//! (lines), each an ordered list of typed fields, plus whether the card group
//! repeats over the block body. The same schema drives both the Rust builder
//! API here and the Python class API — Python lowers its `@keyword` classes to
//! exactly this structure, so there is one parser and one source of truth.
//!
//! ```
//! use dynars::schema::{Schema, Card};
//! // *NODE: one card, repeating over the block (the default).
//! let node = Schema::new("NODE").card(
//!     Card::new().int("nid", 8).float("x", 16).float("y", 16).float("z", 16),
//! );
//! ```

use rayon::prelude::*;

use crate::file::{Block, CardFormat, ParsedFile};
use crate::keywords::{EntityKind, Ref};
use crate::parser::Field;

// --- shared chunking infrastructure (parallel splitting of block bodies) ---

/// Minimum bytes per parallel chunk; below this a block stays one chunk.
const MIN_CHUNK: usize = 256 * 1024;

/// Collect line-aligned chunks (with their format) across every block whose
/// name matches `keyword`, over every file, for parallel parsing.
///
/// The chunk budget is spread over the *whole* matching dataset, not multiplied
/// per block: one huge block fans out into ~`cores·2` chunks; many small blocks
/// (root + includes) each contribute one. (The old per-block split turned a
/// 256-file deck into thousands of chunks.) Each chunk holds only whole lines,
/// so per-chunk row counts sum to the total and outputs concatenate in order.
fn collect_chunks<'a>(files: &'a [ParsedFile], keyword: &str) -> Vec<(&'a [u8], CardFormat)> {
    let mut bodies: Vec<(&[u8], CardFormat)> = Vec::new();
    let mut total = 0usize;
    for parsed in files {
        for block in &parsed.blocks {
            if parsed.keyword_name(block).eq_ignore_ascii_case(keyword) {
                let body = parsed.body(block);
                total += body.len();
                bodies.push((body, block.format));
            }
        }
    }
    let target = (rayon::current_num_threads() * 2).max(1);
    let chunk_bytes = total.div_ceil(target).max(MIN_CHUNK);

    let mut chunks = Vec::new();
    for (body, fmt) in bodies {
        let mut start = 0;
        while start < body.len() {
            let mut end = (start + chunk_bytes).min(body.len());
            if end < body.len() {
                // extend to the next newline so lines never split across chunks.
                match memchr::memchr(b'\n', &body[end..]) {
                    Some(off) => end += off + 1,
                    None => end = body.len(),
                }
            }
            chunks.push((&body[start..end], fmt));
            start = end;
        }
    }
    chunks
}

/// True for comment (`$`) and blank lines, which carry no card data.
#[inline]
fn is_skippable(line: &[u8]) -> bool {
    let indent = line
        .iter()
        .take_while(|&&c| c == b' ' || c == b'\t')
        .count();
    line.is_empty() || line.get(indent) == Some(&b'$') || strip_eol(&line[indent..]).is_empty()
}

/// Strip a trailing `\r`/`\n` without touching interior or leading bytes.
#[inline]
fn strip_eol(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && matches!(s[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Int,
    Float,
    Str,
}

/// One field in a card: a name, a type, a fixed-format column width, a count
/// (`> 1` makes it an array, producing an `N`-wide column), and — for a user
/// schema — what entity its id references, if any (so a registered keyword's
/// references participate in [`Rule::references_resolve`](crate::validate::Rule::references_resolve)).
#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub ty: FieldType,
    pub width: usize,
    pub count: usize,
    /// The entity this field's id points at, if any (default [`Ref::None`]).
    pub reference: Ref,
}

/// One card (line) of a keyword: an ordered list of fields.
#[derive(Debug, Clone, Default)]
pub struct Card {
    pub fields: Vec<FieldSpec>,
}

impl Card {
    pub fn new() -> Self {
        Card::default()
    }
    pub fn int(self, name: &str, width: usize) -> Self {
        self.push(name, FieldType::Int, width, 1)
    }
    pub fn float(self, name: &str, width: usize) -> Self {
        self.push(name, FieldType::Float, width, 1)
    }
    pub fn str(self, name: &str, width: usize) -> Self {
        self.push(name, FieldType::Str, width, 1)
    }
    /// `count` consecutive integer fields as one `count`-wide column.
    pub fn int_array(self, name: &str, count: usize, width: usize) -> Self {
        self.push(name, FieldType::Int, width, count)
    }
    /// `count` consecutive float fields as one `count`-wide column.
    pub fn float_array(self, name: &str, count: usize, width: usize) -> Self {
        self.push(name, FieldType::Float, width, count)
    }
    /// An integer field whose id **references** an entity of `kind`. On a schema
    /// registered with [`Deck::register_schema`](crate::deck::Deck::register_schema),
    /// [`Rule::references_resolve`](crate::validate::Rule::references_resolve)
    /// will check the id resolves, and `Field::reference()` will follow it.
    pub fn ref_to(mut self, name: &str, width: usize, kind: EntityKind) -> Self {
        self.fields.push(FieldSpec {
            name: name.to_string(),
            ty: FieldType::Int,
            width,
            count: 1,
            reference: Ref::To(kind),
        });
        self
    }
    fn push(mut self, name: &str, ty: FieldType, width: usize, count: usize) -> Self {
        self.fields.push(FieldSpec {
            name: name.to_string(),
            ty,
            width,
            count: count.max(1),
            reference: Ref::None,
        });
        self
    }
}

/// A keyword's full layout.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Keyword name without `*`, matched case-insensitively (e.g. `NODE`).
    pub keyword: String,
    pub cards: Vec<Card>,
    /// Whether the card group repeats over the whole block body (`*NODE`,
    /// `*ELEMENT_*`, multiple `*PART`s). Defaults to `true` — the common case.
    /// Use [`Schema::once`] for a keyword that defines a single entity per
    /// block from which you want only the first (rare).
    pub repeat: bool,
}

impl Schema {
    pub fn new(keyword: &str) -> Self {
        Schema {
            keyword: keyword.to_string(),
            cards: Vec::new(),
            repeat: true,
        }
    }
    pub fn card(mut self, card: Card) -> Self {
        self.cards.push(card);
        self
    }
    /// Parse only the first entity in each matching block, not the whole body.
    pub fn once(mut self) -> Self {
        self.repeat = false;
        self
    }

    /// The card governing data row `i` (0-based, past any title). Mirrors
    /// [`Kw::card_for_row`](crate::keywords::Kw::card_for_row) so a user schema
    /// registered on a [`Deck`](crate::deck::Deck) tiles its rows the same way
    /// the built-in table does: a single repeating card governs every row;
    /// otherwise cards map 1:1. Used by the navigation spine, not the columnar
    /// marshaller (which groups rows itself).
    pub fn card_for_row(&self, i: usize) -> Option<&[FieldSpec]> {
        if self.repeat && self.cards.len() == 1 {
            self.cards.first().map(|c| c.fields.as_slice())
        } else {
            self.cards.get(i).map(|c| c.fields.as_slice())
        }
    }
}

/// A parsed column. Numeric columns are contiguous (`ncols == 1` scalar, or
/// `ncols > 1` row-major for array fields); string columns are boxed values.
#[derive(Debug, Clone)]
pub enum Column {
    Int { data: Vec<i64>, ncols: usize },
    Float { data: Vec<f64>, ncols: usize },
    Str { data: Vec<String>, ncols: usize },
}

impl Column {
    pub fn rows(&self) -> usize {
        match self {
            Column::Int { data, ncols } => data.len() / (*ncols).max(1),
            Column::Float { data, ncols } => data.len() / (*ncols).max(1),
            Column::Str { data, ncols } => data.len() / (*ncols).max(1),
        }
    }
    pub fn as_int(&self) -> Option<&[i64]> {
        if let Column::Int { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    pub fn as_float(&self) -> Option<&[f64]> {
        if let Column::Float { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    /// Move the integer data out (for building typed structs without a copy).
    pub fn into_int(self) -> Option<Vec<i64>> {
        if let Column::Int { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    pub fn into_float(self) -> Option<Vec<f64>> {
        if let Column::Float { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    pub fn into_str(self) -> Option<Vec<String>> {
        if let Column::Str { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
    #[inline]
    fn push(&mut self, raw: &[u8]) {
        match self {
            Column::Int { data, .. } => data.push(Field { raw }.as_i64().unwrap_or(0)),
            Column::Float { data, .. } => data.push(Field { raw }.as_f64().unwrap_or(0.0)),
            Column::Str { data, .. } => data.push(Field { raw }.as_str().to_string()),
        }
    }
    fn extend(&mut self, other: Column) {
        match (self, other) {
            (Column::Int { data, .. }, Column::Int { data: d2, .. }) => data.extend(d2),
            (Column::Float { data, .. }, Column::Float { data: d2, .. }) => data.extend(d2),
            (Column::Str { data, .. }, Column::Str { data: d2, .. }) => data.extend(d2),
            _ => unreachable!("column kinds always match — same schema"),
        }
    }
}

/// The columnar result of parsing a keyword against a schema. Columns are in
/// schema field order (all cards flattened).
#[derive(Debug, Clone, Default)]
pub struct Table {
    pub columns: Vec<(String, Column)>,
}

impl Table {
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |(_, c)| c.rows())
    }
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
    /// Remove and return a column by name (used by generated typed structs to
    /// move data out without copying).
    pub fn take(&mut self, name: &str) -> Option<Column> {
        self.columns
            .iter()
            .position(|(n, _)| n == name)
            .map(|i| self.columns.remove(i).1)
    }

    /// Iterate rows as lightweight views — convenient for low-volume keywords
    /// (materials, sections, ...). Costs nothing until used; the columns stay
    /// columnar, so bulk keywords should read them (or numpy) directly.
    pub fn iter(&self) -> impl Iterator<Item = Row<'_>> + '_ {
        (0..self.rows()).map(move |idx| Row { table: self, idx })
    }
}

/// A single-row view into a [`Table`], with scalar field access by name.
pub struct Row<'a> {
    table: &'a Table,
    idx: usize,
}

impl<'a> Row<'a> {
    pub fn int(&self, name: &str) -> Option<i64> {
        match self.table.column(name) {
            Some(Column::Int { data, ncols }) if *ncols == 1 => data.get(self.idx).copied(),
            _ => None,
        }
    }
    pub fn float(&self, name: &str) -> Option<f64> {
        match self.table.column(name) {
            Some(Column::Float { data, ncols }) if *ncols == 1 => data.get(self.idx).copied(),
            _ => None,
        }
    }
    pub fn str(&self, name: &str) -> Option<&'a str> {
        match self.table.column(name) {
            Some(Column::Str { data, ncols }) if *ncols == 1 => {
                data.get(self.idx).map(|s| s.as_str())
            }
            _ => None,
        }
    }
}

/// Implemented by `#[derive(Keyword)]`: provides a keyword's [`Schema`] (for
/// introspection and the runtime/dynamic path). The single parsing entry point
/// on a derived struct is its generated inherent `parse()`, which for the bulk
/// single-card case is specialized code — not this schema being interpreted.
pub trait KeywordSchema {
    fn schema() -> Schema;
}

// --- runtime hooks used by the code `#[derive(Keyword)]` generates ---
// These let the macro emit a specialized, monomorphized per-line parser (no
// Column enum dispatch, offsets known at expansion time) while reusing the
// shared chunking/parallel/merge driver. Not part of the stable surface.

#[inline]
#[doc(hidden)]
pub fn __is_skippable(line: &[u8]) -> bool {
    is_skippable(line)
}
#[inline]
#[doc(hidden)]
pub fn __strip_eol(line: &[u8]) -> &[u8] {
    strip_eol(line)
}
#[inline]
#[doc(hidden)]
pub fn __is_free(line: &[u8], fmt: CardFormat) -> bool {
    fmt == CardFormat::Free || memchr::memchr(b',', line).is_some()
}
#[inline]
#[doc(hidden)]
pub fn __slice(line: &[u8], off: usize, w: usize) -> &[u8] {
    if off >= line.len() {
        &[]
    } else {
        &line[off..(off + w).min(line.len())]
    }
}
#[inline]
#[doc(hidden)]
pub fn __to_int(raw: &[u8]) -> i64 {
    Field { raw }.as_i64().unwrap_or(0)
}
#[inline]
#[doc(hidden)]
pub fn __to_float(raw: &[u8]) -> f64 {
    Field { raw }.as_f64().unwrap_or(0.0)
}
#[inline]
#[doc(hidden)]
pub fn __to_str(raw: &[u8]) -> String {
    Field { raw }.as_str().to_string()
}

/// Drive a generated per-chunk parser across all matching blocks in parallel
/// and merge the columns. The `per_chunk` closure is specialized by the derive.
#[doc(hidden)]
pub fn __drive_single_card<F>(parsed: &ParsedFile, keyword: &str, per_chunk: F) -> Vec<Column>
where
    F: Fn(&[u8], CardFormat) -> Vec<Column> + Sync + Send,
{
    let chunks = collect_chunks(std::slice::from_ref(parsed), keyword);
    if chunks.is_empty() {
        // Run once on empty input to get the (empty) column template.
        return per_chunk(&[], CardFormat::Fixed);
    }
    let partials: Vec<Vec<Column>> = chunks.par_iter().map(|(c, f)| per_chunk(c, *f)).collect();
    let mut it = partials.into_iter();
    let mut base = it.next().unwrap();
    for part in it {
        for (a, b) in base.iter_mut().zip(part) {
            a.extend(b);
        }
    }
    base
}

/// Assemble a [`Table`] from parallel column names and data.
#[doc(hidden)]
pub fn __table(names: Vec<&'static str>, cols: Vec<Column>) -> Table {
    Table {
        columns: names.into_iter().map(|n| n.to_string()).zip(cols).collect(),
    }
}

/// Implemented by `#[derive(Card)]`: provides one card's field layout, so cards
/// can be composed into multi-card keywords.
pub trait CardLayout {
    fn card() -> Card;
}

/// Parse every block matching `schema.keyword` in one file into a columnar
/// [`Table`]. A convenience wrapper over [`parse_schema_files`] for the
/// single-file case (`#[derive(Keyword)]`, tests). Deck-wide reads should go
/// through [`crate::deck::Deck::table`], which spans the root and all includes.
pub fn parse_schema(parsed: &ParsedFile, schema: &Schema) -> Table {
    parse_schema_files(std::slice::from_ref(parsed), schema)
}

/// Parse every block matching `schema.keyword` across `files` (a whole deck:
/// root + includes) into one columnar [`Table`], columns merged in file order.
///
/// Single-card repeating keywords (the bulk ones, `*NODE` / `*ELEMENT_*`) are
/// parsed in parallel across cores; multi-card or single-entity keywords are
/// parsed sequentially (they are almost always low volume).
pub fn parse_schema_files(files: &[ParsedFile], schema: &Schema) -> Table {
    if schema.cards.is_empty() {
        return Table::default();
    }
    if schema.cards.len() == 1 && schema.repeat {
        parse_parallel(files, schema)
    } else {
        parse_sequential(files, schema)
    }
}

/// Fresh, empty columns in schema field order.
fn empty_columns(schema: &Schema) -> Vec<Column> {
    let mut cols = Vec::new();
    for card in &schema.cards {
        for f in &card.fields {
            cols.push(match f.ty {
                FieldType::Int => Column::Int {
                    data: Vec::new(),
                    ncols: f.count,
                },
                FieldType::Float => Column::Float {
                    data: Vec::new(),
                    ncols: f.count,
                },
                FieldType::Str => Column::Str {
                    data: Vec::new(),
                    ncols: f.count,
                },
            });
        }
    }
    cols
}

fn field_names(schema: &Schema) -> Vec<String> {
    schema
        .cards
        .iter()
        .flat_map(|c| c.fields.iter().map(|f| f.name.clone()))
        .collect()
}

/// Parse one card line's fields into `cols` starting at column `ci`; returns
/// the next column index. Every field pushes exactly one value per element, so
/// all columns stay the same length (= row count) even on short/missing input.
fn parse_card_line(
    line: &[u8],
    card: &Card,
    format: CardFormat,
    cols: &mut [Column],
    mut ci: usize,
) -> usize {
    let line = strip_eol(line);
    let free = format == CardFormat::Free || memchr::memchr(b',', line).is_some();

    if free {
        let mut toks = line.split(|&c| c == b',');
        for f in &card.fields {
            for _ in 0..f.count {
                cols[ci].push(toks.next().unwrap_or(&[]));
            }
            ci += 1;
        }
    } else {
        let mut off = 0;
        for f in &card.fields {
            // Long format doubles each field width (I8->I16, E16->E32, ...).
            let fw = if format == CardFormat::Long {
                f.width * 2
            } else {
                f.width
            };
            for _ in 0..f.count {
                let slice = if off >= line.len() {
                    &[][..]
                } else {
                    &line[off..(off + fw).min(line.len())]
                };
                cols[ci].push(slice);
                off += fw;
            }
            ci += 1;
        }
    }
    ci
}

fn parse_sequential(files: &[ParsedFile], schema: &Schema) -> Table {
    let mut cols = empty_columns(schema);
    let k = schema.cards.len();

    for parsed in files {
        for block in &parsed.blocks {
            if !parsed
                .keyword_name(block)
                .eq_ignore_ascii_case(&schema.keyword)
            {
                continue;
            }
            let format = block.format;
            let lines: Vec<&[u8]> = parsed
                .body(block)
                .split(|&c| c == b'\n')
                .filter(|l| !is_skippable(l))
                .collect();

            let groups = if schema.repeat {
                lines.len() / k
            } else {
                usize::from(lines.len() >= k)
            };
            for g in 0..groups {
                let base = g * k;
                let mut ci = 0;
                for (kk, card) in schema.cards.iter().enumerate() {
                    ci = parse_card_line(lines[base + kk], card, format, &mut cols, ci);
                }
            }
        }
    }

    Table {
        columns: field_names(schema).into_iter().zip(cols).collect(),
    }
}

fn parse_parallel(files: &[ParsedFile], schema: &Schema) -> Table {
    // Numeric single-card keywords (`*NODE`, `*ELEMENT_*`) — the bulk ones — take
    // the fast two-pass path. A repeating card with a string field falls back to
    // the general per-chunk-partials path.
    if schema.cards[0].fields.iter().any(|f| f.ty == FieldType::Str) {
        parse_parallel_partials(files, schema)
    } else {
        parse_parallel_numeric(files, schema)
    }
}

/// General parallel parse: each chunk builds partial columns, then they are
/// concatenated. Used when a column is a `String` (can't be filled in place).
fn parse_parallel_partials(files: &[ParsedFile], schema: &Schema) -> Table {
    let card = &schema.cards[0];
    let chunks = collect_chunks(files, &schema.keyword);

    let partials: Vec<Vec<Column>> = chunks
        .par_iter()
        .map(|(chunk, format)| {
            let mut cols = empty_columns(schema);
            for line in chunk.split(|&c| c == b'\n') {
                if is_skippable(line) {
                    continue;
                }
                parse_card_line(line, card, *format, &mut cols, 0);
            }
            cols
        })
        .collect();

    let mut cols = empty_columns(schema);
    for part in partials {
        for (into, from) in cols.iter_mut().zip(part) {
            into.extend(from);
        }
    }

    Table {
        columns: field_names(schema).into_iter().zip(cols).collect(),
    }
}

/// Base pointers into the output columns, for disjoint parallel writes.
///
/// SAFETY: this is only ever used to write, from each worker, the contiguous row
/// range this worker owns (its prefix-sum offset for a length equal to its own
/// row count). Those ranges partition the columns exactly, so no two workers
/// ever touch the same slot — the aliasing is disjoint and the raw writes are
/// sound. `i64`/`f64` are plain data with no invalid bit patterns, and every
/// slot is written before the `Table` is read (see `parse_parallel_numeric`).
struct ColPtrs(Vec<ColPtr>);
enum ColPtr {
    Int(*mut i64),
    Float(*mut f64),
}
// Disjoint parallel writes only (see the type doc). Impl on `ColPtr` too, since
// disjoint closure capture borrows the inner `Vec<ColPtr>`, not the wrapper.
unsafe impl Send for ColPtr {}
unsafe impl Sync for ColPtr {}
unsafe impl Send for ColPtrs {}
unsafe impl Sync for ColPtrs {}

#[inline]
fn write_num(col: &ColPtr, index: usize, raw: &[u8]) {
    match *col {
        ColPtr::Int(p) => unsafe { *p.add(index) = Field { raw }.as_i64().unwrap_or(0) },
        ColPtr::Float(p) => unsafe { *p.add(index) = Field { raw }.as_f64().unwrap_or(0.0) },
    }
}

/// Fast columnar parse for all-numeric single-card keywords. Two passes: count
/// rows per chunk to get each chunk's output offset, allocate the columns once,
/// then parse each chunk directly into its row range in parallel — no per-chunk
/// partial buffers and no final merge copy (both were the bottleneck on
/// GB-scale meshes).
// SAFETY (uninit_vec): the columns are allocated with `set_len` and left
// uninitialized, then pass 2 writes *every* slot (each data row writes one value
// per field across all columns, and the per-chunk row counts partition the rows
// exactly) before the `Table` is returned or read. `i64`/`f64` have no invalid
// bit patterns and no `Drop`, so even a panic mid-fill can't cause UB. Zeroing
// first would cost a full extra pass over gigabytes — the whole point is to touch
// the memory once.
#[allow(clippy::uninit_vec)]
fn parse_parallel_numeric(files: &[ParsedFile], schema: &Schema) -> Table {
    let card = &schema.cards[0];
    let chunks = collect_chunks(files, &schema.keyword);

    // Pass 1: rows (non-skippable lines) per chunk → prefix-sum offsets. Must
    // agree exactly with the line iteration in pass 2, or offsets would drift.
    let counts: Vec<usize> = chunks
        .par_iter()
        .map(|(bytes, _)| bytes.split(|&b| b == b'\n').filter(|l| !is_skippable(l)).count())
        .collect();
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut total_rows = 0usize;
    for &c in &counts {
        offsets.push(total_rows);
        total_rows += c;
    }

    // Allocate each column once, exactly sized. `set_len` leaves the buffer
    // uninitialized; pass 2 writes every slot (each row writes one value per
    // field across all columns), so nothing is read before it is written.
    let mut cols: Vec<Column> = card
        .fields
        .iter()
        .map(|f| {
            let len = total_rows * f.count;
            match f.ty {
                FieldType::Int => {
                    let mut data = Vec::<i64>::with_capacity(len);
                    unsafe { data.set_len(len) };
                    Column::Int { data, ncols: f.count }
                }
                // Str excluded by the caller.
                _ => {
                    let mut data = Vec::<f64>::with_capacity(len);
                    unsafe { data.set_len(len) };
                    Column::Float { data, ncols: f.count }
                }
            }
        })
        .collect();

    // Pass 2: parse each chunk straight into its disjoint row range.
    let ptrs = ColPtrs(
        cols.iter_mut()
            .map(|c| match c {
                Column::Int { data, .. } => ColPtr::Int(data.as_mut_ptr()),
                Column::Float { data, .. } => ColPtr::Float(data.as_mut_ptr()),
                Column::Str { .. } => unreachable!(),
            })
            .collect(),
    );
    let counts_of = |i: usize| card.fields[i].count;
    chunks
        .par_iter()
        .zip(offsets.par_iter())
        .for_each(|((bytes, format), &off)| {
            let mut row = off;
            for line in bytes.split(|&b| b == b'\n') {
                if is_skippable(line) {
                    continue;
                }
                let line = strip_eol(line);
                let free = *format == CardFormat::Free || memchr::memchr(b',', line).is_some();
                if free {
                    let mut toks = line.split(|&c| c == b',');
                    for (fi, f) in card.fields.iter().enumerate() {
                        let base = row * counts_of(fi);
                        for j in 0..f.count {
                            write_num(&ptrs.0[fi], base + j, toks.next().unwrap_or(&[]));
                        }
                    }
                } else {
                    let mut o = 0;
                    for (fi, f) in card.fields.iter().enumerate() {
                        let fw = if *format == CardFormat::Long { f.width * 2 } else { f.width };
                        let base = row * counts_of(fi);
                        for j in 0..f.count {
                            let slice = if o >= line.len() {
                                &[][..]
                            } else {
                                &line[o..(o + fw).min(line.len())]
                            };
                            write_num(&ptrs.0[fi], base + j, slice);
                            o += fw;
                        }
                    }
                }
                row += 1;
            }
        });
    drop(ptrs);

    Table {
        columns: field_names(schema).into_iter().zip(cols).collect(),
    }
}

/// 1-based line of a block's `*KEYWORD` line, for clickable locations.
pub(crate) fn block_line(file: &ParsedFile, block: &Block) -> usize {
    1 + file.src()[..block.name_start]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::ParsedFile;
    use crate::parser::split_blocks;

    fn parsed(src: &[u8]) -> ParsedFile {
        ParsedFile::new("deck.k".into(), src.to_vec(), split_blocks(src))
    }

    #[test]
    fn schema_reproduces_nodes() {
        // Mixed: free-format and fixed-width node cards.
        let fixed = format!("{:>8}{:>16.6}{:>16.6}{:>16.6}", 3, 4.0, 5.0, 6.0);
        let src = format!("*NODE\n1,0.0,0.0,0.0\n2,1.0,2.0,3.0\n{}\n*END\n", fixed);
        let p = parsed(src.as_bytes());

        let schema = Schema::new("NODE").card(
            Card::new()
                .int("nid", 8)
                .float("x", 16)
                .float("y", 16)
                .float("z", 16),
        );
        let t = parse_schema(&p, &schema);

        assert_eq!(t.rows(), 3);
        assert_eq!(t.column("nid").unwrap().as_int().unwrap(), &[1, 2, 3]);
        assert_eq!(t.column("z").unwrap().as_float().unwrap(), &[0.0, 3.0, 6.0]);
    }

    #[test]
    fn schema_handles_multi_card_repeating() {
        // *PART: free-text title line + data line, two parts.
        let src = b"*PART\nsteel bracket\n1,2,3\nalu panel\n10,20,30\n";
        let p = parsed(src);

        let schema = Schema::new("PART")
            .card(Card::new().str("title", 80))
            .card(Card::new().int("pid", 8).int("secid", 8).int("mid", 8));
        let t = parse_schema(&p, &schema);

        assert_eq!(t.rows(), 2);
        assert_eq!(
            t.column("title").unwrap().as_str().unwrap(),
            &["steel bracket", "alu panel"]
        );
        assert_eq!(t.column("pid").unwrap().as_int().unwrap(), &[1, 10]);
        assert_eq!(t.column("mid").unwrap().as_int().unwrap(), &[3, 30]);
    }

    #[test]
    fn schema_array_field_makes_a_wide_column() {
        // *ELEMENT_SHELL: eid, pid, 4 nodes as one 4-wide column.
        let src = b"*ELEMENT_SHELL\n1,10,1,2,3,4\n2,10,5,6,7,8\n";
        let p = parsed(src);

        let schema = Schema::new("ELEMENT_SHELL").card(
            Card::new()
                .int("eid", 8)
                .int("pid", 8)
                .int_array("nodes", 4, 8),
        );
        let t = parse_schema(&p, &schema);

        assert_eq!(t.rows(), 2);
        let nodes = t.column("nodes").unwrap();
        assert_eq!(nodes.rows(), 2);
        assert_eq!(nodes.as_int().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]); // row-major 2x4
    }

    #[test]
    fn table_iter_yields_row_views() {
        let src = b"*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n*MAT_ELASTIC\n2,2.7e-9,70000.0,0.33\n";
        let p = parsed(src);
        let schema = Schema::new("MAT_ELASTIC").card(
            Card::new()
                .int("mid", 8)
                .float("ro", 16)
                .float("e", 16)
                .float("pr", 16),
        );
        let t = parse_schema(&p, &schema);
        let rows: Vec<_> = t
            .iter()
            .map(|r| (r.int("mid").unwrap(), r.float("e").unwrap()))
            .collect();
        assert_eq!(rows, vec![(1, 210000.0), (2, 70000.0)]);
    }

    #[test]
    fn schema_single_entity_per_block() {
        // repeat=false: one row per matching block.
        let src = b"*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n*MAT_ELASTIC\n2,2.7e-9,70000.0,0.33\n";
        let p = parsed(src);

        let schema = Schema::new("MAT_ELASTIC").card(
            Card::new()
                .int("mid", 8)
                .float("ro", 16)
                .float("e", 16)
                .float("pr", 16),
        );
        let t = parse_schema(&p, &schema);

        assert_eq!(t.rows(), 2);
        assert_eq!(t.column("mid").unwrap().as_int().unwrap(), &[1, 2]);
        assert_eq!(
            t.column("e").unwrap().as_float().unwrap(),
            &[210000.0, 70000.0]
        );
    }
}
