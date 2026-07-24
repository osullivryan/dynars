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

use crate::keyword::{CardFormat, ParsedFile};
use crate::parser::Field;

// --- shared chunking infrastructure (parallel splitting of block bodies) ---

/// Minimum bytes per parallel chunk; below this a block stays one chunk.
const MIN_CHUNK: usize = 256 * 1024;

/// Split a block body into line-aligned chunks for parallel parsing. Each chunk
/// holds only whole lines, so concatenating their output preserves file order.
fn line_chunks(body: &[u8], max_chunks: usize) -> Vec<&[u8]> {
    if body.is_empty() {
        return Vec::new();
    }
    let n = (body.len() / MIN_CHUNK).clamp(1, max_chunks.max(1));
    if n <= 1 {
        return vec![body];
    }
    let mut bounds = Vec::with_capacity(n + 1);
    bounds.push(0usize);
    for i in 1..n {
        let target = body.len() * i / n;
        let cut = match memchr::memchr(b'\n', &body[target..]) {
            Some(off) => target + off + 1,
            None => body.len(),
        };
        if cut > *bounds.last().unwrap() && cut < body.len() {
            bounds.push(cut);
        }
    }
    bounds.push(body.len());
    bounds.windows(2).map(|w| &body[w[0]..w[1]]).collect()
}

/// Collect line-aligned chunks (with their format) across every block whose
/// name matches `keyword` — a single huge block fans out into many chunks, many
/// small blocks each contribute one.
fn collect_chunks<'a>(parsed: &'a ParsedFile, keyword: &str) -> Vec<(&'a [u8], CardFormat)> {
    let max_chunks = rayon::current_num_threads() * 4;
    let mut chunks = Vec::new();
    for block in &parsed.blocks {
        if !parsed.keyword_name(block).eq_ignore_ascii_case(keyword) {
            continue;
        }
        for c in line_chunks(parsed.body(block), max_chunks) {
            chunks.push((c, block.format));
        }
    }
    chunks
}

/// True for comment (`$`) and blank lines, which carry no card data.
#[inline]
fn is_skippable(line: &[u8]) -> bool {
    let indent = line.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
    line.is_empty()
        || line.get(indent) == Some(&b'$')
        || strip_eol(&line[indent..]).is_empty()
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

/// One field in a card: a name, a type, a fixed-format column width, and a
/// count (`> 1` makes it an array, producing an `N`-wide column).
#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub ty: FieldType,
    pub width: usize,
    pub count: usize,
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
    fn push(mut self, name: &str, ty: FieldType, width: usize, count: usize) -> Self {
        self.fields.push(FieldSpec {
            name: name.to_string(),
            ty,
            width,
            count: count.max(1),
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
        Schema { keyword: keyword.to_string(), cards: Vec::new(), repeat: true }
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
        if let Column::Int { data, .. } = self { Some(data) } else { None }
    }
    pub fn as_float(&self) -> Option<&[f64]> {
        if let Column::Float { data, .. } = self { Some(data) } else { None }
    }
    pub fn as_str(&self) -> Option<&[String]> {
        if let Column::Str { data, .. } = self { Some(data) } else { None }
    }
    /// Move the integer data out (for building typed structs without a copy).
    pub fn into_int(self) -> Option<Vec<i64>> {
        if let Column::Int { data, .. } = self { Some(data) } else { None }
    }
    pub fn into_float(self) -> Option<Vec<f64>> {
        if let Column::Float { data, .. } = self { Some(data) } else { None }
    }
    pub fn into_str(self) -> Option<Vec<String>> {
        if let Column::Str { data, .. } = self { Some(data) } else { None }
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
    if off >= line.len() { &[] } else { &line[off..(off + w).min(line.len())] }
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
    let chunks = collect_chunks(parsed, keyword);
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
    Table { columns: names.into_iter().map(|n| n.to_string()).zip(cols).collect() }
}

/// Implemented by `#[derive(Card)]`: provides one card's field layout, so cards
/// can be composed into multi-card keywords.
pub trait CardLayout {
    fn card() -> Card;
}

/// Parse every block matching `schema.keyword` into a columnar [`Table`].
///
/// Single-card repeating keywords (the bulk ones, `*NODE` / `*ELEMENT_*`) are
/// parsed in parallel across cores; multi-card or single-entity keywords are
/// parsed sequentially (they are almost always low volume).
pub fn parse_schema(parsed: &ParsedFile, schema: &Schema) -> Table {
    if schema.cards.is_empty() {
        return Table::default();
    }
    if schema.cards.len() == 1 && schema.repeat {
        parse_parallel(parsed, schema)
    } else {
        parse_sequential(parsed, schema)
    }
}

/// Fresh, empty columns in schema field order.
fn empty_columns(schema: &Schema) -> Vec<Column> {
    let mut cols = Vec::new();
    for card in &schema.cards {
        for f in &card.fields {
            cols.push(match f.ty {
                FieldType::Int => Column::Int { data: Vec::new(), ncols: f.count },
                FieldType::Float => Column::Float { data: Vec::new(), ncols: f.count },
                FieldType::Str => Column::Str { data: Vec::new(), ncols: f.count },
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
fn parse_card_line(line: &[u8], card: &Card, format: CardFormat, cols: &mut [Column], mut ci: usize) -> usize {
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
            let fw = if format == CardFormat::Long { f.width * 2 } else { f.width };
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

fn parse_sequential(parsed: &ParsedFile, schema: &Schema) -> Table {
    let mut cols = empty_columns(schema);
    let k = schema.cards.len();

    for block in &parsed.blocks {
        if !parsed.keyword_name(block).eq_ignore_ascii_case(&schema.keyword) {
            continue;
        }
        let format = block.format;
        let lines: Vec<&[u8]> = parsed
            .body(block)
            .split(|&c| c == b'\n')
            .filter(|l| !is_skippable(l))
            .collect();

        let groups = if schema.repeat { lines.len() / k } else { usize::from(lines.len() >= k) };
        for g in 0..groups {
            let base = g * k;
            let mut ci = 0;
            for (kk, card) in schema.cards.iter().enumerate() {
                ci = parse_card_line(lines[base + kk], card, format, &mut cols, ci);
            }
        }
    }

    Table { columns: field_names(schema).into_iter().zip(cols).collect() }
}

fn parse_parallel(parsed: &ParsedFile, schema: &Schema) -> Table {
    let card = &schema.cards[0];
    let chunks = collect_chunks(parsed, &schema.keyword);

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

    Table { columns: field_names(schema).into_iter().zip(cols).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword::ParsedFile;
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
            Card::new().int("nid", 8).float("x", 16).float("y", 16).float("z", 16),
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
        assert_eq!(t.column("title").unwrap().as_str().unwrap(), &["steel bracket", "alu panel"]);
        assert_eq!(t.column("pid").unwrap().as_int().unwrap(), &[1, 10]);
        assert_eq!(t.column("mid").unwrap().as_int().unwrap(), &[3, 30]);
    }

    #[test]
    fn schema_array_field_makes_a_wide_column() {
        // *ELEMENT_SHELL: eid, pid, 4 nodes as one 4-wide column.
        let src = b"*ELEMENT_SHELL\n1,10,1,2,3,4\n2,10,5,6,7,8\n";
        let p = parsed(src);

        let schema = Schema::new("ELEMENT_SHELL").card(
            Card::new().int("eid", 8).int("pid", 8).int_array("nodes", 4, 8),
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
            Card::new().int("mid", 8).float("ro", 16).float("e", 16).float("pr", 16),
        );
        let t = parse_schema(&p, &schema);
        let rows: Vec<_> = t.iter().map(|r| (r.int("mid").unwrap(), r.float("e").unwrap())).collect();
        assert_eq!(rows, vec![(1, 210000.0), (2, 70000.0)]);
    }

    #[test]
    fn schema_single_entity_per_block() {
        // repeat=false: one row per matching block.
        let src = b"*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n*MAT_ELASTIC\n2,2.7e-9,70000.0,0.33\n";
        let p = parsed(src);

        let schema = Schema::new("MAT_ELASTIC").card(
            Card::new().int("mid", 8).float("ro", 16).float("e", 16).float("pr", 16),
        );
        let t = parse_schema(&p, &schema);

        assert_eq!(t.rows(), 2);
        assert_eq!(t.column("mid").unwrap().as_int().unwrap(), &[1, 2]);
        assert_eq!(t.column("e").unwrap().as_float().unwrap(), &[210000.0, 70000.0]);
    }
}
