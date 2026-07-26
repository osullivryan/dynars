use std::fs::File;
use std::path::{Path, PathBuf};

use memchr::{memchr, memrchr};
use memmap2::Mmap;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;

use crate::file::{Block, CardFormat, ParsedFile, Source};
use crate::include::{FileParseResult, IncludeDirective, IncludeKind};

#[inline(always)]
fn match_include_keyword(line: &[u8]) -> Option<IncludeKind> {
    let len = line.len();
    if len < 8 {
        return None;
    }

    if !eq_ci(&line[1..], b"INCLUDE") {
        return None;
    }

    if len == 8 || matches!(line[8], b'\r' | b'\n' | b' ' | b'\t') {
        return Some(IncludeKind::Include);
    }

    let rest = trim_right(&line[8..]);

    if rest.is_empty() {
        return Some(IncludeKind::Include);
    }

    if !rest[0].eq_ignore_ascii_case(&b'_') {
        return None;
    }

    if eq_ci_full(rest, b"_PATH") {
        Some(IncludeKind::IncludePath)
    } else if eq_ci_full(rest, b"_PATH_RELATIVE") {
        Some(IncludeKind::IncludePathRelative)
    } else if eq_ci_full(rest, b"_TRANSFORM") {
        Some(IncludeKind::IncludeTransform)
    } else if eq_ci_full(rest, b"_AUTO_ZZFREE") {
        Some(IncludeKind::IncludeAutoZzfree)
    } else if eq_ci_full(rest, b"_BINARY") {
        Some(IncludeKind::IncludeBinary)
    } else if eq_ci_full(rest, b"_COMPENSATED") {
        Some(IncludeKind::IncludeCompensated)
    } else if eq_ci_full(rest, b"_STAMPED_PART") {
        Some(IncludeKind::IncludeStampedPart)
    } else {
        None
    }
}

#[inline(always)]
fn eq_ci(hay: &[u8], needle: &[u8]) -> bool {
    if hay.len() < needle.len() {
        return false;
    }
    for i in 0..needle.len() {
        if hay[i].to_ascii_uppercase() != needle[i] {
            return false;
        }
    }
    true
}

#[inline(always)]
fn eq_ci_full(hay: &[u8], needle: &[u8]) -> bool {
    hay.len() == needle.len() && eq_ci(hay, needle)
}

#[inline(always)]
fn trim_right(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && matches!(s[end - 1], b' ' | b'\t' | b'\r' | b'\n') {
        end -= 1;
    }
    &s[..end]
}

#[inline(always)]
fn trim(s: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < s.len() && matches!(s[start], b' ' | b'\t') {
        start += 1;
    }
    trim_right(&s[start..])
}

#[inline(always)]
fn find_line_end(data: &[u8], pos: usize) -> usize {
    match memchr(b'\n', &data[pos..]) {
        Some(offset) => pos + offset,
        None => data.len(),
    }
}

#[inline(always)]
fn get_line(data: &[u8], start: usize, end: usize) -> &[u8] {
    if end > start && data[end - 1] == b'\r' {
        &data[start..end - 1]
    } else {
        &data[start..end]
    }
}

/// Files at or above this size get their scan split across cores; smaller
/// files scan on one thread (chunking overhead isn't worth it, and it avoids
/// oversubscribing when the include-tree pool is already parsing many files).
const MIN_PARALLEL_SCAN: usize = 8 * 1024 * 1024; // 8MB

/// Parse a file for include directives.
///
/// The file is memory-mapped (no read() copy) and scanned for '*' at line
/// starts with SIMD memchr — '*' almost never appears in numeric data, so
/// millions of NODE/ELEMENT lines are skipped in 32-byte strides. Large files
/// are scanned in parallel over line-aligned chunks; because the whole file is
/// one contiguous mapping, a chunk that finds a keyword near its end reads
/// forward past the boundary for the filename, so straddling lines are handled
/// with no duplication.
///
/// mmap (rather than streaming read) wins on Linux, where page faults resolve
/// in parallel and madvise drives readahead; on macOS cold faults are
/// single-threaded, but warm data is unaffected and cold huge files are
/// disk-bound regardless.
///
/// Finds includes ANYWHERE in the file — correct because every byte is scanned.
pub fn parse_file_from_path(file_path: &Path, include_paths: &[PathBuf]) -> FileParseResult {
    let parent_dir = file_path.parent().unwrap_or(Path::new("."));

    let file = File::open(file_path).expect("Cannot open file");
    let file_size = file.metadata().map(|m| m.len() as usize).unwrap_or(0);

    if file_size == 0 {
        return FileParseResult {
            path: file_path.to_path_buf(),
            byte_count: 0,
            includes: Vec::new(),
        };
    }

    // SAFETY: standard mmap caveat — undefined behaviour if the file is
    // truncated/modified by another process while mapped.
    let mmap = unsafe { Mmap::map(&file) }.expect("Cannot mmap file");
    #[cfg(unix)]
    let _ = mmap.advise(memmap2::Advice::Sequential);
    let data: &[u8] = &mmap;

    let includes = scan_includes(data, parent_dir, include_paths);

    FileParseResult {
        path: file_path.to_path_buf(),
        byte_count: data.len(),
        includes,
    }
}

/// Scan mapped file bytes for include directives, splitting large files across
/// cores. Results are returned in file order.
fn scan_includes(
    data: &[u8],
    parent_dir: &Path,
    include_paths: &[PathBuf],
) -> Vec<IncludeDirective> {
    if data.len() < MIN_PARALLEL_SCAN {
        return scan_range(data, 0, data.len(), parent_dir, include_paths);
    }

    let bounds = line_aligned_bounds(data, rayon::current_num_threads());
    bounds
        .par_windows(2)
        .map(|w| scan_range(data, w[0], w[1], parent_dir, include_paths))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// Scan `data[start..end]` for `*` at line starts. Every chunk boundary is a
/// line start, so `data[pos - 1] == '\n'` correctly identifies line starts even
/// at `pos == start`. `process_star_line` reads forward into the full `data`
/// slice, so a keyword whose filename lands in the next chunk is still resolved.
fn scan_range(
    data: &[u8],
    start: usize,
    end: usize,
    parent_dir: &Path,
    include_paths: &[PathBuf],
) -> Vec<IncludeDirective> {
    let mut includes = Vec::new();
    for off in memchr::memchr_iter(b'*', &data[start..end]) {
        let pos = start + off;
        if pos == 0 || data[pos - 1] == b'\n' {
            process_star_line(data, pos, parent_dir, include_paths, &mut includes);
        }
    }
    includes
}

/// Split `data` into up to `n` boundaries, each snapped forward to the start of
/// a line, so chunks contain only whole lines.
fn line_aligned_bounds(data: &[u8], n: usize) -> Vec<usize> {
    let n = n.max(1);
    let mut bounds = Vec::with_capacity(n + 1);
    bounds.push(0usize);
    for i in 1..n {
        let target = data.len() * i / n;
        let cut = match memchr(b'\n', &data[target..]) {
            Some(off) => target + off + 1,
            None => data.len(),
        };
        if cut > *bounds.last().unwrap() && cut < data.len() {
            bounds.push(cut);
        }
    }
    bounds.push(data.len());
    bounds
}

#[inline]
fn process_star_line(
    data: &[u8],
    star_pos: usize,
    parent_dir: &Path,
    include_paths: &[PathBuf],
    includes: &mut Vec<IncludeDirective>,
) {
    let line_end_nl = find_line_end(data, star_pos);
    let line = get_line(data, star_pos, line_end_nl);

    if let Some(kind) = match_include_keyword(line) {
        let mut fname_start = if line_end_nl < data.len() {
            line_end_nl + 1
        } else {
            data.len()
        };

        loop {
            if fname_start >= data.len() {
                break;
            }

            let fname_end_nl = find_line_end(data, fname_start);
            let fname_line = get_line(data, fname_start, fname_end_nl);

            if !fname_line.is_empty() && fname_line[0] == b'$' {
                fname_start = if fname_end_nl < data.len() {
                    fname_end_nl + 1
                } else {
                    data.len()
                };
                continue;
            }

            let filename = trim(fname_line);
            if !filename.is_empty() {
                let path_str = String::from_utf8_lossy(filename);
                let path_str = path_str.trim();
                let resolved = resolve_include_path(path_str, parent_dir, include_paths);
                includes.push(IncludeDirective {
                    kind,
                    raw_path: path_str.to_string(),
                    resolved_path: resolved,
                });
            }
            break;
        }
    }
}

fn resolve_include_path(raw: &str, parent_dir: &Path, include_paths: &[PathBuf]) -> PathBuf {
    let p = Path::new(raw);

    if p.is_absolute() {
        return p.to_path_buf();
    }

    let candidate = parent_dir.join(p);
    if candidate.exists() {
        return candidate;
    }

    for ip in include_paths {
        let candidate = ip.join(p);
        if candidate.exists() {
            return candidate;
        }
    }

    parent_dir.join(p)
}

// ---------------------------------------------------------------------------
// Phase 1: block span index
//
// A second, opt-in parsing path for marshalling. Unlike the streaming
// include scanner above, this reads the whole file into an owned buffer so
// blocks can be addressed by stable byte offsets and edited/rewritten later.
// The tree-only path keeps streaming for throughput; this path trades that
// for random access, which is cheap since a single file scans at ~14 GB/s.
// ---------------------------------------------------------------------------

/// Read a file and split it into keyword blocks (see [`ParsedFile`]).
///
/// The blocks tile the source exactly, so `ParsedFile::to_bytes()` reproduces
/// the input byte-for-byte.
pub fn parse_file_blocks(file_path: &Path) -> std::io::Result<ParsedFile> {
    let file = File::open(file_path)?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

    // mmap of a zero-length file fails on some platforms; use an empty buffer.
    if file_size == 0 {
        return Ok(ParsedFile::new(
            file_path.to_path_buf(),
            Vec::new(),
            Vec::new(),
        ));
    }

    // SAFETY: standard mmap caveat — undefined behaviour if the file is
    // truncated/modified by another process while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    #[cfg(unix)]
    let _ = mmap.advise(memmap2::Advice::Sequential);
    let blocks = split_blocks(&mmap);
    Ok(ParsedFile::from_source(
        file_path.to_path_buf(),
        Source::Mapped(mmap),
        blocks,
    ))
}

/// Split raw file bytes into keyword blocks that tile the input.
///
/// Each block spans an optional run of leading trivia (blank / `$`-comment
/// lines), the keyword line starting with `*` in column 1, and the data cards
/// up to the next block. The first block absorbs any leading bytes before the
/// first keyword so the tiling is always gap-free.
pub fn split_blocks(source: &[u8]) -> Vec<Block> {
    // Keyword lines: every '*' that sits in column 1.
    let mut kw: Vec<usize> = Vec::new();
    for pos in memchr::memchr_iter(b'*', source) {
        if pos == 0 || source[pos - 1] == b'\n' {
            kw.push(pos);
        }
    }
    if kw.is_empty() {
        return Vec::new();
    }

    let fmt = if detect_long(source, &kw) {
        CardFormat::Long
    } else {
        CardFormat::Fixed
    };

    // Block boundaries. The first starts at 0 (absorbing any preamble); each
    // later block starts where its leading trivia begins.
    let n = kw.len();
    let mut starts: Vec<usize> = Vec::with_capacity(n);
    starts.push(0);
    for &k in &kw[1..] {
        starts.push(trivia_start(source, k));
    }

    let mut blocks = Vec::with_capacity(n);
    for j in 0..n {
        let span_start = starts[j];
        let span_end = if j + 1 < n {
            starts[j + 1]
        } else {
            source.len()
        };
        let name_start = kw[j];
        let body_start = line_end_after(source, name_start).min(span_end);
        blocks.push(Block {
            span: span_start..span_end,
            name_start,
            body_start,
            format: fmt,
        });
    }
    blocks
}

/// Walk backwards from a keyword line over the contiguous run of blank /
/// `$`-comment lines that should attach to it as leading trivia. Stops at the
/// first data line (belongs to the previous block) or start of file.
fn trivia_start(source: &[u8], name_start: usize) -> usize {
    let mut pos = name_start; // always a line start
    while pos > 0 {
        // The previous line ends at the '\n' at pos-1.
        let line_start = match memrchr(b'\n', &source[..pos - 1]) {
            Some(k) => k + 1,
            None => 0,
        };
        if is_trivia_line(&source[line_start..pos - 1]) {
            pos = line_start;
        } else {
            break;
        }
    }
    pos
}

#[inline]
fn is_trivia_line(line: &[u8]) -> bool {
    let t = trim(line);
    t.is_empty() || t[0] == b'$'
}

/// Offset just past the newline that terminates the line beginning at `from`.
#[inline]
fn line_end_after(source: &[u8], from: usize) -> usize {
    match memchr(b'\n', &source[from..]) {
        Some(off) => from + off + 1,
        None => source.len(),
    }
}

/// Detect deck-wide long format: a `*KEYWORD` line carrying `LONG=Y` or
/// `LONG=S` (case-insensitive, tolerant of spaces around `=`).
fn detect_long(source: &[u8], kw: &[usize]) -> bool {
    for &pos in kw {
        let line = trim_right(&source[pos..line_end_after(source, pos)]);
        // Only *KEYWORD carries the deck-wide LONG option.
        if !eq_ci(&line[1..], b"KEYWORD") {
            continue;
        }
        if let Some(rel) = find_ci(line, b"LONG") {
            let mut i = rel + 4;
            while i < line.len() && matches!(line[i], b' ' | b'\t') {
                i += 1;
            }
            if i < line.len() && line[i] == b'=' {
                i += 1;
                while i < line.len() && matches!(line[i], b' ' | b'\t') {
                    i += 1;
                }
                if i < line.len() && matches!(line[i].to_ascii_uppercase(), b'Y' | b'S') {
                    return true;
                }
            }
        }
    }
    false
}

/// Case-insensitive substring search, returns the start offset of `needle`.
fn find_ci(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    for start in 0..=hay.len() - needle.len() {
        if eq_ci(&hay[start..], needle) {
            return Some(start);
        }
    }
    None
}

/// Extract include directives from a parsed file's block index.
///
/// Equivalent to the streaming scanner's include detection, but driven by the
/// block model so the marshalling path shares a single source of truth.
pub fn extract_includes(parsed: &ParsedFile, include_paths: &[PathBuf]) -> Vec<IncludeDirective> {
    let parent_dir = parsed.path.parent().unwrap_or(Path::new("."));
    let mut includes = Vec::new();

    for block in &parsed.blocks {
        let Some(kind) = match_include_keyword(parsed.name_line(block)) else {
            continue;
        };
        if let Some(filename) = first_filename(parsed.body(block)) {
            let path_str = String::from_utf8_lossy(filename);
            let path_str = path_str.trim();
            let resolved = resolve_include_path(path_str, parent_dir, include_paths);
            includes.push(IncludeDirective {
                kind: kind.clone(),
                raw_path: path_str.to_string(),
                resolved_path: resolved,
            });
        }
    }

    includes
}

/// Find the filename card in an `*INCLUDE` body: skip `$`-comment lines, then
/// take the first following line (matching the streaming scanner's behaviour).
fn first_filename(body: &[u8]) -> Option<&[u8]> {
    let mut start = 0;
    while start < body.len() {
        let end = match memchr(b'\n', &body[start..]) {
            Some(off) => start + off,
            None => body.len(),
        };
        let line = get_line(body, start, end);
        if !line.is_empty() && line[0] == b'$' {
            start = if end < body.len() {
                end + 1
            } else {
                body.len()
            };
            continue;
        }
        let f = trim(line);
        return if f.is_empty() { None } else { Some(f) };
    }
    None
}

// ---------------------------------------------------------------------------
// Phase 2: tokenizer + format-aware field splitter
//
// Lazily turns a block body into rows of fields. This is the "generic
// everything" tier: any of the ~2000 keywords is representable without a
// hand-written struct. Fixed/Long use uniform-width columns (best effort for
// the long tail; typed/columnar parsers in later phases supply exact widths),
// Free splits on commas. Nothing is parsed until the iterator is advanced.
// ---------------------------------------------------------------------------

/// Uniform column widths for the generic splitter. Individual keywords have
/// their own field widths; these are the common defaults used when no typed
/// schema applies.
const FIXED_WIDTH: usize = 8;
const LONG_WIDTH: usize = 20;

/// A zero-copy view of one field, borrowing the source bytes.
#[derive(Clone, Copy, Debug)]
pub struct Field<'a> {
    pub raw: &'a [u8],
}

impl<'a> Field<'a> {
    /// The field with surrounding whitespace removed.
    #[inline]
    pub fn trimmed(&self) -> &'a [u8] {
        trim(self.raw)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.trimmed().is_empty()
    }

    /// Trimmed field as UTF-8 (empty string if the field is not valid UTF-8).
    #[inline]
    pub fn as_str(&self) -> &'a str {
        std::str::from_utf8(self.trimmed()).unwrap_or("")
    }

    /// Parse as an integer, tolerating surrounding whitespace and a leading `+`.
    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        let t = self.trimmed();
        if t.is_empty() {
            return None;
        }
        // Fast path: lexical handles plain and signed integers.
        if let Ok(v) = lexical_core::parse::<i64>(t) {
            return Some(v);
        }
        // Fallback: explicit leading '+', which lexical rejects by default.
        let s = std::str::from_utf8(t).ok()?;
        s.strip_prefix('+').unwrap_or(s).parse().ok()
    }

    /// Parse as a float, tolerating LS-DYNA / Fortran quirks (see
    /// [`parse_dyna_float`]).
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        let t = self.trimmed();
        if t.is_empty() {
            return None;
        }
        // Fast path: lexical on the raw bytes — no UTF-8 validation, no str
        // round-trip. Handles standard/scientific floats, the vast majority.
        if let Ok(v) = lexical_core::parse::<f64>(t) {
            return Some(v);
        }
        // Fallback: Fortran quirks (D exponent, implicit exponent, leading '+').
        parse_dyna_float(std::str::from_utf8(t).ok()?)
    }
}

/// Parse a float the way LS-DYNA (Fortran) writes them:
/// - Fortran `D` exponent: `1.5D+3` -> `1.5E+3`
/// - implicit exponent: `1.234-5` -> `1.234E-5` (sign after a digit with no `E`)
pub fn parse_dyna_float(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    // Fast path: lexical parses standard/scientific floats several times faster
    // than the std parser, and this handles the overwhelming majority of cards.
    if let Ok(v) = lexical_core::parse::<f64>(s.as_bytes()) {
        return Some(v);
    }
    if let Ok(v) = s.parse::<f64>() {
        return v.into();
    }

    // Normalize a Fortran 'D'/'d' exponent to 'E'.
    let mut buf: String = s
        .chars()
        .map(|c| if c == 'd' || c == 'D' { 'E' } else { c })
        .collect();
    if let Ok(v) = buf.parse::<f64>() {
        return v.into();
    }

    // Insert an implicit 'E' before a +/- exponent sign that follows a digit
    // or '.', e.g. "1.234-5" or "1.234+5" (but not a leading sign).
    let bytes = buf.as_bytes();
    let mut insert_at = None;
    for i in 1..bytes.len() {
        if matches!(bytes[i], b'+' | b'-') && matches!(bytes[i - 1], b'0'..=b'9' | b'.') {
            insert_at = Some(i);
            break;
        }
    }
    if let Some(i) = insert_at {
        buf.insert(i, 'E');
        if let Ok(v) = buf.parse::<f64>() {
            return v.into();
        }
    }
    None
}

/// Split one data line into fields according to `format`.
///
/// A line containing a comma is always read as free format, matching
/// LS-DYNA's own rule, regardless of the block's declared format.
pub fn split_fields(line: &[u8], format: CardFormat) -> Vec<Field<'_>> {
    if format == CardFormat::Free || memchr(b',', line).is_some() {
        return trim_eol(line)
            .split(|&c| c == b',')
            .map(|raw| Field { raw })
            .collect();
    }

    let width = if format == CardFormat::Long {
        LONG_WIDTH
    } else {
        FIXED_WIDTH
    };
    let line = trim_right(line); // drop trailing padding / EOL
    let mut fields = Vec::with_capacity(line.len() / width + 1);
    let mut i = 0;
    while i < line.len() {
        let end = (i + width).min(line.len());
        fields.push(Field { raw: &line[i..end] });
        i = end;
    }
    fields
}

/// Strip a trailing `\r`/`\n` without touching interior or leading bytes.
#[inline]
fn trim_eol(s: &[u8]) -> &[u8] {
    let mut end = s.len();
    while end > 0 && matches!(s[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &s[..end]
}

/// Lazy iterator over the data cards of a block body. Skips `$`-comment and
/// blank lines; yields the split fields of every other line.
pub struct CardIter<'a> {
    body: &'a [u8],
    format: CardFormat,
    pos: usize,
}

impl<'a> Iterator for CardIter<'a> {
    type Item = Vec<Field<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.body.len() {
            let end = match memchr(b'\n', &self.body[self.pos..]) {
                Some(off) => self.pos + off,
                None => self.body.len(),
            };
            let line = get_line(self.body, self.pos, end);
            self.pos = if end < self.body.len() {
                end + 1
            } else {
                self.body.len()
            };

            if (!line.is_empty() && line[0] == b'$') || trim(line).is_empty() {
                continue;
            }
            return Some(split_fields(line, self.format));
        }
        None
    }
}

impl ParsedFile {
    /// Lazily iterate the data cards of a block (fields per line), using the
    /// block's detected format. Nothing is parsed until the iterator advances.
    pub fn cards<'a>(&'a self, block: &Block) -> CardIter<'a> {
        CardIter {
            body: self.body(block),
            format: block.format,
            pos: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: owned generic keyword model + editing
//
// `Keyword` is an owned, allocation-backed view of a block — the editable
// counterpart to the zero-copy `cards()` iterator. It carries no borrow of the
// source, so it crosses the FFI boundary to Python cleanly and can be mutated
// and written back. Writing an edited keyword re-emits it in free (comma)
// format, which LS-DYNA always accepts, so regeneration is value-lossless
// without needing per-keyword field widths.
// ---------------------------------------------------------------------------

/// An owned representation of one keyword block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keyword {
    /// Keyword name without the leading `*`, e.g. `ELEMENT_SHELL_THICKNESS`.
    pub name: String,
    /// Whitespace-separated tokens after the name on the keyword line
    /// (e.g. `LONG=Y`).
    pub options: Vec<String>,
    /// Data cards, each a row of trimmed field strings.
    pub cards: Vec<Vec<String>>,
}

impl ParsedFile {
    /// Materialize a block as an owned, editable [`Keyword`].
    pub fn keyword(&self, block: &Block) -> Keyword {
        let name = self.keyword_name(block).to_string();

        // Options are whatever follows the name token on the keyword line.
        let line = trim_right(self.name_line(block));
        let after_star = if line.first() == Some(&b'*') {
            &line[1..]
        } else {
            line
        };
        let options = match after_star.iter().position(|&c| matches!(c, b' ' | b'\t')) {
            Some(sp) => std::str::from_utf8(&after_star[sp..])
                .unwrap_or("")
                .split_whitespace()
                .map(|s| s.to_string())
                .collect(),
            None => Vec::new(),
        };

        let cards = self
            .cards(block)
            .map(|c| c.iter().map(|f| f.as_str().to_string()).collect())
            .collect();

        Keyword {
            name,
            options,
            cards,
        }
    }

    /// Replace a block with an edited [`Keyword`], preserving its leading
    /// trivia. The keyword line and cards are regenerated (cards in free
    /// format); the block is marked dirty.
    pub fn set_keyword(&mut self, block_index: usize, kw: &Keyword) {
        let block = self.blocks[block_index].clone();
        let mut out = Vec::new();
        out.extend_from_slice(self.trivia(&block));
        out.push(b'*');
        out.extend_from_slice(kw.name.as_bytes());
        for opt in &kw.options {
            out.push(b' ');
            out.extend_from_slice(opt.as_bytes());
        }
        out.push(b'\n');
        for card in &kw.cards {
            out.extend_from_slice(card.join(",").as_bytes());
            out.push(b'\n');
        }
        self.set_block_bytes(block_index, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::ParsedFile;

    fn parsed(src: &[u8]) -> ParsedFile {
        ParsedFile::new(PathBuf::from("deck.k"), src.to_vec(), split_blocks(src))
    }

    /// The core lossless guarantee: blocks tile the source, so re-emitting
    /// reproduces the input byte-for-byte across a range of shapes.
    #[test]
    fn round_trip_is_byte_exact() {
        let cases: &[&[u8]] = &[
            b"*KEYWORD\n*NODE\n1,0.0,0.0,0.0\n*END\n",
            b"*KEYWORD\r\n*NODE\r\n1,0.0,0.0,0.0\r\n*END\r\n", // CRLF
            b"*KEYWORD\n*END",                                 // no trailing newline
            b"$ leading comment\n$ another\n*KEYWORD\n*END\n", // preamble comments
            b"garbage before any keyword\n*KEYWORD\n*END\n",   // non-comment preamble
            b"no keywords at all\njust text\n",                // zero blocks
            b"",                                               // empty file
        ];
        for &src in cases {
            let p = parsed(src);
            assert_eq!(p.to_bytes(), src, "round-trip failed for {:?}", src);
        }
    }

    #[test]
    fn blocks_partition_source_without_gaps() {
        let src = b"*KEYWORD\n*NODE\n1,0.0,0.0,0.0\n*ELEMENT_SHELL\n1,1,1,2,3,4\n*END\n";
        let p = parsed(src);
        assert_eq!(p.blocks.len(), 4);
        // Contiguous, covering [0, len).
        assert_eq!(p.blocks[0].span.start, 0);
        for w in p.blocks.windows(2) {
            assert_eq!(w[0].span.end, w[1].span.start);
        }
        assert_eq!(p.blocks.last().unwrap().span.end, src.len());
    }

    #[test]
    fn keyword_names_parsed() {
        let src = b"*KEYWORD\n*ELEMENT_SHELL_THICKNESS\n*INCLUDE_TRANSFORM\nsub.k\n";
        let p = parsed(src);
        let names: Vec<_> = p.blocks.iter().map(|b| p.keyword_name(b)).collect();
        assert_eq!(
            names,
            vec!["KEYWORD", "ELEMENT_SHELL_THICKNESS", "INCLUDE_TRANSFORM"]
        );
    }

    #[test]
    fn comments_attach_as_leading_trivia() {
        let src = b"*KEYWORD\n$ describes the include\n*INCLUDE\nsub.k\n";
        let p = parsed(src);
        // The comment belongs to the *INCLUDE block, not *KEYWORD.
        let inc = p
            .blocks
            .iter()
            .find(|b| p.keyword_name(b) == "INCLUDE")
            .unwrap();
        assert_eq!(p.trivia(inc), b"$ describes the include\n");
        // ...and *KEYWORD's body does not contain it.
        let kw = &p.blocks[0];
        assert!(!p.body(kw).windows(9).any(|w| w == b"describes"));
    }

    #[test]
    fn extract_includes_from_blocks() {
        let src = b"*KEYWORD\n\
                    *INCLUDE\nmesh.k\n\
                    *INCLUDE_TRANSFORM\n$ a comment\ntransform.k\n\
                    *include\nlower.k\n\
                    *END\n";
        let p = parsed(src);
        let incs = extract_includes(&p, &[]);
        let raw: Vec<_> = incs.iter().map(|i| i.raw_path.as_str()).collect();
        assert_eq!(raw, vec!["mesh.k", "transform.k", "lower.k"]);
        assert_eq!(incs[0].kind, IncludeKind::Include);
        assert_eq!(incs[1].kind, IncludeKind::IncludeTransform);
        assert_eq!(incs[2].kind, IncludeKind::Include);
    }

    #[test]
    fn detects_deck_wide_long_format() {
        let fixed = parsed(b"*KEYWORD\n*NODE\n1,0.0\n");
        assert_eq!(fixed.blocks[0].format, CardFormat::Fixed);

        for src in [
            &b"*KEYWORD LONG=Y\n*NODE\n"[..],
            &b"*keyword long=s\n*NODE\n"[..],
            &b"*KEYWORD LONG = Y\n*NODE\n"[..],
        ] {
            let p = parsed(src);
            assert!(
                p.blocks.iter().all(|b| b.format == CardFormat::Long),
                "expected Long for {:?}",
                src
            );
        }
    }

    /// The block-driven include extraction must agree with the streaming
    /// scanner used by the include-tree builder.
    #[test]
    fn block_includes_match_streaming_scanner() {
        let src = b"*KEYWORD\n\
                    $ comment\n\
                    *INCLUDE\nmesh.k\n\
                    *NODE\n1,0.0,0.0,0.0\n2,1.0,1.0,1.0\n\
                    *INCLUDE_TRANSFORM\ntransform.k\n\
                    *END\n";

        let dir = std::env::temp_dir().join(format!("dynars_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("root.k");
        std::fs::write(&path, src).unwrap();

        let streaming = parse_file_from_path(&path, &[]);
        let parsed = parse_file_blocks(&path).unwrap();
        let block_incs = extract_includes(&parsed, &[]);

        let a: Vec<_> = streaming
            .includes
            .iter()
            .map(|i| i.raw_path.clone())
            .collect();
        let b: Vec<_> = block_incs.iter().map(|i| i.raw_path.clone()).collect();
        assert_eq!(a, b);

        std::fs::remove_dir_all(&dir).ok();
    }

    // --- Phase 2: tokenizer ------------------------------------------------

    #[test]
    fn free_format_splits_on_commas() {
        let fields = split_fields(b"1,0.5,-1.0,2.0e3\n", CardFormat::Fixed);
        let vals: Vec<_> = fields.iter().map(|f| f.as_str()).collect();
        assert_eq!(vals, vec!["1", "0.5", "-1.0", "2.0e3"]);
        assert_eq!(fields[0].as_i64(), Some(1));
        assert_eq!(fields[3].as_f64(), Some(2000.0));
    }

    #[test]
    fn fixed_format_splits_on_columns() {
        // Three 8-column fields: "       1" "     1.5" "     2.5".
        let line = b"       1     1.5     2.5";
        let fields = split_fields(line, CardFormat::Fixed);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].as_i64(), Some(1));
        assert_eq!(fields[1].as_f64(), Some(1.5));
        assert_eq!(fields[2].as_f64(), Some(2.5));
    }

    #[test]
    fn long_format_uses_wide_columns() {
        //           |------20 cols-----||------20 cols-----|
        let line = b"                   1                 2.5";
        let fields = split_fields(line, CardFormat::Long);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].as_i64(), Some(1));
        assert_eq!(fields[1].as_f64(), Some(2.5));
    }

    #[test]
    fn parses_fortran_floats() {
        assert_eq!(parse_dyna_float("1.5"), Some(1.5));
        assert_eq!(parse_dyna_float("1.5E+3"), Some(1500.0));
        assert_eq!(parse_dyna_float("1.5D+3"), Some(1500.0)); // Fortran D exponent
        assert_eq!(parse_dyna_float("1.234-5"), Some(1.234e-5)); // implicit exponent
        assert_eq!(parse_dyna_float("1.234+5"), Some(1.234e5));
        assert_eq!(parse_dyna_float("-2.0d-2"), Some(-0.02));
        assert_eq!(parse_dyna_float(""), None);
        assert_eq!(parse_dyna_float("abc"), None);
    }

    // --- Phase 4: owned model + editing -----------------------------------

    #[test]
    fn keyword_materializes_name_options_and_cards() {
        let src = b"*SECTION_SHELL TITLE\n1,2,0.0\n3.0,3.0,3.0,3.0\n";
        let p = parsed(src);
        let kw = p.keyword(&p.blocks[0]);
        assert_eq!(kw.name, "SECTION_SHELL");
        assert_eq!(kw.options, vec!["TITLE"]);
        assert_eq!(kw.cards.len(), 2);
        assert_eq!(kw.cards[0], vec!["1", "2", "0.0"]);
    }

    #[test]
    fn edited_keyword_rewrites_only_its_block() {
        let src = b"*KEYWORD\n\
                    *MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n\
                    *END\n";
        let mut p = parsed(src);
        let mat_idx = p
            .blocks
            .iter()
            .position(|b| p.keyword_name(b) == "MAT_ELASTIC")
            .unwrap();

        let mut kw = p.keyword(&p.blocks[mat_idx]);
        kw.cards[0][2] = "70000.0".to_string(); // change Young's modulus
        p.set_keyword(mat_idx, &kw);

        let text = String::from_utf8(p.to_bytes()).unwrap();
        // Edited block reflects the change...
        assert!(text.contains("*MAT_ELASTIC\n1,7.85e-9,70000.0,0.3\n"));
        // ...and the surrounding blocks are byte-for-byte intact.
        assert!(text.starts_with("*KEYWORD\n"));
        assert!(text.ends_with("*END\n"));
    }

    #[test]
    fn card_iter_skips_comments_and_blanks() {
        let src = b"*NODE\n\
                    $ a comment\n\
                    \n\
                    1,0.0,0.0,0.0\n\
                    2,1.0,2.0,3.0\n";
        let p = parsed(src);
        let node = &p.blocks[0];
        let cards: Vec<Vec<_>> = p
            .cards(node)
            .map(|c| c.iter().map(|f| f.as_str().to_string()).collect())
            .collect();
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0], vec!["1", "0.0", "0.0", "0.0"]);
        assert_eq!(cards[1][0], "2");
        assert_eq!(cards[1][3], "3.0");
    }
}
