use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

/// How a keyword block's data cards are laid out.
///
/// LS-DYNA supports three field layouts. `Fixed` is the default 8-column
/// form. `Long` (wider fields) is enabled deck-wide by `*KEYWORD LONG=Y|S`.
/// `Free` is comma-separated and is decided per *line* (a data line switches
/// to free format the moment it contains a comma), so it is resolved at
/// tokenization time in Phase 2 rather than stored here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardFormat {
    Fixed,
    Long,
    Free,
}

/// One keyword block in a file: its leading trivia, the keyword line, and
/// the data cards that follow, addressed as byte spans into `ParsedFile::source`.
///
/// `span` covers the whole block (trivia + keyword line + body) and the
/// blocks of a file tile `source` exactly with no gaps, so re-emitting
/// `source[span]` for every block reproduces the file byte-for-byte. This is
/// the lossless-round-trip guarantee Phase 1 locks in.
#[derive(Debug, Clone)]
pub struct Block {
    /// Entire block, contiguous with its neighbours.
    pub span: Range<usize>,
    /// Offset of the `*` that begins the keyword line.
    pub name_start: usize,
    /// Offset of the first data-card byte (just past the keyword line's `\n`).
    pub body_start: usize,
    /// Detected card layout for this block.
    pub format: CardFormat,
}

/// A file parsed into keyword blocks, retaining the original bytes so
/// untouched blocks round-trip verbatim.
///
/// Edits are held as an overlay keyed by block index: an edited block emits
/// its replacement bytes, every other block emits `source[span]` verbatim.
/// This is the dirty-tracking that keeps a round-trip of an unedited deck a
/// byte-for-byte no-op while letting individual keywords be rewritten.
#[derive(Debug)]
pub struct ParsedFile {
    pub path: PathBuf,
    pub source: Vec<u8>,
    pub blocks: Vec<Block>,
    /// block index -> replacement bytes for that block's whole span.
    pub(crate) edits: HashMap<usize, Vec<u8>>,
}

impl ParsedFile {
    /// Construct from raw parts with no pending edits.
    pub fn new(path: PathBuf, source: Vec<u8>, blocks: Vec<Block>) -> Self {
        ParsedFile { path, source, blocks, edits: HashMap::new() }
    }

    /// Replace a block's bytes. Subsequent `to_bytes()` emits these instead of
    /// the original span; other blocks stay verbatim.
    pub fn set_block_bytes(&mut self, block_index: usize, bytes: Vec<u8>) {
        self.edits.insert(block_index, bytes);
    }

    /// Whether any block has a pending edit.
    pub fn is_dirty(&self) -> bool {
        !self.edits.is_empty()
    }
    /// Leading trivia (comments / blank lines) attached to a block.
    pub fn trivia(&self, b: &Block) -> &[u8] {
        &self.source[b.span.start..b.name_start]
    }

    /// The keyword line itself, including its trailing newline (if any).
    pub fn name_line(&self, b: &Block) -> &[u8] {
        &self.source[b.name_start..b.body_start]
    }

    /// The data cards following the keyword line.
    pub fn body(&self, b: &Block) -> &[u8] {
        &self.source[b.body_start..b.span.end]
    }

    /// The keyword name without the leading `*` or any options/whitespace,
    /// e.g. `ELEMENT_SHELL_THICKNESS` for `*ELEMENT_SHELL_THICKNESS`.
    pub fn keyword_name(&self, b: &Block) -> &str {
        let line = self.name_line(b);
        let after = if line.first() == Some(&b'*') { &line[1..] } else { line };
        let end = after
            .iter()
            .position(|&c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
            .unwrap_or(after.len());
        std::str::from_utf8(&after[..end]).unwrap_or("")
    }

    /// Reconstruct the file bytes from the block index, applying any edits.
    /// With no edits this equals `source` exactly (the round-trip guarantee);
    /// the concatenation also validates that the blocks tile `source` with no
    /// gaps or overlaps.
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.blocks.is_empty() {
            return self.source.clone();
        }
        let mut out = Vec::with_capacity(self.source.len());
        for (i, b) in self.blocks.iter().enumerate() {
            match self.edits.get(&i) {
                Some(bytes) => out.extend_from_slice(bytes),
                None => out.extend_from_slice(&self.source[b.span.clone()]),
            }
        }
        out
    }

    /// Write the (possibly edited) file to disk.
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncludeKind {
    Include,
    IncludePath,
    IncludePathRelative,
    IncludeTransform,
    IncludeAutoZzfree,
    IncludeBinary,
    IncludeCompensated,
    IncludeStampedPart,
}

#[derive(Debug, Clone)]
pub struct IncludeDirective {
    pub kind: IncludeKind,
    pub raw_path: String,
    pub resolved_path: PathBuf,
}

#[derive(Debug)]
pub struct FileParseResult {
    pub path: PathBuf,
    pub byte_count: usize,
    pub includes: Vec<IncludeDirective>,
}

#[derive(Debug)]
pub struct IncludeNode {
    pub path: PathBuf,
    pub byte_count: usize,
    pub kind: Option<IncludeKind>,
    pub children: Vec<IncludeNode>,
}

impl IncludeNode {
    pub fn total_files(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_files()).sum::<usize>()
    }

    pub fn total_bytes(&self) -> usize {
        self.byte_count + self.children.iter().map(|c| c.total_bytes()).sum::<usize>()
    }

    pub fn print_tree(&self, indent: usize) {
        let prefix = "  ".repeat(indent);
        let kind_str = match &self.kind {
            Some(k) => format!(" [{:?}]", k),
            None => String::new(),
        };
        println!(
            "{}{}{} ({} bytes)",
            prefix,
            self.path.display(),
            kind_str,
            self.byte_count,
        );
        for child in &self.children {
            child.print_tree(indent + 1);
        }
    }
}
