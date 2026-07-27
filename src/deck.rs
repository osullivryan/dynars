//! Parse a whole deck — a root keyword file plus everything it `*INCLUDE`s —
//! in a single pass, **retaining the parsed blocks**.
//!
//! Core already parses individual files fast ([`parse_file_blocks`]) and can
//! build an include *tree* ([`build_include_tree`](crate::include::build_include_tree)), but
//! that tree keeps only paths and byte counts — it discards the parsed blocks,
//! forcing anyone who wants the actual keywords to walk and parse a second
//! time. [`parse_deck`] walks once and hands back every [`ParsedFile`], so
//! downstream consumers (validation, result ingest, …) never re-parse.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use crate::file::ParsedFile;
use crate::include::{IncludeDirective, Parsed, walk_includes};
use crate::parser::{extract_includes, parse_file_blocks};

/// A fully parsed deck: every file (root + includes) with its blocks intact,
/// plus every `*INCLUDE` directive encountered.
///
/// A `Deck` is the single handle a caller holds: parse once with [`parse_deck`],
/// then validate ([`Deck::validate`]) and navigate ([`Deck::part`], …) off the same
/// object. The cross-keyword resolution indices (defined-id sets, entity site
/// map) are built **lazily on first use** and cached — a plain parse pays for
/// neither, validation builds only the id sets, navigation builds only the site
/// map. `OnceLock` (not `OnceCell`) because the Python bindings touch a `Deck`
/// with the GIL released.
pub struct Deck {
    pub files: Vec<ParsedFile>,
    /// `(including-file index, directive)` for every `*INCLUDE`.
    pub includes: Vec<(usize, IncludeDirective)>,
    /// Effective `*INCLUDE_TRANSFORM` offsets per file (parallel to `files`),
    /// composed down the include chain by the walker. Identity for the root,
    /// plain includes, and transform-free decks. The same physical file
    /// instanced at two different offsets appears as two entries here, each with
    /// its own offsets — the source of truth [`file_transforms`] collapses.
    ///
    /// [`file_transforms`]: Deck::file_transforms
    pub(crate) transforms: Vec<crate::keywords::TransformOffsets>,
    /// Defined ids per entity kind — the resolution core (validation + `is_defined`).
    pub(crate) defs: OnceLock<crate::model::Defs>,
    /// `transforms` with identity collapsed to `None`, so the resolution core's
    /// hot paths skip the offset logic on transform-free files. Cached on first use.
    pub(crate) file_transforms: OnceLock<Vec<Option<crate::keywords::TransformOffsets>>>,
    /// `(kind, id) -> defining block` for navigable definition entities.
    pub(crate) sites: OnceLock<crate::model::Sites>,
    /// User schemas for keywords the built-in library doesn't cover, keyed by
    /// canonical base. Consulted **first** when the navigation spine resolves a
    /// keyword's field layout — the escape hatch for rare / vendor / newer-than-
    /// snapshot keywords. Empty for a plain parse. See [`Deck::register_schema`].
    pub(crate) user_schemas: HashMap<String, crate::schema::Schema>,
}

impl Deck {
    /// Total source bytes across all files.
    pub fn total_bytes(&self) -> usize {
        self.files.iter().map(|f| f.src().len()).sum()
    }

    /// Register a user schema for a keyword dynars ships no layout for, so
    /// `deck.keywords("FOO").card(0).field("bar")` gets named, typed field
    /// access — the same runtime [`Schema`](crate::schema::Schema) the columnar
    /// path and `#[derive(Keyword)]` produce. Consulted ahead of the built-in
    /// table; keyed by canonical base, so registering the same base twice
    /// replaces it. (Layout only — user schemas don't participate in
    /// entity-definition/reference resolution, which stays on the built-in
    /// table.)
    pub fn register_schema(&mut self, schema: crate::schema::Schema) {
        let base = crate::keywords::canonical_base(&schema.keyword);
        self.user_schemas.insert(base, schema);
    }
}

/// Parse `root` and every file it includes.
///
/// Traversal, `*INCLUDE_PATH` propagation, and de-duplication all live in the
/// shared [`walk_includes`]; this only says how to parse one file (block index +
/// [`extract_includes`], which reads `*INCLUDE_TRANSFORM` offsets) and how to lay
/// the walker's node list out as a [`Deck`]. Node order is the walker's
/// deterministic BFS, and node index *is* the file index — so the
/// `(file, directive)` pairs in `includes` reference `files` directly, and
/// `transforms[fi]` is `files[fi]`'s effective offsets. A file included twice at
/// different offsets appears as two nodes (see [`walk_includes`]).
pub fn parse_deck(root: &Path) -> Result<Deck, String> {
    let nodes = walk_includes(root, |path, search| {
        let pf = parse_file_blocks(path).ok()?;
        let includes = extract_includes(&pf, search);
        Some(Parsed {
            includes,
            payload: pf,
        })
    })?;

    let mut files: Vec<ParsedFile> = Vec::with_capacity(nodes.len());
    let mut includes: Vec<(usize, IncludeDirective)> = Vec::new();
    let mut transforms: Vec<crate::keywords::TransformOffsets> = Vec::with_capacity(nodes.len());
    for (fi, node) in nodes.into_iter().enumerate() {
        for inc in node.includes {
            includes.push((fi, inc));
        }
        transforms.push(node.transform);
        files.push(node.payload);
    }

    Ok(Deck {
        files,
        includes,
        transforms,
        defs: OnceLock::new(),
        file_transforms: OnceLock::new(),
        sites: OnceLock::new(),
        user_schemas: HashMap::new(),
    })
}
