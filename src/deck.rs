//! Parse a whole deck — a root keyword file plus everything it `*INCLUDE`s —
//! in a single pass, **retaining the parsed blocks**.
//!
//! Core already parses individual files fast ([`parse_file_blocks`]) and can
//! build an include *tree* ([`build_include_tree`](crate::include_tree)), but
//! that tree keeps only paths and byte counts — it discards the parsed blocks,
//! forcing anyone who wants the actual keywords to walk and parse a second
//! time. [`parse_deck`] walks once and hands back every [`ParsedFile`], so
//! downstream consumers (validation, result ingest, …) never re-parse.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::keyword::{IncludeDirective, IncludeKind, ParsedFile};
use crate::parser::{extract_includes, parse_file_blocks};

/// A fully parsed deck: every file (root + includes) with its blocks intact,
/// plus every `*INCLUDE` directive encountered.
pub struct Deck {
    pub files: Vec<ParsedFile>,
    /// `(including-file index, directive)` for every `*INCLUDE`.
    pub includes: Vec<(usize, IncludeDirective)>,
}

impl Deck {
    /// Total source bytes across all files.
    pub fn total_bytes(&self) -> usize {
        self.files.iter().map(|f| f.src().len()).sum()
    }
}

/// Parse `root` and every file it includes, exactly once each.
///
/// A level-by-level BFS parses each frontier in parallel (via [`parse_file_blocks`]),
/// extracts includes straight from the parsed blocks (no re-read), accumulates
/// `*INCLUDE_PATH` search directories for deeper levels, and de-dupes shared
/// includes by canonical path.
pub fn parse_deck(root: &Path) -> Result<Deck, String> {
    let root = std::fs::canonicalize(root).map_err(|e| e.to_string())?;

    let mut files: Vec<ParsedFile> = Vec::new();
    let mut includes: Vec<(usize, IncludeDirective)> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut search_paths: Vec<PathBuf> = Vec::new();
    seen.insert(root.clone());
    let mut frontier = vec![root];

    while !frontier.is_empty() {
        let sp = search_paths.clone();
        let parsed: Vec<(ParsedFile, Vec<IncludeDirective>)> = frontier
            .par_iter()
            .filter_map(|p| {
                let pf = parse_file_blocks(p).ok()?;
                let incs = extract_includes(&pf, &sp);
                Some((pf, incs))
            })
            .collect();

        let mut next = Vec::new();
        for (pf, incs) in parsed {
            let fi = files.len();
            for inc in incs {
                match &inc.kind {
                    // Search-path directives widen resolution for deeper levels.
                    IncludeKind::IncludePath => search_paths.push(inc.resolved_path.clone()),
                    IncludeKind::IncludePathRelative => {
                        if let Some(parent) = pf.path.parent() {
                            search_paths.push(parent.join(&inc.raw_path));
                        }
                    }
                    // A real file include: queue it if it exists and is new.
                    _ => {
                        let resolved = inc.resolved_path.clone();
                        if resolved.exists() {
                            let canon = std::fs::canonicalize(&resolved).unwrap_or(resolved);
                            if seen.insert(canon.clone()) {
                                next.push(canon);
                            }
                        }
                    }
                }
                includes.push((fi, inc));
            }
            files.push(pf);
        }
        frontier = next;
    }

    Ok(Deck { files, includes })
}
