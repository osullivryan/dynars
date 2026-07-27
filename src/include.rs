//! `*INCLUDE` directives and the include tree.
//!
//! An LS-DYNA deck is a root file plus everything it pulls in via `*INCLUDE`
//! (and `*INCLUDE_PATH` for search directories). This module defines the
//! directive kinds and the tree of resolved files ([`build_include_tree`]).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::parser::parse_file_from_path;

/// The flavour of an `*INCLUDE` directive.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
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

/// One `*INCLUDE` directive: its kind, the path as written, the path it
/// resolves to on disk, and — for `*INCLUDE_TRANSFORM` — the id offsets it
/// applies to everything in the included file (identity for every other kind).
#[derive(Debug, Clone)]
pub struct IncludeDirective {
    pub kind: IncludeKind,
    pub raw_path: String,
    pub resolved_path: PathBuf,
    /// Id offsets applied to the included file. [`TransformOffsets::IDENTITY`]
    /// for a plain `*INCLUDE`; populated from the card for `*INCLUDE_TRANSFORM`.
    pub offsets: crate::keywords::TransformOffsets,
}

/// The result of scanning one file for its includes (feeds the tree builder).
#[derive(Debug)]
pub struct FileParseResult {
    pub path: PathBuf,
    pub byte_count: usize,
    pub includes: Vec<IncludeDirective>,
}

/// One node in the resolved include tree.
#[derive(Debug, serde::Serialize)]
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

/// One file discovered while walking the include graph, carrying whatever the
/// caller's parse step produced (`payload`).
pub struct GraphNode<T> {
    pub path: PathBuf,
    /// How this file was pulled in (`None` for the root).
    pub kind: Option<IncludeKind>,
    /// Every directive the file declares, in file order (path directives and
    /// unresolved includes included), so consumers can build a flat edge list.
    pub includes: Vec<IncludeDirective>,
    /// Node indices of the child files actually visited — existing file
    /// includes, de-duplicated — in declaration order.
    pub children: Vec<usize>,
    pub payload: T,
}

/// What a per-file parse step hands back to [`walk_includes`].
pub struct Parsed<T> {
    pub includes: Vec<IncludeDirective>,
    pub payload: T,
}

/// Walk the `*INCLUDE` graph from `root`, once per unique file, and return every
/// visited file as a flat node list — index 0 is the root, `children` are node
/// indices, nodes are in deterministic BFS order.
///
/// The single traversal both consumers share (the byte-count tree builder and
/// the block-parsing [`parse_deck`](crate::deck::parse_deck)); the caller
/// supplies only how to parse one file. The walker owns everything they used to
/// duplicate: parsing each generation in parallel, propagating a file's own
/// `*INCLUDE_PATH[_RELATIVE]` to its descendants, resolving/de-duplicating by
/// canonical path, and skipping includes whose target is missing (recorded in
/// `includes` for the caller to flag, but never traversed — so a missing file
/// can't derail the walk).
///
/// `parse` gets a file's path and the search directories inherited down its
/// include chain, and returns its directives plus a payload, or `None` if the
/// (existing) file could not be read — such a node is pruned from the result.
pub fn walk_includes<T, F>(root: &Path, parse: F) -> Result<Vec<GraphNode<T>>, String>
where
    T: Send,
    F: Fn(&Path, &[PathBuf]) -> Option<Parsed<T>> + Sync,
{
    use rayon::prelude::*;

    let root = std::fs::canonicalize(root)
        .map_err(|e| format!("Cannot resolve root path {}: {}", root.display(), e))?;

    /// A file we've decided to visit, awaiting its parse.
    struct Meta {
        path: PathBuf,
        kind: Option<IncludeKind>,
        search: Vec<PathBuf>,
    }
    let mut metas: Vec<Meta> = vec![Meta {
        path: root.clone(),
        kind: None,
        search: Vec::new(),
    }];
    // Filled in discovery order, index-aligned with `metas`.
    let mut includes_of: Vec<Vec<IncludeDirective>> = Vec::new();
    let mut children_of: Vec<Vec<usize>> = Vec::new();
    let mut payload_of: Vec<Option<T>> = Vec::new(); // None => parse failed / pruned

    let mut seen: HashSet<PathBuf> = HashSet::new();
    seen.insert(root);

    // BFS generation by generation; [cursor, metas.len()) is the current front.
    let mut cursor = 0;
    while cursor < metas.len() {
        let (start, end) = (cursor, metas.len());
        cursor = end;

        // Parse the generation in parallel; assign children sequentially after,
        // so node indices are deterministic regardless of thread timing.
        let parsed: Vec<Option<Parsed<T>>> = metas[start..end]
            .par_iter()
            .map(|m| parse(&m.path, &m.search))
            .collect();

        for (offset, p) in parsed.into_iter().enumerate() {
            let idx = start + offset;
            let Some(Parsed { includes, payload }) = p else {
                includes_of.push(Vec::new());
                children_of.push(Vec::new());
                payload_of.push(None);
                continue;
            };

            let parent_dir = metas[idx]
                .path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
            // Children inherit this file's search set extended by its own
            // path directives (same rule the file used to resolve itself).
            let mut child_search = metas[idx].search.clone();
            for inc in &includes {
                if let Some(dir) =
                    crate::parser::own_search_dir(&inc.kind, &inc.raw_path, &parent_dir)
                {
                    child_search.push(dir);
                }
            }

            let mut children = Vec::new();
            for inc in &includes {
                if crate::parser::own_search_dir(&inc.kind, &inc.raw_path, &parent_dir).is_some() {
                    continue; // a path directive, not a file to visit
                }
                if !inc.resolved_path.exists() {
                    continue; // missing include: kept in `includes`, not traversed
                }
                let canon = std::fs::canonicalize(&inc.resolved_path)
                    .unwrap_or_else(|_| inc.resolved_path.clone());
                if !seen.insert(canon.clone()) {
                    continue; // already visited via another edge
                }
                children.push(metas.len());
                metas.push(Meta {
                    path: canon,
                    kind: Some(inc.kind.clone()),
                    search: child_search.clone(),
                });
            }

            includes_of.push(includes);
            children_of.push(children);
            payload_of.push(Some(payload));
        }
    }

    // Prune parse-failed nodes and renumber to a dense, BFS-ordered index space.
    let mut remap = vec![usize::MAX; metas.len()];
    let mut next = 0;
    for (i, p) in payload_of.iter().enumerate() {
        if p.is_some() {
            remap[i] = next;
            next += 1;
        }
    }

    let mut out = Vec::with_capacity(next);
    for i in 0..metas.len() {
        let Some(payload) = payload_of[i].take() else {
            continue;
        };
        let children = children_of[i]
            .iter()
            .filter(|&&c| remap[c] != usize::MAX)
            .map(|&c| remap[c])
            .collect();
        out.push(GraphNode {
            path: std::mem::take(&mut metas[i].path),
            kind: metas[i].kind.take(),
            includes: std::mem::take(&mut includes_of[i]),
            children,
            payload,
        });
    }
    Ok(out)
}

/// Build the resolved include tree: the file graph as `IncludeNode`s carrying
/// byte counts, via the streaming scanner ([`parse_file_from_path`]).
pub fn build_include_tree(root_path: &Path) -> Result<IncludeNode, String> {
    let nodes = walk_includes(root_path, |path, search| {
        let result = parse_file_from_path(path, search);
        Some(Parsed {
            includes: result.includes,
            payload: result.byte_count,
        })
    })?;
    Ok(build_node(0, &nodes))
}

/// Reassemble the nested [`IncludeNode`] tree from the flat walker output.
fn build_node(i: usize, nodes: &[GraphNode<usize>]) -> IncludeNode {
    let n = &nodes[i];
    IncludeNode {
        path: n.path.clone(),
        byte_count: n.payload,
        kind: n.kind.clone(),
        children: n.children.iter().map(|&c| build_node(c, nodes)).collect(),
    }
}
