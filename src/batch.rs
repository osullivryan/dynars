//! In-process batch parsing/checking across many decks that share `*INCLUDE`s.
//!
//! Model variants (load cases, run configs, parameter sweeps) almost always
//! `*INCLUDE` a large common set of files — the mesh, materials, sections. A
//! plain [`parse_deck`](crate::deck::parse_deck) re-reads, re-`mmap`s, and
//! re-block-parses every file on every call, so a gigabyte shared mesh pays its
//! full cost once *per deck*. A [`Workspace`] parses several decks against one
//! shared cache: each distinct file is read, mapped, and parsed **once**, and
//! every deck that includes it gets a cheap [`Source::Shared`] handle onto the
//! same mapping.
//!
//! The reuse is sound because a `ParsedFile` (bytes + blocks) is a pure function
//! of file *content*: the `*INCLUDE_TRANSFORM` id offset lives separately in
//! [`Deck::transforms`](crate::deck::Deck) and is applied downstream, so one
//! cached file backs many decks even when they include it at different offsets.
//! Include *resolution* is **not** cached (a file's search directories differ per
//! include chain) — only the read + block-parse is, via [`extract_includes`]
//! re-run per deck over the already-parsed blocks.
//!
//! The decks handed back are ordinary [`Deck`]s: validate and navigate them
//! exactly as usual. A [`Deck`] carrying a `Workspace`'s cache additionally
//! reuses per-file *check* work across decks — the definition-index extraction
//! and the connectivity element-row scan (see [`SharedIndex`]) — and
//! [`Workspace::validate_decks`] runs many decks' checks in parallel off it.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::deck::{Deck, assemble_deck};
use crate::file::{Block, ParsedFile, Source};
use crate::include::{Parsed, walk_includes};
use crate::model::{PhysicalDefIds, PhysicalRefIds};
use crate::parser::{extract_includes, parse_file_blocks};
use crate::validate::{Report, Rule};

/// A single-flight per-file derived cache: canonical path → a lazily built
/// `Arc<T>`, computed **at most once** even under concurrent access. Each path's
/// slot is a `OnceLock`, so when many decks that share a file are validated in
/// parallel, one thread builds that file's index and the rest await it — never a
/// duplicated rebuild.
struct FileCache<T> {
    slots: Mutex<HashMap<PathBuf, Arc<OnceLock<Arc<T>>>>>,
    /// Times `build` actually ran — i.e. distinct files whose index was computed
    /// (cache misses). A shared file reused across decks counts once, which is the
    /// whole point; exposed via [`WorkspaceStats`] as the "it ran once" signal.
    builds: AtomicUsize,
}

impl<T> FileCache<T> {
    fn new() -> Self {
        FileCache {
            slots: Mutex::new(HashMap::new()),
            builds: AtomicUsize::new(0),
        }
    }

    /// Return the cached value for `path`, building it with `build` on first
    /// request. The map lock is held only long enough to reach the path's slot;
    /// the (possibly expensive) `build` runs inside the slot's `OnceLock`, off the
    /// map lock, so other paths proceed concurrently.
    fn get_or_build(&self, path: &Path, build: impl FnOnce() -> T) -> Arc<T> {
        let slot = self
            .slots
            .lock()
            .unwrap()
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone();
        slot.get_or_init(|| {
            self.builds.fetch_add(1, Ordering::Relaxed);
            Arc::new(build())
        })
        .clone()
    }

    fn builds(&self) -> usize {
        self.builds.load(Ordering::Relaxed)
    }
}

/// One file parsed once and shared across every deck that includes it: the
/// backing bytes behind an [`Arc`] (so each deck's [`Source::Shared`] is a
/// pointer bump, never a copy) and the keyword blocks (cloned per deck — a small
/// `Vec` even for a multi-gigabyte mesh, which is only a handful of blocks).
struct CachedFile {
    source: Arc<Source>,
    blocks: Vec<Block>,
}

/// The cross-deck reuse cache a [`Workspace`] owns and shares (via [`Arc`]) into
/// every [`Deck`] it produces. All maps are keyed by *canonical* path — the same
/// key the include walker de-duplicates on — so a file reached by different
/// relative paths from different decks still hits one entry.
///
/// Everything here memoizes **content-only** work, correct to reuse verbatim
/// across decks. Deck-specific work (composing transforms, merging the per-deck
/// definition index, resolving references) is never cached.
pub struct SharedIndex {
    /// Read + block-parse cache: canonical path → shared bytes + blocks.
    files: Mutex<HashMap<PathBuf, Arc<CachedFile>>>,
    /// Physical defined-id extraction, memoized per file (single-flight). The
    /// O(bytes) scan of a shared mesh's `*NODE`/`*ELEMENT` ids runs once; each
    /// deck applies only its own `*INCLUDE_TRANSFORM` shift on top (in
    /// `build_defs`). Built lazily on first validation/navigation of a deck.
    def_ids: FileCache<PhysicalDefIds>,
    /// Physical connectivity-reference extraction, memoized per file
    /// (single-flight). The O(incidences) element-row scan a shared mesh would
    /// otherwise re-pay for `references_resolve_with_connectivity` on every deck.
    ref_ids: FileCache<PhysicalRefIds>,
    /// Files read + parsed from disk (cache misses).
    files_parsed: AtomicUsize,
    /// Times a cached file was reused instead of re-read (cache hits).
    files_reused: AtomicUsize,
}

impl SharedIndex {
    fn new() -> Self {
        SharedIndex {
            files: Mutex::new(HashMap::new()),
            def_ids: FileCache::new(),
            ref_ids: FileCache::new(),
            files_parsed: AtomicUsize::new(0),
            files_reused: AtomicUsize::new(0),
        }
    }

    /// The file's physical (pre-transform) defined ids, extracted once and reused
    /// by every deck in the workspace. Keyed by canonical path (the same key the
    /// parse cache uses); single-flight, so concurrent decks don't rebuild it.
    pub(crate) fn physical_def_ids(&self, file: &ParsedFile) -> Arc<PhysicalDefIds> {
        self.def_ids
            .get_or_build(&file.path, || crate::model::collect_def_ids(file, None))
    }

    /// The file's distinct physical connectivity references, extracted once and
    /// reused across decks — lets the connectivity check skip re-walking a shared
    /// mesh's element rows. Keyed by canonical path; single-flight.
    pub(crate) fn physical_ref_ids(&self, file: &ParsedFile) -> Arc<PhysicalRefIds> {
        self.ref_ids
            .get_or_build(&file.path, || crate::model::collect_conn_ref_ids(file))
    }

    /// Get the cached parse for `path`, reading + block-parsing it on a miss.
    /// `path` is the canonical path the include walker hands the parse closure.
    /// The file read/parse happens **outside** the lock, so a large mesh doesn't
    /// serialize other threads; a rare cross-thread double-parse of the same path
    /// is resolved by the `entry` check (one wins, the other is dropped).
    fn get_or_parse(&self, path: &Path) -> std::io::Result<Arc<CachedFile>> {
        if let Some(cached) = self.files.lock().unwrap().get(path) {
            self.files_reused.fetch_add(1, Ordering::Relaxed);
            return Ok(cached.clone());
        }

        let pf = parse_file_blocks(path)?;
        let ParsedFile { source, blocks, .. } = pf;
        let cached = Arc::new(CachedFile {
            source: Arc::new(source),
            blocks,
        });

        match self.files.lock().unwrap().entry(path.to_path_buf()) {
            // Lost the race: another thread inserted first — reuse theirs.
            Entry::Occupied(e) => {
                self.files_reused.fetch_add(1, Ordering::Relaxed);
                Ok(e.get().clone())
            }
            Entry::Vacant(e) => {
                self.files_parsed.fetch_add(1, Ordering::Relaxed);
                Ok(e.insert(cached).clone())
            }
        }
    }

    /// Build this deck's `ParsedFile` for `path`: a fresh handle sharing the
    /// cached bytes ([`Source::Shared`]) with cloned blocks, ready for the
    /// walker. Includes are re-resolved per deck (cheap — walks parsed blocks).
    fn parsed_file_for(&self, path: &Path, search: &[PathBuf]) -> Option<Parsed<ParsedFile>> {
        let cached = self.get_or_parse(path).ok()?;
        let pf = ParsedFile::from_source(
            path.to_path_buf(),
            Source::Shared(cached.source.clone()),
            cached.blocks.clone(),
        );
        let includes = extract_includes(&pf, search);
        Some(Parsed {
            includes,
            payload: pf,
        })
    }
}

/// Cache-hit accounting for a [`Workspace`], for reporting how much work sharing
/// saved. `files_parsed` counts distinct files read from disk; `files_reused`
/// counts times a shared file was served from cache instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceStats {
    pub files_parsed: usize,
    pub files_reused: usize,
    /// Distinct files whose *definition* index was extracted (a shared file counts
    /// once, however many decks used it). The check-work-sharing analogue of
    /// `files_parsed`.
    pub def_indices_built: usize,
    /// Distinct files whose *connectivity-reference* index was extracted — nonzero
    /// only once a connectivity check has run.
    pub ref_indices_built: usize,
}

/// An in-process batch context that parses (and checks) many decks against one
/// shared file cache, so common `*INCLUDE`s are read and parsed exactly once.
///
/// ```no_run
/// use dynars::batch::Workspace;
/// use std::path::Path;
///
/// let ws = Workspace::new();
/// let decks = ws.parse_decks(["variant_a/main.k", "variant_b/main.k"]);
/// for (root, deck) in &decks {
///     match deck {
///         Ok(_d) => { /* d.validate(...) — shared files aren't re-read */ }
///         Err(e) => eprintln!("{}: {e}", root.display()),
///     }
/// }
/// let s = ws.stats();
/// println!("{} files parsed, {} reuses", s.files_parsed, s.files_reused);
/// # let _ = Path::new("");
/// ```
///
/// Decks are parsed sequentially so the cache warms as it goes (deck *N* reuses
/// everything decks *1..N-1* already read); within a single deck the include
/// walk still parses files in parallel.
#[derive(Clone)]
pub struct Workspace {
    shared: Arc<SharedIndex>,
}

impl Workspace {
    /// A fresh workspace with an empty cache.
    pub fn new() -> Self {
        Workspace {
            shared: Arc::new(SharedIndex::new()),
        }
    }

    /// Parse `root` and everything it `*INCLUDE`s, reusing any file this
    /// workspace already read. Equivalent to [`parse_deck`](crate::deck::parse_deck)
    /// but shares cached files (and, for checks, the [`SharedIndex`]) with every
    /// other deck parsed by this workspace.
    pub fn parse_deck(&self, root: &Path) -> Result<Deck, String> {
        let nodes = walk_includes(root, |path, search| {
            self.shared.parsed_file_for(path, search)
        })?;
        Ok(assemble_deck(nodes, Some(self.shared.clone())))
    }

    /// Parse several decks in one batch, sharing all file work across them.
    /// Returns each root paired with its parse result, in input order; one deck
    /// failing does not abort the rest.
    pub fn parse_decks<I, P>(&self, roots: I) -> Vec<(PathBuf, Result<Deck, String>)>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        roots
            .into_iter()
            .map(|r| {
                let root = r.as_ref();
                (root.to_path_buf(), self.parse_deck(root))
            })
            .collect()
    }

    /// Validate several decks **in parallel** against the shared cache, returning
    /// one [`Report`] per deck in input order. Warms the shared definition index
    /// once up front (so decks don't race to build it), then runs `rules` over
    /// every deck concurrently. Intended for decks this workspace parsed: a shared
    /// mesh's parse, id index, and connectivity index are all reused across them.
    ///
    /// ```no_run
    /// # use dynars::Workspace;
    /// # use dynars::validate::Rule;
    /// let ws = Workspace::new();
    /// let decks: Vec<_> = ws.parse_decks(["a/main.k", "b/main.k"])
    ///     .into_iter().filter_map(|(_, d)| d.ok()).collect();
    /// let reports = ws.validate_decks(&decks, [Rule::references_resolve(), Rule::duplicate_ids()]);
    /// ```
    pub fn validate_decks(
        &self,
        decks: &[Deck],
        rules: impl IntoIterator<Item = Rule>,
    ) -> Vec<Report> {
        let rules: Vec<Rule> = rules.into_iter().collect();
        self.prime(decks);
        decks
            .par_iter()
            .map(|d| d.validate(rules.iter().cloned()))
            .collect()
    }

    /// Warm the shared **definition** index for every distinct file across
    /// `decks`, so the parallel validation that follows finds it built rather than
    /// racing to build it. Single-flight makes this dup-free; the connectivity
    /// reference index stays lazy — only decks that actually run a connectivity
    /// check pay to build it.
    fn prime(&self, decks: &[Deck]) {
        // Distinct by canonical path: a shared file is warmed once, not per deck.
        let mut distinct: HashMap<&Path, &ParsedFile> = HashMap::new();
        for d in decks {
            for f in &d.files {
                distinct.entry(f.path.as_path()).or_insert(f);
            }
        }
        let files: Vec<&ParsedFile> = distinct.into_values().collect();
        files.par_iter().for_each(|f| {
            self.shared.physical_def_ids(f);
        });
    }

    /// How much the sharing bought: files read from disk vs. served from cache,
    /// and how many distinct files had their check indices built (a shared file
    /// counts once, not once per deck).
    pub fn stats(&self) -> WorkspaceStats {
        WorkspaceStats {
            files_parsed: self.shared.files_parsed.load(Ordering::Relaxed),
            files_reused: self.shared.files_reused.load(Ordering::Relaxed),
            def_indices_built: self.shared.def_ids.builds(),
            ref_indices_built: self.shared.ref_ids.builds(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Workspace::new()
    }
}
