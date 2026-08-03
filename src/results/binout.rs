use rayon::prelude::*;

use super::LsdaError;
use super::diskfile::Diskfile;
use super::lsda::{build_read_tree, open_read_family};
use super::symbol::{ReadResult, SymNode};

/// LS-DYNA binary output (binout) file reader.
///
/// The file family is memory-mapped and the symbol table is frozen into an
/// immutable, lock-free tree, so reads (including concurrent [`read_many`](Self::read_many)) do
/// no syscalls and take no locks.
///
/// Usage:
/// ```no_run
/// use dynars::results::Binout;
/// let b = Binout::new("path/to/binout*").unwrap();
/// let keys = b.read(&[]).unwrap().keys();            // top-level datasets
/// let data = b.read(&["nodout", "1", "x_displacement"]).unwrap().to_f64_vec();
/// ```
pub struct Binout {
    pub filelist: Vec<String>,
    files: Vec<Diskfile>,
    tree: SymNode,
}

/// A per-state variable aggregated across every time-state directory, as a dense
/// `n_steps × n_channels` row-major matrix plus the per-state `time`. Produced by
/// [`Binout::read_states`] — the clean equivalent of stitching `d000001`, `d000002`,
/// … together by hand.
#[derive(Debug, Clone)]
pub struct StateMatrix {
    /// One time value per state (length `n_steps`).
    pub time: Vec<f64>,
    /// Row-major `n_steps × n_channels`: state `i`, channel `j` lives at `i*n_channels + j`.
    pub values: Vec<f64>,
    /// LS-DYNA entity ID for each column (length `n_channels`), from
    /// `<branch>/metadata/ids` — e.g. the node ID of each `nodout` column. Empty
    /// if the branch has no id metadata.
    pub ids: Vec<i64>,
    pub n_steps: usize,
    pub n_channels: usize,
}

/// A few entity columns of a per-state variable — `time` plus a row-major
/// `n_steps × n_cols` block — produced by [`Binout::read_columns`] **without**
/// building the full `[n_steps, n_channels]` [`StateMatrix`]. The one-column case
/// (`n_cols == 1`) is exactly one entity's contiguous time-history.
#[derive(Debug, Clone)]
pub struct StateColumns {
    /// One time value per state (length `n_steps`).
    pub time: Vec<f64>,
    /// Row-major `n_steps × n_cols`: state `i`, requested column `c` at `i*n_cols + c`.
    pub values: Vec<f64>,
    pub n_steps: usize,
    pub n_cols: usize,
}

impl StateMatrix {
    /// All channels at one state (a row of the matrix).
    pub fn row(&self, step: usize) -> &[f64] {
        &self.values[step * self.n_channels..(step + 1) * self.n_channels]
    }

    /// One channel's time history across all states (a column; strided copy).
    pub fn column(&self, channel: usize) -> Vec<f64> {
        (0..self.n_steps)
            .map(|i| self.values[i * self.n_channels + channel])
            .collect()
    }

    /// Column index of a given LS-DYNA entity ID (e.g. a node ID), if present.
    pub fn index_of(&self, id: i64) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    /// One entity's time history looked up by its LS-DYNA ID (e.g. node 8151's
    /// head acceleration), rather than by column index.
    pub fn column_by_id(&self, id: i64) -> Option<Vec<f64>> {
        self.index_of(id).map(|j| self.column(j))
    }
}

impl Binout {
    pub fn new(glob_pattern: &str) -> Result<Self, LsdaError> {
        let matches =
            glob::glob(glob_pattern).map_err(|e| LsdaError::InvalidPath(e.to_string()))?;
        let mut filelist: Vec<String> = matches
            .filter_map(|e| e.ok())
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        if filelist.is_empty() {
            return Err(LsdaError::FileNotFound);
        }
        filelist.sort();
        // Read-mode open: memory-map the family, then parse the symbol table
        // straight into the lock-free tree — no intermediate mutable/locked tree,
        // no second freeze pass.
        let mut files = open_read_family(&filelist)?;
        let tree = build_read_tree(&mut files)?;
        Ok(Self {
            filelist,
            files,
            tree,
        })
    }

    /// Read at the given path segments. Returns `ReadResult::Directory` if the path is a folder.
    pub fn read(&self, path: &[&str]) -> Result<ReadResult, LsdaError> {
        self.resolve(path)?.read(&self.files)
    }

    /// Read many paths concurrently (lock-free), returning one result per path.
    pub fn read_many(&self, paths: &[Vec<&str>]) -> Vec<Result<ReadResult, LsdaError>> {
        paths.par_iter().map(|p| self.read(p)).collect()
    }

    /// Convenience: read and convert to f64. Useful for extracting scalar/vector results.
    pub fn read_f64(&self, path: &[&str]) -> Result<Vec<f64>, LsdaError> {
        Ok(self.read(path)?.to_f64_vec())
    }

    /// Read a time-history: extracts the channel and its sibling "time" array.
    pub fn read_time_series(&self, path: &[&str]) -> Result<super::TimeSeries, LsdaError> {
        if path.is_empty() {
            return Err(LsdaError::InvalidPath("empty path".into()));
        }
        let values = self.read_f64(path)?;
        let channel = path.last().unwrap().to_string();

        // Try to read "time" from the parent directory
        let mut time_path = path[..path.len() - 1].to_vec();
        time_path.push("time");
        let time = self.read_f64(&time_path).unwrap_or_else(|_| {
            // Synthesize integer time steps if no "time" array present
            (0..values.len()).map(|i| i as f64).collect()
        });

        Ok(super::TimeSeries {
            time,
            values,
            channel,
        })
    }

    /// Aggregate a per-state variable across every time-state directory
    /// (`d000001`, `d000002`, …) into a dense `StateMatrix` — the clean, fast
    /// equivalent of stitching the state dirs together by hand.
    /// `read_states("nodout", "x_acceleration")` returns an `n_steps × n_nodes`
    /// matrix plus `time`; take `.column(i)` for one node's history. State reads
    /// run concurrently and lock-free ([`read_many`](Self::read_many)).
    /// The sorted `dNNNNNN` state-directory names under `branch` (one per output
    /// state). Shared by [`read_states`](Self::read_states) and
    /// [`read_columns`](Self::read_columns).
    fn state_dirs(&self, branch: &str) -> Result<Vec<String>, LsdaError> {
        let mut states: Vec<String> = self
            .read(&[branch])?
            .keys()
            .into_iter()
            .filter(|k| {
                k.len() > 1 && k.starts_with('d') && k[1..].bytes().all(|b| b.is_ascii_digit())
            })
            .collect();
        states.sort();
        if states.is_empty() {
            return Err(LsdaError::SymbolNotFound(format!(
                "no state directories under {branch}"
            )));
        }
        Ok(states)
    }

    /// One time value per state (from each `<branch>/<state>/time`).
    fn state_times(&self, branch: &str, states: &[String]) -> Vec<f64> {
        let time_paths: Vec<Vec<&str>> = states
            .iter()
            .map(|s| vec![branch, s.as_str(), "time"])
            .collect();
        self.read_many(&time_paths)
            .into_iter()
            .map(|r| {
                r.ok()
                    .map(|x| x.to_f64_vec())
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0.0)
            })
            .collect()
    }

    pub fn read_states(&self, branch: &str, var: &str) -> Result<StateMatrix, LsdaError> {
        let states = self.state_dirs(branch)?;
        let n_steps = states.len();
        let var_paths: Vec<Vec<&str>> =
            states.iter().map(|s| vec![branch, s.as_str(), var]).collect();
        let var_res = self.read_many(&var_paths);

        let mut values: Vec<f64> = Vec::new();
        let mut n_channels = 0;
        for (i, r) in var_res.into_iter().enumerate() {
            let row = r?.to_f64_vec();
            if i == 0 {
                n_channels = row.len();
                values.reserve(n_steps * n_channels);
            }
            values.extend_from_slice(&row);
        }
        let time = self.state_times(branch, &states);
        let ids: Vec<i64> = self
            .read(&[branch, "metadata", "ids"])
            .map(|r| r.to_f64_vec().iter().map(|&x| x as i64).collect())
            .unwrap_or_default();
        Ok(StateMatrix {
            time,
            values,
            ids,
            n_steps,
            n_channels,
        })
    }

    /// One entity's time-history (or a few), decoded **only** for the requested
    /// column indices — without building the full `[n_steps, n_channels]`
    /// [`StateMatrix`]. For each state it decodes just those elements straight
    /// from the memory map, so a single node's history out of a wide `nodout`
    /// costs `O(n_steps)` memory and touches ~one page per state instead of the
    /// whole record. `cols` are column indices (see [`StateMatrix::index_of`] /
    /// [`Binout::ids`] to map an LS-DYNA id to its column).
    pub fn read_columns(
        &self,
        branch: &str,
        var: &str,
        cols: &[usize],
    ) -> Result<StateColumns, LsdaError> {
        let states = self.state_dirs(branch)?;
        let n_steps = states.len();
        // Decode only `cols` from each state's record, in parallel, in state order.
        let rows: Vec<Result<Vec<f64>, LsdaError>> = states
            .par_iter()
            .map(|s| {
                let node = self.resolve(&[branch, s.as_str(), var])?;
                let mut row = Vec::with_capacity(cols.len());
                node.read_cols_f64(&self.files, cols, &mut row)?;
                Ok(row)
            })
            .collect();
        let mut values = Vec::with_capacity(n_steps * cols.len());
        for r in rows {
            values.extend_from_slice(&r?);
        }
        let time = self.state_times(branch, &states);
        Ok(StateColumns {
            time,
            values,
            n_steps,
            n_cols: cols.len(),
        })
    }

    /// LS-DYNA entity IDs for a state branch (e.g. the node IDs of `nodout`), from
    /// `<branch>/metadata/ids`.
    pub fn ids(&self, branch: &str) -> Result<Vec<i64>, LsdaError> {
        Ok(self
            .read(&[branch, "metadata", "ids"])?
            .to_f64_vec()
            .iter()
            .map(|&x| x as i64)
            .collect())
    }

    /// Per-entity legend/name strings from `<branch>/metadata/legend` (a packed
    /// fixed-width char block, one name per entity), trimmed. Unnamed entities
    /// come back as empty strings.
    pub fn legend(&self, branch: &str) -> Result<Vec<String>, LsdaError> {
        let n = self
            .read(&[branch, "metadata", "ids"])
            .map(|r| r.to_f64_vec().len())
            .unwrap_or(0);
        let bytes: Vec<u8> = self
            .read(&[branch, "metadata", "legend"])?
            .to_f64_vec()
            .iter()
            .map(|&x| x as u8)
            .collect();
        if n == 0 || bytes.is_empty() {
            return Ok(Vec::new());
        }
        let width = bytes.len() / n;
        Ok(bytes
            .chunks(width.max(1))
            .map(|c| String::from_utf8_lossy(c).trim().to_string())
            .collect())
    }

    /// The dataset title from `<branch>/metadata/title`, trimmed.
    pub fn title(&self, branch: &str) -> Result<String, LsdaError> {
        let bytes: Vec<u8> = self
            .read(&[branch, "metadata", "title"])?
            .to_f64_vec()
            .iter()
            .map(|&x| x as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    }

    /// List variable names (channels) at a directory path in the binout hierarchy.
    pub fn channels(&self, dir_path: &[&str]) -> Result<Vec<String>, LsdaError> {
        let result = self.read(dir_path)?;
        Ok(result.keys())
    }

    fn resolve(&self, path: &[&str]) -> Result<&SymNode, LsdaError> {
        let mut current = &self.tree;
        for part in path {
            current = current
                .child(part.as_bytes())
                .ok_or_else(|| LsdaError::SymbolNotFound(part.to_string()))?;
        }
        Ok(current)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests run against a real binout file if one is present at the
    // path below; they skip cleanly (no failure) on CI / clean checkouts.
    const TEST_BINOUT: &str = "/Users/ryanosullivan/RustroverProjects/lassoBinout/src/binout";

    #[test]
    fn real_binout_top_level() {
        if !std::path::Path::new(TEST_BINOUT).exists() {
            println!("Skipping: test binout not found at {TEST_BINOUT}");
            return;
        }
        let b = Binout::new(TEST_BINOUT).expect("open real binout");
        let top = b.read(&[]).expect("read root").keys();
        assert!(!top.is_empty(), "binout should have top-level channels");
        println!("Top-level channels ({}):", top.len());
        for k in &top {
            println!("  {k}");
        }
    }

    #[test]
    fn real_binout_glstat_channels() {
        if !std::path::Path::new(TEST_BINOUT).exists() {
            return;
        }
        let b = Binout::new(TEST_BINOUT).expect("open binout");
        let channels = b.read(&["glstat"]).expect("read glstat").keys();
        println!("glstat sub-dirs: {} entries", channels.len());
        // Drill into the first sub-dir to see what variables it has.
        if let Some(first) = channels.first() {
            let inner = b
                .read(&["glstat", first])
                .expect("read glstat/first")
                .keys();
            println!("glstat/{first} channels: {inner:?}");
        }
    }

    #[test]
    fn real_binout_nodout_channels() {
        if !std::path::Path::new(TEST_BINOUT).exists() {
            return;
        }
        let b = Binout::new(TEST_BINOUT).expect("open binout");
        let top = b.read(&[]).expect("read root").keys();
        if !top.contains(&"nodout".to_string()) {
            println!("No nodout channel — skipping");
            return;
        }
        let nodes = b.read(&["nodout"]).expect("read nodout").keys();
        println!(
            "nodout node IDs (first 5): {:?}",
            &nodes[..nodes.len().min(5)]
        );
        assert!(!nodes.is_empty());
        // Drill into the first node and print its channels.
        if let Some(first_node) = nodes.first() {
            let channels = b.read(&["nodout", first_node]).expect("read node").keys();
            println!("  channels for node {first_node}: {channels:?}");
            assert!(!channels.is_empty());
            // Read the first channel as f64.
            if let Some(ch) = channels.first() {
                let vals = b
                    .read_f64(&["nodout", first_node, ch])
                    .expect("read channel");
                println!(
                    "  {first_node}/{ch}: {} values, first={:?}",
                    vals.len(),
                    vals.first()
                );
                assert!(!vals.is_empty());
            }
        }
    }

    #[test]
    fn read_states_aggregates_a_nodout_variable() {
        if !std::path::Path::new(TEST_BINOUT).exists() {
            return;
        }
        let b = Binout::new(TEST_BINOUT).expect("open binout");
        if !b.read(&[]).unwrap().keys().contains(&"nodout".to_string()) {
            return;
        }
        let m = b.read_states("nodout", "x_acceleration").expect("read_states");
        assert!(m.n_steps > 0 && m.n_channels > 0);
        assert_eq!(m.values.len(), m.n_steps * m.n_channels);
        assert_eq!(m.time.len(), m.n_steps);
        // Row 0 must equal a direct read of the earliest state dir.
        let first = b
            .read(&["nodout"])
            .unwrap()
            .keys()
            .into_iter()
            .filter(|k| k.starts_with('d') && k[1..].bytes().all(|c| c.is_ascii_digit()))
            .min()
            .unwrap();
        let direct = b.read_f64(&["nodout", &first, "x_acceleration"]).unwrap();
        assert_eq!(m.row(0), direct.as_slice());
        assert_eq!(m.column(0).len(), m.n_steps);
        // IDs align with columns; lookup-by-id resolves to the same column.
        assert_eq!(m.ids.len(), m.n_channels);
        if let Some(&id0) = m.ids.first() {
            assert_eq!(m.index_of(id0), Some(0));
            assert_eq!(m.column_by_id(id0), Some(m.column(0)));
        }
        // legend / title accessors work.
        assert_eq!(b.legend("nodout").unwrap().len(), m.n_channels);
        assert!(!b.title("nodout").unwrap().is_empty());
    }
}
