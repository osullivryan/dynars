//! LS-DYNA **d3plot** reader — control block, geometry, and per-state results
//! across a memory-mapped file family.
//!
//! Handles: single/double precision; the multi-file family (`d3plot01`, …);
//! block-aligned geometry with NARBS numbering + material section; and generic
//! per-state result extraction (node displacement/velocity/acceleration and
//! solid/thick-shell/beam/shell blocks) via [`D3plot::block_data`]. State size
//! accounts for global vars, node thermal/mass-scaling, element results, and
//! element/node deletion. SPH/airbag/rigid-road state terms and FEMZIP are not
//! modelled. Word offsets follow the LS-DYNA Database Manual; validated against
//! open-lasso-python on real decks (node data bit-exact).

use memmap2::Mmap;

#[derive(Debug, thiserror::Error)]
pub enum D3plotError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a recognizable d3plot (could not determine word size)")]
    BadHeader,
    #[error("{0}")]
    Unsupported(String),
}

/// Parsed control block (the values we use).
#[derive(Debug, Clone)]
pub struct Control {
    pub wordsize: u64, // 4 (single) or 8 (double)
    pub ndim: usize,
    pub numnp: usize,
    pub nglbv: usize,
    pub it: i64,
    pub iu: i64,
    pub iv: i64,
    pub ia: i64,
    pub nel8: usize,
    pub nv3d: usize,
    pub nelth: usize,
    pub nv3dt: usize,
    pub nel2: usize,
    pub nv1d: usize,
    pub nel4: usize,
    pub nv2d: usize,
    pub narbs: usize,
    pub maxint: i64,  // shell integration layers; sign encodes element/node deletion (mdlopt)
    pub nmmat: usize, // total number of materials/parts
    pub extra: usize, // extra header words beyond the base 64 (word 57)
}

impl Control {
    /// `mattyp` (per-part material-type section present) is encoded in the NDIM
    /// flag word: NDIM 5 or 7 ⇒ mattyp=1.
    fn mattyp(&self) -> bool {
        self.ndim == 5 || self.ndim == 7
    }

    /// Element-deletion encoding from `maxint` (LS-DYNA mdlopt):
    /// `maxint >= 0` none; `-(n)` node deletion (mdlopt 1); `-(n+10000)` element
    /// deletion (mdlopt 2).
    fn mdlopt(&self) -> i64 {
        if self.maxint >= 0 {
            0
        } else if -self.maxint >= 10000 {
            2
        } else {
            1
        }
    }

    /// Byte offset of the first state database (in the concatenated file family):
    /// all geometry sections, rounded up to the next 512-word block, the way
    /// LS-DYNA zero-pads its files. Node coordinates are always 3-D; the NDIM
    /// header word is a flag set — see `read_control`.
    fn geometry_section_bytes(&self) -> u64 {
        let mut words = 64 + self.extra;
        words += self.numnp * 3; // node coordinates
        words += self.nel8 * 9; // solids (8 nodes + material)
        words += self.nelth * 9; // thick shells
        words += self.nel2 * 6; // beams
        words += self.nel4 * 5; // shells (4 nodes + material)
        if self.mattyp() {
            words += 2 + self.nmmat; // material-type section
        }
        words += self.narbs; // arbitrary node/element numbering section
        // Part & contact-interface titles and any trailing sections fall inside
        // the final partial block, which the 512-word rounding absorbs.
        let block = 512usize;
        let blocks = words.div_ceil(block).max(1);
        (blocks * block) as u64 * self.wordsize
    }

    /// Thermal / mass-scaling variables per node (the IT block), decoded from
    /// the `it` header word: `it % 10` temperature variants, `+1` for mass
    /// scaling when `it >= 10`. Common cases (0/1/10) are exact; exotic thermal
    /// layouts may differ.
    fn node_therm_vars(&self) -> usize {
        let temp = (self.it.rem_euclid(10)).max(0) as usize;
        let mass = if self.it >= 10 { 1 } else { 0 };
        temp + mass
    }

    /// Total words of node data per state: (IU + IV + IA) × 3 + thermal, × NUMNP.
    fn node_data_words(&self) -> usize {
        let vec3 = (self.iu != 0) as usize + (self.iv != 0) as usize + (self.ia != 0) as usize;
        (vec3 * 3 + self.node_therm_vars()) * self.numnp
    }

    /// Deletion flags per state (mdlopt): node- or element-count words.
    fn deletion_words(&self) -> usize {
        match self.mdlopt() {
            1 => self.numnp,
            2 => self.nel8 + self.nel4 + self.nel2 + self.nelth,
            _ => 0,
        }
    }

    /// Element result words per state, summed over all element blocks.
    fn element_words(&self) -> usize {
        self.nel8 * self.nv3d + self.nelth * self.nv3dt + self.nel2 * self.nv1d + self.nel4 * self.nv2d
    }

    /// Bytes per state: time + global vars + node data + element data + deletion.
    /// SPH/airbag/rigid-road terms are not modelled (v1 scope).
    fn bytes_per_state(&self) -> u64 {
        (1 + self.nglbv + self.node_data_words() + self.element_words() + self.deletion_words()) as u64
            * self.wordsize
    }

    /// `(word offset within a state, entity count, vars per entity)` for a result
    /// block, or `None` when that block is absent. This one table is what makes
    /// extraction generic — every block is just a strided slice.
    fn block_spec(&self, block: StateBlock) -> Option<(usize, usize, usize)> {
        let base = 1 + self.nglbv; // after time + globals
        let n3 = self.numnp * 3;
        // Node array order: displacement, then the thermal/mass-scaling block,
        // then velocity, then acceleration (matches LS-DYNA / lasso).
        let therm = self.node_therm_vars() * self.numnp;
        let disp = base;
        let vel = disp + if self.iu != 0 { n3 } else { 0 } + therm;
        let acc = vel + if self.iv != 0 { n3 } else { 0 };
        let elem = base + self.node_data_words();
        let solid = elem;
        let tshell = solid + self.nel8 * self.nv3d;
        let beam = tshell + self.nelth * self.nv3dt;
        let shell = beam + self.nel2 * self.nv1d;
        let some = |cond: bool, off: usize, count: usize, vars: usize| {
            (cond && count > 0 && vars > 0).then_some((off, count, vars))
        };
        match block {
            StateBlock::Displacement => some(self.iu != 0, disp, self.numnp, 3),
            StateBlock::Velocity => some(self.iv != 0, vel, self.numnp, 3),
            StateBlock::Acceleration => some(self.ia != 0, acc, self.numnp, 3),
            StateBlock::Solid => some(true, solid, self.nel8, self.nv3d),
            StateBlock::ThickShell => some(true, tshell, self.nelth, self.nv3dt),
            StateBlock::Beam => some(true, beam, self.nel2, self.nv1d),
            StateBlock::Shell => some(true, shell, self.nel4, self.nv2d),
        }
    }

    /// Byte offset, within a state, of the IU (deformed-coordinate) block.
    fn iu_offset_in_state(&self) -> u64 {
        (1 + self.nglbv) as u64 * self.wordsize
    }
}

/// A per-entity result block in a state. Node blocks are (N, 3); element blocks
/// are (N, vars) where `vars` is the solver's packed per-element layout
/// (stresses, plastic strain, history variables, per integration point/layer) —
/// returned raw for the caller to reshape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateBlock {
    Displacement,
    Velocity,
    Acceleration,
    Solid,
    ThickShell,
    Beam,
    Shell,
}

/// Where one state lives: which family file, and the byte offset of its start.
struct StateLoc {
    file: usize,
    offset: u64,
}

/// A read-only d3plot reader over a file family (base + `d3plot01`, `d3plot02`, …).
/// Files are memory-mapped; each state is located by file + offset so trailing
/// block padding between family files never corrupts the state stride.
pub struct D3plot {
    ctrl: Control,
    files: Vec<Mmap>,
    states: Vec<StateLoc>,
    /// Initial node coordinates (NUMNP × 3), row-major.
    x0: Vec<f64>,
    times: Vec<f64>,
}

/// Memory-map a file read-only.
fn mmap(path: &std::path::Path) -> Result<Mmap, D3plotError> {
    let file = std::fs::File::open(path)?;
    // SAFETY: opened read-only; not mutated elsewhere while mapped.
    Ok(unsafe { Mmap::map(&file)? })
}

impl D3plot {
    pub fn control(&self) -> &Control { &self.ctrl }
    pub fn num_nodes(&self) -> usize { self.ctrl.numnp }
    pub fn num_states(&self) -> usize { self.states.len() }
    /// Simulation time of each state.
    pub fn times(&self) -> &[f64] { &self.times }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, D3plotError> {
        // Memory-map the base file plus any continuation files (d3plot01, …).
        let base = path.as_ref();
        let stem = base.to_string_lossy().into_owned();
        let mut files: Vec<Mmap> = vec![mmap(base)?];
        for i in 1..1000 {
            let name = if i < 100 { format!("{stem}{i:02}") } else { format!("{stem}{i}") };
            if !std::path::Path::new(&name).exists() {
                break;
            }
            files.push(mmap(std::path::Path::new(&name))?);
        }

        let ctrl = read_control_bytes(&files[0])?;
        if ctrl.iu == 0 {
            return Err(D3plotError::Unsupported(
                "d3plot has no nodal displacement data (IU=0)".into(),
            ));
        }
        let ws = ctrl.wordsize;

        // Initial node coordinates: right after the header in the base file.
        let coord_off = (64 + ctrl.extra) as u64 * ws;
        let x0 = read_floats_at(&files[0], coord_off, ctrl.numnp * 3, ws)?;

        // Walk states across the family. In the base file they start after the
        // (block-padded) geometry; in continuation files, at offset 0. A state
        // whose first word is the -999999 EOF marker terminates the sequence.
        let geom = ctrl.geometry_section_bytes();
        let bps = ctrl.bytes_per_state();
        let mut states = Vec::new();
        let mut times = Vec::new();
        if bps > 0 {
            'outer: for (fi, bytes) in files.iter().enumerate() {
                let start = if fi == 0 { geom } else { 0 };
                let len = bytes.len() as u64;
                let mut off = start;
                while off + bps <= len {
                    let t = read_float_at(bytes, off, ws);
                    if is_eof_marker(t) {
                        break 'outer;
                    }
                    times.push(t);
                    states.push(StateLoc { file: fi, offset: off });
                    off += bps;
                }
            }
        }

        Ok(Self { ctrl, files, states, x0, times })
    }

    /// Deformed (current) node coordinates at `state` (0-based), NUMNP × 3 row-major.
    pub fn node_coordinates(&self, state: usize) -> Result<Vec<f64>, D3plotError> {
        let loc = self
            .states
            .get(state)
            .ok_or_else(|| D3plotError::Unsupported(format!("state {state} out of range ({})", self.states.len())))?;
        let ws = self.ctrl.wordsize;
        let off = loc.offset + self.ctrl.iu_offset_in_state();
        read_floats_at(&self.files[loc.file], off, self.ctrl.numnp * 3, ws)
    }

    /// Per-node displacement magnitude at `state`: |current − initial|.
    pub fn displacement_magnitudes(&self, state: usize) -> Result<Vec<f64>, D3plotError> {
        let cur = self.node_coordinates(state)?;
        let mut out = Vec::with_capacity(self.ctrl.numnp);
        for i in 0..self.ctrl.numnp {
            let mut s = 0.0;
            for d in 0..3 {
                let delta = cur[i * 3 + d] - self.x0[i * 3 + d];
                s += delta * delta;
            }
            out.push(s.sqrt());
        }
        Ok(out)
    }

    /// Maximum nodal displacement magnitude over all nodes at the final state — a common
    /// crash/structures response (peak deflection / intrusion).
    pub fn max_displacement_final(&self) -> Result<f64, D3plotError> {
        if self.states.is_empty() {
            return Err(D3plotError::Unsupported("d3plot has no states".into()));
        }
        let mags = self.displacement_magnitudes(self.states.len() - 1)?;
        Ok(mags.into_iter().fold(0.0_f64, f64::max))
    }

    /// The per-entity layout `(count, vars_per_entity)` of a result block, or
    /// `None` if the block is absent. `vars_per_entity` is the solver's packed
    /// element layout (see [`StateBlock`]).
    pub fn block_layout(&self, block: StateBlock) -> Option<(usize, usize)> {
        self.ctrl.block_spec(block).map(|(_, count, vars)| (count, vars))
    }

    /// Generic result extraction: pull a result block across **all** states as a
    /// flat row-major `(n_states, count, vars)` array in the file's **native
    /// precision** (f32 for single-precision d3plots, f64 for double). One code
    /// path serves node displacement/velocity/acceleration and solid/tshell/
    /// beam/shell element results — the layout differences are entirely in
    /// [`Control::block_spec`], not in per-field code.
    ///
    /// Returns `None` if the block is not present in this d3plot.
    pub fn block_data(&self, block: StateBlock) -> Option<(BlockArray, [usize; 3])> {
        let (off_words, count, vars) = self.ctrl.block_spec(block)?;
        let ws = self.ctrl.wordsize as usize;
        let byte_off = off_words * ws;
        let per_state = count * vars;
        let dims = [self.states.len(), count, vars];
        let total = self.states.len() * per_state;
        let need = per_state * ws;

        let out = if ws == 4 {
            let mut out: Vec<f32> = Vec::with_capacity(total);
            for loc in &self.states {
                let start = loc.offset as usize + byte_off;
                let slice = self.files[loc.file].get(start..start + need)?;
                out.extend(slice.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
            }
            BlockArray::F32(out)
        } else {
            let mut out: Vec<f64> = Vec::with_capacity(total);
            for loc in &self.states {
                let start = loc.offset as usize + byte_off;
                let slice = self.files[loc.file].get(start..start + need)?;
                out.extend(slice.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())));
            }
            BlockArray::F64(out)
        };
        Some((out, dims))
    }
}

/// A result block in the d3plot's native floating-point precision.
pub enum BlockArray {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// The LS-DYNA end-of-file marker is the float -999999 (exactly representable in
/// f32 and f64), used to terminate the state sequence.
fn is_eof_marker(v: f64) -> bool {
    v == -999999.0
}

/// Read one float at `off` (single/double per `wordsize`) as f64; 0.0 if short.
fn read_float_at(bytes: &[u8], off: u64, wordsize: u64) -> f64 {
    let o = off as usize;
    match wordsize {
        4 => bytes
            .get(o..o + 4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
            .unwrap_or(0.0),
        _ => bytes
            .get(o..o + 8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0.0),
    }
}

/// Read `n` floats starting at `byte_offset` from an in-memory buffer, as f64.
fn read_floats_at(bytes: &[u8], byte_offset: u64, n: usize, wordsize: u64) -> Result<Vec<f64>, D3plotError> {
    let start = byte_offset as usize;
    let need = n * wordsize as usize;
    let slice = bytes
        .get(start..start + need)
        .ok_or_else(|| D3plotError::Unsupported("d3plot truncated: float block out of range".into()))?;
    let mut out = Vec::with_capacity(n);
    for chunk in slice.chunks_exact(wordsize as usize) {
        out.push(if wordsize == 4 {
            f32::from_le_bytes(chunk.try_into().unwrap()) as f64
        } else {
            f64::from_le_bytes(chunk.try_into().unwrap())
        });
    }
    Ok(out)
}

/// Detect word size and read the control block from an in-memory base file. d3plot has no explicit
/// precision flag, so we read NDIM (word 15) as 32-bit; a sane value (2..=9) means single precision,
/// otherwise 64-bit. NDIM is a flag word (4..9 encode rigid-body/rigid-road/mattyp); node
/// coordinates are always 3-D regardless.
fn read_control_bytes(bytes: &[u8]) -> Result<Control, D3plotError> {
    let read_i = |off: usize, ws: u64| -> Option<i64> {
        let o = off;
        match ws {
            4 => bytes.get(o..o + 4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64),
            _ => bytes.get(o..o + 8).map(|b| i64::from_le_bytes(b.try_into().unwrap())),
        }
    };
    let ndim32 = read_i(15 * 4, 4).ok_or(D3plotError::BadHeader)?;
    let wordsize: u64 = if (2..=9).contains(&ndim32) {
        4
    } else {
        let ndim64 = read_i(15 * 8, 8).ok_or(D3plotError::BadHeader)?;
        if (2..=9).contains(&ndim64) { 8 } else { return Err(D3plotError::BadHeader); }
    };
    let geti = |word: u64| -> Result<i64, D3plotError> {
        read_i((word * wordsize) as usize, wordsize).ok_or(D3plotError::BadHeader)
    };

    Ok(Control {
        wordsize,
        ndim: geti(15)? as usize,
        numnp: geti(16)? as usize,
        nglbv: geti(18)?.max(0) as usize,
        it: geti(19)?,
        iu: geti(20)?,
        iv: geti(21)?,
        ia: geti(22)?,
        nel8: geti(23)?.max(0) as usize,
        nv3d: geti(27)?.max(0) as usize,
        nel2: geti(28)?.max(0) as usize,
        nv1d: geti(30)?.max(0) as usize,
        nel4: geti(31)?.max(0) as usize,
        nv2d: geti(33)?.max(0) as usize,
        maxint: geti(36)?,
        narbs: geti(39)?.max(0) as usize,
        nelth: geti(40)?.max(0) as usize,
        nv3dt: geti(42)?.max(0) as usize,
        nmmat: geti(51)?.max(0) as usize,
        extra: geti(57)?.max(0) as usize,
    })
}

// ───────────────────────────── round-trip tests ─────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("dynars_d3plot_{nanos}_{}.bin", N.fetch_add(1, Ordering::Relaxed)))
    }

    /// Write a minimal single-precision d3plot the way LS-DYNA lays one out:
    /// 2 nodes, no elements, IU only, geometry zero-padded to a 512-word block,
    /// then two states. State `s` moves node 1 to z = s.
    fn write_synthetic(path: &std::path::Path) {
        let numnp = 2usize;
        let mut words: Vec<i32> = vec![0; 64];
        words[15] = 4;             // NDIM (flag word — coords are still 3-D)
        words[16] = numnp as i32;  // NUMNP
        words[18] = 0;             // NGLBV
        words[20] = 1;             // IU
        // everything else (elements, narbs, maxint, extra, it/iv/ia) = 0

        let mut buf: Vec<u8> = Vec::new();
        for &w in &words { buf.write_i32::<LittleEndian>(w).unwrap(); }
        // geometry: initial node coords (row-major); node0 & node1 at origin.
        let x0: [f32; 6] = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0];
        for &c in &x0 { buf.write_f32::<LittleEndian>(c).unwrap(); }
        // Zero-pad the geometry section up to the next 512-word block.
        let block_bytes = 512 * 4;
        while buf.len() % block_bytes != 0 {
            buf.write_i32::<LittleEndian>(0).unwrap();
        }
        // states: each = TIME + IU block (numnp*3 current coords)
        for s in 0..2i32 {
            buf.write_f32::<LittleEndian>(s as f32).unwrap(); // time
            // node0 stays at origin; node1 moves to z = s
            let cur: [f32; 6] = [0.0, 0.0, 0.0,  0.0, 0.0, s as f32];
            for &c in &cur { buf.write_f32::<LittleEndian>(c).unwrap(); }
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    #[test]
    fn round_trip_single_precision() {
        let p = tmp();
        write_synthetic(&p);
        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.control().wordsize, 4);
        assert_eq!(d.num_nodes(), 2);
        assert_eq!(d.num_states(), 2);

        // state 0: node1 at z=0 → zero displacement
        let m0 = d.displacement_magnitudes(0).unwrap();
        assert!((m0[1] - 0.0).abs() < 1e-6, "{m0:?}");
        // state 1: node1 at z=1 → displacement magnitude 1
        let m1 = d.displacement_magnitudes(1).unwrap();
        assert!((m1[1] - 1.0).abs() < 1e-6, "{m1:?}");

        // max displacement at the final state
        assert!((d.max_displacement_final().unwrap() - 1.0).abs() < 1e-6);

        // generic extractor: displacement block == per-state node coordinates
        let (data, dims) = d.block_data(StateBlock::Displacement).unwrap();
        assert_eq!(dims, [2, 2, 3]); // (n_states, n_nodes, 3)
        match data {
            BlockArray::F32(v) => {
                assert_eq!(v.len(), 12);
                assert!((v[11] - 1.0).abs() < 1e-6); // state1, node1, z = 1
            }
            BlockArray::F64(_) => panic!("single-precision synthetic file should yield f32"),
        }
        // no element blocks in this synthetic file
        assert!(d.block_data(StateBlock::Shell).is_none());
        assert!(d.block_data(StateBlock::Solid).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rejects_garbage() {
        let p = tmp();
        std::fs::write(&p, vec![0xABu8; 256]).unwrap();
        assert!(D3plot::open(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    /// Read a real d3plot family if `DYNARS_TEST_D3PLOT` points at a base file
    /// (e.g. an LS-DYNA/open-lasso `d3plot` with `d3plot01…` siblings). Skips
    /// cleanly when unset. Cross-checked against lasso during development:
    /// node/state counts, times, and per-state node coordinates all match.
    #[test]
    fn real_d3plot_family() {
        let Ok(path) = std::env::var("DYNARS_TEST_D3PLOT") else { return };
        if !std::path::Path::new(&path).exists() {
            return;
        }
        let d = D3plot::open(&path).expect("open real d3plot family");
        assert!(d.num_nodes() > 0, "should have nodes");
        assert_eq!(d.times().len(), d.num_states(), "one time per state");
        if d.num_states() > 0 {
            let coords = d.node_coordinates(0).expect("state 0 coords");
            assert_eq!(coords.len(), d.num_nodes() * 3, "NUMNP x 3 coordinates");
            // Final-state peak displacement must be finite and non-negative.
            let peak = d.max_displacement_final().expect("peak displacement");
            assert!(peak.is_finite() && peak >= 0.0);
        }
    }
}
