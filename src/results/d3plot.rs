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
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum D3plotError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a recognizable d3plot (could not determine word size)")]
    BadHeader,
    #[error("{0}")]
    Unsupported(String),
}

/// Control-block word indices (0-based) per the LS-DYNA Database Manual.
mod word {
    pub const FILETYPE: usize = 11;
    // Interface-force (intfor) per-segment field counts (NV2D = their sum).
    pub const NWEAR: usize = 59;
    pub const NPRESU: usize = 60;
    pub const NSHEAR: usize = 61;
    pub const NFORCE: usize = 62;
    pub const NGAPC: usize = 63;
    pub const NDIM: usize = 15;
    pub const NUMNP: usize = 16;
    pub const ICODE: usize = 17;
    pub const NGLBV: usize = 18;
    pub const IT: usize = 19;
    pub const IU: usize = 20;
    pub const IV: usize = 21;
    pub const IA: usize = 22;
    pub const NEL8: usize = 23;
    pub const NV3D: usize = 27;
    pub const NEL2: usize = 28;
    pub const NV1D: usize = 30;
    pub const NEL4: usize = 31;
    /// Interface-force files store contact segment count in the shell (NEL4) slot.
    pub const NUMSG: usize = 31;
    pub const NUMMAT4: usize = 32;
    pub const NV2D: usize = 33;
    pub const NEIPH: usize = 34;
    pub const NEIPS: usize = 35;
    pub const MAXINT: usize = 36;
    pub const NARBS: usize = 39;
    pub const NELTH: usize = 40;
    pub const NV3DT: usize = 42;
    pub const IOSHL1: usize = 43;
    pub const IOSHL2: usize = 44;
    pub const NMMAT: usize = 51;
    pub const EXTRA: usize = 57;
}

/// The 64-word base control block.
const CONTROL_WORDS: usize = 64;
/// Node coordinates are always 3-D (x, y, z); NDIM is a flag word, not a count.
const SPATIAL_DIM: usize = 3;
/// Each state starts with one time word.
const TIME_WORDS: usize = 1;
/// The `it` header word encodes mass scaling at values >= this base.
const IT_ENCODING_BASE: i64 = 10;
/// Connectivity record widths (node indices + one part/material index).
const SOLID_CONN: usize = 9; // 8 nodes + part
const TSHELL_CONN: usize = 9; // 8 nodes + part
const BEAM_CONN: usize = 6; // 5 slots + part
const SHELL_CONN: usize = 5; // 4 nodes + part
/// Per-element stress components (σxx,σyy,σzz,σxy,σyz,σzx).
const ELEM_STRESS_VARS: usize = 6;
/// Base per-element result vars before history: 6 stress + 1 effective plastic strain.
const ELEM_BASE_VARS: usize = ELEM_STRESS_VARS + 1;
/// NARBS numbering-section header word count when material numbering is present.
const NARBS_PART_HEADER: usize = 16;
/// The run title occupies the first 10 words (40 single-precision chars).
const TITLE_WORDS: usize = 10;
const TITLE_BYTES: usize = TITLE_WORDS * 4;
/// LS-DYNA end-of-state marker, the float -999999.
const EOF_MARKER: f64 = -999999.0;
/// `ioshl*` value meaning "field present".
const IOSHL_PRESENT: i32 = 1000;
/// `filetype` value for a d3plot (vs d3part etc.).
const FILETYPE_D3PLOT: i32 = 1;
/// `filetype` value for an interface-force file (`intfor`).
const FILETYPE_INTFOR: i64 = 4;
/// `icode` value identifying an LS-DYNA database.
const ICODE_LSDYNA: i32 = 6;
/// NDIM flag value for a plain structural model (no rigid body/road, mattyp=0).
const NDIM_STRUCTURAL: i32 = 4;
/// `maxint` magnitude offset that flags element (vs node) deletion in mdlopt.
const MDLOPT_ELEMENT_DELETION: i64 = 10000;
/// Valid NDIM flag range, used to detect word size / a real d3plot header.
const NDIM_RANGE: std::ops::RangeInclusive<i64> = 2..=9;

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
    pub maxint: i64, // shell integration layers; sign encodes element/node deletion (mdlopt)
    pub nmmat: usize, // total number of materials/parts
    pub extra: usize, // extra header words beyond the base 64 (word 57)
    // Interface-force (intfor) fields. `filetype == 4` marks an intfor file, in
    // which the "shell" slot (nel4) holds interface segments and nv2d their
    // per-segment values, split into these counts.
    pub filetype: i64,
    pub fsifor: bool, // intfor with negative NV2D ⇒ FSIFOR (ALE) file
    pub nwear: usize,
    pub npresu: usize,
    pub nshear: usize,
    pub nforce: usize,
    pub ngapc: usize,
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
        } else if -self.maxint >= MDLOPT_ELEMENT_DELETION {
            2
        } else {
            1
        }
    }

    /// Byte offset of the first state database: the sum of all geometry sections.
    /// States begin immediately after the exact geometry (LS-DYNA does not pad it
    /// to a block boundary). Node coordinates are always 3-D; the NDIM header
    /// word is a flag set — see `read_control`.
    ///
    /// Note: the part & contact-interface title section (present in some real
    /// single-file decks) is not accounted for here. It's harmless for the
    /// multi-file family case (the base file's leftover is smaller than one state)
    /// and for files dynars writes (which omit it).
    fn geometry_section_bytes(&self) -> u64 {
        let mut words = CONTROL_WORDS + self.extra;
        words += self.numnp * SPATIAL_DIM; // node coordinates
        words += self.nel8 * SOLID_CONN; // solids
        words += self.nelth * TSHELL_CONN; // thick shells
        words += self.nel2 * BEAM_CONN; // beams
        words += self.nel4 * SHELL_CONN; // shells
        if self.mattyp() {
            words += 2 + self.nmmat; // material-type section
        }
        words += self.narbs; // arbitrary node/element numbering section
        words as u64 * self.wordsize
    }

    /// Thermal / mass-scaling variables per node (the IT block), decoded from
    /// the `it` header word: `it % 10` temperature variants, `+1` for mass
    /// scaling when `it >= 10`. Common cases (0/1/10) are exact; exotic thermal
    /// layouts may differ.
    fn node_therm_vars(&self) -> usize {
        let temp = self.it.rem_euclid(IT_ENCODING_BASE).max(0) as usize;
        let mass = if self.it >= IT_ENCODING_BASE { 1 } else { 0 };
        temp + mass
    }

    /// Total words of node data per state: (IU + IV + IA) × 3 + thermal, × NUMNP.
    fn node_data_words(&self) -> usize {
        let vec3 = (self.iu != 0) as usize + (self.iv != 0) as usize + (self.ia != 0) as usize;
        (vec3 * SPATIAL_DIM + self.node_therm_vars()) * self.numnp
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
        self.nel8 * self.nv3d
            + self.nelth * self.nv3dt
            + self.nel2 * self.nv1d
            + self.nel4 * self.nv2d
    }

    /// Bytes per state: time + global vars + node data + element data + deletion.
    /// SPH/airbag/rigid-road terms are not modelled (v1 scope).
    fn bytes_per_state(&self) -> u64 {
        (TIME_WORDS
            + self.nglbv
            + self.node_data_words()
            + self.element_words()
            + self.deletion_words()) as u64
            * self.wordsize
    }

    /// `(word offset within a state, entity count, vars per entity)` for a result
    /// block, or `None` when that block is absent. This one table is what makes
    /// extraction generic — every block is just a strided slice.
    fn block_spec(&self, block: StateBlock) -> Option<(usize, usize, usize)> {
        let base = TIME_WORDS + self.nglbv; // after the time word + global vars
        let n3 = self.numnp * SPATIAL_DIM;
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
            StateBlock::Displacement => some(self.iu != 0, disp, self.numnp, SPATIAL_DIM),
            StateBlock::Velocity => some(self.iv != 0, vel, self.numnp, SPATIAL_DIM),
            StateBlock::Acceleration => some(self.ia != 0, acc, self.numnp, SPATIAL_DIM),
            StateBlock::Solid => some(true, solid, self.nel8, self.nv3d),
            StateBlock::ThickShell => some(true, tshell, self.nelth, self.nv3dt),
            StateBlock::Beam => some(true, beam, self.nel2, self.nv1d),
            StateBlock::Shell => some(true, shell, self.nel4, self.nv2d),
        }
    }

    /// Byte offset, within a state, of the IU (deformed-coordinate) block.
    fn iu_offset_in_state(&self) -> u64 {
        (TIME_WORDS + self.nglbv) as u64 * self.wordsize
    }
}

/// A per-entity result block in a state. Node blocks are (N, 3); element blocks
/// are (N, vars) where `vars` is the solver's packed per-element layout
/// (stresses, plastic strain, history variables, per integration point/layer) —
/// returned raw for the caller to reshape.
///
/// This is the single source of truth for block identity: the reader/writer use
/// it directly, and (with the `python` feature) it is exported to Python as the
/// `StateBlock` enum — no magic strings.
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "StateBlock")
)]
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

/// A per-segment field in an interface-force (`intfor`) file. These partition
/// the segment result block (`StateBlock::Shell`) in this order and sum to
/// `nv2d`. Exported to Python as the `InterfaceField` enum — no magic strings.
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "InterfaceField")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceField {
    Wear,
    Pressure,
    Shear,
    Force,
    Gap,
}

/// A per-segment field in an **FSIFOR** (ALE interface-force) file. These are
/// single-value fields in this fixed order; the file carries as many as `|nv2d|`.
/// Exported to Python as the `FsiforField` enum.
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "FsiforField")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiforField {
    Pressure,
    ForceX,
    ForceY,
    ForceZ,
    RelativeVelocity,
    VelocityX,
    VelocityY,
    VelocityZ,
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

/// Family member paths for a base d3plot: `base`, `base01`, `base02`, … For
/// reading, `n = None` stops at the first missing sibling; for writing, `n =
/// Some(k)` returns exactly `k` names.
fn family_paths(base: &std::path::Path, n: Option<usize>) -> Vec<std::path::PathBuf> {
    let stem = base.to_string_lossy().into_owned();
    let mut out = vec![base.to_path_buf()];
    let mut i = 1;
    loop {
        if let Some(k) = n
            && out.len() >= k
        {
            break;
        }
        let name = if i < 100 {
            format!("{stem}{i:02}")
        } else {
            format!("{stem}{i}")
        };
        let p = std::path::PathBuf::from(&name);
        if n.is_none() && !p.exists() {
            break;
        }
        out.push(p);
        i += 1;
    }
    out
}

/// Read the control block from the base and walk all states across the family,
/// returning per-state `(file, offset)` locations and times.
fn index_family(files: &[&[u8]]) -> Result<(Control, Vec<StateLoc>, Vec<f64>), D3plotError> {
    let ctrl = read_control_bytes(files[0])?;
    if ctrl.iu == 0 {
        return Err(D3plotError::Unsupported(
            "d3plot has no nodal displacement data (IU=0)".into(),
        ));
    }
    let ws = ctrl.wordsize;
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
                states.push(StateLoc {
                    file: fi,
                    offset: off,
                });
                off += bps;
            }
        }
    }
    Ok((ctrl, states, times))
}

impl D3plot {
    pub fn control(&self) -> &Control {
        &self.ctrl
    }
    pub fn num_nodes(&self) -> usize {
        self.ctrl.numnp
    }
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
    /// Simulation time of each state.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, D3plotError> {
        // Memory-map the base file plus any continuation files (d3plot01, …).
        let files: Vec<Mmap> = family_paths(path.as_ref(), None)
            .iter()
            .map(|p| mmap(p))
            .collect::<Result<_, _>>()?;
        let file_slices: Vec<&[u8]> = files.iter().map(|m| &m[..]).collect();
        let (ctrl, states, times) = index_family(&file_slices)?;
        let coord_off = (CONTROL_WORDS + ctrl.extra) as u64 * ctrl.wordsize;
        let x0 = read_floats_at(
            &files[0],
            coord_off,
            ctrl.numnp * SPATIAL_DIM,
            ctrl.wordsize,
        )?;
        Ok(Self {
            ctrl,
            files,
            states,
            x0,
            times,
        })
    }

    /// Deformed (current) node coordinates at `state` (0-based), NUMNP × 3 row-major.
    pub fn node_coordinates(&self, state: usize) -> Result<Vec<f64>, D3plotError> {
        let loc = self.states.get(state).ok_or_else(|| {
            D3plotError::Unsupported(format!(
                "state {state} out of range ({})",
                self.states.len()
            ))
        })?;
        let ws = self.ctrl.wordsize;
        let off = loc.offset + self.ctrl.iu_offset_in_state();
        read_floats_at(
            &self.files[loc.file],
            off,
            self.ctrl.numnp * SPATIAL_DIM,
            ws,
        )
    }

    /// Deformed node coordinates for **every** state in one pass: a flat
    /// `num_states × NUMNP × 3` row-major buffer (state outermost, then node,
    /// then x/y/z). One allocation, one call — pulls the whole coordinate history
    /// without the per-state call and allocation overhead of repeatedly invoking
    /// [`node_coordinates`](Self::node_coordinates) (the difference is stark
    /// across a language boundary, where each state would be its own round-trip).
    pub fn node_coordinates_all(&self) -> Result<Vec<f64>, D3plotError> {
        let per = self.ctrl.numnp * SPATIAL_DIM;
        let mut out = vec![0.0f64; self.states.len() * per];
        let ws = self.ctrl.wordsize;
        let iu = self.ctrl.iu_offset_in_state();
        for (s, loc) in self.states.iter().enumerate() {
            let off = loc.offset + iu;
            read_floats_into(
                &self.files[loc.file],
                off,
                &mut out[s * per..(s + 1) * per],
                ws,
            )?;
        }
        Ok(out)
    }

    /// Per-node displacement magnitude at `state`: |current − initial|.
    pub fn displacement_magnitudes(&self, state: usize) -> Result<Vec<f64>, D3plotError> {
        let cur = self.node_coordinates(state)?;
        let mut out = Vec::with_capacity(self.ctrl.numnp);
        for i in 0..self.ctrl.numnp {
            let mut s = 0.0;
            for d in 0..SPATIAL_DIM {
                let delta = cur[i * SPATIAL_DIM + d] - self.x0[i * SPATIAL_DIM + d];
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

    /// Initial node coordinates (`numnp * 3`, row-major) — the reference geometry.
    pub fn initial_coordinates(&self) -> &[f64] {
        &self.x0
    }

    /// File type from the control block (1 = d3plot, 4 = intfor, …).
    pub fn filetype(&self) -> i64 {
        self.ctrl.filetype
    }

    /// Whether this is an interface-force (`intfor`) database. Such a file is a
    /// d3plot-family binary whose "shell" slot holds contact **segments**: read
    /// them with `block(StateBlock::Shell)` (shape `(n_states, n_segments, nv2d)`)
    /// and `shell_connectivity()`, and split the per-segment values with
    /// [`interface_fields`](Self::interface_fields).
    pub fn is_interface_force(&self) -> bool {
        self.ctrl.filetype == FILETYPE_INTFOR
    }

    /// For an `intfor` file, the per-segment field counts that partition the
    /// segment result block, in order: wear, pressure, shear, force, gap. Their
    /// sum is `nv2d`.
    pub fn interface_fields(&self) -> InterfaceFields {
        InterfaceFields {
            wear: self.ctrl.nwear,
            pressure: self.ctrl.npresu,
            shear: self.ctrl.nshear,
            force: self.ctrl.nforce,
            gap: self.ctrl.ngapc,
        }
    }

    /// Whether this is an **FSIFOR** (ALE) interface-force file — an intfor file
    /// with negative NV2D and a fixed per-segment field layout ([`FsiforField`]).
    pub fn is_fsifor(&self) -> bool {
        self.ctrl.fsifor
    }

    /// `(offset, count)` of an [`InterfaceField`] within the per-segment block
    /// (`StateBlock::Shell`, shape `(n_states, n_segments, nv2d)`). `count == 0`
    /// means the field is absent from this file.
    pub fn interface_field_span(&self, field: InterfaceField) -> (usize, usize) {
        let c = &self.ctrl;
        match field {
            InterfaceField::Wear => (0, c.nwear),
            InterfaceField::Pressure => (c.nwear, c.npresu),
            InterfaceField::Shear => (c.nwear + c.npresu, c.nshear),
            InterfaceField::Force => (c.nwear + c.npresu + c.nshear, c.nforce),
            InterfaceField::Gap => (c.nwear + c.npresu + c.nshear + c.nforce, c.ngapc),
        }
    }

    /// `(offset, count)` of an [`FsiforField`] within the per-segment block. Each
    /// FSIFOR field is one value at a fixed column; `count == 0` if the file
    /// doesn't carry that column (`|nv2d|` fields total).
    pub fn fsifor_field_span(&self, field: FsiforField) -> (usize, usize) {
        let idx = field as usize;
        if idx < self.ctrl.nv2d {
            (idx, 1)
        } else {
            (idx, 0)
        }
    }

    /// Shell connectivity as `(node_indices, part_indices)`: `node_indices` is
    /// `n_shells * 4` one-based node numbers, `part_indices` is `n_shells`.
    pub fn shell_connectivity(&self) -> (Vec<i64>, Vec<i64>) {
        self.connectivity(self.shells_offset_words(), self.ctrl.nel4, 4)
    }

    /// Solid connectivity as `(node_indices, part_indices)`: `node_indices` is
    /// `n_solids * 8` one-based node numbers, `part_indices` is `n_solids`.
    pub fn solid_connectivity(&self) -> (Vec<i64>, Vec<i64>) {
        self.connectivity(self.solids_offset_words(), self.ctrl.nel8, 8)
    }

    /// User node IDs (`numnp`), or `1..=numnp` if the file has no numbering section.
    pub fn node_ids(&self) -> Vec<i64> {
        self.narbs_slice(0, self.ctrl.numnp)
            .unwrap_or_else(|| (1..=self.ctrl.numnp as i64).collect())
    }

    /// User solid element IDs.
    pub fn solid_ids(&self) -> Vec<i64> {
        self.narbs_slice(self.ctrl.numnp, self.ctrl.nel8)
            .unwrap_or_else(|| (1..=self.ctrl.nel8 as i64).collect())
    }

    /// User shell element IDs.
    pub fn shell_ids(&self) -> Vec<i64> {
        let before = self.ctrl.numnp + self.ctrl.nel8 + self.ctrl.nel2;
        self.narbs_slice(before, self.ctrl.nel4)
            .unwrap_or_else(|| (1..=self.ctrl.nel4 as i64).collect())
    }

    /// User part/material IDs.
    pub fn part_ids(&self) -> Vec<i64> {
        let before =
            self.ctrl.numnp + self.ctrl.nel8 + self.ctrl.nel2 + self.ctrl.nel4 + self.ctrl.nelth;
        self.narbs_slice(before, self.ctrl.nmmat)
            .unwrap_or_else(|| (1..=self.ctrl.nmmat as i64).collect())
    }

    // --- geometry-section offset helpers (words into the base file) ---
    fn solids_offset_words(&self) -> usize {
        CONTROL_WORDS + self.ctrl.extra + self.ctrl.numnp * SPATIAL_DIM
    }
    fn shells_offset_words(&self) -> usize {
        self.solids_offset_words()
            + self.ctrl.nel8 * SOLID_CONN
            + self.ctrl.nelth * TSHELL_CONN
            + self.ctrl.nel2 * BEAM_CONN
    }
    fn narbs_offset_words(&self) -> usize {
        let mut w = self.shells_offset_words() + self.ctrl.nel4 * SHELL_CONN;
        if self.ctrl.mattyp() {
            w += 2 + self.ctrl.nmmat;
        }
        w
    }

    /// Read `count` connectivity records of `nodes_per` node indices + 1 part.
    fn connectivity(
        &self,
        off_words: usize,
        count: usize,
        nodes_per: usize,
    ) -> (Vec<i64>, Vec<i64>) {
        let ws = self.ctrl.wordsize as usize;
        let stride = nodes_per + 1;
        let raw = read_ints_at(&self.files[0], off_words * ws, count * stride, ws);
        let mut nodes = Vec::with_capacity(count * nodes_per);
        let mut parts = Vec::with_capacity(count);
        for rec in raw.chunks_exact(stride) {
            nodes.extend_from_slice(&rec[..nodes_per]);
            parts.push(rec[nodes_per]);
        }
        (nodes, parts)
    }

    /// A slice of `n` IDs from the NARBS numbering section, `skip` IDs in (node,
    /// solid, beam, shell, tshell, material order). `None` if no NARBS section.
    fn narbs_slice(&self, skip: usize, n: usize) -> Option<Vec<i64>> {
        if self.ctrl.narbs == 0 || n == 0 {
            return None;
        }
        let ws = self.ctrl.wordsize as usize;
        let narbs_off = self.narbs_offset_words();
        // NSORT<0 ⇒ 16-word header (with material numbering), else 10.
        let nsort = read_ints_at(&self.files[0], narbs_off * ws, 1, ws);
        let header = if nsort.first().copied().unwrap_or(0) < 0 {
            16
        } else {
            10
        };
        let off_words = narbs_off + header + skip;
        Some(read_ints_at(&self.files[0], off_words * ws, n, ws))
    }

    /// The per-entity layout `(count, vars_per_entity)` of a result block, or
    /// `None` if the block is absent. `vars_per_entity` is the solver's packed
    /// element layout (see [`StateBlock`]).
    pub fn block_layout(&self, block: StateBlock) -> Option<(usize, usize)> {
        self.ctrl
            .block_spec(block)
            .map(|(_, count, vars)| (count, vars))
    }

    /// Resolve a state selection to concrete indices. `None` = all states;
    /// negative indices count from the end. Errors on out-of-range indices.
    pub fn resolve_states(&self, sel: Option<&[i64]>) -> Result<Vec<usize>, D3plotError> {
        let n = self.states.len() as i64;
        match sel {
            None => Ok((0..self.states.len()).collect()),
            Some(idx) => idx
                .iter()
                .map(|&i| {
                    let j = if i < 0 { i + n } else { i };
                    if j < 0 || j >= n {
                        Err(D3plotError::Unsupported(format!(
                            "state index {i} out of range ({n} states)"
                        )))
                    } else {
                        Ok(j as usize)
                    }
                })
                .collect(),
        }
    }

    /// Info for a **zero-copy** strided view of a block over the memory map for
    /// the given (already-resolved) states, or `None` when that's not possible:
    /// block absent, double precision, empty selection, or the selected states
    /// don't lie in one file at a single constant byte stride. Returns
    /// `(file_index, byte_offset_of_first_block, [n, count, vars], state_stride_bytes)`.
    /// Byte offsets are 4-aligned (word-aligned data in a page-aligned map).
    pub fn block_view(
        &self,
        block: StateBlock,
        states: &[usize],
    ) -> Option<(usize, usize, [usize; 3], usize)> {
        if self.ctrl.wordsize != 4 || states.is_empty() {
            return None;
        }
        let fi = self.states[states[0]].file;
        let base = self.states[states[0]].offset as usize;
        // All selected states must live in the same file at a constant, positive
        // (ascending) stride, so they form one strided NumPy view.
        let stride = if states.len() >= 2 {
            let s1 = &self.states[states[1]];
            if s1.file != fi || (s1.offset as usize) <= base {
                return None;
            }
            s1.offset as usize - base
        } else {
            self.ctrl.bytes_per_state() as usize
        };
        for w in states.windows(2) {
            let a = &self.states[w[0]];
            let c = &self.states[w[1]];
            if c.file != fi || (c.offset as usize).checked_sub(a.offset as usize) != Some(stride) {
                return None;
            }
        }
        let (off_words, count, vars) = self.ctrl.block_spec(block)?;
        let byte_off = base + off_words * 4;
        Some((fi, byte_off, [states.len(), count, vars], stride))
    }

    /// Raw bytes of a family file (for building zero-copy views).
    pub fn file_bytes(&self, i: usize) -> &[u8] {
        &self.files[i]
    }

    /// Generic result extraction (copy path): pull a result block for the given
    /// `states` as a flat row-major `(n, count, vars)` array in the file's
    /// **native precision** (f32 single / f64 double). One code path serves node
    /// displacement/velocity/acceleration and solid/tshell/beam/shell element
    /// results — the layout differences live entirely in [`Control::block_spec`].
    /// The per-state copies run in parallel for large selections.
    ///
    /// Returns `None` if the block is not present in this d3plot.
    pub fn block_data(
        &self,
        block: StateBlock,
        states: &[usize],
    ) -> Option<(BlockArray, [usize; 3])> {
        let (off_words, count, vars) = self.ctrl.block_spec(block)?;
        let ws = self.ctrl.wordsize as usize;
        let byte_off = off_words * ws;
        let per_state = count * vars;
        let total = states.len() * per_state;
        let dims = [states.len(), count, vars];

        // Copy state `si`'s block into `dst` (exactly `per_state` elements).
        macro_rules! fill {
            ($ty:ty, $n:expr) => {{
                // SAFETY: every element is written before use; `$ty` has no Drop.
                let mut out: Vec<$ty> = Vec::with_capacity(total);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    out.set_len(total)
                };
                let fill_one = |dst: &mut [$ty], si: usize| -> Option<()> {
                    let loc = &self.states[si];
                    let start = loc.offset as usize + byte_off;
                    let slice = self.files[loc.file].get(start..start + per_state * $n)?;
                    for (d, c) in dst.iter_mut().zip(slice.chunks_exact($n)) {
                        *d = <$ty>::from_le_bytes(c.try_into().unwrap());
                    }
                    Some(())
                };
                // Parallelize across states once the copy is big enough to matter.
                if total >= (1 << 16) && states.len() > 1 {
                    let ok = out
                        .par_chunks_mut(per_state)
                        .zip(states.par_iter())
                        .all(|(dst, &si)| fill_one(dst, si).is_some());
                    if !ok {
                        return None;
                    }
                } else {
                    for (dst, &si) in out.chunks_mut(per_state).zip(states.iter()) {
                        fill_one(dst, si)?;
                    }
                }
                out
            }};
        }

        let out = if ws == 4 {
            BlockArray::F32(fill!(f32, 4))
        } else {
            BlockArray::F64(fill!(f64, 8))
        };
        Some((out, dims))
    }

    /// The per-element result `block` across **all** states as `f64`, with dims
    /// `[n_states, n_elem, nv]` and the per-element part index — ready for the
    /// [`element`](super::element) per-part reductions. Supports `Solid`/`Shell`
    /// (which carry connectivity); the ready-made extractors
    /// [`von_mises_stress`](super::element::von_mises_stress) /
    /// [`effective_plastic_strain`](super::element::effective_plastic_strain)
    /// assume the base "6 stress + plastic strain" element layout.
    pub fn element_block_f64(&self, block: StateBlock) -> Option<(Vec<f64>, [usize; 3], Vec<i64>)> {
        let states: Vec<usize> = (0..self.states.len()).collect();
        let (arr, dims) = self.block_data(block, &states)?;
        let part_ids = match block {
            StateBlock::Solid => self.solid_connectivity().1,
            StateBlock::Shell => self.shell_connectivity().1,
            _ => return None,
        };
        Some((arr.to_f64(), dims, part_ids))
    }
}

/// A result block in the d3plot's native floating-point precision.
pub enum BlockArray {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl BlockArray {
    /// Values as `f64` (casts an `f32` block, clones an `f64` one).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            BlockArray::F32(v) => v.iter().map(|&x| x as f64).collect(),
            BlockArray::F64(v) => v.clone(),
        }
    }

    /// Number of scalar values in the block.
    pub fn len(&self) -> usize {
        match self {
            BlockArray::F32(v) => v.len(),
            BlockArray::F64(v) => v.len(),
        }
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Edits an existing d3plot family in place: overwrite node coordinates or any
/// result block at chosen states, then re-emit. Everything not overwritten
/// (header, geometry, IDs, flags, other results) is preserved byte-for-byte, so
/// the result reads back identically — including in LS-PrePost / lasso, whose
/// element-field interpretation depends on flags we leave untouched.
pub struct D3plotEditor {
    files: Vec<Vec<u8>>,
    paths: Vec<std::path::PathBuf>,
    ctrl: Control,
    states: Vec<StateLoc>,
}

impl D3plotEditor {
    /// Load a d3plot family (base + `d3plot01`, …) into memory for editing.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, D3plotError> {
        let paths = family_paths(path.as_ref(), None);
        let files: Vec<Vec<u8>> = paths.iter().map(std::fs::read).collect::<Result<_, _>>()?;
        let slices: Vec<&[u8]> = files.iter().map(|v| &v[..]).collect();
        let (ctrl, states, _) = index_family(&slices)?;
        if ctrl.wordsize != 4 {
            return Err(D3plotError::Unsupported(
                "editing double-precision d3plots is not supported".into(),
            ));
        }
        Ok(Self {
            files,
            paths,
            ctrl,
            states,
        })
    }

    pub fn control(&self) -> &Control {
        &self.ctrl
    }
    pub fn num_nodes(&self) -> usize {
        self.ctrl.numnp
    }
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Overwrite a result block for one state with `data` (`count * vars` f32
    /// values, native single precision). Only these bytes change.
    pub fn set_block(
        &mut self,
        block: StateBlock,
        state: usize,
        data: &[f32],
    ) -> Result<(), D3plotError> {
        let (off_words, count, vars) = self
            .ctrl
            .block_spec(block)
            .ok_or_else(|| D3plotError::Unsupported("block not present in this d3plot".into()))?;
        if data.len() != count * vars {
            return Err(D3plotError::Unsupported(format!(
                "data length {} != count*vars ({})",
                data.len(),
                count * vars
            )));
        }
        let loc = self.states.get(state).ok_or_else(|| {
            D3plotError::Unsupported(format!(
                "state {state} out of range ({})",
                self.states.len()
            ))
        })?;
        let base = loc.offset as usize + off_words * 4;
        let buf = &mut self.files[loc.file];
        if base + data.len() * 4 > buf.len() {
            return Err(D3plotError::Unsupported(
                "block extent past end of file".into(),
            ));
        }
        for (i, &v) in data.iter().enumerate() {
            buf[base + i * 4..base + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    /// Overwrite deformed node coordinates (`numnp * 3` f32) for one state.
    pub fn set_node_coordinates(
        &mut self,
        state: usize,
        coords: &[f32],
    ) -> Result<(), D3plotError> {
        self.set_block(StateBlock::Displacement, state, coords)
    }

    /// Overwrite the original files in place.
    pub fn save(&self) -> Result<(), D3plotError> {
        for (p, bytes) in self.paths.iter().zip(&self.files) {
            std::fs::write(p, bytes)?;
        }
        Ok(())
    }

    /// Write the edited family to a new base path (`path`, `path01`, …).
    pub fn write<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), D3plotError> {
        for (p, bytes) in family_paths(path.as_ref(), Some(self.files.len()))
            .iter()
            .zip(&self.files)
        {
            std::fs::write(p, bytes)?;
        }
        Ok(())
    }
}

/// Emit the NARBS numbering section (part-id form) for a geometry with `numnp`
/// nodes, `nel8` solids, `nel4` shells (or intfor segments), and `nmmat`
/// materials. IDs default to `1..=N` when not supplied. Returns the section's
/// word count (= `NARBS`).
#[allow(clippy::too_many_arguments)]
fn write_narbs(
    buf: &mut Vec<u8>,
    numnp: usize,
    nel8: usize,
    nel4: usize,
    nmmat: usize,
    node_ids: Option<&[i32]>,
    solid_ids: Option<&[i32]>,
    shell_ids: Option<&[i32]>,
    part_ids: Option<&[i32]>,
) -> usize {
    let put = |buf: &mut Vec<u8>, v: i32| buf.extend_from_slice(&v.to_le_bytes());
    let seq = |n: usize| -> Vec<i32> { (1..=n as i32).collect() };
    let (nel8i, nel4i, nmmi) = (nel8 as i32, nel4 as i32, nmmat as i32);
    // numbering header (part-id form, 16 words)
    put(buf, -1); // NSORT (negative: material numbering present)
    let nsrh = 1 + numnp as i32;
    put(buf, nsrh);
    let nsrb = nsrh + nel8i;
    put(buf, nsrb);
    let nsrs = nsrb; // + nel2 (=0)
    put(buf, nsrs);
    let nsrt = nsrs + nel4i;
    put(buf, nsrt);
    put(buf, numnp as i32); // NSORTD
    put(buf, nel8i); // NSRHD
    put(buf, 0); // NSRBD
    put(buf, nel4i); // NSRSD
    put(buf, 0); // NSRTD
    let nsrmu = nsrt;
    let nsrma = nsrmu + nmmi;
    let nsrmp = nsrma + nmmi;
    put(buf, nsrma);
    put(buf, nsrmu);
    put(buf, nsrmp);
    put(buf, nmmi);
    put(buf, 0); // NUMRBS
    put(buf, nmmi);
    // ID arrays
    for &v in &node_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(numnp)) {
        put(buf, v);
    }
    for &v in &solid_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nel8)) {
        put(buf, v);
    }
    for &v in &shell_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nel4)) {
        put(buf, v);
    }
    let parts = part_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nmmat));
    for i in 0..nmmat {
        put(buf, parts.get(i).copied().unwrap_or(i as i32 + 1)); // material ids
    }
    for _ in 0..nmmat {
        put(buf, 0); // unordered material ids
    }
    for _ in 0..nmmat {
        put(buf, 0); // material cross-references
    }
    numnp + nel8 + nel4 + 3 * nmmat + NARBS_PART_HEADER
}

/// The LS-DYNA end-of-file marker is the float -999999 (exactly representable in
/// f32 and f64), used to terminate the state sequence.
fn is_eof_marker(v: f64) -> bool {
    v == EOF_MARKER
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

/// Read `out.len()` floats starting at `byte_offset` into `out`, as f64. Lets
/// callers fill a slice of a larger buffer with no intermediate allocation.
fn read_floats_into(
    bytes: &[u8],
    byte_offset: u64,
    out: &mut [f64],
    wordsize: u64,
) -> Result<(), D3plotError> {
    let start = byte_offset as usize;
    let need = out.len() * wordsize as usize;
    let slice = bytes.get(start..start + need).ok_or_else(|| {
        D3plotError::Unsupported("d3plot truncated: float block out of range".into())
    })?;
    for (dst, chunk) in out.iter_mut().zip(slice.chunks_exact(wordsize as usize)) {
        *dst = if wordsize == 4 {
            f32::from_le_bytes(chunk.try_into().unwrap()) as f64
        } else {
            f64::from_le_bytes(chunk.try_into().unwrap())
        };
    }
    Ok(())
}

/// Read `n` floats starting at `byte_offset` from an in-memory buffer, as f64.
fn read_floats_at(
    bytes: &[u8],
    byte_offset: u64,
    n: usize,
    wordsize: u64,
) -> Result<Vec<f64>, D3plotError> {
    let mut out = vec![0.0f64; n];
    read_floats_into(bytes, byte_offset, &mut out, wordsize)?;
    Ok(out)
}

/// Read `n` integers (single/double word) starting at `byte_offset` as i64;
/// stops early if the buffer is short.
fn read_ints_at(bytes: &[u8], byte_offset: usize, n: usize, wordsize: usize) -> Vec<i64> {
    let end = (byte_offset + n * wordsize).min(bytes.len());
    let slice = &bytes[byte_offset.min(bytes.len())..end];
    slice
        .chunks_exact(wordsize)
        .map(|c| {
            if wordsize == 4 {
                i32::from_le_bytes(c.try_into().unwrap()) as i64
            } else {
                i64::from_le_bytes(c.try_into().unwrap())
            }
        })
        .collect()
}

/// Detect word size and read the control block from an in-memory base file. d3plot has no explicit
/// precision flag, so we read NDIM as 32-bit; a value in [`NDIM_RANGE`] means single precision,
/// otherwise 64-bit. NDIM is a flag word (4..9 encode rigid-body/rigid-road/mattyp); node
/// coordinates are always 3-D regardless.
fn read_control_bytes(bytes: &[u8]) -> Result<Control, D3plotError> {
    let read_i = |off: usize, ws: u64| -> Option<i64> {
        match ws {
            4 => bytes
                .get(off..off + 4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64),
            _ => bytes
                .get(off..off + 8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap())),
        }
    };
    let ndim32 = read_i(word::NDIM * 4, 4).ok_or(D3plotError::BadHeader)?;
    let wordsize: u64 = if NDIM_RANGE.contains(&ndim32) {
        4
    } else {
        let ndim64 = read_i(word::NDIM * 8, 8).ok_or(D3plotError::BadHeader)?;
        if NDIM_RANGE.contains(&ndim64) {
            8
        } else {
            return Err(D3plotError::BadHeader);
        }
    };
    let geti = |w: usize| -> Result<i64, D3plotError> {
        read_i(w * wordsize as usize, wordsize).ok_or(D3plotError::BadHeader)
    };

    Ok(Control {
        wordsize,
        ndim: geti(word::NDIM)? as usize,
        numnp: geti(word::NUMNP)? as usize,
        nglbv: geti(word::NGLBV)?.max(0) as usize,
        it: geti(word::IT)?,
        iu: geti(word::IU)?,
        iv: geti(word::IV)?,
        ia: geti(word::IA)?,
        nel8: geti(word::NEL8)?.max(0) as usize,
        nv3d: geti(word::NV3D)?.max(0) as usize,
        nel2: geti(word::NEL2)?.max(0) as usize,
        nv1d: geti(word::NV1D)?.max(0) as usize,
        nel4: geti(word::NEL4)?.max(0) as usize,
        // NV2D is negative in FSIFOR (ALE interface-force) files; the magnitude
        // is the per-segment value count.
        nv2d: geti(word::NV2D)?.unsigned_abs() as usize,
        maxint: geti(word::MAXINT)?,
        narbs: geti(word::NARBS)?.max(0) as usize,
        nelth: geti(word::NELTH)?.max(0) as usize,
        nv3dt: geti(word::NV3DT)?.max(0) as usize,
        nmmat: geti(word::NMMAT)?.max(0) as usize,
        extra: geti(word::EXTRA)?.max(0) as usize,
        filetype: geti(word::FILETYPE)?,
        fsifor: geti(word::NV2D)? < 0,
        nwear: geti(word::NWEAR)?.max(0) as usize,
        npresu: geti(word::NPRESU)?.max(0) as usize,
        nshear: geti(word::NSHEAR)?.max(0) as usize,
        nforce: geti(word::NFORCE)?.max(0) as usize,
        ngapc: geti(word::NGAPC)?.max(0) as usize,
    })
}

/// Which optional nodal result arrays each state carries.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeFields {
    pub velocity: bool,
    pub acceleration: bool,
}

/// How an interface-force (`intfor`) file's per-segment result block splits.
/// The blocks appear in this order (wear, pressure, shear, force, gap) and sum
/// to `nv2d`. Typical values: pressure 1–3, shear 3, force 12 (x/y/z at 4
/// nodes), gap 5, wear 4.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterfaceFields {
    pub wear: usize,
    pub pressure: usize,
    pub shear: usize,
    pub force: usize,
    pub gap: usize,
}

/// Builds a single-precision d3plot from a mesh (nodes + shell/solid
/// connectivity) and per-state nodal results (deformed coordinates, and
/// optionally velocity/acceleration).
///
/// Scope (v1): NDIM=4 structural layout, implicit 1..N numbering (NARBS=0), no
/// global variables, and no per-element result fields — a mesh you can display
/// and animate by nodal motion. Output is a single file: header + block-aligned
/// geometry + states, terminated by the `-999999` marker. Reads back through
/// [`D3plot`] and open-lasso-python (node data bit-exact).
pub struct D3plotWriter {
    numnp: usize,
    x0: Vec<f32>,          // numnp*3, row-major x,y,z
    solids: Vec<[i32; 9]>, // 8 node connectivity indices (1-based) + part index
    shells: Vec<[i32; 5]>, // 4 node connectivity indices (1-based) + part index
    states: Vec<StateData>,
    fields: NodeFields,
    title: String,
    // Optional user IDs for the NARBS numbering section (default 1..N).
    node_ids: Option<Vec<i32>>,
    solid_ids: Option<Vec<i32>>,
    shell_ids: Option<Vec<i32>>,
    part_ids: Option<Vec<i32>>,
    // Optional per-element result blocks: (vars_per_element, flat
    // n_states*count*vars, row-major). Written raw after node data each state.
    solid_results: Option<(usize, Vec<f32>)>,
    shell_results: Option<(usize, Vec<f32>)>,
}

struct StateData {
    time: f32,
    disp: Vec<f32>, // numnp*3 current coordinates
    vel: Vec<f32>,  // numnp*3 or empty
    acc: Vec<f32>,  // numnp*3 or empty
}

impl D3plotWriter {
    /// Start from initial node coordinates (`numnp*3`, row-major x,y,z).
    pub fn new(node_coords: Vec<f64>) -> Result<Self, D3plotError> {
        if node_coords.is_empty() || !node_coords.len().is_multiple_of(3) {
            return Err(D3plotError::Unsupported(
                "node_coords length must be a non-zero multiple of 3".into(),
            ));
        }
        Ok(Self {
            numnp: node_coords.len() / 3,
            x0: node_coords.iter().map(|&c| c as f32).collect(),
            solids: Vec::new(),
            shells: Vec::new(),
            states: Vec::new(),
            fields: NodeFields::default(),
            title: String::new(),
            node_ids: None,
            solid_ids: None,
            shell_ids: None,
            part_ids: None,
            solid_results: None,
            shell_results: None,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.numnp
    }

    /// Set the 40-char run title.
    pub fn set_title(&mut self, title: &str) {
        self.title = title.to_string();
    }

    /// User node IDs (length NUMNP) written into the NARBS numbering section.
    pub fn set_node_ids(&mut self, ids: Vec<i64>) {
        self.node_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }
    /// User shell element IDs (length = number of shells).
    pub fn set_shell_ids(&mut self, ids: Vec<i64>) {
        self.shell_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }
    /// User solid element IDs (length = number of solids).
    pub fn set_solid_ids(&mut self, ids: Vec<i64>) {
        self.solid_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }
    /// User part IDs (length = number of parts / materials).
    pub fn set_part_ids(&mut self, ids: Vec<i64>) {
        self.part_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }

    /// Per-solid result block: `vars` values per solid, flat row-major
    /// `n_states * n_solids * vars` (the same raw layout [`D3plot::block_data`]
    /// returns). Sets NV3D.
    pub fn set_solid_results(&mut self, vars: usize, data: Vec<f64>) {
        self.solid_results = Some((vars, data.into_iter().map(|x| x as f32).collect()));
    }

    /// Per-shell result block: `vars` values per shell, flat row-major
    /// `n_states * n_shells * vars`. Sets NV2D.
    pub fn set_shell_results(&mut self, vars: usize, data: Vec<f64>) {
        self.shell_results = Some((vars, data.into_iter().map(|x| x as f32).collect()));
    }

    /// Add a quad/tri shell (4 one-based node ids; repeat the last for a tri).
    pub fn add_shell(&mut self, nodes: [i32; 4], part: i32) {
        self.shells
            .push([nodes[0], nodes[1], nodes[2], nodes[3], part]);
    }

    /// Add a hex/tet solid (8 one-based node ids).
    pub fn add_solid(&mut self, nodes: [i32; 8], part: i32) {
        let mut c = [0i32; 9];
        c[..8].copy_from_slice(&nodes);
        c[8] = part;
        self.solids.push(c);
    }

    /// Append a state: `time` and deformed node coordinates (`numnp*3`), plus
    /// optional velocity/acceleration (each `numnp*3` or empty). The presence of
    /// velocity/acceleration is fixed by the first state.
    pub fn add_state(
        &mut self,
        time: f64,
        disp: Vec<f64>,
        vel: Option<Vec<f64>>,
        acc: Option<Vec<f64>>,
    ) -> Result<(), D3plotError> {
        let n = self.numnp * SPATIAL_DIM;
        let check = |v: &[f64], what: &str| {
            if v.len() != n {
                Err(D3plotError::Unsupported(format!(
                    "{what} length {} != numnp*3 ({n})",
                    v.len()
                )))
            } else {
                Ok(())
            }
        };
        check(&disp, "disp")?;
        if let Some(v) = &vel {
            check(v, "vel")?;
        }
        if let Some(a) = &acc {
            check(a, "acc")?;
        }
        if self.states.is_empty() {
            self.fields = NodeFields {
                velocity: vel.is_some(),
                acceleration: acc.is_some(),
            };
        } else if vel.is_some() != self.fields.velocity || acc.is_some() != self.fields.acceleration
        {
            return Err(D3plotError::Unsupported(
                "velocity/acceleration presence must match across states".into(),
            ));
        }
        let f32v = |v: Vec<f64>| v.into_iter().map(|c| c as f32).collect();
        self.states.push(StateData {
            time: time as f32,
            disp: f32v(disp),
            vel: vel.map(f32v).unwrap_or_default(),
            acc: acc.map(f32v).unwrap_or_default(),
        });
        Ok(())
    }

    /// Serialize to a complete single-precision d3plot byte image.
    pub fn to_bytes(&self) -> Vec<u8> {
        let nel8 = self.solids.len();
        let nel4 = self.shells.len();
        let nmmat = self
            .shells
            .iter()
            .map(|s| s[4])
            .chain(self.solids.iter().map(|s| s[8]))
            .max()
            .unwrap_or(0)
            .max(0);

        // --- control block ---
        let mut words = [0i32; CONTROL_WORDS];
        // title (first TITLE_WORDS words), space-padded
        let mut title = [b' '; TITLE_BYTES];
        for (i, b) in self.title.bytes().take(TITLE_BYTES).enumerate() {
            title[i] = b;
        }
        let set = |w: &mut [i32; CONTROL_WORDS], i: usize, v: i32| w[i] = v;
        set(&mut words, word::FILETYPE, FILETYPE_D3PLOT);
        set(&mut words, word::NDIM, NDIM_STRUCTURAL); // flag word; coords are 3-D
        set(&mut words, word::NUMNP, self.numnp as i32);
        set(&mut words, word::ICODE, ICODE_LSDYNA);
        set(&mut words, word::NGLBV, 0); // no global vars
        set(&mut words, word::IU, 1);
        set(&mut words, word::IV, i32::from(self.fields.velocity));
        set(&mut words, word::IA, i32::from(self.fields.acceleration));
        let nmmat_u = nmmat.max(0) as usize;
        // NARBS numbering section size (we always emit it, with a part-id header).
        let narbs = self.numnp + nel8 + nel4 + 3 * nmmat_u + NARBS_PART_HEADER;
        let nv3d = self.solid_results.as_ref().map_or(0, |(v, _)| *v);
        let nv2d = self.shell_results.as_ref().map_or(0, |(v, _)| *v);
        set(&mut words, word::NEL8, nel8 as i32);
        set(&mut words, word::NV3D, nv3d as i32);
        set(&mut words, word::NEL4, nel4 as i32);
        set(&mut words, word::NUMMAT4, nmmat);
        set(&mut words, word::NV2D, nv2d as i32);
        set(&mut words, word::MAXINT, 1); // 1 integration point/layer
        set(&mut words, word::NARBS, narbs as i32);
        set(&mut words, word::NMMAT, nmmat);

        // Result-field flags so LS-PrePost/lasso *name* the raw element vars.
        // One layer, solver order: 6 stress, 1 plastic strain, then history.
        // ioshl1/ioshl2 are shared by solids and shells (lasso derives
        // has_solid_stress from ioshl1), so set them from whichever is present.
        if nv3d > 0 || nv2d > 0 {
            let has_stress = nv3d >= ELEM_STRESS_VARS || nv2d >= ELEM_STRESS_VARS;
            let has_pstrain = nv3d >= ELEM_BASE_VARS || nv2d >= ELEM_BASE_VARS;
            let base = ELEM_STRESS_VARS * has_stress as usize + has_pstrain as usize;
            set(
                &mut words,
                word::IOSHL1,
                if has_stress { IOSHL_PRESENT } else { 0 },
            );
            set(
                &mut words,
                word::IOSHL2,
                if has_pstrain { IOSHL_PRESENT } else { 0 },
            );
            if nv3d > 0 {
                set(&mut words, word::NEIPH, nv3d.saturating_sub(base) as i32); // solid history
            }
            if nv2d > 0 {
                set(&mut words, word::NEIPS, nv2d.saturating_sub(base) as i32); // shell history
            }
        }

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&title);
        for w in &words[TITLE_WORDS..] {
            buf.extend_from_slice(&w.to_le_bytes());
        }

        // --- geometry ---
        for &c in &self.x0 {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        for s in &self.solids {
            for &w in s {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        for s in &self.shells {
            for &w in s {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }

        // --- NARBS: arbitrary node/element/material numbering ---
        write_narbs(
            &mut buf,
            self.numnp,
            nel8,
            nel4,
            nmmat_u,
            self.node_ids.as_deref(),
            self.solid_ids.as_deref(),
            self.shell_ids.as_deref(),
            self.part_ids.as_deref(),
        );

        // States follow the exact geometry (LS-DYNA does not block-pad it).

        // --- states: time + node data (IU/IV/IA) + element data (solids, shells) ---
        let per_solid = nv3d * self.solids.len();
        let per_shell = nv2d * self.shells.len();
        for (si, st) in self.states.iter().enumerate() {
            buf.extend_from_slice(&st.time.to_le_bytes());
            for v in [&st.disp, &st.vel, &st.acc] {
                for &c in v {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
            // element results (order matches the reader: solids, then shells)
            if let Some((_, data)) = &self.solid_results {
                for &c in &data[si * per_solid..(si + 1) * per_solid] {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
            if let Some((_, data)) = &self.shell_results {
                for &c in &data[si * per_shell..(si + 1) * per_shell] {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        // end-of-file marker
        buf.extend_from_slice(&(EOF_MARKER as f32).to_le_bytes());
        buf
    }

    /// Write the d3plot to `path`.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), D3plotError> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }
}

/// Builds an interface-force (`intfor`) file: contact segments + per-state nodal
/// motion (displacement, velocity) + per-segment interface values (pressure,
/// shear, forces, gap — or the FSIFOR/ALE fixed layout).
///
/// Scope (v1): single precision, implicit numbering (NARBS=0), no global
/// variables. Round-trips through [`D3plot`]; **validate in LS-PrePost before
/// relying on it.**
pub struct IntforWriter {
    numnp: usize,
    x0: Vec<f32>,
    segments: Vec<[i32; 5]>, // 4 one-based node ids + segment id
    n_interfaces: usize,
    nwear: usize,
    npresu: usize,
    nshear: usize,
    nforce: usize,
    ngapc: usize,
    fsifor_fields: usize, // >0 ⇒ FSIFOR file with this many per-segment values
    node_ids: Option<Vec<i32>>,
    states: Vec<IntforState>,
    title: String,
}

struct IntforState {
    time: f32,
    disp: Vec<f32>, // numnp*3
    vel: Vec<f32>,  // numnp*3
    seg: Vec<f32>,  // numsg*nv2d
}

impl IntforWriter {
    /// Start from node coordinates (`numnp*3`) and the number of sliding
    /// interfaces (sets NUMMAT4 = 2 × interfaces).
    pub fn new(node_coords: Vec<f64>, n_interfaces: usize) -> Result<Self, D3plotError> {
        if node_coords.is_empty() || !node_coords.len().is_multiple_of(3) {
            return Err(D3plotError::Unsupported(
                "node_coords length must be a non-zero multiple of 3".into(),
            ));
        }
        Ok(Self {
            numnp: node_coords.len() / 3,
            x0: node_coords.iter().map(|&c| c as f32).collect(),
            segments: Vec::new(),
            n_interfaces: n_interfaces.max(1),
            nwear: 0,
            npresu: 0,
            nshear: 0,
            nforce: 0,
            ngapc: 0,
            fsifor_fields: 0,
            node_ids: None,
            states: Vec::new(),
            title: String::new(),
        })
    }

    pub fn set_title(&mut self, title: &str) {
        self.title = title.to_string();
    }

    /// User node IDs (length NUMNP) written into the NARBS numbering section.
    pub fn set_node_ids(&mut self, ids: Vec<i64>) {
        self.node_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }

    /// Add a 4-node contact segment (one-based node ids) with a segment id.
    pub fn add_segment(&mut self, nodes: [i32; 4], id: i32) {
        self.segments
            .push([nodes[0], nodes[1], nodes[2], nodes[3], id]);
    }

    /// Declare the intfor per-segment field layout (NV2D = their sum): wear,
    /// pressure, shear, force, gap. Typical: pressure 1, shear 3, force 12, gap 5.
    pub fn set_fields(
        &mut self,
        wear: usize,
        pressure: usize,
        shear: usize,
        force: usize,
        gap: usize,
    ) {
        self.nwear = wear;
        self.npresu = pressure;
        self.nshear = shear;
        self.nforce = force;
        self.ngapc = gap;
        self.fsifor_fields = 0;
    }

    /// Mark this an FSIFOR (ALE) file with `n` fixed per-segment values (NV2D is
    /// written negative). See [`FsiforField`] for the column meanings.
    pub fn set_fsifor(&mut self, n: usize) {
        self.fsifor_fields = n;
        self.nwear = 0;
        self.npresu = 0;
        self.nshear = 0;
        self.nforce = 0;
        self.ngapc = 0;
    }

    /// Values per segment in each state.
    pub fn nv2d(&self) -> usize {
        if self.fsifor_fields > 0 {
            self.fsifor_fields
        } else {
            self.nwear + self.npresu + self.nshear + self.nforce + self.ngapc
        }
    }

    /// Append a state: `time`, deformed coords `disp` (`numnp*3`), `vel`
    /// (`numnp*3`), and `segment_values` (`n_segments * nv2d`, row-major).
    pub fn add_state(
        &mut self,
        time: f64,
        disp: Vec<f64>,
        vel: Vec<f64>,
        segment_values: Vec<f64>,
    ) -> Result<(), D3plotError> {
        let n3 = self.numnp * SPATIAL_DIM;
        let need = self.segments.len() * self.nv2d();
        if disp.len() != n3 || vel.len() != n3 {
            return Err(D3plotError::Unsupported(format!(
                "disp/vel length must be numnp*3 ({n3})"
            )));
        }
        if segment_values.len() != need {
            return Err(D3plotError::Unsupported(format!(
                "segment_values length {} != n_segments*nv2d ({need})",
                segment_values.len()
            )));
        }
        let f32v = |v: Vec<f64>| v.into_iter().map(|c| c as f32).collect();
        self.states.push(IntforState {
            time: time as f32,
            disp: f32v(disp),
            vel: f32v(vel),
            seg: f32v(segment_values),
        });
        Ok(())
    }

    /// Serialize to a complete single-precision intfor byte image.
    pub fn to_bytes(&self) -> Vec<u8> {
        let numsg = self.segments.len();
        let nv2d = self.nv2d();

        let mut words = [0i32; CONTROL_WORDS];
        let mut title = [b' '; TITLE_BYTES];
        for (i, b) in self.title.bytes().take(TITLE_BYTES).enumerate() {
            title[i] = b;
        }
        let set = |w: &mut [i32; CONTROL_WORDS], i: usize, v: i32| w[i] = v;
        set(&mut words, word::FILETYPE, FILETYPE_INTFOR as i32);
        set(&mut words, word::NDIM, NDIM_STRUCTURAL);
        set(&mut words, word::NUMNP, self.numnp as i32);
        set(&mut words, word::ICODE, ICODE_LSDYNA);
        set(&mut words, word::NGLBV, 0);
        set(&mut words, word::IU, 1);
        set(&mut words, word::IV, 1);
        set(&mut words, word::NUMSG, numsg as i32);
        // Materials = the interface surfaces (slave + master per interface).
        let nmmat = 2 * self.n_interfaces;
        set(&mut words, word::NUMMAT4, nmmat as i32);
        set(&mut words, word::NMMAT, nmmat as i32);
        // NV2D is negative for FSIFOR.
        set(
            &mut words,
            word::NV2D,
            if self.fsifor_fields > 0 {
                -(nv2d as i32)
            } else {
                nv2d as i32
            },
        );
        // NARBS numbering section: numnp nodes + numsg segments (shell slot) + materials.
        let narbs = numsg + self.numnp + 3 * nmmat + NARBS_PART_HEADER;
        set(&mut words, word::NARBS, narbs as i32);
        if self.fsifor_fields == 0 {
            set(&mut words, word::NWEAR, self.nwear as i32);
            set(&mut words, word::NPRESU, self.npresu as i32);
            set(&mut words, word::NSHEAR, self.nshear as i32);
            set(&mut words, word::NFORCE, self.nforce as i32);
            set(&mut words, word::NGAPC, self.ngapc as i32);
        }

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&title);
        for w in &words[TITLE_WORDS..] {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        // geometry: node coords + segment connectivity (4 nodes + id)
        for &c in &self.x0 {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        for s in &self.segments {
            for &w in s {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        // NARBS: node IDs + segment IDs (in the shell slot) + material IDs.
        let seg_ids: Vec<i32> = self.segments.iter().map(|s| s[4]).collect();
        write_narbs(
            &mut buf,
            self.numnp,
            0,
            numsg,
            nmmat,
            self.node_ids.as_deref(),
            None,
            Some(&seg_ids),
            None,
        );

        // states: time + disp + vel + per-segment values
        for st in &self.states {
            buf.extend_from_slice(&st.time.to_le_bytes());
            for v in [&st.disp, &st.vel, &st.seg] {
                for &c in v {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        buf.extend_from_slice(&(EOF_MARKER as f32).to_le_bytes());
        buf
    }

    /// Write the intfor file to `path`.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), D3plotError> {
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }
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
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "dynars_d3plot_{nanos}_{}.bin",
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// Write a minimal single-precision d3plot the way LS-DYNA lays one out:
    /// 2 nodes, no elements, IU only; states follow the exact geometry (no
    /// block padding). State `s` moves node 1 to z = s.
    fn write_synthetic(path: &std::path::Path) {
        let numnp = 2usize;
        let mut words: Vec<i32> = vec![0; 64];
        words[15] = 4; // NDIM (flag word — coords are still 3-D)
        words[16] = numnp as i32; // NUMNP
        words[18] = 0; // NGLBV
        words[20] = 1; // IU
        // everything else (elements, narbs, maxint, extra, it/iv/ia) = 0

        let mut buf: Vec<u8> = Vec::new();
        for &w in &words {
            buf.write_i32::<LittleEndian>(w).unwrap();
        }
        // geometry: initial node coords (row-major); node0 & node1 at origin.
        let x0: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for &c in &x0 {
            buf.write_f32::<LittleEndian>(c).unwrap();
        }
        // states follow immediately: each = TIME + IU block (numnp*3 coords)
        for s in 0..2i32 {
            buf.write_f32::<LittleEndian>(s as f32).unwrap(); // time
            // node0 stays at origin; node1 moves to z = s
            let cur: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 0.0, s as f32];
            for &c in &cur {
                buf.write_f32::<LittleEndian>(c).unwrap();
            }
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

        // bulk all-states read == per-state reads concatenated
        let bulk = d.node_coordinates_all().unwrap();
        let per: Vec<f64> = (0..d.num_states())
            .flat_map(|s| d.node_coordinates(s).unwrap())
            .collect();
        assert_eq!(bulk, per);
        assert_eq!(bulk.len(), d.num_states() * d.num_nodes() * 3);

        // generic extractor: displacement block == per-state node coordinates
        let all = d.resolve_states(None).unwrap();
        let (data, dims) = d.block_data(StateBlock::Displacement, &all).unwrap();
        assert_eq!(dims, [2, 2, 3]); // (n_states, n_nodes, 3)
        match data {
            BlockArray::F32(v) => {
                assert_eq!(v.len(), 12);
                assert!((v[11] - 1.0).abs() < 1e-6); // state1, node1, z = 1
            }
            BlockArray::F64(_) => panic!("single-precision synthetic file should yield f32"),
        }
        // selective read: just the last state
        let (last, ldims) = d.block_data(StateBlock::Displacement, &[1]).unwrap();
        assert_eq!(ldims, [1, 2, 3]);
        if let BlockArray::F32(v) = last {
            assert!((v[5] - 1.0).abs() < 1e-6);
        }
        // this synthetic file is single-file → zero-copy view is available
        assert!(d.block_view(StateBlock::Displacement, &all).is_some());
        // no element blocks in this synthetic file
        assert!(d.block_data(StateBlock::Shell, &all).is_none());
        assert!(d.block_data(StateBlock::Solid, &all).is_none());
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
        let Ok(path) = std::env::var("DYNARS_TEST_D3PLOT") else {
            return;
        };
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

    #[test]
    fn writer_roundtrips_through_reader() {
        // 5 nodes, 1 quad shell, 3 states with displacement + velocity.
        let nodes: Vec<f64> = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0,
        ];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.set_title("dynars writer test");
        w.add_shell([1, 2, 3, 4], 1);
        for s in 0..3 {
            let dz = s as f64;
            let disp: Vec<f64> = nodes
                .chunks(3)
                .flat_map(|p| [p[0], p[1], p[2] + dz])
                .collect();
            let vel: Vec<f64> = vec![0.5; nodes.len()];
            w.add_state(s as f64 * 0.1, disp, Some(vel), None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.num_nodes(), 5);
        assert_eq!(d.num_states(), 3);
        for (a, b) in d.times().iter().zip([0.0, 0.1, 0.2]) {
            assert!((a - b).abs() < 1e-6);
        }
        // final state: node4 moved to z = 1.0 (initial) + 2.0 = 3.0
        let c2 = d.node_coordinates(2).unwrap();
        assert!((c2[4 * 3 + 2] - 3.0).abs() < 1e-6);
        // velocity block present and correct
        let all = d.resolve_states(None).unwrap();
        let (vel, dims) = d.block_data(StateBlock::Velocity, &all).unwrap();
        assert_eq!(dims, [3, 5, 3]);
        if let BlockArray::F32(v) = vel {
            assert!(v.iter().all(|&x| (x - 0.5).abs() < 1e-6));
        }
        // acceleration absent
        assert!(d.block_data(StateBlock::Acceleration, &all).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn writer_narbs_and_element_results_and_editor() {
        let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
        w.set_node_ids((0..8).map(|i| 100 + i).collect());
        w.set_part_ids(vec![7]);
        // 2 states × 1 solid × 5 raw result vars
        w.set_solid_results(5, (0..2 * 5).map(|i| i as f64).collect());
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        // reader sees NARBS-sized geometry + element results
        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.num_states(), 2);
        let all = d.resolve_states(None).unwrap();
        let (solid, dims) = d.block_data(StateBlock::Solid, &all).unwrap();
        assert_eq!(dims, [2, 1, 5]);
        if let BlockArray::F32(v) = solid {
            assert_eq!(v.len(), 10);
            assert!((v[9] - 9.0).abs() < 1e-6);
        }

        // editor: overwrite node coords at state 1, everything else preserved
        let orig_state0 = d.node_coordinates(0).unwrap();
        drop(d);
        let mut e = D3plotEditor::open(&p).unwrap();
        let new_coords = vec![7.0f32; 8 * 3];
        e.set_node_coordinates(1, &new_coords).unwrap();
        e.save().unwrap();

        let d2 = D3plot::open(&p).unwrap();
        let s1 = d2.node_coordinates(1).unwrap();
        assert!(s1.iter().all(|&c| (c - 7.0).abs() < 1e-6));
        // state 0 untouched
        assert_eq!(d2.node_coordinates(0).unwrap(), orig_state0);
        // element results untouched
        let (solid2, _) = d2
            .block_data(StateBlock::Solid, &d2.resolve_states(None).unwrap())
            .unwrap();
        if let BlockArray::F32(v) = solid2 {
            assert!((v[9] - 9.0).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn element_criteria_over_a_solid_part() {
        use crate::results::element;
        // 1 solid (part index 1), nv=7 (6 stress + eff plastic strain), 2 states.
        let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
        w.set_part_ids(vec![7]);
        w.set_solid_results(
            7,
            vec![
                100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, // state 0: uniaxial 100, eps 0.1
                0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.3, // state 1: pure shear 50, eps 0.3
            ],
        );
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let (data, dims, parts) = d.element_block_f64(StateBlock::Solid).unwrap();
        assert_eq!(dims, [2, 1, 7]);
        assert_eq!(parts, vec![1]); // connectivity part index
        let vm =
            element::part_max_history(&data, dims[0], dims[1], dims[2], &parts, 1, element::von_mises_stress);
        assert!((vm[0] - 100.0).abs() < 1e-2 && (vm[1] - 3.0f64.sqrt() * 50.0).abs() < 1e-2, "{vm:?}");
        let eps = element::part_max_history(
            &data, dims[0], dims[1], dims[2], &parts, 1, element::effective_plastic_strain,
        );
        assert!((eps[0] - 0.1).abs() < 1e-4 && (eps[1] - 0.3).abs() < 1e-4, "{eps:?}");
        // failure fraction at eps > 0.2: none in state 0, all in state 1
        let ff = element::part_failure_fraction_history(
            &data, dims[0], dims[1], dims[2], &parts, 1, 0.2, element::effective_plastic_strain,
        );
        assert_eq!(ff, vec![0.0, 1.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn reads_intfor_interface_force_file() {
        // Hand-build a minimal interface-force (intfor) file: one quad segment
        // (4 nodes) with per-segment values = pressure(1) + shear(3) + force(12),
        // plus nodal displacement + velocity. It is a d3plot-family file whose
        // "shell" slot holds the segments.
        let (numnp, numsg) = (4usize, 1usize);
        let (nwear, npresu, nshear, nforce, ngapc) = (0usize, 1, 3, 12, 0);
        let nv2d = nwear + npresu + nshear + nforce + ngapc; // 16
        let n_states = 2;

        let mut words = [0i32; 64];
        words[11] = 4; // FILETYPE = intfor
        words[15] = 4; // NDIM
        words[16] = numnp as i32; // NUMNP
        words[17] = 6; // ICODE
        words[20] = 1; // IU
        words[21] = 1; // IV
        words[31] = numsg as i32; // NUMSG (segments live in the shell slot)
        words[32] = 2; // NUMMAT4
        words[33] = nv2d as i32; // NV2D
        words[59] = nwear as i32;
        words[60] = npresu as i32;
        words[61] = nshear as i32;
        words[62] = nforce as i32;
        words[63] = ngapc as i32;

        let mut buf: Vec<u8> = Vec::new();
        for &w in &words {
            buf.write_i32::<LittleEndian>(w).unwrap();
        }
        // geometry: node coords + segment connectivity (4 nodes + segment id)
        let coords: [f32; 12] = [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        for &c in &coords {
            buf.write_f32::<LittleEndian>(c).unwrap();
        }
        for &w in &[1i32, 2, 3, 4, 101] {
            buf.write_i32::<LittleEndian>(w).unwrap();
        }
        // states: time + disp + vel + per-segment values
        for s in 0..n_states {
            buf.write_f32::<LittleEndian>(s as f32 * 0.1).unwrap();
            for c in coords.chunks(3) {
                buf.write_f32::<LittleEndian>(c[0]).unwrap();
                buf.write_f32::<LittleEndian>(c[1]).unwrap();
                buf.write_f32::<LittleEndian>(c[2] + s as f32 * 0.5)
                    .unwrap();
            }
            for _ in 0..numnp * 3 {
                buf.write_f32::<LittleEndian>(2.0).unwrap(); // velocity
            }
            for v in 0..nv2d {
                buf.write_f32::<LittleEndian>(v as f32 + s as f32).unwrap(); // segment values
            }
        }
        buf.write_f32::<LittleEndian>(-999999.0).unwrap();

        let p = tmp();
        std::fs::write(&p, &buf).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert!(d.is_interface_force());
        assert!(!d.is_fsifor());
        assert_eq!(d.filetype(), 4);
        assert_eq!(d.num_nodes(), numnp);
        assert_eq!(d.num_states(), n_states);
        let f = d.interface_fields();
        assert_eq!(
            (f.wear, f.pressure, f.shear, f.force, f.gap),
            (0, 1, 3, 12, 0)
        );
        // enum-based field spans: pressure at 0 (1), shear at 1 (3), force at 4 (12)
        assert_eq!(d.interface_field_span(InterfaceField::Pressure), (0, 1));
        assert_eq!(d.interface_field_span(InterfaceField::Shear), (1, 3));
        assert_eq!(d.interface_field_span(InterfaceField::Force), (4, 12));
        assert_eq!(d.interface_field_span(InterfaceField::Wear), (0, 0)); // absent

        let all = d.resolve_states(None).unwrap();
        // node velocity
        let (BlockArray::F32(vel), vdims) = d.block_data(StateBlock::Velocity, &all).unwrap()
        else {
            panic!("expected f32 velocity");
        };
        assert_eq!(vdims, [n_states, numnp, 3]);
        assert!(vel.iter().all(|&x| (x - 2.0).abs() < 1e-6));
        // per-segment interface values (the "shell" slot)
        let (BlockArray::F32(seg), sdims) = d.block_data(StateBlock::Shell, &all).unwrap() else {
            panic!("expected f32 segment data");
        };
        assert_eq!(sdims, [n_states, numsg, nv2d]);
        // state1, segment0, value index 5 = 5 + 1
        assert!((seg[numsg * nv2d + 5] - 6.0).abs() < 1e-6);
        // segment connectivity
        assert_eq!(d.shell_connectivity().0, vec![1, 2, 3, 4]);

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn intfor_writer_roundtrips_through_reader() {
        // Write an intfor with pressure(1)+shear(3)+force(12) per segment, then
        // read it back and confirm the fields land where InterfaceField says.
        let coords: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = IntforWriter::new(coords.clone(), 1).unwrap();
        w.set_title("dynars intfor demo");
        w.add_segment([1, 2, 3, 4], 501);
        w.set_fields(0, 1, 3, 12, 0); // nv2d = 16
        assert_eq!(w.nv2d(), 16);
        for s in 0..2 {
            let disp = coords.clone();
            let vel = vec![0.0; coords.len()];
            // one segment, 16 values: pressure=100+s, shear=[1,2,3], force=[10..21]
            let mut seg = vec![100.0 + s as f64, 1.0, 2.0, 3.0];
            seg.extend((0..12).map(|i| 10.0 + i as f64));
            w.add_state(s as f64 * 0.01, disp, vel, seg).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert!(d.is_interface_force());
        assert!(!d.is_fsifor());
        assert_eq!(d.num_states(), 2);
        assert_eq!(d.interface_field_span(InterfaceField::Pressure), (0, 1));
        assert_eq!(d.interface_field_span(InterfaceField::Force), (4, 12));
        // NARBS: the segment id (in the shell slot) round-trips
        assert_eq!(d.shell_ids(), vec![501]);
        assert_eq!(d.shell_connectivity().0, vec![1, 2, 3, 4]);

        let all = d.resolve_states(None).unwrap();
        let (BlockArray::F32(seg), dims) = d.block_data(StateBlock::Shell, &all).unwrap() else {
            panic!("expected f32 segment data");
        };
        assert_eq!(dims, [2, 1, 16]);
        assert!((seg[0] - 100.0).abs() < 1e-6); // state0 pressure
        assert!((seg[16] - 101.0).abs() < 1e-6); // state1 pressure
        assert!((seg[4] - 10.0).abs() < 1e-6); // state0 force[0] at offset 4
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn intfor_writer_fsifor() {
        let coords: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = IntforWriter::new(coords.clone(), 1).unwrap();
        w.add_segment([1, 2, 3, 4], 1);
        w.set_fsifor(7); // 7 fixed FSIFOR fields, NV2D negative
        w.add_state(0.0, coords.clone(), vec![0.0; 12], vec![9.0; 7])
            .unwrap();
        let p = tmp();
        w.write(&p).unwrap();
        let d = D3plot::open(&p).unwrap();
        assert!(d.is_fsifor());
        assert_eq!(d.fsifor_field_span(FsiforField::Pressure), (0, 1));
        assert_eq!(d.fsifor_field_span(FsiforField::VelocityY), (6, 1));
        assert_eq!(d.fsifor_field_span(FsiforField::VelocityZ), (7, 0)); // absent (only 7)
        let _ = std::fs::remove_file(&p);
    }
}
