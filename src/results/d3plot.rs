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
    pub const NMSPH: usize = 37; // number of SPH nodes
    pub const NARBS: usize = 39;
    pub const NELTH: usize = 40;
    pub const NV3DT: usize = 42;
    pub const IOSHL1: usize = 43;
    pub const IOSHL2: usize = 44;
    pub const IOSHL3: usize = 45;
    pub const IOSHL4: usize = 46;
    pub const IALEMAT: usize = 47; // ALE material count
    pub const NMMAT: usize = 51;
    pub const NPEFG: usize = 54; // airbag: n_airbags = npefg % 1000
    pub const NEL48: usize = 55; // 8-node shell count
    pub const IDTDT: usize = 56; // flags: node temp-gradient / residual forces / strain tensors
    pub const EXTRA: usize = 57;
    pub const NT3D: usize = 65; // solid thermal vars per solid (first extra header word past 64: 65)
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
    pub neips: usize, // extra history vars per shell integration point
    pub ioshl1: i64,  // 1000 ⇒ shell/tshell stress (6) written per layer
    pub ioshl2: i64,  // 1000 ⇒ shell/tshell effective plastic strain (1) written per layer
    pub ioshl3: i64,  // 1000 ⇒ shell force resultants (8) written at element level
    pub ioshl4: i64,  // 1000 ⇒ shell "extra" (thickness, energy: 4) at element level
    pub idtdt: i64,   // digit flags: temp-gradient / residual forces+moments / strain tensors
    pub nt3d: usize,  // solid thermal vars per solid (a thermal block before the solid results)
    pub n_rigid_shells: usize, // shells in rigid bodies: they write NO state data (NUMRBE)
    pub ialemat: usize, // ALE material count (fluid material id list in geometry)
    pub npefg: i64,   // airbag/particle flag word (n_airbags = npefg % 1000)
    pub nel48: usize, // 8-node shell count (extra connectivity, 5 words each)
    pub nmsph: usize, // number of SPH nodes
    // Geometry-walk-derived (filled by `walk_geometry` at open): the exact byte
    // offset of the node coordinate block and of the first state, plus the
    // per-state SPH/airbag/rigid-road/rigid-body counts these state tails need.
    pub geom_bytes: u64,
    pub coord_off: u64,
    pub n_sph_vars: usize,
    pub n_airbags: usize,
    pub n_particles: usize,
    pub n_airbag_state_vars: usize,   // nstgeom, per airbag per state
    pub n_particle_state_vars: usize, // nvar, per particle per state
    pub n_geom_vars: usize,           // ngeom, per airbag (geometry only)
    pub n_rigids: usize,
    pub reduced_rigid: bool,
    pub n_roads: usize,
    pub nmmat: usize, // total number of materials/parts
    // Rigid walls live in the global block after the 6 scalars + 7*nmmat part
    // block; derived from NGLBV (assumes the modern 4-vars/wall layout when the
    // tail is a multiple of 4, else 1 var/wall — force only).
    pub n_rigid_walls: usize,
    pub n_rigid_wall_vars: usize,
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

    /// Number of shell/thick-shell through-thickness integration points (layers),
    /// decoding the mdlopt sign packed into `maxint`.
    pub fn n_shell_layers(&self) -> usize {
        let m = self.maxint;
        let n = if m >= 0 {
            m
        } else if -m >= MDLOPT_ELEMENT_DELETION {
            -m - MDLOPT_ELEMENT_DELETION
        } else {
            -m
        };
        n.max(0) as usize
    }

    /// A shell layer carries the 6 stress components (`ioshl1 == 1000`).
    pub fn has_shell_stress(&self) -> bool {
        self.ioshl1 == IOSHL_PRESENT as i64
    }

    /// A shell layer carries effective plastic strain (`ioshl2 == 1000`).
    pub fn has_shell_pstrain(&self) -> bool {
        self.ioshl2 == IOSHL_PRESENT as i64
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
        // Computed by `walk_geometry` at open (full section walk in LS-DYNA order,
        // including material / SPH / airbag / rigid-body / rigid-road sections).
        self.geom_bytes
    }

    /// Non-kinematic node words that sit between displacement and velocity in the
    /// node stream: the IT thermal/mass-scaling block ([`therm_vars_for_it`]) plus
    /// the IDTDT temperature-gradient / residual force+moment block
    /// ([`idtdt_node_vars`]). Per node.
    fn node_therm_vars(&self) -> usize {
        therm_vars_for_it(self.it) + idtdt_node_vars(self.idtdt)
    }

    /// Solid thermal words per state (`NT3D` per solid), a block written before
    /// the solid results.
    fn solid_thermal_words(&self) -> usize {
        self.nt3d * self.nel8
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
    /// Shells that actually write state data: rigid-body shells (NUMRBE) are
    /// omitted from the shell result block.
    fn n_shells_with_data(&self) -> usize {
        self.nel4.saturating_sub(self.n_rigid_shells)
    }

    fn element_words(&self) -> usize {
        self.nel8 * self.nv3d
            + self.nelth * self.nv3dt
            + self.nel2 * self.nv1d
            + self.n_shells_with_data() * self.nv2d
    }

    /// Bytes per state, in LS-DYNA order: time + global vars + node data + solid
    /// thermal + element data + SPH + deletion + airbag particles + rigid road +
    /// rigid-body motion.
    fn bytes_per_state(&self) -> u64 {
        // Single source of truth: the last tail block's offset + its own size.
        // rigid_off already chains time → globals → nodes → solid-thermal →
        // elements → deletion → sph → airbag → road, so this stays in lockstep
        // with the per-block readers.
        (self.rigid_off() + self.n_rigids * self.rigid_vars()) as u64 * self.wordsize
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
        // Element blocks follow all node data, then the solid thermal block.
        let elem = base + self.node_data_words() + self.solid_thermal_words();
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
            // Rigid-body shells write no data, so the shell block is shorter. NOTE:
            // when n_rigid_shells > 0 the data index no longer equals the connectivity
            // index (rigid shells are skipped), so part-based shell reductions need a
            // material-type remap — a documented gap until we parse the IRBRTYP list.
            StateBlock::Shell => some(true, shell, self.n_shells_with_data(), self.nv2d),
        }
    }

    /// Byte offset, within a state, of the IU (deformed-coordinate) block.
    fn iu_offset_in_state(&self) -> u64 {
        (TIME_WORDS + self.nglbv) as u64 * self.wordsize
    }

    /// `(word offset within a state, words per node)` of a thermal/auxiliary node
    /// field, or `None` when absent. The node stream is displacement, temperature
    /// (±layers), heat flux, mass scaling, temperature gradient, residual force,
    /// residual moment, velocity, acceleration.
    fn node_field_spec(&self, field: NodeField) -> Option<(usize, usize)> {
        let it0 = self.it.rem_euclid(10);
        let has_temp = (1..=3).contains(&it0);
        let temp_words = if it0 == 3 { 3 } else { 1 };
        let has_flux = it0 == 2 || it0 == 3;
        let has_mass = self.it.div_euclid(10).rem_euclid(10) == 1;
        let has_grad = digit(self.idtdt, 0) == 1;
        let has_resid = digit(self.idtdt, 1) == 1;
        let n = self.numnp;
        let mut off = TIME_WORDS + self.nglbv + if self.iu != 0 { n * SPATIAL_DIM } else { 0 };
        let mut step = |present: bool, per: usize| {
            let at = off;
            if present {
                off += n * per;
            }
            (at, per)
        };
        let temp = step(has_temp, temp_words);
        let flux = step(has_flux, 3);
        let mass = step(has_mass, 1);
        let grad = step(has_grad, 1);
        let residf = step(has_resid, 3);
        let residm = step(has_resid, 3);
        match field {
            NodeField::Temperature => has_temp.then_some(temp),
            NodeField::HeatFlux => has_flux.then_some(flux),
            NodeField::MassScaling => has_mass.then_some(mass),
            NodeField::TemperatureGradient => has_grad.then_some(grad),
            NodeField::ResidualForce => has_resid.then_some(residf),
            NodeField::ResidualMoment => has_resid.then_some(residm),
        }
    }

    // Word offsets, within a state, of the tail blocks — in LS-DYNA *read* order
    // (deletion, SPH, airbag particles, rigid road, rigid-body motion), which is
    // what places the arrays. bytes_per_state sums the same set (order-independent).
    fn deletion_off(&self) -> usize {
        TIME_WORDS
            + self.nglbv
            + self.node_data_words()
            + self.solid_thermal_words()
            + self.element_words()
    }
    fn sph_off(&self) -> usize {
        self.deletion_off() + self.deletion_words()
    }
    fn sph_words(&self) -> usize {
        self.nmsph * self.n_sph_vars
    }
    fn airbag_off(&self) -> usize {
        self.sph_off() + self.sph_words()
    }
    fn airbag_words(&self) -> usize {
        self.n_airbags * self.n_airbag_state_vars + self.n_particles * self.n_particle_state_vars
    }
    fn road_off(&self) -> usize {
        self.airbag_off() + self.airbag_words()
    }
    fn rigid_off(&self) -> usize {
        self.road_off() + self.n_roads * 6
    }
    fn rigid_vars(&self) -> usize {
        if self.reduced_rigid { 12 } else { 24 }
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

/// A global (whole-model) per-state scalar/vector in the global-variables block
/// (right after the state time word). See [`D3plot::global_history`].
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "GlobalField")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalField {
    /// Total kinetic energy (1).
    KineticEnergy,
    /// Total internal energy (1).
    InternalEnergy,
    /// Total energy (1).
    TotalEnergy,
    /// Global velocity vector X (first of 3).
    VelocityX,
    /// Global velocity vector Y.
    VelocityY,
    /// Global velocity vector Z.
    VelocityZ,
}

/// Airbag/CPM per-state data from [`D3plot::airbag_state`]: the airbag chamber
/// state block and the particle state block, each with its `[rows, cols]` dims.
#[derive(Debug, Clone)]
pub struct AirbagState {
    /// Airbag chamber state, `n_airbags × nstgeom` row-major.
    pub airbag: Vec<f64>,
    /// `[n_airbags, nstgeom]`.
    pub airbag_dims: [usize; 2],
    /// Particle state, `n_particles × nvar` row-major.
    pub particle: Vec<f64>,
    /// `[n_particles, nvar]`.
    pub particle_dims: [usize; 2],
}

/// A per-part per-state scalar in the global-variables block (after the global
/// scalars). See [`D3plot::part_field_history`].
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "PartField")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartField {
    /// Internal energy per part.
    InternalEnergy,
    /// Kinetic energy per part.
    KineticEnergy,
    /// Mass per part.
    Mass,
    /// Hourglass energy per part.
    HourglassEnergy,
}

/// A per-node thermal / auxiliary state field (beyond displacement / velocity /
/// acceleration), present per the IT and IDTDT header flags. Each maps to a
/// strided slice of the node block. See [`D3plot::node_field`].
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(eq, eq_int, from_py_object, name = "NodeField")
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeField {
    /// Nodal temperature (1 value/node, or 3 through-thickness layers when IT%10==3).
    Temperature,
    /// Nodal heat flux (3/node).
    HeatFlux,
    /// Nodal mass scaling (1/node).
    MassScaling,
    /// Nodal temperature gradient (1/node).
    TemperatureGradient,
    /// Nodal residual force (3/node).
    ResidualForce,
    /// Nodal residual moment (3/node).
    ResidualMoment,
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

/// A self-describing d3plot result array: a flat, row-major `data` buffer plus
/// its `dims` (state-major for per-state blocks — `dims[0]` is the state count)
/// and, when known, `components` naming the innermost axis (e.g. the six stress
/// components). The buffer stays flat/columnar for throughput; this type only
/// labels it, so bulk callers can read [`data`](Self::data) directly while
/// convenience callers use [`state`](Self::state) / [`component`](Self::component).
///
/// It is the single currency of the d3plot results API: readers return it and
/// the writer accepts it, so a read → edit → write round-trip needs no manual
/// reshaping.
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(name = "ResultBlock", skip_from_py_object)
)]
#[derive(Debug, Clone, PartialEq)]
pub struct ResultBlock {
    data: Vec<f64>,
    dims: Vec<usize>,
    components: Vec<String>,
}

impl ResultBlock {
    /// A block with the given row-major `dims` and no component labels.
    pub fn new(dims: impl Into<Vec<usize>>, data: Vec<f64>) -> Self {
        Self {
            data,
            dims: dims.into(),
            components: Vec::new(),
        }
    }

    /// A block whose innermost axis carries the given `components` labels.
    pub fn named(
        dims: impl Into<Vec<usize>>,
        components: impl IntoIterator<Item = impl Into<String>>,
        data: Vec<f64>,
    ) -> Self {
        Self {
            data,
            dims: dims.into(),
            components: components.into_iter().map(Into::into).collect(),
        }
    }

    /// The flat, row-major data buffer.
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    /// Consume the block, yielding the flat buffer.
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }
    /// The shape, outermost axis first.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    /// Labels for the innermost axis, or empty when unlabeled.
    pub fn components(&self) -> &[String] {
        &self.components
    }
    /// Total element count of the flat buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Number of states (the outermost dimension), or 0 if the block is scalar-rank.
    pub fn n_states(&self) -> usize {
        self.dims.first().copied().unwrap_or(0)
    }

    /// The row-major slice for outermost index `i` (one state, when state-major):
    /// `dims[1..].product()` values. `None` if out of range or scalar-rank.
    pub fn state(&self, i: usize) -> Option<&[f64]> {
        if self.dims.len() < 2 {
            return None;
        }
        let stride: usize = self.dims[1..].iter().product();
        self.data.get(i * stride..(i + 1) * stride)
    }

    /// Gather a named innermost-axis component across the whole block: one value
    /// per (all-but-last-axis) entry. `None` if the name is unknown.
    pub fn component(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.components.iter().position(|c| c == name)?;
        let last = *self.dims.last()?;
        if last == 0 {
            return Some(Vec::new());
        }
        Some(self.data.iter().skip(idx).step_by(last).copied().collect())
    }
}

/// Standard component labels for a raw element result record of `vars` values:
/// the 6 stress components, effective plastic strain, then `hist_k`. Truncated or
/// extended to exactly `vars` entries so it always matches the innermost axis.
fn element_component_names(vars: usize) -> Vec<String> {
    const BASE: [&str; 7] = [
        "sigma_xx",
        "sigma_yy",
        "sigma_zz",
        "sigma_xy",
        "sigma_yz",
        "sigma_zx",
        "eff_plastic_strain",
    ];
    (0..vars)
        .map(|i| {
            BASE.get(i)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("hist_{}", i - BASE.len()))
        })
        .collect()
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

/// Family member paths for a base d3plot. For writing, `n = Some(k)` returns
/// exactly `k` contiguous names (`base`, `base01`, …). For reading, `n = None`
/// enumerates the base plus every sibling matching `<basename><digits>` present
/// in the directory, sorted by **numeric** suffix — matching LS-DYNA / lasso, so
/// non-contiguous or >99 numbering (`d3plot01, 02, 10, 22, 100`) is handled and a
/// gap does not truncate the family.
fn family_paths(base: &std::path::Path, n: Option<usize>) -> Vec<std::path::PathBuf> {
    let stem = base.to_string_lossy().into_owned();
    if let Some(k) = n {
        // Writer: contiguous names.
        let mut out = vec![base.to_path_buf()];
        let mut i = 1;
        while out.len() < k {
            let name = if i < 100 {
                format!("{stem}{i:02}")
            } else {
                format!("{stem}{i}")
            };
            out.push(std::path::PathBuf::from(name));
            i += 1;
        }
        return out;
    }

    // Reader: glob siblings `<basename><digits>` and numeric-sort.
    let mut out = vec![base.to_path_buf()];
    let (dir, basename) = match (base.parent(), base.file_name()) {
        (Some(d), Some(f)) => (
            if d.as_os_str().is_empty() { std::path::Path::new(".") } else { d },
            f.to_string_lossy().into_owned(),
        ),
        _ => return out,
    };
    let mut sibs: Vec<(u64, std::path::PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let fname = e.file_name();
            let s = fname.to_string_lossy();
            if let Some(rest) = s.strip_prefix(&basename)
                && !rest.is_empty()
                && rest.bytes().all(|b| b.is_ascii_digit())
                && let Ok(num) = rest.parse::<u64>()
            {
                sibs.push((num, e.path()));
            }
        }
    }
    sibs.sort_by_key(|(num, _)| *num);
    out.extend(sibs.into_iter().map(|(_, p)| p));
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
        for (fi, bytes) in files.iter().enumerate() {
            let start = if fi == 0 { geom } else { 0 };
            let len = bytes.len() as u64;
            let mut off = start;
            while off + bps <= len {
                let t = read_float_at(bytes, off, ws);
                // An EOF marker ends the states in THIS family member, not the whole
                // family: a geometry-only base file carries the marker right after
                // its geometry (before the part-title section), with the actual
                // states in the continuation files — so continue to the next file.
                if is_eof_marker(t) {
                    break;
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
        // Node coordinates follow the material / SPH / airbag flag sections, so use
        // the offset computed by the geometry walk (not just the header size).
        let x0 = read_floats_at(
            &files[0],
            ctrl.coord_off,
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

    /// Part titles `(id, name)` from the header part/contact title section (the
    /// EOF-marker-gated block after the geometry, ntype 90001). Empty if the file
    /// carries no title section. Titles are the fixed 18-word (72-byte) LS-DYNA
    /// part names, trimmed. Ids here are the user part ids.
    pub fn part_titles(&self) -> Vec<(i64, String)> {
        let ws = self.ctrl.wordsize;
        let wsz = ws as usize;
        let bytes: &[u8] = &self.files[0];
        let mut pos = self.ctrl.geom_bytes as usize;
        // The section exists only if an EOF marker sits right after the geometry.
        if !is_eof_marker(read_float_at(bytes, pos as u64, ws)) {
            return Vec::new();
        }
        pos += wsz; // consume the marker
        const TITLE_BYTES: usize = 18 * 4; // always 18 4-byte words, even in double
        let mut out = Vec::new();
        loop {
            let ntype = read_int_at(bytes, pos, ws);
            match ntype {
                90000 => pos += wsz + TITLE_BYTES, // single title2
                90001 | 90002 | 90020 => {
                    pos += wsz;
                    let count = read_int_at(bytes, pos, ws).max(0) as usize;
                    pos += wsz;
                    for _ in 0..count {
                        let id = read_int_at(bytes, pos, ws);
                        let tb = bytes.get(pos + wsz..pos + wsz + TITLE_BYTES).unwrap_or(&[]);
                        let name = String::from_utf8_lossy(tb).trim().to_string();
                        if ntype == 90001 {
                            out.push((id, name));
                        }
                        pos += wsz + TITLE_BYTES;
                    }
                }
                90100 => {
                    pos += wsz;
                    let nline = read_int_at(bytes, pos, ws).max(0) as usize;
                    pos += wsz + nline * 20 * 4; // d3prop keywords, 20 4-byte words each
                }
                _ => break,
            }
        }
        out
    }

    /// A global per-state scalar (energy / velocity component) as a time history:
    /// one value per state, in state order. The global-variables block holds
    /// kinetic, internal, total energy, then the 3-component global velocity; this
    /// returns `None` if the requested field isn't present (`nglbv` too small).
    pub fn global_history(&self, field: GlobalField) -> Option<Vec<f64>> {
        let idx = match field {
            GlobalField::KineticEnergy => 0,
            GlobalField::InternalEnergy => 1,
            GlobalField::TotalEnergy => 2,
            GlobalField::VelocityX => 3,
            GlobalField::VelocityY => 4,
            GlobalField::VelocityZ => 5,
        };
        if idx >= self.ctrl.nglbv {
            return None;
        }
        let off_words = TIME_WORDS + idx;
        let ws = self.ctrl.wordsize;
        Some(
            self.states
                .iter()
                .map(|loc| {
                    read_float_at(&self.files[loc.file], loc.offset + off_words as u64 * ws, ws)
                })
                .collect(),
        )
    }

    /// A per-part per-state scalar as a dense `(n_states, n_parts)` row-major
    /// matrix (row = state). The part block follows the 6 global scalars in the
    /// global-variables section, laid out internal-energy, kinetic-energy,
    /// velocity(3), mass, hourglass-energy — each `n_parts` wide. `None` if the
    /// global block is too small to contain it.
    pub fn part_field_history(&self, field: PartField) -> Option<ResultBlock> {
        const GLOBAL_SCALARS: usize = 6; // kinetic, internal, total, vel x/y/z
        let np = self.ctrl.nmmat;
        if np == 0 {
            return None;
        }
        // Offset of this field within the per-part block (in units of n_parts).
        let (field_words_before, width) = match field {
            PartField::InternalEnergy => (0, np),
            PartField::KineticEnergy => (np, np),
            // velocity occupies 3*np between kinetic and mass
            PartField::Mass => (2 * np + 3 * np, np),
            PartField::HourglassEnergy => (3 * np + 3 * np, np),
        };
        let start = GLOBAL_SCALARS + field_words_before;
        if start + width > self.ctrl.nglbv {
            return None;
        }
        let off_words = TIME_WORDS + start;
        let ws = self.ctrl.wordsize;
        let mut out = vec![0.0f64; self.states.len() * np];
        for (s, loc) in self.states.iter().enumerate() {
            let base = loc.offset + off_words as u64 * ws;
            let row = read_floats_at(&self.files[loc.file], base, np, ws).ok()?;
            out[s * np..(s + 1) * np].copy_from_slice(&row);
        }
        Some(ResultBlock::new([self.states.len(), np], out))
    }

    /// Element deletion ("is alive") flags for a block at `state`: one value per
    /// element (0 = deleted). The deletion block sits at the end of the state, in
    /// the order solid, thick-shell, shell, beam. `None` if the file carries no
    /// element deletion data (mdlopt != 2) or the block is empty.
    pub fn element_alive(&self, block: StateBlock, state: usize) -> Option<Vec<f64>> {
        if self.ctrl.mdlopt() != 2 {
            return None; // node deletion (1) or none (0) — no per-element flags
        }
        let (n_solid, n_tshell, n_shell, n_beam) =
            (self.ctrl.nel8, self.ctrl.nelth, self.ctrl.n_shells_with_data(), self.ctrl.nel2);
        let (before, count) = match block {
            StateBlock::Solid => (0, n_solid),
            StateBlock::ThickShell => (n_solid, n_tshell),
            StateBlock::Shell => (n_solid + n_tshell, n_shell),
            StateBlock::Beam => (n_solid + n_tshell + n_shell, n_beam),
            _ => return None,
        };
        if count == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        let off_words = self.ctrl.deletion_off() + before;
        let base = loc.offset + off_words as u64 * self.ctrl.wordsize;
        read_floats_at(&self.files[loc.file], base, count, self.ctrl.wordsize).ok()
    }

    /// Node deletion ("is alive") flags for `state`: one value per node (0 =
    /// deleted). Present only when the file carries **node** deletion (mdlopt 1);
    /// returns `None` for element deletion (mdlopt 2) or no deletion data. The
    /// block sits at the end of the state, before any SPH/airbag/rigid tail.
    pub fn node_alive(&self, state: usize) -> Option<Vec<f64>> {
        if self.ctrl.mdlopt() != 1 {
            return None;
        }
        let loc = self.states.get(state)?;
        let base = loc.offset + self.ctrl.deletion_off() as u64 * self.ctrl.wordsize;
        read_floats_at(&self.files[loc.file], base, self.ctrl.numnp, self.ctrl.wordsize).ok()
    }

    /// **SPH** per-state block: `NMSPH × n_sph_vars` row-major (raw solver words;
    /// the per-particle layout is set by the ISPHFG flags — typically material,
    /// then radius/pressure/stress(6)/plastic-strain/density/energy/…). `None` if
    /// the file has no SPH nodes or the state is out of range.
    ///
    /// ⚠ Not yet validated against a real SPH d3plot (no such test file available);
    /// the state *sizing* is correct (files with SPH step right), but treat the
    /// per-variable interpretation as provisional.
    pub fn sph_state(&self, state: usize) -> Option<ResultBlock> {
        if self.ctrl.nmsph == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        let base = loc.offset + self.ctrl.sph_off() as u64 * self.ctrl.wordsize;
        let v = read_floats_at(&self.files[loc.file], base, self.ctrl.sph_words(), self.ctrl.wordsize).ok()?;
        Some(ResultBlock::new([self.ctrl.nmsph, self.ctrl.n_sph_vars], v))
    }

    /// **Airbag/CPM** per-state data (see [`AirbagState`]): airbag chamber state
    /// (`n_airbags × nstgeom`) plus particle state (`n_particles × nvar`), both
    /// row-major. `None` if the file has no airbag particle data.
    ///
    /// ⚠ Not yet validated against a real airbag d3plot; sizing is correct, the
    /// per-variable interpretation is provisional.
    pub fn airbag_state(&self, state: usize) -> Option<AirbagState> {
        if self.ctrl.n_airbags == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        let ws = self.ctrl.wordsize;
        let a_off = loc.offset + self.ctrl.airbag_off() as u64 * ws;
        let a_n = self.ctrl.n_airbags * self.ctrl.n_airbag_state_vars;
        let airbag = read_floats_at(&self.files[loc.file], a_off, a_n, ws).ok()?;
        let p_off = a_off + a_n as u64 * ws;
        let p_n = self.ctrl.n_particles * self.ctrl.n_particle_state_vars;
        let particle = read_floats_at(&self.files[loc.file], p_off, p_n, ws).ok()?;
        Some(AirbagState {
            airbag,
            airbag_dims: [self.ctrl.n_airbags, self.ctrl.n_airbag_state_vars],
            particle,
            particle_dims: [self.ctrl.n_particles, self.ctrl.n_particle_state_vars],
        })
    }

    /// **Rigid-body motion** per-state block: `n_rigids × k` row-major, where `k`
    /// is 12 (reduced: coordinates + 3×3 rotation) or 24 (adds translational and
    /// rotational velocity/acceleration). `None` if the file has no rigid-body data.
    ///
    /// ⚠ Not yet validated against a real rigid-body d3plot; sizing is correct, the
    /// per-variable interpretation is provisional.
    pub fn rigid_body_motion(&self, state: usize) -> Option<ResultBlock> {
        if self.ctrl.n_rigids == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        let base = loc.offset + self.ctrl.rigid_off() as u64 * self.ctrl.wordsize;
        let k = self.ctrl.rigid_vars();
        let v = read_floats_at(&self.files[loc.file], base, self.ctrl.n_rigids * k, self.ctrl.wordsize).ok()?;
        Some(ResultBlock::new([self.ctrl.n_rigids, k], v))
    }

    /// **Rigid-road** per-state motion block: `n_roads × 6` row-major. `None` if
    /// the file has no rigid road surfaces.
    ///
    /// ⚠ Not yet validated against a real rigid-road d3plot; sizing is correct.
    pub fn rigid_road_state(&self, state: usize) -> Option<ResultBlock> {
        if self.ctrl.n_roads == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        let base = loc.offset + self.ctrl.road_off() as u64 * self.ctrl.wordsize;
        let v = read_floats_at(&self.files[loc.file], base, self.ctrl.n_roads * 6, self.ctrl.wordsize).ok()?;
        Some(ResultBlock::new([self.ctrl.n_roads, 6], v))
    }

    /// Rigid-wall per-state data from the global-variables block: the force per
    /// wall (`ResultBlock` of dims `[n_walls]`), or `None` if the file has none.
    /// See [`rigid_wall_position`](Self::rigid_wall_position). Assumes the modern
    /// 971 layout (4 vars/wall) when the global tail is a multiple of 4.
    pub fn rigid_wall_force(&self, state: usize) -> Option<ResultBlock> {
        let (nw, base) = self.rigid_wall_layout(state)?;
        let v = read_floats_at(&self.files[self.states[state].file], base, nw, self.ctrl.wordsize).ok()?;
        Some(ResultBlock::new([nw], v))
    }

    /// Rigid-wall position per wall (`ResultBlock` of dims `[n_walls, 3]`), or
    /// `None` if the file carries force only (1 var/wall) or has no rigid walls.
    pub fn rigid_wall_position(&self, state: usize) -> Option<ResultBlock> {
        let (nw, base) = self.rigid_wall_layout(state)?;
        let ws = self.ctrl.wordsize;
        // Positions (if present) follow the force block; 4 vars/wall total.
        if self.ctrl.n_rigid_walls == 0 || self.ctrl.n_rigid_wall_vars != 4 {
            return None;
        }
        let pos_base = base + nw as u64 * ws;
        let v = read_floats_at(&self.files[self.states[state].file], pos_base, 3 * nw, ws).ok()?;
        Some(ResultBlock::named([nw, 3], ["x", "y", "z"], v))
    }

    /// `(n_walls, byte offset of the force block)` for `state`, or `None` if the
    /// file has no rigid walls / the state is out of range.
    fn rigid_wall_layout(&self, state: usize) -> Option<(usize, u64)> {
        let nw = self.ctrl.n_rigid_walls;
        if nw == 0 {
            return None;
        }
        let loc = self.states.get(state)?;
        // force block sits after the 6 global scalars + the 7*nmmat part block.
        let off_words = TIME_WORDS + 6 + 7 * self.ctrl.nmmat;
        Some((nw, loc.offset + off_words as u64 * self.ctrl.wordsize))
    }

    /// A thermal/auxiliary per-node field at `state` as a [`ResultBlock`] of dims
    /// `[NUMNP, k]`, where `k` is the per-node width (1, or 3 for vectors /
    /// temperature layers; 3-vectors are labeled `x`/`y`/`z`). `None` if the field
    /// is absent (per IT/IDTDT) or the state is out of range.
    pub fn node_field(&self, field: NodeField, state: usize) -> Option<ResultBlock> {
        let (off_words, per) = self.ctrl.node_field_spec(field)?;
        let loc = self.states.get(state)?;
        let byte = loc.offset + off_words as u64 * self.ctrl.wordsize;
        let v =
            read_floats_at(&self.files[loc.file], byte, self.ctrl.numnp * per, self.ctrl.wordsize)
                .ok()?;
        let block = if per == 3 {
            ResultBlock::named([self.ctrl.numnp, per], ["x", "y", "z"], v)
        } else {
            ResultBlock::new([self.ctrl.numnp, per], v)
        };
        Some(block)
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
    /// results — the layout differences live entirely in `Control::block_spec`.
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
    ///
    /// ⚠ Materializes the whole block as `f64` (`n_states·n_elem·nv·8` bytes) — fine
    /// for small/medium models, but for tens of millions of elements use the
    /// streaming [`part_max_history`](Self::part_max_history) /
    /// [`part_failure_fraction_history`](Self::part_failure_fraction_history)
    /// instead, which reduce straight off the memory map.
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

    /// A per-element result `block` for the given `states` as a [`ResultBlock`]
    /// shaped `[n_states, n_elem, vars]`, with the innermost axis labeled
    /// `sigma_xx…sigma_zx`, `eff_plastic_strain`, then `hist_k`. The ergonomic,
    /// self-describing form of [`block_data`](Self::block_data).
    pub fn element_results(&self, block: StateBlock, states: &[usize]) -> Option<ResultBlock> {
        let (arr, dims) = self.block_data(block, states)?;
        Some(ResultBlock::named(
            dims.to_vec(),
            element_component_names(dims[2]),
            arr.to_f64(),
        ))
    }

    /// All-states solid result block as a [`ResultBlock`] (`[n_states, n_solids,
    /// nv3d]`). See [`element_results`](Self::element_results).
    pub fn solid_results(&self) -> Option<ResultBlock> {
        self.element_results(StateBlock::Solid, &self.all_states())
    }
    /// All-states shell result block as a [`ResultBlock`] (`[n_states, n_shells, nv2d]`).
    pub fn shell_results(&self) -> Option<ResultBlock> {
        self.element_results(StateBlock::Shell, &self.all_states())
    }
    /// All-states beam result block as a [`ResultBlock`] (`[n_states, n_beams, nv1d]`).
    pub fn beam_results(&self) -> Option<ResultBlock> {
        self.element_results(StateBlock::Beam, &self.all_states())
    }
    /// All-states thick-shell result block as a [`ResultBlock`] (`[n_states, n_tshells, nv3dt]`).
    pub fn tshell_results(&self) -> Option<ResultBlock> {
        self.element_results(StateBlock::ThickShell, &self.all_states())
    }

    /// `0..n_states` — the default selection for the whole-history accessors.
    fn all_states(&self) -> Vec<usize> {
        (0..self.states.len()).collect()
    }

    /// Column indices of `part`'s elements for a `Solid`/`Shell` block.
    fn part_element_indices(&self, block: StateBlock, part: i64) -> Option<Vec<usize>> {
        let parts = match block {
            StateBlock::Solid => self.solid_connectivity().1,
            StateBlock::Shell => self.shell_connectivity().1,
            _ => return None,
        };
        Some((0..parts.len()).filter(|&e| parts[e] == part).collect())
    }

    /// **Streaming** max of `quantity` over `part`'s elements of `block`, per state
    /// — read directly from the memory map (f32 or f64), **without materializing**
    /// the block, and parallelized across states. Scales to tens of millions of
    /// elements (memory = OS page cache + `O(n_states)`). `quantity` receives one
    /// element's `nv` result words (f32 promoted to f64); use
    /// [`element::von_mises_stress`](super::element::von_mises_stress) etc.
    pub fn part_max_history(
        &self,
        block: StateBlock,
        part: i64,
        quantity: impl Fn(&[f64]) -> f64 + Sync,
    ) -> Option<Vec<f64>> {
        let (off_words, _count, vars) = self.ctrl.block_spec(block)?;
        let idx = self.part_element_indices(block, part)?;
        let (ws, byte_off) = (self.ctrl.wordsize as usize, off_words * self.ctrl.wordsize as usize);
        Some(
            (0..self.states.len())
                .into_par_iter()
                .map(|s| {
                    let bytes: &[u8] = &self.files[self.states[s].file];
                    let base = self.states[s].offset as usize + byte_off;
                    let mut buf = vec![0.0f64; vars];
                    idx.iter().fold(0.0_f64, |m, &e| {
                        if read_element(bytes, base, e, vars, ws, &mut buf) {
                            m.max(quantity(&buf))
                        } else {
                            m
                        }
                    })
                })
                .collect(),
        )
    }

    /// **Streaming** fraction (0–1) of `part`'s elements whose `quantity` exceeds
    /// `threshold`, per state — same memory-map streaming + state parallelism as
    /// [`part_max_history`](Self::part_max_history).
    pub fn part_failure_fraction_history(
        &self,
        block: StateBlock,
        part: i64,
        threshold: f64,
        quantity: impl Fn(&[f64]) -> f64 + Sync,
    ) -> Option<Vec<f64>> {
        let (off_words, _count, vars) = self.ctrl.block_spec(block)?;
        let idx = self.part_element_indices(block, part)?;
        if idx.is_empty() {
            return Some(vec![0.0; self.states.len()]);
        }
        let inv = 1.0 / idx.len() as f64;
        let (ws, byte_off) = (self.ctrl.wordsize as usize, off_words * self.ctrl.wordsize as usize);
        Some(
            (0..self.states.len())
                .into_par_iter()
                .map(|s| {
                    let bytes: &[u8] = &self.files[self.states[s].file];
                    let base = self.states[s].offset as usize + byte_off;
                    let mut buf = vec![0.0f64; vars];
                    idx.iter()
                        .filter(|&&e| {
                            read_element(bytes, base, e, vars, ws, &mut buf)
                                && quantity(&buf) > threshold
                        })
                        .count() as f64
                        * inv
                })
                .collect(),
        )
    }

    /// Per state, **which** of `part`'s elements maximizes `quantity`, and that
    /// value: `(block-order element index, value)`. Same streaming/mmap +
    /// state-parallel shape as [`part_max_history`](Self::part_max_history) —
    /// `O(n_states)` memory, no per-element histories materialized. Locate the
    /// critical element with this, then loop [`element_result`](Self::element_result)
    /// over states for its full record. `None` if the block or part is empty.
    pub fn part_argmax_history(
        &self,
        block: StateBlock,
        part: i64,
        quantity: impl Fn(&[f64]) -> f64 + Sync,
    ) -> Option<Vec<(usize, f64)>> {
        let (off_words, _count, vars) = self.ctrl.block_spec(block)?;
        let idx = self.part_element_indices(block, part)?;
        if idx.is_empty() {
            return None;
        }
        let (ws, byte_off) = (self.ctrl.wordsize as usize, off_words * self.ctrl.wordsize as usize);
        Some(
            (0..self.states.len())
                .into_par_iter()
                .map(|s| {
                    let bytes: &[u8] = &self.files[self.states[s].file];
                    let base = self.states[s].offset as usize + byte_off;
                    let mut buf = vec![0.0f64; vars];
                    idx.iter().fold((idx[0], f64::NEG_INFINITY), |best, &e| {
                        if read_element(bytes, base, e, vars, ws, &mut buf) {
                            let v = quantity(&buf);
                            if v > best.1 {
                                return (e, v);
                            }
                        }
                        best
                    })
                })
                .collect(),
        )
    }

    /// Every element of `part`, its `quantity` over **all states**: a dense
    /// `(n_states, n_part_elems)` row-major matrix (row `s` = every element at
    /// state `s`; column `e` = element `e`'s history). Also returns the block-order
    /// element indices, so column `e` ↔ `indices[e]` (feed to
    /// [`element_result`](Self::element_result) for that element's raw record).
    ///
    /// Filled in one parallel streaming pass off the memory map (rayon over states,
    /// each state writes one contiguous row — no contention, no full-block copy).
    /// Unlike the scalar reductions this **materializes** the matrix, so memory is
    /// `n_states · n_part_elems · 8` bytes — bounded by the *part* size, but that
    /// can still be GB for a multi-million-element part. Returns `None` if the
    /// block is absent.
    pub fn part_element_history(
        &self,
        block: StateBlock,
        part: i64,
        quantity: impl Fn(&[f64]) -> f64 + Sync,
    ) -> Option<(Vec<f64>, [usize; 2], Vec<usize>)> {
        let (off_words, _count, vars) = self.ctrl.block_spec(block)?;
        let idx = self.part_element_indices(block, part)?;
        let (ns, ne) = (self.states.len(), idx.len());
        let (ws, byte_off) = (self.ctrl.wordsize as usize, off_words * self.ctrl.wordsize as usize);
        let mut out = vec![0.0f64; ns * ne];
        out.par_chunks_mut(ne.max(1)).enumerate().for_each(|(s, row)| {
            let loc = &self.states[s];
            let bytes: &[u8] = &self.files[loc.file];
            let base = loc.offset as usize + byte_off;
            let mut buf = vec![0.0f64; vars];
            for (col, &e) in idx.iter().enumerate() {
                if read_element(bytes, base, e, vars, ws, &mut buf) {
                    row[col] = quantity(&buf);
                }
            }
        });
        Some((out, [ns, ne], idx))
    }

    /// The packed result record (all `vars` words) for a **single element** at one
    /// `state` — O(1) random access straight off the memory map: no scan, no other
    /// element or state is touched (only the page(s) holding this element fault
    /// in). `elem` is the element's 0-based position within the block (file
    /// order), *not* its user element id. Returns `None` if the block is absent or
    /// an index is out of range. Feed the result to
    /// [`element::von_mises_stress`](super::element) etc. to derive a quantity.
    pub fn element_result(&self, block: StateBlock, state: usize, elem: usize) -> Option<Vec<f64>> {
        let (off_words, count, vars) = self.ctrl.block_spec(block)?;
        if state >= self.states.len() || elem >= count {
            return None;
        }
        let ws = self.ctrl.wordsize as usize;
        let loc = &self.states[state];
        let base = loc.offset as usize + off_words * ws;
        let mut buf = vec![0.0f64; vars];
        read_element(&self.files[loc.file], base, elem, vars, ws, &mut buf).then_some(buf)
    }

    /// Through-thickness layer layout of the shell result block, for reading a
    /// shell record (from [`element_result`](Self::element_result) or the
    /// reductions) with [`element::shell_von_mises`](super::element::shell_von_mises)
    /// / [`shell_plastic_strain`](super::element::shell_plastic_strain) /
    /// [`ShellLayout::resultants`](super::element::ShellLayout::resultants). Feed a
    /// closure to any reduction, e.g. `|rec| element::shell_von_mises(rec,
    /// &layout, LayerSelect::Max)`.
    pub fn shell_layout(&self) -> super::element::ShellLayout {
        let c = &self.ctrl;
        let (has_stress, has_pstrain) = (c.has_shell_stress(), c.has_shell_pstrain());
        super::element::ShellLayout {
            n_layers: c.n_shell_layers(),
            stride: 6 * has_stress as usize + has_pstrain as usize + c.neips,
            has_stress,
            has_pstrain,
            has_forces: c.ioshl3 == IOSHL_PRESENT as i64,
            has_extra: c.ioshl4 == IOSHL_PRESENT as i64,
        }
    }
}

/// Read element `e`'s `vars` result words at byte `base` (state block start) into
/// `buf` as f64 (promoting f32). `false` if the read runs past the map.
#[inline]
fn read_element(bytes: &[u8], base: usize, e: usize, vars: usize, ws: usize, buf: &mut [f64]) -> bool {
    let eb = base + e * vars * ws;
    if eb + vars * ws > bytes.len() {
        return false;
    }
    if ws == 4 {
        for (k, b) in buf.iter_mut().enumerate().take(vars) {
            let o = eb + k * 4;
            *b = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
        }
    } else {
        for (k, b) in buf.iter_mut().enumerate().take(vars) {
            let o = eb + k * 8;
            *b = f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        }
    }
    true
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
/// Append one little-endian float as a `ws`-byte word (f32 for ws==4, f64 for 8).
fn push_word_f(buf: &mut Vec<u8>, v: f64, ws: usize) {
    if ws == 8 {
        buf.extend_from_slice(&v.to_le_bytes());
    } else {
        buf.extend_from_slice(&(v as f32).to_le_bytes());
    }
}
/// Append one little-endian integer as a `ws`-byte word (i32 for ws==4, i64 for 8).
fn push_word_i(buf: &mut Vec<u8>, v: i64, ws: usize) {
    if ws == 8 {
        buf.extend_from_slice(&v.to_le_bytes());
    } else {
        buf.extend_from_slice(&(v as i32).to_le_bytes());
    }
}
/// Append a slice of floats as `ws`-byte words.
fn push_word_fs(buf: &mut Vec<u8>, s: &[f64], ws: usize) {
    for &c in s {
        push_word_f(buf, c, ws);
    }
}
/// Append state `si`'s `count`-wide slice of a whole-history optional block,
/// filling `fill` when the block is absent (0.0 for results, 1.0 for "alive").
fn push_opt_block(
    buf: &mut Vec<u8>,
    opt: &Option<Vec<f64>>,
    si: usize,
    count: usize,
    fill: f64,
    ws: usize,
) {
    match opt {
        Some(d) => push_word_fs(buf, &d[si * count..(si + 1) * count], ws),
        None => {
            for _ in 0..count {
                push_word_f(buf, fill, ws);
            }
        }
    }
}

/// NARBS arbitrary node/element/material numbering section. IDs are laid out
/// nodes, solids, beams, shells, thick-shells, then materials — matching the
/// skips the [`D3plot`] reader computes. Returns the number of words written
/// (== the NARBS header value).
#[allow(clippy::too_many_arguments)]
fn write_narbs(
    buf: &mut Vec<u8>,
    numnp: usize,
    nel8: usize,
    nel2: usize,
    nel4: usize,
    nelth: usize,
    nmmat: usize,
    node_ids: Option<&[i32]>,
    solid_ids: Option<&[i32]>,
    beam_ids: Option<&[i32]>,
    shell_ids: Option<&[i32]>,
    tshell_ids: Option<&[i32]>,
    part_ids: Option<&[i32]>,
    ws: usize,
) -> usize {
    let put = |buf: &mut Vec<u8>, v: i32| push_word_i(buf, v as i64, ws);
    let seq = |n: usize| -> Vec<i32> { (1..=n as i32).collect() };
    let (nel8i, nel2i, nel4i, nelthi, nmmi) =
        (nel8 as i32, nel2 as i32, nel4 as i32, nelth as i32, nmmat as i32);
    // numbering header (part-id form, 16 words). Pointers are 1-based offsets to
    // each id block in node, solid, beam, shell, tshell order.
    put(buf, -1); // NSORT (negative: material numbering present)
    let nsrh = 1 + numnp as i32; // solids
    put(buf, nsrh);
    let nsrb = nsrh + nel8i; // beams
    put(buf, nsrb);
    let nsrs = nsrb + nel2i; // shells
    put(buf, nsrs);
    let nsrt = nsrs + nel4i; // thick shells
    put(buf, nsrt);
    put(buf, numnp as i32); // NSORTD
    put(buf, nel8i); // NSRHD
    put(buf, nel2i); // NSRBD
    put(buf, nel4i); // NSRSD
    put(buf, nelthi); // NSRTD
    let nsrmu = nsrt + nelthi;
    let nsrma = nsrmu + nmmi;
    let nsrmp = nsrma + nmmi;
    put(buf, nsrma);
    put(buf, nsrmu);
    put(buf, nsrmp);
    put(buf, nmmi);
    put(buf, 0); // NUMRBS
    put(buf, nmmi);
    // ID arrays, in the same order the reader skips them.
    for &v in &node_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(numnp)) {
        put(buf, v);
    }
    for &v in &solid_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nel8)) {
        put(buf, v);
    }
    for &v in &beam_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nel2)) {
        put(buf, v);
    }
    for &v in &shell_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nel4)) {
        put(buf, v);
    }
    for &v in &tshell_ids.map(<[i32]>::to_vec).unwrap_or_else(|| seq(nelth)) {
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
    numnp + nel8 + nel2 + nel4 + nelth + 3 * nmmat + NARBS_PART_HEADER
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
/// Node thermal words per node from the IT header flag (LS-DYNA / lasso).
/// Ones digit: 1 = nodal temperature (1); 2 = temperature (1) + heat flux (3);
/// 3 = temperature layers (3) + heat flux (3). Tens digit == 1 adds nodal mass
/// scaling (1).
fn therm_vars_for_it(it: i64) -> usize {
    let temp_flux = match it.rem_euclid(IT_ENCODING_BASE) {
        1 => 1,
        2 => 1 + 3,
        3 => 3 + 3,
        _ => 0,
    };
    let mass = usize::from(it.div_euclid(IT_ENCODING_BASE).rem_euclid(IT_ENCODING_BASE) == 1);
    temp_flux + mass
}

/// Decimal digit `n` (0 = ones) of `x` (matches lasso's `get_digit`).
fn digit(x: i64, n: u32) -> i64 {
    x.div_euclid(10i64.pow(n)).rem_euclid(10)
}

/// Extra node words from the IDTDT flag: temperature gradient (digit 0 → 1) and
/// residual forces + moments (digit 1 → 3 + 3). Per node, sits between the IT
/// thermal block and velocity in the node stream.
fn idtdt_node_vars(idtdt: i64) -> usize {
    let grad = usize::from(digit(idtdt, 0) == 1);
    let residual = if digit(idtdt, 1) == 1 { 3 + 3 } else { 0 };
    grad + residual
}

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

    let mut ctrl = Control {
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
        neips: geti(word::NEIPS)?.max(0) as usize,
        ioshl1: geti(word::IOSHL1)?,
        ioshl2: geti(word::IOSHL2)?,
        ioshl3: geti(word::IOSHL3)?,
        ioshl4: geti(word::IOSHL4)?,
        idtdt: geti(word::IDTDT)?,
        // NT3D lives in the extra header words (past 64); present only when EXTRA
        // is large enough to reach word 65.
        nt3d: if geti(word::EXTRA)?.max(0) as usize > word::NT3D - CONTROL_WORDS {
            geti(word::NT3D)?.max(0) as usize
        } else {
            0
        },
        // The material-type section (present when NDIM is 5/7) leads the geometry;
        // its first word is NUMRBE, the count of shells belonging to rigid bodies,
        // which write no state data. Read it directly at the first geometry word.
        n_rigid_shells: {
            let ndim = geti(word::NDIM)?;
            if ndim == 5 || ndim == 7 {
                let extra = geti(word::EXTRA)?.max(0) as usize;
                let pos = (CONTROL_WORDS + extra) * wordsize as usize;
                read_i(pos, wordsize).unwrap_or(0).max(0) as usize
            } else {
                0
            }
        },
        ialemat: geti(word::IALEMAT)?.max(0) as usize,
        npefg: geti(word::NPEFG)?,
        nel48: geti(word::NEL48)?.max(0) as usize,
        nmsph: geti(word::NMSPH)?.max(0) as usize,
        // Filled by walk_geometry below (needs the whole geometry, not just the header).
        geom_bytes: 0,
        coord_off: 0,
        n_sph_vars: 0,
        n_airbags: 0,
        n_particles: 0,
        n_airbag_state_vars: 0,
        n_particle_state_vars: 0,
        n_geom_vars: 0,
        n_rigids: 0,
        reduced_rigid: false,
        n_roads: 0,
        narbs: geti(word::NARBS)?.max(0) as usize,
        nelth: geti(word::NELTH)?.max(0) as usize,
        nv3dt: geti(word::NV3DT)?.max(0) as usize,
        nmmat: geti(word::NMMAT)?.max(0) as usize,
        n_rigid_walls: 0,     // derived below from NGLBV
        n_rigid_wall_vars: 0, // derived below from NGLBV
        extra: geti(word::EXTRA)?.max(0) as usize,
        filetype: geti(word::FILETYPE)?,
        fsifor: geti(word::NV2D)? < 0,
        nwear: geti(word::NWEAR)?.max(0) as usize,
        npresu: geti(word::NPRESU)?.max(0) as usize,
        nshear: geti(word::NSHEAR)?.max(0) as usize,
        nforce: geti(word::NFORCE)?.max(0) as usize,
        ngapc: geti(word::NGAPC)?.max(0) as usize,
    };
    // Rigid walls occupy the NGLBV tail after the 6 global scalars + 7*nmmat part
    // block. Their per-wall width isn't in the header, so infer the modern 971
    // layout (4 vars/wall) when the tail divides by 4, else force-only (1).
    let rw_tail = ctrl.nglbv.saturating_sub(6 + 7 * ctrl.nmmat);
    ctrl.n_rigid_wall_vars = if rw_tail == 0 {
        0
    } else if rw_tail.is_multiple_of(4) {
        4
    } else {
        1
    };
    ctrl.n_rigid_walls = rw_tail.checked_div(ctrl.n_rigid_wall_vars).unwrap_or(0);

    walk_geometry(&mut ctrl, bytes);
    Ok(ctrl)
}

/// Read one integer word at byte offset `off` (0 past the buffer end).
fn read_int_at(bytes: &[u8], off: usize, ws: u64) -> i64 {
    match ws {
        4 => bytes
            .get(off..off + 4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .unwrap_or(0),
        _ => bytes
            .get(off..off + 8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0),
    }
}

/// Walk the geometry section in LS-DYNA order (matching lasso's `_parse_geometry`),
/// recording the node-coordinate offset, the total size (= first state offset), and
/// the SPH/airbag/rigid counts the per-state tails need. Sections whose flags are
/// off contribute nothing, so plain files reduce to header + coords + connectivity
/// + numbering exactly as before.
fn walk_geometry(c: &mut Control, bytes: &[u8]) {
    let ws = c.wordsize;
    let wsz = ws as usize;
    let word = |p: usize| read_int_at(bytes, p, ws);
    let mattyp = c.ndim == 5 || c.ndim == 7;
    let has_rigid_body = c.ndim == 8 || c.ndim == 9;
    let has_rigid_road = c.ndim == 6 || c.ndim == 9;
    c.reduced_rigid = c.ndim == 9;

    let mut pos = (CONTROL_WORDS + c.extra) * wsz; // start of geometry

    // 1. material-type section: NUMRBE + NMMAT + material types.
    if mattyp {
        pos += (2 + c.nmmat) * wsz;
    }
    // 2. ALE fluid material id list.
    pos += c.ialemat * wsz;
    // 3. SPH element data flags (isphfg1..11): isphfg1 = flag-word count; the rest
    //    give n_sph_vars.
    if c.nmsph > 0 {
        let f = |k: usize| word(pos + k * wsz);
        let isphfg1 = f(0).max(0) as usize;
        let n_hist = if f(0) == 10 { 0 } else { f(10).max(0) as usize };
        c.n_sph_vars = (f(1) + f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8).abs() + f(9)).max(0)
            as usize
            + n_hist
            + 1; // material number
        pos += isphfg1 * wsz;
    }
    // 4. airbag/particle flags: ngeom, nvar(particle state), npart, nstgeom(airbag
    //    state), [n_chambers if subver==4], then 9 words per airbag variable.
    if c.npefg > 0 && c.npefg <= 10_000_000 {
        c.n_airbags = (c.npefg % 1000) as usize;
        let subver = c.npefg / 1000;
        c.n_geom_vars = word(pos).max(0) as usize;
        c.n_particle_state_vars = word(pos + wsz).max(0) as usize;
        c.n_particles = word(pos + 2 * wsz).max(0) as usize;
        c.n_airbag_state_vars = word(pos + 3 * wsz).max(0) as usize;
        pos += 4 * wsz;
        if subver == 4 {
            pos += wsz; // n_chambers
        }
        let n_airbag_vars = c.n_geom_vars + c.n_particle_state_vars + c.n_airbag_state_vars;
        pos += 9 * n_airbag_vars * wsz; // variable types (1) + names (8) each
    }
    // 5. geometry: node coordinates + element connectivity.
    c.coord_off = pos as u64;
    pos += c.numnp * SPATIAL_DIM * wsz;
    pos += (c.nel8 * SOLID_CONN + c.nelth * TSHELL_CONN + c.nel2 * BEAM_CONN + c.nel4 * SHELL_CONN)
        * wsz;
    // 6. user id numbering section.
    pos += c.narbs * wsz;
    // 7. rigid body description: nrigid, then per body (mrigid, numnodr, node list,
    //    numnoda, active node list).
    if has_rigid_body {
        c.n_rigids = word(pos).max(0) as usize;
        pos += wsz;
        for _ in 0..c.n_rigids {
            let numnodr = word(pos + wsz).max(0) as usize;
            pos += 2 * wsz + numnodr * wsz;
            let numnoda = word(pos).max(0) as usize;
            pos += wsz + numnoda * wsz;
        }
    }
    // 8. SPH node and material list: 2 words per SPH node.
    if c.nmsph > 0 {
        pos += c.nmsph * 2 * wsz;
    }
    // 9. airbag particle geometry: ngeom words per airbag.
    if c.n_airbags > 0 {
        pos += c.n_airbags * c.n_geom_vars * wsz;
    }
    // 10. rigid road surface: header (nnode, nseg, nsurf, motion) + node ids + node
    //     coords + per-surface (id, nseg, 4 words per segment).
    if has_rigid_road {
        let nnode = word(pos).max(0) as usize;
        c.n_roads = word(pos + 2 * wsz).max(0) as usize;
        pos += 4 * wsz + nnode * wsz + nnode * SPATIAL_DIM * wsz;
        for _ in 0..c.n_roads {
            let nseg = word(pos + wsz).max(0) as usize;
            pos += 2 * wsz + 4 * nseg * wsz;
        }
    }
    // 11. extra connectivity for 8-node shells (higher-order solids not modelled).
    pos += 5 * c.nel48 * wsz;

    c.geom_bytes = pos as u64;
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

/// Per-part-per-state fields packed into the global-variables block after the 6
/// global scalars, in LS-DYNA order (each `n_parts` wide unless noted). Set via
/// [`D3plotWriter::set_part_field`] / [`D3plotWriter::set_part_velocity`]; any
/// set part field forces the full `7*n_parts` block to be written (unset ones
/// zero-filled) so the reader's fixed offsets line up.
#[derive(Default)]
struct PartBlock {
    internal_energy: Option<Vec<f64>>, // n_states*n_parts
    kinetic_energy: Option<Vec<f64>>,  // n_states*n_parts
    velocity: Option<Vec<f64>>,        // n_states*n_parts*3
    mass: Option<Vec<f64>>,            // n_states*n_parts
    hourglass_energy: Option<Vec<f64>>, // n_states*n_parts
}

impl PartBlock {
    fn any(&self) -> bool {
        self.internal_energy.is_some()
            || self.kinetic_energy.is_some()
            || self.velocity.is_some()
            || self.mass.is_some()
            || self.hourglass_energy.is_some()
    }
}

/// Whole-model global scalars and the per-part block that together form the
/// per-state global-variables section (NGLBV). The 6 scalars are indexed exactly
/// as the reader expects them: 0 kinetic, 1 internal, 2 total, 3/4/5 velocity
/// x/y/z. Any set scalar forces all 6 to be written (unset ones zero).
#[derive(Default)]
struct Globals {
    scalars: [Option<Vec<f64>>; 6], // each len n_states
    parts: PartBlock,
    /// Rigid-wall force (`n_states*n_walls`) and optional position
    /// (`n_states*n_walls*3`); modern 971 layout keeps 4 vars/wall.
    rigid_wall: Option<RigidWall>,
}

impl Globals {
    fn any_scalar(&self) -> bool {
        self.scalars.iter().any(Option::is_some)
    }
    fn any(&self) -> bool {
        self.any_scalar() || self.parts.any() || self.rigid_wall.is_some()
    }
}

/// Rigid-wall per-state data living in the global-variables block after the
/// per-part block. `force` is `n_states*n_walls`; `position`, when present, is
/// `n_states*n_walls*3`. The writer emits the modern 971 layout (4 vars/wall)
/// whenever positions are given, else 1 var/wall (force only).
struct RigidWall {
    n_walls: usize,
    force: Vec<f64>,
    position: Option<Vec<f64>>,
}

impl RigidWall {
    /// Words per wall in the global block: 4 with positions, else 1.
    fn vars(&self) -> usize {
        if self.position.is_some() {
            4
        } else {
            1
        }
    }
}

/// Per-node thermal / auxiliary fields that sit between displacement and velocity
/// in the node stream, gated by the IT and IDTDT header flags. Set via
/// [`D3plotWriter::set_node_field`]; the writer derives IT/IDTDT from which of
/// these are present (see [`Control::node_field_spec`] for the inverse layout).
#[derive(Default)]
struct NodeThermal {
    temperature: Option<Vec<f64>>,     // n_states*numnp*{1|3}
    heat_flux: Option<Vec<f64>>,       // n_states*numnp*3
    mass_scaling: Option<Vec<f64>>,    // n_states*numnp
    temp_gradient: Option<Vec<f64>>,   // n_states*numnp
    residual_force: Option<Vec<f64>>,  // n_states*numnp*3
    residual_moment: Option<Vec<f64>>, // n_states*numnp*3
}

/// Element/node deletion ("is alive") state, terminating each state block. Only
/// one mode may be active (it is encoded in the sign/magnitude of MAXINT).
enum Deletion {
    /// mdlopt 1: one flag per node (`n_states*numnp`).
    Node(Vec<f64>),
    /// mdlopt 2: one flag per element, written in block order solid, thick-shell,
    /// shell, beam (unset blocks zero-filled to 1.0 = alive). Each `n_states*count`.
    Element {
        solid: Option<Vec<f64>>,
        tshell: Option<Vec<f64>>,
        shell: Option<Vec<f64>>,
        beam: Option<Vec<f64>>,
    },
}

/// SPH (smoothed-particle hydrodynamics) geometry + per-state results. The
/// per-particle variable count is carried in an ISPHFG flag section the writer
/// synthesizes so the reader recomputes exactly `n_vars` (see `walk_geometry`).
struct Sph {
    materials: Vec<i32>, // len = n particles; per-particle material number
    n_vars: usize,       // per-particle state var count (incl. material word)
    results: Vec<f64>,   // n_states * n_particles * n_vars, row-major
}

/// Airbag / CPM particle geometry + per-state results. `geom` is `n_airbags *
/// n_geom_vars` (written once, in geometry); the per-state block is the airbag
/// chamber state (`n_airbags*n_airbag_vars`) followed by particle state
/// (`n_particles*n_particle_vars`).
struct Airbag {
    n_airbags: usize,
    n_particles: usize,
    n_geom_vars: usize,          // ngeom
    n_airbag_vars: usize,        // nstgeom (per airbag, per state)
    n_particle_vars: usize,      // nvar (per particle, per state)
    geom: Vec<f64>,              // n_airbags * n_geom_vars
    airbag_state: Vec<f64>,      // n_states * n_airbags * n_airbag_vars
    particle_state: Vec<f64>,    // n_states * n_particles * n_particle_vars
}

/// A single rigid body's geometry (a part index plus its node index lists).
struct RigidBody {
    part: i32,
    nodes: Vec<i32>,        // all nodes (1-based)
    active_nodes: Vec<i32>, // active subset (1-based)
}

/// Rigid-body geometry + per-state motion. The per-body motion width is fixed by
/// the file layout: 12 vars (coords + 3x3 rotation) when a rigid road is also
/// present (NDIM=9), else the full 24 vars (adds translational/rotational
/// velocity and acceleration, NDIM=8).
struct RigidBodies {
    bodies: Vec<RigidBody>,
    motion: Vec<f64>, // n_states * n_bodies * (12|24), row-major
}

/// Rigid-road surface geometry + per-state motion (`n_roads*6` per state).
struct RigidRoad {
    node_ids: Vec<i32>,
    node_coords: Vec<f64>,           // n_road_nodes * 3
    segments: Vec<(i32, Vec<i32>)>,  // per surface: (road id, flat 4-node segment ids)
    motion: Vec<f64>,                // n_states * n_roads * 6
}

/// Builds a single-precision d3plot from a mesh (nodes + element connectivity)
/// and per-state results. Beyond nodal motion, it supports the full complement
/// of LS-DYNA state arrays — element results for every family (solid, thick
/// shell, beam, shell), global & per-part energies, nodal thermal/residual
/// fields, element/node deletion, and the SPH / airbag / rigid-body / rigid-road
/// blocks — each set through a strongly-typed method (no magic strings). Which
/// header flags and state layout get emitted is *derived* from which arrays were
/// populated, so callers only pay for what they set.
///
/// Output is single precision (f32): header + geometry + states, terminated by
/// the `-999999` marker. It reads back through [`D3plot`] and open-lasso-python
/// (node data bit-exact).
pub struct D3plotWriter {
    numnp: usize,
    wordsize: usize,        // 4 (single, default) or 8 (double) precision output
    x0: Vec<f64>,           // numnp*3, row-major x,y,z
    solids: Vec<[i32; 9]>,  // 8 node connectivity indices (1-based) + part index
    tshells: Vec<[i32; 9]>, // 8 node connectivity indices (1-based) + part index
    beams: Vec<[i32; 6]>,   // 5 connectivity slots (1-based) + part index
    shells: Vec<[i32; 5]>,  // 4 node connectivity indices (1-based) + part index
    states: Vec<StateData>,
    fields: NodeFields,
    title: String,
    // Optional user IDs for the NARBS numbering section (default 1..N).
    node_ids: Option<Vec<i32>>,
    solid_ids: Option<Vec<i32>>,
    beam_ids: Option<Vec<i32>>,
    shell_ids: Option<Vec<i32>>,
    tshell_ids: Option<Vec<i32>>,
    part_ids: Option<Vec<i32>>,
    // Optional per-element result blocks: (vars_per_element, flat
    // n_states*count*vars, row-major). Written raw after node data each state.
    solid_results: Option<(usize, Vec<f64>)>,
    tshell_results: Option<(usize, Vec<f64>)>,
    beam_results: Option<(usize, Vec<f64>)>,
    shell_results: Option<(usize, Vec<f64>)>,
    shell_layers: usize, // shell through-thickness integration points (MAXINT)
    // Whole-model / auxiliary state arrays, all derived into header flags.
    globals: Globals,
    node_thermal: NodeThermal,
    deletion: Option<Deletion>,
    sph: Option<Sph>,
    airbag: Option<Airbag>,
    rigid_bodies: Option<RigidBodies>,
    rigid_road: Option<RigidRoad>,
}

struct StateData {
    time: f64,
    disp: Vec<f64>, // numnp*3 current coordinates
    vel: Vec<f64>,  // numnp*3 or empty
    acc: Vec<f64>,  // numnp*3 or empty
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
            wordsize: 4,
            x0: node_coords,
            solids: Vec::new(),
            tshells: Vec::new(),
            beams: Vec::new(),
            shells: Vec::new(),
            states: Vec::new(),
            fields: NodeFields::default(),
            title: String::new(),
            node_ids: None,
            solid_ids: None,
            beam_ids: None,
            shell_ids: None,
            tshell_ids: None,
            part_ids: None,
            solid_results: None,
            tshell_results: None,
            beam_results: None,
            shell_results: None,
            shell_layers: 1,
            globals: Globals::default(),
            node_thermal: NodeThermal::default(),
            deletion: None,
            sph: None,
            airbag: None,
            rigid_bodies: None,
            rigid_road: None,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.numnp
    }

    /// Emit double-precision (8-byte word) output when `double` is true; the
    /// default is single precision (f32). Values are stored as f64 internally, so
    /// double-precision output is lossless.
    pub fn set_double_precision(&mut self, double: bool) {
        self.wordsize = if double { 8 } else { 4 };
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

    /// Per-solid result block as a [`ResultBlock`] shaped `[n_states, n_solids,
    /// vars]` (state-major, row-major; `vars` = the innermost axis). Sets NV3D.
    pub fn set_solid_results(&mut self, block: ResultBlock) {
        let vars = block.dims().last().copied().unwrap_or(0);
        self.solid_results = Some((vars, block.into_data()));
    }

    /// Per-shell result block as a [`ResultBlock`] shaped `[n_states, n_shells,
    /// vars]`. `vars` packs the through-thickness layers (see
    /// [`set_shell_layers`](Self::set_shell_layers)). Sets NV2D.
    pub fn set_shell_results(&mut self, block: ResultBlock) {
        let vars = block.dims().last().copied().unwrap_or(0);
        self.shell_results = Some((vars, block.into_data()));
    }

    /// Number of through-thickness integration points (layers) packed into each
    /// shell result record (sets MAXINT). `set_shell_results`' `vars` must equal
    /// `n_layers * per_layer`, where `per_layer` = 6 stress + 1 plastic strain +
    /// history; the record is laid out layer-by-layer. Default 1.
    pub fn set_shell_layers(&mut self, n_layers: usize) {
        self.shell_layers = n_layers.max(1);
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

    /// Add a thick shell (8 one-based node ids).
    pub fn add_tshell(&mut self, nodes: [i32; 8], part: i32) {
        let mut c = [0i32; 9];
        c[..8].copy_from_slice(&nodes);
        c[8] = part;
        self.tshells.push(c);
    }

    /// Add a beam: two end nodes plus an orientation node (`[n1, n2, n3]`, all
    /// one-based; pass 0 for an unused orientation node). The two extra
    /// connectivity slots LS-DYNA reserves are written as 0.
    pub fn add_beam(&mut self, nodes: [i32; 3], part: i32) {
        self.beams
            .push([nodes[0], nodes[1], nodes[2], 0, 0, part]);
    }

    /// User beam element IDs (length = number of beams).
    pub fn set_beam_ids(&mut self, ids: Vec<i64>) {
        self.beam_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }
    /// User thick-shell element IDs (length = number of thick shells).
    pub fn set_tshell_ids(&mut self, ids: Vec<i64>) {
        self.tshell_ids = Some(ids.into_iter().map(|x| x as i32).collect());
    }

    /// Per-thick-shell result block as a [`ResultBlock`] shaped `[n_states,
    /// n_tshells, vars]`. Sets NV3DT.
    pub fn set_tshell_results(&mut self, block: ResultBlock) {
        let vars = block.dims().last().copied().unwrap_or(0);
        self.tshell_results = Some((vars, block.into_data()));
    }

    /// Per-beam result block as a [`ResultBlock`] shaped `[n_states, n_beams,
    /// vars]`. Sets NV1D.
    pub fn set_beam_results(&mut self, block: ResultBlock) {
        let vars = block.dims().last().copied().unwrap_or(0);
        self.beam_results = Some((vars, block.into_data()));
    }

    /// A whole-model global scalar history (one value per state), written into the
    /// global-variables block at this field's fixed slot. Setting any global
    /// scalar or part field makes the writer emit the block (NGLBV > 0).
    pub fn set_global_history(&mut self, field: GlobalField, data: Vec<f64>) {
        let idx = match field {
            GlobalField::KineticEnergy => 0,
            GlobalField::InternalEnergy => 1,
            GlobalField::TotalEnergy => 2,
            GlobalField::VelocityX => 3,
            GlobalField::VelocityY => 4,
            GlobalField::VelocityZ => 5,
        };
        self.globals.scalars[idx] = Some(data);
    }

    /// A per-part scalar history (`n_states * n_parts`, row-major, part index
    /// matching the material/part order). Lives in the per-part portion of the
    /// global-variables block; setting any part field forces the full per-part
    /// block to be written.
    pub fn set_part_field(&mut self, field: PartField, data: Vec<f64>) {
        match field {
            PartField::InternalEnergy => self.globals.parts.internal_energy = Some(data),
            PartField::KineticEnergy => self.globals.parts.kinetic_energy = Some(data),
            PartField::Mass => self.globals.parts.mass = Some(data),
            PartField::HourglassEnergy => self.globals.parts.hourglass_energy = Some(data),
        }
    }

    /// Per-part velocity history (`n_states * n_parts * 3`, row-major). Occupies
    /// the velocity slot of the per-part global block.
    pub fn set_part_velocity(&mut self, data: Vec<f64>) {
        self.globals.parts.velocity = Some(data);
    }

    /// Rigid-wall per-state data: `force` is `n_states * n_walls` (row-major);
    /// `position`, when given, is `n_states * n_walls * 3`. Providing positions
    /// selects the modern 971 layout (4 vars/wall). Lives in the global block, so
    /// setting it also forces the per-part block to be emitted.
    pub fn set_rigid_walls(&mut self, n_walls: usize, force: Vec<f64>, position: Option<Vec<f64>>) {
        self.globals.rigid_wall = Some(RigidWall {
            n_walls,
            force,
            position,
        });
    }

    /// A per-node thermal / auxiliary field history. Data length fixes the per-node
    /// width the writer records into IT/IDTDT: `Temperature` is `n_states*numnp`
    /// (single) or `n_states*numnp*3` (through-thickness layers); `HeatFlux`,
    /// `ResidualForce`, `ResidualMoment` are ×3; `MassScaling` and
    /// `TemperatureGradient` are ×1. `HeatFlux` requires `Temperature` (LS-DYNA's
    /// IT encoding has no flux-without-temperature form).
    pub fn set_node_field(&mut self, field: NodeField, data: Vec<f64>) {
        match field {
            NodeField::Temperature => self.node_thermal.temperature = Some(data),
            NodeField::HeatFlux => self.node_thermal.heat_flux = Some(data),
            NodeField::MassScaling => self.node_thermal.mass_scaling = Some(data),
            NodeField::TemperatureGradient => self.node_thermal.temp_gradient = Some(data),
            NodeField::ResidualForce => self.node_thermal.residual_force = Some(data),
            NodeField::ResidualMoment => self.node_thermal.residual_moment = Some(data),
        }
    }

    /// Per-element deletion ("is alive") flags for one element family (mdlopt 2):
    /// `n_states * n_elements`, 1.0 = alive, 0.0 = deleted. Element and node
    /// deletion are mutually exclusive; the last mode set wins.
    pub fn set_element_deletion(&mut self, block: StateBlock, alive: Vec<f64>) {
        let v = alive;
        let del = match self.deletion.take() {
            Some(Deletion::Element {
                solid,
                tshell,
                shell,
                beam,
            }) => (solid, tshell, shell, beam),
            _ => (None, None, None, None),
        };
        let (mut solid, mut tshell, mut shell, mut beam) = del;
        match block {
            StateBlock::Solid => solid = Some(v),
            StateBlock::ThickShell => tshell = Some(v),
            StateBlock::Shell => shell = Some(v),
            StateBlock::Beam => beam = Some(v),
            _ => return,
        }
        self.deletion = Some(Deletion::Element {
            solid,
            tshell,
            shell,
            beam,
        });
    }

    /// Per-node deletion flags (mdlopt 1): `n_states * numnp`, 1.0 = alive. Note
    /// the [`D3plot`] reader exposes only element deletion, so node deletion is
    /// write-only for round-tripping purposes. Mutually exclusive with element
    /// deletion.
    pub fn set_node_deletion(&mut self, alive: Vec<f64>) {
        self.deletion = Some(Deletion::Node(alive));
    }

    /// SPH particle geometry + per-state results. `materials` (length = particle
    /// count) gives each particle's material number; `n_vars` is the per-particle
    /// state width and `results` is `n_states * n_particles * n_vars` row-major.
    pub fn set_sph(&mut self, materials: Vec<i64>, n_vars: usize, results: Vec<f64>) {
        self.sph = Some(Sph {
            materials: materials.into_iter().map(|x| x as i32).collect(),
            n_vars,
            results,
        });
    }

    /// Airbag / CPM geometry + per-state results. `geom` is `n_airbags *
    /// n_geom_vars`; `airbag_state` is `n_states * n_airbags * n_airbag_vars` and
    /// `particle_state` is `n_states * n_particles * n_particle_vars`, all
    /// row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn set_airbag(
        &mut self,
        n_airbags: usize,
        n_particles: usize,
        n_geom_vars: usize,
        n_airbag_vars: usize,
        n_particle_vars: usize,
        geom: Vec<f64>,
        airbag_state: Vec<f64>,
        particle_state: Vec<f64>,
    ) {
        self.airbag = Some(Airbag {
            n_airbags,
            n_particles,
            n_geom_vars,
            n_airbag_vars,
            n_particle_vars,
            geom,
            airbag_state,
            particle_state,
        });
    }

    /// Rigid-body geometry + per-state motion. Each body is `(part_id, node_ids,
    /// active_node_ids)`. `motion` is `n_states * n_bodies * k` row-major, where
    /// `k` is 12 when a rigid road is also set (coords + rotation) or 24
    /// otherwise (adds translational/rotational velocity and acceleration).
    pub fn set_rigid_bodies(&mut self, bodies: Vec<(i64, Vec<i64>, Vec<i64>)>, motion: Vec<f64>) {
        let bodies = bodies
            .into_iter()
            .map(|(part, nodes, active)| RigidBody {
                part: part as i32,
                nodes: nodes.into_iter().map(|x| x as i32).collect(),
                active_nodes: active.into_iter().map(|x| x as i32).collect(),
            })
            .collect();
        self.rigid_bodies = Some(RigidBodies { bodies, motion });
    }

    /// Rigid-road surface geometry + per-state motion. `segments` is one
    /// `(road_id, [4 node ids per segment])` per road surface; `motion` is
    /// `n_states * n_roads * 6` row-major.
    pub fn set_rigid_road(
        &mut self,
        node_ids: Vec<i64>,
        node_coords: Vec<f64>,
        segments: Vec<(i64, Vec<i64>)>,
        motion: Vec<f64>,
    ) {
        self.rigid_road = Some(RigidRoad {
            node_ids: node_ids.into_iter().map(|x| x as i32).collect(),
            node_coords,
            segments: segments
                .into_iter()
                .map(|(id, seg)| (id as i32, seg.into_iter().map(|x| x as i32).collect()))
                .collect(),
            motion,
        });
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
        self.states.push(StateData {
            time,
            disp,
            vel: vel.unwrap_or_default(),
            acc: acc.unwrap_or_default(),
        });
        Ok(())
    }

    /// The material/part count: the max part id across every element family and
    /// rigid-body part (clamped to ≥ 0). This is NUMMAT / the per-part block width.
    fn part_count(&self) -> usize {
        self.solids
            .iter()
            .map(|s| s[8])
            .chain(self.tshells.iter().map(|s| s[8]))
            .chain(self.beams.iter().map(|s| s[5]))
            .chain(self.shells.iter().map(|s| s[4]))
            .chain(
                self.rigid_bodies
                    .iter()
                    .flat_map(|rb| rb.bodies.iter().map(|b| b.part)),
            )
            .max()
            .unwrap_or(0)
            .max(0) as usize
    }

    /// Check every whole-history array's length against the state count and its
    /// entity count, returning a clear error instead of letting
    /// [`to_bytes`](Self::to_bytes) panic on an out-of-range slice. Called by
    /// [`write`](Self::write).
    pub fn validate(&self) -> Result<(), D3plotError> {
        let ns = self.states.len();
        let numnp = self.numnp;
        let np = self.part_count();
        let bad = |what: &str, got: usize, want: usize| {
            Err(D3plotError::Unsupported(format!(
                "{what}: length {got} != expected {want}"
            )))
        };
        let chk = |what: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                bad(what, got, want)
            }
        };
        // element results (vars, flat): n_states * count * vars
        for (label, count, res) in [
            ("solid_results", self.solids.len(), &self.solid_results),
            ("tshell_results", self.tshells.len(), &self.tshell_results),
            ("beam_results", self.beams.len(), &self.beam_results),
            ("shell_results", self.shells.len(), &self.shell_results),
        ] {
            if let Some((vars, data)) = res {
                chk(label, data.len(), ns * count * vars)?;
            }
        }
        // globals + part block
        for (i, s) in self.globals.scalars.iter().enumerate() {
            if let Some(d) = s {
                chk(&format!("global scalar {i}"), d.len(), ns)?;
            }
        }
        let p = &self.globals.parts;
        for (label, opt, w) in [
            ("part internal_energy", &p.internal_energy, np),
            ("part kinetic_energy", &p.kinetic_energy, np),
            ("part velocity", &p.velocity, 3 * np),
            ("part mass", &p.mass, np),
            ("part hourglass_energy", &p.hourglass_energy, np),
        ] {
            if let Some(d) = opt {
                chk(label, d.len(), ns * w)?;
            }
        }
        if let Some(w) = &self.globals.rigid_wall {
            chk("rigid_wall force", w.force.len(), ns * w.n_walls)?;
            if let Some(pos) = &w.position {
                chk("rigid_wall position", pos.len(), ns * 3 * w.n_walls)?;
            }
        }
        // node thermal fields
        let nt = &self.node_thermal;
        if let Some(t) = &nt.temperature {
            let unit = ns * numnp;
            if unit == 0 || (t.len() != unit && t.len() != 3 * unit) {
                return bad("node Temperature", t.len(), unit);
            }
        }
        for (label, opt, per) in [
            ("node HeatFlux", &nt.heat_flux, 3),
            ("node MassScaling", &nt.mass_scaling, 1),
            ("node TemperatureGradient", &nt.temp_gradient, 1),
            ("node ResidualForce", &nt.residual_force, 3),
            ("node ResidualMoment", &nt.residual_moment, 3),
        ] {
            if let Some(d) = opt {
                chk(label, d.len(), ns * numnp * per)?;
            }
        }
        // deletion
        match &self.deletion {
            Some(Deletion::Node(d)) => chk("node deletion", d.len(), ns * numnp)?,
            Some(Deletion::Element {
                solid,
                tshell,
                shell,
                beam,
            }) => {
                for (label, opt, count) in [
                    ("solid deletion", solid, self.solids.len()),
                    ("tshell deletion", tshell, self.tshells.len()),
                    ("shell deletion", shell, self.shells.len()),
                    ("beam deletion", beam, self.beams.len()),
                ] {
                    if let Some(d) = opt {
                        chk(label, d.len(), ns * count)?;
                    }
                }
            }
            None => {}
        }
        // sph / airbag / rigid
        if let Some(s) = &self.sph {
            if s.n_vars == 0 {
                return Err(D3plotError::Unsupported("sph n_vars must be ≥ 1".into()));
            }
            chk("sph results", s.results.len(), ns * s.materials.len() * s.n_vars)?;
        }
        if let Some(a) = &self.airbag {
            chk("airbag geom", a.geom.len(), a.n_airbags * a.n_geom_vars)?;
            chk(
                "airbag state",
                a.airbag_state.len(),
                ns * a.n_airbags * a.n_airbag_vars,
            )?;
            chk(
                "particle state",
                a.particle_state.len(),
                ns * a.n_particles * a.n_particle_vars,
            )?;
        }
        if let Some(rb) = &self.rigid_bodies {
            let k = if self.rigid_road.is_some() { 12 } else { 24 };
            chk("rigid_body motion", rb.motion.len(), ns * rb.bodies.len() * k)?;
        }
        if let Some(rr) = &self.rigid_road {
            chk("rigid_road node_coords", rr.node_coords.len(), 3 * rr.node_ids.len())?;
            for (i, (_, seg)) in rr.segments.iter().enumerate() {
                if !seg.len().is_multiple_of(4) {
                    return Err(D3plotError::Unsupported(format!(
                        "rigid_road segment {i}: {} node ids not a multiple of 4",
                        seg.len()
                    )));
                }
            }
            chk("rigid_road motion", rr.motion.len(), ns * rr.segments.len() * 6)?;
        }
        Ok(())
    }

    /// Serialize to a complete d3plot byte image (single or double precision per
    /// [`set_double_precision`](Self::set_double_precision)). Panics on malformed
    /// inputs; call [`validate`](Self::validate) (or [`write`](Self::write)) first
    /// for a checked path.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_states = self.states.len();
        let nel8 = self.solids.len();
        let nelth = self.tshells.len();
        let nel2 = self.beams.len();
        let nel4 = self.shells.len();
        let nmmat = self.part_count() as i32;
        let nmmat_u = nmmat as usize;
        let ws = self.wordsize;

        // --- element result var counts ---
        let nv3d = self.solid_results.as_ref().map_or(0, |(v, _)| *v);
        let nv3dt = self.tshell_results.as_ref().map_or(0, |(v, _)| *v);
        let nv1d = self.beam_results.as_ref().map_or(0, |(v, _)| *v);
        let nv2d = self.shell_results.as_ref().map_or(0, |(v, _)| *v);

        // --- node thermal / auxiliary fields → IT / IDTDT ---
        let nt = &self.node_thermal;
        let per_node = |d: &[f64]| if n_states > 0 && self.numnp > 0 { d.len() / (n_states * self.numnp) } else { 0 };
        let temp_w = nt.temperature.as_deref().map_or(0, per_node); // 1 or 3
        let want_flux = nt.heat_flux.is_some();
        // IT ones digit: 1 = temp(1); 2 = temp(1)+flux(3); 3 = temp(3)+flux(3).
        let (it0, tw) = if temp_w == 3 {
            (3, 3)
        } else if nt.temperature.is_some() && want_flux {
            (2, 1)
        } else if nt.temperature.is_some() {
            (1, 1)
        } else if want_flux {
            (2, 1) // flux implies a (zero) temperature slot in LS-DYNA's encoding
        } else {
            (0, 0)
        };
        let has_temp = it0 >= 1;
        let has_flux = it0 >= 2;
        let has_mass = nt.mass_scaling.is_some();
        let has_grad = nt.temp_gradient.is_some();
        let has_resid = nt.residual_force.is_some() || nt.residual_moment.is_some();
        let it = it0 + if has_mass { 10 } else { 0 };
        let idtdt = if has_grad { 1 } else { 0 } + if has_resid { 10 } else { 0 };

        // --- global-variables block (NGLBV) ---
        // Rigid-wall data sits after the per-part block, so any rigid wall forces
        // the full per-part block to be emitted (zero-filled if no part fields).
        let rw = self.globals.rigid_wall.as_ref();
        let rw_words = rw.map_or(0, |w| w.n_walls * w.vars());
        let has_part_block = self.globals.parts.any() || rw.is_some();
        let nglbv = if self.globals.any() {
            6 + if has_part_block { 7 * nmmat_u } else { 0 } + rw_words
        } else {
            0
        };

        // --- deletion (mdlopt sign packed into MAXINT) ---
        let shell_layers = self.shell_layers.max(1);
        let maxint: i32 = match &self.deletion {
            None => shell_layers as i32,
            Some(Deletion::Node(_)) => -(shell_layers as i32),
            Some(Deletion::Element { .. }) => {
                -((shell_layers + MDLOPT_ELEMENT_DELETION as usize) as i32)
            }
        };

        // --- SPH / airbag / rigid-body / rigid-road presence and sizing ---
        let nmsph = self.sph.as_ref().map_or(0, |s| s.materials.len());
        let n_sph_vars = self.sph.as_ref().map_or(0, |s| s.n_vars);
        let npefg = self.airbag.as_ref().map_or(0, |a| a.n_airbags as i32);
        let has_rb = self.rigid_bodies.is_some();
        let has_road = self.rigid_road.is_some();
        // NDIM is a flag word: 8 = rigid body, 6 = rigid road, 9 = both (reduced
        // 12-var rigid motion), else 4 (plain structural).
        let ndim = if has_rb && has_road {
            9
        } else if has_rb {
            8
        } else if has_road {
            6
        } else {
            NDIM_STRUCTURAL
        };
        let rigid_k = if ndim == 9 { 12 } else { 24 };

        // --- control block ---
        let mut words = [0i32; CONTROL_WORDS];
        // Title occupies TITLE_WORDS words (space-padded), so its byte width
        // follows the word size (40 bytes single, 80 double).
        let title_bytes = TITLE_WORDS * ws;
        let mut title = vec![b' '; title_bytes];
        for (i, b) in self.title.bytes().take(title_bytes).enumerate() {
            title[i] = b;
        }
        let set = |w: &mut [i32; CONTROL_WORDS], i: usize, v: i32| w[i] = v;
        set(&mut words, word::FILETYPE, FILETYPE_D3PLOT);
        set(&mut words, word::NDIM, ndim);
        set(&mut words, word::NUMNP, self.numnp as i32);
        set(&mut words, word::ICODE, ICODE_LSDYNA);
        set(&mut words, word::NGLBV, nglbv as i32);
        set(&mut words, word::IT, it);
        set(&mut words, word::IU, 1);
        set(&mut words, word::IV, i32::from(self.fields.velocity));
        set(&mut words, word::IA, i32::from(self.fields.acceleration));
        set(&mut words, word::NEL8, nel8 as i32);
        set(&mut words, word::NV3D, nv3d as i32);
        set(&mut words, word::NELTH, nelth as i32);
        set(&mut words, word::NV3DT, nv3dt as i32);
        set(&mut words, word::NEL2, nel2 as i32);
        set(&mut words, word::NV1D, nv1d as i32);
        set(&mut words, word::NEL4, nel4 as i32);
        set(&mut words, word::NUMMAT4, nmmat);
        set(&mut words, word::NV2D, nv2d as i32);
        set(&mut words, word::MAXINT, maxint);
        set(&mut words, word::NMSPH, nmsph as i32);
        set(&mut words, word::NPEFG, npefg);
        set(&mut words, word::IDTDT, idtdt);
        set(&mut words, word::NMMAT, nmmat);
        // NARBS numbering section size (always emitted, part-id header form).
        let narbs = self.numnp + nel8 + nel2 + nel4 + nelth + 3 * nmmat_u + NARBS_PART_HEADER;
        set(&mut words, word::NARBS, narbs as i32);

        // Result-field flags so LS-PrePost/lasso name the raw element vars. Per
        // layer, solver order is 6 stress, 1 plastic strain, then history.
        // ioshl1/ioshl2 are shared by solids and shells.
        let shell_per_layer = if nv2d > 0 { nv2d / shell_layers } else { 0 };
        if nv3d > 0 || nv2d > 0 {
            let has_stress = nv3d >= ELEM_STRESS_VARS || shell_per_layer >= ELEM_STRESS_VARS;
            let has_pstrain = nv3d >= ELEM_BASE_VARS || shell_per_layer >= ELEM_BASE_VARS;
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
                set(&mut words, word::NEIPH, nv3d.saturating_sub(base) as i32);
            }
            if nv2d > 0 {
                set(&mut words, word::NEIPS, shell_per_layer.saturating_sub(base) as i32);
            }
        }

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&title);
        for w in &words[TITLE_WORDS..] {
            push_word_i(&mut buf, *w as i64, ws);
        }

        // ================= geometry (LS-DYNA walk order) =================
        // 3. SPH element-data flags (ISPHFG): 11 words. We synthesize a layout the
        //    reader decodes back to exactly n_sph_vars (see `walk_geometry`).
        if nmsph > 0 {
            push_word_i(&mut buf, 11, ws); // isphfg1 = flag-word count
            push_word_i(&mut buf, n_sph_vars.max(1) as i64 - 1, ws); // f1: all but the material word
            for _ in 2..11 {
                push_word_i(&mut buf, 0, ws);
            }
        }
        // 4. airbag / particle flags: ngeom, nvar, npart, nstgeom, then 9 words
        //    (type + 8 name) per airbag variable.
        if let Some(a) = &self.airbag {
            push_word_i(&mut buf, a.n_geom_vars as i64, ws);
            push_word_i(&mut buf, a.n_particle_vars as i64, ws);
            push_word_i(&mut buf, a.n_particles as i64, ws);
            push_word_i(&mut buf, a.n_airbag_vars as i64, ws);
            let n_vars = a.n_geom_vars + a.n_particle_vars + a.n_airbag_vars;
            for _ in 0..9 * n_vars {
                push_word_i(&mut buf, 0, ws);
            }
        }
        // 5. node coordinates + element connectivity (solids, tshells, beams, shells).
        push_word_fs(&mut buf, &self.x0, ws);
        for s in &self.solids {
            for &w in s {
                push_word_i(&mut buf, w as i64, ws);
            }
        }
        for s in &self.tshells {
            for &w in s {
                push_word_i(&mut buf, w as i64, ws);
            }
        }
        for b in &self.beams {
            for &w in b {
                push_word_i(&mut buf, w as i64, ws);
            }
        }
        for s in &self.shells {
            for &w in s {
                push_word_i(&mut buf, w as i64, ws);
            }
        }
        // 6. NARBS arbitrary numbering.
        write_narbs(
            &mut buf,
            self.numnp,
            nel8,
            nel2,
            nel4,
            nelth,
            nmmat_u,
            self.node_ids.as_deref(),
            self.solid_ids.as_deref(),
            self.beam_ids.as_deref(),
            self.shell_ids.as_deref(),
            self.tshell_ids.as_deref(),
            self.part_ids.as_deref(),
            ws,
        );
        // 7. rigid-body description: nrigid, then per body (part, numnodr, node
        //    list, numnoda, active-node list).
        if let Some(rb) = &self.rigid_bodies {
            push_word_i(&mut buf, rb.bodies.len() as i64, ws);
            for body in &rb.bodies {
                push_word_i(&mut buf, body.part as i64, ws);
                push_word_i(&mut buf, body.nodes.len() as i64, ws);
                for &n in &body.nodes {
                    push_word_i(&mut buf, n as i64, ws);
                }
                push_word_i(&mut buf, body.active_nodes.len() as i64, ws);
                for &n in &body.active_nodes {
                    push_word_i(&mut buf, n as i64, ws);
                }
            }
        }
        // 8. SPH node & material list: 2 words per particle (node number, material).
        if let Some(s) = &self.sph {
            for (i, &m) in s.materials.iter().enumerate() {
                push_word_i(&mut buf, i as i64 + 1, ws);
                push_word_i(&mut buf, m as i64, ws);
            }
        }
        // 9. airbag particle geometry: ngeom words per airbag.
        if let Some(a) = &self.airbag {
            push_word_fs(&mut buf, &a.geom, ws);
        }
        // 10. rigid-road surface: header (nnode, nseg_total, nsurf, motion) +
        //     node ids + node coords + per surface (id, nseg, 4 ids per segment).
        if let Some(rr) = &self.rigid_road {
            let nnode = rr.node_ids.len();
            let total_seg: usize = rr.segments.iter().map(|(_, s)| s.len() / 4).sum();
            push_word_i(&mut buf, nnode as i64, ws);
            push_word_i(&mut buf, total_seg as i64, ws);
            push_word_i(&mut buf, rr.segments.len() as i64, ws);
            push_word_i(&mut buf, 1, ws);
            for &n in &rr.node_ids {
                push_word_i(&mut buf, n as i64, ws);
            }
            push_word_fs(&mut buf, &rr.node_coords, ws);
            for (id, seg) in &rr.segments {
                push_word_i(&mut buf, *id as i64, ws);
                push_word_i(&mut buf, (seg.len() / 4) as i64, ws);
                for &n in seg {
                    push_word_i(&mut buf, n as i64, ws);
                }
            }
        }

        // ================= states =================
        // Per-node thermal words in stream order (temp, flux, mass, grad,
        // residual-force, residual-moment), used to walk each state's node block.
        let numnp = self.numnp;
        let per_solid = nv3d * nel8;
        let per_tshell = nv3dt * nelth;
        let per_beam = nv1d * nel2;
        let per_shell = nv2d * nel4;
        for (si, st) in self.states.iter().enumerate() {
            push_word_f(&mut buf, st.time, ws);
            // global variables
            if nglbv > 0 {
                for s in &self.globals.scalars {
                    push_word_f(&mut buf, s.as_ref().map_or(0.0, |d| d[si]), ws);
                }
                if has_part_block {
                    let p = &self.globals.parts;
                    push_opt_block(&mut buf, &p.internal_energy, si, nmmat_u, 0.0, ws);
                    push_opt_block(&mut buf, &p.kinetic_energy, si, nmmat_u, 0.0, ws);
                    push_opt_block(&mut buf, &p.velocity, si, 3 * nmmat_u, 0.0, ws);
                    push_opt_block(&mut buf, &p.mass, si, nmmat_u, 0.0, ws);
                    push_opt_block(&mut buf, &p.hourglass_energy, si, nmmat_u, 0.0, ws);
                }
                // rigid-wall block: force per wall, then optional 3-vector position.
                if let Some(w) = rw {
                    let nw = w.n_walls;
                    push_word_fs(&mut buf, &w.force[si * nw..(si + 1) * nw], ws);
                    if let Some(pos) = &w.position {
                        let per = 3 * nw;
                        push_word_fs(&mut buf, &pos[si * per..(si + 1) * per], ws);
                    }
                }
            }
            // node data: displacement, thermal block, velocity, acceleration.
            push_word_fs(&mut buf, &st.disp, ws);
            if has_temp {
                push_opt_block(&mut buf, &nt.temperature, si, tw * numnp, 0.0, ws);
            }
            if has_flux {
                push_opt_block(&mut buf, &nt.heat_flux, si, 3 * numnp, 0.0, ws);
            }
            if has_mass {
                push_opt_block(&mut buf, &nt.mass_scaling, si, numnp, 0.0, ws);
            }
            if has_grad {
                push_opt_block(&mut buf, &nt.temp_gradient, si, numnp, 0.0, ws);
            }
            if has_resid {
                push_opt_block(&mut buf, &nt.residual_force, si, 3 * numnp, 0.0, ws);
                push_opt_block(&mut buf, &nt.residual_moment, si, 3 * numnp, 0.0, ws);
            }
            push_word_fs(&mut buf, &st.vel, ws);
            push_word_fs(&mut buf, &st.acc, ws);
            // element results: solids, thick shells, beams, shells.
            if let Some((_, data)) = &self.solid_results {
                push_word_fs(&mut buf, &data[si * per_solid..(si + 1) * per_solid], ws);
            }
            if let Some((_, data)) = &self.tshell_results {
                push_word_fs(&mut buf, &data[si * per_tshell..(si + 1) * per_tshell], ws);
            }
            if let Some((_, data)) = &self.beam_results {
                push_word_fs(&mut buf, &data[si * per_beam..(si + 1) * per_beam], ws);
            }
            if let Some((_, data)) = &self.shell_results {
                push_word_fs(&mut buf, &data[si * per_shell..(si + 1) * per_shell], ws);
            }
            // deletion (mdlopt). Element order: solid, tshell, shell, beam.
            match &self.deletion {
                Some(Deletion::Node(d)) => {
                    push_word_fs(&mut buf, &d[si * numnp..(si + 1) * numnp], ws)
                }
                Some(Deletion::Element {
                    solid,
                    tshell,
                    shell,
                    beam,
                }) => {
                    push_opt_block(&mut buf, solid, si, nel8, 1.0, ws);
                    push_opt_block(&mut buf, tshell, si, nelth, 1.0, ws);
                    push_opt_block(&mut buf, shell, si, nel4, 1.0, ws);
                    push_opt_block(&mut buf, beam, si, nel2, 1.0, ws);
                }
                None => {}
            }
            // SPH state.
            if let Some(s) = &self.sph {
                let per = nmsph * n_sph_vars;
                push_word_fs(&mut buf, &s.results[si * per..(si + 1) * per], ws);
            }
            // airbag state: chamber block then particle block.
            if let Some(a) = &self.airbag {
                let per_a = a.n_airbags * a.n_airbag_vars;
                let per_p = a.n_particles * a.n_particle_vars;
                push_word_fs(&mut buf, &a.airbag_state[si * per_a..(si + 1) * per_a], ws);
                push_word_fs(&mut buf, &a.particle_state[si * per_p..(si + 1) * per_p], ws);
            }
            // rigid-road motion.
            if let Some(rr) = &self.rigid_road {
                let per = rr.segments.len() * 6;
                push_word_fs(&mut buf, &rr.motion[si * per..(si + 1) * per], ws);
            }
            // rigid-body motion.
            if let Some(rb) = &self.rigid_bodies {
                let per = rb.bodies.len() * rigid_k;
                push_word_fs(&mut buf, &rb.motion[si * per..(si + 1) * per], ws);
            }
        }
        // end-of-file marker
        push_word_f(&mut buf, EOF_MARKER, ws);
        buf
    }

    /// Validate the configured arrays, then write the d3plot to `path`.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), D3plotError> {
        self.validate()?;
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
            0,
            numsg,
            0,
            nmmat,
            self.node_ids.as_deref(),
            None,
            None,
            Some(&seg_ids),
            None,
            None,
            4, // intfor writer is single precision
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

    #[test]
    fn material_section_rigid_shell_count() {
        // NDIM 5 ⇒ material-type section present; its first geometry word is NUMRBE
        // (rigid-body shells that write no state data).
        let mut words = vec![0i32; 66];
        words[15] = 5; // NDIM 5 ⇒ mattyp, wordsize 4
        words[31] = 10; // NEL4
        words[33] = 8; // NV2D
        words[51] = 2; // NMMAT
        words[64] = 3; // material section first word = n_rigid_shells
        let mut buf = Vec::new();
        for &w in &words {
            buf.write_i32::<LittleEndian>(w).unwrap();
        }
        let ctrl = read_control_bytes(&buf).unwrap();
        assert_eq!(ctrl.n_rigid_shells, 3);
        assert_eq!(ctrl.n_shells_with_data(), 7);
        assert_eq!(ctrl.element_words(), 7 * 8); // only shells: (10-3)*8
        let (_, count, vars) = ctrl.block_spec(StateBlock::Shell).unwrap();
        assert_eq!((count, vars), (7, 8));
    }

    // Encode a header + hand-set geometry words to a little-endian d3plot buffer.
    fn control_from_words(words: &[i32]) -> Control {
        let mut buf = Vec::new();
        for &w in words {
            buf.write_i32::<LittleEndian>(w).unwrap();
        }
        read_control_bytes(&buf).unwrap()
    }

    #[test]
    fn geometry_walk_airbag_sizing() {
        // NPEFG=2 (2 airbags, subver 0). Airbag flags at word 64: ngeom, nvar
        // (particle state), npart, nstgeom (airbag state).
        let mut w = vec![0i32; 72];
        w[15] = 3; // NDIM plain
        w[16] = 1; // NUMNP
        w[20] = 1; // IU
        w[54] = 2; // NPEFG
        w[64] = 3; // ngeom
        w[65] = 5; // nvar (particle state vars)
        w[66] = 4; // npart (particles)
        w[67] = 2; // nstgeom (airbag state vars)
        let c = control_from_words(&w);
        assert_eq!(
            (c.n_airbags, c.n_particles, c.n_airbag_state_vars, c.n_particle_state_vars, c.n_geom_vars),
            (2, 4, 2, 5, 3)
        );
        // flag section = 4 + 9*(3+5+2) = 94 words ⇒ coords start at word 64+94.
        assert_eq!(c.coord_off, (64 + 94) * 4);
        // airbag state tail = 2*2 + 4*5 = 24 words; + time(1) + node(iu:3) = 28.
        assert_eq!(c.bytes_per_state(), 28 * 4);
    }

    #[test]
    fn geometry_walk_sph_sizing() {
        // NMSPH=2. isphfg1..11 at word 64. isphfg1=11 (flag count).
        let mut w = vec![0i32; 80];
        w[15] = 3; // NDIM
        w[16] = 1; // NUMNP
        w[20] = 1; // IU
        w[37] = 2; // NMSPH
        w[64] = 11; // isphfg1 = flag-word count
        for k in 65..=71 {
            w[k] = 1; // isphfg2..8 = 1 each (7)
        }
        w[72] = 6; // isphfg9 (stress-ish)
        w[73] = 1; // isphfg10 (mass)
        w[74] = 2; // isphfg11 (history vars)
        let c = control_from_words(&w);
        // n_sph_vars = 7 + |6| + 1 + 2 + 1(material) = 17.
        assert_eq!(c.n_sph_vars, 17);
        assert_eq!(c.coord_off, (64 + 11) * 4); // flags = isphfg1 = 11 words
        // sph state tail = nmsph * n_sph_vars = 2*17 = 34; + time(1) + node(3) = 38.
        assert_eq!(c.bytes_per_state(), 38 * 4);
    }

    #[test]
    fn geometry_walk_rigid_body_sizing() {
        // NDIM=8 ⇒ rigid-body data (not reduced). One rigid body with 0 nodes.
        // Geometry: coords(numnp*3=3) then rigid-body description at word 67.
        let mut w = vec![0i32; 74];
        w[15] = 8; // NDIM ⇒ rigid body
        w[16] = 1; // NUMNP
        w[20] = 1; // IU
        w[67] = 1; // nrigid = 1
        w[68] = 0; // mrigid
        w[69] = 0; // numnodr
        w[70] = 0; // numnoda
        let c = control_from_words(&w);
        assert!(!c.reduced_rigid);
        assert_eq!(c.n_rigids, 1);
        // rigid-body motion tail = 1*24 = 24; + time(1) + node(3) = 28.
        assert_eq!(c.bytes_per_state(), 28 * 4);
        // Tail-block offsets chain correctly: rigid block is last, at word 4.
        assert_eq!(c.rigid_off(), TIME_WORDS + 3); // time(1)+node(3), no elements
        assert_eq!(c.rigid_vars(), 24);
    }

    #[test]
    fn tail_block_offsets_chain_in_read_order() {
        // Solids + SPH + element deletion: verify deletion precedes SPH (read order)
        // and every tail offset chains to bytes_per_state.
        let mut w = vec![0i32; 80];
        w[15] = 3; // NDIM
        w[16] = 2; // NUMNP
        w[20] = 1; // IU
        w[23] = 4; // NEL8 (solids)
        w[27] = 7; // NV3D
        w[36] = -10004; // MAXINT ⇒ mdlopt 2 (element deletion), 4 layers
        w[37] = 3; // NMSPH
        w[64] = 11; // isphfg1
        for k in 65..=74 {
            w[k] = 1; // isphfg2..11 = 1 → n_sph_vars = 9 + 0hist? isphfg9=1,others 1
        }
        let c = control_from_words(&w);
        let time_glob_node = TIME_WORDS + 0 + 2 * 3; // time + no globals + iu*numnp*3
        let elements = 4 * 7; // solids
        assert_eq!(c.deletion_off(), time_glob_node + elements);
        // deletion words (mdlopt 2) = nel8+nel4+nel2+nelth = 4.
        assert_eq!(c.sph_off(), c.deletion_off() + 4);
        // sph before airbag/road/rigid (all zero here) → rigid_off == end of sph.
        assert_eq!(c.rigid_off(), c.sph_off() + c.sph_words());
        assert_eq!(c.bytes_per_state(), c.rigid_off() as u64 * 4);
    }

    #[test]
    fn it_flag_decodes_node_thermal_words() {
        assert_eq!(therm_vars_for_it(0), 0); // none
        assert_eq!(therm_vars_for_it(1), 1); // temperature
        assert_eq!(therm_vars_for_it(2), 4); // temperature + heat flux
        assert_eq!(therm_vars_for_it(3), 6); // temperature layers (3) + heat flux (3)
        assert_eq!(therm_vars_for_it(11), 2); // temperature + mass scaling
        assert_eq!(therm_vars_for_it(13), 7); // temp layers + flux + mass scaling
    }

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
        w.set_solid_results(ResultBlock::new([2, 1, 5], (0..2 * 5).map(|i| i as f64).collect()));
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
        w.set_solid_results(ResultBlock::new(
            [2, 1, 7],
            vec![
                100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, // state 0: uniaxial 100, eps 0.1
                0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.3, // state 1: pure shear 50, eps 0.3
            ],
        ));
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

        // Streaming reductions (read off the mmap, no materialization) must agree
        // exactly with the materialized element:: reductions above.
        let vm_s = d.part_max_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert_eq!(vm_s, vm);
        let ff_s = d
            .part_failure_fraction_history(StateBlock::Solid, 1, 0.2, element::effective_plastic_strain)
            .unwrap();
        assert_eq!(ff_s, ff);

        // Argmax: 1-element part → element 0 wins every state, value == part max.
        let am = d.part_argmax_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert_eq!(am.iter().map(|&(e, _)| e).collect::<Vec<_>>(), vec![0, 0]);
        assert!((am[0].1 - vm[0]).abs() < 1e-9 && (am[1].1 - vm[1]).abs() < 1e-9);

        // Full per-element history matrix (n_states, n_part_elems). 1-elem part →
        // it flattens to the per-state max, and column 0 is that element's curve.
        let (mat, mdims, cols) =
            d.part_element_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert_eq!(mdims, [2, 1]);
        assert_eq!(cols, vec![0]);
        assert_eq!(mat, vm);
        let (pmat, _, _) =
            d.part_element_history(StateBlock::Solid, 1, element::effective_plastic_strain).unwrap();
        assert!((pmat[0] - 0.1).abs() < 1e-4 && (pmat[1] - 0.3).abs() < 1e-4, "{pmat:?}");

        // Single-element O(1) random access: state 1, element 0 = pure shear 50.
        let e = d.element_result(StateBlock::Solid, 1, 0).unwrap();
        let expect = [0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.3]; // results are stored f32
        assert!(e.iter().zip(expect).all(|(a, b)| (a - b).abs() < 1e-5), "{e:?}");
        assert!((element::von_mises_stress(&e) - 3.0f64.sqrt() * 50.0).abs() < 1e-2);
        assert!(d.element_result(StateBlock::Solid, 2, 0).is_none()); // state out of range
        assert!(d.element_result(StateBlock::Solid, 0, 1).is_none()); // elem out of range
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn shell_multilayer_round_trip() {
        use crate::results::element::{self, LayerSelect};
        // 2 shells, part 1, 3 layers, per-layer = 6 stress + 1 pstrain (neips 0),
        // so nv2d = 21. 1 state. Each layer uniaxial so von Mises == |sxx|.
        let nodes: Vec<f64> = (0..4 * 3).map(|i| i as f64).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1);
        w.add_shell([1, 2, 3, 4], 1);
        w.set_part_ids(vec![5]);
        w.set_shell_layers(3);
        let uni = |sxx: f64, eps: f64| [sxx, 0.0, 0.0, 0.0, 0.0, 0.0, eps];
        // shell 0 layers: bottom 100/top 300 ; shell 1: bottom 400/top 200
        let mut data = Vec::new();
        for lay in [(100.0, 0.01), (200.0, 0.02), (300.0, 0.03)] {
            data.extend_from_slice(&uni(lay.0, lay.1)); // shell 0, layers b/m/t
        }
        for lay in [(400.0, 0.04), (150.0, 0.015), (200.0, 0.02)] {
            data.extend_from_slice(&uni(lay.0, lay.1)); // shell 1
        }
        w.set_shell_results(ResultBlock::new([1, 2, 21], data));
        let disp: Vec<f64> = nodes.iter().map(|&c| c + 1.0).collect();
        w.add_state(0.0, disp, None, None).unwrap();
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let ly = d.shell_layout();
        assert_eq!((ly.n_layers, ly.stride, ly.has_stress, ly.has_pstrain), (3, 7, true, true));

        let s0 = d.element_result(StateBlock::Shell, 0, 0).unwrap();
        assert!((element::shell_von_mises(&s0, &ly, LayerSelect::Bottom) - 100.0).abs() < 1e-2);
        assert!((element::shell_von_mises(&s0, &ly, LayerSelect::Mid) - 200.0).abs() < 1e-2);
        assert!((element::shell_von_mises(&s0, &ly, LayerSelect::Top) - 300.0).abs() < 1e-2);
        assert!((element::shell_von_mises(&s0, &ly, LayerSelect::Max) - 300.0).abs() < 1e-2);
        assert!((element::shell_plastic_strain(&s0, &ly, LayerSelect::Max) - 0.03).abs() < 1e-4);

        // Streaming reduction over the part with a shell-aware closure (worst layer).
        let vm_max = d
            .part_max_history(StateBlock::Shell, 1, |rec| {
                element::shell_von_mises(rec, &ly, LayerSelect::Max)
            })
            .unwrap();
        assert!((vm_max[0] - 400.0).abs() < 1e-2, "{vm_max:?}"); // shell 1 bottom = 400 wins
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn part_element_history_and_argmax_over_multiple_elements() {
        use crate::results::element;
        // 4 solids, parts 1,2,1,2 → part 1 = elements {0,2}. nv=7, 2 states.
        // Each element uniaxial (sxx only) so von Mises == |sxx|.
        let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        for &pt in &[1, 2, 1, 2] {
            w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], pt);
        }
        w.set_part_ids(vec![10, 20]);
        let uni = |sxx: f64| [sxx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // state 0: e0=100 e1=10 e2=150 e3=30 ; state 1: e0=200 e1=20 e2=50 e3=40
        let mut data = Vec::new();
        for &sxx in &[100.0, 10.0, 150.0, 30.0, 200.0, 20.0, 50.0, 40.0] {
            data.extend_from_slice(&uni(sxx));
        }
        w.set_solid_results(ResultBlock::new([2, 4, 7], data));
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();
        let d = D3plot::open(&p).unwrap();

        // Full matrix: part 1 = elements {0,2}, so columns map to those.
        let (mat, dims, cols) =
            d.part_element_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert_eq!(dims, [2, 2]);
        assert_eq!(cols, vec![0, 2]);
        // row s = [vm(e0), vm(e2)]: [[100,150],[200,50]]
        assert!(mat.iter().zip([100.0, 150.0, 200.0, 50.0]).all(|(a, b)| (a - b).abs() < 1e-2), "{mat:?}");

        // part max is the row-wise max of the matrix.
        let vm = d.part_max_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert!((vm[0] - 150.0).abs() < 1e-2 && (vm[1] - 200.0).abs() < 1e-2, "{vm:?}");

        // argmax picks the winning element index (block order): state0→e2, state1→e0.
        let am = d.part_argmax_history(StateBlock::Solid, 1, element::von_mises_stress).unwrap();
        assert_eq!(am[0].0, 2);
        assert_eq!(am[1].0, 0);
        assert!((am[0].1 - 150.0).abs() < 1e-2 && (am[1].1 - 200.0).abs() < 1e-2);

        // A single element's history == the matching matrix column, via element_result.
        let e2_hist: Vec<f64> = (0..d.num_states())
            .map(|s| element::von_mises_stress(&d.element_result(StateBlock::Solid, s, 2).unwrap()))
            .collect();
        assert!((e2_hist[0] - 150.0).abs() < 1e-2 && (e2_hist[1] - 50.0).abs() < 1e-2, "{e2_hist:?}");
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

    /// f32 round-trip of a flat block through the reader.
    fn assert_block_eq(got: &BlockArray, expect: &[f64]) {
        if let BlockArray::F32(v) = got {
            assert_eq!(v.len(), expect.len());
            for (i, (&a, &b)) in v.iter().zip(expect).enumerate() {
                assert!((a as f64 - b).abs() < 1e-4, "idx {i}: {a} != {b}");
            }
        } else {
            panic!("expected single-precision block");
        }
    }

    #[test]
    fn beams_and_tshells_round_trip() {
        let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64 * 0.1).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_beam([1, 2, 3], 1);
        w.add_tshell([1, 2, 3, 4, 5, 6, 7, 8], 2);
        let beam: Vec<f64> = (0..2 * 6).map(|i| i as f64).collect(); // 2 states, 1 beam, nv1d=6
        let tsh: Vec<f64> = (0..2 * 8).map(|i| 100.0 + i as f64).collect(); // nv3dt=8
        w.set_beam_results(ResultBlock::new([2, 1, 6], beam.clone()));
        w.set_tshell_results(ResultBlock::new([2, 1, 8], tsh.clone()));
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        // read back through the ResultBlock accessors
        let bl = d.beam_results().unwrap();
        assert_eq!(bl.dims(), [2, 1, 6]);
        assert_eq!(bl.data(), beam.as_slice());
        let ts = d.tshell_results().unwrap();
        assert_eq!(ts.dims(), [2, 1, 8]);
        assert_eq!(ts.data(), tsh.as_slice());
        // component naming + per-state slicing on the object
        assert_eq!(bl.component("sigma_xx").unwrap(), vec![0.0, 6.0]);
        assert_eq!(ts.state(1).unwrap(), &tsh[8..]);
        // nodal motion still aligned with the extra element blocks present.
        let c1 = d.node_coordinates(1).unwrap();
        assert!((c1[0] - 1.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn globals_and_part_fields_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1);
        w.add_shell([1, 2, 3, 4], 2); // two parts
        w.set_global_history(GlobalField::KineticEnergy, vec![1., 2., 3.]);
        w.set_global_history(GlobalField::TotalEnergy, vec![10., 11., 12.]);
        w.set_part_field(PartField::InternalEnergy, vec![1., 2., 3., 4., 5., 6.]);
        w.set_part_field(PartField::Mass, vec![10., 20., 30., 40., 50., 60.]);
        for s in 0..3 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.global_history(GlobalField::KineticEnergy).unwrap(), vec![1., 2., 3.]);
        assert_eq!(d.global_history(GlobalField::TotalEnergy).unwrap(), vec![10., 11., 12.]);
        // unset scalar reads back as zeros (slot still written).
        assert_eq!(d.global_history(GlobalField::InternalEnergy).unwrap(), vec![0., 0., 0.]);
        let ie = d.part_field_history(PartField::InternalEnergy).unwrap();
        assert_eq!(ie.dims(), [3, 2]);
        assert_eq!(ie.data(), &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(ie.state(1).unwrap(), &[3., 4.]); // state 1 row
        let mass = d.part_field_history(PartField::Mass).unwrap();
        assert_eq!(mass.data(), &[10., 20., 30., 40., 50., 60.]);
        // unset part field reads back as zeros.
        let ke = d.part_field_history(PartField::KineticEnergy).unwrap();
        assert!(ke.data().iter().all(|&x| x == 0.0));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn node_thermal_fields_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let numnp = 4;
        let ns = 2;
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1);
        let temp: Vec<f64> = (0..ns * numnp).map(|i| i as f64).collect();
        let flux: Vec<f64> = (0..ns * numnp * 3).map(|i| 100.0 + i as f64).collect();
        let mass: Vec<f64> = (0..ns * numnp).map(|i| 200.0 + i as f64).collect();
        let grad: Vec<f64> = (0..ns * numnp).map(|i| 300.0 + i as f64).collect();
        let rf: Vec<f64> = (0..ns * numnp * 3).map(|i| 400.0 + i as f64).collect();
        let rm: Vec<f64> = (0..ns * numnp * 3).map(|i| 500.0 + i as f64).collect();
        w.set_node_field(NodeField::Temperature, temp.clone());
        w.set_node_field(NodeField::HeatFlux, flux.clone());
        w.set_node_field(NodeField::MassScaling, mass.clone());
        w.set_node_field(NodeField::TemperatureGradient, grad.clone());
        w.set_node_field(NodeField::ResidualForce, rf.clone());
        w.set_node_field(NodeField::ResidualMoment, rm.clone());
        for s in 0..ns {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.node_field(NodeField::Temperature, 0).unwrap().data(), &temp[..numnp]);
        assert_eq!(d.node_field(NodeField::HeatFlux, 1).unwrap().data(), &flux[numnp * 3..]);
        assert_eq!(d.node_field(NodeField::MassScaling, 0).unwrap().data(), &mass[..numnp]);
        assert_eq!(d.node_field(NodeField::TemperatureGradient, 1).unwrap().data(), &grad[numnp..]);
        assert_eq!(d.node_field(NodeField::ResidualForce, 0).unwrap().data(), &rf[..numnp * 3]);
        assert_eq!(d.node_field(NodeField::ResidualMoment, 1).unwrap().data(), &rm[numnp * 3..]);
        // 3-vector fields are component-labeled
        assert_eq!(d.node_field(NodeField::HeatFlux, 1).unwrap().components(), &["x", "y", "z"]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn element_deletion_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1);
        w.add_shell([1, 2, 3, 4], 1);
        // state 0: both alive; state 1: shell 2 deleted.
        w.set_element_deletion(StateBlock::Shell, vec![1., 1., 1., 0.]);
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.element_alive(StateBlock::Shell, 0).unwrap(), vec![1., 1.]);
        assert_eq!(d.element_alive(StateBlock::Shell, 1).unwrap(), vec![1., 0.]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn sph_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        // 3 particles, 4 vars each, 2 states.
        let results: Vec<f64> = (0..2 * 3 * 4).map(|i| i as f64).collect();
        w.set_sph(vec![11, 12, 13], 4, results.clone());
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let v0 = d.sph_state(0).unwrap();
        assert_eq!(v0.dims(), [3, 4]);
        assert_eq!(v0.data(), &results[..12]);
        assert_eq!(d.sph_state(1).unwrap().data(), &results[12..]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn airbag_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        let (na, npart, ngeom, nav, npv) = (2usize, 3usize, 2usize, 2usize, 4usize);
        let geom: Vec<f64> = (0..na * ngeom).map(|i| i as f64).collect();
        let ab: Vec<f64> = (0..2 * na * nav).map(|i| 10.0 + i as f64).collect();
        let pt: Vec<f64> = (0..2 * npart * npv).map(|i| 100.0 + i as f64).collect();
        w.set_airbag(na, npart, ngeom, nav, npv, geom, ab.clone(), pt.clone());
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let st = d.airbag_state(1).unwrap();
        assert_eq!(st.airbag_dims, [na, nav]);
        assert_eq!(st.particle_dims, [npart, npv]);
        assert_eq!(st.airbag, ab[na * nav..]);
        assert_eq!(st.particle, pt[npart * npv..]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rigid_body_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        // no rigid road ⇒ full 24-var motion.
        let motion: Vec<f64> = (0..2 * 2 * 24).map(|i| i as f64).collect();
        w.set_rigid_bodies(
            vec![(1, vec![1, 2], vec![1]), (2, vec![3, 4], vec![4])],
            motion.clone(),
        );
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let m0 = d.rigid_body_motion(0).unwrap();
        assert_eq!(m0.dims(), [2, 24]);
        assert_eq!(m0.data(), &motion[..48]);
        assert_eq!(d.rigid_body_motion(1).unwrap().data(), &motion[48..]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rigid_road_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        let road_nodes: Vec<f64> = (0..3 * 3).map(|i| i as f64 * 0.5).collect();
        let motion: Vec<f64> = (0..2 * 1 * 6).map(|i| i as f64).collect();
        w.set_rigid_road(
            vec![1, 2, 3],
            road_nodes,
            vec![(5, vec![1, 2, 3, 3])],
            motion.clone(),
        );
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let r0 = d.rigid_road_state(0).unwrap();
        assert_eq!(r0.dims(), [1, 6]);
        assert_eq!(r0.data(), &motion[..6]);
        assert_eq!(d.rigid_road_state(1).unwrap().data(), &motion[6..]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rigid_body_and_road_reduced_round_trip() {
        // rigid body + rigid road ⇒ NDIM=9, reduced 12-var rigid motion.
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        let rb_motion: Vec<f64> = (0..2 * 1 * 12).map(|i| i as f64).collect();
        let road_motion: Vec<f64> = (0..2 * 1 * 6).map(|i| 100.0 + i as f64).collect();
        w.set_rigid_bodies(vec![(1, vec![1, 2], vec![1])], rb_motion.clone());
        w.set_rigid_road(
            vec![1, 2],
            vec![0., 0., 0., 1., 0., 0.],
            vec![(7, vec![1, 2, 2, 2])],
            road_motion.clone(),
        );
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let rb = d.rigid_body_motion(1).unwrap();
        assert_eq!(rb.dims(), [1, 12]);
        assert_eq!(rb.data(), &rb_motion[12..]);
        let rr = d.rigid_road_state(1).unwrap();
        assert_eq!(rr.dims(), [1, 6]);
        assert_eq!(rr.data(), &road_motion[6..]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn combined_blocks_keep_state_offsets_aligned() {
        // globals + node thermal + beams + element deletion together: every block
        // offset must still line up when the whole stack is present.
        let nodes: Vec<f64> = (0..4 * 3).map(|i| i as f64).collect();
        let numnp = 4;
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_beam([1, 2, 3], 1);
        w.set_global_history(GlobalField::TotalEnergy, vec![5., 6.]);
        w.set_node_field(NodeField::Temperature, (0..2 * numnp).map(|i| i as f64).collect());
        w.set_beam_results(ResultBlock::new([2, 1, 3], (0..2 * 3).map(|i| 9.0 + i as f64).collect()));
        w.set_element_deletion(StateBlock::Beam, vec![1., 1.]);
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, Some(vec![7.0; numnp * 3]), None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.global_history(GlobalField::TotalEnergy).unwrap(), vec![5., 6.]);
        assert_eq!(d.node_field(NodeField::Temperature, 1).unwrap().data(), &[4., 5., 6., 7.]);
        let all = d.resolve_states(None).unwrap();
        let (vel, _) = d.block_data(StateBlock::Velocity, &all).unwrap();
        assert_block_eq(&vel, &vec![7.0; 2 * numnp * 3]);
        let (beam, bd) = d.block_data(StateBlock::Beam, &all).unwrap();
        assert_eq!(bd, [2, 1, 3]);
        assert_block_eq(&beam, &(0..2 * 3).map(|i| 9.0 + i as f64).collect::<Vec<_>>());
        assert_eq!(d.element_alive(StateBlock::Beam, 1).unwrap(), vec![1.]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn node_deletion_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1);
        // state 0 all alive; state 1 node 3 deleted.
        w.set_node_deletion(vec![1., 1., 1., 1., 1., 1., 0., 1.]);
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.node_alive(0).unwrap(), vec![1., 1., 1., 1.]);
        assert_eq!(d.node_alive(1).unwrap(), vec![1., 1., 0., 1.]);
        // node deletion (mdlopt 1) ⇒ no per-element flags.
        assert!(d.element_alive(StateBlock::Shell, 0).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rigid_wall_round_trip() {
        let nodes: Vec<f64> = vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.];
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_shell([1, 2, 3, 4], 1); // one part ⇒ nmmat = 1
        // 2 walls, force + position, 2 states.
        let force: Vec<f64> = (0..2 * 2).map(|i| i as f64).collect();
        let pos: Vec<f64> = (0..2 * 2 * 3).map(|i| 100.0 + i as f64).collect();
        w.set_rigid_walls(2, force.clone(), Some(pos.clone()));
        for s in 0..2 {
            let disp: Vec<f64> = nodes.iter().map(|&c| c + s as f64).collect();
            w.add_state(s as f64, disp, None, None).unwrap();
        }
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        let f1 = d.rigid_wall_force(1).unwrap();
        assert_eq!(f1.dims(), [2]);
        assert_eq!(f1.data(), &force[2..4]);
        let p1 = d.rigid_wall_position(1).unwrap();
        assert_eq!(p1.dims(), [2, 3]);
        assert_eq!(p1.components(), &["x", "y", "z"]);
        assert_eq!(p1.data(), &pos[6..12]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn double_precision_preserves_f64() {
        // 0.1 and 1/3 aren't exact in f32; double precision must round-trip them.
        let coords: Vec<f64> = vec![0.1, 0.2, 0.3, 1.0 / 3.0, 0.7, 0.9, 1.1, 1.3, 1.7, 2.2, 2.4, 2.6];
        let mut w = D3plotWriter::new(coords.clone()).unwrap();
        w.set_double_precision(true);
        w.add_solid([1, 2, 3, 4, 1, 2, 3, 4], 1);
        let sres = vec![1.0 / 3.0; 7]; // 1 state, 1 solid, 7 vars
        w.set_solid_results(ResultBlock::new([1, 1, 7], sres.clone()));
        w.add_state(0.25, coords.clone(), None, None).unwrap();
        let p = tmp();
        w.write(&p).unwrap();

        let d = D3plot::open(&p).unwrap();
        assert_eq!(d.control().wordsize, 8); // reader detected double precision
        // bit-exact: no f32 rounding.
        assert_eq!(d.node_coordinates(0).unwrap(), coords);
        let solid = d.solid_results().unwrap();
        assert_eq!(solid.data(), sres.as_slice());
        assert_eq!(solid.component("eff_plastic_strain").unwrap(), vec![1.0 / 3.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn validate_rejects_wrong_length() {
        let nodes: Vec<f64> = (0..8 * 3).map(|i| i as f64).collect();
        let mut w = D3plotWriter::new(nodes.clone()).unwrap();
        w.add_solid([1, 2, 3, 4, 5, 6, 7, 8], 1);
        // 2 states × 1 solid × 5 vars ⇒ 10 values expected, provide 9.
        w.set_solid_results(ResultBlock::new([2, 1, 5], vec![0.0; 9]));
        for s in 0..2 {
            w.add_state(s as f64, nodes.clone(), None, None).unwrap();
        }
        assert!(w.validate().is_err());
        let p = tmp();
        assert!(w.write(&p).is_err()); // write validates first
        let _ = std::fs::remove_file(&p);
    }
}
