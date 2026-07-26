"""
High-performance LS-DYNA keyword file include tree parser.
"""

import enum
from collections.abc import Sequence
from typing import Any, final

import numpy as np
import numpy.typing as npt

@final
class StateBlock(enum.Enum):
    """A d3plot per-state result block (pass to `D3plot.block`)."""
    Displacement = ...
    Velocity = ...
    Acceleration = ...
    Solid = ...
    ThickShell = ...
    Beam = ...
    Shell = ...

@final
class InterfaceField(enum.Enum):
    """A per-segment field in an intfor file (pass to `D3plot.segment_field`)."""
    Wear = ...
    Pressure = ...
    Shear = ...
    Force = ...
    Gap = ...

@final
class FsiforField(enum.Enum):
    """A per-segment field in an FSIFOR (ALE) file (pass to `D3plot.segment_field`)."""
    Pressure = ...
    ForceX = ...
    ForceY = ...
    ForceZ = ...
    RelativeVelocity = ...
    VelocityX = ...
    VelocityY = ...
    VelocityZ = ...

@final
class IncludeNode:
    def __repr__(self, /) -> str: ...
    @property
    def byte_count(self, /) -> int: ...
    @property
    def children(self, /) -> list[IncludeNode]: ...
    @property
    def kind(self, /) -> str |None: ...
    @property
    def path(self, /) -> str: ...
    def total_bytes(self, /) -> int:
        """
        Total bytes across all files in this subtree.
        """
    def total_files(self, /) -> int:
        """
        Total number of files in this subtree (including self).
        """

@final
class KeywordFile:
    """
    A parsed LS-DYNA keyword file: keyword blocks with lossless round-trip,
    columnar bulk access as numpy arrays, and block-level editing.
    """
    def __repr__(self, /) -> str: ...
    def block_names(self, /) -> list[str]:
        """
        The keyword name of every block, in file order.
        """
    @property
    def dirty(self, /) -> bool:
        """
        Whether any block has a pending edit.
        """
    def keyword(self, /, index: int) -> dict:
        """
        A block as a dict: `{"name": str, "options": [str], "cards": [[str]]}`.
        """
    @property
    def num_blocks(self, /) -> int:
        """
        Number of keyword blocks in the file.
        """
    def parse_builtin(self, /, keyword: str) -> dict:
        """
        Parse a keyword using dynars' built-in library (generated from the
        pyDYNA field database), returning the same column dict. Errors if the
        keyword is not in the library.
        """
    def parse_schema(self, /, keyword: str, cards: Sequence[Sequence[tuple[str, str, int, int]]], repeat: bool = False) -> dict:
        """
        Parse a keyword against a user-defined schema, returning a dict of
        columns (numpy arrays for numeric fields, lists for strings).
        
        Low-level: the Python `@keyword` class layer lowers to this. `cards`
        is a list of cards, each a list of `(name, type, width, count)` field
        tuples where `type` is "int" | "float" | "str".
        """
    def set_keyword(self, /, index: int, name: str, cards: Sequence[Sequence[str]], options: Sequence[str] |None = None) -> None:
        """
        Replace a block's keyword. Cards are re-emitted in free format; the
        rest of the file stays byte-for-byte intact.
        """
    def to_bytes(self, /) -> bytes:
        """
        The (possibly edited) file contents as bytes.
        """
    def write(self, /, path: str) -> None:
        """
        Write the (possibly edited) file to disk.
        """

@final
class Binout:
    """
    LS-DYNA binout reader: walk the LSDA tree by path, read channels as numpy.
    """
    def __init__(self, /, pattern: str) -> None: ...
    def __repr__(self, /) -> str: ...
    @property
    def files(self, /) -> list[str]:
        """The binout files backing this reader, in order."""
    def read(self, /, path: Sequence[str] = ...) -> npt.NDArray[Any] | list[str]:
        """
        Read at `path` (list of segments). A leaf returns a numpy array of the
        channel's native dtype; a directory returns list[str] of child names.
        Empty path returns the top-level datasets.
        """
    def read_many(self, /, paths: Sequence[Sequence[str]]) -> list[npt.NDArray[Any] | list[str]]:
        """
        Read many paths concurrently (lock-free, GIL released), returning a list
        aligned with `paths`. Faster than a Python loop when pulling many channels.
        """
    def read_f64(self, /, path: Sequence[str]) -> npt.NDArray[np.float64]:
        """Read a leaf and coerce to float64 (any numeric dtype)."""
    def read_time_series(self, /, path: Sequence[str]) -> dict:
        """
        Read a time-history: `{"time": float64[T], "values": float64[T], "channel": str}`.
        """
    def channels(self, /, path: Sequence[str] = ...) -> list[str]:
        """Child names at a directory path (empty path = top level)."""

@final
class D3plot:
    """
    LS-DYNA d3plot reader: control block, geometry, per-state nodal results.
    """
    def __init__(self, /, path: str) -> None: ...
    def __repr__(self, /) -> str: ...
    @property
    def num_nodes(self, /) -> int: ...
    @property
    def num_states(self, /) -> int: ...
    def times(self, /) -> npt.NDArray[np.float64]:
        """Simulation time of each state, as a float64 array."""
    def node_coordinates(self, /, state: int) -> npt.NDArray[np.float64]:
        """Deformed node coordinates at `state` (0-based) as an (NUMNP, NDIM) array."""
    def displacement_magnitudes(self, /, state: int) -> npt.NDArray[np.float64]:
        """Per-node displacement magnitude at `state` as a (NUMNP,) array."""
    def max_displacement_final(self, /) -> float:
        """Peak nodal displacement magnitude at the final state."""
    def initial_coordinates(self, /) -> npt.NDArray[np.floating]:
        """Initial (reference) node coordinates as an (N, 3) array."""
    def shell_connectivity(self, /) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """(conn (n_shells, 4) one-based node numbers, parts (n_shells,))."""
    def solid_connectivity(self, /) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """(conn (n_solids, 8) one-based node numbers, parts (n_solids,))."""
    def node_ids(self, /) -> npt.NDArray[np.int64]:
        """User node IDs (default 1..=N)."""
    def shell_ids(self, /) -> npt.NDArray[np.int64]:
        """User shell element IDs."""
    def solid_ids(self, /) -> npt.NDArray[np.int64]:
        """User solid element IDs."""
    def part_ids(self, /) -> npt.NDArray[np.int64]:
        """User part/material IDs."""
    @property
    def filetype(self, /) -> int:
        """Control-block file type (1 = d3plot, 4 = intfor, ...)."""
    @property
    def is_interface_force(self, /) -> bool:
        """
        Whether this is an interface-force (intfor) database. In an intfor file
        the contact segments are in the shell slot: block(StateBlock.Shell) gives
        (n_states, n_segments, nv2d); split with interface_fields().
        """
    @property
    def is_fsifor(self, /) -> bool:
        """Whether this is an FSIFOR (ALE) interface-force file (use FsiforField)."""
    def segment_field(
        self, /, field: InterfaceField | FsiforField, states: int | Sequence[int] | None = None
    ) -> npt.NDArray[np.floating]:
        """
        Extract one interface-force field's values from the per-segment block as
        (n_states, n_segments, k). `field` is an InterfaceField (intfor) or
        FsiforField (FSIFOR). Raises if the field is absent from this file.
        """
    def available_blocks(self, /) -> list[StateBlock]:
        """The result blocks present in this d3plot."""
    def block(self, /, block: StateBlock, states: int | Sequence[int] | None = None) -> npt.NDArray[np.floating]:
        """
        Generic result extraction: a result block as an (n, count, vars) array in
        the file's native precision (float32 single / float64 double). `block` is
        a StateBlock. `states` selects which states: None = all; an int (negatives
        from the end) = one; a sequence = those. When the selection is
        single-precision and contiguous within one family file, the result is a
        zero-copy read-only view over the memory map; otherwise it is copied (in
        parallel for large blocks). Node blocks are (..., 3); element blocks
        return the solver's raw packed per-entity layout for you to reshape.
        """
    def block_layout(self, /, block: StateBlock) -> tuple[int, int] | None:
        """The (count, vars_per_entity) layout of a result block, or None."""

@final
class BinoutEditor:
    """
    Editable binout: a directory tree of typed datasets that writes back a
    complete LSDA file. Construct new, or open an existing file and mutate it
    (save re-emits the whole file).
    """
    def __init__(self, /, path: str | None = None) -> None: ...
    def list(self, /, path: Sequence[str] = ...) -> list[str] | None:
        """Child names at a directory path (empty = top level); None if it's a dataset."""
    def get(self, /, path: Sequence[str]) -> npt.NDArray[Any] | str | None:
        """The dataset at `path` as a numpy array / str, or None."""
    def set(self, /, path: Sequence[str], values: npt.NDArray[Any] | Sequence[float] | str) -> None:
        """Create or overwrite the dataset at `path` (parent dirs autocreated)."""
    def remove(self, /, path: Sequence[str]) -> bool:
        """Remove the dataset/directory at `path`; returns whether it existed."""
    def to_bytes(self, /) -> bytes:
        """The whole tree serialized as LSDA bytes."""
    def write(self, /, path: str) -> None:
        """Write the whole tree to `path` as an LSDA (binout) file."""

@final
class D3plotWriter:
    """
    Build a single-precision d3plot from a mesh (nodes + shell/solid
    connectivity) and per-state nodal results (deformed coordinates, and
    optionally velocity/acceleration). v1 scope: NDIM=4 structural layout,
    implicit numbering, no global variables, no per-element result fields.
    Output reads back through dynars and open-lasso-python.
    """
    def __init__(self, /, node_coords: npt.NDArray[np.floating], title: str | None = None) -> None: ...
    def add_shells(self, /, conn: npt.NDArray[np.integer], parts: Sequence[int] | None = None) -> None:
        """Add shells: `conn` is (M, 4) one-based node ids; `parts` optional (M,) part ids."""
    def add_solids(self, /, conn: npt.NDArray[np.integer], parts: Sequence[int] | None = None) -> None:
        """Add solids: `conn` is (M, 8) one-based node ids; `parts` optional (M,) part ids."""
    def set_ids(
        self,
        /,
        node_ids: Sequence[int] | None = None,
        shell_ids: Sequence[int] | None = None,
        solid_ids: Sequence[int] | None = None,
        part_ids: Sequence[int] | None = None,
    ) -> None:
        """User IDs written into the NARBS numbering section (default 1..N)."""
    def set_solid_results(self, /, results: npt.NDArray[np.floating]) -> None:
        """Per-solid result block (n_states, n_solids, vars). Sets NV3D."""
    def set_shell_results(self, /, results: npt.NDArray[np.floating]) -> None:
        """Per-shell result block (n_states, n_shells, vars). Sets NV2D."""
    def add_state(
        self,
        /,
        time: float,
        disp: npt.NDArray[np.floating],
        vel: npt.NDArray[np.floating] | None = None,
        acc: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Append a state: `time`, deformed coords `disp` (N,3), optional `vel`/`acc` (N,3)."""
    def to_bytes(self, /) -> bytes:
        """The d3plot as bytes."""
    def write(self, /, path: str) -> None:
        """Write the d3plot to `path`."""

@final
class D3plotEditor:
    """
    Edit an existing d3plot family in place: overwrite node coordinates or a
    result block at chosen states; everything else is preserved byte-for-byte.
    """
    def __init__(self, /, path: str) -> None: ...
    @property
    def num_nodes(self, /) -> int: ...
    @property
    def num_states(self, /) -> int: ...
    def set_block(self, /, block: StateBlock, state: int, data: npt.NDArray[np.floating]) -> None:
        """Overwrite a result block at `state` with `data` (count, vars)."""
    def set_node_coordinates(self, /, state: int, coords: npt.NDArray[np.floating]) -> None:
        """Overwrite deformed node coordinates (N, 3) at `state`."""
    def save(self, /) -> None:
        """Overwrite the original files in place."""
    def write(self, /, path: str) -> None:
        """Write the edited family to a new base path (`path`, `path01`, ...)."""

@final
class IntforWriter:
    """
    Build an interface-force (intfor) file: contact segments + per-state nodal
    motion (displacement, velocity) + per-segment interface values (pressure,
    shear, forces, gap — or the FSIFOR/ALE fixed layout). Reads back through
    dynars; validate in LS-PrePost before relying on it.
    """
    def __init__(self, /, node_coords: npt.NDArray[np.floating], n_interfaces: int = 1, title: str | None = None) -> None: ...
    def add_segments(self, /, conn: npt.NDArray[np.integer], ids: Sequence[int] | None = None) -> None:
        """Add contact segments: `conn` is (M, 4) one-based node ids; `ids` optional (M,)."""
    def set_node_ids(self, /, node_ids: Sequence[int]) -> None:
        """User node IDs (length N) for the NARBS numbering section."""
    def set_fields(self, /, wear: int = 0, pressure: int = 0, shear: int = 0, force: int = 0, gap: int = 0) -> None:
        """Declare the intfor per-segment field layout (nv2d = their sum)."""
    def set_fsifor(self, /, n: int) -> None:
        """Mark this an FSIFOR (ALE) file with `n` fixed per-segment values."""
    @property
    def nv2d(self, /) -> int:
        """Values per segment in each state."""
    def add_state(
        self,
        /,
        time: float,
        disp: npt.NDArray[np.floating],
        vel: npt.NDArray[np.floating],
        segment_values: npt.NDArray[np.floating],
    ) -> None:
        """Append a state: time, disp (N,3), vel (N,3), segment_values (n_segments, nv2d)."""
    def to_bytes(self, /) -> bytes:
        """The intfor file as bytes."""
    def write(self, /, path: str) -> None:
        """Write the intfor file to `path`."""

def parse_binout(pattern: str) -> Binout:
    """Open an LS-DYNA binout for reading."""

def open_d3plot(path: str) -> D3plot:
    """Open an LS-DYNA d3plot for reading."""

def parse_include_tree(path: str) -> IncludeNode:
    """
    Parse an LS-DYNA keyword file and return the include tree.
    
    Releases the GIL during parsing so other Python threads can run.
    """

def parse_keyword_file(path: str) -> KeywordFile:
    """
    Parse an LS-DYNA keyword file into an editable [`PyKeywordFile`].
    
    Releases the GIL during the file read and block split.
    """

# ── Deck: parse once, validate + navigate ────────────────────────────────────

@final
class Severity(enum.Enum):
    """How serious a validation finding is."""
    Error = ...
    Warning = ...
    Info = ...

@final
class Cmp(enum.Enum):
    """A comparison operator for field predicates."""
    Eq = ...
    Ne = ...
    Lt = ...
    Le = ...
    Gt = ...
    Ge = ...

@final
class Predicate:
    """A boolean predicate tree over card fields (evaluated in Rust)."""
    @staticmethod
    def field(field: str, cmp: Cmp, value: int | float | str) -> Predicate:
        """`field <cmp> value`."""
    @staticmethod
    def all_(preds: Sequence[Predicate]) -> Predicate:
        """All sub-predicates must hold (logical AND)."""
    @staticmethod
    def any_(preds: Sequence[Predicate]) -> Predicate:
        """Any sub-predicate holds (logical OR)."""
    @staticmethod
    def not_(pred: Predicate) -> Predicate:
        """Negation."""

@final
class Rule:
    """A built-in declarative validation rule. Constructed in Python, run in Rust."""
    @staticmethod
    def keyword_forbidden(keyword: str) -> Rule: ...
    @staticmethod
    def field_forbidden_values(keyword: str, field: str, values: Sequence[int | float | str]) -> Rule: ...
    @staticmethod
    def field_required(keyword: str, require: Predicate, when: Predicate | None = None) -> Rule: ...
    @staticmethod
    def include_missing() -> Rule:
        """Every `*INCLUDE` must resolve to a file that exists."""
    @staticmethod
    def references_resolve() -> Rule:
        """Every cross-keyword id reference resolves (PART.mid -> *MAT, *LOAD.lcid -> *DEFINE_CURVE, ...). Does not check element connectivity."""
    @staticmethod
    def references_resolve_with_connectivity() -> Rule:
        """As `references_resolve`, and additionally checks every element's nodes are defined. Heavy on large meshes."""
    def with_severity(self, severity: Severity) -> Rule: ...
    def only_in(self, patterns: Sequence[str]) -> Rule: ...
    def except_in(self, patterns: Sequence[str]) -> Rule: ...

@final
class Finding:
    """One rule violation with a clickable `file:line`."""
    @property
    def rule(self) -> str: ...
    @property
    def severity(self) -> Severity: ...
    @property
    def keyword(self) -> str: ...
    @property
    def file(self) -> str: ...
    @property
    def line(self) -> int: ...
    @property
    def message(self) -> str: ...
    def location(self) -> str:
        """`file:line`."""

@final
class Report:
    """The result of a validation run."""
    @property
    def findings(self) -> list[Finding]: ...
    def is_clean(self) -> bool:
        """True if there are no Error-severity findings."""
    def count(self, severity: Severity) -> int: ...
    def __len__(self) -> int: ...

@final
class Entity:
    """A handle to one definition entity: fields, source location, reference-following."""
    @property
    def id(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def keyword(self) -> str:
        """The exact keyword defining this entity (e.g. `MAT_RIGID_TITLE`)."""
    @property
    def file(self) -> str:
        """The include file this entity is defined in."""
    @property
    def line(self) -> int:
        """1-based line of the entity's `*KEYWORD` line (jump-to location)."""
    @property
    def offsets(self) -> dict[str, int] | None:
        """Effective ``*INCLUDE_TRANSFORM`` id offsets applied to this entity's
        file (``idnoff``, ``ideoff``, …), or ``None`` outside a transform."""
    def field(self, name: str) -> int | float | str | None:
        """Read a card field by name (case-insensitive)."""
    def reference(self, name: str) -> Entity | None:
        """Follow the reference in field `name` to the entity it points at."""
    def material(self) -> Entity | None: ...
    def section(self) -> Entity | None: ...
    def eos(self) -> Entity | None: ...
    def hourglass(self) -> Entity | None: ...

@final
class Deck:
    """
    A parsed LS-DYNA deck (root + all includes).

    Parse once with `parse_deck`, then validate and navigate off the same
    object. Resolution indices are built lazily on first use.
    """
    def __init__(self, path: str) -> None: ...
    def validate(self, rules: Sequence[Rule]) -> Report:
        """Run a set of rules over this deck (reuses the parse). No default rule set."""
    def part(self, id: int) -> Entity | None: ...
    def material(self, id: int) -> Entity | None: ...
    def section(self, id: int) -> Entity | None: ...
    def curve(self, id: int) -> Entity | None: ...
    def parts(self) -> list[Entity]:
        """Every part in the deck (enumerate, don't guess ids)."""
    def materials(self) -> list[Entity]: ...
    def sections(self) -> list[Entity]: ...
    def curves(self) -> list[Entity]: ...
    def definition_counts(self) -> list[tuple[str, int]]:
        """`(kind, count)` of defined ids, most-numerous first."""
    def table(self, keyword: str) -> dict:
        """
        Bulk columnar read of a keyword across the whole deck (root + includes)
        using the built-in library: a dict of numpy arrays (numeric fields) and
        string lists. Include-aware, unlike the per-file `KeywordFile`. Raises
        `KeyError` if the keyword is not built in (use `table_with`).
        """
    def table_with(self, keyword: str, cards: Sequence[Sequence[tuple[str, str, int, int]]], repeat: bool = False) -> dict:
        """
        Bulk columnar read across the whole deck against a user-defined schema —
        the escape hatch for a keyword not in the built-in library. Each card is
        a list of `(name, type, width, count)` tuples; `type` is int/float/str.
        """
    def register_schema(self, keyword: str, cards: Sequence[Sequence[tuple[str, str, int, int]]], repeat: bool = False) -> None:
        """
        Register a user schema for a keyword the built-in library doesn't cover,
        so navigation (`keywords`, `part`, …) gets named, typed field access for
        it. Each card is a list of `(name, type, width, count)` tuples; `type`
        is int/float/str. Keyed by canonical base (re-registering replaces).
        """

def parse_deck(path: str) -> Deck:
    """Parse a deck (root + all includes) once and return a navigable `Deck`."""
