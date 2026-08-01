"""
High-performance LS-DYNA keyword file include tree parser.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Final, final

@final
class Binout:
    """
    LS-DYNA binout reader: walk the LSDA tree by path, read channels as numpy.
    """
    def __new__(cls, /, pattern: str) -> Binout:
        """
        Open a binout (glob pattern; continuation files `binout%NNN` are
        picked up automatically). Releases the GIL while indexing.
        """
    def __repr__(self, /) -> str: ...
    def channels(self, /, path: Sequence[str] = ...) -> list[str]:
        """
        Child names at a directory path (empty path = top level).
        """
    @property
    def files(self, /) -> list[str]:
        """
        The binout files backing this reader, in order.
        """
    def ids(self, /, branch: str) -> Any:
        """
        LS-DYNA entity IDs for a state branch (e.g. `nodout` node IDs), as int64.
        """
    def legend(self, /, branch: str) -> list[str]:
        """
        Per-entity legend/name strings for a state branch (trimmed).
        """
    def read(self, /, path: Sequence[str] = ...) -> Any:
        """
        Read at `path` (list of segments). A leaf returns a numpy array of
        the channel's native dtype; a directory returns `list[str]` of child
        names. Empty path returns the top-level datasets.
        """
    def read_f64(self, /, path: Sequence[str]) -> Any:
        """
        Read a leaf and coerce to float64 (any numeric dtype).
        """
    def read_many(self, /, paths: Sequence[Sequence[str]]) -> list[Any]:
        """
        Read many paths concurrently (lock-free, GIL released), returning a
        list aligned with `paths`. Faster than a Python loop when pulling
        many channels: the reads run in parallel across cores.
        """
    def read_states(self, /, branch: str, var: str) -> dict:
        """
        Aggregate a per-state variable across all state dirs into a dense matrix:
        `{"time": float64[T], "values": float64[T, C], "ids": int64[C],
        "n_steps": int, "n_channels": int}`. One node's history by ID:
        `values[:, np.nonzero(ids == node_id)[0][0]]`.
        """
    def read_time_series(self, /, path: Sequence[str]) -> dict:
        """
        Read a time-history: `{"time": float64[T], "values": float64[T], "channel": str}`.
        `time` is read from the sibling `time` array, or synthesized as 0..T.
        """
    def title(self, /, branch: str) -> str:
        """
        Dataset title for a state branch.
        """

@final
class BinoutEditor:
    """
    Editable binout: a directory tree of typed datasets that writes back a
    complete LSDA file. Construct new, or open an existing file and mutate it
    (save re-emits the whole file).
    """
    def __new__(cls, /, path: str |None = None) -> BinoutEditor:
        """
        `BinoutEditor()` starts empty; `BinoutEditor(path)` loads an existing
        binout (glob pattern) fully into memory.
        """
    def get(self, /, path: Sequence[str]) -> Any |None:
        """
        The dataset at `path` as a numpy array / str, or None.
        """
    def list(self, /, path: Sequence[str] = ...) -> list[str] |None:
        """
        Child names at a directory path (empty path = top level); None if the
        path is a dataset.
        """
    def remove(self, /, path: Sequence[str]) -> bool:
        """
        Remove the dataset/directory at `path`; returns whether it existed.
        """
    def set(self, /, path: Sequence[str], values: Any) -> None:
        """
        Create or overwrite the dataset at `path` (parent dirs autocreated).
        """
    def to_bytes(self, /) -> bytes:
        """
        The whole tree serialized as LSDA bytes.
        """
    def write(self, /, path: str) -> None:
        """
        Write the whole tree to `path` as an LSDA (binout) file.
        """

@final
class Cmp:
    """
    A comparison operator — used instead of a stringly `"eq"`/`"ne"`.
    """
    Eq: Final[Cmp]
    Ge: Final[Cmp]
    Gt: Final[Cmp]
    Le: Final[Cmp]
    Lt: Final[Cmp]
    Ne: Final[Cmp]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class D3plot:
    """
    LS-DYNA d3plot reader: control block, geometry, per-state nodal results.
    """
    def __new__(cls, /, path: str) -> D3plot:
        """
        Open a d3plot file (single-file, structural layout — see the Rust
        `d3plot` module docs for scope).
        """
    def __repr__(self, /) -> str: ...
    def available_blocks(self, /) -> list[StateBlock]:
        """
        The result blocks present in this d3plot, as `StateBlock` values.
        """
    def block(self, /, block: StateBlock, states: Any |None = None) -> Any:
        """
        Generic result extraction: any result block across all states as an
        `(n_states, count, vars)` numpy array in native precision. `block` is
        a `StateBlock` (or its lowercase name string). Node blocks are
        `(…, 3)`; element blocks return the solver's raw packed per-entity
        layout — reshape by integration points/layers as needed. Raises if
        the block is absent.
        
        `states` selects which states to return: `None` = all; an int (or
        negative int, from the end) = one state; a sequence of ints = those
        states. Selecting fewer states reads/copies only those.
        
        When the selected states are single-precision and contiguous within
        one family file, the result is a **zero-copy** read-only view straight
        over the memory map (no allocation, no copy). Otherwise the selection
        is copied into a fresh array (in parallel for large blocks).
        """
    def block_layout(self, /, block: StateBlock) -> tuple[int, int] |None:
        """
        The `(count, vars_per_entity)` layout of a result block, or None.
        """
    def displacement_magnitudes(self, /, state: int) -> Any:
        """
        Per-node displacement magnitude at `state` as a `(NUMNP,)` array.
        """
    @property
    def filetype(self, /) -> int:
        """
        Control-block file type (1 = d3plot, 4 = intfor, …).
        """
    def initial_coordinates(self, /) -> Any:
        """
        Initial (reference) node coordinates as an `(N, 3)` array.
        """
    @property
    def is_fsifor(self, /) -> bool:
        """
        Whether this is an FSIFOR (ALE) interface-force file — use
        `FsiforField` values with `segment_field`.
        """
    @property
    def is_interface_force(self, /) -> bool:
        """
        Whether this is an interface-force (`intfor`) database. In an intfor
        file the contact **segments** are in the shell slot:
        `block(StateBlock.Shell)` gives `(n_states, n_segments, nv2d)` and
        `shell_connectivity()` the segment nodes; split the per-segment values
        with `interface_fields`.
        """
    def max_displacement_final(self, /) -> float:
        """
        Peak nodal displacement magnitude at the final state.
        """
    def node_coordinates(self, /, state: int) -> Any:
        """
        Deformed node coordinates at `state` (0-based) as an `(NUMNP, 3)` array.
        """
    def node_coordinates_all(self, /) -> Any:
        """
        Deformed node coordinates for every state as a `(num_states, NUMNP, 3)`
        array — one call, one allocation, instead of a Python loop over
        `node_coordinates`.
        """
    def node_ids(self, /) -> Any:
        """
        User node IDs (`N`), default `1..=N`.
        """
    @property
    def num_nodes(self, /) -> int: ...
    @property
    def num_states(self, /) -> int: ...
    def part_ids(self, /) -> Any:
        """
        User part/material IDs.
        """
    def segment_field(self, /, field: Any, states: Any |None = None) -> Any:
        """
        Extract one interface-force field's values from the per-segment block
        as `(n_states, n_segments, k)`. `field` is an `InterfaceField` (intfor)
        or `FsiforField` (FSIFOR) — no magic strings. `states` selects states
        like `block`. Raises if the field isn't present in this file.
        """
    def shell_connectivity(self, /) -> tuple[Any, Any]:
        """
        Shell connectivity: `(conn, parts)` where `conn` is `(n_shells, 4)`
        one-based node numbers and `parts` is `(n_shells,)`.
        """
    def shell_ids(self, /) -> Any:
        """
        User shell element IDs.
        """
    def solid_connectivity(self, /) -> tuple[Any, Any]:
        """
        Solid connectivity: `(conn, parts)` where `conn` is `(n_solids, 8)`.
        """
    def solid_ids(self, /) -> Any:
        """
        User solid element IDs.
        """
    def times(self, /) -> Any:
        """
        Simulation time of each state, as a float64 array.
        """

@final
class D3plotEditor:
    """
    Edit an existing d3plot family in place: overwrite node coordinates or a
    result block at chosen states; everything else is preserved byte-for-byte.
    """
    def __new__(cls, /, path: str) -> D3plotEditor:
        """
        Load a d3plot family (base + `d3plot01`, …) for editing.
        """
    @property
    def num_nodes(self, /) -> int: ...
    @property
    def num_states(self, /) -> int: ...
    def save(self, /) -> None:
        """
        Overwrite the original files in place.
        """
    def set_block(self, /, block: StateBlock, state: int, data: Any) -> None:
        """
        Overwrite a result `block` (a `StateBlock`) at `state` with `data`
        `(count, vars)` — the same layout `D3plot.block(...)` returns.
        """
    def set_node_coordinates(self, /, state: int, coords: Any) -> None:
        """
        Overwrite deformed node coordinates `(N, 3)` at `state`.
        """
    def write(self, /, path: str) -> None:
        """
        Write the edited family to a new base path (`path`, `path01`, …).
        """

@final
class D3plotWriter:
    """
    Build a single-precision d3plot from a mesh + per-state nodal results.
    """
    def __new__(cls, /, node_coords: Any, title: str |None = None) -> D3plotWriter:
        """
        `node_coords` is `(N, 3)` (or flat `3N`) initial coordinates.
        """
    def add_shells(self, /, conn: Incomplete, parts: Sequence[int] |None = None) -> None:
        """
        Add shell elements: `conn` is `(M, 4)` one-based node ids; `parts` is
        an optional `(M,)` part id per shell (default 1).
        """
    def add_solids(self, /, conn: Incomplete, parts: Sequence[int] |None = None) -> None:
        """
        Add solid elements: `conn` is `(M, 8)` one-based node ids; `parts` is
        an optional `(M,)` part id per solid (default 1).
        """
    def add_state(self, /, time: float, disp: Any, vel: Any |None = None, acc: Any |None = None) -> None:
        """
        Append a state: `time`, deformed coords `disp` `(N,3)`, and optional
        `vel`/`acc` `(N,3)`. Velocity/acceleration presence is fixed by the
        first state added.
        """
    def set_ids(self, /, node_ids: Sequence[int] |None = None, shell_ids: Sequence[int] |None = None, solid_ids: Sequence[int] |None = None, part_ids: Sequence[int] |None = None) -> None:
        """
        Set user IDs written into the NARBS numbering section (default 1..N):
        node IDs (length N), shell/solid element IDs, and part IDs.
        """
    def set_shell_results(self, /, results: Any) -> None:
        """
        Per-shell result block, `(n_states, n_shells, vars)`. Sets NV2D.
        """
    def set_solid_results(self, /, results: Any) -> None:
        """
        Per-solid result block, `(n_states, n_solids, vars)` — the same raw
        layout `D3plot.block(StateBlock.Solid)` returns. Sets NV3D.
        """
    def to_bytes(self, /) -> bytes:
        """
        The d3plot as bytes.
        """
    def write(self, /, path: str) -> None:
        """
        Write the d3plot to `path`.
        """

@final
class Deck:
    """
    A parsed LS-DYNA deck (root + all includes). Parse once with
    [`parse_deck`], then validate (`validate`) and navigate
    (`part`, `material`, …) off the same object — no second parse. The
    resolution indices are built lazily on first use.
    """
    def __new__(cls, /, path: str) -> Deck: ...
    def __repr__(self, /) -> str: ...
    def curve(self, /, id: int) -> Entity |None: ...
    def curves(self, /) -> list[Entity]: ...
    def definition_counts(self, /) -> list[tuple[str, int]]:
        """
        `(kind, count)` of defined ids, most-numerous first.
        """
    def material(self, /, id: int) -> Entity |None: ...
    def materials(self, /) -> list[Entity]: ...
    def part(self, /, id: int) -> Entity |None: ...
    def parts(self, /) -> list[Entity]:
        """
        Every part in the deck (enumerate, don't guess ids).
        """
    def register_schema(self, /, keyword: str, cards: Sequence[Sequence[tuple[str, str, int, int]]], repeat: bool = False) -> None:
        """
        Register a user schema for a keyword the built-in library doesn't cover,
        so navigation (`keywords`, `part`, …) gets named, typed field access for
        it. `cards` is a list of cards, each a list of `(name, type, width,
        count)` field tuples; `type` is "int" | "float" | "str". Keyed by
        canonical base — registering the same base twice replaces it.
        """
    def section(self, /, id: int) -> Entity |None: ...
    def sections(self, /) -> list[Entity]: ...
    def table(self, /, keyword: str) -> dict:
        """
        Bulk **columnar** read of every occurrence of `keyword` across the whole
        deck (root + includes) using the built-in library, as a dict of numpy
        arrays (numeric fields) and string lists. The fast path alongside
        `part`/`material`/… navigation — the deck is the one columnar entry,
        include-aware (unlike the per-file `KeywordFile`). Raises `KeyError` if
        the keyword isn't in the built-in library (use `table_with`).
        """
    def table_with(self, /, keyword: str, cards: Sequence[Sequence[tuple[str, str, int, int]]], repeat: bool = False) -> dict:
        """
        Bulk columnar read across the whole deck against a user-defined schema —
        the escape hatch for a keyword not in the built-in library. `cards` is a
        list of cards, each a list of `(name, type, width, count)` field tuples;
        `type` is "int" | "float" | "str".
        """
    def validate(self, /, rules: Sequence[Rule]) -> Report:
        """
        Run a set of rules over this deck, reusing the parse. No default
        rule set — pass the rules you want (e.g. `Rule.references_resolve()`).
        """

@final
class Entity:
    """
    A handle to one entity: typed field access, source location, and
    reference-following. Keeps its [`PyDeck`] alive.
    """
    def __repr__(self, /) -> str: ...
    def eos(self, /) -> Entity |None: ...
    def field(self, /, name: str) -> Any |None:
        """
        Read a field by name (case-insensitive) → int / float / str.
        """
    @property
    def file(self, /) -> str:
        """
        The include file this entity is defined in.
        """
    def hourglass(self, /) -> Entity |None: ...
    @property
    def id(self, /) -> int: ...
    @property
    def keyword(self, /) -> str: ...
    @property
    def kind(self, /) -> str: ...
    @property
    def line(self, /) -> int:
        """
        1-based line of the entity's `*KEYWORD` line (jump-to location).
        """
    def material(self, /) -> Entity |None: ...
    @property
    def offsets(self, /) -> dict[str, int] |None:
        """
        The effective `*INCLUDE_TRANSFORM` offsets applied to this entity's file
        (composed down the include chain) as a dict `{"idnoff": …, "ideoff": …}`,
        or `None` if it sits in the root or a plain `*INCLUDE`. These are the
        shifts that turn the file-local ids into the global ones `id` reports.
        """
    def reference(self, /, name: str) -> Entity |None:
        """
        Follow the reference in field `name` to the entity it points at.
        """
    def section(self, /) -> Entity |None: ...

@final
class Finding:
    """
    One rule violation with a clickable `file:line`.
    """
    def __repr__(self, /) -> str: ...
    @property
    def file(self, /) -> str: ...
    @property
    def keyword(self, /) -> str: ...
    @property
    def line(self, /) -> int: ...
    def location(self, /) -> str: ...
    @property
    def message(self, /) -> str: ...
    @property
    def rule(self, /) -> str: ...
    @property
    def severity(self, /) -> Severity: ...

@final
class FsiforField:
    """
    A per-segment field in an **FSIFOR** (ALE interface-force) file. These are
    single-value fields in this fixed order; the file carries as many as `|nv2d|`.
    Exported to Python as the `FsiforField` enum.
    """
    ForceX: Final[FsiforField]
    ForceY: Final[FsiforField]
    ForceZ: Final[FsiforField]
    Pressure: Final[FsiforField]
    RelativeVelocity: Final[FsiforField]
    VelocityX: Final[FsiforField]
    VelocityY: Final[FsiforField]
    VelocityZ: Final[FsiforField]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

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
class InterfaceField:
    """
    A per-segment field in an interface-force (`intfor`) file. These partition
    the segment result block (`StateBlock::Shell`) in this order and sum to
    `nv2d`. Exported to Python as the `InterfaceField` enum — no magic strings.
    """
    Force: Final[InterfaceField]
    Gap: Final[InterfaceField]
    Pressure: Final[InterfaceField]
    Shear: Final[InterfaceField]
    Wear: Final[InterfaceField]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class IntforWriter:
    """
    Build an interface-force (`intfor`) file: contact segments + per-state
    nodal motion + per-segment interface values.
    """
    def __new__(cls, /, node_coords: Any, n_interfaces: int = 1, title: str |None = None) -> IntforWriter:
        """
        `node_coords` is `(N, 3)`; `n_interfaces` sliding interfaces.
        """
    def add_segments(self, /, conn: Incomplete, ids: Sequence[int] |None = None) -> None:
        """
        Add contact segments: `conn` is `(M, 4)` one-based node ids; `ids` is
        an optional `(M,)` segment id per segment (default 1..M).
        """
    def add_state(self, /, time: float, disp: Any, vel: Any, segment_values: Any) -> None:
        """
        Append a state: `time`, deformed `disp` `(N,3)`, `vel` `(N,3)`, and
        `segment_values` `(n_segments, nv2d)`.
        """
    @property
    def nv2d(self, /) -> int:
        """
        Values per segment in each state.
        """
    def set_fields(self, /, wear: int = 0, pressure: int = 0, shear: int = 0, force: int = 0, gap: int = 0) -> None:
        """
        Declare the intfor per-segment field layout (nv2d = their sum).
        """
    def set_fsifor(self, /, n: int) -> None:
        """
        Mark this an FSIFOR (ALE) file with `n` fixed per-segment values.
        """
    def set_node_ids(self, /, node_ids: Sequence[int]) -> None:
        """
        User node IDs (length N) for the NARBS numbering section.
        """
    def to_bytes(self, /) -> bytes:
        """
        The intfor file as bytes.
        """
    def write(self, /, path: str) -> None:
        """
        Write the intfor file to `path`.
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
class Predicate:
    """
    A boolean predicate tree over card fields (tier 2). Evaluated in Rust.
    """
    @staticmethod
    def all_(preds: Sequence[Predicate]) -> Predicate:
        """
        All sub-predicates must hold (logical AND).
        """
    @staticmethod
    def any_(preds: Sequence[Predicate]) -> Predicate:
        """
        Any sub-predicate holds (logical OR).
        """
    @staticmethod
    def field(field: str, cmp: Cmp, value: Any) -> Predicate:
        """
        `field <cmp> value`.
        """
    @staticmethod
    def not_(pred: Predicate) -> Predicate:
        """
        Negation.
        """

@final
class Report:
    """
    The result of a validation run.
    """
    def __len__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    def count(self, /, severity: Severity) -> int: ...
    @property
    def findings(self, /) -> list[Finding]: ...
    def is_clean(self, /) -> bool: ...

@final
class Rule:
    """
    A built-in declarative rule. Constructed in Python, executed in Rust.
    """
    @staticmethod
    def duplicate_ids() -> Rule:
        """
        No two labelled definition entities of the same kind share an id (two
        *PART pid=5, duplicate *MAT/*SET/*SECTION/*DEFINE_CURVE ids, …). Compared
        on logical ids, so *INCLUDE_TRANSFORM instances don't collide.
        """
    def except_in(self, /, patterns: Sequence[str]) -> Rule:
        """
        Apply everywhere except files whose path contains one of `patterns`.
        """
    @staticmethod
    def field_forbidden_values(keyword: str, field: str, values: Sequence[Any]) -> Rule: ...
    @staticmethod
    def field_required(keyword: str, require: Predicate, when: Predicate |None = None) -> Rule: ...
    @staticmethod
    def include_missing() -> Rule: ...
    @staticmethod
    def keyword_forbidden(keyword: str) -> Rule: ...
    def only_in(self, /, patterns: Sequence[str]) -> Rule:
        """
        Apply only within files whose path contains one of `patterns`.
        """
    @staticmethod
    def references_resolve() -> Rule:
        """
        Cross-keyword referential integrity: every id reference resolves
        (PART.mid → *MAT, *LOAD.lcid → *DEFINE_CURVE, …). Does not check
        element connectivity.
        """
    @staticmethod
    def references_resolve_with_connectivity() -> Rule:
        """
        As `references_resolve`, and additionally checks that every element's
        nodes are defined. Heavy on large meshes.
        """
    @staticmethod
    def rigid_context() -> Rule:
        """
        Rigid-body keywords (*LOAD_RIGID_BODY, *CONSTRAINED_RIGID_BODIES,
        *CONSTRAINED_EXTRA_NODES, *BOUNDARY_PRESCRIBED_MOTION_RIGID, …) must
        target a *MAT_RIGID part; flags a reference to a deformable part.
        """
    @staticmethod
    def unreferenced_entities() -> Rule:
        """
        Library definition entities nothing references — dead *MAT, *SECTION,
        *DEFINE_CURVE, *SET, *DEFINE_COORDINATE, … Reports at Warning severity.
        """
    def with_severity(self, /, severity: Severity) -> Rule:
        """
        Set severity (default Error).
        """

@final
class Severity:
    """
    How serious a violation is.
    """
    Error: Final[Severity]
    Info: Final[Severity]
    Warning: Final[Severity]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class StateBlock:
    """
    A per-entity result block in a state. Node blocks are (N, 3); element blocks
    are (N, vars) where `vars` is the solver's packed per-element layout
    (stresses, plastic strain, history variables, per integration point/layer) —
    returned raw for the caller to reshape.
    
    This is the single source of truth for block identity: the reader/writer use
    it directly, and (with the `python` feature) it is exported to Python as the
    `StateBlock` enum — no magic strings.
    """
    Acceleration: Final[StateBlock]
    Beam: Final[StateBlock]
    Displacement: Final[StateBlock]
    Shell: Final[StateBlock]
    Solid: Final[StateBlock]
    ThickShell: Final[StateBlock]
    Velocity: Final[StateBlock]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class Workspace:
    """
    An in-process batch context: parse and validate many decks that share
    `*INCLUDE`s against one shared cache, so common files (mesh, materials) are
    read, parsed, and indexed **once** no matter how many decks include them.
    
    ```python
    import dynars
    ws = dynars.Workspace()
    decks = ws.parse_decks(["variant_a/main.k", "variant_b/main.k"])
    reports = ws.validate_decks(decks, [
        dynars.Rule.references_resolve(),
        dynars.Rule.duplicate_ids(),
    ])
    print(ws.stats())  # {'files_parsed': ..., 'files_reused': ..., ...}
    ```
    
    The decks handed back are ordinary `Deck`s — validate or navigate them
    individually too; a deck from a workspace reuses the shared indices whether
    you call `validate_decks` or its own `.validate(...)`.
    """
    def __new__(cls, /) -> Workspace: ...
    def __repr__(self, /) -> str: ...
    def parse_deck(self, /, path: str) -> Deck:
        """
        Parse one deck (root + all includes), reusing any file this workspace has
        already read. Returns a navigable `Deck`.
        """
    def parse_decks(self, /, paths: Sequence[str]) -> list[Deck]:
        """
        Parse several decks in one batch, sharing all file work across them.
        Returns a list of `Deck`s in input order; raises `RuntimeError` naming the
        first root that fails to parse.
        """
    def stats(self, /) -> dict[str, int]:
        """
        Cache stats as a dict: `files_parsed` / `files_reused` (disk reads vs.
        cache hits) and `def_indices_built` / `ref_indices_built` (distinct files
        whose definition / connectivity index was extracted — a shared file counts
        once).
        """
    def validate_decks(self, /, decks: Sequence[Deck], rules: Sequence[Rule]) -> list[Report]:
        """
        Validate several decks in parallel against the shared cache. Returns one
        `Report` per deck, in order. Warms the shared definition index first, then
        runs `rules` over every deck concurrently — a shared file's id and
        connectivity indices are built once, not per deck.
        """

def bric(wx: Incomplete, wy: Incomplete, wz: Incomplete, crit_x: float, crit_y: float, crit_z: float) -> float:
    """
    Brain Injury Criterion from the three head angular-velocity channels (rad/s)
    and their critical values.
    """

def butterworth(values: Incomplete, order: int, cutoff: float, fs: float, btype: str = "low") -> Any:
    """
    Zero-phase Butterworth filter: `order`-pole, corner `cutoff` Hz at sample
    rate `fs` Hz, `btype` = "low" or "high".
    """

def cfc(values: Incomplete, cfc: float, dt: float) -> Any:
    """
    Apply an SAE J211 CFC low-pass filter (zero-phase). `cfc` is the class in Hz
    (60/180/600/1000 or any value); `dt` is the sample interval in seconds.
    """

def clip(a: Incomplete, dt: float, window: float = 0.003) -> float:
    """
    The "3 ms clip": highest acceleration (g) sustained for `window` seconds
    (default 3 ms).
    """

def decimate(values: Incomplete, factor: int) -> Any:
    """
    Decimate by an integer factor (keep every Nth sample); new dt = dt·factor.
    Lossless after a CFC/low-pass; use before an O(n·w) criterion (HIC) on very
    fine dt to cut cost by ~1/factor².
    """

def differentiate(values: Incomplete, dt: float) -> Any:
    """
    Central-difference derivative (e.g. velocity → acceleration). Same length as
    `values`.
    """

def filtfilt(b: Sequence[float], a: Sequence[float], values: Incomplete) -> Any:
    """
    Zero-phase forward-backward filtering of a `(b, a)` filter — the analogue of
    `scipy.signal.filtfilt` with default odd padding.
    """

def hic(a: Incomplete, dt: float, window: float = 0.036) -> float:
    """
    Head Injury Criterion over a `window`-second interval; `a` is resultant head
    acceleration in g sampled every `dt` seconds.
    """

def hic15(a: Incomplete, dt: float) -> float:
    """
    HIC15 — [`hic`] over a 15 ms window.
    """

def hic36(a: Incomplete, dt: float) -> float:
    """
    HIC36 — [`hic`] over a 36 ms window.
    """

def integrate(values: Incomplete, dt: float) -> Any:
    """
    Cumulative trapezoidal integral (e.g. acceleration → velocity). Same length
    as `values`, starting at 0.
    """

def nic(a_t1: Incomplete, a_head: Incomplete, dt: float) -> float:
    """
    Rear-impact Neck Injury Criterion NIC (max) from T1 and head accel (m/s²).
    """

def nij(fx: Incomplete, fz: Incomplete, my: Incomplete, distance: float, fzc_te: float, fzc_co: float, myc_fl: float, myc_ex: float) -> float:
    """
    Neck Injury Criterion Nij (max) — see the Rust docs for the signed-critical
    convention (compression/extension criticals are negative).
    """

def open_d3plot(path: str) -> D3plot:
    """
    Open an LS-DYNA d3plot for reading (mirrors [`PyD3plot::new`]).
    """

def parse_binout(pattern: str) -> Binout:
    """
    Open an LS-DYNA binout for reading (mirrors [`PyBinout::new`]).
    """

def parse_deck(path: str) -> Deck:
    """
    Parse a deck (root + all includes) once and return a navigable [`PyDeck`].
    """

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

def resample_linear(values: Incomplete, dt_in: float, dt_out: float) -> Any:
    """
    Linear resample from `dt_in` to `dt_out` (up- or down-sample).
    """

def resultant(x: Incomplete, y: Incomplete, z: Incomplete) -> Any:
    """
    Elementwise resultant magnitude √(x²+y²+z²) of three channels.
    """

def severity_index(a: Incomplete, dt: float) -> float:
    """
    Gadd Severity Index (CSI on a chest resultant): ∫ a^2.5 dt over the pulse.
    """

def tibia_index(mx: Incomplete, my: Incomplete, fz: Incomplete, critical_bending_moment: float, critical_compression_force: float) -> float:
    """
    Tibia Index (max) from bending moments (N·m) and axial force (N).
    """

def ubric(wx: Incomplete, wy: Incomplete, wz: Incomplete, ax: Incomplete, ay: Incomplete, az: Incomplete, crit_wx: float, crit_wy: float, crit_wz: float, crit_ax: float, crit_ay: float, crit_az: float) -> float:
    """
    Universal Brain Injury Criterion (uBRIC) from angular velocity + acceleration
    channels and their critical values.
    """

def vc(y: Incomplete, dt: float, scaling_factor: float, deformation_constant: float) -> float:
    """
    Viscous Criterion (VC)max from a chest deflection channel `y` (m).
    """
