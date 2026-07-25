"""
High-performance LS-DYNA keyword file include tree parser.
"""

from collections.abc import Sequence
from typing import Any, final

import numpy as np
import numpy.typing as npt

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
    def available_blocks(self, /) -> list[str]:
        """
        Result blocks present: any of 'displacement', 'velocity',
        'acceleration', 'solid', 'tshell', 'beam', 'shell'.
        """
    def block(self, /, name: str) -> npt.NDArray[np.floating]:
        """
        Generic result extraction: a result block across all states as an
        (n_states, count, vars) array in the file's native precision (float32
        for single-precision d3plots, float64 for double). Node blocks are
        (..., 3); element blocks return the solver's raw packed per-entity
        layout (stresses, strains, history vars per integration point/layer)
        for you to reshape.
        """
    def block_layout(self, /, name: str) -> tuple[int, int] | None:
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
