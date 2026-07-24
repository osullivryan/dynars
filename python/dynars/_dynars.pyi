"""
High-performance LS-DYNA keyword file include tree parser.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import final

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
    def elements_shell(self, /) -> tuple[Incomplete, Incomplete, Incomplete]:
        """
        `*ELEMENT_SHELL` as `(eids, pids, nodes: int64[N, 4])`.
        """
    def elements_solid(self, /) -> tuple[Incomplete, Incomplete, Incomplete]:
        """
        `*ELEMENT_SOLID` as `(eids, pids, nodes: int64[N, 8])`.
        """
    def keyword(self, /, index: int) -> dict:
        """
        A block as a dict: `{"name": str, "options": [str], "cards": [[str]]}`.
        """
    def nodes(self, /) -> tuple[Incomplete, Incomplete]:
        """
        `*NODE` data as `(ids: int64[N], coords: float64[N, 3])`, zero-copy.
        """
    @property
    def num_blocks(self, /) -> int:
        """
        Number of keyword blocks in the file.
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
    def set_node_coords(self, /, coords: Incomplete) -> None:
        """
        Rewrite all `*NODE` blocks from a new `(N, 3)` coordinate array.
        """
    def to_bytes(self, /) -> bytes:
        """
        The (possibly edited) file contents as bytes.
        """
    def write(self, /, path: str) -> None:
        """
        Write the (possibly edited) file to disk.
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
