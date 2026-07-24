"""Declarative keyword schemas.

Define how to marshal a keyword as a class of typed fields (one card), or a
class listing several card classes (multi-card). A ``@keyword`` decorator lowers
the class to the schema the Rust core parses — the class is purely the authoring
surface, so parsing stays entirely in Rust.

    from dynars import keyword, Card, Int, Float, Str, IntArray, parse_keyword

    @keyword("NODE")                       # one card, repeats over the block
    class Node(Card):
        nid = Int(8)
        x = Float(16); y = Float(16); z = Float(16)

    class Heading(Card):                    # reusable card classes
        title = Str(80)

    class PartData(Card):
        pid = Int(8); secid = Int(8); mid = Int(8)

    @keyword("PART", repeat=True)
    class Part:
        cards = [Heading, PartData]         # multi-card: a list of cards

    cols = parse_keyword(kf, Node)          # {"nid": int64[N], "x": float64[N], ...}
    cols = parse_keyword(kf, "PART")        # by registered name
"""

from __future__ import annotations


class _Field:
    """A field marker placed on a card class attribute."""

    __slots__ = ("ty", "width", "count")

    def __init__(self, ty: str, width: int, count: int = 1):
        self.ty = ty
        self.width = width
        self.count = count


def Int(width: int) -> _Field:
    """A signed-integer field `width` columns wide (fixed format)."""
    return _Field("int", width)


def Float(width: int) -> _Field:
    """A floating-point field `width` columns wide (fixed format)."""
    return _Field("float", width)


def Str(width: int) -> _Field:
    """A string field `width` columns wide (fixed format)."""
    return _Field("str", width)


def IntArray(count: int, width: int) -> _Field:
    """`count` consecutive integer fields, returned as one `(N, count)` column."""
    return _Field("int", width, count)


def FloatArray(count: int, width: int) -> _Field:
    """`count` consecutive float fields, returned as one `(N, count)` column."""
    return _Field("float", width, count)


class Card:
    """Base for a keyword card (one line). Subclass and assign fields in order."""


_REGISTRY: dict[str, type] = {}


def _card_fields(card_cls: type) -> list[tuple[str, str, int, int]]:
    # Field markers in declaration order (class __dict__ preserves insertion
    # order on every supported Python).
    return [
        (name, f.ty, f.width, f.count)
        for name, f in vars(card_cls).items()
        if isinstance(f, _Field)
    ]


def _lower(name: str, cls: type, repeat: bool):
    cards_attr = vars(cls).get("cards")
    card_classes = cards_attr if cards_attr else [cls]
    cards = [_card_fields(cc) for cc in card_classes]
    return (name, cards, repeat)


def keyword(name: str, repeat: bool = True):
    """Register a class as the schema for keyword `name`.

    Fields directly on the class define a single card; a ``cards = [...]`` list
    of card classes defines a multi-card layout. `repeat=True` (the default)
    parses the card group repeatedly over the block body — the common case for
    `*NODE`, `*ELEMENT_*`, multiple `*PART`s, etc. Pass `repeat=False` only to
    read a single entity per block.
    """

    def deco(cls: type) -> type:
        cls._dynars_schema = _lower(name, cls, repeat)
        _REGISTRY[name.upper()] = cls
        return cls

    return deco


def parse_keyword(kf, schema):
    """Parse a keyword from `kf` (a `KeywordFile`) using a `@keyword` class or a
    registered keyword name. Returns a dict of columns: numpy arrays for numeric
    fields, lists for string fields.
    """
    if isinstance(schema, str):
        cls = _REGISTRY.get(schema.upper())
        if cls is None:
            raise KeyError(f"no keyword schema registered for '{schema}'")
    else:
        cls = schema
    name, cards, repeat = cls._dynars_schema
    return kf.parse_schema(name, cards, repeat)
