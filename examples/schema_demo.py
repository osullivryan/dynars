"""Marshalling keywords with declarative classes — Python API.

    python examples/schema_demo.py   (after `maturin develop`)

Each keyword is a class: fields directly on it form one card, or a ``cards``
list composes several card classes. The ``@keyword`` decorator lowers the class
to a schema that Rust parses — the class is purely the authoring surface, so the
parse runs entirely in Rust and numeric columns come back as numpy arrays.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np

import dynars
from dynars import Card, Float, Int, IntArray, Str, keyword, parse_keyword


# --- single card, repeats over the block (the default) ---
@keyword("NODE")
class Node(Card):
    nid = Int(8)
    x = Float(16)
    y = Float(16)
    z = Float(16)


# --- single card with an array field: 4 nodes -> one (N, 4) column ---
@keyword("ELEMENT_SHELL")
class ElementShell(Card):
    eid = Int(8)
    pid = Int(8)
    nodes = IntArray(4, width=8)


# --- multi-card: reusable card classes composed into one keyword ---
class Heading(Card):
    title = Str(80)


class PartData(Card):
    pid = Int(8)
    secid = Int(8)
    mid = Int(8)


@keyword("PART")
class Part:
    cards = [Heading, PartData]


def main() -> None:
    # *NODE fixed-width; other data cards comma-free — handled per line.
    node_lines = "\n".join(
        f"{i + 1:>8}{x:>16.6f}{y:>16.6f}{z:>16.6f}"
        for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
    )
    deck = f"*KEYWORD\n*NODE\n{node_lines}\n" + textwrap.dedent("""\
        *ELEMENT_SHELL
        1,10,1,2,3,4
        2,10,3,4,1,2
        *PART
        steel bracket
        10,20,1
        aluminium panel
        11,21,2
        *END
        """)

    path = Path(tempfile.gettempdir()) / "dynars_schema_demo.k"
    path.write_text(deck)
    kf = dynars.parse_keyword_file(str(path))

    # Pass the class...
    nodes = parse_keyword(kf, Node)
    print("NODE —", nodes["nid"].shape[0], "rows")
    print("  nid =", nodes["nid"], nodes["nid"].dtype)
    print("  x   =", nodes["x"])

    # ...or the registered keyword name.
    shells = parse_keyword(kf, "ELEMENT_SHELL")
    print("\nELEMENT_SHELL — nodes is a 2-D array:", shells["nodes"].shape)
    print(shells["nodes"])

    parts = parse_keyword(kf, Part)
    print("\nPART —", len(parts["title"]), "rows")
    for title, pid in zip(parts["title"], parts["pid"]):
        print(f"  pid {pid}: {title}")

    # Columns are ordinary numpy arrays — vectorized math just works.
    centroid = np.column_stack([nodes["x"], nodes["y"], nodes["z"]]).mean(axis=0)
    print("\nnode centroid =", centroid)

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
