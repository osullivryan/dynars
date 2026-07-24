"""Built-in keyword library — parse common LS-DYNA keywords by name with no
declaration at all (generated from the pyDYNA field database).

    python examples/builtin_demo.py   (after `maturin develop`)

Pass a keyword *name* (or the typo-proof `dynars.kw.*` constant) and it resolves
from the built-in library — no `@keyword` class needed.
"""

import tempfile
from pathlib import Path

import dynars
from dynars import kw


def main() -> None:
    node_lines = "\n".join(
        f"{i + 1:>8}{x:>16.6f}{y:>16.6f}{z:>16.6f}"
        for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
    )
    # Two materials = two separate *MAT_ELASTIC blocks (as in a real deck).
    deck = f"*KEYWORD\n*NODE\n{node_lines}\n" + (
        "*MAT_ELASTIC\n1,7.85e-9,210000.0,0.3\n"
        "*MAT_ELASTIC\n2,2.70e-9,70000.0,0.33\n*END\n"
    )
    path = Path(tempfile.gettempdir()) / "dynars_builtin_demo.k"
    path.write_text(deck)
    kf = dynars.parse_keyword_file(str(path))

    # Built-in schema, via the name constant — no @keyword class defined here.
    # parse_keyword gathers *every* *MAT_ELASTIC block into one table.
    n_blocks = kf.block_names().count("MAT_ELASTIC")
    mats = dynars.parse_keyword(kf, kw.MAT_ELASTIC)
    print(f"MAT_ELASTIC — {n_blocks} blocks in file, aggregated into {mats['MID'].shape[0]} rows")
    print("  MID =", mats["MID"], mats["MID"].dtype)
    print("  E   =", mats["E"])

    # *NODE comes from the supplement (pyDYNA omits it); a plain string works too.
    nodes = dynars.parse_keyword(kf, "NODE")
    print("\nNODE (supplement) —", nodes["nid"].shape[0], "rows")
    print("  nid =", nodes["nid"], " x =", nodes["x"])

    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
