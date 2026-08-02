"""Surgical single-field editing — change one field and write the deck back
byte-identical everywhere else (no whole-deck rewrite, comments/rulers kept).

    python examples/edit_demo.py   (after `maturin develop --features python`)

Builds a tiny two-file deck (root + one *INCLUDE), edits three fields through
three different navigation paths, writes the touched files, and re-parses to
prove the new values round-trip.
"""

import tempfile
from pathlib import Path

import dynars


def cols(*vals: str) -> str:
    """Join values into 10-wide right-justified LS-DYNA columns."""
    return "".join(f"{v:>10}" for v in vals)


def main() -> None:
    d = Path(tempfile.mkdtemp())

    # Root: a control card + a material, then *INCLUDE a second file.
    (d / "root.k").write_text(
        "*KEYWORD\n"
        "*CONTROL_TERMINATION\n"
        "$#  endtim    endcyc     dtmin\n"
        "0.05\n"
        "*MAT_ELASTIC\n"
        "$#     mid        ro         e        pr\n"
        f"{cols('1', '7.85e-9', '2.1e11', '0.3')}\n"
        "*INCLUDE\n"
        "sub.k\n"
        "*END\n"
    )
    # Include: a second material (same keyword, different file).
    (d / "sub.k").write_text(
        "*KEYWORD\n"
        "*MAT_ELASTIC\n"
        f"{cols('2', '2.70e-9', '7.0e10', '0.33')}\n"
        "*END\n"
    )

    deck = dynars.parse_deck(str(d / "root.k"))
    print("files:", [(f.index, Path(f.path).name) for f in deck.files()])

    # 1) by name (schema-aware): retard the termination time.
    term = deck.keywords("CONTROL_TERMINATION")[0]
    print("\nENDTIM before:", term.field("endtim"))
    print("  set ->", term.set_field("endtim", 0.02))

    # 2) by entity id: change material 1's Young's modulus (in root.k).
    mat = deck.material(1)
    print("MAT 1 E before:", mat.field("e"))
    print("  set ->", mat.set_field("e", 2.0e11))

    # 3) file-first: material 2 lives in the include — scope to it.
    sub_mat = deck.file("sub.k").keywords("MAT_ELASTIC")[0]
    print("sub.k MAT E before:", sub_mat.field("e"))
    print("  set ->", sub_mat.set_field("e", 8.0e10))

    # Edits are a write-time overlay — write the touched files (into a new dir).
    out = Path(tempfile.mkdtemp())
    for f in deck.files():
        f.write(str(out / Path(f.path).name))
        if f.dirty:
            print(f"wrote (edited): {Path(f.path).name}")

    # Re-parse and confirm the new values.
    re = dynars.parse_deck(str(out / "root.k"))
    assert abs(re.keywords("CONTROL_TERMINATION")[0].field("endtim") - 0.02) < 1e-12
    assert abs(re.material(1).field("e") - 2.0e11) < 1.0
    assert abs(re.file("sub.k").keywords("MAT_ELASTIC")[0].field("e") - 8.0e10) < 1.0
    print("\nverified: edits round-tripped and re-parse to the new values")


if __name__ == "__main__":
    main()
