# Concepts

A short tour of the ideas the API is built on. Read it once and the rest of the
guides — and the choices they make — will read easily.

## Two ways in: keyword file vs. deck

dynars gives you two entry points, and picking the right one saves a lot of
friction.

- **`Deck`** (`parse_deck`) is the **root plus every file it includes**, parsed in
  one pass and presented as one model. It navigates by id, follows references,
  bulk-reads columns across every file, validates — and **edits**: change a
  single field (`set_field`) and write the deck back byte-identical everywhere
  else, include-aware. Ids are global across the whole include graph. This is the
  tool you reach for most.
- **`KeywordFile`** (`parse_keyword_file`) is **one standalone file**, seen as a
  flat list of keyword blocks that tile the bytes exactly, with the same lossless
  round-trip and block-level editing. It does *not* follow `*INCLUDE`s — use it
  for a lone file with no include graph around it.

```text
parse_deck("main.k")           -> Deck          (root + all *INCLUDEs; navigate, validate, edit)
parse_keyword_file("part.k")   -> KeywordFile   (one standalone file, editable blocks)
```

Rule of thumb: reach for a **`Deck`** to read, check, or edit a model; reach for a
**`KeywordFile`** only for a lone file with no includes. (See [editing a
deck](decks.md#editing-a-deck-round-trip).)

## The navigation spine

Inside a `Deck`, LS-DYNA's tangle of cross-references is presented as a small,
regular vocabulary:

```text
Deck  ──part(id)──▶  Entity ──material()──▶ Entity ──field("RO")──▶ 7.85e-9
      ──material(id)─▶        ──section()───▶        ──reference("lcid")─▶ *DEFINE_CURVE
      ──section(id)──▶        ──field(name)─▶ value
      ──curve(id)────▶
```

An **`Entity`** (called `Keyword` in Rust) is one occurrence of a keyword. It
knows its `id`, its source `file` and `line`, its typed `field(name)` values, and
how to **follow references** — `.material()`, `.section()`, `.eos()`,
`.hourglass()`, or the generic `.reference(field_name)`. Everything you'd do by
hand — "this part's `mid` points at that `*MAT`" — is one method call, and it's
correct across includes and transforms.

## Ids are global and unsigned

References in a deck are by **id**, and dynars resolves them in a single **global
namespace** per entity kind. Two consequences worth internalizing:

- **The sign is ignored.** LS-DYNA uses signed ids in some fields (a negative
  curve id means "use the absolute value with a flag"). `deck.curve(5)` and a
  reference to `-5` resolve to the same `*DEFINE_CURVE 5`.
- **Ids are logical, post-transform.** When a file is pulled in through an
  `*INCLUDE_TRANSFORM` with an id offset, its entities take their **shifted**
  global ids. `deck.part(id)` and reference-following both speak these global ids,
  so a part in a transformed submodel is reached by the id it actually has in the
  assembled model — not its file-local id.

## Includes and transforms

`parse_deck` walks the whole `*INCLUDE` graph — `*INCLUDE`, `*INCLUDE_PATH`,
`*INCLUDE_TRANSFORM`, and friends — from the root, in parallel, and folds the
result into one model.

- A plain **`*INCLUDE`** contributes its entities unchanged.
- An **`*INCLUDE_TRANSFORM`** applies id offsets (`idnoff`, `ideoff`, …) and a
  geometric transform. dynars applies the **id offsets** when it assembles the
  global namespace, so duplicate-id and dangling-reference checks are correct even
  when the same mesh file is instanced several times with different offsets. An
  entity's effective offsets are visible on `Entity.offsets` (Python).
- A **missing `*INCLUDE`** is *never parsed* — it adds no file and no phantom
  content. It is still recorded as a directive, so
  [`Rule.include_missing()`](validation.md) can flag it. Don't rely on
  `references_resolve` alone to catch a missing file: if that file was the only
  source of some entity kind, references to it are left unflagged by design (the
  dangling check is conservative).

You can inspect the graph itself without a full parse — see [the include
tree](decks.md#the-include-tree).

## Two access patterns: columns vs. handles

The same deck offers two ways to read the same data, tuned for two very different
volumes:

| | **Columnar** (`table`) | **Navigation** (`part`, `keywords`, …) |
|---|---|---|
| Shape | dict of arrays / a `Table` | per-entity `Entity`/`Keyword` handles |
| Best for | `*NODE`, `*ELEMENT_*` — millions of rows | `*PART`, `*MAT`, `*SECTION` — following references |
| Cost | one parallel pass, zero-copy to NumPy | lazy id/reference index, built on first use |

They share one vocabulary — the same keyword names and field names — so you move
between "give me all node coordinates as an array" and "walk this part to its
material" without changing mental gears. Use columns when you want the *numbers*;
use handles when you want the *relationships*.

## Schemas: how a keyword becomes typed

Columnar reads and typed `field(name)` access both rest on a **schema** — a
declaration of a keyword's card layout (each field's name, type, and width).
dynars ships schemas for **~3,170 LS-DYNA keywords**, generated from the pyDYNA
field database, so the common keywords "just work" with no declaration.

For a vendor, rare, or newer-than-our-snapshot keyword, you describe the layout
**once** and it becomes first-class — columns, typed fields, and (in Rust)
dangling-checked references. That's the [Schemas](schemas.md) page. The library
covers each keyword's *static* card layout; conditional or count-driven cards
(e.g. `*DEFINE_CURVE`) fall back to the generic, lazy field model.

## Validation is à la carte

There is **no default rule set**. You hand `validate` exactly the checks you want
and get back a `Report`. This is deliberate: what counts as an error is a policy
decision that differs per team and per model. Rules are typed values you compose,
scope to files, and re-severity — see [Validation](validation.md). Custom logic
that the built-ins don't cover drops to a `Check` (Rust) or a plain loop over the
same views (Python).

## One core, three languages

Rust is the engine. The Python package (PyO3) and the C/Fortran bindings (a C ABI)
drive the *same* code — the parser, the validator, the schema marshaller, and the
signal/injury math are written once in Rust. So a number you get in Python is the
same number Rust computes, and the injury criteria are verified bit-exact against
SciPy. The Python examples in these docs and their Rust tabs aren't
reimplementations of each other; they're two front doors to one house.

## Where to go next

- [Decks & navigation](decks.md) — put the spine to work.
- [Schemas](schemas.md) — teach dynars a keyword it doesn't ship.
- [Validation](validation.md) — assemble a rule set.
- [Results](results.md) — read the binary output the run produced.
