# Plan: one deck→keyword→card→field spine, driven by the generated schema

## The one idea

There is **one handle** for a keyword occurrence, reached by several axes, and it
mirrors the LS-DYNA structure as four nesting levels:

```
Deck  →  Keyword (one *KEYWORD block)  →  Card (one data row)  →  Field (one slot)
```

Everything a reader can do — navigate by id, iterate by name, validate fields,
pull columns in bulk — is a view over this spine. Field layout is **never
hand-coded**: it comes from the generated `keywords` table, which is the single
schema authority. And because a deck can contain a keyword we have *no* schema
for (rare, vendor-specific, or newer than our `kwd.json` snapshot), the spine is
split into two layers:

- **Document layer** — blocks, rows, raw tokens, source locations, round-trip.
  Always available. Needs no schema.
- **Schema layer** — field *names*, *types*, *references*, and entity *identity*.
  An enrichment applied when a schema resolves; absent (not broken) when it
  doesn't.

This split is what makes "one clean way to do everything" true even for keywords
we've never seen: you always get rows and positional raw fields; you get named,
typed, reference-following fields whenever a schema is available.

## Why the current code misses this

- Two near-identical handles — `model::Entity` (by id) and `validate::Keyword`
  (by name) — both wrap `(deck, file, block)`, both read via
  `model::entity_field`, differing only by an optional `(kind, id)` and nav
  sugar. `Entity::field`/`Keyword::field`, `Entity::file`/`Keyword::file`, and
  `Entity::line`/`Keyword::line` (which reimplements `schema::block_line`) are
  the same logic written twice.
- Two identical `Value` unions (`model::Value`, `validate::Value`) with a
  pointless conversion in `Keyword::field`.
- Field access has **no card level and no row iteration** — `field(name)` scans
  all cards and returns the *first* matching value, so repeated/tabular keywords
  (`*NODE`, `*ELEMENT_*`, `*DEFINE_CURVE` points, `*SET_*`) can't be read at all
  through navigation.
- Three "keyword" names collide: `crate::keywords` (the schema library),
  `crate::keyword` (actually the parsed-file/block model — misnamed), and the
  `Keyword` handle. Plus `dynars::Keyword` is already taken by the
  `#[derive(Keyword)]` macro.
- Two disconnected field-access worlds: navigation (`entity_field`, scalar,
  read-only) and columnar marshalling (`schema::parse_schema`, `KeywordFile`,
  reached through a *different entry point* that knows nothing about includes,
  identity, or references). Same job, no shared vocabulary.

## Module layout

| now | becomes | holds |
| --- | --- | --- |
| `crate::keyword` (misnamed) | **`crate::block`** | `Source`, `Block`, `CardFormat`, `ParsedFile` — the parsed-document model (round-trip + edits). Named for the `Block` it's built on. |
| `IncludeKind`/`IncludeDirective`/`IncludeNode` (in `keyword.rs`) + `include_tree.rs` | **`crate::include`** | one home for include directives and the include tree (they already share these types). |
| `crate::keywords` | `crate::keywords` (unchanged name) | the **schema authority**: the generated `Kw`/`Fld` table, `find`/`schema`/`names`, `Ref`/`EntityKind`, and (new) def-side metadata — see below. |
| `crate::model` | **`crate::deck`-adjacent core** (keep `model` or rename to `nav`) | the `Keyword`/`Card`/`Field`/`Value` spine + resolution indices (`Defs`, `Sites`). The primary public surface. |
| `crate::schema` | `crate::schema` | the marshaller (`Schema`/`FieldSpec`/`Table`/`parse_schema`) — now for **user-defined** schemas + the columnar fast path, not a second copy of built-in layouts. |
| `crate::validate` | `crate::validate` | pure consumer: `Rule`/`Check`/`Expr`/`Cmp`/`Report`. No handle, no `Value`, no keyword-iter of its own. |

## The generated table is the only schema authority

`keywords::Kw { name, cards: &[&[Fld]] }` with `Fld { n, t, w, r }` already
carries everything the spine needs — name, type, fixed-width, and reference
target — for thousands of keywords. The Field/Card layer is a thin typed view
over `keywords::find(base)`; nothing about field layout is written by hand.

Two authority items to consolidate into `keywords` (today they live elsewhere or
are scattered):

1. **Def-side metadata** — "this keyword *defines* entity kind K, with its id at
   card C field 0, one-per-block vs one-per-line." This is `model::DEF_RULES`
   today. Move it next to the table as `keywords::definition_of(base) ->
   Option<DefSpec>`, so identity (`Keyword::id`/`kind`), `build_defs`, and
   `build_sites` all read one source. (Left hand-maintained for now; a later
   codegen pass can populate it from `kwd.json` link/define codes — noted, not
   required.)

2. **Repeat / conditional semantics** — the generated schema is the *static*
   layout; a minority of keywords have cards that repeat `N` times or appear only
   under a flag (`keywords/mod.rs` scope note). This knowledge is currently
   split across `parse_schema(repeat)`, `model::DEF_RULES.per_line`, and the
   `typed`/`Columns` row iterator. Fold it into the schema as a per-card
   `repeat` marker so **one** `Keyword::cards()` implementation serves both a
   scalar control card and a million `*NODE` rows.

Also collapse the static-vs-runtime schema duplication: `keywords::Kw`/`Fld`
(static, built-in) and `schema::Schema`/`FieldSpec` (owned, runtime) describe the
same thing, bridged by `Kw::to_schema()`. Keep the static `Kw`/`Fld` as *the
built-in library the spine reads directly* (zero-alloc); keep runtime `Schema`
only for **user-supplied** schemas.

## Schema resolution + graceful degradation (the "no generated keyword" case)

When a `Keyword` needs its schema, resolve in this order:

1. a **user schema** registered on the deck for this base (escape hatch for
   rare/new/vendor keywords — see below),
2. the hand-written **`SUPPLEMENT`** (`NODE`, `PART`, …),
3. the generated **`data::KEYWORDS`** table,
4. **none** → raw-only mode.

Behavior by layer:

- **Always works, schema or not:** `name()`, `base()`, `file()`, `line()`,
  `cards()` / `card(i)` (rows come from splitting the block body, not the
  schema), positional `card.at(col)` / `card.raw(col)`, and `Field::raw()` /
  `as_i64()` / `as_f64()` (a numeric parse of the raw token still succeeds on an
  unknown keyword).
- **Needs a schema, returns `None`/raw without one:** `field(name)` (name→slot
  lookup), `Field::name()`, typed `Field::value()` (falls back to
  `Value::Str(raw)`), `Field::reference()`, and `Keyword::id()`/`kind()` +
  reference sugar.
- `Keyword::has_schema()` lets a caller check up front.

**Escape hatch:** the deck holds an optional user-schema overlay consulted first
(`deck.register_schema(Schema)` / supplied at parse), so an analyst hitting a
keyword we don't ship can describe it once and get full named/typed/reference
access — the same runtime `Schema` the columnar path and `#[derive(Keyword)]`
already produce. No fork in the API; just a schema that happens to come from the
user instead of the table.

## The handle types (core)

```rust
// ── Deck — retrieval axes, each yields Keyword occurrences ──
impl Deck {
    fn keywords(&self, name: &str) -> impl Iterator<Item = Keyword<'_>>;       // by name (1:N)
    fn get(&self, kind: EntityKind, id: i64) -> Option<Keyword<'_>>;           // by identity
    fn part(&self, id: i64) -> Option<Keyword<'_>>;  // section/material/curve …
    fn entities(&self, kind: EntityKind) -> impl Iterator<Item = Keyword<'_>>; // by kind
    fn parts(&self) -> impl Iterator<Item = Keyword<'_>>;  // sections/materials …
    fn register_schema(&mut self, schema: Schema);   // user schema for unknown keywords
}

// ── Keyword — one occurrence = one *KEYWORD block ──
struct Keyword<'d> { /* deck, file, block */ }
impl Keyword<'d> {
    // document layer — always available
    fn name(&self) -> &str;                 // "SECTION_SHELL_TITLE" (exact)
    fn base(&self) -> &str;                 // "SECTION_SHELL" (canonical)
    fn file(&self) -> &Path;
    fn line(&self) -> usize;
    fn has_schema(&self) -> bool;
    fn cards(&self) -> impl Iterator<Item = Card<'d>>;   // one per data row (repeats included)
    fn card(&self, i: usize) -> Option<Card<'d>>;

    // schema layer — None / raw without a schema
    fn field(&self, name: &str) -> Option<Field<'d>>;    // flatten: first field named X across cards
    fn id(&self)   -> Option<i64>;                       // when this block defines an entity
    fn kind(&self) -> Option<EntityKind>;
    fn reference(&self, field: &str) -> Option<Keyword<'d>>;
    fn reference_to(&self, kind: EntityKind) -> Option<Keyword<'d>>;
    fn material(&self) -> Option<Keyword<'d>>;  // section/eos/hourglass sugar
}

// ── Card — one data row + (optionally) the schema for that row ──
struct Card<'d> { /* deck, file, block, row */ }
impl Card<'d> {
    fn raw(&self, col: usize) -> Option<&str>;           // untyped token — never needs schema
    fn at(&self, col: usize)  -> Option<Field<'d>>;      // positional field
    fn field(&self, name: &str) -> Option<Field<'d>>;    // by name — needs schema
    fn fields(&self) -> impl Iterator<Item = Field<'d>>; // schema-driven when present
    fn line(&self) -> &str;                              // the whole data line
}

// ── Field — an addressed slot: value + name + type + position + ref ──
struct Field<'d> { /* … */ }
impl Field<'d> {
    fn value(&self) -> Value;               // typed with schema; Str(raw) without
    fn as_i64(&self) -> Option<i64>;
    fn as_f64(&self) -> Option<f64>;
    fn as_str(&self) -> Option<&str>;
    fn raw(&self) -> &str;                  // the untrimmed source slice
    fn name(&self) -> Option<&str>;         // Some(schema name) / None if positional-only
    fn reference(&self) -> Option<Keyword<'d>>;  // follow, if this slot is a Ref
}
```

Field access reads the same verb at every altitude, precision on demand:

```rust
deck.keywords("SECTION_SHELL").next()?.field("NIP")?.as_i64()   // flatten shortcut
deck.section(2)?.card(0)?.field("NIP")?.as_i64()                // precise: control card
deck.section(2)?.card(1)?.field("T1")?.as_f64()                // precise: thickness card

for node in deck.keywords("NODE").flat_map(|k| k.cards()) {     // tabular — impossible today
    let nid = node.field("NID").and_then(|f| f.as_i64());
    let x   = node.at(1).and_then(|f| f.as_f64());              // positional works schema-or-not
}

part.field("secid")?.reference()      // the field carries its Ref::To(Section)
part.section()                        // sugar: first field whose Ref targets Section
```

Notes:
- `Keyword::field(name)` stays as **sugar** ("first field named X across my
  cards"); the canonical address is `card(i).field(name)`, used when a name
  repeats across cards or you're targeting a slot to edit.
- References move onto `Field` (it already knows its `Ref` from the schema),
  instead of `Entity::reference(name)` re-parsing the slot.
- One `Keyword` handle parses its data rows once and shares them across field
  reads, replacing the per-call `data_lines` re-split in `entity_field`.

## One `Value`

Keep a single core `Value { Int, Float, Str }` in the core module, with the pure
accessors (`as_i64`, `as_f64`, `as_str`, `display`). Delete `validate::Value`.
Fold the comparison into `Cmp` as `cmp.test(&a, &b)` (it already has `test_num`),
so `Cmp`-coupled logic sits with `Cmp` and `Value` stays pure data. `validate`
re-exports the core `Value` for callers that expect `dynars::validate::Value`.

## validate becomes a pure consumer

- Delete `validate/keyword.rs` and `Deck::keywords` in `validate/mod.rs`; rules
  do `for kw in deck.keywords(name)` importing the core `Keyword`.
- `Expr::eval(&Keyword)` and `FieldPredicate` use the core `Keyword`/`Value`.
- `Rule::field_forbidden_values` / `pred` take the core `Value`.
- Built-ins that iterate occurrences (`FieldForbiddenValues`, `FieldRequired`)
  are unchanged in spirit — same `kw.field(...)`, now against the core handle.

## Python surface

- `PyEntity` → the one handle (rename to `PyKeyword`, keep `Entity` alias or drop
  it — see open decision). Add `cards()` / `card(i)` / positional `at` and the
  `has_schema` flag; keep `field(name)` returning native `int|float|str|None`.
  **Python keeps no `Value` type** — values stay native scalars (they already
  are; the stub has no `Value` class). This is smaller fallout than the old plan
  implied.
- `py_to_value` builds the core `Value`; rule constructors unchanged.
- Expose `deck.keywords(name)` to Python — the natural "iterate my
  `*SECTION_SHELL`s" API for analysts — yielding `PyKeyword`.
- Consider `deck.register_schema(...)` from Python for unknown keywords.

## Phasing

**Phase 1 — the spine + rename (breaking, do first while there are no users).**
✅ **Done.** Renamed `keyword`→`file` (not `block` — the module holds
`Source`/`Block`/`ParsedFile`, so `file` fit better), split includes into
`crate::include`. Introduced `Keyword`/`Card`/`Field` in the core with the
document/schema split and graceful degradation; `keywords(name)`, `get/part/…`,
`entities/parts/…` all return `Keyword`; `Entity` collapsed into it (identity
optional, derived on demand). One core `Value`; `validate` consumes it. Demos,
`.pyi`, Python bindings updated.

**Phase 2 — schema authority consolidation.** ✅ **Done (with one deferral).**
- `DEF_RULES` moved → `keywords::definition_of(base) -> Option<DefSpec>` (the
  single def-side authority; folds modifier detection in, so modifiers/controls
  return `None`). `collect_defs`, `build_sites`, `Keyword::id`/`kind` all read it.
- Per-card repeat lives in `keywords::Kw::card_for_row(i)` — the one place that
  knows how a keyword's cards tile over rows. A `per_line` def's single card
  repeats over every row, so `*NODE`/`*ELEMENT_*` rows now type through the schema
  (`card(i).field("nid")`), which they couldn't before. *Deferral:* repeating
  **tails** of list keywords (`*SET_*`, `*DEFINE_CURVE` points) are still mapped
  1:1 and fall back to raw positional access — the acknowledged minority; needs
  per-keyword head/tail metadata (a Phase 4 codegen candidate).
- Static `Kw`/`Fld` is already the sole built-in layout the **navigation spine**
  reads directly (zero-alloc). Runtime `Schema` remains only on the columnar/user
  path; unifying that entry point is Phase 3, not a duplication to remove now.

**Phase 3 — unify the two field-access worlds.** ✅ **Done (with two deferrals).**
- The columnar marshaller (`schema::parse_schema`) generalized from one file to a
  whole deck: `schema::parse_schema_files(&[ParsedFile], &Schema)` collects chunks
  across the root **and every include** and merges columns in file order;
  `parse_schema(&ParsedFile, …)` is now a thin wrapper over it.
- Bulk/columnar is a method on the spine: `Deck::table(keyword) -> Option<Table>`
  (built-in schema) and `Deck::table_with(&Schema) -> Table` (user schema). Same
  keyword names and field names as `Deck::keywords` navigation — one vocabulary,
  fast columnar path underneath. `Table::column(field)` gives the column, so
  `deck.table("NODE")?.column("nid")` is the realized form of the plan's
  aspirational `deck.keywords(name).column(...)`.
- Python: the **deck** is now the include-aware columnar entry too —
  `PyDeck::table` / `table_with`, sharing the lowering (`build_schema`) and dict
  conversion (`table_to_pydict`) with the per-file `KeywordFile`. `KeywordFile`
  stays as the single-file edit/round-trip adapter over the *same* marshaller, no
  longer a parallel universe. `.pyi` + README updated.
- The **navigation-side user-schema overlay** is now done (was a deferral):
  `Deck::register_schema(Schema)` stores a schema keyed by canonical base; the
  spine consults it *first* when resolving field layout, via a `CardRef<'d>`
  {`Static(&'static [Fld])` | `User(&'d [FieldSpec])`} that one `Card`/`Field`
  impl reads. So `deck.keywords("VENDOR_WIDGET").card(0).field("mass")` gets
  named, typed access for a keyword we ship no layout for — the same runtime
  `Schema` the columnar path and `#[derive(Keyword)]` produce, no API fork.
  Layout only: user schemas don't participate in entity-definition/reference
  resolution (that stays on the built-in table), and carry no `Ref` metadata.
  Python `PyDeck::register_schema` + `.pyi` + README updated.
- *Remaining deferral:* no iterator-level `.column` sugar
  (`deck.keywords(name).column`) — `deck.table(name).column(field)` covers it
  without a bespoke iterator type.

**Phase 4 (optional) — generate def-side metadata from `kwd.json`.**
❌ **Investigated, not worth doing — premise doesn't hold.** kwd.json has **no
define/primary-key codes**: field attributes are only
`{default, help, name, position, type, width, link, options, transform, used}`,
`link` appears solely on *reference* fields, and the defining primary-key field
is unmarked (`link=None`). So there is nothing to "generate from link/define
codes." An audit of the current hand rules against kwd.json found them **already
correct** — all 1032 rule-classified definers have an integer primary-key first
field (0 mismatches), the id is always at card0/field0 (so a generated `id_card`
would be a constant `0`), and `per_line` is only ever `ELEMENT_`. A per-keyword
generated table would be ~1032 identical-shaped entries — pure bloat, zero
behavioral change, family→kind knowledge still hand-authored. The compact prefix
rules in `keywords::definition_of` already prefix-generalize and are kept as-is.
(Only tiny real gap: `Sensor`/`Define` kinds are referenced but never defined —
a coverage note, not a generation task.)

## Fallout checklist

- `examples/validate_demo.rs` (`Value::Int` pattern, `deck.keywords` import),
  `examples/nav_demo.rs`, `examples/model_demo.rs`.
- `crate::keyword::` imports across the tree (`include_tree.rs`, `parser.rs`,
  `schema.rs`, `deck.rs`, tests) → `crate::block::` / `crate::include::`.
- `schema::block_line` — delete; the handle's single `line()` replaces both it
  and the inline `Entity::line` counter.
- `.pyi` stub: `Keyword`/`Entity`, `cards`/`card`/`at`, `has_schema`.
- README "Keyword marshalling" section once bulk unifies (phase 3).

## One open decision

`dynars::Keyword` is currently the `#[derive(Keyword)]` macro. The occurrence
handle is the star public type and wants that name. Proposed: the handle is
`dynars::Keyword` (re-exported from the core), and the derive macros move to
`dynars::derive::{Card, Keyword}`. If you'd rather leave the derive macro at the
root, the handle stays `dynars::model::Keyword` (not re-exported at root). Either
works; everything else in this plan is independent of the choice.
