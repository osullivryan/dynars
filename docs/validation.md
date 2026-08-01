# Validation

Rules run over a parsed `Deck`, reusing the parse and fanning out across cores.
There is **no default rule set** — you pass exactly the checks you want and get
back a `Report` of findings, each with a clickable `file:line`.

## Built-in rules

=== "Python"

    ```python
    from dynars import Rule, Predicate, Cmp, Severity

    report = deck.validate([
        Rule.references_resolve(),                                # ids resolve
        Rule.duplicate_ids(),                                     # no id collisions
        Rule.unreferenced_entities(),                             # dead *MAT/*SET/... (warns)
        Rule.include_missing(),                                   # missing *INCLUDEs
        Rule.field_forbidden_values("MAT_ELASTIC", "PR", [0.5]),  # PR may not be 0.5
        Rule.field_required(                                      # if ELFORM==2, NIP > 0
            "SECTION_SHELL",
            require=Predicate.field("NIP", Cmp.Gt, 0),
            when=Predicate.field("ELFORM", Cmp.Eq, 2),
        ),
    ])
    ```

=== "Rust"

    ```rust
    use dynars::validate::{Rule, Cmp, Expr, Value};

    let report = deck.validate([
        Rule::references_resolve(),                                    // ids resolve
        Rule::duplicate_ids(),                                         // no id collisions
        Rule::unreferenced_entities(),                                 // dead *MAT/*SET/... (warns)
        Rule::include_missing(),                                       // missing *INCLUDEs
        Rule::field_forbidden_values("MAT_ELASTIC", "PR", [Value::Float(0.5)]),
        Rule::field_required(
            "SECTION_SHELL",
            Some(Expr::field("ELFORM", Cmp::Eq, Value::Int(2))), // when
            Expr::field("NIP", Cmp::Gt, Value::Int(0)),          // require
        ),
    ]);
    ```

The built-ins:

| Rule | Checks |
|------|--------|
| `references_resolve` | every cross-keyword id reference resolves (`*PART.mid → *MAT`, `*LOAD.lcid → *DEFINE_CURVE`, …) |
| `references_resolve_with_connectivity` | as above, **and** every element's nodes exist (heavy on big meshes) |
| `duplicate_ids` | no two entities of a kind claim one id (logical ids, so `*INCLUDE_TRANSFORM` instances don't collide) |
| `unreferenced_entities` | library definitions nothing references — dead `*MAT`/`*SECTION`/`*DEFINE_CURVE`/`*SET`/… (warns) |
| `rigid_context` | rigid-body keywords (`*LOAD_RIGID_BODY`, `*CONSTRAINED_RIGID_BODIES`, …) target a `*MAT_RIGID` part |
| `include_missing` | every `*INCLUDE` resolves to a file on disk |
| `field_forbidden_values` / `field_required` / `keyword_forbidden` | field- and keyword-level policy |

## Severity and file scope

Every rule takes a severity override and a file scope. Scopes match on a path
substring (case-insensitive).

=== "Python"

    ```python
    Rule.keyword_forbidden("MAT_ADD_EROSION").with_severity(Severity.Warning)
    Rule.references_resolve().except_in(["submodel/"])   # everywhere but these
    Rule.duplicate_ids().only_in(["assembly/"])          # only within these
    ```

=== "Rust"

    ```rust
    use dynars::validate::Severity;

    Rule::keyword_forbidden("MAT_ADD_EROSION").with_severity(Severity::Warning);
    Rule::references_resolve().except_in(["submodel/"]); // everywhere but these
    Rule::duplicate_ids().only_in(["assembly/"]);        // only within these
    ```

Compose field predicates with `all` / `any` / `not` (`Predicate.all_ / any_ /
not_` in Python, `Expr::all / any / not` in Rust).

## Custom checks

When the built-ins don't cover a policy, drop to arbitrary logic. In Rust,
implement the `Check` trait and wrap it in `Rule::custom`; in Python, walk the
deck yourself and build findings.

=== "Python"

    ```python
    # Arbitrary logic in plain Python — iterate the same views the built-ins use.
    bad = []
    for m in deck.materials():
        ro = m.field("RO")
        if ro is not None and ro <= 0.0:
            bad.append((m.file, m.line, f"RO = {ro} must be positive"))
    for file, line, msg in bad:
        print(f"{file}:{line} — {msg}")
    ```

=== "Rust"

    ```rust
    use dynars::validate::{Check, Deck, Finding, Rule, Severity};

    struct DensityPositive;
    impl Check for DensityPositive {
        fn name(&self) -> String { "density_positive".into() }
        fn run(&self, deck: &Deck) -> Vec<Finding> {
            deck.keywords("MAT_ELASTIC").filter_map(|m| {
                let ro = m.field("RO")?.as_f64()?;
                (ro <= 0.0).then(|| Finding {
                    rule: self.name(), severity: Severity::Warning,
                    keyword: "MAT_ELASTIC".into(), file: m.file().to_path_buf(),
                    line: m.line(), message: format!("RO = {ro} must be positive"),
                })
            }).collect()
        }
    }

    let report = deck.validate([Rule::custom(DensityPositive)]);
    ```

## Validating many decks

Checking a batch of decks that share includes? Do it once against a shared cache
— see [Workspace (batch)](workspace.md).
