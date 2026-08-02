# Validation

Rules run over a parsed `Deck`, reusing the parse and fanning out across cores.
There is **no default rule set** — you pass exactly the checks you want and get
back a `Report` of findings, each with a clickable `file:line`. What counts as an
error is your policy, so you assemble it.

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

## Reading the report

`validate` returns a `Report`. Three things to know:

- `is_clean()` is `True` when there are **no `Error`-severity findings** — warnings
  don't sink it.
- `count(severity)` totals findings at one level; `len(report)` (Python) is the
  grand total.
- `findings` is the list; each `Finding` has `.severity`, `.rule`, `.keyword`,
  `.message`, and a clickable `.location()`.

=== "Python"

    ```python
    report = deck.validate([dynars.Rule.references_resolve(), dynars.Rule.duplicate_ids()])

    print(report.is_clean(),
          report.count(dynars.Severity.Error),
          report.count(dynars.Severity.Warning))

    # Group findings by rule for a tidy summary:
    from collections import Counter
    by_rule = Counter(f.rule for f in report.findings)
    for rule, n in by_rule.most_common():
        print(f"{n:>5}  {rule}")

    for f in report.findings:
        print(f"{f.severity}  {f.location()}  {f.message}")
    ```

=== "Rust"

    ```rust
    use dynars::validate::Severity;

    let report = deck.validate([Rule::references_resolve(), Rule::duplicate_ids()]);
    println!("{} error(s), {} warning(s)",
             report.count(Severity::Error), report.count(Severity::Warning));

    for f in &report.findings {
        println!("{:?}  {}  {}", f.severity, f.location(), f.message);
    }
    ```

## Severity and file scope

Every rule takes a severity override and a file scope. Scopes match on a path
substring (case-insensitive), so `"submodel/"` matches any include under a
`submodel` directory.

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

## Composing field predicates

`field_required` takes a `when` (optional guard) and a `require` predicate.
Compose them with `all` / `any` / `not` to express real policies — "solid
sections must use a valid element formulation **and** a positive thickness".

=== "Python"

    ```python
    from dynars import Rule, Predicate, Cmp

    # If it's a shell section (ELFORM in {2, 16}), require NIP >= 3 and a positive T1.
    Rule.field_required(
        "SECTION_SHELL",
        when=Predicate.any_([
            Predicate.field("ELFORM", Cmp.Eq, 2),
            Predicate.field("ELFORM", Cmp.Eq, 16),
        ]),
        require=Predicate.all_([
            Predicate.field("NIP", Cmp.Ge, 3),
            Predicate.field("T1", Cmp.Gt, 0.0),
        ]),
    )
    ```

=== "Rust"

    ```rust
    use dynars::validate::{Rule, Expr, Cmp, Value, pred};

    Rule::field_required(
        "SECTION_SHELL",
        Some(Expr::any([
            pred("ELFORM", Cmp::Eq, Value::Int(2)),
            pred("ELFORM", Cmp::Eq, Value::Int(16)),
        ])),
        Expr::all([
            pred("NIP", Cmp::Ge, Value::Int(3)),
            pred("T1", Cmp::Gt, Value::Float(0.0)),
        ]),
    );
    ```

## A house rule set

In practice you keep a standing list of rules — your team's "house rules" — and
run it on every model. Assemble it once and reuse it.

=== "Python"

    ```python
    import dynars
    from dynars import Rule, Predicate, Cmp, Severity

    HOUSE_RULES = [
        Rule.references_resolve(),
        Rule.duplicate_ids(),
        Rule.include_missing(),
        Rule.rigid_context(),
        Rule.unreferenced_entities().with_severity(Severity.Warning),
        Rule.keyword_forbidden("MAT_ADD_EROSION"),          # not allowed in production decks
        Rule.field_forbidden_values("CONTROL_TIMESTEP", "DT2MS", [0.0]),
        Rule.field_required(
            "SECTION_SHELL",
            when=Predicate.field("ELFORM", Cmp.Eq, 2),
            require=Predicate.field("NIP", Cmp.Gt, 0),
        ),
    ]

    report = dynars.parse_deck("main.k").validate(HOUSE_RULES)
    print(report.is_clean(), report.count(Severity.Error))
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;
    use dynars::validate::{Rule, Expr, Cmp, Value, Severity};

    fn house_rules() -> Vec<Rule> {
        vec![
            Rule::references_resolve(),
            Rule::duplicate_ids(),
            Rule::include_missing(),
            Rule::rigid_context(),
            Rule::unreferenced_entities().with_severity(Severity::Warning),
            Rule::keyword_forbidden("MAT_ADD_EROSION"),
            Rule::field_forbidden_values("CONTROL_TIMESTEP", "DT2MS", [Value::Float(0.0)]),
            Rule::field_required(
                "SECTION_SHELL",
                Some(Expr::field("ELFORM", Cmp::Eq, Value::Int(2))),
                Expr::field("NIP", Cmp::Gt, Value::Int(0)),
            ),
        ]
    }

    let report = parse_deck(std::path::Path::new("main.k")).unwrap().validate(house_rules());
    println!("{}", report.count(Severity::Error));
    ```

## Gate a pipeline on the result

Because `is_clean()` is a plain boolean and findings carry `file:line`, wiring
validation into CI or a pre-submit hook is a few lines: print the findings and
exit non-zero if there are errors.

=== "Python"

    ```python
    import sys, dynars

    report = dynars.parse_deck(sys.argv[1]).validate(HOUSE_RULES)
    for f in report.findings:
        print(f"{f.location()}: {f.severity}: {f.message}")   # editor-clickable
    sys.exit(0 if report.is_clean() else 1)
    ```

=== "Rust"

    ```rust
    use dynars::deck::parse_deck;

    let root = std::env::args().nth(1).expect("usage: check <root.k>");
    let report = parse_deck(std::path::Path::new(&root)).unwrap().validate(house_rules());
    for f in &report.findings {
        println!("{}: {:?}: {}", f.location(), f.severity, f.message);
    }
    std::process::exit(if report.is_clean() { 0 } else { 1 });
    ```

## Custom checks

When the built-ins don't cover a policy, drop to arbitrary logic. In Rust,
implement the `Check` trait and wrap it in `Rule::custom`, so it composes with the
built-ins in one `validate` call and fans out with them. In Python, walk the same
views the built-ins use and build your own findings.

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

    // Composes with the built-ins in one call:
    let report = deck.validate([Rule::references_resolve(), Rule::custom(DensityPositive)]);
    ```

## Validating many decks

Checking a batch of decks that share includes? Do it once against a shared cache
— see [Workspace (batch)](workspace.md). The same rules apply; only the driver
changes.
