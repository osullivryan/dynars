"""Batch-validate several decks that share `*INCLUDE`s off one shared cache.

A `Workspace` reads/parses/indexes each common file (mesh, materials, sections)
exactly once, no matter how many decks include it, then validates the decks in
parallel. The decks it returns are ordinary `Deck`s — navigate or validate them
individually too.

    python examples/batch_demo.py <a/main.k> [<b/main.k> ...]
"""

import sys

import dynars

roots = sys.argv[1:]
if not roots:
    print("usage: python examples/batch_demo.py <main.k> [<main2.k> ...]")
    raise SystemExit(2)

ws = dynars.Workspace()
decks = ws.parse_decks(roots)

reports = ws.validate_decks(
    decks,
    [
        dynars.Rule.references_resolve_with_connectivity(),
        dynars.Rule.duplicate_ids(),
        dynars.Rule.include_missing().with_severity(dynars.Severity.Warning),
    ],
)

for root, report in zip(roots, reports):
    errors = report.count(dynars.Severity.Error)
    warnings = report.count(dynars.Severity.Warning)
    print(f"{root}: {errors} error(s), {warnings} warning(s)")
    for f in report.findings[:10]:
        print(f"  [{f.severity}] {f.location()} — {f.message}")

# What the sharing bought: files read once, indices built once (a shared mesh
# counts once, not once per deck).
print("\ncache:", ws.stats())
