/* Parse an LS-DYNA deck and validate it, from C.
 *
 * Build (from examples/ffi/):  make c_example
 * Run:                         ./c_example path/to/main.k
 */
#include <stdio.h>
#include "dynars.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to-deck.k>\n", argv[0]);
        return 2;
    }

    DynarsDeck *deck = dynars_parse_deck(argv[1]);
    if (!deck) {
        fprintf(stderr, "parse failed: %s\n", dynars_last_error());
        return 1;
    }
    printf("parsed %zu file(s), %zu bytes\n",
           dynars_deck_file_count(deck), dynars_deck_total_bytes(deck));

    /* Assemble the checks we want to run. */
    DynarsRuleSet *rules = dynars_ruleset_new();
    dynars_ruleset_add_references_resolve(rules);
    dynars_ruleset_add_include_missing(rules);

    DynarsReport *report = dynars_deck_validate(deck, rules);
    size_t n = dynars_report_len(report);
    printf("%zu finding(s): %zu error, %zu warning, %zu info\n", n,
           dynars_report_count(report, DYNARS_ERROR),
           dynars_report_count(report, DYNARS_WARNING),
           dynars_report_count(report, DYNARS_INFO));

    static const char *SEV[] = {"ERROR", "WARNING", "INFO"};
    for (size_t i = 0; i < n; i++) {
        printf("  [%s] %s:%zu  %s\n",
               SEV[dynars_report_finding_severity(report, i)],
               dynars_report_finding_file(report, i),
               dynars_report_finding_line(report, i),
               dynars_report_finding_message(report, i));
    }

    int clean = dynars_report_is_clean(report);
    printf("%s\n", clean ? "deck is clean (no errors)" : "deck has errors");

    dynars_report_free(report);
    dynars_ruleset_free(rules);
    dynars_deck_free(deck);
    return clean ? 0 : 1;
}
