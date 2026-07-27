/* dynars — C ABI for the deck parse + validate path.
 *
 * This header is the authoritative, hand-maintained declaration of the FFI
 * exported by `src/ffi.rs` (built with `cargo build --features ffi`). It can
 * also be regenerated with cbindgen (see examples/ffi/README.md).
 *
 * Ownership: every handle returned by a `*_new`/`parse`/`validate` call is owned
 * by the caller and must be released with the matching `*_free`. Strings handed
 * out by the finding accessors live inside their DynarsReport and are valid
 * until dynars_report_free; copy them if you need them longer.
 */
#ifndef DYNARS_H
#define DYNARS_H

#include <stddef.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct DynarsDeck    DynarsDeck;
typedef struct DynarsRuleSet DynarsRuleSet;
typedef struct DynarsReport  DynarsReport;

/* Finding severity (ABI-stable integer values). */
typedef enum DynarsSeverity {
    DYNARS_ERROR   = 0,
    DYNARS_WARNING = 1,
    DYNARS_INFO    = 2
} DynarsSeverity;

/* Message for the most recent failing dynars_* call on THIS thread, or NULL.
 * Owned by the library; valid only until the next dynars_* call on this thread. */
const char *dynars_last_error(void);

/* ── Deck: parse + basic queries ──────────────────────────────────────── */

/* Parse a deck (root file + all *INCLUDEs). NULL on failure (see
 * dynars_last_error). Free with dynars_deck_free. */
DynarsDeck *dynars_parse_deck(const char *path);
void        dynars_deck_free(DynarsDeck *deck);
size_t      dynars_deck_file_count(const DynarsDeck *deck);
size_t      dynars_deck_total_bytes(const DynarsDeck *deck);

/* ── Rule set: build the checks to run ─────────────────────────────────── */

DynarsRuleSet *dynars_ruleset_new(void);
void           dynars_ruleset_free(DynarsRuleSet *rules);

/* Cross-keyword referential integrity (every id reference resolves). */
void dynars_ruleset_add_references_resolve(DynarsRuleSet *rules);
/* As above, and also checks every element's nodes are defined (heavy). */
void dynars_ruleset_add_references_resolve_with_connectivity(DynarsRuleSet *rules);
/* Flag every *INCLUDE whose target file is missing on disk. */
void dynars_ruleset_add_include_missing(DynarsRuleSet *rules);
/* Flag any occurrence of `keyword`. Returns 0 on success, -1 on error. */
int  dynars_ruleset_add_keyword_forbidden(DynarsRuleSet *rules, const char *keyword);

/* ── Report: run the rules, read findings back ─────────────────────────── */

/* NULL if deck or rules is NULL (see dynars_last_error). Free with
 * dynars_report_free. */
DynarsReport *dynars_deck_validate(const DynarsDeck *deck, const DynarsRuleSet *rules);
void          dynars_report_free(DynarsReport *report);

size_t dynars_report_len(const DynarsReport *report);
size_t dynars_report_count(const DynarsReport *report, DynarsSeverity severity);
int    dynars_report_is_clean(const DynarsReport *report); /* 1 = no ERROR findings */

/* Per-finding accessors. Index must be < dynars_report_len; out-of-range
 * returns 0 / NULL / DYNARS_ERROR. Strings are valid until dynars_report_free. */
DynarsSeverity dynars_report_finding_severity(const DynarsReport *report, size_t i);
size_t         dynars_report_finding_line(const DynarsReport *report, size_t i);
const char    *dynars_report_finding_rule(const DynarsReport *report, size_t i);
const char    *dynars_report_finding_keyword(const DynarsReport *report, size_t i);
const char    *dynars_report_finding_file(const DynarsReport *report, size_t i);
const char    *dynars_report_finding_message(const DynarsReport *report, size_t i);

#ifdef __cplusplus
}
#endif

#endif /* DYNARS_H */
