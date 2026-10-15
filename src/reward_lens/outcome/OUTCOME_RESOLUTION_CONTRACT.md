# Outcome resolution contract

This package exposes a separate resolution layer after, and only after, an outcome protocol has
qualified. Qualification asks whether the protocol is fit to support a correctness claim.
Resolution records what that qualified protocol returned for each item. A protocol is never made
qualified by a collection of per-item labels.

## Semantic states

`OutcomeState` has exactly three values: `PASS`, `FAIL`, and `UNQUALIFIED`. `Outcome` carries a
resolved or unqualified item state. `PASS` and `FAIL` are resolved outcomes. `UNQUALIFIED` means the item has no outcome measurement under this protocol; it
is never treated as a failure, false value, zero, or denominator member.

Every `Outcome` requires a non-empty item id, reason, protocol digest, provenance,
task-validity record, and scope. The protocol digest identifies the protocol that qualified, not
the evaluated candidate or the individual task. Provenance names the actual evaluator attempt.
Task validity and scope make the item-level claim bounded rather than a statement about a broader
task population.

## Construction rules

`resolved_pass(...)` and `resolved_fail(...)` accept only an explicit `TaskValidity.VALID` value.
They reject booleans and all truthiness coercion. `unqualified(...)` accepts only an
`UnqualifiedReason`; it too rejects strings, booleans, negation, and default-zero substitutes.

`AMBIGUITY`, `MISSING_CHECK`, `TIMEOUT`, `EVALUATOR_ERROR`, `PANEL_TIE`, and `REFUSAL` are all
unqualified reasons. No constructor, aggregation function, or panel helper converts any of them
to `FAIL`.

## Aggregation

`OutcomeSummary.from_outcomes(...)` keeps the total, pass, fail, and unqualified counts. Its
resolved denominator contains only `PASS` and `FAIL`. Its resolved rate is `None` and its
measurement state is `NOT_MEASURED` when that denominator is zero. Unqualified reasons remain
counted so an empty denominator cannot be mistaken for a zero success rate or a complete test.

## Relation to existing P-OUTCOME interfaces

`qualify(...)` retains its existing role: it qualifies the protocol and emits the assay-level
entry. The new resolution types consume a qualified protocol digest and do not alter that function,
the round ledger, partition access policy, candidate-process isolation, or sealed-round anti-forgery
checks. Consumers must first verify qualification, then construct or consume item resolutions.
