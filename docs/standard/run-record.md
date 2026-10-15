# Run record

A run record is a directory produced by `reward_lens.record.RecordWriter`. Its
root contains `run.json` and `manifest.json`, and its partitions contain the
`steps`, `groups`, `trajectories`, and `turns` tables. The writer chooses JSONL
or Parquet but preserves this table layout. Tensor sidecars are referenced from
the row that uses them.

## Required fields and meanings

`schema_version` occurs in both `run.json` and `manifest.json` and identifies
the layout written by this version of the record writer. `id`, `kind`,
`components`, `access`, `regime`, `arms`, `lineage`, and `sampling_policy` in
`run.json` describe the run before any step is materialised. `manifest.json`
declares the format, table list, chunk size, and written step indices.

`steps` records the step index, schedule, optimizer telemetry, probes,
instrument effect, and measured regime. `groups` records the declared group
membership and estimator. `trajectories` records task identity, turns, score
tree, trajectory advantage, token advantage when captured, provenance,
compaction, labels, features, and capture reference. `turns` records the model
action, environment response, token and loss-mask information when captured,
and any step score or step advantage.

`components` names declared run components. A component is not automatically a
recorded reward or an optimizer objective. A score tree preserves its named
leaves and its composition. A weight belongs to that composition, where one was
captured, rather than being folded into a component value.

## The trace projection

The accepted trace reader projects a source into four fixed series:

| Field | Meaning |
| --- | --- |
| `reward_recorded` | A recorded reward-side value whose role is declared by the source. |
| `reward_applied` | The value or weight documented as entering the applied loss path. |
| `advantage` | The estimator output used for relative preference within a group. |
| `selection` | Evidence that a rollout was selected into the update path. |

An absent series stays absent with its source-specific reason. `reward_applied`
is not recoverable from a logged scalar alone, and `selection` is not established
by a completion state alone.

### A prohibited column

A standard run-record table MUST NOT name a standard field `reward`. A source
column named `reward` remains outside the standard field set. The name erases
whether a number is recorded output, a configured objective, an
advantage, or an optimizer-applied contribution. Those quantities can differ
for the same rollout. A source reader may preserve its raw `reward` input as a
labelled convenience value while stating that its optimization role is unknown;
it MUST NOT promote that raw value into `reward_recorded` or `reward_applied`
without source evidence.

## Versioning and refusal

Every writer MUST include `schema_version` in both required root files. A
missing version is an invalid instance for this standard and MUST be refused by
a validating importer rather than guessed from filenames or table columns. A
change that adds optional fields without changing the meaning of existing fields
is additive. A change that reinterprets, removes, or changes the type of an
existing field requires a new major and a reader that does not support that
major MUST report an incompatible major without rewriting the source record.

An unknown field is not evidence that a reader understands it. A reader may
preserve an explicitly declared extension where its contract allows one; it
MUST otherwise refuse or leave the record uninterpreted. An unverifiable example
is not a validated instance: missing provenance, an artifact digest mismatch,
or an undeclared row identifier prevents interpretation.

## Per-component `verifiers` instance

`tests/fixtures/readers/verifiers/` is a checked-in result directory produced by
the framework writer. Its `metadata.json` declares the rows, `results.jsonl`
contains saved rollout outputs and named `metrics` components, and
`PROVENANCE.json` binds the results bytes to the writer record. This instance is
validated in two parts. The source artifact is opened through the public
`verifiers_results` reader, and its per-component `metrics` mapping is supplied
to the actual `score_tree` converter with an explicit component-weight mapping.

The fixture's raw `reward` value is deliberately not a standard trace field.
The `verifiers_results` reader reports it only as `source_reward` and reports
the four standard series as absent when the saved files cannot establish their
roles. The fixture therefore validates the provenance and per-component route
without claiming a missing loss path.

`verl` and `prime-rl` reader routes are unsupported in this build. No dummy
converter or invented instance stands in for either framework.
