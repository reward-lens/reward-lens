# The reward-lens record standard

This standard keeps a measurement record from making a stronger claim than its
artifact supports. An assay describes what was measured about a reward system.
A run record describes what a training or evaluation run captured. A trace
reader may project a run into four named series. These are separate contracts.

The normative assay schema is
`schema/assay/1.0/assay.schema.json`. The normative run-record writer is
`reward_lens.record.RecordWriter`; it writes
`schema_version` from `reward_lens.record.schema.RECORD_SCHEMA_VERSION` into
both `run.json` and `manifest.json`. This directory explains those contracts. It
does not replace either source with a generated or copied schema.

A producer MUST write every required field of its applicable contract. A reader
MUST retain the distinction between a recorded value, a configured objective, an
advantage, and an optimizer-applied contribution. A reader MUST report an
absence when its source did not capture the required quantity.

Read [the assay record](assay.md), [the run record](run-record.md), and [the
validation procedure](validation.md) together. The supplied examples are
generated or opened by the local implementation and are checked by
`tests/standard`.
