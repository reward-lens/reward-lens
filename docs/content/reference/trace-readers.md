# Trace readers

`reward-lens trace` discovers readers from the public trace package and records the four separate
series `reward_recorded`, `reward_applied`, `advantage`, and `selection`. A reader names a missing
series instead of creating a field named `reward` or guessing a loss path from an evaluator output.

## verl

veRL writes rollout data to `{rollout_data_dir}/{step}.jsonl`. Its rows carry `input`, `output`,
`gts`, and `score`; the relevant scalar is named `score`, not `reward`. This release does not ship
a veRL reader because no real veRL artifact was available to read and bind to provenance. Importing
the optional framework or transcribing its source layout is not evidence that this reader works.

## prime-rl

prime-rl writes verifiers-shaped rows at
`rollouts/step_N/{train,eval}/{all,effective}/traces.jsonl`. This release does not ship a prime-rl
reader because no real artifact at that path was available to read and bind to provenance. The path
difference alone does not establish that every field retained the verifiers semantics.
