# Repository guidance

Use `reward-lens` in prose and commands, `reward_lens` for Python imports, and
`REWARD_LENS_*` for environment variables. The published 3.1.0 tag points to
`2775965d641ca48cf2745bf9c41e6b5659562306`; continue development on the 3.x line and check current tags and
PyPI metadata before changing version declarations. Preserve public symbols,
serialized keys, `rewardlens.yaml`, and existing record identities.

This standalone clone is the maintained product development repository. Base new
branches on current `origin/main`, use focused pull requests, and keep source,
tests, resources, and documentation together. Follow `CONTRIBUTING.md` and the
machine-readable matrix in `release/validation-matrix.json`. Run builds and
tests in independent disposable clones with isolated environments.

Keep private operational records, credentials, raw research data, environments,
and release workspaces outside source history. Historical development snapshots
and the pinned reproducibility subproject retain their own provenance and remain
outside the product distribution. The narrow `fleet/golden` and
`fleet/gates/fixtures` directories are portable test inputs, not a controller.

Normal pushes and pull requests must only validate. Production publication uses
the reviewed release workflow, exact main commit, existing immutable tag,
successful main CI, and that run's validated wheel and sdist. Never use
`skip-existing`, move a published tag, or replace an existing release asset.
See `docs/releasing.md` for the release sequence.
