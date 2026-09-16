# reward-lens identity and versioning

`reward-lens` is the canonical project and product name. The latest legitimate public 3.x release is `3.0.0` (`v3.0.0`, 2026-08-08). The product checkout had incorrectly declared the unreleased work as `4.0.0`; its canonical target is the unreleased `3.1.0`, the next minor release after `3.0.0`. Until release, install this checkout with `pip install .` or install the locally built wheel, not `pip install reward-lens` from PyPI.

This is metadata normalization, not a release or history rewrite. The active product tree is on the legacy technical branch `product/4.0`, which is deliberately retained while its packet worktrees still resolve it. It is a branch name, never evidence of a fourth major release. The tree is not descended from the public `v3.0.0` release, so a reviewed reconciliation decision is required before a `v3.1.0` tag is made.

## Version sources

`pyproject.toml` is the Python distribution authority. `src/reward_lens/__init__.py` is the runtime fallback. `uv.lock`, the frontend workspace manifests, and `frontend/package-lock.json` mirror the same `3.1.0` release identity. `tests/packaging/test_release_identity.py` makes divergence among those sources a test failure; `tests/packaging/test_wheel_metadata.py` reads the version back from the built wheel.

## Deliberate residual forms

| Form | Why it remains |
| --- | --- |
| `reward_lens` | Python import namespace, package-data path, and wheel stem. A hyphen is invalid in those identifiers. |
| `RewardLensError` | Public Python exception base class. Renaming it would break downstream exception handling. |
| `RewardLens` in migration/legacy API material and old examples/notebook cells | A historical public API identifier, retained only to explain or demonstrate the retired v1 surface. It is not current product branding. |
| `rewardlens.yaml`, `rewardlens.schema.json`, and related schema IDs/keys | Existing user configuration contract and compatibility filename. Renaming it would break projects and schema resolution. |
| `product/4.0` | Legacy technical Git branch and worktree reference. It remains resolvable until the packet fleet is retired or deliberately migrated. |
| `4.0.0` in `fleet/TEST_EVIDENCE/**`, `chain/repair/proofs/**`, and dated handoff/review receipts | Literal record of commands, wheels, and environment state actually observed before this normalization. These receipts must not be rewritten as if they had run under 3.1.0. |
| `@v4`, `DeepSeek-V4`, `transformers v4.45`, dependency constraints, schema revisions, and study/wave labels | Third-party products, dependency versions, schemas, and unrelated identifiers. They do not denote the reward-lens release line. |

Active code, package metadata, CLI-facing error text, README route, frontend metadata, current fixtures, goldens, and fleet controller text use `reward-lens` and `3.1.0`. Post-migration audits should classify every remaining legacy match using this table before changing it.
