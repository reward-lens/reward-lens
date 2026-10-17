# reward-lens identity and versioning

`reward-lens` is the canonical project and product name. Version 3.1.0 is the
published minor release after `v3.0.0`, whose annotated tag is dated
2026-08-08. The `v3.1.0` tag points to `2775965d641ca48cf2745bf9c41e6b5659562306`. Its files are on
[PyPI](https://pypi.org/project/reward-lens/3.1.0/)
and the [GitHub release](https://github.com/reward-lens/reward-lens/releases/tag/v3.1.0). Install the exact version with
`python -m pip install reward-lens==3.1.0`.

The former product checkout had declared unfinished work as `4.0.0` and retained
the technical branch `product/4.0` for its packet worktrees. Neither
represented a fourth major release. That checkout was not descended from public
`v3.0.0`, so the 3.1.0 source was compared and reconciled in this independent
development repository before tagging. Older public tags and research receipts
were not rewritten.

## Version sources

`pyproject.toml` is the Python distribution authority. `src/reward_lens/__init__.py` is the runtime fallback. `uv.lock`, the frontend workspace manifests, and `frontend/package-lock.json` mirror the same `3.1.0` release identity. `tests/packaging/test_release_identity.py` makes divergence among those sources a test failure; `tests/packaging/test_wheel_metadata.py` reads the version back from the built wheel.

## Deliberate residual forms

| Form | Why it remains |
| --- | --- |
| `reward_lens` | Python import namespace, package-data path, and wheel stem. A hyphen is invalid in those identifiers. |
| `RewardLensError` | Public Python exception base class. Renaming it would break downstream exception handling. |
| `RewardLens` in migration/legacy API material and old examples/notebook cells | A historical public API identifier, retained only to explain or demonstrate the retired v1 surface. It is not current product branding. |
| `rewardlens.yaml`, `rewardlens.schema.json`, and related schema IDs/keys | Existing user configuration contract and compatibility filename. Renaming it would break projects and schema resolution. |
| `product/4.0` | Legacy technical branch in the former product checkout, retained there for packet worktrees. It did not define a distribution release. |
| `4.0.0` in `fleet/TEST_EVIDENCE/**`, `chain/repair/proofs/**`, and dated handoff/review receipts | Literal record of commands, wheels, and environment state actually observed before this normalization. These receipts must not be rewritten as if they had run under 3.1.0. |
| `@v4`, `DeepSeek-V4`, `transformers v4.45`, dependency constraints, schema revisions, and study/wave labels | Third-party products, dependency versions, schemas, and unrelated identifiers. They do not denote the reward-lens release line. |

Active code, package metadata, CLI-facing error text, README route, frontend
metadata, current fixtures, and goldens use `reward-lens` and `3.1.0`. Classify
historical legacy matches with this table before changing them.
