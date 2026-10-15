# Release validation

The release matrix is defined in `release/validation-matrix.json` and executed
by `.github/workflows/ci.yml`. Publication requires a fully successful run for
the exact main commit, including its immutable validated wheel and sdist. A
passing subset or a result from another source tree does not meet that gate.

The four positive profiles cover distinct environments:

- `dev`: the declared development group on CPU, Python 3.11, 3.12, and 3.13.
  It covers contracts, SDK and CLI, audit/reuse, trace, forecast/improve, stores,
  export/import, report/workbench, packaging, and portable fixtures.
- `all_extras`: the declared optional extras and development group on the same
  Python versions, adding offline integration and library regression checks.
- `trl_110`: controlled-intervention learning and compare tests on Python 3.11
  with TRL 1.10.0, which their execution and restoration contracts require.
- `trl_113`: the matched bare/tapped CPU trainer conformance test on Python 3.11
  with TRL 1.13.0. It uses random weights and a local tokenizer.

All four profiles are required. Tests are grouped into separate pytest
processes to keep local `conftest` imports and duplicate historical module
basenames from colliding. `REWARD_LENS_EXIT_CHECK=1` enables the unconditional
shipped-asset assertions; `REWARD_LENS_SLOW=1` enables whole-record schema
parity. The earlier API/CLI/error deselections are retired.

The E1 and bank checks require inputs rebuilt by
[`release/public_inputs`](release/public_inputs/README.md). The builder verifies
the pinned public parquet and 401 Inspect archives, checks a one-to-one join
and scorer agreement, and writes a new derived table with its own digest. It
also recreates three campaign banks with their original byte/content hashes.
CI supplies only these derived files to test jobs after checking their receipts,
source identity, and hashes. Missing or inconsistent required inputs fail.
Original checkouts, private caches, model downloads, and paid providers are not
inputs to this release gate.

The matrix names 34 retained historical tests outside these profiles. They
import separate, unshipped top-level research projects or need a historical v1
cache. The inventory test requires every test file to be covered by a positive
profile or explicitly classified. Other historical scientific assertions keep
their existing optional archive guards, using explicit
`REWARD_LENS_CAMPAIGN_STORE`, `REWARD_LENS_CAMPAIGN_SPECS`, or
`REWARD_LENS_VERL_SOURCE` inputs. Their skips do not establish scientific
results. Always-on offline product controls remain required.

Separate required jobs check source/metadata/lock consistency, the audited
secret baseline, both frontend builds and TypeScript, strict documentation and
local links, distribution contents, a wheel rebuilt from its sdist without
Node, and a clean base installation outside source trees. Dependency checks
run real npm and OSV scans. The frontend advisory gate records the exact
reviewed build-only braces dependency path and configuration that makes its
pattern matcher unreachable; a changed path/configuration or new advisory
fails. The source-map-js advisory is patched in the lockfile.

The generated code-reward example deliberately has defects. A complete offline
audit writes a saved assay and embedded report, keeps honest absences, and exits
2 with an unresolved decision. Reuse must preserve the saved bytes; signed
export/import must verify them and reject tampering. This is product
conformance, not an all-clear reward-system finding.

Exact source, dependency versions, CI runs, artifact hashes, and public
installation/deployment evidence are recorded with the release. Until those
checks and production publication complete, 3.1.0 remains an unreleased source
candidate. Offline checks do not qualify live providers, a GPU intervention,
forecasting skill, or a new scientific training result.
