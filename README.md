# reward-lens

`reward-lens` turns a reward system into an auditable assay record and an offline report. It is for work where the reward written in code may differ from the objective a policy learned to optimize.

## Versions and installation

The stable package on PyPI is **3.0.0**:

```bash
python -m pip install reward-lens==3.0.0
```

This repository's `main` branch is the unreleased **3.1.0** development target. Install that source revision only when you need work that has not been released to PyPI:

```bash
python -m pip install "reward-lens @ git+https://github.com/reward-lens/reward-lens.git@main"
```

Python 3.11 or newer is required. The base package does not require a model runtime. Optional extras in `pyproject.toml` cover integrations that need them.

## First audit

```bash
reward-lens init --example code-reward ./demo
reward-lens audit ./demo
reward-lens doctor
```

`init` writes the example project. `audit` writes a record and report in that project. `doctor` reports the measurements available from the supplied inputs and what further access would be required. The bundled example includes deliberately wrong tasks. Its audit produces a report with blocking findings and exits 2; this smoke check is not an all-clear result. Command help is available for the full command tree. A development command whose implementation is not present returns a typed refusal instead of simulating a result.

## What is here

- `src/reward_lens/`: the installable library, CLI, schemas, report renderer, adapters, and optional integrations.
- `schema/` and `spec/`: public schema and catalogue sources used by the package.
- `frontend/`: source for the offline report bundle. The built JavaScript and stylesheet are committed under the Python package so normal package builds do not require Node.
- `tests/`: reusable contract, API, CLI, packaging, and compatibility tests. [`TEST_STATUS.md`](TEST_STATUS.md) states which broader suites remain incomplete or need optional dependencies.
- `docs/`: API, schema, command, and method documentation.

The accompanying [GRPO Reward Trace](https://github.com/reward-lens/grpo-reward-trace) repository contains protocol and fixture code for studying how GRPO rewards become update pressure and learned behavior. Both source imports became public on 24 September 2026 (Asia/Kolkata). The independently installed public study fixture produced 2 pass, 1 fail, 1 defect, and 32 unrun rows; those are routing outcomes, not a completed GPU experiment.

## Compatibility and history

The 3.1.0 source keeps the supported `reward_lens` import name, the `reward-lens` and `rlens` command names, and established serialized keys. [`docs/compatibility.md`](docs/compatibility.md) records the six retained v3 behavior fixes and their public tests.

This repository was assembled from reviewed source snapshots. [`docs/publication-history.md`](docs/publication-history.md) explains the reconstructed import history, source revisions, and its limits. Commit dates in that import history are not evidence of public preregistration, experiment execution, or package release dates.

## Limits

The repository contains reusable code, small fixtures, and documentation. It excludes private fleet records, raw run outputs, local provider configuration, credentials, and large captured research material. A clean fixture or package check does not establish that an unfinished empirical study ran or that a reward mechanism was validated on a new model.

For contributor setup and the supported checks, read [CONTRIBUTING.md](CONTRIBUTING.md) and [TEST_STATUS.md](TEST_STATUS.md).

[Development snapshots](development-snapshots/README.md) preserve selected unfinished alternatives as full files. They are outside installed packages and default tests. The [historical C1 dependency](reproducibility/c1-reward-lens-snapshot/README.md) is an isolated installable subproject, excluded from the current product distributions.
