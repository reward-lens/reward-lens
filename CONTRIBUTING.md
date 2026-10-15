# Contributing to reward-lens

Develop the product in the official repository,
[reward-lens/reward-lens](https://github.com/reward-lens/reward-lens). Use a new
branch from current `origin/main` and open a pull request for review.

## Development setup

Python 3.11 or newer is required. CI covers Python 3.11, 3.12, and 3.13 on Linux.
The base wheel requires neither Node nor a model runtime. Some optional training
and scientific integrations have narrower platform and dependency constraints.

```bash
git clone https://github.com/reward-lens/reward-lens.git
cd reward-lens
uv venv
uv pip install --python .venv/bin/python --torch-backend cpu . --group dev
```

Development dependencies use the PEP 735 `dev` group. There is no `dev` package
extra. Install only the optional extras needed for the feature under test;
`pyproject.toml` documents them. The command above uses the official PyTorch CPU wheels. `uv lock --check`
checks metadata/lock consistency separately. `uv sync --group dev` replays the
full platform lock, including its accelerator dependencies on Linux.

The declared release matrix is `release/validation-matrix.json`. Run test
directories in separate pytest invocations where their local `conftest` modules
use direct imports. A missing fixture or collection error is a failure. Build the public inputs described in
`release/public_inputs/README.md` and set their three explicit environment
variables before running a profile. The `trl_110` and `trl_113` profiles need
separate environments with their exact declared TRL versions. Do not run the
entire historical test tree under an arbitrary trainer version.

```bash
.venv/bin/ruff check src tests tools scripts hatch_build.py
.venv/bin/ruff format --check src tests tools scripts hatch_build.py
.venv/bin/python scripts/run_release_matrix.py dev
```

Deliberately malformed negative-control inputs and unchanged third-party code
are excluded from Python formatting. Their execution, provenance, and integrity
checks remain required.

## Frontends and packages

Node 20.19 or newer is a development dependency for the report and workbench.

```bash
cd frontend
npm ci
npm run typecheck
npm run build:report
npm run build:workbench
cd ..
python -m build
```

Commit the generated report and workbench assets with their reviewed source.
The wheel build can use those assets when Node is absent. Validate both wheel
and sdist in clean environments, including a wheel rebuilt from the sdist.
`TEST_STATUS.md` records the current evidence and optional boundaries.

## Pull requests and releases

Explain the behavior changed, its regression test, compatibility implications,
and remaining limits. Keep historical records and scientific claims accurate.
Fix failing required checks before merge. Release publication follows
[`docs/releasing.md`](docs/releasing.md) and uses the exact artifacts validated by
main CI. Account permissions and production approvals remain in force.
