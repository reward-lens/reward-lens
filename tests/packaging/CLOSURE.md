# The measured base closure

Written by `tests/packaging/test_closure.py`. This is what `uv pip list` prints in a venv
holding only the built wheel, which is the count D-58 says binds.

12 distributions.

| distribution | version |
|---|---|
| annotated-types | 0.8.0 |
| click | 8.5.0 |
| coverage | 7.16.1 |
| jsonschema-rs | 0.56.0 |
| libcst | 1.9.0 |
| pydantic | 2.13.5 |
| pydantic-core | 2.46.5 |
| pyyaml | 6.0.3 |
| reward-lens | 3.1.0 |
| rfc8785 | 0.1.4 |
| typing-extensions | 4.16.0 |
| typing-inspection | 0.4.4 |

## Addendum A-002: what `coverage` costs the base closure

`verifier/coverage.py` imports the `coverage` distribution inside `trace_corpus`, and D-63
says the first hour runs with no extras, so `measure_coverage` has to reach a Reading on
the base closure. Three measurements taken on this platform (Linux x86_64, CPython 3.11.13)
before A-002 was applied:

- **Wheel size.** `coverage-7.16.0-cp311-cp311-manylinux1_x86_64.manylinux_2_28_x86_64.
  manylinux_2_5_x86_64.whl` is 255,846 bytes (249.8 KiB), read from the PyPI JSON API for
  release 7.16.0 and selected by the cp311 manylinux x86_64 tag this interpreter resolves.
- **Distributions added beyond itself: none.** `uv venv` then `uv pip install
  'coverage>=7.15.0'` into an otherwise empty 3.11 venv, then `uv pip list`, prints one
  row: `coverage 7.16.0`.
- **Cold `import coverage`.** 152, 162 and 185 ms cumulative over three runs of
  `python -X importtime -c 'import coverage'` with every `__pycache__` removed between
  runs; 70 ms warm. The number is the whole subtree, stdlib imports included, which is
  what a caller pays.

The alternative A-002 rejected was a second tracer on `sys.settrace` (3.11) and
`sys.monitoring` (3.12+), which would replace a tracer whose semantics the inherited tests
pin, to save one pure-payload distribution. `requires-python >= 3.11` is unchanged.

## Cold `uvx reward-lens --help`

pending P-CLI. The console script's entry point is in the wheel's metadata and is asserted
by `test_wheel_metadata.py`, but `reward_lens.cli.main:main` does not exist yet, so there
is nothing to time. D-30's ceiling is 100 ms and gate 5 checks the import set as well as
the clock.
