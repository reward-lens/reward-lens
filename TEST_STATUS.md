# Test status

The public CPU workflow is deliberately a supported regression contract, not a claim that every retained development suite is green. It builds a wheel and sdist, checks that both omit unreleased reproducibility and development snapshot paths, then runs public contract, API, CLI, error, and v3 compatibility tests on Python 3.11 and 3.12. It does not deploy Pages, publish a package, create a release, provision cloud resources, or run GPU or provider-backed work.

The isolated 3.12 candidate validation on 2026-09-23 passed these selected checks:

- 170 API tests after excluding the one dry-plan absence mismatch and the four repeated-audit reuse cases.
- 130 CLI tests after excluding four source-reproduced failures: the text report line, two repeated-audit reuse assertions, and the 100 ms help-time threshold.
- 64 error tests after excluding `tests/errors/test_error_catalogue.py`, which reports source-raised codes missing from the catalogue.
- Contract tests and v3 compatibility tests.
- A wheel and sdist from the candidate, plus a wheel rebuilt from its sdist. Both include schemas and the offline report bundle, and both exclude `reproducibility/**` and `development-snapshots/**`.

The broader suite remains public where it documents intended behavior, but it is visibly incomplete. A separately copied immutable source snapshot reproduced the API dry-plan and repeated-audit failures, all four CLI failures, and the missing error-catalogue codes. The public-surface suite also requires optional scientific and configuration dependencies that are outside the base CPU closure; after NumPy was added, it still required SciPy and `pydantic-settings`. Those results are retained in the publication-run validation logs rather than being represented as green.

The generated `code-reward` example is a fixture. `reward-lens audit` completes offline with zero spend but exits 2 because its known-wrong tasks produce blocking findings and the outcome remains unresolved. It is a working report-generation smoke check, not an all-clear audit result.

Run the supported public contract after creating an isolated environment:

```bash
python -m pip install -e . pytest hypothesis hypothesis-jsonschema
pytest -q tests/contracts
pytest -q tests/api --deselect tests/api/test_absence_records.py::test_dry_run_promises_what_the_run_fills_and_nothing_it_cannot_measure --deselect tests/api/test_reuse_installed.py
pytest -q tests/cli --deselect tests/cli/test_output_contract.py::test_the_envelope_names_the_project_and_the_text_names_the_working_directory --deselect tests/cli/test_reuse.py::test_a_second_audit_of_the_written_example_returns_the_stored_record --deselect tests/cli/test_reuse.py::test_the_second_audit_prints_the_stored_verdict_over_the_reused_line --deselect tests/cli/test_root_help.py::test_help_completes_under_100_ms_as_the_minimum_of_twenty_runs
pytest -q tests/errors --ignore tests/errors/test_error_catalogue.py
pytest -q tests/compat
```

Package closure checks create local build and virtual-environment output. Run them only in a disposable clone or staging checkout.
