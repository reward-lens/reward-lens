# Validation

Run the focused standard checks from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/standard
```

The checks execute the E1 assay builder and `validate_record`, open the
provenance-bound `verifiers` fixture through `verifiers_results`, and exercise
the real refusal paths below. They do not accept documentation text as proof of
an instance.

## Required refusals

| Condition | Required result |
| --- | --- |
| Unknown field in a closed assay object | Refuse the invalid instance with `RL0604 RECORD_INVALID`. |
| Incompatible major in `$schema` | Refuse with `RL0601 SCHEMA_UNKNOWN_MAJOR` and preserve original bytes. |
| Missing version field | Refuse the invalid instance. Do not infer a schema from its contents. |
| Invalid instance | Refuse before measurement or reporting. |
| Missing or mismatched example provenance | Treat the example as unverifiable and refuse interpretation. |
| Raw source `reward` column without declared role | Preserve only as a convenience value; do not label it a standard reward field. |

For an assay input, call `validate_record(record, raw=original_bytes)` while
the original bytes are still available. For a run-record import, check both
root `schema_version` fields before decoding table rows. For a framework result
directory, open it through its declared reader so that its provenance verifier
checks the recorded artifact before values are projected.

The standard has two validated local instances: the generated E1 assay in
[the assay procedure](assay.md#generated-e1-instance), and the provenance-bound
per-component `verifiers` instance in [the run-record procedure](run-record.md#per-component-verifiers-instance).
