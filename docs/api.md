# The Python API

`reward_lens.api` is the whole product. The terminal, the MCP server, the CI action and the
workbench are thin over this module: they parse, map, call and format, and none of them computes
anything (D-67). A notebook cell and a shell pipeline get the same record, and a gate proves it by
running the demo both ways and diffing the records outside the volatile fields.

```python
from pathlib import Path
from reward_lens.api import AuditRequest, audit

record = audit(AuditRequest(path=Path("./demo")))
print(len(record.holes), "things this build could not measure")
```

## The verbs

| Function | Request | Result |
|---|---|---|
| `audit(request)` | `AuditRequest` | `Assay` |
| `trace(request)` | `TraceRequest` | `Assay` |
| `compare(request)` | `CompareRequest` | `Assay` |
| `forecast_issue(request)` | `ForecastIssueRequest` | `Assay` |
| `forecast_resolve(request)` | `ForecastResolveRequest` | `Assay` |
| `forecast_ledger(request)` | `ForecastLedgerRequest` | `Assay` |
| `improve(request)` | `ImproveRequest` | `Assay` |
| `open_record(path)` | a path | `Assay` |
| `export(request)` | `ExportRequest` | a written file |
| `doctor(request=None, *, project=None)` | `DoctorRequest` | `Capabilities` |
| `dry_run(request)` | `AuditRequest` | `Plan` |
| `last_reuse()` | nothing | `Reused` or `None` |

`last_reuse()` is the one function that takes no request. An audit of a subject the store has
already measured, with nothing it depends on changed, does not measure it again: it hands back the
record that is on disk, byte for byte. That record therefore says nothing about the run that
fetched it, and it should not. The run says it here instead: which record answered, the subject
version the store and the project agree on, the file it was read from, whether the report or the
manifest had to be composed again, and a sentence to print. The answer is `None` when the last
audit measured, and `None` in a build with no audit engine, where nothing can be reused because
nothing can be measured. `Reused` is the audit engine's own type, read through the same lazy seam
as every verb; the same fields travel out in the result envelope's `execution.reused`, so it is
not a type this surface keeps to itself.

`dry_run` returns a `Plan`, which carries `estimate_s` beside `estimate_usd`: the seconds the run
is expected to take, before it runs. Zero is what a plan carries when nothing estimated it.

Every result other than `Reused` is a contract type from `reward_lens.contracts`. There is no result type that exists
for one surface only, and there is no option the CLI exposes as a flag that is not a field on a
request here: the run directory, the budget cap, offline, the sandbox tier, the seed and the rest
are fields rather than flags.

Section 8.0 names `ImproveResult` as the result of `improve`. No such contract type exists, and a
result type on one surface only is forbidden, so `improve` returns the record.

## What this build does not hold

Most engines land in later waves. A verb whose engine is absent returns a record whose ten
sections are honest absences: one entry per section with `missing_access` set to
`instrument not in this build`, a remedy that says what would fix it, and a `holes` index with one
row per section. It does not raise, and it does not report a zero where it measured nothing.

Two fields the frozen schema requires non-null have no honest value in a build that reads no
subject. `subject.version.digest` is the digest of a mapping naming the unread request, and
`subject.digests.instrument_method` is the digest of the instrument set that ran, which is empty.
Every entry carries a limitation that says so. Nothing is a zero digest and nothing is a plausible
hash of something that was never read.

`export` is the one verb that refuses instead. It reads the same seam as every other verb, and
what it finds there while the renderer is absent is nothing, so it raises `CapabilityUnavailable`
with code `RL0701` rather than writing an empty file that looks like a report. The refusal is the
seam's emptiness and not a decision taken in the verb: when the renderer lands, `export` reaches it
without a line here changing.

`doctor` returns `Capabilities`: the list of `Capability` the interface freezes, plus the sandbox
and the installation. It is the list, so a caller iterates it, measures it, indexes it and appends
to it; `sandbox` is a `reward_lens.execution.Sandbox` or `None`, checked when one is supplied
rather than imported when the module loads.

## Errors

Every failure is a `RewardLensError` carrying the `RL` code the CLI prints, a remediation, and the
exit code of the D-22 table.

| Code | Raised by | Exit |
|---|---|---|
| `RL0003` | any request, on a field whose value it cannot use | 4 |
| `RL0002` | any request, on an identifier it cannot use | 4 |
| `RL0701` | `export`, while the renderer is not in the build | 5 |
| `RL0621` | `open_record` on a path that holds no record | 4 |
| `RL0604` | `open_record` on a record that does not validate | 4 |

A request that refuses a value refuses it as one of those two codes and never as a
`pydantic.ValidationError`: the models validate through one path, which names the field and says
what is wrong with the value it was given. `RL0002` is for the fields that carry an identifier
(`run`, `resume`, `baseline`, `candidate`, `forecast_id`) and `RL0003` for every other field,
including a field the request does not have. A caller therefore catches one family, which is why
the CLI needs no error logic of its own. The one pydantic error left is an assignment to a request
already built: a request is frozen, and a write to one is a mistake in the calling code rather than
an input.

`RL0604` carries the schema path that caught the record and the original bytes, unaltered, on
`error.context["original_bytes"]`. A record this build cannot interpret is handed back as it
arrived rather than rewritten (D-64).

`doctor` never raises for a missing capability. Reporting that a capability is unavailable is what
it is for, and it exits 0 whenever it produced its report.

## Stability

The SDK carries the schema's stability contract (D-64). Within a major version, functions, request
fields and result fields are added; nothing is removed and nothing is renamed. A name that has to
change keeps working for one major behind a deprecation shim. An option added to the CLI is added
here first, because the CLI has no way to reach anything this module does not expose.

Two things are not part of the contract and may change in a patch release: the private modules
(`reward_lens.api._dispatch`, `reward_lens.api._record`), and which engines a given build holds,
which `doctor` reports and the record's holes record.

## Requests

Requests are pydantic models, strict, closed and frozen. A misspelt field is refused rather than
ignored; a float is never money (`max_budget_usd` is a string such as `"5.00"`); a field typed
`Path` also accepts the string a terminal or a JSON payload hands it, and nothing else is coerced.

```python
AuditRequest(
    path,                      # the project or the grader
    tasks=None, responses=None, outcome=None,
    seeker="off",              # off | api | local | agent
    max_budget_usd=None,       # "5.00"
    offline=True,
    sandbox="auto", require_tier=None,
    dry_run=False,
    seed=20260911,
    run_dir=None, name=None, resume=None,
    only=(),                   # panel names
    policy=None,
    non_interactive=False,
)
```

## Imports

Importing `reward_lens.api` loads the contracts and the standard library. It loads nothing
numeric, no HTTP client, and no engine: the seam between the verbs and the engines under them is a
pointer table resolved on demand, so a build that holds an engine pays for it and a build that
does not pays nothing. A test asserts this in a fresh interpreter.
