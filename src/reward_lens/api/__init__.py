"""The public SDK (D-67): audit, trace, compare, forecast_issue, forecast_resolve,
forecast_ledger, improve, open_record, export and doctor, taking typed requests and returning the
record's own contract types. Every other surface is thin over this module. Owned by P-SDK.

The stability rule is the schema's (D-64): fields and functions are added, nothing is removed or
renamed inside a major version, and a deprecated name keeps working for one major with a shim.
The page at `docs/api.md` is the contract a caller reads.

In wave 1 most engines are not in the build. A verb whose engine is absent returns a record whose
ten sections are honest absences and whose holes index says what is missing and what would fix it;
it does not raise, and it does not guess. `export` is the exception: it reads the same seam as the
other verbs, and when the renderer behind it is absent it refuses with RL0701 rather than writing
an empty file that looks like a report.
"""

import json
import pathlib
import platform
import sys
import time
import typing

from reward_lens import contracts

from . import _dispatch, _record
from .requests import (
    DEFAULT_SEED,
    AuditRequest,
    Capabilities,
    Capability,
    CompareRequest,
    DoctorRequest,
    ExportRequest,
    ForecastIssueRequest,
    ForecastLedgerRequest,
    ForecastResolveRequest,
    ImproveRequest,
    InstallInfo,
    Plan,
    TraceRequest,
)

if typing.TYPE_CHECKING:  # pragma: no cover - the engine is never imported at runtime here
    from reward_lens.product.audit import Reused

__all__ = [
    "AuditRequest",
    "Capabilities",
    "Capability",
    "CompareRequest",
    "DEFAULT_SEED",
    "DoctorRequest",
    "ExportRequest",
    "ForecastIssueRequest",
    "ForecastLedgerRequest",
    "ForecastResolveRequest",
    "ImproveRequest",
    "InstallInfo",
    "Plan",
    "TraceRequest",
    "audit",
    "compare",
    "doctor",
    "dry_run",
    "export",
    "forecast_issue",
    "forecast_resolve",
    "forecast_ledger",
    "improve",
    "last_reuse",
    "open_record",
    "trace",
]


def _honest(
    verb: str,
    *,
    path=None,
    offline: bool = True,
    seed: int = DEFAULT_SEED,
    reproduce=(),
    started: float,
) -> contracts.Assay:
    return _record.absent_record(
        command=verb,
        reproduce=reproduce or [f"reward-lens {verb.replace('_', ' ')}"],
        path=path,
        offline=offline,
        seed=seed,
        started_monotonic=started,
    )


def audit(request: AuditRequest) -> contracts.Assay:
    """Measure one reward system and return the record (section 8.0).

    Dispatches to `reward_lens.product.audit.run(req, *, project, sandbox)` when that module is in
    the build, and builds the honest all-absence record when it is not.
    """
    started = time.monotonic()
    engine = _dispatch.engine("audit")
    if engine is not None:
        return engine(request, **_dispatch.context(request.path))
    return _honest(
        "audit",
        path=request.path,
        offline=request.offline,
        seed=request.seed,
        reproduce=[f"reward-lens audit {request.path} --seed {request.seed}"],
        started=started,
    )


def last_reuse() -> "Reused | None":
    """What the last `audit` in this context did instead of measuring, or `None` if it measured.

    An audit of a subject the store has already measured, with nothing it depends on changed,
    hands back the record that is on disk rather than measuring again (A-016, D-28). What comes
    back is the stored bytes: the sealed record, unchanged, which therefore says nothing about
    the run that fetched it. That sentence lives here instead, beside the record rather than in
    it: which record answered, the subject version the store and the project agree on, the file
    it was read from, and whether the run had to compose the report or the manifest again from
    it. `None` is the answer when the last audit measured, and also in a build that does not
    hold the audit engine, where nothing can be reused because nothing can be measured.
    """
    reader = _dispatch.load("reward_lens.product.audit:last_reuse")
    if reader is None:
        return None
    return reader()


def trace(request: TraceRequest) -> contracts.Assay:
    """Read back what a training run actually optimised (section 8.0)."""
    started = time.monotonic()
    engine = _dispatch.engine("trace")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    return _honest(
        "trace",
        path=request.project,
        offline=request.offline,
        seed=request.seed,
        reproduce=[f"reward-lens trace {request.run or '<run>'}"],
        started=started,
    )


def compare(request: CompareRequest) -> contracts.Assay:
    """Compare two versions of one reward system on the rungs this build holds (section 8.0)."""
    started = time.monotonic()
    engine = _dispatch.engine("compare")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    return _honest(
        "compare",
        path=request.project,
        offline=request.offline,
        seed=request.seed,
        reproduce=[f"reward-lens compare {request.baseline} {request.candidate}"],
        started=started,
    )


def forecast_issue(request: ForecastIssueRequest) -> contracts.Assay:
    """Write a claim down before the evidence arrives (section 8.0)."""
    started = time.monotonic()
    engine = _dispatch.engine("forecast_issue")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    return _honest(
        "forecast_issue",
        path=request.project,
        offline=request.offline,
        seed=request.seed,
        reproduce=["reward-lens forecast issue"],
        started=started,
    )


def forecast_resolve(request: ForecastResolveRequest) -> contracts.Assay:
    """Resolve a claim already on the ledger against what happened (section 8.0)."""
    started = time.monotonic()
    engine = _dispatch.engine("forecast_resolve")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    return _honest(
        "forecast_resolve",
        path=request.project,
        offline=request.offline,
        seed=request.seed,
        reproduce=[f"reward-lens forecast resolve {request.forecast_id}"],
        started=started,
    )


def forecast_ledger(request: ForecastLedgerRequest) -> contracts.Assay:
    """The standing ledger of issued and resolved claims (section 8.0)."""
    started = time.monotonic()
    engine = _dispatch.engine("forecast_ledger")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    return _honest(
        "forecast_ledger",
        path=request.project,
        offline=request.offline,
        seed=request.seed,
        reproduce=["reward-lens forecast ledger"],
        started=started,
    )


def improve(request: ImproveRequest) -> contracts.Assay:
    """What to change about a reward system, and what changing it would cost (section 8.0).

    Section 8.0 names the result `ImproveResult`. No such contract type exists, and a result type
    that lives on one surface only is forbidden, so the result is the record: an `Assay` whose
    `improve` findings are entries like any other. The frozen wave-1 interface says the same.
    """
    started = time.monotonic()
    engine = _dispatch.engine("improve")
    if engine is not None:
        return engine(request, **_dispatch.context(request.path))
    return _honest(
        "improve",
        path=request.path,
        offline=request.offline,
        seed=request.seed,
        reproduce=[f"reward-lens improve {request.path}"],
        started=started,
    )


def open_record(path: pathlib.Path) -> contracts.Assay:
    """Read a record off disk, refusing anything this build cannot trust.

    A missing file is RL0621. A file that does not validate is RL0604, carrying the schema path
    that caught it and the original bytes, unaltered, so the caller can hand them on (D-64).
    """
    path = pathlib.Path(path)
    try:
        raw = path.read_bytes()
    except OSError as why:
        raise contracts.UsageError(
            code="RL0621",
            message=f"no record at {path}",
            remediation="check the path, or list what the store holds with `reward-lens runs`",
            context={"path": str(path), "detail": str(why)},
        ) from why
    try:
        data = json.loads(raw)
    except ValueError as why:
        raise contracts.RecordInvalid(
            message=f"the file at {path} is not JSON: {why}",
            remediation="a record is one JSON object; the schema is at schema/assay/1.0/",
            context={"path": str(path), "schema_path": ["$"], "original_bytes": raw},
        ) from why
    try:
        contracts.validate_record(data, raw=raw)
    except contracts.RecordInvalid as invalid:
        invalid.context.setdefault("original_bytes", raw)
        invalid.context.setdefault("path", str(path))
        raise
    return contracts.Assay.model_validate(data)


def export(request: ExportRequest):
    """Write a record out as something a reader can open (section 8.0).

    Dispatches to `reward_lens.product.export.run(req, *, project, sandbox)` through the same seam
    as every other verb. The refusal below is what the seam's absence means, not a decision taken
    here: when the renderer lands, this verb reaches it without a line changing. There is no honest
    absence record to return instead, because the output of `export` is a file rather than a
    record, and an empty file that claims to be a report is the one answer worse than refusing.
    """
    engine = _dispatch.engine("export")
    if engine is not None:
        return engine(request, **_dispatch.context(request.project))
    raise contracts.CapabilityUnavailable(
        code="RL0701",
        message="export needs the report renderer, which is not in this build",
        remediation=(
            "install a reward-lens build that ships the renderer, or read the record itself "
            "with `reward-lens open`"
        ),
        context={"format": request.format, "dest": str(request.dest) if request.dest else None},
    )


def doctor(request: DoctorRequest = None, *, project: pathlib.Path = None) -> Capabilities:
    """What this installation can and cannot do (section 8.0, D-22: it never exits non-zero).

    Dispatches to `reward_lens.product.access.doctor()` when that module is in the build.
    """
    where = project if request is None else request.project
    engine = _dispatch.engine("doctor")
    if engine is not None:
        return engine(project=where)
    return Capabilities(
        capabilities=[
            Capability(
                id=verb,
                status="unavailable",
                reason=_record.MISSING_ACCESS,
                unlocks=[target.split(":")[0]],
                cost="0.00",
            )
            for verb, target in sorted(_dispatch.ENGINES.items())
        ],
        sandbox=None,
        install=InstallInfo(
            version=_record._tool_version(),
            python=platform.python_version(),
            platform=sys.platform,
            extras=(),
        ),
    )


def dry_run(request: AuditRequest) -> Plan:
    """What a run would do before it does it (section 8.0)."""
    engine = _dispatch.engine("dry_run")
    if engine is not None:
        return engine(request, **_dispatch.context(request.path))
    return Plan(
        command="audit",
        panels=(),
        paid_calls=0,
        estimate_usd="0.00",
        notes=(
            f"{_record.MISSING_ACCESS}: this build runs no panel, so the plan is empty",
            f"reward-lens audit {request.path} would produce a record of ten honest absences",
        ),
    )
