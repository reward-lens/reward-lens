"""The trace verb (section 7.4, D-42): reconstruct what the optimiser saw, verify the composition, answer the five-way question, write the incident packet. Spine and primary readers owned by P-TRACE-1; answer.py owned by P-FIVEWAY.

The surface here is frozen for P-TRACE-2, which lands readers as modules under this package and
edits neither this file nor `readers.py`. Everything a reader has to satisfy is in `readers.Reader`;
everything it has to produce is in `record.TraceRecord`; everything it has to check before it
trusts a row is in `integrity.RULES`.

`run` is the engine `reward_lens.api._dispatch.ENGINES` points at. Defining it here is what makes
`reward-lens trace` stop refusing with RL0703 and stop being marked `(not in this build)` in the
root help, because `cli/availability.py` reads that pointer table and looks for this name.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .answer import (
    ANSWER_ENTRY_ID,
    Answer,
    AnswerKind,
    EntryRef,
    RemediationKind,
    answer,
    answer_entry,
    answer_from_entry,
)
from .errors import RecordIncomplete, RecordIntegrity
from .follow import Chunk, chunk_path, follow, seal_chunk, seal_path
from .integrity import (
    RULES,
    CheckInput,
    IntegrityRule,
    RuleResult,
    Violation,
    check,
    enforce,
    require_series,
    rows_digest,
)
from .readers import PACKAGE, Reader, ReaderNotFound, discover, import_failures, reader_for
from .record import (
    SERIES_NAMES,
    CompositionTelemetry,
    DeclaredReward,
    PlannedContrastDesign,
    Policy,
    PriceSource,
    ScalarLossStep,
    ScalarLossToken,
    Series,
    StepMetric,
    TornTail,
    TraceMetadata,
    TraceRecord,
    TraceRow,
)

__all__ = [
    "ANSWER_ENTRY_ID",
    "PACKAGE",
    "RULES",
    "SERIES_NAMES",
    "Answer",
    "AnswerKind",
    "CheckInput",
    "Chunk",
    "CompositionTelemetry",
    "DeclaredReward",
    "EntryRef",
    "IntegrityRule",
    "Policy",
    "PlannedContrastDesign",
    "PriceSource",
    "Reader",
    "ReaderNotFound",
    "RemediationKind",
    "RecordIncomplete",
    "RecordIntegrity",
    "RuleResult",
    "Series",
    "ScalarLossStep",
    "ScalarLossToken",
    "StepMetric",
    "TornTail",
    "TraceRecord",
    "TraceMetadata",
    "TraceRow",
    "Violation",
    "answer",
    "answer_entry",
    "answer_from_entry",
    "check",
    "chunk_path",
    "discover",
    "enforce",
    "follow",
    "import_failures",
    "open_trace",
    "reader_for",
    "require_series",
    "rows_digest",
    "run",
    "seal_chunk",
    "seal_path",
]


def open_trace(path: Any, *, policy: Policy = "strict", reader: str | None = None) -> TraceRecord:
    """Read one artifact with whichever reader recognises it."""
    return reader_for(path, name=reader).read(Path(path), policy=policy)


def run(request: Any, *, project: Any = None, sandbox: Any = None) -> Any:
    """Read back what a training run actually optimised, and return the record (section 8.0).

    A request names a run. It may name it as a run id inside a project, or as the path of the
    artifact itself, and both are tried in that order. When nothing readable is there the honest
    all-absence record stands rather than a refusal: `trace` is in this build, and this particular
    input was not readable, which are two different facts.
    """
    from reward_lens import contracts
    from reward_lens.api import _record as api_record

    started = time.monotonic()
    record: TraceRecord | None = None
    root: Path | None = None
    for candidate in (getattr(request, "project", None), getattr(request, "run", None)):
        if not candidate:
            continue
        here = Path(str(candidate))
        if not here.exists():
            continue
        try:
            found = reader_for(here)
        except ReaderNotFound:
            continue
        # A completions parquet is deliberately a flat table. Its caller must name the reward
        # functions, extras, optional complete weight vector and source group metadata; forwarding
        # them here is the only public route to the reader. Other readers do not receive these
        # TRL-specific declarations and therefore cannot accidentally reinterpret their records.
        trl_declarations: dict[str, Any] = {}
        if found.name == "trl_completions":
            if not getattr(request, "reward_functions", ()):
                raise RecordIncomplete(
                    message=(
                        "the TRL completions parquet has no declared reward-function column set, "
                        "so a flat table cannot establish which values are native reward components"
                    ),
                    remediation=(
                        "pass --reward-function once for every reward-function column and "
                        "--extra-column for every remaining non-reserved column"
                    ),
                    context={"series": ["reward_recorded"], "source": str(here)},
                )
            trl_declarations = {
                "reward_functions": getattr(request, "reward_functions", ()),
                "extra_columns": getattr(request, "extra_columns", ()),
                "weights": getattr(request, "weight_map", {}),
                "group_column": getattr(request, "group_column", None),
                "group_size": getattr(request, "group_size", None),
            }
        record = found.read(here, policy="strict", **trl_declarations)
        root = here
        break

    named = getattr(request, "run", None) or (str(root) if root else "<run>")
    assay = api_record.absent_record(
        command="trace",
        reproduce=[f"reward-lens trace {named}"],
        path=root,
        offline=bool(getattr(request, "offline", True)),
        seed=int(getattr(request, "seed", 20260911)),
        started_monotonic=started,
    )
    if record is None:
        return assay
    trace_entries = list(record.entries())
    assay.subject.version.digest = record.subject_ref
    assay.measurement.trace = [
        *trace_entries,
        answer_entry(
            record,
            trace_entries=trace_entries,
            provenance=trace_entries[0].provenance,
        ),
    ]
    for section in type(assay.measurement).model_fields:
        for entry in getattr(assay.measurement, section, []) or []:
            entry.subject_ref = record.subject_ref
    assay.decision.reasons = [
        reason for reason in assay.decision.reasons if reason != "required_missing:trace"
    ]
    assay.holes_from_entries()
    assay.assay_id = contracts.digest(assay)
    return assay
