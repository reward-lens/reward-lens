"""Source-derived contract tests for the five-way trace answer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from reward_lens import contracts
from reward_lens.product.trace import open_trace
from reward_lens.product.trace.answer import (
    Answer,
    AnswerKind,
    EntryRef,
    RemediationKind,
    answer,
    answer_entry,
    answer_from_entry,
)
from reward_lens.product.trace.record import Series, TraceRecord, TraceRow


def _record(
    rows: tuple[TraceRow, ...],
    *,
    complete: bool = True,
    torn: bool = False,
    outcome: tuple[float | None, ...] | None = None,
    oracle_disagreement: tuple[float | None, ...] | None = None,
    present: tuple[str, ...] = ("reward_recorded", "reward_applied", "advantage", "selection"),
) -> TraceRecord:
    rows = tuple(rows)
    values = {
        name: tuple(getattr(row, name) for row in rows)
        for name in ("reward_recorded", "reward_applied", "advantage", "selection")
    }
    series = {
        name: Series(
            name=name,
            values=values[name],
            present=name in present,
            absent_reason="source did not record this series" if name not in present else "",
        )
        for name in values
    }
    return TraceRecord(
        reader="test",
        source="source-derived.jsonl",
        run_id="test",
        rows=rows,
        series=series,
        components={
            **(
                {"independent_outcome": Series("independent_outcome", outcome)}
                if outcome is not None
                else {}
            ),
            **(
                {"oracle_disagreement": Series("oracle_disagreement", oracle_disagreement)}
                if oracle_disagreement is not None
                else {}
            ),
        },
        access_level="RECORD",
        access_statement="record evidence",
        policy="strict",
        complete=complete,
        completeness_statement="test record",
        subject_ref="sha256:" + "a" * 64,
        depends_on=("digest:training_semantics",),
        torn_tail=(object() if torn else None),
    )


def _rows(
    *, score: tuple[float, ...], advantage: tuple[float, ...], selection: tuple[float | None, ...]
) -> tuple[TraceRow, ...]:
    return tuple(
        TraceRow(
            index=index,
            step=index // 2,
            group_id=f"g{index // 2}",
            row_id=f"r{index}",
            components={},
            reward_recorded=value,
            reward_applied=value,
            advantage=advantage[index],
            selection=selection[index],
        )
        for index, value in enumerate(score)
    )


def _provenance() -> contracts.EntryProvenance:
    return contracts.EntryProvenance(
        started="2026-09-26T00:00:00Z", duration_s=0.25, sandbox_tier="L3", offline=True
    )


def _qualified_outcome(subject_ref: str) -> contracts.Entry:
    return contracts.estimate(
        "validity",
        "validity.outcome_qualification",
        "protected outcome qualification",
        1.0,
        "fraction",
        2,
        "task",
        contracts.Uncertainty(
            method="no_interval_below_15_clusters", level=0.95, reason="fixture", clusters=2
        ),
        subject_ref=subject_ref,
        provenance=_provenance(),
        method=contracts.Method(
            id="outcome.protected_suite",
            version="1.0.0",
            params_digest=contracts.digest({"fixture": "qualified-outcome"}),
            procedure="run known-good and known-wrong protected controls",
        ),
        result={
            "known_good_passed": 2,
            "known_good_total": 2,
            "known_wrong_failed": 2,
            "known_wrong_total": 2,
            "partition_digest": "sha256:" + "b" * 64,
        },
    )


def _reproduced_seeker_exploit(subject_ref: str) -> contracts.Entry:
    return contracts.Entry(
        entry_id="exploits.seeker.scripted.witness_0",
        section="exploits",
        kind="witness",
        measurand="reward accepted response whose independent outcome was unmet",
        method=contracts.Method(
            id="exploits.seeker",
            version="1.0.0",
            params_digest=contracts.digest({"fixture": "seeker"}),
            procedure="reproduce audited reward and independent outcome pair",
        ),
        scope="evaluator_comparison",
        subject_ref=subject_ref,
        depends_on=[],
        state="complete",
        provenance=_provenance(),
        limitations=["fixture is one reproduced proposal"],
        result={
            "actor": "scripted",
            "budget_usd": "0.00",
            "spent_usd": "0.00",
            "proposal_id": "sha256:" + "c" * 64,
            "reward": 1.0,
            "threshold": 0.9,
            "outcome_passed": False,
            "arm": "black_box",
        },
        witness=contracts.Witness(
            inputs={"task": "fixture"},
            procedure="the recorded pair reproduced",
            observed="reward accepted and outcome rejected",
            artifact_identity="sha256:" + "c" * 64,
        ),
    )


def _public_run_source() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "short"


def _reopened_fiveway(assay: contracts.Assay):
    entries = tuple(assay.measurement.trace)
    persisted = next(entry for entry in entries if entry.entry_id == "trace.fiveway")
    return persisted, answer_from_entry(
        persisted, evidence_entries=tuple(entry for entry in entries if entry is not persisted)
    )


def test_the_frozen_observational_domain_has_exactly_five_kinds() -> None:
    assert {kind.name for kind in AnswerKind} == {
        "OPTIMISED_AS_INTENDED",
        "OPTIMISED_PROXY",
        "SELECTED_NOT_TRAINED",
        "NO_SIGNAL",
        "INCONCLUSIVE_INPUTS",
    }


def test_each_observational_kind_is_reachable_from_record_evidence() -> None:
    intended = _record(
        _rows(score=(0.1, 0.9, 0.2, 1.0), advantage=(-1.0, 1.0, -1.0, 1.0), selection=(None,) * 4),
        outcome=(0.0, 1.0, 0.0, 1.0),
    )
    proxy = _record(
        _rows(score=(0.1, 0.9, 0.2, 1.0), advantage=(-1.0, 1.0, -1.0, 1.0), selection=(None,) * 4),
        outcome=(1.0, 0.0, 1.0, 0.0),
    )
    selected = _record(_rows(score=(0.1, 0.9), advantage=(0.0, 0.0), selection=(0.0, 1.0)))
    signal = _record(
        _rows(score=(0.5, 0.5, 0.5, 0.5), advantage=(0.0, 0.0, 0.0, 0.0), selection=(None,) * 4)
    )
    incomplete = _record(
        _rows(score=(0.1, 0.9), advantage=(None, None), selection=(None, None)),
        complete=False,
        torn=True,
        present=("reward_recorded",),
    )

    assert answer(intended).kind is AnswerKind.OPTIMISED_AS_INTENDED
    assert answer(proxy).kind is AnswerKind.OPTIMISED_PROXY
    assert answer(selected).kind is AnswerKind.SELECTED_NOT_TRAINED
    assert answer(signal).kind is AnswerKind.NO_SIGNAL
    assert answer(incomplete).kind is AnswerKind.INCONCLUSIVE_INPUTS


def test_answer_references_resolve_and_every_other_kind_is_distinguished() -> None:
    record = _record(_rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)))
    result = answer(record)
    assert result.evidence
    assert all(reference.resolve(record) for reference in result.evidence)
    assert {kind for kind in result.distinguished_from} == set(AnswerKind) - {result.kind}
    assert all(text.strip() for text in result.distinguished_from.values())


def test_real_grpo_trace_stays_unresolved_without_an_independent_outcome() -> None:
    """The pinned trainer fixture proves the reader route, not a reward-hacking conclusion."""
    root = _public_run_source()
    result = answer(open_trace(root))
    assert result.kind is AnswerKind.INCONCLUSIVE_INPUTS
    assert result.remediation is RemediationKind.RECORD_NEXT_TIME
    assert all(reference.resolve(result.record) for reference in result.evidence)


def test_trace_sdk_persists_a_digest_bound_unresolved_answer_for_the_actual_reader_input() -> None:
    from reward_lens import api

    source = _public_run_source()
    assay = api.trace(api.TraceRequest(run=source.name, project=source))
    persisted, restored = _reopened_fiveway(assay)
    assert persisted.kind == "witness"
    assert restored.kind is AnswerKind.INCONCLUSIVE_INPUTS
    assert restored.remediation is RemediationKind.RECORD_NEXT_TIME
    all_entries = tuple(entry for entry in assay.measurement.trace if entry is not persisted)
    assert all(reference.resolve(all_entries) for reference in restored.evidence)
    assert all(reference.resolve(all_entries) for reference in restored.remediation_evidence)


def test_trace_cli_writes_a_reopenable_fiveway_answer_and_report(tmp_path: Path) -> None:
    from reward_lens import api

    source = tmp_path / "short"
    shutil.copytree(_public_run_source(), source)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "reward_lens.cli.main",
            "trace",
            source.name,
            "--format",
            "json",
            "--no-color",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2, completed.stderr
    envelope = json.loads(completed.stdout)
    assay_path = source / envelope["artifacts"]["assay"]
    report_path = source / envelope["artifacts"]["report"]
    assert assay_path.is_file()
    assert report_path.is_file()
    persisted, restored = _reopened_fiveway(api.open_record(assay_path))
    assert persisted.entry_id == "trace.fiveway"
    assert restored.remediation is RemediationKind.RECORD_NEXT_TIME


def test_scalar_shortcuts_leave_causal_remediation_unresolved() -> None:
    rows = _rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None))
    records = (
        _record(
            (
                TraceRow(**{**row.__dict__, "reward_applied": row.reward_recorded + 0.1})
                for row in rows
            ),
            outcome=(0.0, 1.0),
        ),
        _record(rows, outcome=(1.0, 0.0)),
        _record(rows, outcome=(0.0, 1.0), oracle_disagreement=(1.0, 1.0)),
        _record(
            _rows(score=(0.5, 0.5), advantage=(0.0, 0.0), selection=(None, None)),
            outcome=(0.0, 0.0),
        ),
    )
    for record in records:
        result = answer(record)
        assert result.remediation is RemediationKind.RECORD_NEXT_TIME
        assert result.remediation_evidence
        assert all(text.strip() for text in result.remediation_distinguished_from.values())
    summary = answer(records[0]).summary.lower()
    assert "does not establish a gradient" in summary
    assert "or parameter update" in summary


def test_answer_materializes_trace_entries_once_for_digest_bound_unresolved_refs(
    monkeypatch,
) -> None:
    record = _record(_rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)))
    first = tuple(record.entries())
    different = tuple(entry.model_copy(deep=True) for entry in first)
    different[0].provenance.started = "2030-01-01T00:00:00Z"
    calls = 0

    def alternating_entries(_record: TraceRecord):
        nonlocal calls
        calls += 1
        return first if calls == 1 else different

    monkeypatch.setattr(TraceRecord, "entries", alternating_entries)
    result = answer(record)

    assert calls == 1
    assert result.remediation is RemediationKind.RECORD_NEXT_TIME
    assert all(reference.resolve(result.entries) for reference in result.remediation_evidence)


def test_digest_bound_entry_round_trips_a_reproduced_evaluator_mismatch() -> None:
    record = _record(
        _rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)),
        outcome=(1.0, 0.0),
    )
    support = (
        _qualified_outcome(record.subject_ref),
        _reproduced_seeker_exploit(record.subject_ref),
    )
    result = answer(record, evidence_entries=support)
    assert result.kind is AnswerKind.OPTIMISED_PROXY
    assert result.remediation is RemediationKind.EVALUATOR_OR_SURFACE

    persisted = answer_entry(record, evidence_entries=support, provenance=_provenance())
    restored = answer_from_entry(persisted, evidence_entries=(*record.entries(), *support))
    assert persisted.entry_id == "trace.fiveway"
    assert persisted.kind == "witness"
    assert restored.remediation is RemediationKind.EVALUATOR_OR_SURFACE
    assert {ref.entry_id for ref in restored.remediation_evidence} == {
        "validity.outcome_qualification",
        "exploits.seeker.scripted.witness_0",
    }
    assert all(
        ref.digest and ref.resolve((*record.entries(), *support)) for ref in restored.evidence
    )


def test_positive_evaluator_remedy_requires_one_trace_subject_and_bound_producer_values() -> None:
    record = _record(_rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)))
    other_subject = "sha256:" + "d" * 64
    mixed_subject = (
        _qualified_outcome(other_subject),
        _reproduced_seeker_exploit(record.subject_ref),
    )
    assert (
        answer(record, evidence_entries=mixed_subject).remediation
        is RemediationKind.RECORD_NEXT_TIME
    )

    mismatched_artifact = _reproduced_seeker_exploit(record.subject_ref)
    mismatched_artifact.result["proposal_id"] = "sha256:" + "e" * 64
    assert (
        answer(
            record,
            evidence_entries=(_qualified_outcome(record.subject_ref), mismatched_artifact),
        ).remediation
        is RemediationKind.RECORD_NEXT_TIME
    )

    boolean_control_count = _qualified_outcome(record.subject_ref)
    boolean_control_count.result["known_good_passed"] = True
    assert (
        answer(
            record,
            evidence_entries=(
                boolean_control_count,
                _reproduced_seeker_exploit(record.subject_ref),
            ),
        ).remediation
        is RemediationKind.RECORD_NEXT_TIME
    )


def test_persisted_entry_refuses_a_forged_causal_remediation() -> None:
    record = _record(_rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)))
    persisted = answer_entry(record, provenance=_provenance())
    persisted.result["answer"]["remediation"] = RemediationKind.EVALUATOR_OR_SURFACE.value
    try:
        answer_from_entry(persisted, evidence_entries=tuple(record.entries()))
    except ValueError as error:
        assert "not supported" in str(error)
    else:
        raise AssertionError(
            "a scalar-only record may not be relabelled as causal evaluator evidence"
        )


def test_answer_construction_refuses_empty_or_unresolvable_evidence_and_blank_neighbour() -> None:
    record = _record(_rows(score=(0.1, 0.9), advantage=(-1.0, 1.0), selection=(None, None)))
    try:
        Answer(
            kind=AnswerKind.OPTIMISED_AS_INTENDED,
            evidence=(),
            distinguished_from={AnswerKind.NO_SIGNAL: "scores would be flat"},
            remediation=RemediationKind.RECORD_NEXT_TIME,
            remediation_evidence=(EntryRef("trace.reconstruction"),),
            remediation_distinguished_from={
                RemediationKind.OBJECTIVE_OR_TRAINER: "need a causal intervention"
            },
            summary="x",
            record=record,
        )
    except ValueError as error:
        assert "evidence" in str(error)
    else:
        raise AssertionError("empty evidence must be refused")

    try:
        Answer(
            kind=AnswerKind.OPTIMISED_AS_INTENDED,
            evidence=(EntryRef("trace.not-in-record"),),
            distinguished_from={
                kind: "a recorded contrary observation"
                for kind in AnswerKind
                if kind is not AnswerKind.OPTIMISED_AS_INTENDED
            },
            remediation=RemediationKind.RECORD_NEXT_TIME,
            remediation_evidence=(EntryRef("trace.reconstruction"),),
            remediation_distinguished_from={
                kind: "a recorded causal discriminator"
                for kind in RemediationKind
                if kind is not RemediationKind.RECORD_NEXT_TIME
            },
            summary="source-bound",
            record=record,
        )
    except ValueError as error:
        assert "does not resolve" in str(error)
    else:
        raise AssertionError("unresolvable evidence must be refused")
