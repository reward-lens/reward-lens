"""Evaluate every acceptance-policy clause against a canonical Assay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .types import (
    CheckResult,
    Consequence,
    Decision,
    Policy,
    PolicyCheck,
    PolicyDigestMismatch,
    UnresolvedTradeoff,
)

_UNRESOLVED = {"missing", "stale", "invalid", "inconclusive"}


def _entries(record: Any, section: str | None) -> list[Any]:
    if not section:
        return []
    measurement = getattr(record, "measurement", None)
    values = getattr(measurement, section, None)
    return list(values or ())


def _fresh(result: CheckResult, entries: list[Any], freshness: timedelta | None) -> CheckResult:
    if result.state != "qualified" or freshness is None:
        return result
    try:
        newest = max(
            datetime.strptime(entry.provenance.started, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            for entry in entries
        )
    except (AttributeError, TypeError, ValueError):
        return CheckResult("invalid", "evidence has no valid producer timestamp")
    if datetime.now(timezone.utc) - newest > freshness:
        return CheckResult("stale", "evidence is older than the policy freshness interval")
    return result


def _section_measured(check: PolicyCheck, record: Any) -> CheckResult:
    entries = _entries(record, check.section)
    if not entries or any(entry.kind == "absence" for entry in entries):
        return CheckResult("missing", "the required producer entry is absent")
    if any(entry.state != "complete" for entry in entries):
        return CheckResult("inconclusive", "the required producer entry is incomplete")
    return _fresh(
        CheckResult("qualified", "required producer entries are complete"), entries, check.fresh_for
    )


def _all_checks_complete(check: PolicyCheck, record: Any) -> CheckResult:
    entries = _entries(record, check.section)
    if not entries or any(entry.kind == "absence" for entry in entries):
        return CheckResult("missing", "a required validity check is absent")
    checks = [entry.check for entry in entries if entry.check is not None]
    if not checks:
        return CheckResult("inconclusive", "the validity panel has no producer check entries")
    if any(entry.state != "complete" for entry in entries):
        return CheckResult("inconclusive", "a validity check is incomplete")
    if any(not item.passed for item in checks):
        return CheckResult("rejected", "a validity check failed")
    return _fresh(CheckResult("qualified", "all validity checks passed"), entries, check.fresh_for)


def _outcome_qualified(check: PolicyCheck, record: Any) -> CheckResult:
    outcome = getattr(getattr(record, "intent", None), "outcome_check", None)
    state = getattr(outcome, "state", None)
    if state is None:
        return CheckResult("missing", "the record names no outcome check")
    if state == "qualified":
        return CheckResult("qualified", "the outcome check is qualified")
    if state == "unqualified":
        return CheckResult("rejected", "the outcome check is unqualified")
    return CheckResult("inconclusive", "the outcome check has no final qualification")


def _finding_guard(check: PolicyCheck, record: Any) -> CheckResult:
    entries = _entries(record, check.section)
    if not entries or any(entry.kind == "absence" for entry in entries):
        return CheckResult("missing", "the required panel is absent")
    entry_ids = {entry.entry_id for entry in entries}
    findings = list(getattr(record, "findings", ()) or ())
    severity = {"none": 0, "note": 1, "warning": 2, "error": 3}
    threshold = severity[check.threshold or "warning"]
    if any(
        set(getattr(finding, "entries", ()) or ()) & entry_ids
        and getattr(finding, "kind", None) in {"open", "fail"}
        and severity.get(getattr(finding, "level", None), -1) >= threshold
        for finding in findings
    ):
        return CheckResult("rejected", "the panel has a finding at or above its declared severity")
    return _fresh(
        CheckResult("qualified", "the panel has no blocking finding"), entries, check.fresh_for
    )


def _producer_entry(
    check: PolicyCheck,
    record: Any,
    *,
    allow_alternate: bool = False,
    require_check: bool = False,
) -> tuple[Any | None, CheckResult | None]:
    identifiers = {check.entry_id}
    if allow_alternate:
        identifiers.update(check.alternate_entry_ids)
    entries = [entry for entry in _entries(record, check.section) if entry.entry_id in identifiers]
    if not entries:
        return None, CheckResult("missing", "the policy-named producer entry is missing")
    if len(entries) != 1:
        return None, CheckResult("invalid", "the policy-named alternate producers are ambiguous")
    entry = entries[0]
    if entry.kind == "absence":
        return None, CheckResult("missing", "the policy-named producer entry is absent")
    if entry.state != "complete":
        return None, CheckResult("inconclusive", "the policy-named producer entry is incomplete")
    if require_check:
        if entry.check is None:
            return None, CheckResult(
                "missing", "the policy-named producer entry has no retained check"
            )
        if entry.check.passed is False:
            return None, CheckResult("rejected", "the policy-named producer check failed")
        if entry.check.passed is not True:
            return None, CheckResult(
                "invalid", "the policy-named producer check has a non-boolean verdict"
            )
    elif entry.check is not None and not entry.check.passed:
        return None, CheckResult("rejected", "the policy-named producer check failed")
    return entry, _fresh(
        CheckResult("qualified", "the policy-named producer entry is complete"),
        entries,
        check.fresh_for,
    )


def _entry_complete(check: PolicyCheck, record: Any) -> CheckResult:
    _entry, result = _producer_entry(check, record)
    assert result is not None
    return result


def _entry_required_check(check: PolicyCheck, record: Any) -> CheckResult:
    _entry, result = _producer_entry(check, record, require_check=True)
    assert result is not None
    return result


def _entry_result_field(check: PolicyCheck, record: Any) -> CheckResult:
    entry, complete = _producer_entry(check, record)
    assert complete is not None
    if complete.state != "qualified":
        return complete
    result = getattr(entry, "result", None)
    if not isinstance(result, dict) or not result.get(check.result_field):
        return CheckResult("missing", "the policy-named producer result field is missing")
    return complete


def _entry_any_complete(check: PolicyCheck, record: Any) -> CheckResult:
    _entry, result = _producer_entry(check, record, allow_alternate=True)
    assert result is not None
    return result


def _entry_any_control_field(check: PolicyCheck, record: Any) -> CheckResult:
    entry, complete = _producer_entry(check, record, allow_alternate=True)
    assert complete is not None
    if complete.state != "qualified":
        return complete
    control = (
        getattr(entry, "result", {}).get(check.result_field)
        if isinstance(getattr(entry, "result", None), dict)
        else None
    )
    if not isinstance(control, dict) or "passed" not in control:
        return CheckResult("missing", "the producer has no retained declared control")
    if control["passed"] is True:
        return complete
    if control["passed"] is False:
        return CheckResult("rejected", "the producer's retained declared control failed")
    return CheckResult(
        "invalid", "the producer's retained declared control has a non-boolean verdict"
    )


def _future_interface(check: PolicyCheck, record: Any) -> CheckResult:
    return CheckResult("missing", "the declared future producer has not emitted a canonical entry")


def _evaluate(check: PolicyCheck, record: Any) -> CheckResult:
    sources = {
        "section_measured": _section_measured,
        "all_checks_complete": _all_checks_complete,
        "outcome_qualified": _outcome_qualified,
        "finding_guard": _finding_guard,
        "entry_complete": _entry_complete,
        "entry_required_check": _entry_required_check,
        "entry_result_field": _entry_result_field,
        "entry_any_complete": _entry_any_complete,
        "entry_any_control_field": _entry_any_control_field,
        "future_interface": _future_interface,
    }
    evaluator = sources.get(check.source)
    if evaluator is None:
        return CheckResult("invalid", f"policy names unknown evidence source {check.source!r}")
    return evaluator(check, record)


def decide(
    record: Any,
    policy: Policy,
    *,
    action: str | None = None,
    overrides: Mapping[str, CheckResult] | None = None,
    recorded_policy_digest: str | None = None,
    tradeoff: tuple[tuple[str, str], tuple[str, str]] | None = None,
) -> Decision:
    """Return the D-13 decision after evaluating every required policy check."""
    named_digest = recorded_policy_digest
    if named_digest is None:
        named_digest = getattr(
            getattr(getattr(record, "decision", None), "policy", None), "digest", None
        )
    if named_digest is not None and named_digest != policy.digest:
        raise PolicyDigestMismatch(expected=policy.digest, recorded=named_digest)
    chosen_action = action or policy.actions[0]
    if chosen_action not in policy.actions:
        raise ValueError(f"{chosen_action!r} is not named by policy {policy.id!r}")
    supplied = dict(overrides or {})
    evaluated = {
        check.key: supplied.get(check.key, _evaluate(check, record)) for check in policy.checks
    }
    rejected = [key for key, result in evaluated.items() if result.state == "rejected"]
    unresolved = [key for key, result in evaluated.items() if result.state in _UNRESOLVED]
    if rejected:
        state = "rejected"
        reasons = [f"blocking_finding:{key}" for key in rejected]
    elif unresolved:
        state = "unresolved"
        reasons = [f"required_{evaluated[key].state}:{key}" for key in unresolved]
    else:
        state = "qualified"
        reasons = []
    if tradeoff is not None:
        consequences = tuple(Consequence(*item) for item in tradeoff)
        return UnresolvedTradeoff(
            "unresolved",
            chosen_action,
            policy.id,
            policy.digest,
            evaluated,
            reasons + ["tradeoff_unsettled:policy.preference"],
            consequences,  # type: ignore[arg-type]
        )
    return Decision(state, chosen_action, policy.id, policy.digest, evaluated, reasons)
