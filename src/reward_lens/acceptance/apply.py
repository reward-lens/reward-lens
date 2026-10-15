"""Attach, retain and persist policy decisions on canonical Assay records."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decide import decide
from .policies import load_policy
from .types import CheckResult, Decision, Policy, PolicyInvalid, RegressionCase

_EXTENSION = "https://reward-lens.org/acceptance-policy/v1"
RegressionRunner = Callable[[RegressionCase, Any], CheckResult]
_RUNNERS: dict[str, RegressionRunner] = {}


@dataclass(frozen=True)
class PersistedDecision:
    """The re-opened record and paths returned by the existing Audit writer."""

    record: Any
    decision: Decision
    paths: Mapping[str, Path]


def register_regression_runner(evidence_ref: str, runner: RegressionRunner) -> None:
    """Register the one runnable producer for a retained failure reference."""
    if evidence_ref in _RUNNERS:
        raise PolicyInvalid(f"a regression runner is already registered for {evidence_ref!r}")
    _RUNNERS[evidence_ref] = runner


def _dependency_snapshot(record: Any, case: RegressionCase) -> dict[str, str]:
    values: dict[str, str] = {}
    digests = getattr(getattr(record, "subject", None), "digests", None)
    for dependency in case.depends_on:
        if not dependency.startswith("digest:"):
            raise PolicyInvalid(
                f"regression case {case.id!r} names invalid dependency {dependency!r}"
            )
        value = getattr(digests, dependency.removeprefix("digest:"), None)
        if not isinstance(value, str):
            raise PolicyInvalid(
                f"regression case {case.id!r} cannot retain unavailable dependency {dependency!r}"
            )
        values[dependency] = value
    return values


def _case_payload(
    record: Any, policy: Policy, cases: Iterable[RegressionCase]
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for case in cases:
        if case.from_version != policy.version:
            raise PolicyInvalid(
                f"regression case {case.id!r} belongs to policy version {case.from_version!r}, "
                f"not {policy.version!r}"
            )
        retained.append(
            {
                "id": case.id,
                "from_version": case.from_version,
                "evidence_ref": case.evidence_ref,
                "depends_on": list(case.depends_on),
                "dependency_digests": _dependency_snapshot(record, case),
            }
        )
    return retained


def apply_policy(
    record: Any,
    policy: Policy | Path | str,
    *,
    regression_cases: Iterable[RegressionCase] = (),
    action: str | None = None,
) -> Decision:
    """Evaluate a declared policy and attach its decision to this Assay.

    The Assay's existing ``decision`` contract is the persisted authority.  The
    extension only retains the policy-versioned failure inputs needed for a
    later registered rerun; it does not duplicate the policy's required keys.
    """
    declared = load_policy(policy) if isinstance(policy, (Path, str)) else policy
    decision = decide(record, declared, action=action)
    cases = _case_payload(record, declared, regression_cases)
    contract = decision.as_contract()
    contract.regression_cases = [case["id"] for case in cases]
    record.decision = contract
    extensions = dict(getattr(record, "extensions", None) or {})
    extensions[_EXTENSION] = {
        "policy_version": declared.version,
        "policy_digest": declared.digest,
        "regression_cases": cases,
    }
    record.extensions = extensions
    return decision


def retained_regression_cases(record: Any) -> tuple[RegressionCase, ...]:
    """Read retained cases only when their decision still names their policy version."""
    payload = (getattr(record, "extensions", None) or {}).get(_EXTENSION, {})
    policy_version = payload.get("policy_version")
    result: list[RegressionCase] = []
    for raw in payload.get("regression_cases", []):
        if not isinstance(raw, dict) or raw.get("from_version") != policy_version:
            raise PolicyInvalid(
                "retained regression case is not bound to its recorded policy version"
            )
        result.append(
            RegressionCase(
                id=raw["id"],
                from_version=raw["from_version"],
                evidence_ref=raw["evidence_ref"],
                depends_on=tuple(raw["depends_on"]),
                dependency_digests=tuple(sorted(dict(raw.get("dependency_digests", {})).items())),
            )
        )
    return tuple(result)


def rerun_changed_regressions(previous: Any, candidate: Any) -> dict[str, CheckResult]:
    """Run only retained cases whose actual saved dependency digest changed."""
    outcomes: dict[str, CheckResult] = {}
    for case in retained_regression_cases(previous):
        stored = dict(case.dependency_digests)
        current = _dependency_snapshot(candidate, case)
        if stored == current:
            continue
        runner = _RUNNERS.get(case.evidence_ref)
        if runner is None:
            raise PolicyInvalid(f"changed regression case {case.id!r} has no registered runner")
        outcomes[case.id] = runner(case, candidate)
    return outcomes


def apply_and_persist(
    record: Any,
    policy: Policy | Path | str,
    *,
    project: Any,
    name: str,
    fallback_dir: Path,
    regression_cases: Iterable[RegressionCase] = (),
    action: str | None = None,
) -> PersistedDecision:
    """Public finalization seam for Audit after it assembles its canonical Assay.

    The import is intentionally lazy: acceptance owns the application of the
    policy, while the released Audit writer remains the sole persistence
    authority.  Before C045 is merged this fails at that explicit seam rather
    than writing a competing bundle format.
    """
    decision = apply_policy(record, policy, regression_cases=regression_cases, action=action)
    from reward_lens.product.audit.run import persist_assay_bundle

    saved, paths = persist_assay_bundle(
        record, project=project, name=name, fallback_dir=fallback_dir
    )
    assay_path = paths.get("assay")
    if assay_path is None:
        raise PolicyInvalid("Audit persistence did not return an assay path")
    from reward_lens.contracts import Assay

    reopened = Assay.model_validate(json.loads(Path(assay_path).read_text(encoding="utf-8")))
    return PersistedDecision(record=reopened, decision=decision, paths=paths)
