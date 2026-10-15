"""Decision-state precedence and durable policy-version tests."""

from __future__ import annotations

from pathlib import Path

from reward_lens.acceptance import CheckResult, UnresolvedTradeoff, decide, load_policy
from reward_lens.api._record import absent_record


def _policy(name: str):
    from pathlib import Path

    return load_policy(Path(__file__).parents[2] / "src/reward_lens/acceptance/policies" / name)


def test_rejected_precedes_unresolved_and_qualified() -> None:
    policy = _policy("default-evaluator-v1.json")
    values = {key: CheckResult("qualified", "ok") for key in policy.required_evidence}
    values[policy.required_evidence[0]] = CheckResult("rejected", "failed")
    values[policy.required_evidence[1]] = CheckResult("missing", "not present")
    decision = decide(
        absent_record(command="audit", reproduce=["reward-lens audit"]), policy, overrides=values
    )
    assert decision.state == "rejected"
    assert set(decision.evaluated) == set(policy.required_evidence)


def test_unresolved_tradeoff_preserves_two_consequences() -> None:
    policy = _policy("default-evaluator-v1.json")
    decision = decide(
        absent_record(command="audit", reproduce=["reward-lens audit"]),
        policy,
        tradeoff=(("merge", "keeps a useful grader"), ("hold", "avoids an open reach finding")),
    )
    assert isinstance(decision, UnresolvedTradeoff)
    assert decision.state == "unresolved"
    assert len(decision.competing_consequences) == 2


def test_old_policy_digest_still_decides_old_way_after_new_version(tmp_path) -> None:
    original = _policy("default-evaluator-v1.json")
    evolved = tmp_path / "evaluator-v2.json"
    evolved.write_text(
        (
            Path(__file__).parents[2]
            / "src/reward_lens/acceptance/policies/default-evaluator-v1.json"
        )
        .read_text()
        .replace('"version": "1.0.0"', '"version": "2.0.0"')
        .replace('"threshold": "warning"', '"threshold": "error"'),
        encoding="utf-8",
    )
    updated = load_policy(evolved)
    assert original.digest != updated.digest
    assert original.checks[3].threshold == "warning"
    assert updated.checks[3].threshold == "error"
    assert (
        decide(
            absent_record(command="audit", reproduce=["reward-lens audit"]), original
        ).policy_digest
        == original.digest
    )


def test_threshold_version_changes_verdict_for_the_same_canonical_record(tmp_path) -> None:
    """A warning blocks v1 but is below v2's error threshold on the same saved Assay."""
    import json

    from reward_lens import contracts
    from reward_lens.contracts import Assay

    record = absent_record(command="audit", reproduce=["reward-lens audit"])
    absent = record.measurement.exploits[0]
    measured = absent.model_copy(
        update={
            "kind": "check",
            "state": "complete",
            "method": contracts.Method(
                id="exploits.reproducible",
                version="1.0.0",
                params_digest="sha256:" + "3" * 64,
                procedure="repeat the declared exploit",
            ),
            "check": contracts.Check(
                predicate="exploit replay passed", passed=True, scope_tested="test"
            ),
            "result": {"replayed": True},
            "absence": None,
        }
    )
    record.measurement.exploits = [measured]
    record.findings = [
        contracts.Finding(
            id="RGX-local-0001",
            rule="exploits.threshold",
            level="warning",
            kind="open",
            severity_rationale="reproducible warning",
            scope="evaluator_comparison",
            entries=[measured.entry_id],
            partial_fingerprint="12345678",
        )
    ]
    record.holes_from_entries()
    saved = Assay.model_validate(record.to_dict())
    base = {
        "id": "threshold-test",
        "actions": ["merge_this_grader"],
        "default_interval": "0.95",
        "multiplicity": {"rule": "benjamini_hochberg_fdr", "family": "threshold test"},
        "checks": [
            {
                "key": "exploits.reproducible_severity",
                "source": "finding_guard",
                "section": "exploits",
                "threshold": "warning",
            }
        ],
    }
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps({**base, "version": "1.0.0"}), encoding="utf-8")
    new.write_text(
        json.dumps(
            {**base, "version": "2.0.0", "checks": [{**base["checks"][0], "threshold": "error"}]}
        ),
        encoding="utf-8",
    )
    old_policy, new_policy = load_policy(old), load_policy(new)
    assert decide(saved, old_policy).state == "rejected"
    assert decide(saved, new_policy).state == "qualified"
    assert decide(saved, old_policy).policy_digest == old_policy.digest
