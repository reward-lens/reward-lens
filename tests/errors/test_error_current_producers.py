"""Rows for the producer refusals and findings that are live at the resume boundary."""

from __future__ import annotations

from reward_lens.acceptance.types import PolicyDigestMismatch, PolicyInvalid
from reward_lens.contracts import BudgetExceeded, CapabilityUnavailable, UsageError
from reward_lens.errors import CATALOGUE, make
from reward_lens.errors.catalogue import surface_for

SOUNDNESS_TITLES = {
    "RL0221": "a known-wrong solution was accepted, or rejected for a reason other than the expected one",
    "RL0222": "a known-good solution was rejected, which is a defect in the evaluator and not in the task",
    "RL0223": "the grader accepted work the owner's stricter reference verifier rejected",
    "RL0224": "a mutant of the grader survived the protected suite, naming something the suite does not test",
    "RL0225": "the score moved under a transformation of the response that cannot change correctness",
    "RL0226": "an injected real defect did not move the score, so the grader is not reading the response",
    "RL0227": "an exploiting response outscored the best honest response on the same task",
    "RL0228": "the grader source was not available, so mutation and the metamorphic relations did not run",
}

SIGNAL_TITLES = {
    "RL0241": "contrast present",
    "RL0242": "low effective sample size",
    "RL0243": "presentation carries reward",
    "RL0244": "components disagree",
    "RL0245": "TRL zero-std reconciliation differs",
}


def test_soundness_findings_keep_their_producer_titles_and_unresolved_exit() -> None:
    for code, title in SOUNDNESS_TITLES.items():
        spec = CATALOGUE[code]
        assert spec.title == title
        assert spec.cause == title
        assert spec.surface == "finding"
        assert spec.exit_code == 2


def test_soundness_self_check_has_the_error_and_record_finding_surfaces() -> None:
    spec = CATALOGUE["RL0229"]
    assert spec.title == "the panel's own self-check refused a number it was about to report"
    assert spec.exit_code == 4
    assert spec.surface == "error"
    assert spec.verdicts == ("finding",)
    assert surface_for("RL0229", internal=True) == ("error", 4)
    assert surface_for("RL0229", internal=False) == ("verdict", None)


def test_signal_findings_keep_their_declared_descriptions_and_no_invented_context() -> None:
    for code, title in SIGNAL_TITLES.items():
        spec = CATALOGUE[code]
        assert spec.title == title
        assert spec.cause == title
        assert spec.exit_code == 2
        assert spec.surface == "finding"
        assert spec.context_keys == ()


def test_reward_semantics_finding_keeps_declared_and_observed_values() -> None:
    error = make(
        "RL0251", declared={"weights": {"quality": 1.0}}, observed={"weights": {"quality": 0.7}}
    )
    assert error.exit_code == 2
    assert error.context == {
        "declared": {"weights": {"quality": 1.0}},
        "observed": {"weights": {"quality": 0.7}},
    }
    assert "declared" in error.message and "observed" in error.message


def test_applied_weight_identity_finding_keeps_both_scalar_terms() -> None:
    error = make("RL0261", sum_applied_weight=-1.25, negative_loss=-1.0)
    assert error.exit_code == 2
    assert error.context == {"sum_applied_weight": -1.25, "negative_loss": -1.0}
    assert "-1.25" in error.message and "-1.0" in error.message


def test_seeker_budget_refusal_keeps_the_actor_and_field_context() -> None:
    error = make("RL0502", actor="api", field="max_budget_usd")
    assert isinstance(error, BudgetExceeded)
    assert error.exit_code == 6
    assert error.context == {"actor": "api", "field": "max_budget_usd"}
    assert "api" in error.message
    assert "no cap" in error.message


def test_bank_refusals_name_the_missing_bank_and_exact_budget_context() -> None:
    missing = make("RL0134", bank="required")
    assert isinstance(missing, UsageError)
    assert missing.exit_code == 4
    assert missing.context == {"bank": "required"}
    assert "response bank" in missing.message

    budget = make("RL0132", N=4, requested_budget=3)
    assert isinstance(budget, UsageError)
    assert budget.exit_code == 4
    assert budget.context == {"N": 4, "requested_budget": 3}
    assert "N=4" in budget.message and "3" in budget.message


def test_replay_calibration_and_policy_refusals_keep_their_live_contracts() -> None:
    replay = make("RL0131", field="run_dir", path="run")
    assert isinstance(replay, UsageError)
    assert replay.exit_code == 4
    assert replay.context == {"field": "run_dir", "path": "run"}
    assert "closed run record" in replay.message

    calibration = make("RL0133", field="calibration", reading="susceptibility")
    assert isinstance(calibration, UsageError)
    assert calibration.exit_code == 4
    assert calibration.context == {"field": "calibration", "reading": "susceptibility"}
    assert "calibration" in calibration.message

    invalid = PolicyInvalid("policy has an unknown evidence source")
    assert invalid.code == "RL0310"
    assert invalid.exit_code == 4

    mismatch = PolicyDigestMismatch(expected="sha256:expected", recorded="sha256:recorded")
    assert mismatch.code == "RL0311"
    assert mismatch.exit_code == 4
    assert mismatch.context == {"expected": "sha256:expected", "recorded": "sha256:recorded"}


def test_import_and_tracker_refusals_preserve_their_allocated_context() -> None:
    version = make(
        "RL0630", major=99, declared="https://example/schema/99.0", original_bytes=b"received"
    )
    assert isinstance(version, UsageError)
    assert version.exit_code == 4
    assert version.context == {
        "major": 99,
        "declared": "https://example/schema/99.0",
        "original_bytes": b"received",
    }

    bundle = make("RL0631", member="payload-link", rule="symlink")
    assert isinstance(bundle, UsageError)
    assert bundle.exit_code == 4
    assert bundle.context == {"member": "payload-link", "rule": "symlink"}

    signature = make(
        "RL0632", signer="ed25519:fixture", digest="sha256:bad", verification={"valid": False}
    )
    assert isinstance(signature, UsageError)
    assert signature.exit_code == 4
    assert signature.context == {
        "signer": "ed25519:fixture",
        "digest": "sha256:bad",
        "verification": {"valid": False},
    }

    tracker = make("RL0720", tracker="wandb", credential="WANDB_API_KEY")
    assert isinstance(tracker, CapabilityUnavailable)
    assert tracker.exit_code == 5
    assert tracker.context == {"tracker": "wandb", "credential": "WANDB_API_KEY"}


def test_export_refusals_preserve_their_reallocated_context() -> None:
    secret = make(
        "RL0641",
        hits=[{"file": "record.json", "line": 7, "pattern": "credential"}],
        redacted_path="record.redacted.json",
    )
    assert isinstance(secret, UsageError)
    assert secret.exit_code == 4
    assert secret.context == {
        "hits": [{"file": "record.json", "line": 7, "pattern": "credential"}],
        "redacted_path": "record.redacted.json",
    }

    self_reference = make("RL0642", value="report.html")
    assert isinstance(self_reference, UsageError)
    assert self_reference.exit_code == 4
    assert self_reference.context == {"value": "report.html"}


def test_trace_refusals_name_the_series_or_integrity_row() -> None:
    incomplete = make("RL0610", series="reward_recorded")
    assert incomplete.exit_code == 4
    assert incomplete.context == {"series": "reward_recorded"}
    assert "reward_recorded" in incomplete.message

    integrity = make("RL0611", rule="trace.incomplete_group", row="completions_00000.parquet:0")
    assert integrity.exit_code == 4
    assert integrity.context == {
        "rule": "trace.incomplete_group",
        "row": "completions_00000.parquet:0",
    }
    assert "trace.incomplete_group" in integrity.message
    assert "completions_00000.parquet:0" in integrity.message
