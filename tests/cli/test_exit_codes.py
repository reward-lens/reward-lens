"""D-22's table: nine codes, each produced here by a test that names it."""

from __future__ import annotations

import json
import pathlib

import pytest


@pytest.fixture
def grader(tmp_path) -> pathlib.Path:
    path = tmp_path / "grader.py"
    path.write_text("def score(task, response):\n    return 1.0\n", encoding="utf-8")
    return path


_POLICY = {"id": "policy.exit-codes", "digest": "sha256:" + "0" * 64}


def _record(grader: pathlib.Path, state: str):
    """The honest all-absence record the SDK returns, with one decision state swapped in.

    D-13 makes a settled decision name the policy it was settled under and the action it
    settles, so qualified and rejected carry both; unresolved carries neither.
    """
    from reward_lens import api
    from reward_lens.contracts import Assay

    doc = api.audit(api.AuditRequest(path=grader)).to_dict()
    settled = state in ("qualified", "rejected")
    doc["decision"] = dict(
        doc["decision"],
        state=state,
        action="merge_this_grader" if settled else None,
        policy=_POLICY if settled else None,
    )
    return Assay.model_validate(doc)


def _pin_decision(monkeypatch, grader: pathlib.Path, state: str) -> None:
    """Swap one decision state into the record, built before api.audit is replaced.

    The record has to be made first: _record calls api.audit itself, so a lambda that
    builds it lazily would recurse into the patch.
    """
    from reward_lens import api

    record = _record(grader, state)
    monkeypatch.setattr(api, "audit", lambda request: record)


def test_exit_0_when_the_decision_qualified(invoke, monkeypatch, grader):
    _pin_decision(monkeypatch, grader, "qualified")
    assert invoke("audit", str(grader), "--format", "json").exit_code == 0


def test_exit_1_when_the_decision_was_rejected(invoke, monkeypatch, grader):
    _pin_decision(monkeypatch, grader, "rejected")
    assert invoke("audit", str(grader), "--format", "json").exit_code == 1


def test_exit_2_when_the_decision_is_unresolved(invoke, grader):
    """No monkeypatch: this build's audit is all absence, so unresolved is the honest answer."""
    assert invoke("audit", str(grader), "--format", "json").exit_code == 2


def test_exit_zero_flag_folds_1_and_2_to_0(invoke, grader):
    assert invoke("audit", str(grader), "--format", "json", "--exit-zero").exit_code == 0


def test_exit_3_when_a_decision_is_pending_and_there_is_no_tty(cli, state_home):
    from reward_lens.store import RunDir

    run_id = RunDir.create(project=None, name="doomed").run_id
    refused = cli("runs", "rm", run_id, "--format", "json")
    assert refused.exit_code == 3
    payload = json.loads(refused.stdout)
    assert payload["error"]["code"] == "RL0801"


def test_exit_4_on_an_unknown_format(cli, grader):
    run = cli("audit", str(grader), "--format", "xml")
    assert run.exit_code == 4
    assert "RL0004" in run.stderr
    assert "xml" in run.stderr


def test_exit_5_when_a_needed_extra_is_unavailable(cli, grader):
    run = cli("audit", str(grader), "--format", "sarif")
    assert run.exit_code == 5
    assert "RL0701" in run.stderr


def test_exit_6_when_the_money_cap_cannot_cover_the_run(cli, grader):
    run = cli("audit", str(grader), "--seeker", "api", "--max-budget-usd", "0.00", "--yes")
    assert run.exit_code == 6
    assert "RL0501" in run.stderr


def test_exit_7_on_an_internal_failure_and_no_traceback(invoke, monkeypatch, grader):
    from reward_lens import api

    def explode(request):
        raise ZeroDivisionError("a defect nobody planned for")

    monkeypatch.setattr(api, "audit", explode)
    run = invoke("audit", str(grader))
    assert run.exit_code == 7
    assert "RL0900" in run.stderr
    assert "Traceback (most recent call last)" not in run.stderr


def test_an_internal_failure_names_the_action_and_carries_the_exception(invoke, monkeypatch, grader):
    """RL0900 is the one error whose cause is a defect in reward-lens, so it has to say enough
    to go and find that defect: which verb was running, and what was raised inside it.

    The exception rides in the error's context, which reaches `error.context` in the envelope and
    an indented line in the `help:` block. A traceback is still never printed: the type and the
    message are what a caller can act on, and the run directory keeps the rest.
    """
    from reward_lens import api

    def explode(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "audit", explode)

    run = invoke("audit", str(grader))
    assert run.exit_code == 7
    assert "RL0900" in run.stderr
    assert "failed while running audit" in run.stderr
    assert "RuntimeError: boom" in run.stderr
    assert "<action>" not in run.stderr, "the {action} slot was left unfilled"
    assert "Traceback (most recent call last)" not in run.stderr

    structured = invoke("audit", str(grader), "--format", "json")
    assert structured.exit_code == 7
    error = json.loads(structured.stdout)["error"]
    assert error["code"] == "RL0900"
    assert "failed while running audit" in error["message"]
    assert error["context"]["exception"] == "RuntimeError: boom"


def test_exit_130_on_interrupt(invoke, monkeypatch, grader):
    from reward_lens import api

    def interrupted(request):
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "audit", interrupted)
    run = invoke("audit", str(grader))
    assert run.exit_code == 130
    assert "RL0130" in run.stderr
    assert "Traceback (most recent call last)" not in run.stderr


def test_an_interrupt_a_serializer_wrapped_is_still_an_interrupt(invoke, monkeypatch, grader):
    """D-22: a SIGINT that lands inside someone else's callback comes back wrapped.

    Reproduced against pydantic-core, which catches what a wrap serializer raised and re-raises it
    as `PydanticSerializationError`: an `Exception`, so `except KeyboardInterrupt` misses it and
    the run reported RL0900, a defect in reward-lens, for a key the user pressed. A `RuntimeError`
    with the interrupt as its cause is the same shape without the dependency.
    """
    from reward_lens import api

    def wrapped(request):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt as interrupt:
            raise RuntimeError("Error calling function '_drop_absent_optionals'") from interrupt

    monkeypatch.setattr(api, "audit", wrapped)
    run = invoke("audit", str(grader))
    assert run.exit_code == 130, run.stderr
    assert "RL0130" in run.stderr
    assert "RL0900" not in run.stderr
    assert "Traceback (most recent call last)" not in run.stderr


def test_no_expected_failure_prints_a_traceback(cli, grader, tmp_path):
    missing = tmp_path / "no-such.assay.json"
    for argv in (("audit", str(grader), "--format", "xml"), ("explain", "RL9999"), ("open", str(missing))):
        run = cli(*argv)
        assert "Traceback (most recent call last)" not in run.stderr, argv
        assert run.stderr.startswith("error: "), (argv, run.stderr[:200])


def test_exit_1_through_a_real_record_the_policy_rejected(cli, grader, tmp_path):
    """D-22's 1, with nothing patched: a rejected decision on disk, read back by the install.

    No wave-1 verb settles a decision against a policy on its own, so the record that carries one
    is written here and opened through the installed CLI. That is a real record taking a real
    path through the exit-code table, which is what the gate's demonstration needs.
    """
    path = tmp_path / "rejected.assay.json"
    path.write_text(json.dumps(_record(grader, "rejected").to_dict()), encoding="utf-8")

    run = cli("open", str(path), "--format", "json")

    assert run.exit_code == 1, (run.stdout, run.stderr)
    payload = json.loads(run.stdout)
    assert payload["decision"]["state"] == "rejected"
    assert payload["decision"]["policy"]["id"] == _POLICY["id"]


def test_exit_3_when_a_paid_step_is_pending_and_the_budget_is_not_what_stopped_it(cli, grader):
    """D-31 under `--non-interactive`, reached before D-33's cap rather than behind it.

    The cap here could cover the work, so the money is not the answer and what is left is a
    decision to make. Nothing may be asked for, so the pending action is printed and the run
    exits 3. A cap that could not cover it is exit 6, and that is the test above.
    """
    run = cli(
        "audit", str(grader), "--seeker", "api", "--max-budget-usd", "5.00",
        "--non-interactive", "--format", "json",
    )

    assert run.exit_code == 3, (run.stdout, run.stderr)
    assert "RL0801" in run.stderr
    payload = json.loads(run.stdout)
    assert payload["execution"]["state"] == "pending"
    assert "api seeker" in payload["pending"]["decision"]
    assert payload["pending"]["argv"][-1] == "--yes"
    assert payload["pending"]["command"].endswith("--yes")


def test_exit_7_through_the_documented_self_test_hook(cli, grader):
    """The gate has to demonstrate exit 7 from a command line, and a defect has no invocation."""
    run = cli(
        "audit", str(grader), "--format", "json",
        env={"REWARD_LENS_SELFTEST": "1", "REWARD_LENS_SELFTEST_RAISE": "a defect nobody planned for"},
    )

    assert run.exit_code == 7, (run.stdout, run.stderr)
    assert "RL0900" in run.stderr
    assert "failed while running audit" in run.stderr
    assert "a defect nobody planned for" in run.stderr
    assert "Traceback (most recent call last)" not in run.stderr


def test_the_self_test_hook_does_nothing_without_the_flag_that_arms_it(cli, grader):
    """A stray variable never turns a working install into one that reports a defect in itself."""
    run = cli(
        "audit", str(grader), "--format", "json",
        env={"REWARD_LENS_SELFTEST_RAISE": "a defect nobody planned for"},
    )

    assert run.exit_code != 7, (run.stdout, run.stderr)
    assert "RL0900" not in run.stderr
    assert "REWARD_LENS_SELFTEST=1" in run.stderr
