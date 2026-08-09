"""The requests: strict, closed, frozen, and carrying exactly the defaults the fleet agreed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reward_lens import api, contracts

#: One bad field per request type: what to build, which field is wrong, and the reserved code the
#: SDK owes for it. An identifier gets RL0002, anything else RL0003, and both carry exit 4.
ONE_BAD_FIELD = (
    ("AuditRequest", {"path": ".", "seed": "20260911"}, "seed", "RL0003"),
    ("TraceRequest", {"run": 7}, "run", "RL0002"),
    ("CompareRequest", {"baseline": 1, "candidate": "v2"}, "baseline", "RL0002"),
    ("ForecastIssueRequest", {"claim": 3}, "claim", "RL0003"),
    ("ForecastResolveRequest", {"forecast_id": 9}, "forecast_id", "RL0002"),
    ("ForecastLedgerRequest", {"since": 5}, "since", "RL0003"),
    ("ImproveRequest", {"path": ".", "offline": "no"}, "offline", "RL0003"),
    ("ExportRequest", {"format": 3}, "format", "RL0003"),
    ("DoctorRequest", {"project": 3}, "project", "RL0003"),
)

FROZEN_AUDIT_FIELDS = (
    "path",
    "tasks",
    "responses",
    "outcome",
    "seeker",
    "max_budget_usd",
    "offline",
    "sandbox",
    "require_tier",
    "dry_run",
    "seed",
    "run_dir",
    "name",
    "resume",
    "only",
    "policy",
    "non_interactive",
)


def test_audit_request_carries_exactly_the_frozen_fields() -> None:
    assert tuple(api.AuditRequest.model_fields) == FROZEN_AUDIT_FIELDS


def test_audit_request_carries_the_frozen_defaults(tmp_path) -> None:
    request = api.AuditRequest(path=tmp_path)
    assert (request.tasks, request.responses, request.outcome) == (None, None, None)
    assert request.seeker == "off"
    assert request.max_budget_usd is None
    assert request.offline is True
    assert request.sandbox == "auto"
    assert request.require_tier is None
    assert request.dry_run is False
    assert request.seed == 20260911
    assert (request.run_dir, request.name, request.resume, request.policy) == (None,) * 4
    assert request.only == ()
    assert request.non_interactive is False


def test_a_request_refuses_a_field_it_does_not_know(tmp_path) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.AuditRequest(path=tmp_path, sekeer="api")
    error = caught.value
    assert error.code == "RL0003"
    assert error.exit_code == 4
    assert "sekeer" in error.message
    assert "no field of that name" in error.message


def test_a_request_refuses_a_value_of_the_wrong_type(tmp_path) -> None:
    for field, value in (("offline", "yes"), ("seed", "20260911"), ("seeker", "sometimes")):
        with pytest.raises(contracts.UsageError) as caught:
            api.AuditRequest(path=tmp_path, **{field: value})
        assert caught.value.code == "RL0003"
        assert field in caught.value.message


def test_a_request_is_frozen(tmp_path) -> None:
    request = api.AuditRequest(path=tmp_path)
    with pytest.raises(ValidationError):
        request.offline = False


def test_a_path_field_accepts_the_string_a_terminal_hands_it(tmp_path) -> None:
    request = api.AuditRequest(path=str(tmp_path), tasks=str(tmp_path / "t.jsonl"))
    assert request.path == Path(tmp_path)
    assert request.tasks == tmp_path / "t.jsonl"


def test_money_is_a_string_because_a_float_is_not_money(tmp_path) -> None:
    with pytest.raises(contracts.UsageError):
        api.AuditRequest(path=tmp_path, max_budget_usd=5.0)
    assert api.AuditRequest(path=tmp_path, max_budget_usd="5.00").max_budget_usd == "5.00"


def test_a_plan_estimates_the_seconds_and_the_default_is_zero() -> None:
    """A-016's `estimate_s`: how long the run is expected to take, before it runs.

    Zero is what a plan carries when nothing has estimated it. It is the additive default the
    field landed with, not a promise that the run is instant.
    """
    assert api.Plan(command="audit").estimate_s == 0


def test_the_estimate_in_seconds_round_trips_like_every_other_plan_field() -> None:
    plan = api.Plan(command="audit", panels=("validity",), estimate_s=42)
    assert plan.estimate_s == 42
    assert plan.model_dump()["estimate_s"] == 42
    assert json.loads(plan.model_dump_json())["estimate_s"] == 42
    assert api.Plan.model_validate(plan.model_dump()) == plan


def test_doctor_takes_either_a_request_or_the_keyword(tmp_path) -> None:
    by_request = api.doctor(api.DoctorRequest(project=tmp_path))
    by_keyword = api.doctor(project=tmp_path)
    assert by_request.install == by_keyword.install


# The refusal at the SDK boundary: every request model validates through one path, and what comes
# out of it is the reserved code-4 error the CLI can print and exit on, never a pydantic object.


@pytest.mark.parametrize("name,kwargs,field,code", ONE_BAD_FIELD, ids=[r[0] for r in ONE_BAD_FIELD])
def test_one_bad_field_per_request_type_is_the_reserved_refusal(
    name: str, kwargs: dict, field: str, code: str, tmp_path
) -> None:
    built = {k: (tmp_path if v == "." else v) for k, v in kwargs.items()}
    with pytest.raises(contracts.RewardLensError) as caught:
        getattr(api, name)(**built)
    error = caught.value
    assert isinstance(error, contracts.UsageError)
    assert not isinstance(error, ValidationError)
    assert error.code == code
    assert error.exit_code == 4
    assert field in error.message, f"{name} does not name {field} in {error.message!r}"
    assert error.remediation
    assert error.context["request"] == name
    assert error.context["field"] == field


@pytest.mark.parametrize("name,kwargs,field,code", ONE_BAD_FIELD, ids=[r[0] for r in ONE_BAD_FIELD])
def test_no_pydantic_error_escapes_any_request_type(
    name: str, kwargs: dict, field: str, code: str, tmp_path
) -> None:
    built = {k: (tmp_path if v == "." else v) for k, v in kwargs.items()}
    try:
        getattr(api, name)(**built)
    except ValidationError as escaped:  # pragma: no cover - the failure this test exists for
        pytest.fail(f"{name} let a pydantic ValidationError reach the caller: {escaped}")
    except contracts.RewardLensError:
        pass
    else:  # pragma: no cover - the bad field was accepted
        pytest.fail(f"{name} accepted a bad {field}")


def test_a_missing_identifier_is_rl0002_and_a_missing_setting_is_rl0003() -> None:
    with pytest.raises(contracts.UsageError) as missing_id:
        api.CompareRequest(candidate="v2")
    assert missing_id.value.code == "RL0002"
    assert "baseline" in missing_id.value.message

    with pytest.raises(contracts.UsageError) as missing_path:
        api.ImproveRequest()
    assert missing_path.value.code == "RL0003"
    assert "path" in missing_path.value.message


def test_the_payload_path_refuses_the_same_way_as_the_constructor(tmp_path) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.AuditRequest.model_validate({"path": str(tmp_path), "offline": "yes"})
    assert caught.value.code == "RL0003"
    assert "offline" in caught.value.message
    good = api.AuditRequest.model_validate({"path": str(tmp_path)})
    assert good.path == Path(tmp_path)


def test_the_refusal_carries_the_catalogue_text_and_its_remedy(tmp_path) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.TraceRequest(run=7)
    error = caught.value
    assert error.message.startswith("the value given for run is not a usable identifier")
    assert "reward-lens" in error.remediation
    assert json.loads(error.to_json())["code"] == "RL0002"


# The frozen types: `Capabilities` is the list the interface freezes, and `sandbox` is the probe.


def test_capabilities_is_the_list_the_frozen_line_names_and_a_consumer_can_append_to_it() -> None:
    capabilities = api.doctor()
    assert isinstance(capabilities.capabilities, list)
    before = len(capabilities)
    extra = api.Capability(
        id="seeker", status="pending", reason="not in this build", unlocks=["reward_lens.seeker"]
    )
    capabilities.append(extra)
    assert len(capabilities) == before + 1
    assert capabilities[-1] is extra
    assert list(capabilities)[-1] is extra
    assert capabilities.capabilities[-1] is extra
    assert isinstance(extra.unlocks, list)


def test_the_sandbox_field_holds_the_probe_result_and_the_dump_keeps_every_measurement() -> None:
    """The frozen field is `sandbox: SandboxProbe`: the measurement, carried out of the dump whole."""
    import dataclasses

    from reward_lens.execution import SandboxProbe, probe

    measured = probe()
    assert isinstance(measured, SandboxProbe)
    install = api.doctor().install

    capabilities = api.Capabilities(capabilities=[], sandbox=measured, install=install)
    assert capabilities.sandbox is measured

    dumped = capabilities.model_dump()
    assert dumped["sandbox"] == dataclasses.asdict(measured)
    assert dumped["sandbox"]["tier_held"] == measured.tier_held
    assert (
        json.loads(json.dumps(capabilities.model_dump(mode="json")))["sandbox"] == dumped["sandbox"]
    )

    assert api.Capabilities(install=install).sandbox is None


def test_a_sandbox_runner_is_refused_as_the_sandbox_of_a_capabilities() -> None:
    """A runner has a `tier` and a `run` and is not a measurement: RL0003, naming the field."""

    class Runner:
        tier = "L0"

        def run(self, argv, *, limits, cwd, env, stdin=None):  # pragma: no cover - never called
            raise AssertionError("the wave-1 SDK runs nothing")

    install = api.doctor().install
    for bad in (Runner(), "a string is not a probe"):
        with pytest.raises(contracts.UsageError) as caught:
            api.Capabilities(install=install, sandbox=bad)
        assert caught.value.code == "RL0003"
        assert "sandbox" in caught.value.message
        assert "SandboxProbe" in caught.value.message


def test_the_doctor_verb_reports_the_probe_as_its_sandbox() -> None:
    from reward_lens.execution import SandboxProbe, probe

    reported = api.doctor()
    assert isinstance(reported, api.Capabilities)
    assert isinstance(reported.sandbox, SandboxProbe)
    assert reported.sandbox == probe()
