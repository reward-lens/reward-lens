"""Conformance for the local, protocol-faithful LLM-judge adapter fixture.

No test in this module contacts a provider.  The fixture's only purpose is to exercise the
wire protocol, and every emitted record labels that fact explicitly.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from reward_lens.contracts import Counters
from reward_lens.execution import Limits
from reward_lens.execution.supervisor import Result
from reward_lens.graders.base import AdapterCapabilityUnproven
from reward_lens.graders.judge import (
    JudgeAdapter,
    JudgeBattery,
    JudgeIdentity,
    ProviderReply,
    SandboxJSONTransport,
)


class FixtureJudgeTransport:
    """A local protocol fixture, deliberately not a named commercial provider."""

    def __init__(self, replies: list[object] | None = None) -> None:
        self.replies = list(replies or [{"verdict": "left", "score": 0.75}])
        self.requests: list[dict[str, Any]] = []

    def __call__(self, prompt: str, completion: str, **kw: object) -> object:
        self.requests.append({"prompt": prompt, "completion": completion, "kw": kw})
        return self.replies.pop(0)


class FakeSandbox:
    tier = "fake-provider-sandbox"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def run(self, argv, *, limits, cwd, env, stdin=None, **kw):
        self.calls.append({"argv": argv, "limits": limits, "cwd": cwd, "env": env, "stdin": stdin})
        return Result(
            exit_code=0,
            stdout=json.dumps(self.payload).encode(),
            stderr=b"",
            counters=Counters(api_calls=1, input_tokens=7, output_tokens=3),
            breach=None,
            tier=self.tier,
        )


def adapter(*, replies: list[object] | None = None, fixture: bool = True) -> JudgeAdapter:
    return JudgeAdapter.from_transport(
        provider="local-protocol-fixture",
        model="fixture-judge-v1",
        prompt="Choose the more correct response and return JSON.",
        settings={"temperature": 0, "max_tokens": 32},
        transport=FixtureJudgeTransport(replies),
        fixture_conformance=fixture,
        passing_probes={"fixture_conformance": fixture},
    )


def pair_task() -> dict[str, str]:
    return {"id": "judge-1", "prompt": "What is two plus two?"}


def test_identity_and_raw_exchange_are_digest_references_in_the_record() -> None:
    result = adapter().score_pair(pair_task(), "four", "five", sandbox=None, limits=Limits())
    identity = result.observations["judge_identity"]
    assert JudgeIdentity.from_dict(identity).provider == "local-protocol-fixture"
    assert identity["model"] == "fixture-judge-v1"
    assert identity["prompt_digest"].startswith("sha256:")
    assert identity["settings_digest"].startswith("sha256:")
    exchange = result.observations["raw_exchange"]
    assert set(exchange) == {"request_digest", "response_digest"}
    assert all(value.startswith("sha256:") for value in exchange.values())
    assert {item["kind"] for item in result.artifacts} == {"raw_request", "raw_response"}
    assert result.observations["mode"] == "fixture_conformance"
    assert "fixture" in result.limitations[0].lower()


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("not-json", "parse_error"),
        (RuntimeError("fixture unavailable"), "provider_unavailable"),
    ],
)
def test_parse_and_provider_failures_are_verdicts_never_scores(reply: object, verdict: str) -> None:
    transport = FixtureJudgeTransport([reply])
    value = JudgeAdapter.from_transport(
        provider="local-protocol-fixture",
        model="fixture-judge-v1",
        prompt="json",
        settings={},
        transport=transport,
        fixture_conformance=True,
        passing_probes={"fixture_conformance": True},
    ).score_pair(pair_task(), "four", "five", sandbox=None, limits=Limits())
    assert value.verdict == verdict
    assert value.score is None
    if verdict == "provider_unavailable":
        refusal = value.observations["refusal"]
        assert refusal["code"] == "RL0711"
        assert refusal["exit_code"] == 5
        assert set(refusal) >= {"provider", "operation", "detail"}


def test_battery_reports_actual_repeat_order_bias_and_independent_measures() -> None:
    report = JudgeBattery(
        adapter=replies_adapter(
            [
                {"verdict": "left", "score": 0.7},
                {"verdict": "left", "score": 0.7},
                {"verdict": "right", "score": 0.2},
                {"verdict": "left", "score": 0.8},
                {"verdict": "left", "score": 0.4},
                {"verdict": "right", "score": 0.3},
                {"verdict": "left", "score": 0.9},
            ]
        )
    ).run(pair_task(), "four", "five", independent_check="left", sandbox=None, limits=Limits())
    measures = {entry.metric: entry for entry in report.measures}
    assert set(measures) == {
        "repeat_agreement",
        "order_effect",
        "position_bias",
        "length_effect",
        "self_preference_effect",
        "style_effect",
        "independent_check_disagreement",
    }
    assert measures["repeat_agreement"].value == 1.0
    assert measures["order_effect"].value == pytest.approx(0.5)
    assert measures["independent_check_disagreement"].value == 0.0
    assert all(
        entry.scope == "fixture_conformance" and entry.comparison for entry in report.measures
    )
    swapped = measures["order_effect"]
    assert swapped.envelope.observations["pair_order"] == ["right", "left"]


def replies_adapter(replies: list[object]) -> JudgeAdapter:
    return adapter(replies=replies)


def test_every_bias_control_is_its_own_entry() -> None:
    report = JudgeBattery(adapter=replies_adapter([{"verdict": "left", "score": 0.7}] * 12)).run(
        pair_task(), "four", "five", independent_check="left", sandbox=None, limits=Limits()
    )
    assert {entry.name for entry in report.entries} >= {
        "position",
        "length",
        "self_preference",
        "style",
    }


def test_independent_check_disagreement_is_reported_not_reconciled() -> None:
    report = JudgeBattery(adapter=replies_adapter([{"verdict": "left", "score": 0.7}] * 12)).run(
        pair_task(), "four", "five", independent_check="right", sandbox=None, limits=Limits()
    )
    assert report.independent_check == "right"
    assert report.disagreement is True
    assert report.final_verdict == "left"


def test_manifest_claims_only_capabilities_with_a_passing_probe() -> None:
    with pytest.raises(AdapterCapabilityUnproven):
        JudgeAdapter.from_transport(
            provider="local-protocol-fixture",
            model="fixture-judge-v1",
            prompt="json",
            settings={},
            transport=FixtureJudgeTransport(),
            fixture_conformance=False,
            passing_probes={"live_provider": False},
            claimed_capabilities={"live_provider": True},
        )


def test_fixture_conformance_is_not_presented_as_a_commercial_measurement() -> None:
    result = adapter().score_pair(pair_task(), "four", "five", sandbox=None, limits=Limits())
    assert result.observations["mode"] == "fixture_conformance"
    assert result.observations["judge_identity"]["provider"] == "local-protocol-fixture"


def test_retained_raw_exchange_is_immutable_after_caller_mutation() -> None:
    response = {"verdict": "left", "score": 0.75, "nested": {"values": [1]}}
    task = {**pair_task(), "nested": {"values": [2]}}
    judge = adapter(replies=[response])
    value = judge.score_pair(task, "four", "five", sandbox=None, limits=Limits())
    request_ref = value.observations["raw_exchange"]["request_digest"]
    response_ref = value.observations["raw_exchange"]["response_digest"]
    response["nested"]["values"].append(99)
    task["nested"]["values"].append(88)
    # The bytes are a tamper-evident snapshot, not the mutable object the caller still owns.
    retained_request = json.loads(judge.raw_artifact(request_ref).decode())["judge_raw"]
    retained_response = json.loads(judge.raw_artifact(response_ref).decode())["judge_raw"]
    assert retained_request["task"]["nested"]["values"] == [2]
    assert retained_response["nested"]["values"] == [1]


def test_sandbox_provider_transport_receives_limits_and_records_costs(tmp_path) -> None:
    sandbox = FakeSandbox({"verdict": "left", "score": 0.75})
    provider = SandboxJSONTransport(("fake-provider-client",), cwd=tmp_path)
    live = JudgeAdapter.from_transport(
        provider="fake-provider",
        model="fake-model",
        prompt="json",
        settings={},
        transport=provider,
        fixture_conformance=False,
        passing_probes={"live_provider": True},
    )
    result = live.score_pair(
        pair_task(),
        "four",
        "five",
        sandbox=sandbox,
        limits=Limits(network=True),
        live_key="authorized",
        budget_usd=1,
    )
    assert result.verdict == "scored"
    assert sandbox.calls[0]["limits"].network is True
    assert result.counters.api_calls == 1
    assert result.observations["mode"] == "sandbox_provider_transport"
    assert result.observations["provider_attempts"][0]["tier"] == sandbox.tier
    assert result.observations["provider_attempts"][0]["execution"] == "sandbox.run"
    assert result.observations["provider_attempts"][0]["limits"]["network"] is True


def test_callback_spoof_cannot_upgrade_to_provider_scope() -> None:
    spoof = FixtureJudgeTransport(
        [
            # A callback can manufacture a provider-shaped reply, but cannot manufacture its scope.
            ProviderReply(
                raw={"verdict": "left", "score": 0.75},
                transport_scope="sandbox_provider_transport",
                attempts=({"execution": "sandbox.run", "limits": {"network": True}},),
            )
        ]
    )
    live = JudgeAdapter.from_transport(
        provider="spoofed-provider",
        model="spoofed-model",
        prompt="json",
        settings={},
        transport=spoof,
        fixture_conformance=False,
        passing_probes={"live_provider": True},
    )
    result = live.score_pair(
        pair_task(),
        "four",
        "five",
        sandbox=FakeSandbox({"verdict": "left", "score": 0.75}),
        limits=Limits(network=True),
        live_key="authorized",
        budget_usd=1,
    )
    assert result.verdict == "provider_unavailable"
    assert result.score is None
    assert result.errors == ("RL0711",)
    assert result.observations["mode"] == "provider_refused_unexecuted"
    assert spoof.requests == []


def test_local_missing_live_prerequisites_are_a_non_skipped_rl0711_refusal() -> None:
    live = adapter(fixture=False)
    result = live.score_pair(
        pair_task(), "four", "five", sandbox=None, limits=Limits(network=False)
    )
    assert result.verdict == "provider_unavailable"
    assert result.score is None
    assert result.errors == ("RL0711",)


_LIVE_AUTHORIZED = (
    os.getenv("REWARD_LENS_JUDGE_LIVE_OPT_IN") == "1"
    and bool(os.getenv("REWARD_LENS_JUDGE_LIVE_KEY"))
    and float(os.getenv("REWARD_LENS_JUDGE_LIVE_BUDGET_USD", "0")) > 0
)


@pytest.mark.skipif(
    not _LIVE_AUTHORIZED,
    reason="LIVE_JUDGE_REQUIRES_EXPLICIT_AUTHORIZED_KEY_AND_BUDGET",
)
def test_live_route_requires_key_and_budget() -> None:
    assert _LIVE_AUTHORIZED
