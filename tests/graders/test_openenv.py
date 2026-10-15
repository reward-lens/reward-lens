"""C042's configured OpenEnv adapter, exercised through the real local protocol."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from reward_lens.contracts import ProjectConfig, RewardLensError
from reward_lens.contracts.project import project_schema

OPENENV_PROJECT = {
    "reward": {
        "kind": "openenv",
        "entry": "bridge:make_client",
        "openenv": {
            "package": "openenv",
            "version": "0.6.0",
            "wheel_sha256": "dd1033006d17c3c0147a02f73fc6deef881e15dc5501aea28930765708292641",
            "base_url": "ws://127.0.0.1:8000",
            "connect_timeout_s": 10.0,
            "message_timeout_s": 60.0,
            "reset": "bridge:reset_from_task",
            "action": "bridge:action_from_task_response",
        },
    },
    "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
    "outcome": None,
}


def test_openenv_project_contract_round_trips_and_schema_accepts():
    import jsonschema_rs

    validator = jsonschema_rs.validator_for(project_schema())
    config = ProjectConfig.model_validate(OPENENV_PROJECT)
    assert config.reward.kind == "openenv"
    assert config.reward.openenv is not None
    assert config.to_dict()["reward"]["openenv"] == OPENENV_PROJECT["reward"]["openenv"]
    assert validator.is_valid(OPENENV_PROJECT), list(validator.iter_errors(OPENENV_PROJECT))


def test_openenv_contract_rejects_bad_explicit_binding():
    malformed = {**OPENENV_PROJECT, "reward": {**OPENENV_PROJECT["reward"]}}
    malformed["reward"]["openenv"] = {**OPENENV_PROJECT["reward"]["openenv"], "extra": "no"}
    with pytest.raises(RewardLensError) as caught:
        ProjectConfig.model_validate(malformed)
    assert caught.value.code == "RL0003"
    assert "reward.openenv.extra" in str(caught.value)


def test_openenv_adapter_imports_after_its_contract_exists():
    from reward_lens.graders.openenv import OpenEnvGrader

    assert OpenEnvGrader is not None


@pytest.fixture
def bridge(monkeypatch):
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "openenv"
    monkeypatch.syspath_prepend(str(fixture_dir))
    sys.modules.pop("bridge", None)
    return importlib.import_module("bridge")


def _grader(bridge):
    from reward_lens.graders.openenv import OpenEnvGrader

    config = ProjectConfig.model_validate(OPENENV_PROJECT)
    assert config.reward.openenv is not None
    return OpenEnvGrader.from_config(config.reward.openenv, entry=config.reward.entry)


def test_openenv_uses_real_sync_client_step_result_and_typed_action(bridge):
    grader = _grader(bridge)
    envelope = grader.score({"id": "episode-7", "seed": 7, "prompt": "2+2"}, "4")

    client = bridge.LAST_CLIENT
    assert client is not None
    assert client.connected == client.closed == 1
    assert client.reset_calls == [{"seed": 7, "episode_id": "episode-7"}]
    assert [action.response for action in client.step_calls] == ["4"]
    assert envelope.verdict == "scored"
    assert envelope.score == 0.75
    assert envelope.provenance["done"] is True
    assert envelope.provenance["step_count"] == 1
    assert envelope.provenance["action_class"] == "Submit"
    assert envelope.provenance["version"] == "0.6.0"
    assert "response" not in envelope.provenance
    assert envelope.from_dict(envelope.to_dict()) == envelope
    manifest = grader.describe().to_dict()
    assert manifest["family"] == "stateful_tool_environment"
    assert manifest["state_model"] == "reset_then_single_step"
    assert manifest["network_policy"] == "local_endpoint_only"


@pytest.mark.parametrize("reward", [None, True, float("nan"), float("inf")])
def test_openenv_rejects_unscorable_step_rewards(bridge, reward):
    grader = _grader(bridge)
    bridge.make_client(base_url="ws://127.0.0.1:8000", connect_timeout_s=1.0, message_timeout_s=1.0)
    original_factory = grader._client_factory

    def factory(**kwargs):
        client = original_factory(**kwargs)
        client.reward = reward
        client.done = False
        return client

    grader._client_factory = factory
    envelope = grader.score({"id": "episode-7", "seed": 7, "prompt": "2+2"}, "4")
    assert envelope.verdict == "grader_error"
    assert envelope.score is None
    assert envelope.errors


def test_openenv_rejects_raw_dict_action_as_bad_explicit_binding(bridge):
    grader = _grader(bridge)
    grader._action_bridge = lambda task, response: {"response": "raw"}
    with pytest.raises(RewardLensError) as caught:
        grader.score({"id": "episode-7", "seed": 7, "prompt": "2+2"}, "4")
    assert caught.value.code == "RL0003"
    assert caught.value.exit_code == 4
    assert caught.value.context["field"] == "reward.openenv.action"


def test_openenv_missing_bridge_is_rl0003(bridge):
    from reward_lens.graders.openenv import OpenEnvGrader

    raw = {**OPENENV_PROJECT, "reward": {**OPENENV_PROJECT["reward"]}}
    raw["reward"]["openenv"] = {**OPENENV_PROJECT["reward"]["openenv"], "reset": "bridge:missing"}
    config = ProjectConfig.model_validate(raw)
    assert config.reward.openenv is not None
    with pytest.raises(RewardLensError) as caught:
        OpenEnvGrader.from_config(config.reward.openenv, entry=config.reward.entry)
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "reward.openenv.reset"


def test_openenv_missing_optional_package_is_rl0704(bridge, monkeypatch):
    import reward_lens.graders.openenv as adapter

    actual_import = adapter.importlib.import_module

    def no_openenv(name, *args, **kwargs):
        if name.startswith("openenv"):
            raise ModuleNotFoundError(name)
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", no_openenv)
    with pytest.raises(RewardLensError) as caught:
        _grader(bridge).score({"id": "episode-7", "seed": 7, "prompt": "2+2"}, "4")
    assert caught.value.code == "RL0704"
    assert caught.value.exit_code == 5


def test_open_episode_is_retained_without_a_second_step(bridge):
    grader = _grader(bridge)
    original_factory = grader._client_factory

    def factory(**kwargs):
        client = original_factory(**kwargs)
        client.done = False
        return client

    grader._client_factory = factory
    envelope = grader.score({"id": "episode-7", "seed": 7, "prompt": "2+2"}, "4")
    client = bridge.LAST_CLIENT
    assert client is not None
    assert len(client.step_calls) == 1
    assert envelope.verdict == "scored"
    assert envelope.provenance["done"] is False
    assert envelope.limitations == ("one step only; episode remained open",)
