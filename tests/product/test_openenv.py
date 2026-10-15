"""The released Audit route for a configured local OpenEnv reward."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from reward_lens import api, contracts, errors
from reward_lens.cli.main import cli

OPENENV_EXTENSION = "https://reward-lens.dev/extensions/openenv"


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "openenv-project"
    root.mkdir()
    fixture = Path(__file__).parents[1] / "fixtures" / "openenv" / "bridge.py"
    shutil.copyfile(fixture, root / "bridge.py")
    (root / "tasks.jsonl").write_text(
        '{"id":"episode-7","seed":7,"prompt":"2+2"}\n', encoding="utf-8"
    )
    (root / "responses.jsonl").write_text(
        '{"id":"episode-7","response_id":"response-7","response":"4"}\n', encoding="utf-8"
    )
    (root / "rewardlens.yaml").write_text(
        """reward:
  kind: openenv
  entry: bridge:make_client
  openenv:
    package: openenv
    version: 0.6.0
    wheel_sha256: dd1033006d17c3c0147a02f73fc6deef881e15dc5501aea28930765708292641
    base_url: ws://127.0.0.1:8000
    connect_timeout_s: 1.0
    message_timeout_s: 1.0
    reset: bridge:reset_from_task
    action: bridge:action_from_task_response
tasks:
  path: tasks.jsonl
  prompt: prompt
responses:
  path: responses.jsonl
outcome: null
""",
        encoding="utf-8",
    )
    return root


def test_openenv_project_runs_actual_adapter_through_sdk_and_cli_and_saves_bundle(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)

    record = api.audit(api.AuditRequest(path=project, name="openenv-sdk"))
    route = record.extensions[OPENENV_EXTENSION]
    entry = next(
        item for item in record.measurement.validity if item.entry_id == "validity.openenv_step"
    )
    assert entry.value == 0.75
    assert entry.result["done"] is True
    assert route["config_digest"].startswith("sha256:")
    assert route["adapter_manifest_digest"].startswith("sha256:")
    assert route["server"]["base_url"] == "ws://127.0.0.1:8000"
    assert route["result"]["step_count"] == 1
    selected = route["selected_rollout"]
    assert selected["rollout_id"] == "response-7"
    assert selected["task_digest"].startswith("sha256:")
    assert selected["response_digest"].startswith("sha256:")
    assert selected["request_digest"] == entry.result["selected_rollout"]["request_digest"]
    assert entry.result["adapter"]["reset_digest"].startswith("sha256:")
    assert entry.result["adapter"]["action_class"] == "Submit"
    assert entry.result["adapter"]["observation_class"] == "KnownObservation"

    saved = project / "assays" / "openenv-sdk.assay.json"
    report = project / "assays" / "openenv-sdk.assay.html"
    assert saved.is_file() and report.is_file()
    saved_record = contracts.Assay.model_validate(json.loads(saved.read_text(encoding="utf-8")))
    assert saved_record.assay_id == record.assay_id
    assert saved_record.extensions[OPENENV_EXTENSION]["selected_rollout"] == selected

    result = CliRunner().invoke(
        cli,
        ["audit", str(project), "--name", "openenv-cli", "--format", "json", "--exit-zero"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr
    delivered = json.loads(result.stdout)
    # An unchanged subject with the same method scope reuses its sealed measurement. The later
    # requested name does not create a second alias whose bytes could drift from the first.
    assert delivered["execution"]["reused"]["name"] == "openenv-sdk"
    assert delivered["artifacts"]["assay"] == "assays/openenv-sdk.assay.json"
    assert not (project / "assays" / "openenv-cli.assay.json").exists()
    assert (project / delivered["artifacts"]["report"]).is_file()
    cli_saved = json.loads((project / delivered["artifacts"]["assay"]).read_text(encoding="utf-8"))
    assert cli_saved["extensions"][OPENENV_EXTENSION]["selected_rollout"] == selected


def test_openenv_audit_records_an_explicit_absence_without_a_joined_rollout(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    config = project / "rewardlens.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("responses:\n  path: responses.jsonl\n", ""),
        encoding="utf-8",
    )

    record = api.audit(api.AuditRequest(path=project, name="openenv-no-rollout"))

    entry = next(
        item for item in record.measurement.validity if item.entry_id == "validity.openenv_step"
    )
    assert entry.kind == "absence"
    assert entry.absence.state == "NOT_MEASURED"
    route = record.extensions[OPENENV_EXTENSION]
    assert route["state"] == "absent"
    assert route["missing"]["state"] == "NOT_MEASURED"


def test_openenv_audit_records_rl0704_when_the_optional_protocol_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    from reward_lens.graders import openenv

    project = _project(tmp_path)

    def unavailable():
        raise errors.make("RL0704", extra="openenv", capability="stateful_tool_environment")

    monkeypatch.setattr(openenv, "_protocol", unavailable)
    record = api.audit(api.AuditRequest(path=project, name="openenv-rl0704"))

    entry = next(
        item for item in record.measurement.validity if item.entry_id == "validity.openenv_step"
    )
    assert entry.kind == "absence"
    assert entry.absence.state == "REFUSED"
    route = record.extensions[OPENENV_EXTENSION]
    assert route["missing"]["code"] == "RL0704"


def test_openenv_audit_retains_selected_rollout_when_the_adapter_is_unscored(
    tmp_path: Path, monkeypatch
) -> None:
    from reward_lens.graders.openenv import OpenEnvGrader

    project = _project(tmp_path)

    def unscored(self, task, response, **_kwargs):
        return SimpleNamespace(
            verdict="grader_error",
            score=None,
            errors=("fixture refused to score",),
            request_digest=contracts.digest({"task": task, "response": response}),
            provenance={
                "reset_digest": "sha256:" + "1" * 64,
                "action_class": "Submit",
                "returned_observation_class": "KnownObservation",
            },
            to_dict=lambda: {"fixture": "unscored"},
        )

    monkeypatch.setattr(OpenEnvGrader, "score", unscored)
    record = api.audit(api.AuditRequest(path=project, name="openenv-unscored"))

    entry = next(
        item for item in record.measurement.validity if item.entry_id == "validity.openenv_step"
    )
    assert entry.kind == "absence"
    assert entry.absence.state == "COULD_NOT_CHECK"
    route = record.extensions[OPENENV_EXTENSION]
    assert route["selected_rollout"]["rollout_id"] == "response-7"
    assert route["attempt"]["request_digest"] == route["selected_rollout"]["request_digest"]
    assert "response" not in route["selected_rollout"]
