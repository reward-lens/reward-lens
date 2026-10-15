"""Reach executes configured OpenEnv through its typed adapter, never its factory as a grader."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from reward_lens.contracts import ProjectConfig
from reward_lens.execution import default_sandbox
from reward_lens.instruments.base import RunContext
from reward_lens.instruments.reach import ReachSurfaces, Surface
from reward_lens.product.audit.run import AuditSubject
from reward_lens.store import Project


class _NoSocketListener:
    """The source-captured route does not need a real network listener for this contract."""

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 0)

    def accept(self):
        raise BlockingIOError

    def close(self) -> None:
        pass


def _openenv_project(tmp_path: Path) -> Project:
    fixture = Path(__file__).parents[1] / "fixtures" / "openenv" / "bridge.py"
    shutil.copy2(fixture, tmp_path / "bridge.py")
    (tmp_path / "rewardlens.yaml").write_text(
        """
reward:
  kind: openenv
  entry: bridge:make_client
  openenv:
    package: openenv
    version: "0.6.0"
    wheel_sha256: dd1033006d17c3c0147a02f73fc6deef881e15dc5501aea28930765708292641
    base_url: ws://127.0.0.1:8000
    connect_timeout_s: 1.0
    message_timeout_s: 1.0
    reset: bridge:reset_from_task
    action: bridge:action_from_task_response
tasks:
  path: tasks.jsonl
  prompt: prompt
outcome: null
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "tasks.jsonl").write_text(
        '{"id":"episode-7","seed":7,"prompt":"2+2"}\n', encoding="utf-8"
    )
    return Project.open(tmp_path)


def test_openenv_reach_runs_adapter_process_probes_without_claiming_response_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zero-argument client factory is not a two-argument reward callable."""
    project = _openenv_project(tmp_path)
    config = ProjectConfig.model_validate(project.config.to_dict())
    assert config.reward.kind == "openenv"
    bridge = tmp_path / "bridge.py"
    subject = AuditSubject(
        grader_path=bridge,
        entrypoint="make_client",
        project_dir=tmp_path,
        tasks_path=tmp_path / "tasks.jsonl",
        responses_path=None,
        trainer=None,
        name="openenv",
        source=bridge.read_text(encoding="utf-8"),
    )
    corpus = type(
        "Corpus",
        (),
        {"tasks": ({"id": "episode-7", "seed": 7, "prompt": "2+2"},)},
    )()
    monkeypatch.setattr("reward_lens.instruments.reach.witness._listener", _NoSocketListener)

    entries = ReachSurfaces().run(
        subject, corpus, RunContext(sandbox=default_sandbox(require_tier="T0"), offline=True)
    )

    assert len(entries) == len(Surface)
    assert all(entry.kind == "estimate" for entry in entries)
    for entry in entries:
        assert entry.result["execution_basis"] == "adapter_process"
        assert entry.result["grader_executed_the_response"] is False
        assert entry.result["attempted"] > 0
        assert entry.result["attempts"]
        assert entry.witness.inputs["source_captured_scope"] == "source_captured_adapter_process"
        assert "response" not in entry.witness.observed.lower()
        assert all(
            "response" not in attempt["procedure"].lower() for attempt in entry.result["attempts"]
        )
    source_entry = next(entry for entry in entries if entry.result["surface"] == "grader_source")
    assert (
        "declared bridge source in the contained adapter process" in source_entry.witness.observed
    )
