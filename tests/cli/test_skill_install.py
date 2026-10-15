"""The portable reward-lens skill is installed through the public init command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def invoke(monkeypatch, tmp_path):
    from reward_lens.cli.main import cli

    def run(*args: str):
        monkeypatch.chdir(tmp_path)
        return CliRunner().invoke(cli, list(args), catch_exceptions=False)

    return run


@pytest.mark.parametrize(
    ("agent", "marker"),
    (("claude", ".claude"), ("codex", ".agents"), ("copilot", ".github")),
)
def test_init_skill_explicit_target_writes_and_reads_back(invoke, tmp_path, agent, marker):
    project = tmp_path / agent
    project.mkdir()

    result = invoke("init", str(project), "--skill", "--agent", agent, "--format", "text")

    written = project / marker / "skills" / "reward-lens" / "SKILL.md"
    assert result.exit_code == 0, result.output
    assert written.is_file()
    assert "reward-lens skill installed for " + agent in result.output
    assert str(written) in result.output
    assert written.read_text(encoding="utf-8") == (
        (REPO / "skills/reward-lens/SKILL.md").read_text(encoding="utf-8")
    )


def test_init_skill_detects_one_existing_agent_marker(invoke, tmp_path):
    project = tmp_path / "detected"
    (project / ".agents").mkdir(parents=True)

    result = invoke("init", str(project), "--skill", "--format", "text")

    assert result.exit_code == 0, result.output
    assert "reward-lens skill installed for codex" in result.output
    assert (project / ".agents" / "skills" / "reward-lens" / "SKILL.md").is_file()


def test_init_skill_refuses_unknown_target(invoke, tmp_path):
    project = tmp_path / "unknown"
    project.mkdir()

    result = invoke("init", str(project), "--skill")

    assert result.exit_code == 4
    assert "RL0001" in result.output
    assert "cannot detect an agent" in result.output


def test_init_skill_refuses_unsafe_symlinked_marker(invoke, tmp_path):
    project = tmp_path / "unsafe"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / ".claude").symlink_to(outside, target_is_directory=True)

    result = invoke("init", str(project), "--skill")

    assert result.exit_code == 4
    assert "RL0001" in result.output
    assert "unsafe skill destination" in result.output
    assert not (outside / "skills").exists()


def test_init_skill_refuses_a_project_reached_through_a_symlink(invoke, tmp_path):
    outside = tmp_path / "outside"
    project = outside / "project"
    alias = tmp_path / "alias"
    project.mkdir(parents=True)
    alias.symlink_to(outside, target_is_directory=True)

    result = invoke("init", str(alias / "project"), "--skill", "--agent", "codex")

    assert result.exit_code == 4
    assert "unsafe skill destination" in result.output
    assert not (project / ".agents").exists()


def test_init_skill_refuses_an_existing_skill_conflict(invoke, tmp_path):
    project = tmp_path / "conflict"
    installed = project / ".github" / "skills" / "reward-lens"
    installed.mkdir(parents=True)
    sentinel = installed / "SKILL.md"
    sentinel.write_text("do not replace\n", encoding="utf-8")

    result = invoke("init", str(project), "--skill")

    assert result.exit_code == 4
    assert "RL0001" in result.output
    assert "already exists" in result.output
    assert sentinel.read_text(encoding="utf-8") == "do not replace\n"


def test_init_example_route_is_unchanged_when_skill_is_not_requested(invoke, tmp_path):
    project = tmp_path / "example"

    result = invoke("init", str(project), "--example", "code-reward")

    assert result.exit_code == 0, result.output
    assert (project / "rewardlens.yaml").is_file()


def test_skill_has_portable_fields_triggers_and_side_effect_guard():
    skill = (REPO / "skills/reward-lens/SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, _ = skill.split("---\n", 2)

    for field in (
        "name:",
        "description:",
        "license:",
        "compatibility:",
        "metadata:",
        "allowed-tools:",
    ):
        assert field in frontmatter
    assert "name: reward-lens" in frontmatter
    assert "disable-model-invocation: true" in frontmatter
    assert len(frontmatter.split("description:", 1)[1].split("\n", 1)[0].strip()) < 1024
    sentence = (
        "when you write or change a reward function, rubric, grader or environment, run "
        "`reward-lens audit`, paste the exploit table into your reply, and do not hand back "
        "a grader with an open reach finding."
    )
    assert sentence in skill
    assert "seeker.propose" in skill
