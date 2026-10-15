"""Local, real-CLI checks for the D-56 composite action.

The test invokes the composite shell with GitHub's input environment recreated
from its declared inputs. It does not substitute a Python helper for either
the Audit or Export CLI: a fresh local wheel supplies the executable.
"""

from __future__ import annotations

import json
import os
import subprocess
import venv
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / "action.yml"
CODEQL_SOURCE = ROOT / "tests" / "ci" / "fixtures" / "github-codeql-upload-sarif-v4.action.yml"
PRE_COMMIT_HOOKS = ROOT / ".pre-commit-hooks.yaml"


def _action() -> dict:
    assert ACTION.is_file(), "P-CI must ship the root composite action.yml"
    loaded = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _ci_step(action: dict) -> dict:
    steps = action["runs"]["steps"]
    return next(step for step in steps if step.get("id") == "audit")


def _wheel_cli(tmp_path: Path) -> tuple[Path, Path]:
    """Build and install this checkout without resolving from the network."""
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(wheelhouse), str(ROOT)],
        check=True,
        text=True,
        capture_output=True,
    )
    wheel = next(wheelhouse.glob("reward_lens-3.1.0-*.whl"))
    environment = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--offline", "--python", str(python), str(wheel)],
        check=True,
        text=True,
        capture_output=True,
    )
    return python, environment / "bin" / "reward-lens"


def _fixture(
    path: Path,
    *,
    planted: bool,
    policy_name: str = "default-ci-v1.json",
) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    malformed = "return None" if planted else "return 0.0"
    scorer = (
        "def score(task, response):\n"
        "    if response == '' or response.startswith('```') or len(response) > 100000:\n"
        f"        {malformed}\n"
        "    return 1.0\n"
    )
    (path / "grader.py").write_text(scorer, encoding="utf-8")
    (path / "tasks.jsonl").write_text('{"id":"t1","prompt":"answer"}\n', encoding="utf-8")
    (path / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome: null\n",
        encoding="utf-8",
    )
    policy = ROOT / "src" / "reward_lens" / "acceptance" / "policies" / policy_name
    (path / "policy.json").write_bytes(policy.read_bytes())
    return path


def _run_action(
    action: dict,
    *,
    fixture: Path,
    cli: Path,
    event_name: str = "pull_request",
    sarif_path: str = "artifacts/ci-audit.sarif",
) -> subprocess.CompletedProcess[str]:
    step = _ci_step(action)
    summary = fixture / "summary.md"
    outputs = fixture / "outputs.txt"
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_WORKSPACE": str(fixture),
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_OUTPUT": str(outputs),
            "INPUT_SUBJECT": ".",
            "INPUT_POLICY": "policy.json",
            "INPUT_AUDIT_NAME": "ci-audit",
            "INPUT_RECORD_PATH": "assays/ci-audit.assay.json",
            "INPUT_SARIF_PATH": sarif_path,
            "INPUT_COMMAND": str(cli),
            "INPUT_UPLOAD_SARIF": "false",
            "GITHUB_EVENT_NAME": event_name,
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=fixture,
        env=env,
        text=True,
        capture_output=True,
    )
    result.summary_path = summary  # type: ignore[attr-defined]
    result.outputs_path = outputs  # type: ignore[attr-defined]
    return result


def test_action_is_composite_and_uses_verified_node24_upload_action() -> None:
    action = _action()
    assert action["runs"]["using"] == "composite"
    assert action["inputs"]["command"]["default"] == "reward-lens"
    assert 'uvx "reward-lens==3.1.0"' in _ci_step(action)["run"]
    assert 'GITHUB_EVENT_NAME:-}" == "pull_request_target"' in _ci_step(action)["run"]
    upload = next(step for step in action["runs"]["steps"] if "uses" in step)
    assert upload["uses"] == "github/codeql-action/upload-sarif@v4"
    assert upload["continue-on-error"] is True
    source = yaml.safe_load(CODEQL_SOURCE.read_text(encoding="utf-8"))
    assert source["runs"]["using"] == "node24"
    assert "node20" not in ACTION.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("planted, expected_exit", [(True, 1), (False, 0)])
def test_composite_runs_real_cli_policy_sarif_summary_and_reopened_record(
    tmp_path: Path, planted: bool, expected_exit: int
) -> None:
    action = _action()
    installed_python, cli = _wheel_cli(tmp_path)
    fixture = _fixture(tmp_path / ("planted" if planted else "repaired"), planted=planted)
    result = _run_action(action, fixture=fixture, cli=cli)
    assert result.returncode == expected_exit, result.stderr
    record = fixture / "assays" / "ci-audit.assay.json"
    sarif = fixture / "artifacts" / "ci-audit.sarif"
    assert record.is_file() and sarif.is_file()
    reopened = subprocess.run(
        [
            str(installed_python),
            "-c",
            "from reward_lens import api; import sys; print(api.open_record(sys.argv[1]).assay_id)",
            str(record),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert reopened.stdout.strip()
    document = json.loads(record.read_text(encoding="utf-8"))
    sarif_document = json.loads(sarif.read_text(encoding="utf-8"))
    levels = [row["level"] for row in sarif_document["runs"][0]["results"]]
    summary = result.summary_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert len(summary.encode("utf-8")) < 1024 * 1024
    assert document["decision"] is not None
    assert "| policy decision |" in summary
    if planted:
        assert document["decision"]["state"] == "rejected"
        assert "error" in levels
        assert "::error file=" in result.stdout
        assert "blocked=true" in result.outputs_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    else:
        assert document["decision"]["state"] == "qualified"
        assert "error" not in levels
        assert "blocked=false" in result.outputs_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]


def test_composite_fails_on_real_unresolved_persisted_policy_decision(tmp_path: Path) -> None:
    """An absent default-evaluator requirement is unresolved, never a CI success."""
    action = _action()
    installed_python, cli = _wheel_cli(tmp_path)
    fixture = _fixture(
        tmp_path / "unresolved",
        planted=False,
    )
    # The registered default CI producer stays canonical, but its adapter cannot import this
    # subject. Audit retains `validity.input_handling` as an explicit absence, so the persisted
    # policy state is genuinely unresolved rather than a synthetic policy key.
    grader = fixture / "grader.py"
    grader.write_text(
        "import reward_lens_fixture_missing_dependency\n" + grader.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = _run_action(action, fixture=fixture, cli=cli)
    assert result.returncode == 2, result.stdout + result.stderr
    record = fixture / "assays" / "ci-audit.assay.json"
    sarif = fixture / "artifacts" / "ci-audit.sarif"
    assert record.is_file() and sarif.is_file()
    reopened = subprocess.run(
        [
            str(installed_python),
            "-c",
            "from reward_lens import api; import sys; print(api.open_record(sys.argv[1]).assay_id)",
            str(record),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert reopened.stdout.strip()
    document = json.loads(record.read_text(encoding="utf-8"))
    summary = result.summary_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    outputs = result.outputs_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    assert document["decision"]["state"] == "unresolved"
    assert "| policy decision | unresolved |" in summary
    assert "| reason | policy unresolved |" in summary
    assert "| SARIF export exit |" in summary
    assert "blocked=true" in outputs


def test_fixture_refuses_the_two_d56_unsafe_controls() -> None:
    action = _action()
    source = ACTION.read_text(encoding="utf-8")
    assert "MAX_SUMMARY_BYTES=1048576" in _ci_step(action)["run"]
    assert "continue-on-error" in source
    assert 'GITHUB_EVENT_NAME:-}" == "pull_request_target"' in _ci_step(action)["run"]


def test_hook_manifest_is_the_fast_four_check_subset() -> None:
    hooks = yaml.safe_load(PRE_COMMIT_HOOKS.read_text(encoding="utf-8"))
    assert hooks == [
        {
            "id": "reward-lens-static-validity",
            "name": "reward-lens static validity subset",
            "description": (
                "Run reward-lens' four fast validity checks only. This hook does not run a full "
                "Audit and does not create an Assay record."
            ),
            "entry": ".github/actions/reward-lens-static-validity/pre_commit.py",
            "language": "python",
            "additional_dependencies": ["."],
            "pass_filenames": False,
            "require_serial": True,
        }
    ]


def test_source_check_rejects_a_node20_third_party_action() -> None:
    clean = yaml.safe_load(CODEQL_SOURCE.read_text(encoding="utf-8"))
    node20 = yaml.safe_load(CODEQL_SOURCE.read_text(encoding="utf-8").replace("node24", "node20"))
    assert clean["runs"]["using"] == "node24"
    with pytest.raises(AssertionError):
        assert node20["runs"]["using"] == "node24"


def test_composite_refuses_pull_request_target_before_it_reads_workspace(tmp_path: Path) -> None:
    action = _action()
    result = _run_action(
        action,
        fixture=tmp_path,
        cli=Path("/bin/false"),
        event_name="pull_request_target",
    )
    assert result.returncode == 4
    assert "unsafe workflow trigger" in result.stdout
    assert not (tmp_path / "assays").exists()


def test_composite_fails_when_the_real_sarif_export_cannot_write(tmp_path: Path) -> None:
    action = _action()
    _installed_python, cli = _wheel_cli(tmp_path)
    fixture = _fixture(tmp_path / "sarif-write-failure", planted=False)
    result = _run_action(action, fixture=fixture, cli=cli, sarif_path=".")
    assert result.returncode != 0
    assert (fixture / "assays" / "ci-audit.assay.json").is_file()
    assert not result.outputs_path.exists()  # type: ignore[attr-defined]


def test_distributed_pre_commit_hook_runs_four_validity_checks_in_a_disposable_git_repo(
    tmp_path: Path,
) -> None:
    """A disposable Git commit invokes the installed hook script and its real four checks."""
    installed_python, _cli = _wheel_cli(tmp_path)
    fixture = _fixture(tmp_path / "hook-repaired", planted=False)
    hook = fixture / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        f"exec {installed_python} {ROOT / '.github/actions/reward-lens-static-validity/pre_commit.py'}\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    subprocess.run(["git", "config", "user.email", "ci@example.test"], cwd=fixture, check=True)
    subprocess.run(["git", "config", "user.name", "Reward Lens CI"], cwd=fixture, check=True)
    subprocess.run(
        ["git", "add", "grader.py", "tasks.jsonl", "rewardlens.yaml", "policy.json"],
        cwd=fixture,
        check=True,
    )
    result = subprocess.run(
        ["git", "commit", "-m", "exercise reward-lens static validity hook"],
        cwd=fixture,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    for check_id in (
        "validity.writable_then_read",
        "validity.input_leakage",
        "validity.unreachable_score_branch",
        "validity.input_handling",
    ):
        assert check_id in output
    assert not (fixture / "assays").exists()
