"""The run directory can be discovered and resumed from a fresh CLI process."""

from __future__ import annotations

from pathlib import Path


def test_runs_list_and_show_expose_a_resumable_audit(cli, state_home, tmp_path: Path) -> None:
    from reward_lens.store import RunDir

    run = RunDir.create(project=tmp_path / "project", name="nightly", command="audit")
    environment = {"XDG_STATE_HOME": str(state_home)}
    listed = cli("runs", "list", "--format", "json", env=environment)
    assert listed.exit_code == 0, listed.stderr
    row = next(item for item in listed.json()["runs"] if item["run_id"] == run.run_id)
    assert row["state"] == "running"
    assert row["resume_command"][-2:] == ["--resume", "nightly"]

    shown = cli("runs", "show", run.run_id, "--format", "text", env=environment)
    assert shown.exit_code == 0, shown.stderr
    assert "Continue it with:" in shown.stdout
    assert "--resume nightly" in shown.stdout
