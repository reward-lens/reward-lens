"""Offline tracker exports, including the registry seam P-EXPORT consumes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from reward_lens.api._record import absent_record


def _summary(tmp_path: Path):
    from reward_lens.integrations.trackers import RecordSummary

    return RecordSummary.from_record(
        absent_record(
            command="tracker-test",
            reproduce=["reward-lens audit ."],
            path=tmp_path / "source.jsonl",
        )
    )


def test_file_backend_persists_summary_and_report_link_and_reads_it_back(tmp_path: Path) -> None:
    from reward_lens.integrations.trackers import export_to

    report = tmp_path / "report.html"
    report.write_text("<html>report</html>", encoding="utf-8")
    result = export_to("file", summary=_summary(tmp_path), report_url=report.as_uri())

    assert result.backend == "file"
    assert result.path is not None and result.path.is_file()
    persisted = json.loads(result.path.read_text(encoding="utf-8"))
    assert persisted["summary"]["assay_id"] == _summary(tmp_path).assay_id
    assert persisted["report_url"] == report.as_uri()
    assert result.readback == persisted


@pytest.mark.parametrize(
    "backend, directory, expected_file",
    [
        ("wandb", "REWARD_LENS_WANDB_DIR", "reward-lens-summary.json"),
        ("mlflow", "REWARD_LENS_MLFLOW_DIR", "summary.json"),
    ],
)
def test_optional_backends_write_offline_target_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    directory: str,
    expected_file: str,
) -> None:
    from reward_lens.integrations.trackers import export_to

    root = tmp_path / backend
    monkeypatch.setenv(directory, str(root))
    report = tmp_path / "report.html"
    report.write_text("<html>report</html>", encoding="utf-8")
    summary = _summary(tmp_path)
    result = export_to(backend, summary=summary, report_url=report.as_uri())

    assert result.path is not None and result.path.name == expected_file
    assert result.path.is_file()
    assert result.readback["summary"]["assay_id"] == summary.assay_id
    assert result.readback["report_url"] == report.as_uri()
    if backend == "wandb":
        assert list(root.rglob("*.wandb")), "the real W&B offline run was not persisted"
    else:
        assert "artifacts/reward-lens/summary.json" in result.path.as_posix()


def test_missing_optional_tracker_capability_is_named_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reward_lens.contracts import CapabilityUnavailable
    from reward_lens.integrations.trackers import export_to
    from reward_lens.integrations.trackers.wandb import WandbTracker

    monkeypatch.setattr(WandbTracker, "_sdk", lambda self: None)
    with pytest.raises(CapabilityUnavailable) as raised:
        export_to(
            "wandb", summary=_summary(tmp_path), report_url=(tmp_path / "report.html").as_uri()
        )

    assert raised.value.code == "RL0720"
    assert raised.value.exit_code == 5
    assert raised.value.context == {"tracker": "wandb", "credential": "reward-lens[trackers]"}
    assert "reward-lens[trackers]" in raised.value.message


def test_cli_export_to_file_reopens_saved_audit_and_returns_tracker_readback(
    tmp_path: Path,
) -> None:
    """Public route: saved audit -> CLI export -> registry -> file write -> receipt/readback."""
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.examples.code_reward import write as write_example

    project = tmp_path / "project"
    write_example(project)
    record = api.audit(api.AuditRequest(path=project, name="tracker-public-source"))
    saved = next((project / "assays").glob("*.assay.json"))
    destination = tmp_path / "bundle"

    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(destination),
            "--kind",
            "bundle",
            "--to",
            "file",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    tracker = receipt["artifacts"]["tracker"]
    assert tracker["backend"] == "file"
    assert tracker["report_url"] == (destination / "report.html").resolve().as_uri()
    persisted = Path(tracker["path"])
    assert persisted.is_file()
    assert json.loads(persisted.read_text(encoding="utf-8")) == tracker["readback"]
    assert tracker["readback"]["summary"]["assay_id"] == record.assay_id


def test_cli_export_to_missing_optional_tracker_refuses_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from reward_lens import api
    from reward_lens.cli.main import cli
    from reward_lens.examples.code_reward import write as write_example
    from reward_lens.integrations.trackers.wandb import WandbTracker

    monkeypatch.setattr(WandbTracker, "_sdk", lambda self: None)
    project = tmp_path / "project"
    write_example(project)
    api.audit(api.AuditRequest(path=project, name="tracker-refusal-source"))
    saved = next((project / "assays").glob("*.assay.json"))
    result = CliRunner().invoke(
        cli,
        [
            "export",
            "--record",
            str(saved),
            "--dest",
            str(tmp_path / "bundle"),
            "--kind",
            "bundle",
            "--to",
            "wandb",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 5
    assert "RL0720" in result.output
    assert "wandb" in result.output
    assert "reward-lens[trackers]" in result.output


def test_registry_discovers_a_fourth_backend_without_a_verb_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reward_lens.integrations.trackers as trackers
    from reward_lens.integrations.trackers.registry import discover

    module = tmp_path / "test_fourth.py"
    module.write_text(
        "from reward_lens.integrations.trackers.base import Availability, TrackerRef\n"
        "class Fourth:\n"
        "    name = 'fourth'\n"
        "    def available(self): return Availability.available()\n"
        "    def write(self, summary, report_url): return TrackerRef('fourth', report_url, None, {})\n"
        "def backend(): return Fourth()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(trackers, "__path__", [*trackers.__path__, str(tmp_path)])
    importlib.invalidate_caches()

    assert "fourth" in discover()


def test_tracker_dependencies_remain_out_of_base_closure() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    base = text.split("[project.optional-dependencies]", maxsplit=1)[0].lower()
    assert "wandb" not in base
    assert "mlflow" not in base
