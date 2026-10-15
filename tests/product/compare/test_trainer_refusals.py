"""A declared unsupported trainer is a capability refusal at public boundaries."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from reward_lens import api, contracts
from reward_lens.cli.main import cli
from reward_lens.product.compare.rung4 import CpuControlledUpdateConfig
from reward_lens.product.compare.rung5 import default_cpu_short_fork_config


def _config(tmp_path: Path, rung: int) -> tuple[Path, str]:
    path = tmp_path / "execution.json"
    subject = contracts.digest({"unread_subject": {"command": "compare", "path": str(path)}})
    if rung == 4:
        payload = CpuControlledUpdateConfig(
            seed=37, max_completion_length=4, candidate_offsets=(0.5, -0.25), subject_ref=subject
        ).canonical_mapping()
        field = "controlled_update_config"
    else:
        payload = default_cpu_short_fork_config(subject_ref=subject).to_mapping()
        field = "short_fork_config"
    path.write_text(json.dumps(payload) + "\n")
    return path, field


@pytest.mark.parametrize("rung", [4, 5])
@pytest.mark.parametrize("installed", ["9.99.0", None])
def test_sdk_and_cli_refuse_unsupported_trainer_before_writing_artifacts(
    tmp_path, monkeypatch, rung, installed
):
    path, field = _config(tmp_path, rung)
    original = importlib.metadata.version

    def version(name):
        if name != "trl":
            return original(name)
        if installed is None:
            raise importlib.metadata.PackageNotFoundError("trl")
        return installed

    monkeypatch.setattr(importlib.metadata, "version", version)
    request = api.CompareRequest(
        baseline="before", candidate="after", rungs=(rung,), **{field: path}
    )
    with pytest.raises(contracts.CapabilityUnavailable) as refusal:
        api.compare(request)
    assert refusal.value.code == "RL0702" and refusal.value.exit_code == 5
    assert refusal.value.context["installed_version"] == (installed or "not installed")
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "before",
            "after",
            "--rung",
            str(rung),
            "--" + field.replace("_", "-"),
            str(path),
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 5
    assert "RL0702" in result.stderr and "RL0900" not in result.stderr
    assert not list(tmp_path.rglob("runtime-config.json"))
    assert not list(tmp_path.rglob("*.assay.json"))
