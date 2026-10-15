"""The public trace route delivers the source record it just read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_trl_completions(directory: Path) -> Path:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    directory.mkdir(parents=True)
    path = directory / "completions_00000.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "step": [0, 0, 0, 0],
                "prompt": ["same"] * 4,
                "completion": ["a", "b", "c", "d"],
                "length_reward": [0.0, 1.0, 2.0, 3.0],
                "format_reward": [1.0, 1.0, 1.0, 1.0],
                "group": ["g0"] * 4,
                "advantage": [-1.5, -0.5, 0.5, 1.5],
            }
        ),
        path,
    )
    return directory


def test_trace_cli_forwards_declared_trl_partition_and_delivers_its_assay(
    cli, tmp_path: Path
) -> None:
    run_dir = _write_trl_completions(tmp_path / "trl-run")
    result = cli(
        "trace",
        run_dir.name,
        "--reward-function",
        "length_reward",
        "--reward-function",
        "format_reward",
        "--extra-column",
        "group",
        "--weight",
        "length_reward=0.5",
        "--weight",
        "format_reward=2",
        "--group-column",
        "group",
        "--group-size",
        "4",
        "--format",
        "json",
        cwd=tmp_path,
    )
    assert result.exit_code == 2, (result.stdout, result.stderr)
    document = json.loads(result.stdout)
    assert document["command"] == "trace"
    assert document["artifacts"]
    # Trace artifacts are named from the source run, as `artifacts_for_record` names every
    # record from its own project root. The command's CWD is only where the relative run path was
    # resolved, not the base of the envelope paths.
    assay = run_dir / document["artifacts"]["assay"]
    report = run_dir / document["artifacts"]["report"]
    assert assay.is_file()
    assert report.is_file()
    payload = json.loads(assay.read_text(encoding="utf-8"))
    assert payload["measurement"]["trace"]
    assert payload["measurement"]["trace"][0]["result"]["reader"] == "trl_completions"


def test_trace_text_names_its_trace_evidence(cli, tmp_path: Path) -> None:
    run_dir = _write_trl_completions(tmp_path / "trl-run")
    result = cli(
        "trace",
        run_dir.name,
        "--reward-function",
        "length_reward",
        "--reward-function",
        "format_reward",
        "--extra-column",
        "group",
        "--group-column",
        "group",
        "--group-size",
        "4",
        "--format",
        "text",
        cwd=tmp_path,
    )
    assert result.exit_code == 2, (result.stdout, result.stderr)
    assert "Trace evidence" in result.stdout
    assert "trl_completions" in result.stdout
    assert "assays/" in result.stdout


def test_trace_cli_refuses_a_nonfinite_weight_before_it_reads_a_run(cli) -> None:
    result = cli("trace", "missing-run", "--weight", "length_reward=nan", "--format", "json")
    assert result.exit_code == 4
    assert "weights" in result.stderr
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "RL0003"


def test_trace_cli_refuses_trl_extras_without_an_explicit_reward_partition(
    cli, tmp_path: Path
) -> None:
    run_dir = _write_trl_completions(tmp_path / "trl-run")
    result = cli(
        "trace",
        run_dir.name,
        "--extra-column",
        "group",
        "--group-column",
        "group",
        "--group-size",
        "4",
        "--format",
        "json",
        cwd=tmp_path,
    )
    assert result.exit_code == 4
    assert "reward-function" in result.stderr
    assert not (run_dir / "assays").exists(), result.stdout
