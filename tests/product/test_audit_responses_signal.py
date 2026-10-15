"""The response-bank score carrier used by Signal's public audit route."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from reward_lens import api, contracts
from reward_lens.cli.main import cli
from reward_lens.examples.code_reward import write as write_example
from reward_lens.execution import default_sandbox
from reward_lens.product.audit import response_scores
from reward_lens.product.audit.instruments import build_corpus
from reward_lens.product.audit.response_scores import score_response_bank

_GRADER = """\
def score(task, response):
    return 1.0 if response == task["expected"] else 0.0
"""


def _launch_contained_child_from_thread(ready: multiprocessing.Queue) -> None:
    """Exercise the Linux non-leader-thread child accounting path."""

    def launch() -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        ready.put((child.pid, os.getpgid(child.pid)))
        child.wait()

    thread = threading.Thread(target=launch)
    thread.start()
    thread.join()


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _bank(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    grader = tmp_path / "grader.py"
    grader.write_text(_GRADER, encoding="utf-8")
    tasks = _jsonl(
        tmp_path / "tasks.jsonl",
        [
            {"task_id": "prompt-a", "prompt": "a", "expected": "yes"},
            {"task_id": "prompt-b", "prompt": "b", "expected": "up"},
        ],
    )
    responses = _jsonl(
        tmp_path / "responses.jsonl",
        [
            {"response_id": "a-yes", "task_id": "prompt-a", "response": "yes"},
            {"response_id": "a-no", "task_id": "prompt-a", "response": "no"},
            {"response_id": "b-up", "task_id": "prompt-b", "response": "up"},
            {"response_id": "b-down", "task_id": "prompt-b", "response": "down"},
        ],
    )
    return grader, tasks, responses


def _nontrivial_bank(tmp_path: Path) -> tuple[Path, Path, Path]:
    grader, _, _ = _bank(tmp_path)
    tasks = _jsonl(
        tmp_path / "tasks.jsonl",
        [
            {"task_id": f"prompt-{task}", "prompt": str(task), "expected": "yes"}
            for task in range(15)
        ],
    )
    responses = _jsonl(
        tmp_path / "responses.jsonl",
        [
            {
                "response_id": f"prompt-{task}-{position}",
                "task_id": f"prompt-{task}",
                "response": "no"
                if task == 0 and position == 1
                else ("yes" if task % 2 == 0 else "no"),
            }
            for task in range(15)
            for position in range(2 + task % 3)
        ],
    )
    return grader, tasks, responses


def _by_id(record, entry_id: str):
    return next(entry for entry in record.measurement.signal if entry.entry_id == entry_id)


def test_response_scores_execute_the_real_grader_and_preserve_declared_prompt_groups(
    tmp_path: Path,
) -> None:
    grader, tasks, responses = _bank(tmp_path)

    corpus = build_corpus(tasks_path=tasks, responses_path=responses)
    scores = score_response_bank(
        grader_path=grader,
        entrypoint="score",
        corpus=corpus,
        sandbox=default_sandbox(),
        project_dir=tmp_path,
    )

    assert [(row.response_id, row.task_id, row.score, row.status) for row in scores] == [
        ("a-yes", "prompt-a", 1.0, "measured"),
        ("a-no", "prompt-a", 0.0, "measured"),
        ("b-up", "prompt-b", 1.0, "measured"),
        ("b-down", "prompt-b", 0.0, "measured"),
    ]


def test_response_scores_ignore_an_unwritable_diagnostic_profile_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader, tasks, responses = _bank(tmp_path)
    monkeypatch.setenv("REWARD_LENS_MCP_PROFILE_PATH", "/proc/reward-lens-profile")

    scores = score_response_bank(
        grader_path=grader,
        entrypoint="score",
        corpus=build_corpus(tasks_path=tasks, responses_path=responses),
        sandbox=default_sandbox(),
        project_dir=tmp_path,
    )

    assert [(row.score, row.status) for row in scores] == [
        (1.0, "measured"),
        (0.0, "measured"),
        (1.0, "measured"),
        (0.0, "measured"),
    ]


def test_response_scores_keep_input_order_and_failures_under_bounded_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent score subprocesses may finish out of order without moving evidence rows."""
    grader, tasks, responses = _bank(tmp_path)
    active = 0
    peak = 0
    completed: list[str] = []
    lock = threading.Lock()
    delays = {"yes": 0.06, "no": 0.01, "up": 0.04, "down": 0.02}

    class ControlledGrader:
        def score(self, task, response, **_kwargs):  # noqa: ANN001
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(delays[response])
                if response == "no":
                    return SimpleNamespace(
                        score=None,
                        verdict="grader_error",
                        errors=("controlled score failure",),
                    )
                return SimpleNamespace(score=1.0, verdict="scored", errors=())
            finally:
                with lock:
                    active -= 1
                    completed.append(response)

    controlled = ControlledGrader()
    monkeypatch.setattr(
        response_scores,
        "PythonGrader",
        SimpleNamespace(from_path=lambda _path, _entrypoint: controlled),
    )

    rows = score_response_bank(
        grader_path=grader,
        entrypoint="score",
        corpus=build_corpus(tasks_path=tasks, responses_path=responses),
        sandbox=object(),
        project_dir=tmp_path,
    )

    assert [row.response_id for row in rows] == ["a-yes", "a-no", "b-up", "b-down"]
    assert [(row.status, row.score) for row in rows] == [
        ("measured", 1.0),
        ("error", None),
        ("measured", 1.0),
        ("measured", 1.0),
    ]
    assert rows[1].reason == "controlled score failure"
    assert completed[0] == "no"
    assert len(completed) == len(rows) == 4
    assert len(set(completed)) == len(rows)
    assert 1 < peak <= response_scores._MAX_RESPONSE_SCORE_WORKERS


def test_response_scores_share_one_capacity_bound_across_concurrent_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent audit callers cannot multiply the response-grader process cap."""
    grader, tasks, responses = _bank(tmp_path)
    active = 0
    peak = 0
    lock = threading.Lock()

    class ControlledGrader:
        def score(self, _task, _response, **_kwargs):  # noqa: ANN001
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.03)
                return SimpleNamespace(score=1.0, verdict="scored", errors=())
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr(
        response_scores,
        "PythonGrader",
        SimpleNamespace(from_path=lambda _path, _entrypoint: ControlledGrader()),
    )
    monkeypatch.setattr(response_scores, "_MAX_RESPONSE_SCORE_WORKERS", 2)
    monkeypatch.setattr(response_scores, "_RESPONSE_SCORE_CAPACITY", threading.BoundedSemaphore(2))
    corpus = build_corpus(tasks_path=tasks, responses_path=responses)
    start = threading.Barrier(3)
    results: list[tuple[object, ...]] = []

    def score() -> None:
        start.wait()
        results.append(
            score_response_bank(
                grader_path=grader,
                entrypoint="score",
                corpus=corpus,
                sandbox=object(),
                project_dir=tmp_path,
            )
        )

    first = threading.Thread(target=score)
    second = threading.Thread(target=score)
    first.start()
    second.start()
    start.wait()
    first.join()
    second.join()

    assert len(results) == 2
    assert all(len(rows) == 4 for rows in results)
    assert peak == 2


def test_response_score_reaper_finds_children_started_by_nonleader_threads() -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    worker = context.Process(target=_launch_contained_child_from_thread, args=(ready,))
    worker.start()
    _pid, group = ready.get(timeout=5)
    response_scores._reap_contained_child_groups(worker)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"contained group {group} remained after reaping")


def test_response_scores_retain_later_rows_after_an_operational_worker_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader, tasks, responses = _bank(tmp_path)

    class RaisingGrader:
        def score(self, _task, response, **_kwargs):  # noqa: ANN001
            if response == "yes":
                raise RuntimeError("sandbox transport failed")
            return SimpleNamespace(score=1.0, verdict="scored", errors=())

    monkeypatch.setattr(
        response_scores,
        "PythonGrader",
        SimpleNamespace(from_path=lambda _path, _entrypoint: RaisingGrader()),
    )

    rows = score_response_bank(
        grader_path=grader,
        entrypoint="score",
        corpus=build_corpus(tasks_path=tasks, responses_path=responses),
        sandbox=object(),
        project_dir=tmp_path,
    )

    assert [row.response_id for row in rows] == ["a-yes", "a-no", "b-up", "b-down"]
    assert [(row.status, row.score) for row in rows] == [
        ("error", None),
        ("measured", 1.0),
        ("measured", 1.0),
        ("measured", 1.0),
    ]
    assert rows[0].reason == "operational scoring failure: RuntimeError: sandbox transport failed"


def test_response_scores_preserve_an_unscored_grader_result_as_an_absence(tmp_path: Path) -> None:
    grader = tmp_path / "none.py"
    grader.write_text("def score(task, response):\n    return None\n", encoding="utf-8")
    tasks = _jsonl(
        tmp_path / "tasks.jsonl",
        [{"task_id": "prompt-a", "prompt": "a", "expected": "yes"}],
    )
    responses = _jsonl(
        tmp_path / "responses.jsonl",
        [{"response_id": "a-none", "task_id": "prompt-a", "response": "yes"}],
    )

    (row,) = score_response_bank(
        grader_path=grader,
        entrypoint="score",
        corpus=build_corpus(tasks_path=tasks, responses_path=responses),
        sandbox=default_sandbox(),
        project_dir=tmp_path,
    )

    assert (row.response_id, row.task_id, row.score, row.status) == (
        "a-none",
        "prompt-a",
        None,
        "error",
    )
    assert row.reason and "unscored" in row.reason


def test_sdk_response_bank_route_saves_and_reopens_scoped_signal_evidence(tmp_path: Path) -> None:
    grader, tasks, responses = _bank(tmp_path)

    record = api.audit(
        api.AuditRequest(
            path=grader,
            tasks=tasks,
            responses=responses,
            name="signal-sdk",
        )
    )

    contrast = _by_id(record, "signal.response_score_contrast_fraction")
    assert contrast.kind == "estimate" and contrast.value == 1.0
    assert contrast.method.id == "signal.response_bank_score"
    assert contrast.result["scored_response_count"] == 4
    assert contrast.result["invalid_response_count"] == 0
    assert contrast.result["execution_error_count"] == 0
    assert "not a recorded applied training reward" in contrast.limitations[0]
    saved = tmp_path / "assays" / "signal-sdk.assay.json"
    reopened = api.open_record(saved)
    assert _by_id(reopened, "signal.response_score_contrast_fraction").result == contrast.result


def test_cli_responses_route_forwards_bank_and_persists_signal_evidence(tmp_path: Path) -> None:
    grader, tasks, responses = _bank(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "audit",
            str(grader),
            "--tasks",
            str(tasks),
            "--responses",
            str(responses),
            "--name",
            "signal-cli",
            "--format",
            "json",
            "--exit-zero",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    saved = api.open_record(tmp_path / "assays" / "signal-cli.assay.json")
    assert (
        _by_id(saved, "signal.response_score_contrast_fraction").result["scored_response_count"]
        == 4
    )


def test_sdk_response_bank_with_nontrivial_ess_saves_quantised_signal_uncertainty(
    tmp_path: Path,
) -> None:
    grader, tasks, responses = _nontrivial_bank(tmp_path)

    record = api.audit(
        api.AuditRequest(
            path=grader, tasks=tasks, responses=responses, name="signal-sdk-nontrivial"
        )
    )

    contrast = _by_id(record, "signal.response_score_contrast_fraction")
    assert contrast.result["design_effect"] == pytest.approx(3.084159506499229)
    assert contrast.uncertainty.design_effect == contracts.quantise(
        contrast.result["design_effect"], 4
    )
    reopened = api.open_record(tmp_path / "assays" / "signal-sdk-nontrivial.assay.json")
    assert _by_id(reopened, contrast.entry_id).uncertainty == contrast.uncertainty


def test_cli_response_bank_with_nontrivial_ess_saves_and_reopens_quantised_uncertainty(
    tmp_path: Path,
) -> None:
    grader, tasks, responses = _nontrivial_bank(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "audit",
            str(grader),
            "--tasks",
            str(tasks),
            "--responses",
            str(responses),
            "--name",
            "signal-cli-nontrivial",
            "--format",
            "json",
            "--exit-zero",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    reopened = api.open_record(tmp_path / "assays" / "signal-cli-nontrivial.assay.json")
    contrast = _by_id(reopened, "signal.response_score_contrast_fraction")
    assert contrast.result["design_effect"] == pytest.approx(3.084159506499229)
    assert contrast.uncertainty.design_effect == contracts.quantise(
        contrast.result["design_effect"], 4
    )


def test_shipped_example_sdk_audit_saves_and_reopens_quantised_signal_uncertainty(
    tmp_path: Path,
) -> None:
    project = tmp_path / "example"
    write_example(project)

    record = api.audit(api.AuditRequest(path=project, name="signal-example"))

    contrast = _by_id(record, "signal.response_score_contrast_fraction")
    assert contrast.result["design_effect"] == pytest.approx(2.132923322008713)
    assert contrast.uncertainty.design_effect == contracts.quantise(
        contrast.result["design_effect"], 4
    )
    reopened = api.open_record(project / "assays" / "signal-example.assay.json")
    assert _by_id(reopened, contrast.entry_id).uncertainty == contrast.uncertainty


def test_sdk_response_bank_with_no_eligible_prompt_group_remains_an_absence(tmp_path: Path) -> None:
    grader = tmp_path / "grader.py"
    grader.write_text(_GRADER, encoding="utf-8")
    tasks = _jsonl(
        tmp_path / "tasks.jsonl",
        [{"task_id": "prompt-a", "prompt": "a", "expected": "yes"}],
    )
    responses = _jsonl(
        tmp_path / "responses.jsonl",
        [{"response_id": "only", "task_id": "prompt-a", "response": "yes"}],
    )

    record = api.audit(
        api.AuditRequest(path=grader, tasks=tasks, responses=responses, name="signal-missing")
    )

    absence = _by_id(record, "signal.response_score_contrast_fraction")
    assert absence.kind == "absence"
    assert (
        "no declared prompt group has two finite scored responses" in absence.absence.missing_access
    )
    assert absence.method.id == "signal.response_bank_score"
