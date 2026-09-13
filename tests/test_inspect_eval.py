"""The Inspect archive reader, against archives whose answers are known by construction.

Checking a reader against the real artifact only proves it agrees with itself. Every case here
builds the archive, so the expected value is arithmetic rather than a recorded observation, and the
one property that matters most is tested in the direction that can fail: `verify_reductions` is
given an archive whose reductions were written under a *different* grade mapping, and asserted to
catch it. A verifier that only ever passes is measuring its own assumptions.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.record.convert.inspect_eval import (
    GRADE_VALUES,
    InspectHeader,
    iter_scores,
    read_eval,
    read_header,
    verify_reductions,
)


def _sample(
    sample_id: str,
    epoch: int,
    *,
    fmt: float,
    passed: str,
    mentions: str,
    completion_ids: list[int] | None = None,
) -> dict:
    ids = completion_ids if completion_ids is not None else [7, 8, 9]
    return {
        "id": sample_id,
        "epoch": epoch,
        "scores": {
            "thinking_format_scorer": {"value": fmt, "explanation": "fmt note"},
            "proxy_scorer": {
                "value": {"passed": passed, "solved": "I"},
                "explanation": "## Detected: []",
            },
            "cot_scorer": {"value": {"mentions": mentions}},
        },
        "metadata": {
            "problem_id": f"p{sample_id}",
            "difficulty": 11,
            "cf_rating": 2700,
            "test_count": 1,
        },
        "output": {
            "choices": [
                {
                    "message": {"content": f"answer-{sample_id}-{epoch}"},
                    "stop_reason": "stop",
                    "logprobs": {"content": [{"token": "a", "logprob": -0.5}] * len(ids)},
                }
            ],
            "metadata": {
                "trl_completion_data": [
                    {
                        "prompt_ids": [1, 2, 3, 4],
                        "completion_ids": ids,
                        "logprobs": [-0.5] * len(ids),
                    }
                ]
            },
        },
    }


def _write(
    path: Path,
    samples: list[dict],
    *,
    header: bool = True,
    reductions: list | None = None,
    summaries: bool = True,
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        if header:
            archive.writestr(
                "header.json",
                json.dumps(
                    {
                        "eval": {
                            "eval_id": "E1",
                            "run_id": "R1",
                            "created": "2026-05-07T16:37:37+00:00",
                            "task": "t/task",
                            "model": "trl-vllm/base",
                            "revision": {
                                "type": "git",
                                "origin": "git@example.com:o/r.git",
                                "commit": "bf8c4dd",
                            },
                            "task_args": {"training": True},
                            "dataset": {"samples": 2, "sample_ids": ["0", "1"]},
                            "config": {"epochs": 2},
                            "packages": {"inspect_ai": "0.3.190"},
                            "scorers": [
                                {
                                    "name": "thinking_format_scorer",
                                    "metrics": [{"name": "accuracy"}],
                                },
                                {"name": "proxy_scorer", "metrics": {"passed": [], "solved": []}},
                                {"name": "cot_scorer", "metrics": {"mentions": []}},
                            ],
                        }
                    }
                ),
            )
        if summaries:
            archive.writestr("summaries.json", json.dumps([]))
        if reductions is not None:
            archive.writestr("reductions.json", json.dumps(reductions))
        archive.writestr("_journal/start.json", json.dumps({"eval": {"eval_id": "E1"}}))
        for sample in samples:
            archive.writestr(
                f"samples/{sample['id']}_epoch_{sample['epoch']}.json", json.dumps(sample)
            )
    return path


@pytest.fixture()
def complete(tmp_path: Path) -> Path:
    samples = [
        _sample("0", 1, fmt=0.25, passed="C", mentions="I"),
        _sample("0", 2, fmt=0.75, passed="I", mentions="C"),
        _sample("1", 1, fmt=1.0, passed="C", mentions="C"),
        _sample("1", 2, fmt=0.0, passed="C", mentions="I"),
    ]
    reductions = [
        {
            "scorer": "thinking_format_scorer",
            "reducer": "mean",
            "samples": [{"sample_id": "0", "value": 0.5}, {"sample_id": "1", "value": 0.5}],
        },
        {
            "scorer": "proxy_scorer",
            "reducer": "mean",
            "samples": [
                {"sample_id": "0", "value": {"passed": 0.5, "solved": 0.0}},
                {"sample_id": "1", "value": {"passed": 1.0, "solved": 0.0}},
            ],
        },
    ]
    return _write(tmp_path / "complete.eval", samples, reductions=reductions)


def test_header_carries_provenance_and_flattens_two_scorer_metric_shapes(complete: Path) -> None:
    header = read_header(complete)
    assert isinstance(header, InspectHeader)
    assert header.revision is not None and header.revision["commit"] == "bf8c4dd"
    assert header.epochs == 2
    assert header.sample_ids == ("0", "1")
    # One scorer declares metrics as a list and two as a mapping. Both flatten, and the count is
    # what a caller needs to know how many reward functions a configuration vector must match.
    assert header.scorer_metrics == (
        ("thinking_format_scorer", "thinking_format_scorer"),
        ("proxy_scorer", "passed"),
        ("proxy_scorer", "solved"),
        ("cot_scorer", "mentions"),
    )


def test_scores_flatten_with_grades_mapped_and_fractional_values_preserved(complete: Path) -> None:
    log = read_eval(complete)
    assert log.is_complete and len(log.samples) == 4
    first = next(s for s in log.samples if s.sample_id == "0" and s.epoch == 1)
    # The fractional component survives. This is the whole point of the module: a binarising reader
    # would report 0.0 here and lose the reward's fractional part.
    assert first.scores["thinking_format_scorer"] == 0.25
    assert first.scores["passed"] == 1.0
    assert first.scores["solved"] == 0.0
    assert first.scores["mentions"] == 0.0
    assert first.explanations["proxy_scorer"] == "## Detected: []"


def test_token_counts_come_from_ids_and_never_from_text(complete: Path) -> None:
    log = read_eval(complete)
    sample = log.samples[0]
    assert sample.n_completion_tokens == 3
    assert len(sample.prompt_ids) == 4
    assert len(sample.sampling_logprobs) == 3
    assert sample.completion_text is not None
    # The text is far longer than three tokens; the count must not be derived from it.
    assert len(sample.completion_text) > sample.n_completion_tokens


def test_completion_ids_absent_yields_none_rather_than_a_character_fallback(tmp_path: Path) -> None:
    sample = _sample("0", 1, fmt=1.0, passed="C", mentions="I")
    sample["output"]["metadata"] = {}
    path = _write(tmp_path / "notokens.eval", [sample])
    log = read_eval(path)
    assert log.samples[0].n_completion_tokens is None


def test_verify_reductions_confirms_the_grade_mapping_from_the_archives_own_arithmetic(
    complete: Path,
) -> None:
    report = verify_reductions(complete)
    assert not isinstance(report, Refusal)
    assert report["verified"] is True
    assert report["n_mismatch"] == 0
    assert report["n_checked"] > 0
    assert report["max_abs_difference"] == pytest.approx(0.0, abs=1e-12)


def test_verify_reductions_catches_a_wrong_grade_mapping(complete: Path) -> None:
    """The direction that can fail. Reverse C and I; the reductions must stop reproducing."""
    report = verify_reductions(complete, grades={"C": 0.0, "I": 1.0})
    assert not isinstance(report, Refusal)
    assert report["verified"] is False
    assert report["n_mismatch"] > 0


def test_verify_reductions_refuses_rather_than_passing_when_there_is_nothing_to_check(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "noreductions.eval", [_sample("0", 1, fmt=1.0, passed="C", mentions="I")]
    )
    report = verify_reductions(path)
    assert isinstance(report, Refusal)
    assert report.reason is RefusalReason.RECORD_INCOMPLETE
    assert "reductions" in report.detail or "reductions" in report.remedy


def test_a_started_but_empty_archive_reports_what_it_lacks_instead_of_raising(
    tmp_path: Path,
) -> None:
    """The companion run publishes one archive holding only `_journal/start.json`.

    A reader that assumed `header.json` exists would raise on it. An evaluation that started and
    wrote no samples is a fact about the run, and the reader's job is to report it.
    """
    path = tmp_path / "empty.eval"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("_journal/start.json", json.dumps({"eval": {"eval_id": "E9"}}))
    log = read_eval(path)
    assert log.is_complete is False
    assert log.header is None
    assert log.samples == ()
    assert "header.json" in log.missing
    assert "samples/" in log.missing


def test_iter_scores_emits_a_basename_join_key_and_one_row_per_rollout(complete: Path) -> None:
    rows = list(iter_scores([complete]))
    assert len(rows) == 4
    assert {r["source_eval_file"] for r in rows} == {"complete.eval"}
    assert "/" not in rows[0]["source_eval_file"]
    assert rows[0]["git_commit"] == "bf8c4dd"
    assert rows[0]["n_completion_tokens"] == 3
    assert "thinking_format_scorer" in rows[0] and "passed" in rows[0]


def test_grade_alphabet_is_the_documented_one() -> None:
    assert GRADE_VALUES["C"] == 1.0
    assert GRADE_VALUES["I"] == 0.0
    assert GRADE_VALUES["P"] == 0.5
