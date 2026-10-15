"""The public Compare scorecard survives its saved Assay without reconstruction."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from reward_lens import api, contracts
from reward_lens.bank import load_bank
from reward_lens.cli.main import cli
from reward_lens.product.compare.persisted_scorecard import (
    SCORECARD_EXTENSION,
    PersistedScorecardError,
    read_persisted_scorecards,
)
from reward_lens.product.compare.registry import Rung
from reward_lens.product.compare.rung1 import rescore


def _project(destination: Path) -> Path:
    from reward_lens.examples import REGISTRY

    module_name, _, attribute = REGISTRY["code-reward"].partition(":")
    getattr(importlib.import_module(module_name), attribute)(destination)
    return destination


def _bank(destination: Path) -> Path:
    destination.write_text(
        "\n".join(
            (
                json.dumps({"name": "bank"}),
                json.dumps(
                    {
                        "task_id": "one",
                        "responses": [
                            {"text": "a", "scores": {"before": "1", "after": "2"}},
                            {"text": "b", "scores": {"before": "3", "after": "4"}},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "two",
                        "responses": [
                            {"text": "c", "scores": {"before": "3", "after": "6"}},
                            {"text": "d", "scores": {"before": "5", "after": "8"}},
                        ],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _expected(bank: Path):
    return rescore(load_bank(bank, name=bank.stem), "before", "after").scorecard


def test_sdk_compare_saves_and_reopens_the_actual_six_row_scorecard(tmp_path: Path) -> None:
    root = _project(tmp_path / "sdk-project")
    bank = _bank(root / "bank.jsonl")

    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank,
            project=root,
            rungs=(1,),
            name="sdk-scorecard",
        )
    )
    saved = root / "assays" / "sdk-scorecard.assay.json"
    reopened = api.open_record(saved)

    observed = read_persisted_scorecards(reopened)
    assert observed == (_expected(bank),)
    extension = reopened.extensions[SCORECARD_EXTENSION]
    assert extension["baseline"] == "before"
    assert extension["candidate"] == "after"
    assert extension["subject_ref"] == reopened.subject.version.digest
    assert extension["scorecards"][0]["rung"] == Rung.RE_SCORE.value
    assert [row["id"] for row in extension["scorecards"][0]["rows"]] == [
        "intended_task_success",
        "legitimate_solutions_kept",
        "solution_diversity",
        "exploit_pressure",
        "signal",
        "cost",
    ]
    evidence = extension["scorecards"][0]["evidence"]
    assert evidence and all(
        item["subject_ref"] == reopened.subject.version.digest for item in evidence
    )
    assert record.assay_id == reopened.assay_id


def test_cli_compare_saves_and_reopens_the_same_typed_scorecard(tmp_path: Path) -> None:
    from click.testing import CliRunner

    root = _project(tmp_path / "cli-project")
    bank = _bank(root / "bank.jsonl")
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "before",
            "after",
            "--rung",
            "1",
            "--bank",
            str(bank),
            "--project",
            str(root),
            "--name",
            "cli-scorecard",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2, result.output
    reopened = api.open_record(root / "assays" / "cli-scorecard.assay.json")
    assert read_persisted_scorecards(reopened) == (_expected(bank),)


def test_reader_refuses_missing_foreign_and_tampered_scorecard_payloads(tmp_path: Path) -> None:
    root = _project(tmp_path / "controls-project")
    bank = _bank(root / "bank.jsonl")
    record = api.compare(
        api.CompareRequest(
            baseline="before",
            candidate="after",
            bank=bank,
            project=root,
            rungs=(1,),
            name="control-scorecard",
        )
    )

    missing = copy.deepcopy(record)
    del missing.extensions[SCORECARD_EXTENSION]
    missing.assay_id = contracts.digest(missing.to_dict())
    with pytest.raises(PersistedScorecardError, match="no persisted compare scorecard"):
        read_persisted_scorecards(missing)

    foreign = copy.deepcopy(record)
    foreign.extensions[SCORECARD_EXTENSION]["subject_ref"] = "sha256:" + "0" * 64
    foreign.assay_id = contracts.digest(foreign.to_dict())
    with pytest.raises(PersistedScorecardError, match="foreign subject"):
        read_persisted_scorecards(foreign)

    tampered = copy.deepcopy(record)
    tampered.extensions[SCORECARD_EXTENSION]["scorecards"][0]["rows"][0]["value"]["value"] = "9"
    with pytest.raises(PersistedScorecardError, match="record digest"):
        read_persisted_scorecards(tampered)
