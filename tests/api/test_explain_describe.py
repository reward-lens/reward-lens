"""SDK, CLI, and agent tools read one error catalogue and command tree."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from reward_lens import api
from reward_lens.cli.main import cli
from reward_lens.contracts import RewardLensError


def test_explain_matches_the_cli_document() -> None:
    run = CliRunner().invoke(cli, ["explain", "RL0210", "--format", "json"])
    assert run.exit_code == 0, run.output
    expected = json.loads(run.stdout)
    assert api.explain("RL0210").model_dump() == expected


def test_explain_unknown_code_is_a_typed_usage_failure() -> None:
    with pytest.raises(RewardLensError) as caught:
        api.explain("RL9999")
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4


def test_describe_matches_the_cli_document() -> None:
    run = CliRunner().invoke(cli, ["describe"])
    assert run.exit_code == 0, run.output
    expected = json.loads(run.stdout)
    assert api.describe() == expected
