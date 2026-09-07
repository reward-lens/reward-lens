"""The two forbidden things, and the codes this packet is allowed to use.

Forbidden by the packet: a field named `hidden` described as protection, and a containment claim
the sandbox probe did not establish on this machine. The second is tested in `test_separation.py`;
this file holds the field ban and the code allocation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import reward_lens.outcome as outcome
import reward_lens.partitions as partitions

SRC = Path(__file__).resolve().parents[2] / "src" / "reward_lens"
PACKAGES = (SRC / "outcome", SRC / "partitions")
FIELD = re.compile(r"\bhidden\b")
CODE = re.compile(r"RL[0-9]{4}")


def sources() -> list[Path]:
    return sorted(path for package in PACKAGES for path in package.rglob("*.py"))


def test_the_packages_exist_and_hold_source() -> None:
    assert len(sources()) >= 2


def test_no_source_line_names_a_field_called_hidden() -> None:
    offenders = [
        f"{path}:{number}"
        for path in sources()
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if FIELD.search(line)
    ]
    assert offenders == []


def test_the_only_code_allocated_here_is_RL0302() -> None:
    found: dict[str, list[str]] = {}
    for path in sources():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for code in CODE.findall(line):
                found.setdefault(code, []).append(f"{path.name}:{number}")
    assert "RL0302" in found
    assert set(found) <= {"RL0001", "RL0002", "RL0301", "RL0302", "RL0341", "RL0401", "RL0402"}


def test_RL0301_is_imported_and_never_redefined() -> None:
    for path in sources():
        text = path.read_text()
        assert 'code="RL0301"' not in text
        assert "class OutcomeUnqualified" not in text


def test_the_exhaustion_error_declares_its_own_code_and_exit() -> None:
    error = outcome.AcceptanceRoundExhausted(
        candidate_set="sha256:" + "1" * 64,
        partition_id="acceptance-suite-1",
        reason="the round was already sealed",
    )
    assert error.code == "RL0302"
    assert error.exit_code == 4
    assert error.name == "ACCEPTANCE_ROUND_EXHAUSTED"


def test_both_packages_state_what_they_export() -> None:
    for module in (outcome, partitions):
        assert module.__all__
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__} exports {name} and does not have it"


def test_no_module_here_reaches_for_the_network() -> None:
    for path in sources():
        text = path.read_text()
        for banned in ("import requests", "import socket", "urllib.request", "httpx"):
            assert banned not in text, f"{path} reaches the network with {banned}"


@pytest.mark.parametrize("name", ["Partition", "PartitionKind", "AccessLog"])
def test_the_frozen_partition_names_are_the_ones_shipped(name: str) -> None:
    assert hasattr(partitions, name)


@pytest.mark.parametrize("name", ["Round", "Panel", "SealedRequest", "AcceptanceRoundExhausted"])
def test_the_frozen_outcome_names_are_the_ones_shipped(name: str) -> None:
    assert hasattr(outcome, name)
