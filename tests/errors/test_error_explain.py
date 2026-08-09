"""`explain` prints the long form, and it does it with the network gone."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from reward_lens.contracts import RewardLensError
from reward_lens.errors import CATALOGUE, explain, render_two_line
from reward_lens.errors.catalogue import EXIT_MEANINGS

# A slot as a template writes it, and the marker `fill` leaves where it cannot fill one. A
# metavariable a remedy writes on purpose reads as words, `<the record path>`, so the marker is
# the single-token form and nothing a sentence means to say collides with it.
TEMPLATE_SLOT = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
UNFILLED_MARKER = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")

PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
DRIVER = Path(__file__).resolve().parent / "_offline_driver.py"
REPO = Path(__file__).resolve().parents[2]


def test_explain_opens_no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("explain opened a socket")

    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    text = explain("RL0210")
    assert "RL0210" in text
    assert len(text.splitlines()) > 4


def test_explain_works_when_sockets_are_blocked_before_the_import() -> None:
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env["PYTHONPATH"] = str(REPO / "src")
    result = subprocess.run(
        [sys.executable, str(DRIVER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "RL0210" in result.stdout


def test_explain_prints_the_long_form_for_every_code() -> None:
    for code, spec in CATALOGUE.items():
        text = explain(code)
        assert text.splitlines()[0].startswith(code), code
        assert spec.title in text, code
        assert spec.long_form_cause in text, code
        for remedy in spec.long_form_remedies:
            assert remedy in text, (code, remedy)
        assert EXIT_MEANINGS[spec.exit_code] in text, code


def test_no_long_form_leaves_a_placeholder_for_the_reader_to_decode() -> None:
    # `explain` is read by someone who has not hit the code yet, so there is no context and
    # nothing to fill a slot from. The long form is written to read without one (A-026).
    for code in CATALOGUE:
        text = explain(code)
        assert not TEMPLATE_SLOT.search(text), (code, TEMPLATE_SLOT.findall(text))
        assert not UNFILLED_MARKER.search(text), (code, UNFILLED_MARKER.findall(text))


def test_rl0201_names_the_file_in_words_and_keeps_the_slot_for_the_finding() -> None:
    # The cold read of wave 1 got `the grader reads {path}` from this entry. The long form now
    # says which file in words; the template it was rendered from still carries the slot, so a
    # finding with a real path in hand renders it.
    text = explain("RL0201")
    assert "the grader reads the file the finding names after" in text
    assert "{path}" not in text and "<path>" not in text
    assert "{path}" in CATALOGUE["RL0201"].cause


def test_explain_names_no_internal_module_and_no_document_section() -> None:
    internal_module = re.compile(r"reward_lens[./][A-Za-z_]")
    document_section = re.compile(r"\b[A-Z]-[0-9]{1,3}\b|\bsection [0-9]|§")

    for code in CATALOGUE:
        text = explain(code)
        assert not internal_module.search(text), code
        assert not document_section.search(text), code


def test_explain_of_an_unknown_code_is_a_two_line_rl0001_error() -> None:
    with pytest.raises(RewardLensError) as caught:
        explain("RL9999")
    error = caught.value
    assert error.code == "RL0001"
    assert error.exit_code == 4
    assert "RL9999" in error.message
    lines = render_two_line(error).splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("error: ")
    assert lines[1].startswith("help:  ")
    assert lines[2] == "       reward-lens explain RL0001"


@pytest.mark.parametrize("bad", ["", "rl0341", "RL341", "RL03410", "0341", "RL0341 ", None, 341])
def test_explain_refuses_a_malformed_code(bad: object) -> None:
    with pytest.raises(RewardLensError) as caught:
        explain(bad)  # type: ignore[arg-type]
    assert caught.value.code == "RL0001"
    assert caught.value.exit_code == 4
