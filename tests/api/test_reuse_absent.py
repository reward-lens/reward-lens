"""`last_reuse()` where the audit engine is not in the build (A-016).

Kept apart from `test_reuse_installed.py` on purpose: nothing here runs a real audit, and nothing
here reads whether this tree happens to hold the engine. The absence is made at the seam, the one
lookup the verb performs, so each test states the rule rather than the shape of the day's tree.
"""

from __future__ import annotations

import inspect

import pytest

from reward_lens import api, contracts
from reward_lens.api import _dispatch

from .conftest import engines_absent

#: The pointer A-016 names. The verb resolves this string and nothing else.
TARGET = "reward_lens.product.audit:last_reuse"


@pytest.fixture
def reader_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam answers `None` for the reuse reader, as it does in a build without `product/`."""
    real = _dispatch.load

    def load(target: str):
        return None if target == TARGET else real(target)

    monkeypatch.setattr(_dispatch, "load", load)


def test_the_signal_is_a_no_argument_function_on_the_public_surface() -> None:
    """The signature A-016 fixes: `last_reuse() -> Reused | None`, exported and taking nothing."""
    assert "last_reuse" in api.__all__
    assert inspect.signature(api.last_reuse).parameters == {}
    assert api.last_reuse.__doc__


def test_nothing_is_reported_when_the_reader_is_not_in_the_build(reader_absent) -> None:
    assert _dispatch.load(TARGET) is None
    assert api.last_reuse() is None


def test_an_absent_build_answers_none_rather_than_raising(
    reader_absent, monkeypatch, project_dir
) -> None:
    """A build that cannot measure cannot reuse, and says so twice over.

    The verb still returns the honest all-absence record, and the signal beside it is `None`:
    the true answer for a run that measured nothing because there was nothing to measure with,
    rather than a fabricated reuse or an exception the caller has to catch.
    """
    engines_absent(monkeypatch, "audit")
    record = api.audit(api.AuditRequest(path=project_dir))
    assert isinstance(record, contracts.Assay)
    assert {entry.kind for entry in record.entries()} == {"absence"}
    assert api.last_reuse() is None


def test_the_seam_is_asked_on_every_call_and_the_answer_is_never_cached(
    monkeypatch, project_dir
) -> None:
    """A build is read at the call, not at import: the same process answers both ways.

    `_dispatch.load` resolves its pointer on demand and caches nothing, so a caller that has
    seen `None` once is not stuck with it. The absence is put in place, read, and taken away.
    """
    real = _dispatch.load
    monkeypatch.setattr(
        _dispatch, "load", lambda target: None if target == TARGET else real(target)
    )
    assert api.last_reuse() is None
    monkeypatch.undo()
    assert (_dispatch.load(TARGET) is None) == (
        _dispatch.load("reward_lens.product.audit:run") is None
    )
