"""`last_reuse()` where the audit engine is in the build, over a real project (A-016).

Kept apart from `test_reuse_absent.py`: every test here runs the shipped example through the
verb twice and reads what the run says about itself afterwards. The assertions are A-016's
contract, not the engine's code: a first audit measures and reports no reuse, a second audit of
an unchanged project answers out of the store and reports which record answered, and that record
is the bytes on disk rather than anything this run composed.

The sequence runs inside a fresh `contextvars.Context`, because the signal is per context and
the contract starts at "a context that has audited nothing has nothing to report". A context
built here has never been written to, so the claim is made rather than inherited from whatever
ran before it in the session.
"""

from __future__ import annotations

import contextvars
from pathlib import Path

import pytest

from reward_lens import api


@pytest.fixture
def audited_twice(project_dir) -> dict[str, object]:
    """One unchanged project, audited twice, with the signal read at each of the three moments."""
    seen: dict[str, object] = {}

    def sequence() -> None:
        seen["before"] = api.last_reuse()
        seen["first"] = api.audit(api.AuditRequest(path=project_dir))
        seen["after_first"] = api.last_reuse()
        seen["second"] = api.audit(api.AuditRequest(path=project_dir))
        seen["after_second"] = api.last_reuse()

    contextvars.Context().run(sequence)
    return seen


def test_a_context_that_has_audited_nothing_reports_nothing(audited_twice) -> None:
    assert audited_twice["before"] is None


def test_a_run_that_measured_reports_no_reuse(audited_twice) -> None:
    """The first audit of a project the store has never seen measures it, so there is no reuse."""
    assert audited_twice["after_first"] is None


def test_the_second_audit_of_an_unchanged_project_names_the_first_record(audited_twice) -> None:
    """D-28's no-op: same inputs, same record, and the run says which record it handed back."""
    reused = audited_twice["after_second"]
    assert reused is not None
    assert reused.assay_id == audited_twice["first"].assay_id
    assert audited_twice["second"].assay_id == audited_twice["first"].assay_id
    assert reused.says


def test_what_it_names_is_the_record_on_disk_and_not_a_fresh_one(audited_twice) -> None:
    """The stored bytes: the file the signal names reads back as the record that was returned."""
    reused = audited_twice["after_second"]
    stored = Path(reused.record_path)
    assert stored.is_file()
    assert api.open_record(stored).assay_id == reused.assay_id
    assert stored.stem.startswith(reused.name)
