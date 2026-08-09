"""Reuse: an unchanged dependency set says so, and an old measurement's subject is never rewritten."""

from __future__ import annotations

import pytest

from .conftest import ALL_IDS
from .test_invalidation import GRADER_SOURCE_STALE, GRADER_SOURCE_STANDING
from .test_project import align

STARTED = "2026-09-13T12:00:00Z"
CREATED = "2026-09-13T12:00:01Z"


@pytest.fixture()
def previous(project, record_dict):
    from reward_lens.contracts import Assay

    return Assay.model_validate(align(record_dict, project))


def test_an_unchanged_dependency_set_reuses_and_says_so(project, previous):
    reuse = project.reuse(previous)
    assert reuse.unchanged is True
    assert reuse.changed == frozenset()
    assert set(reuse.reused_entry_ids) == set(ALL_IDS)
    assert reuse.stale_entry_ids == ()
    assert "reused" in reuse.says
    assert str(len(ALL_IDS)) in reuse.says


def test_changed_since_reads_the_change_off_the_digests(project, previous, project_root):
    assert project.changed_since(previous) == frozenset()
    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.0\n", encoding="utf-8")
    assert project.changed_since(previous) == frozenset({"source"})


def test_a_changed_grader_source_reuses_what_still_stands(project, previous, project_root):
    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.0\n", encoding="utf-8")
    reuse = project.reuse(previous)
    assert reuse.unchanged is False
    assert set(reuse.stale_entry_ids) == GRADER_SOURCE_STALE
    assert set(reuse.reused_entry_ids) == GRADER_SOURCE_STANDING
    assert "source" in reuse.says


def test_every_stale_entry_becomes_an_absence_with_the_run_s_real_provenance(
    project, previous, project_root
):
    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.0\n", encoding="utf-8")
    reuse = project.reuse(previous, started=STARTED)
    assert {entry.entry_id for entry in reuse.absences} == GRADER_SOURCE_STALE
    for entry in reuse.absences:
        assert entry.kind == "absence"
        assert entry.state == "absent"
        assert entry.absence.state == "NOT_MEASURED"
        assert "source" in entry.absence.missing_access
        assert entry.provenance.started == STARTED  # this run, not a placeholder epoch
        assert entry.provenance.actor
        assert entry.subject_ref == project.version_of(previous).digest()


def test_the_new_assay_references_the_eligible_old_entries(project, previous, project_root):
    from reward_lens.contracts import Assay

    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.0\n", encoding="utf-8")
    reuse = project.reuse(previous, started=STARTED)
    new = reuse.apply(previous, created=CREATED, started=STARTED)
    assert isinstance(new, Assay)

    carried = {e.entry_id: e for e in new.entries() if e.kind != "absence"}
    assert set(carried) == GRADER_SOURCE_STANDING
    for entry_id, entry in carried.items():
        before = next(e for e in previous.entries() if e.entry_id == entry_id)
        assert entry.subject_ref == before.subject_ref  # the old entry is referenced, not rewritten
        assert entry.depends_on == before.depends_on
        assert entry.provenance.started == before.provenance.started

    assert {e.entry_id for e in new.entries() if e.kind == "absence"} == GRADER_SOURCE_STALE
    assert {h.entry_id for h in new.holes} == GRADER_SOURCE_STALE
    assert new.subject.version.digest == project.version_of(previous).digest()
    assert new.created == CREATED
    assert new.decision.state == "unresolved"
    assert any(r.startswith("required_stale:") for r in new.decision.reasons)


def test_an_old_measurement_is_never_rewritten(project, previous, project_root):
    first = project.record(previous, name="first")
    before = first.read_bytes()

    (project_root / "grader.py").write_text("def score(t, r):\n    return 0.0\n", encoding="utf-8")
    new = project.reuse(previous, started=STARTED).apply(previous, created=CREATED, started=STARTED)
    second = project.record(new, name="second")

    assert first.read_bytes() == before
    assert second != first
    assert project.open_record("first").subject.version.digest != new.subject.version.digest


def test_a_rerun_after_no_change_writes_the_same_entries_again(project, previous):
    reuse = project.reuse(previous, started=STARTED)
    new = reuse.apply(previous, created=CREATED, started=STARTED)
    assert [e.entry_id for e in new.entries()] == [e.entry_id for e in previous.entries()]
    assert not [e for e in new.entries() if e.kind == "absence"]


# --- the rewrite rule at the store's own layer (A-016) --------------------------------------------


def test_one_version_measured_the_same_way_twice_is_a_rewrite_and_is_refused(project, record_dict):
    """Same version, same entry ids, different bytes: the second record is refused, RL0620.

    This is the rule at its own layer rather than through an audit: two records built by hand, the
    only difference between them one the entry ids do not see.
    """
    from reward_lens.contracts import Assay
    from reward_lens.store.errors import SubjectVersionRewritten

    first = Assay.model_validate(align(record_dict, project))
    second = Assay.model_validate(align(dict(record_dict, created="2026-09-14T00:00:00Z"), project))
    assert second.to_dict() != first.to_dict()
    assert {entry.entry_id for entry in second.entries()} == {e.entry_id for e in first.entries()}
    assert second.subject.version.digest == first.subject.version.digest

    path = project.record(first, name="first")
    before = path.read_bytes()
    with pytest.raises(SubjectVersionRewritten) as excinfo:
        project.record(second, name="second")
    assert excinfo.value.code == "RL0620"
    assert excinfo.value.context["existing"] == "first"
    assert path.read_bytes() == before
    assert tuple(project.names()) == ("first",)


def test_one_version_measured_under_a_moved_method_is_recorded_beside_the_first(
    project, record_dict
):
    """The same entry ids measured by a method whose parameters moved is a second measurement.

    The rule keys on the pair (entry id, method identity), so re-running one entry with new
    parameters writes a new record rather than an RL0620 refusal: the ids have not moved and what
    was measured of the subject has. The audit's reuse check reads the same key, so this is exactly
    the case where it declines to hand the old record back.
    """
    import copy

    from reward_lens.contracts import Assay, digest

    moved = copy.deepcopy(record_dict)
    target = moved["measurement"]["validity"][0]
    assert target["method"]["params_digest"]
    target["method"] = dict(target["method"], params_digest=digest({"probes": 16}))

    first = Assay.model_validate(align(record_dict, project))
    second = Assay.model_validate(align(moved, project))
    assert [e.entry_id for e in second.entries()] == [e.entry_id for e in first.entries()]
    assert second.subject.version.digest == first.subject.version.digest

    path = project.record(first, name="first")
    before = path.read_bytes()
    other = project.record(second, name="second")
    assert other.is_file() and other != path
    assert tuple(project.names()) == ("first", "second")
    assert path.read_bytes() == before
    assert project.open_record("first").to_dict() == first.to_dict()


def test_one_version_measured_further_is_recorded_beside_the_first_and_not_over_it(
    project, record_dict
):
    """The same version with one entry id the first record lacks is a second measurement, not a
    rewrite: a panel that lands between two audits is exactly this case, and refusing it would
    leave a build that ships more instruments unable to record what they found."""
    from reward_lens.contracts import Assay

    from .conftest import CORPUS, SUBJECT, build_record

    wider = build_record(
        CORPUS + (("reach.attack_surface", "check", ("digest:source",), SUBJECT, "evaluator_comparison"),)
    )
    first = Assay.model_validate(align(record_dict, project))
    second = Assay.model_validate(align(wider, project))
    lacked = {e.entry_id for e in second.entries()} - {e.entry_id for e in first.entries()}
    assert lacked == {"reach.attack_surface"}

    project.record(first, name="first")
    path = project.record(second, name="second")
    assert path.is_file()
    assert tuple(project.names()) == ("first", "second")
    assert project.open_record("first").to_dict() == first.to_dict()
