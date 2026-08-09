"""`Project.open`, `init`, `record`, `latest`: files are canonical, the index is rebuildable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import PROJECT_YAML, build_record, revise_grader, write_project


def align(record: dict, project, *, methods: tuple[str, ...] | None = None) -> dict:
    """The record, moved onto the version this project currently declares.

    Everything a version is moves together: the nine digests, the subject version they compose,
    and the `subject_ref` of every entry that was about that subject. An entry that named a
    different subject keeps the one it named. The id is recomputed last, because it is the digest
    of everything above it. `methods` names the instrument method set by hand, where a test wants
    to say what ran rather than let the helper read it off the record.
    """
    from reward_lens.contracts import Assay, digest

    version = project.version(methods=methods) if methods is not None else project.version_of(record)
    was = record["subject"]["version"]["digest"]
    now = version.digest()
    record = dict(record)
    record["subject"] = dict(record["subject"])
    record["subject"]["digests"] = version.digests.to_dict()
    record["subject"]["version"] = dict(record["subject"]["version"], id=version.id, digest=now)
    record["measurement"] = {
        section: [
            dict(entry, subject_ref=now) if entry["subject_ref"] == was else dict(entry)
            for entry in entries
        ]
        for section, entries in record["measurement"].items()
    }
    data = Assay.model_validate(record).to_dict()
    data["assay_id"] = digest(data)
    return data


@pytest.fixture()
def aligned(project, record_dict):
    from reward_lens.contracts import Assay

    return Assay.model_validate(align(record_dict, project))


# --- open and init ------------------------------------------------------------------------------


def test_open_reads_the_config(project):
    from reward_lens.contracts import ProjectConfig

    assert isinstance(project.config, ProjectConfig)
    assert project.config.reward.entry == "grader.py:score"


def test_init_writes_a_config_a_later_open_reads(tmp_path: Path):
    from reward_lens.contracts import ProjectConfig
    from reward_lens.store import Project

    config = ProjectConfig.model_validate(
        {
            "reward": {"kind": "plain", "entry": "grader.py:score"},
            "tasks": {"path": "tasks.jsonl", "prompt": "prompt"},
            "outcome": None,
        }
    )
    root = tmp_path / "fresh"
    created = Project.init(root, config)
    assert (root / "rewardlens.yaml").is_file()
    assert Project.open(root).config.to_dict() == created.config.to_dict()


def test_init_refuses_to_overwrite_an_existing_project(project_root):
    from reward_lens.store import Project
    from reward_lens.store.errors import BadConfigField

    before = (project_root / "rewardlens.yaml").read_bytes()
    with pytest.raises(BadConfigField):
        Project.init(project_root, Project.open(project_root).config)
    assert (project_root / "rewardlens.yaml").read_bytes() == before


# --- the refusals -------------------------------------------------------------------------------


def test_open_without_a_config_names_the_file_it_wanted(tmp_path: Path):
    from reward_lens.store import Project
    from reward_lens.store.errors import BadConfigField

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BadConfigField) as excinfo:
        Project.open(empty)
    error = excinfo.value
    assert error.code == "RL0003"
    assert error.exit_code == 4
    assert error.context["field"] == "rewardlens.yaml"
    assert "rewardlens.yaml" in error.message
    assert error.remediation


def test_open_with_a_bad_field_names_the_field(tmp_path: Path):
    from reward_lens.store import Project
    from reward_lens.store.errors import BadConfigField

    root = write_project(tmp_path / "bad", yaml_text="reward:\n  kind: plain\ntasks: null\n")
    with pytest.raises(BadConfigField) as excinfo:
        Project.open(root)
    assert excinfo.value.code == "RL0003"
    assert excinfo.value.context["field"] in ("reward.entry", "tasks")
    assert excinfo.value.context["field"] in excinfo.value.message


def test_open_keeps_the_contracts_refusal_as_the_cause(tmp_path: Path):
    """The store's public error is the one a caller catches; the refusal under it stays readable.

    `BadConfigField` is what the store promises, so the contracts' own RL0003 is chained rather
    than swallowed: the cause still holds pydantic's every location, and everything the refusal
    carried is in the store error's context too.
    """
    from reward_lens.contracts.errors import UsageError
    from reward_lens.store import Project
    from reward_lens.store.errors import BadConfigField

    root = write_project(tmp_path / "bad", yaml_text="reward:\n  kind: plain\ntasks: null\n")
    with pytest.raises(BadConfigField) as excinfo:
        Project.open(root)
    cause = excinfo.value.__cause__
    assert isinstance(cause, UsageError)
    assert not isinstance(cause, BadConfigField)
    assert cause.code == "RL0003"
    assert excinfo.value.context["detail"] == cause.context["detail"]
    assert len(cause.__cause__.errors()) == 3


def test_open_with_an_unknown_reward_kind_names_the_field_and_the_file(tmp_path: Path):
    """Valid YAML, a valid mapping, and still not a project the contracts will admit."""
    from reward_lens.store import Project
    from reward_lens.store.errors import BadConfigField

    text = PROJECT_YAML.replace("kind: composite", "kind: no_such_kind")
    root = write_project(tmp_path / "unknown-kind", yaml_text=text)
    with pytest.raises(BadConfigField) as excinfo:
        Project.open(root)
    error = excinfo.value
    assert error.code == "RL0003"
    assert error.context["field"] == "reward.kind"
    assert error.context["path"] == str(root / "rewardlens.yaml")
    assert "reward.kind" in error.message
    assert str(root / "rewardlens.yaml") in error.message
    assert error.remediation


def test_recording_a_record_whose_digests_disagree_names_the_digest(project, record_dict):
    from reward_lens.contracts import Assay
    from reward_lens.store.errors import DependencyDigestMismatch

    record = align(record_dict, project)
    record["subject"]["digests"] = dict(record["subject"]["digests"], source="sha256:" + "9" * 64)
    with pytest.raises(DependencyDigestMismatch) as excinfo:
        project.record(Assay.model_validate(record), name="mismatched")
    error = excinfo.value
    assert error.code == "RL0620"
    assert error.exit_code == 4
    assert error.context["digest"] == "source"
    assert "source" in error.message
    assert not (project.root / "assays" / "mismatched.assay.json").exists()


def test_recording_a_different_record_under_a_taken_name_is_refused(
    project, aligned, record_dict, project_root
):
    from reward_lens.contracts import UsageError

    path = project.record(aligned, name="first")
    before = path.read_bytes()
    other = _second_record(project, project_root, record_dict)  # a later version, a taken name
    with pytest.raises(UsageError) as excinfo:
        project.record(other, name="first")
    assert excinfo.value.code == "RL0001"
    assert path.read_bytes() == before


def test_recording_a_record_whose_subject_version_is_not_the_project_s_is_refused(
    project, record_dict, project_root
):
    """The record measures the version it names; a project that has moved on refuses to hold it."""
    from reward_lens.contracts import Assay
    from reward_lens.store.errors import DependencyDigestMismatch

    measured = align(record_dict, project)
    was = measured["subject"]["version"]["digest"]
    revise_grader(project_root)  # the project is now a different version of the reward system
    with pytest.raises(DependencyDigestMismatch) as excinfo:
        project.record(Assay.model_validate(measured), name="elsewhere")
    error = excinfo.value
    assert error.code == "RL0620"
    assert error.context["digest"] in ("source", "subject.version")
    assert not (project.root / "assays" / "elsewhere.assay.json").exists()

    # and the version digest itself is checked, not only the nine it is composed of
    stale = dict(measured)
    stale["subject"] = dict(measured["subject"])
    stale["subject"]["digests"] = project.version_of(measured).digests.to_dict()
    with pytest.raises(DependencyDigestMismatch) as excinfo:
        project.record(Assay.model_validate(stale), name="elsewhere")
    assert excinfo.value.context["digest"] == "subject.version"
    assert excinfo.value.context["found"] == was


def test_recording_a_record_whose_id_disagrees_with_its_bytes_is_refused(project, record_dict):
    """`record()` checks the id it was given; it never quietly writes a corrected one."""
    from reward_lens.contracts import Assay, digest
    from reward_lens.store.errors import DependencyDigestMismatch

    record = align(record_dict, project)
    honest = record["assay_id"]
    record["assay_id"] = "sha256:" + "7" * 64
    with pytest.raises(DependencyDigestMismatch) as excinfo:
        project.record(Assay.model_validate(record), name="misnamed")
    error = excinfo.value
    assert error.code == "RL0620"
    assert error.context["digest"] == "assay_id"
    assert error.context["found"] == "sha256:" + "7" * 64
    assert error.context["expected"] == honest == digest(record)
    assert not (project.root / "assays" / "misnamed.assay.json").exists()


def test_a_second_measurement_of_one_subject_version_is_refused_under_any_name(
    project, aligned, record_dict
):
    """One version, one measurement: neither the name it was written under nor a fresh one."""
    from reward_lens.contracts import Assay
    from reward_lens.store.errors import SubjectVersionRewritten

    path = project.record(aligned, name="first")
    before = path.read_bytes()
    version = aligned.subject.version.digest

    different = align(dict(record_dict, created="2026-09-14T00:00:00Z"), project)
    assert different["subject"]["version"]["digest"] == version  # same version, different record

    for name in ("first", "second"):
        with pytest.raises(SubjectVersionRewritten) as excinfo:
            project.record(Assay.model_validate(different), name=name)
        error = excinfo.value
        assert error.code == "RL0620"
        assert error.exit_code == 4
        assert error.context["existing"] == "first"
        assert error.context["version"] == version
        assert version in error.message

    assert path.read_bytes() == before
    assert tuple(project.names()) == ("first",)


def test_a_record_that_names_its_instrument_methods_records_and_reads_back(project, record_dict):
    """`record()` and `version(methods=...)` derive the method set the same way, so they agree."""
    from reward_lens.contracts import Assay

    named = {
        section: [
            dict(entry, method=dict(entry["method"], id="rl.verifier.replay_fidelity",
                                    version="3.1.0"))
            for entry in entries
        ]
        for section, entries in record_dict["measurement"].items()
    }
    record = align(
        dict(record_dict, measurement=named), project, methods=("rl.verifier.replay_fidelity@3.1.0",)
    )
    path = project.record(Assay.model_validate(record), name="instrumented")
    assert path.is_file()
    reopened = project.open_record("instrumented")
    assert {e.method.version for e in reopened.entries()} == {"3.1.0"}
    assert project.changed_since(reopened) == frozenset()


# --- record and latest --------------------------------------------------------------------------


def test_record_writes_canonical_bytes_under_the_name(project, aligned):
    import rfc8785

    from reward_lens.contracts import digest

    path = project.record(aligned, name="first")
    assert path == project.root / "assays" / "first.assay.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert path.read_bytes() == rfc8785.dumps(written)  # the file itself is canonical
    assert written["assay_id"] == digest(written)  # and the id is the digest of what it holds


def test_recording_the_same_record_twice_is_a_no_op(project, aligned):
    first = project.record(aligned, name="first")
    before = first.read_bytes()
    again = project.record(aligned, name="first")
    assert again == first
    assert first.read_bytes() == before


def test_latest_is_none_before_anything_is_recorded(project):
    assert project.latest() is None


def _second_record(project, project_root, record_dict):
    """A second measurement, of a second version: the grader moved between the two."""
    from reward_lens.contracts import Assay

    revise_grader(project_root)
    later = dict(record_dict, created="2026-09-14T00:00:00Z")
    return Assay.model_validate(align(later, project))


def test_latest_returns_the_most_recently_recorded(project, aligned, record_dict, project_root):
    project.record(aligned, name="first")
    project.record(_second_record(project, project_root, record_dict), name="second")
    latest = project.latest()
    assert latest is not None
    assert latest.created == "2026-09-14T00:00:00Z"


def test_the_index_is_a_cache_and_the_files_are_the_truth(
    project, aligned, record_dict, project_root
):
    project.record(aligned, name="first")
    project.record(_second_record(project, project_root, record_dict), name="second")
    with_index = (project.latest().assay_id, tuple(project.names()))

    index = project.root / "assays" / "index.json"
    assert index.is_file()
    index.unlink()
    assert (project.latest().assay_id, tuple(project.names())) == with_index
    assert index.is_file()  # rebuilt


def test_a_corrupt_index_is_rebuilt_rather_than_believed(project, aligned):
    project.record(aligned, name="first")
    index = project.root / "assays" / "index.json"
    index.write_text("not json at all", encoding="utf-8")
    assert project.latest() is not None
    assert tuple(project.names()) == ("first",)


def test_open_record_round_trips_the_reference_fixture(project):
    from reward_lens.contracts import Assay

    record = Assay.model_validate(align(build_record(), project))
    path = project.record(record, name="reference")
    reopened = project.open_record("reference")
    assert isinstance(reopened, Assay)
    assert reopened.to_dict()["measurement"] == record.to_dict()["measurement"]
    assert path.is_file()


def test_a_record_survives_a_reload_byte_for_byte(project, aligned):
    """STEER: write, reload, write again; the bytes and the digest are the same both times."""
    first = project.record(aligned, name="round")
    payload = first.read_bytes()
    second = project.record(project.open_record("round"), name="round-again")
    assert second.read_bytes() == payload
    assert json.loads(payload)["assay_id"] == json.loads(second.read_bytes())["assay_id"]


def test_the_reference_record_written_by_the_store_reads_back_identical(project):
    """STEER: the frozen reference fixture, written by the store, round-trips byte for byte."""
    from .conftest import reference_dict

    from reward_lens.contracts import Assay

    path = project.record(Assay.model_validate(align(reference_dict(), project)), name="ref")
    payload = path.read_bytes()
    again = project.record(project.open_record("ref"), name="ref-again")
    assert again.read_bytes() == payload


def test_no_database_is_the_source_of_truth(project, aligned):
    project.record(aligned, name="first")
    assert not list(project.root.rglob("*.sqlite"))
    assert not list(project.root.rglob("*.db"))
