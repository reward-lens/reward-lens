"""The refusals: a capability that is not here, a record that is not there, one that is bad, and
a path that is not a project.

Each raises a typed error carrying the `RL` code the CLI prints and the exit code of D-22.
"""

from __future__ import annotations

import json

import pytest

from reward_lens import api, contracts
from reward_lens.api import _dispatch


def test_export_refuses_with_rl0701_until_its_engine_lands(tmp_path) -> None:
    with pytest.raises(contracts.CapabilityUnavailable) as caught:
        api.export(api.ExportRequest(record=tmp_path / "a.assay.json", dest=tmp_path / "out"))
    error = caught.value
    assert error.code == "RL0701"
    assert error.exit_code == 5
    assert error.remediation
    assert json.loads(error.to_json())["code"] == "RL0701"


def test_open_record_on_a_missing_path_refuses_with_rl0621(tmp_path) -> None:
    missing = tmp_path / "nowhere.assay.json"
    with pytest.raises(contracts.RewardLensError) as caught:
        api.open_record(missing)
    error = caught.value
    assert error.code == "RL0621"
    assert error.exit_code == 4
    assert str(missing) in error.context["path"]


def test_open_record_on_an_invalid_record_refuses_with_rl0604(tmp_path) -> None:
    path = tmp_path / "bad.assay.json"
    raw = b'{"$schema": "https://reward-lens.github.io/schema/assay/1.0/assay.schema.json"}'
    path.write_bytes(raw)
    with pytest.raises(contracts.RecordInvalid) as caught:
        api.open_record(path)
    error = caught.value
    assert error.code == "RL0604"
    assert error.exit_code == 4
    assert error.context["schema_path"]
    assert error.context["original_bytes"] == raw


def test_open_record_on_a_file_that_is_not_json_refuses_with_rl0604(tmp_path) -> None:
    path = tmp_path / "notjson.assay.json"
    path.write_bytes(b"not json at all")
    with pytest.raises(contracts.RecordInvalid) as caught:
        api.open_record(path)
    assert caught.value.code == "RL0604"
    assert caught.value.context["original_bytes"] == b"not json at all"


def test_audit_refuses_a_path_that_is_not_a_project_with_rl0001(tmp_path) -> None:
    """A directory holding no `rewardlens.yaml` is not a project, and the refusal names the path.

    The refusal belongs to the engine, so the premise is stated: with the seam empty `audit`
    answers with the honest record instead, which is `test_dispatch`\'s case and not this one.
    """
    assert _dispatch.engine("audit") is not None
    bare = tmp_path / "not-a-project"
    bare.mkdir()
    with pytest.raises(contracts.UsageError) as caught:
        api.audit(api.AuditRequest(path=bare))
    error = caught.value
    assert error.code == "RL0001"
    assert error.exit_code == 4
    assert str(bare) in error.message
    assert error.context["path"] == str(bare)
    assert error.remediation


def test_open_record_reads_back_what_a_verb_produced(project_dir, tmp_path) -> None:
    record = api.audit(api.AuditRequest(path=project_dir))
    path = tmp_path / "good.assay.json"
    path.write_bytes(contracts.canonical_bytes(record))
    with pytest.raises(contracts.RecordInvalid):
        api.open_record(path)
    path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    reopened = api.open_record(path)
    assert isinstance(reopened, contracts.Assay)
    assert reopened.assay_id == record.assay_id
    # STEER, point 2: a record built by the constructors and the same record loaded from disk
    # hash the same, and the digest on it is the digest of it.
    assert contracts.digest(reopened) == contracts.digest(record) == record.assay_id
    assert contracts.canonical_bytes(reopened) == contracts.canonical_bytes(record)
