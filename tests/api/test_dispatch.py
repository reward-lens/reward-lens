"""The lazy seam: the engine when it is in the build, the honest record when it is not."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from reward_lens import api, contracts
from reward_lens.api import _dispatch

from .conftest import RECORD_VERBS, engines_absent


@pytest.fixture
def fake_audit_engine(project_dir, monkeypatch):
    """A stand-in engine behind the `audit` pointer, resolved the way every engine is.

    The pointer is moved rather than `reward_lens.product.audit` replaced in `sys.modules`: that
    module is in the build now, and a fixture that overwrote it would be measuring its own
    eviction. A hand-built module has no `__spec__`, which is the case `_dispatch.load` consults
    `sys.modules` for, so this still exercises that branch.
    """
    calls: list[dict] = []
    with monkeypatch.context() as absent:
        engines_absent(absent, "audit")
        record = api.audit(api.AuditRequest(path=project_dir))
    module = types.ModuleType("reward_lens.product.stand_in_audit")

    def run(req, *, project, sandbox):
        calls.append({"req": req, "project": project, "sandbox": sandbox})
        return record

    module.run = run
    monkeypatch.setitem(sys.modules, "reward_lens.product.stand_in_audit", module)
    monkeypatch.setitem(_dispatch.ENGINES, "audit", "reward_lens.product.stand_in_audit:run")
    return module, calls, record


def test_audit_dispatches_to_the_engine_when_the_module_is_there(
    fake_audit_engine, project_dir
) -> None:
    _, calls, record = fake_audit_engine
    request = api.AuditRequest(path=project_dir)
    assert api.audit(request) is record
    assert len(calls) == 1
    assert calls[0]["req"] is request
    assert set(calls[0]) == {"req", "project", "sandbox"}


def test_audit_builds_the_honest_record_when_the_module_is_not_there(
    project_dir, monkeypatch
) -> None:
    """The engine is taken away and then the verb is asked, rather than the build being read.

    `sys.modules` after the call answers nothing: resolving any verb imports the module its
    pointer names, so the module is there afterwards whichever way the call went. The case is the
    one the seam decides, `engine("audit")` finding nothing, and the evidence is the record.
    """
    engines_absent(monkeypatch, "audit")
    assert _dispatch.engine("audit") is None
    record = api.audit(api.AuditRequest(path=project_dir))
    assert isinstance(record, contracts.Assay)
    assert len(record.holes) == 10
    assert {entry.kind for entry in record.entries()} == {"absence"}
    assert record.subject.version.id == "unread"
    assert record.decision.state == "unresolved"


def test_doctor_dispatches_to_the_access_engine_when_it_is_there(monkeypatch) -> None:
    module = types.ModuleType("reward_lens.product.stand_in_access")
    sentinel = api.doctor()

    def doctor(*, project=None):
        return sentinel

    module.doctor = doctor
    monkeypatch.setitem(sys.modules, "reward_lens.product.stand_in_access", module)
    monkeypatch.setitem(_dispatch.ENGINES, "doctor", "reward_lens.product.stand_in_access:doctor")
    assert api.doctor() is sentinel


def test_the_engine_table_names_every_verb() -> None:
    assert set(_dispatch.ENGINES) >= set(RECORD_VERBS) | {"doctor", "dry_run", "export"}
    for target in _dispatch.ENGINES.values():
        module, sep, attribute = target.partition(":")
        assert sep == ":" and module.startswith("reward_lens.") and attribute


def test_explicit_missing_forecast_project_refuses_instead_of_relocating(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing-project"
    with pytest.raises(contracts.UsageError) as caught:
        api.forecast_ledger(api.ForecastLedgerRequest(project=missing))
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "project"
    assert not (tmp_path / ".reward-lens").exists()


def test_unknown_forecast_id_is_a_typed_identifier_refusal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(contracts.UsageError) as caught:
        api.forecast_resolve(
            api.ForecastResolveRequest(forecast_id="sha256:" + "0" * 64, outcome="true")
        )
    assert caught.value.code == "RL0002"
    assert caught.value.context["field"] == "forecast_id"


def test_malformed_issue_record_is_a_typed_input_refusal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    issue_record = tmp_path / "issue.json"
    issue_record.write_text("{}", encoding="utf-8")
    with pytest.raises(contracts.UsageError) as caught:
        api.forecast_issue(api.ForecastIssueRequest(claim="test", record=issue_record))
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "record"


def test_improve_missing_project_is_a_typed_path_refusal(tmp_path) -> None:
    with pytest.raises(contracts.UsageError) as caught:
        api.improve(api.ImproveRequest(path=tmp_path / "missing", describe="test"))
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "path"


def test_forecast_resolve_rejects_invalid_outcome_as_typed_input(tmp_path, monkeypatch) -> None:
    from tests.product.test_forecast import _issue_row

    monkeypatch.chdir(tmp_path)
    issue_record = tmp_path / "issue.json"
    issue_record.write_text(json.dumps({"issues": [_issue_row("T1")]}), encoding="utf-8")
    api.forecast_issue(api.ForecastIssueRequest(claim="test", record=issue_record))
    rows = [
        json.loads(line)
        for line in (tmp_path / ".reward-lens/forecast-ledger.jsonl").read_text().splitlines()
    ]
    digest = next(row["sealed_digest"] for row in rows if row["kind"] == "issue")
    with pytest.raises(contracts.UsageError) as caught:
        api.forecast_resolve(api.ForecastResolveRequest(forecast_id=digest, outcome="maybe"))
    assert caught.value.code == "RL0003"
    assert caught.value.context["field"] == "outcome"


def test_expected_refusal_mapping_does_not_hide_programming_value_errors() -> None:
    def broken(request, **kwargs):
        raise ValueError("unexpected engine bug")

    with pytest.raises(ValueError, match="unexpected engine bug"):
        _dispatch.call_expected_refusal(broken, object(), verb="forecast_issue")


def test_an_absent_module_a_missing_attribute_and_a_non_callable_all_resolve_to_none() -> None:
    assert _dispatch.load("reward_lens.product.nothing_here:run") is None
    module = types.ModuleType("reward_lens.product.halfway")
    module.not_run = 1
    module.run = 7
    sys.modules["reward_lens.product.halfway"] = module
    try:
        assert _dispatch.load("reward_lens.product.halfway:missing") is None
        assert _dispatch.load("reward_lens.product.halfway:run") is None
    finally:
        sys.modules.pop("reward_lens.product.halfway", None)


def test_resolution_imports_only_the_requested_engine_tree() -> None:
    """The lazy seam must not load other engines merely because one verb is resolved.

    An engine module may import its own implementation dependencies while it defines its callable.
    The public lazy-dispatch boundary is earlier: resolving audit must not also resolve a different
    pointer such as compare, improve, export, trace, or forecast.
    """
    source_root = Path(__file__).parents[2] / "src"
    probe = """
import json
import sys
from reward_lens.api import _dispatch
before = {name for name in sys.modules if name.startswith('reward_lens.product')}
assert _dispatch.engine('audit') is not None
after = {name for name in sys.modules if name.startswith('reward_lens.product')}
print(json.dumps(sorted(after - before)))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=source_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    touched = set(json.loads(result.stdout))
    assert any(
        name == "reward_lens.product.audit" or name.startswith("reward_lens.product.audit.")
        for name in touched
    )
    unrelated = ("compare", "improve", "export", "trace", "forecast")
    assert not any(
        name.startswith("reward_lens.product." + branch) for branch in unrelated for name in touched
    )


@pytest.fixture
def fake_export_engine(project_dir, monkeypatch):
    """A stand-in renderer behind the `export` pointer, resolved the way every engine is.

    Moved rather than shadowed, for the reason `fake_audit_engine` gives: `product.export` is a
    v3 leftover today and will be a real module the day its packet lands.
    """
    calls: list[dict] = []
    written = project_dir / "report.html"
    module = types.ModuleType("reward_lens.product.stand_in_export")

    def run(req, *, project, sandbox):
        calls.append({"req": req, "project": project, "sandbox": sandbox})
        return written

    module.run = run
    monkeypatch.setitem(sys.modules, "reward_lens.product.stand_in_export", module)
    monkeypatch.setitem(_dispatch.ENGINES, "export", "reward_lens.product.stand_in_export:run")
    return calls, written


def test_export_dispatches_to_the_renderer_when_the_module_is_there(
    fake_export_engine, project_dir
) -> None:
    calls, written = fake_export_engine
    request = api.ExportRequest(project=project_dir, dest=written)
    assert api.export(request) is written
    assert len(calls) == 1
    assert calls[0]["req"] is request
    assert set(calls[0]) == {"req", "project", "sandbox"}


def test_export_propagates_the_missing_record_refusal_from_the_landed_engine(
    project_dir,
) -> None:
    """The landed renderer opens its record before rendering, preserving RL0621 for a missing one."""
    assert _dispatch.engine("export") is not None
    with pytest.raises(contracts.UsageError) as caught:
        api.export(
            api.ExportRequest(project=project_dir, record=project_dir / "missing.assay.json")
        )
    assert caught.value.code == "RL0621"
    assert caught.value.exit_code == 4
