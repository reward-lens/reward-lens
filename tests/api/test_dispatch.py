"""The lazy seam: the engine when it is in the build, the honest record when it is not."""

from __future__ import annotations

import sys
import types

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


def test_resolution_imports_nothing_outside_the_pointer_table() -> None:
    """Resolving a verb must import the module the table names, its parents, and nothing else.

    Three of the names the table points at are already taken by modules left over from v3
    (`product.trace`, `product.improve`, `product.export`). Resolution imports them and finds no
    engine attribute, which is why a leftover module cannot quietly answer a verb.

    A name the table points at may be a package rather than a module, and a package imports its
    own submodules: `product.access` brings its `doctor`, `ladder`, `matrix` and `endpoints` with
    it. What is forbidden is a module outside the pointed names' own subtrees.
    """
    pointed = {target.split(":")[0] for target in _dispatch.ENGINES.values()}
    parents = {name.rsplit(".", 1)[0] for name in pointed}
    for target in _dispatch.ENGINES.values():
        _dispatch.load(target)
    touched = {name for name in sys.modules if name.startswith("reward_lens.product")}
    allowed = pointed | parents
    stray = {
        name
        for name in touched
        if name not in allowed and not any(name.startswith(f"{one}.") for one in pointed)
    }
    assert stray == set()


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


def test_export_refuses_with_rl0701_because_the_seam_is_empty_and_not_by_decree(
    project_dir,
) -> None:
    """RL0701 is what the absent seam means. The leftover `product.export` carries no `run`."""
    assert _dispatch.engine("export") is None
    with pytest.raises(contracts.CapabilityUnavailable) as caught:
        api.export(api.ExportRequest(project=project_dir))
    assert caught.value.code == "RL0701"
    assert caught.value.exit_code == 5
