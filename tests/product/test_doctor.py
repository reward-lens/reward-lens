"""P-DOCTOR: the doctor and the access ladder (D-20, D-37, section 5.6).

The panel is not an install check. Every line of it is measured on this machine and this input, and
every line that says "no" says what would make it "yes" and what that would cost.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import re
import socket
import sys
import threading
import types
from fnmatch import fnmatch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reward_lens import execution
from reward_lens.api import AuditRequest, Capabilities, Capability
from reward_lens.execution.probe import SandboxProbe, TierResult
from reward_lens.product import access
from reward_lens.product.access import endpoints, ladder, matrix

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "fleet" / "golden" / "wave-1" / "doctor.txt"
VOLATILE = ROOT / "fleet" / "golden" / "VOLATILE.md"
DEMO_NUMBERS = ROOT / "fleet" / "golden" / "DEMO_NUMBERS.md"


# --- helpers -------------------------------------------------------------------------------------


def _linux_tiers(held_through: str) -> dict[str, TierResult]:
    """A full Linux ladder whose rungs hold up to and including `held_through`."""
    ladder_names = ("T0", "L0", "L1", "L2", "L3")
    out: dict[str, TierResult] = {}
    still = True
    for name in ladder_names:
        out[name] = TierResult(
            held=still,
            reason=f"{name} held" if still else f"{name} did not hold on this machine",
            ms=1.0,
        )
        if name == held_through:
            still = False
    return out


def _fake_probe(tier: str) -> SandboxProbe:
    return SandboxProbe(
        os="linux",
        tier_held=tier,
        tiers=_linux_tiers(tier),
        kernel="6.8.0-test",
        landlock_abi=4,
        platform_matrix={},
    )


def _stub_audit(**attrs: object) -> types.ModuleType:
    """A `reward_lens.product.audit` that exists but runs nothing: the wave-1 seam, by hand."""
    module = types.ModuleType("reward_lens.product.audit")
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def wave_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The tree the wave-1 golden was cut from: the audit engine present, nothing else."""
    monkeypatch.setitem(sys.modules, "reward_lens.product.audit", _stub_audit())
    monkeypatch.setattr(endpoints, "scan", lambda **_: ())
    # Two of the tokens are read off the machine rather than declared, so the tree the golden was
    # cut from has to pin them too: no API key in the environment, and no reference set installed.
    for variable in ladder.api_key_variables():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv(ladder.REFERENCE_VARIABLE, raising=False)
    monkeypatch.setenv(ladder.REFERENCE_HOME_VARIABLE, str(tmp_path / "no-home"))


#: One declared rule cannot match the line its own replacement describes. `install-path` reads the
#: os field as a single run of non-space characters, and the doctor prints it as `linux x86_64`, so
#: the rule never fires and the python version and the platform stay in the compared text. They are
#: the machine block, which section 5 declares volatile, and the rule's own replacement says it
#: masks them. The one field is widened here to the run of words it is, which masks exactly what
#: the rule already says it masks. `fleet/golden/VOLATILE.md` is the controller's file and owes the
#: same correction; until it has it, this is the only place the test does not quote the file.
_RULE_CORRECTIONS = {
    "install-path": r"python 3\.\d+\.\d+  ·  \S+(?: \S+)*  ·  \S+",
}


def _declared_masks(name: str) -> list[tuple[re.Pattern[str], str]]:
    """The masks `fleet/golden/VOLATILE.md` declares for one golden file, read from that file.

    The declaration is the single source it says it is, so the test applies it rather than keeping
    a second copy that could drift wider: whichever rules name a glob this file matches, in the
    order they are written, plus the numeral mask while `DEMO_NUMBERS.md` does not exist.
    """
    block = re.search(r"```json\n(.*?)\n```", VOLATILE.read_text(), re.S)
    assert block is not None, "VOLATILE.md carries the machine-readable rule set in a JSON block"
    declared = json.loads(block.group(1))
    masks = [
        (re.compile(_RULE_CORRECTIONS.get(rule["name"], rule["pattern"])), rule["replace"])
        for rule in declared["rules"]
        if any(fnmatch(name, glob) for glob in rule["files"])
    ]
    numerals = declared.get("numerals_until_demo_numbers") or {}
    if any(fnmatch(name, glob) for glob in numerals.get("files", ())) and not DEMO_NUMBERS.exists():
        masks.append((re.compile(numerals["pattern"]), numerals["replace"]))
    return masks


def _normalise(text: str) -> str:
    """Mask exactly what the commission declared volatile for `doctor.txt`, and nothing wider."""
    for pattern, replacement in _declared_masks(GOLDEN.name):
        text = pattern.sub(replacement, text)
    return text


class _Models(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's own spelling
        if self.path.startswith("/v1/models"):
            body = json.dumps(
                {"object": "list", "data": [{"id": "qwen2.5-coder-7b"}, {"id": "llama3.2"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:  # pragma: no cover - the scanner never asks for another path
            self.send_error(404)

    def log_message(self, *args: object) -> None:
        return None


@contextlib.contextmanager
def _openai_compatible_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Models)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --- the panel -----------------------------------------------------------------------------------


def test_text_matches_the_golden_outside_the_volatile_fields(wave_one: None) -> None:
    golden = GOLDEN.read_text()
    body = "\n".join(golden.split("\n")[1:])  # the `$ reward-lens doctor` line is the shell's
    produced = access.render_text(access.doctor())
    assert _normalise(produced) == _normalise(body)


def test_the_report_is_a_capabilities_object(wave_one: None) -> None:
    caps = access.doctor()
    assert isinstance(caps, Capabilities)
    assert len(caps) > 0
    assert all(isinstance(c, Capability) for c in caps)
    assert caps.install.version and caps.install.python


@pytest.mark.parametrize("tier", ["L0", "L2"])
def test_the_sandbox_line_carries_the_os_and_the_tier_that_held(
    monkeypatch: pytest.MonkeyPatch, wave_one: None, tier: str
) -> None:
    monkeypatch.setattr(execution, "probe", lambda **_: _fake_probe(tier))
    caps = access.doctor()
    assert caps.sandbox.os == "linux"
    assert caps.sandbox.tier_held == tier
    line = next(
        ln for ln in access.render_text(caps).split("\n") if ln.strip().startswith(f"{tier}:")
    )
    assert line.startswith(f"    {tier}: ")
    assert line.endswith(", held")
    assert "probed in " in line


def test_the_sandbox_label_is_the_probe_s_and_never_a_claim(wave_one: None) -> None:
    caps = access.doctor()
    assert caps.sandbox.tier_held == execution.probe().tier_held
    assert caps.sandbox.os == execution.probe().os


def test_every_unavailable_capability_names_what_would_unlock_it_and_what_it_costs(
    wave_one: None,
) -> None:
    caps = access.doctor()
    unavailable = [c for c in caps if c.status != "available"]
    assert unavailable, "wave 1 does not hold every instrument; some capability must say so"
    for capability in unavailable:
        assert capability.reason, capability.id
        assert capability.unlocks, capability.id
        assert all(isinstance(u, str) and u for u in capability.unlocks), capability.id
        assert re.fullmatch(r"\d+\.\d{2}", capability.cost), (capability.id, capability.cost)


def test_an_api_key_in_the_environment_is_what_settles_the_seeker_s_api_arm(
    wave_one: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`key:api` is read off the machine, so the rung moves when the environment does."""
    monkeypatch.setitem(
        sys.modules, "reward_lens.product.seeker", types.ModuleType("reward_lens.product.seeker")
    )
    variable = ladder.api_key_variables()[0]
    before = {c.id: c for c in access.doctor()}["seeker.api"]
    assert before.status == "unavailable"
    assert variable in before.reason, before.reason

    monkeypatch.setenv(variable, "a-key-this-test-never-spends")
    after = {c.id: c for c in access.doctor()}["seeker.api"]
    assert after.status == "available"


def test_a_reference_set_on_the_machine_is_what_settles_the_calibration_rung(
    wave_one: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`reference` is a lookup, not a constant: install one and calibration comes up."""
    before = {c.id: c for c in access.doctor()}["calibration"]
    assert before.status == "unavailable"
    assert any(ladder.REFERENCE_VARIABLE in one for one in before.unlocks), before.unlocks

    materials = tmp_path / "reference"
    materials.mkdir()
    (materials / "crm-1.json").write_text('{"assigned_value": 0.5, "uncertainty": 0.1}')
    monkeypatch.setenv(ladder.REFERENCE_VARIABLE, str(materials))
    root, found = ladder.reference_set()
    assert root == materials and len(found) == 1
    after = {c.id: c for c in access.doctor()}["calibration"]
    assert after.status == "available"


def test_json_form_is_one_object_per_capability_with_the_five_fields(wave_one: None) -> None:
    caps = access.doctor()
    payload = access.render_json(caps)
    assert isinstance(payload, list)
    assert len(payload) == len(caps)
    for obj, capability in zip(payload, caps, strict=True):
        assert set(obj) == {"id", "status", "reason", "unlocks", "cost"}
        assert obj["id"] == capability.id
        assert obj["status"] in {"available", "unavailable", "pending"}
    json.dumps(payload)  # an agent branches on a field, so the form has to serialise


def test_the_panel_is_derived_from_what_is_in_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """No audit engine, so the panels that wave 1 fills are not printed as available."""
    monkeypatch.delitem(sys.modules, "reward_lens.product.audit", raising=False)
    monkeypatch.setattr(endpoints, "scan", lambda **_: ())
    monkeypatch.setattr(ladder, "_audit_module", lambda: None)
    caps = access.doctor()
    by_id = {c.id: c for c in caps}
    assert by_id["panel.validity"].status == "unavailable"
    assert by_id["panel.validity"].reason == "not in this build"
    assert "validity" not in access.render_text(caps).split("What it cannot")[0]


def test_a_panel_the_audit_engine_declares_is_reported_as_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoints, "scan", lambda **_: ())
    monkeypatch.setitem(
        sys.modules,
        "reward_lens.product.audit",
        _stub_audit(PANELS_FILLED={"validity": "full", "signal": "full"}),
    )
    by_id = {c.id: c for c in access.doctor()}
    assert by_id["panel.validity"].status == "available"
    assert by_id["panel.signal"].status == "available"
    assert by_id["panel.reach"].status == "unavailable"


# --- the compatibility matrix (section 7.1) ------------------------------------------------------


def test_the_matrix_covers_the_families_in_this_build() -> None:
    built = matrix.build()
    assert "local_deterministic" in built
    row = built["local_deterministic"]
    assert row["conformance"] == "ran"
    for name in ("component_dag", "token_quantities", "cost_metering"):
        assert row[name] in {"yes", "no"}


def test_a_hand_edited_matrix_entry_does_not_survive_a_rebuild() -> None:
    built = matrix.build()
    before = built["local_deterministic"]["cost_metering"]
    built["local_deterministic"]["cost_metering"] = "hand-written"
    built["a_family_nobody_ran"] = {"conformance": "ran"}
    again = matrix.build()
    assert again["local_deterministic"]["cost_metering"] == before
    assert "a_family_nobody_ran" not in again


def test_the_matrix_follows_the_family_s_conformance_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Change what the family's conformance reports and the cell changes: it is derived."""

    def stub() -> matrix.Conformance:
        return matrix.Conformance(
            family="stub_family",
            capabilities={"component_dag": True, "cost_metering": False},
            probed={"component_dag": True},
            determinism_class="declared_deterministic",
            network_policy="deny",
            access_level="source_visible",
            replay_mode="local_replay_unproven",
        )

    monkeypatch.setitem(matrix.FAMILIES, "stub_family", stub)
    built = matrix.build()
    assert built["stub_family"]["component_dag"] == "yes"
    assert built["stub_family"]["cost_metering"] == "no"


def test_a_family_whose_conformance_cannot_run_is_not_reported_as_conformant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> matrix.Conformance:
        raise RuntimeError("the fixture grader is not in this build")

    monkeypatch.setitem(matrix.FAMILIES, "broken_family", broken)
    row = matrix.build()["broken_family"]
    assert row["conformance"] == "did not run"
    assert "the fixture grader is not in this build" in row["reason"]


# --- the local endpoint scan ---------------------------------------------------------------------


def test_the_scan_finds_a_local_openai_compatible_server() -> None:
    with _openai_compatible_server() as port:
        found = endpoints.scan(ports=(port,))
    assert len(found) == 1
    endpoint = found[0]
    assert endpoint.port == port
    assert endpoint.base_url == f"http://127.0.0.1:{port}/v1"
    assert "qwen2.5-coder-7b" in endpoint.models


def test_the_scan_finds_nothing_when_none_runs() -> None:
    assert endpoints.scan(ports=(_closed_port(),)) == ()


def test_a_found_endpoint_is_named_as_a_seeker_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "reward_lens.product.audit", _stub_audit())
    real = endpoints.scan
    with _openai_compatible_server() as port:
        monkeypatch.setattr(endpoints, "scan", lambda **_: real(ports=(port,)))
        caps = access.doctor()
    local = next(c for c in caps if c.id == "seeker.local")
    assert any(f"127.0.0.1:{port}" in line for line in local.unlocks), local.unlocks
    assert any("actor" in line for line in local.unlocks), local.unlocks


def test_the_scan_refuses_a_host_that_is_not_loopback() -> None:
    with pytest.raises(ValueError):
        endpoints.scan(host="203.0.113.7", ports=(80,))
    for host in endpoints.DEFAULT_HOSTS:
        assert ipaddress.ip_address(host).is_loopback


# --- dry run -------------------------------------------------------------------------------------


def test_dry_run_executes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "rewardlens.yaml").write_text(
        "reward:\n  kind: plain\n  entry: grader.py\n  shape: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n"
    )

    def never(*args: object, **kwargs: object):  # pragma: no cover - the point is it is not called
        raise AssertionError("dry_run ran something in the sandbox")

    for name in ("LinuxSandbox", "MacSandbox", "WindowsSandbox"):
        module = {"LinuxSandbox": "linux", "MacSandbox": "macos", "WindowsSandbox": "windows"}[name]
        target = sys.modules.get(f"reward_lens.execution.{module}")
        if target is not None and hasattr(target, name):
            monkeypatch.setattr(getattr(target, name), "run", never)
    monkeypatch.setattr(execution, "run_python", never)

    plan = access.dry_run(AuditRequest(path=project, dry_run=True))
    assert plan.command == "audit"
    assert plan.paid_calls == 0
    assert plan.estimate_usd == "0.00"
    assert plan.panels


def test_dry_run_prints_the_same_panel_scoped_to_the_planned_run(
    wave_one: None, tmp_path: Path
) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "rewardlens.yaml").write_text(
        "reward:\n  kind: plain\n  entry: grader.py\n  shape: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n"
    )
    plan = access.dry_run(AuditRequest(path=project, dry_run=True))
    panel = access.render_text(access.doctor())
    scoped = access.render_text(access.doctor(project=project), plan=plan)
    for line in panel.split("\n"):
        if line.strip() and not line.startswith("    reward-lens "):
            assert line in scoped, line
    assert "audit" in scoped


def test_dry_run_falls_back_to_the_interfaces_panel_list_without_an_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The seam is the one `dry_run` asks, which is the dispatch lookup of the audit engine's
    # `plan`, not the ladder's module check: the audit packet has since landed, so a build without
    # the engine is now made by refusing that one lookup.
    from reward_lens.api import _dispatch

    resolved = _dispatch.load
    monkeypatch.setattr(
        _dispatch,
        "load",
        lambda target: None if target.startswith("reward_lens.product.audit") else resolved(target),
    )
    project = tmp_path / "p"
    project.mkdir()
    (project / "rewardlens.yaml").write_text(
        "reward:\n  kind: plain\n  entry: grader.py\n  shape: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\noutcome: null\n"
    )
    plan = access.dry_run(AuditRequest(path=project, dry_run=True))
    assert plan.panels == access.INTERFACE_PANELS
    assert any("not in this build" in note for note in plan.notes)


# --- the two refusals ----------------------------------------------------------------------------


def test_doctor_whose_probe_raised_refuses_with_RL0401_naming_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raises(**_: object) -> SandboxProbe:
        raise OSError("landlock_create_ruleset: ENOSYS")

    monkeypatch.setattr(execution, "probe", raises)
    with pytest.raises(Exception) as caught:
        access.doctor()
    error = caught.value
    assert error.code == "RL0401"
    assert error.exit_code != 0
    assert "probe" in error.context
    assert "landlock_create_ruleset" in str(error)


def test_doctor_exits_zero_whenever_it_produced_its_report(
    wave_one: None, tmp_path: Path
) -> None:
    """A directory that is not a project is a fact about the input, not a reason to refuse."""
    caps = access.doctor(project=tmp_path)
    assert isinstance(caps, Capabilities)
    project = next(c for c in caps if c.id == "input.project")
    assert project.status == "unavailable"
    assert "rewardlens.yaml" in project.reason


def test_dry_run_without_a_rewardlens_yaml_refuses_with_RL0003(tmp_path: Path) -> None:
    with pytest.raises(Exception) as caught:
        access.dry_run(AuditRequest(path=tmp_path, dry_run=True))
    assert caught.value.code == "RL0003"
    assert caught.value.exit_code == 4


# --- what the doctor is not ----------------------------------------------------------------------


def test_the_report_is_not_a_list_of_installed_versions(wave_one: None) -> None:
    """D-20's forbidden case: a capability panel whose content is `pip list`."""
    caps = access.doctor()
    ids = {c.id for c in caps}
    assert ids == set(ladder.rung_ids(project=None))
    text = access.render_text(caps)
    body = text.split("What this machine can measure now")[1]
    assert "pytest" not in body and "pydantic" not in body
    assert "what each would take" in text
