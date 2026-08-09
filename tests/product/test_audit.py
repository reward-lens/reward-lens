"""The audit spine: three real instruments, four static checks, and an absence for everything else.

The fixture graders live here as source strings rather than under `tests/product/fixtures/audit/`
because the fleet guard refuses that directory to this packet (see the handoff's premise checks).
Each one is still one fixture per static check, built to violate exactly that check.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from reward_lens import api, contracts
from reward_lens.contracts import RewardLensError
from reward_lens.examples.code_reward import write as write_example
from reward_lens.execution import default_sandbox
from reward_lens.instruments.base import Panel
from reward_lens.product import audit
from reward_lens.product.audit import absences, registry, static_checks
from reward_lens.product.audit.run import (
    DEFAULT_SYSTEM_NAME,
    MANIFEST_NAME,
    NAME_DIGEST_CHARS,
    _system_name,
    manifest_bytes,
    record_bytes,
)
from reward_lens.product.audit.plan import BUILTIN_PANELS, SECTIONS
from reward_lens.store import Project

# --- fixture graders, one per static check -------------------------------------------------------

CLEAN = '''
"""Passes all four static checks: no exec, no answer key, no task-only branch."""
from typing import Any


def score(task: Any, response: Any) -> float:
    if not isinstance(task, dict) or not isinstance(response, str):
        return 0.0
    wanted = set(str(task.get("prompt", "")).lower().split())
    if not wanted:
        return 0.0
    return float(len(wanted & set(response.lower().split()))) / float(len(wanted))
'''

WRITES_THEN_READS = '''
"""RL0201: the response runs in the working directory and the grader reads a file in it."""
from pathlib import Path
from typing import Any

TESTS = Path("outcome") / "test_solution.py"


def score(task: Any, response: Any) -> float:
    namespace: dict[str, Any] = {}
    try:
        exec(compile(str(response), "response.py", "exec"), namespace)
    except Exception:
        return 0.0
    if not TESTS.exists():
        return 0.0
    return 1.0 if "assert" in TESTS.read_text(encoding="utf-8") else 0.0
'''

LEAKY_TASK = '''
"""RL0202: the task carries the answer and the grader reads it straight out of the input."""
from typing import Any


def score(task: Any, response: Any) -> float:
    if not isinstance(task, dict) or not isinstance(response, str):
        return 0.0
    answer = task["solution"]
    return 1.0 if str(answer).strip() in response else 0.0
'''

TASK_ONLY_BRANCH = '''
"""RL0203: a score branch decided by the task alone, and a branch no input can reach."""
from typing import Any


def score(task: Any, response: Any) -> float:
    if not isinstance(task, dict):
        return 0.0
    if task.get("difficulty") == "easy":
        return 1.0
    if False:
        return 0.5
    return 0.0 if not isinstance(response, str) else float(len(response) > 0)
'''

NONE_ON_MALFORMED = '''
"""The input-handling check: None on anything it does not recognise (D-65's first mode)."""
from typing import Any


def score(task: Any, response: Any) -> float | None:
    if not isinstance(task, dict) or not isinstance(response, str):
        return None
    expected = task.get("expected")
    if not isinstance(expected, str) or not expected.strip():
        return None
    return 1.0 if expected in response else 0.0
'''

NEEDS_NUMPY = '''
"""A grader whose module imports a distribution the base closure does not carry."""
import numpy as np
from typing import Any


def score(task: Any, response: Any) -> float:
    return float(np.mean([len(str(response))]))
'''


def fixture(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source.lstrip("\n"), encoding="utf-8")
    return path


def request_for(path: Path) -> api.AuditRequest:
    return api.AuditRequest(path=path)


def audit_of(path: Path, *, project: Project | None = None) -> contracts.Assay:
    return audit.run(request_for(path), project=project, sandbox=default_sandbox())


def entries(record: contracts.Assay) -> list[contracts.Entry]:
    out: list[contracts.Entry] = []
    for section in contracts.models.Measurement.model_fields:
        out.extend(getattr(record.measurement, section))
    return out


def by_id(record: contracts.Assay, entry_id: str) -> contracts.Entry:
    for entry in entries(record):
        if entry.entry_id == entry_id:
            return entry
    raise AssertionError(f"no entry {entry_id!r}; have {[e.entry_id for e in entries(record)]}")


def codes(record: contracts.Assay) -> set[str]:
    return {finding.code for finding in record.findings if finding.code}


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("demo") / "demo"
    write_example(dest)
    return dest


@pytest.fixture(scope="module")
def demo_record(demo: Path) -> contracts.Assay:
    return audit_of(demo, project=Project.open(demo))


# --- the demo audit ------------------------------------------------------------------------------


def test_demo_record_validates(demo_record: contracts.Assay) -> None:
    contracts.validate_record(demo_record.to_dict())


def test_demo_record_has_a_witness_and_five_absences(demo_record: contracts.Assay) -> None:
    kinds = [entry.kind for entry in entries(demo_record)]
    assert kinds.count("witness") >= 1
    assert kinds.count("absence") >= 5


def test_holes_equal_the_absence_entries(demo_record: contracts.Assay) -> None:
    absences = {entry.entry_id for entry in entries(demo_record) if entry.kind == "absence"}
    assert {hole.entry_id for hole in demo_record.holes} == absences
    assert absences


def test_every_panel_runs_or_writes_its_absence(demo_record: contracts.Assay) -> None:
    for section in contracts.models.Measurement.model_fields:
        assert getattr(demo_record.measurement, section), f"{section} is empty"


def test_the_demo_defect_is_found_with_a_witness(demo_record: contracts.Assay) -> None:
    entry = by_id(demo_record, "validity.writable_then_read")
    assert entry.kind == "witness"
    assert entry.witness is not None
    assert entry.witness.path and entry.witness.path.endswith("grader.py")
    assert entry.witness.artifact_identity and entry.witness.artifact_identity.startswith("sha256:")
    assert "RL0201" in codes(demo_record)
    finding = next(f for f in demo_record.findings if f.code == "RL0201")
    assert finding.id == "RGX-local-0001"
    assert finding.arm == "static"
    assert finding.location is not None and finding.location.line >= 1


TWO_CROSSINGS = """
import subprocess


def score(task, response):
    subprocess.run(["python", "solution.py"], check=False)
    expected = open("outcome/test_solution.py").read()
    hints = open("fixtures/hints.txt").read()
    return 1.0 if expected and hints else 0.0
"""


def test_the_reach_panel_counts_the_crossings_the_isolation_witness_measured(
    demo_record: contracts.Assay,
) -> None:
    """An exposure is a crossing, not a call site, and the two panels read one set of facts.

    The demo's source has twelve access sites and one of them is a path the graded process can
    write and the grader then reads. The panel used to report the twelve as `surfaces exposed`
    beside `crossings: 0`, in the same record as an isolation witness that had measured the
    crossing; the census of call sites is still there, under `by_kind`, where it is a fact about
    the source rather than a claim about reach.
    """
    entry = by_id(demo_record, "reach.attack_surface")
    isolation = [f for f in demo_record.findings if f.entries == ["validity.writable_then_read"]]
    assert [f.code for f in isolation] == ["RL0201"]

    assert entry.result["crossings"] == 1
    assert entry.result["summary"] == "1 surface exposed, by static analysis only"
    assert entry.result["surfaces_exposed"] == entry.n == 1
    assert entry.result["exposure_id"] == "REACH-EDIT-TESTS"
    assert entry.result["note"] == "no executed witness: not in this build"
    assert entry.result["by_kind"] == {"execute": 5, "read": 3, "write": 4}
    assert entry.result["accesses"] == 12


def test_two_crossings_read_as_two_surfaces_exposed(tmp_path: Path) -> None:
    """The count is measured, not the word `1` written into the sentence."""
    grader = fixture(tmp_path, "two.py", TWO_CROSSINGS)
    tasks = fixture(tmp_path, "tasks.jsonl", '{"id": "t1", "prompt": "p"}\n')
    record = audit.run(
        api.AuditRequest(path=grader, tasks=tasks), project=None, sandbox=default_sandbox()
    )
    entry = by_id(record, "reach.attack_surface")
    assert entry.result["crossings"] == 2
    assert entry.result["summary"] == "2 surfaces exposed, by static analysis only"
    assert entry.result["exposure_id"] == "REACH-EDIT-TESTS"


def test_reach_is_static_and_never_an_executed_claim(demo_record: contracts.Assay) -> None:
    entry = by_id(demo_record, "reach.attack_surface")
    assert entry.state == "partial"
    assert "static: no executed witness" in entry.limitations
    assert entry.kind != "witness"


def test_decision_coverage_is_an_estimate_without_invented_endpoints(
    demo_record: contracts.Assay,
) -> None:
    entry = by_id(demo_record, "validity.decision_coverage")
    assert entry.kind == "estimate"
    assert entry.value is not None and entry.unit and entry.n and entry.sampling_unit
    assert entry.uncertainty is not None
    if entry.uncertainty.method == "no_interval_below_15_clusters":
        assert entry.uncertainty.interval is None
        assert entry.uncertainty.bounded is False


def test_replay_determinism_ran(demo_record: contracts.Assay) -> None:
    entry = by_id(demo_record, "validity.replay_determinism")
    assert entry.kind in {"check", "estimate"}
    assert entry.state in {"complete", "partial"}


def test_outcome_is_unqualified_and_the_decision_is_unresolved(
    demo_record: contracts.Assay,
) -> None:
    """The unqualified outcome is a hole and a decision reason, and not a finding (A-025 point 2).

    It used to be all three. A finding is something the run measured about the grader, and this is
    something the run did not measure: a reader scanning the findings for defects found a hole
    sitting among them, and the report's count of findings disagreed with the list printed under
    it. The record still says it twice, where it belongs, as the absence and as the reason.
    """
    assert demo_record.intent.outcome_check.state == "unqualified"
    assert demo_record.decision.state == "unresolved"
    assert "RL0301" not in {finding.code for finding in demo_record.findings}
    assert by_id(demo_record, "soundness.instrument_absent").kind == "absence"
    assert "outcome_unqualified:protected_test_suite" in demo_record.decision.reasons
    assert [finding.level for finding in demo_record.findings] == ["error"]


def test_the_record_and_the_report_were_written(demo: Path, demo_record: contracts.Assay) -> None:
    written = sorted((demo / "assays").glob("*.assay.json"))
    assert written, "no record written through the store"
    assert json.loads(written[0].read_text(encoding="utf-8"))["assay_id"] == demo_record.assay_id
    assert sorted((demo / "assays").glob("*.assay.html"))


def test_no_panel_prints_a_number_it_did_not_compute(demo_record: contracts.Assay) -> None:
    for entry in entries(demo_record):
        if entry.kind == "absence":
            assert entry.value is None and entry.result is None
        if entry.absence is not None:
            assert entry.absence.missing_access and entry.absence.remedy


def test_the_sdk_seam_reaches_this_engine(tmp_path: Path) -> None:
    own = tmp_path / "demo"
    write_example(own)
    record = api.audit(api.AuditRequest(path=own, name="via-sdk"))
    assert isinstance(record, contracts.Assay)
    assert any(entry.kind == "witness" for entry in entries(record))


# --- the bare grader -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bare(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import reward_lens.examples.code_reward as package

    dest = tmp_path_factory.mktemp("bare")
    shipped = Path(package.__file__).resolve().parent / "fixtures" / "my_grader.py"
    source = shipped.read_text(encoding="utf-8")
    target = dest / "my_grader.py"
    target.write_text(source, encoding="utf-8")
    return target


def test_bare_grader_audit_is_a_record(bare: Path) -> None:
    record = audit_of(bare)
    contracts.validate_record(record.to_dict())
    assert record.subject.reward_system.name == "my_grader.py"
    assert any(entry.kind == "absence" for entry in entries(record))


def test_bare_grader_names_its_input_handling_defect(bare: Path) -> None:
    record = audit_of(bare)
    entry = by_id(record, "validity.input_handling")
    assert entry.check is not None and entry.check.passed is False
    assert "RL0210" in codes(record)


def test_bare_grader_task_set_checks_say_they_need_a_task_set(bare: Path) -> None:
    record = audit_of(bare)
    for entry_id in ("validity.task_validity", "validity.pass_rate_floor", "validity.flakiness"):
        entry = by_id(record, entry_id)
        assert entry.kind == "absence"
        assert "task set" in entry.absence.missing_access


# --- the four static checks, each on a fixture built to violate it -------------------------------


def test_rl0201_fires_on_its_fixture(tmp_path: Path) -> None:
    record = audit_of(fixture(tmp_path, "writes_then_reads.py", WRITES_THEN_READS))
    assert "RL0201" in codes(record)
    assert by_id(record, "validity.writable_then_read").kind == "witness"


def test_rl0202_fires_on_its_fixture(tmp_path: Path) -> None:
    record = audit_of(fixture(tmp_path, "leaky_task.py", LEAKY_TASK))
    assert "RL0202" in codes(record)
    entry = by_id(record, "validity.input_leakage")
    assert entry.check is None or entry.check.passed is False


def test_rl0203_fires_on_its_fixture(tmp_path: Path) -> None:
    record = audit_of(fixture(tmp_path, "task_only_branch.py", TASK_ONLY_BRANCH))
    assert "RL0203" in codes(record)
    entry = by_id(record, "validity.unreachable_score_branch")
    assert entry.check is None or entry.check.passed is False


def test_input_handling_fires_on_its_fixture(tmp_path: Path) -> None:
    record = audit_of(fixture(tmp_path, "none_on_malformed.py", NONE_ON_MALFORMED))
    assert "RL0210" in codes(record)
    entry = by_id(record, "validity.input_handling")
    assert entry.check is not None and entry.check.passed is False
    assert any("trainer" in note for note in entry.limitations + (entry.assumptions or []))


def test_the_clean_fixture_passes_all_four(tmp_path: Path) -> None:
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    assert codes(record) & {"RL0201", "RL0202", "RL0203", "RL0210", "RL0211", "RL0212"} == set()
    for entry_id in (
        "validity.writable_then_read",
        "validity.input_leakage",
        "validity.unreachable_score_branch",
        "validity.input_handling",
    ):
        assert by_id(record, entry_id).check.passed is True


# --- the plan ------------------------------------------------------------------------------------


def test_plan_prices_zero_paid_calls(demo: Path) -> None:
    plan = audit.plan(request_for(demo))
    assert plan.paid_calls == 0
    assert plan.estimate_usd == "0.00"
    assert "validity" in plan.panels and "reach" in plan.panels


def test_the_plan_estimates_the_seconds_over_the_panels_that_will_run(demo: Path) -> None:
    """A-016: `estimate_s` is a sum over the panels, so it moves when the panels do."""
    from reward_lens.product.audit.plan import PANEL_SECONDS, estimate_s

    plan = audit.plan(request_for(demo))
    assert plan.estimate_s == sum(PANEL_SECONDS[section] for section in plan.panels)
    assert plan.estimate_s == estimate_s(plan.panels) > 0
    # Not a literal: one panel fewer is fewer seconds, which a written-down 30 could not be.
    assert estimate_s(plan.panels[:1]) < plan.estimate_s
    assert estimate_s(()) == 0


# --- the refusal cases ---------------------------------------------------------------------------


def test_outcome_unqualified_is_the_state_when_no_independent_check_is_supplied(
    tmp_path: Path,
) -> None:
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    assert record.intent.outcome_check.state == "unqualified"
    assert "RL0301" not in {f.code for f in record.findings}
    assert by_id(record, "soundness.instrument_absent").kind == "absence"
    assert any(
        reason.startswith("outcome_unqualified:") for reason in record.decision.reasons
    ), record.decision.reasons
    assert record.decision.state == "unresolved"


def test_an_instrument_refusal_becomes_an_absence_and_never_a_number(tmp_path: Path) -> None:
    """An empty corpus makes `measure_coverage` refuse. The refusal is a value, so it is a hole."""
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    entry = by_id(record, "validity.decision_coverage")
    assert entry.kind == "absence"
    assert entry.value is None
    assert entry.absence.state in {"REFUSED", "NOT_MEASURED"}
    assert entry.absence.missing_access
    assert any(hole.entry_id == entry.entry_id for hole in record.holes)


def test_a_grader_file_that_does_not_exist_is_rl0001(tmp_path: Path) -> None:
    with pytest.raises(RewardLensError) as raised:
        audit_of(tmp_path / "absent_grader.py")
    assert raised.value.code == "RL0001"
    assert "absent_grader.py" in str(raised.value.context) + raised.value.message


def test_an_import_the_base_closure_lacks_is_a_could_not_check_hole(tmp_path: Path) -> None:
    """P-PKG's required hole: `verifier.load()` raises before any Reading exists."""
    pytest.importorskip  # noqa: B018 - numpy may be installed; the hole is asserted either way
    record = audit_of(fixture(tmp_path, "needs_numpy.py", NEEDS_NUMPY))
    contracts.validate_record(record.to_dict())
    try:
        import numpy  # noqa: F401
    except ImportError:
        holes = [hole for hole in record.holes if hole.state == "COULD_NOT_CHECK"]
        assert holes, "an import the closure lacks must be a COULD_NOT_CHECK hole"
        assert any("numpy" in hole.missing_access for hole in holes)
        assert all("inconclusive" not in hole.missing_access.lower() for hole in holes)
    # the other panels ran either way
    assert by_id(record, "validity.writable_then_read").check is not None
    assert by_id(record, "reach.attack_surface") is not None


def test_every_hole_from_a_failed_import_names_the_module_and_the_exception(
    tmp_path: Path,
) -> None:
    """A hole that says only that something failed cannot be acted on.

    The reader needs the module to install and the exception to recognise, on every entry that
    needed the grader rather than on whichever one reached it first. The static checks, which read
    the source and never import it, still run and still pass.
    """
    pytest.importorskip("reward_lens")  # the hole is about the grader's imports, not ours
    missing_module = "reward_lens_absent_dependency"
    source = NEEDS_NUMPY.replace("numpy as np", f"{missing_module} as np")
    record = audit_of(fixture(tmp_path, "needs_absent.py", source))
    contracts.validate_record(record.to_dict())

    holes = [hole for hole in record.holes if hole.state == "COULD_NOT_CHECK"]
    assert holes, "an import this environment lacks is a COULD_NOT_CHECK hole"
    for hole in holes:
        assert missing_module in hole.missing_access, hole.missing_access
        assert "ModuleNotFoundError" in hole.missing_access, hole.missing_access
        assert "pip install" in hole.missing_access, hole.missing_access
        assert "pip install" in hole.remedy, hole.remedy

    # every entry that needed the grader, not only the first of them
    needed_the_grader = {
        "validity.input_handling",
        "validity.replay_determinism",
        "validity.decision_coverage",
    }
    held = {hole.entry_id for hole in holes}
    assert needed_the_grader <= held, sorted(held)

    # and the entries that did not need it still ran
    # `reach.attack_surface` is not in this list any more: it reads the source and needs no
    # import, but with no task set the reach panel is an absence for that reason instead (A-015),
    # and an absence written for the wrong reason is what this test exists to catch.
    for entry_id in ("validity.writable_then_read", "validity.input_leakage"):
        entry = by_id(record, entry_id)
        assert entry.kind != "absence", f"{entry_id} needs no import and must still run"


def test_a_declared_extra_is_named_as_the_command_that_closes_the_hole() -> None:
    """A module a declared extra carries is remediated by that extra, not by a bare module name.

    Asserted against the diagnosis rather than against a run, so the statement holds whether or not
    this environment happens to have numpy: the mapping is read out of the installed distribution's
    metadata, which is the same in both cases.
    """
    from reward_lens.product.audit.absences import import_diagnosis

    module, command = import_diagnosis(ModuleNotFoundError("No module named 'numpy'", name="numpy"))
    assert module == "numpy"
    assert command is not None and command.startswith("pip install 'reward-lens[")
    assert "verifier" in command  # numpy is carried by more than one extra, and all are named

    other, bare = import_diagnosis(
        ModuleNotFoundError("No module named 'nothing_declares_me'", name="nothing_declares_me")
    )
    assert (other, bare) == ("nothing_declares_me", "pip install nothing_declares_me")

    assert import_diagnosis(ValueError("not an import at all")) == (None, None)


def test_the_blocking_finding_is_the_first_reason_the_decision_gives(
    demo_record: contracts.Assay,
) -> None:
    """The demo exits non-zero on a finding at level error, so the decision names that finding."""
    blocking = [finding for finding in demo_record.findings if finding.level == "error"]
    assert blocking, "the demo fixture is the one with a blocking finding"
    reasons = demo_record.decision.reasons
    assert reasons[0] == "blocking_finding:RGX-local-0001", reasons
    # One finding is one reason, named by its id, which is the form the envelope golden pins. The
    # rule is the stabler name and is one field away in the record's own findings, which carry
    # both; the report's verdict prints the count and not the name either way.
    for finding in blocking:
        assert f"blocking_finding:{finding.id}" in reasons, reasons
        assert f"blocking_finding:{finding.rule}" not in reasons, reasons
    first_other = next(
        index for index, reason in enumerate(reasons) if not reason.startswith("blocking_finding:")
    )
    assert all(
        reason.startswith("blocking_finding:") for reason in reasons[:first_other]
    ), reasons
    assert "outcome_unqualified:protected_test_suite" in reasons
    assert any(reason.startswith("required_missing:") for reason in reasons)


def test_the_rule_names_the_arm_and_the_entry_id_does_not_move(
    demo_record: contracts.Assay,
) -> None:
    """A-025 point 2: the finding's rule is `validity.static.writable_then_read`.

    A measurand can be asked by more than one arm. Reading the grader's source and watching a
    sampled response take the path are two claims about the same hole, and a rule that named only
    the measurand could not tell them apart: a reader told the path is writable wants to know
    whether that was read off the source or seen happening. The entry id stays put, because it is
    where the record keeps the hole and moving it would move the hole.
    """
    finding = next(f for f in demo_record.findings if f.code == "RL0201")
    assert finding.rule == "validity.static.writable_then_read"
    assert finding.entries == ["validity.writable_then_read"]
    assert finding.arm == "static"
    assert by_id(demo_record, "validity.writable_then_read").kind == "witness"
    assert static_checks.CHECK_IDS["validity.static.writable_then_read"] == "RL0201"


def test_the_reasons_are_in_the_goldens_order(demo_record: contracts.Assay) -> None:
    """A-025 point 1: blocking findings, then the outcome, then the sections, as printed.

    The reasons used to run findings, sections, outcome, and the sections in the record's field
    order, which puts reward_statistics before signal. The report prints signal first and the
    envelope is the report's projection, so a reader comparing the two found the same eight names
    in two orders and had no way to tell which was the list.
    """
    reasons = demo_record.decision.reasons
    kinds = [reason.split(":", 1)[0] for reason in reasons]
    assert kinds == ["blocking_finding"] + ["outcome_unqualified"] + ["required_missing"] * 8
    assert reasons[1] == "outcome_unqualified:protected_test_suite"
    assert [reason.split(":", 1)[1] for reason in reasons[2:]] == [
        "soundness",
        "exploits",
        "framing",
        "signal",
        "reward_statistics",
        "trace",
        "forecast",
        "calibration",
    ]


def test_the_written_record_is_the_returned_record_byte_for_byte(tmp_path: Path) -> None:
    """One serialisation: what the pipeline writes is what the SDK hands back."""
    from reward_lens.product.audit.run import record_bytes

    grader = fixture(tmp_path, "clean.py", CLEAN)
    record = audit_of(grader)
    written = sorted((grader.parent / "assays").glob("*.assay.json"))
    assert len(written) == 1, [path.name for path in written]
    on_disk = written[0].read_bytes()
    assert on_disk == record_bytes(record)
    assert json.loads(on_disk) == record.to_dict()
    assert json.loads(on_disk)["assay_id"] == record.assay_id
    assert "$schema" in json.loads(on_disk)


def test_a_project_writes_the_same_serialisation_the_pipeline_does(tmp_path: Path) -> None:
    """The two write paths are one: the store's bytes and the pipeline's bytes are the same rule."""
    from reward_lens.product.audit.run import record_bytes

    root = tmp_path / "proj"
    write_example(root)
    project = Project.open(root)
    record = audit_of(root, project=project)
    written = sorted(root.rglob("*.assay.json"))
    assert len(written) == 1, [path.name for path in written]
    assert written[0].read_bytes() == record_bytes(record)


# --- the panels a wave-2 build plugs in -----------------------------------------------------------


class FakeInstrument:
    """Stands in for a wave-2 instrument. It returns `count` entries, which may be none of them."""

    def __init__(self, instrument_id: str, section: str, count: int) -> None:
        self.id = instrument_id
        self.section = section
        self.count = count

    def run(self, subject, corpus, ctx):  # noqa: ANN001, ANN201
        subject_ref = contracts.digest({"source": subject.source})
        return [
            ctx.absence(
                self.section,
                f"{self.id}.{n}",
                f"what {self.id} measured, part {n}",
                "nothing: this instrument stands in for a wave-2 panel",
                "land the real panel",
                (f"no {self.section} claim from {self.id}",),
                subject_ref=subject_ref,
            )
            for n in range(self.count)
        ]


def discovering(*panels: Panel):
    """`registry.discover` returning exactly these panels, as a landed wave-2 build would."""

    def discover(package: str = registry.PACKAGE) -> tuple[Panel, ...]:
        registry.import_failures.clear()
        return tuple(panels)

    return discover


def test_every_entry_an_instrument_returns_is_kept(tmp_path: Path, monkeypatch) -> None:
    """Three entries in, three entries in the record. The runner kept the first one only."""
    panel = Panel(section="soundness", instruments=(FakeInstrument("soundness.battery", "soundness", 3),))
    monkeypatch.setattr(registry, "discover", discovering(panel))
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    contracts.validate_record(record.to_dict())
    assert [entry.entry_id for entry in record.measurement.soundness] == [
        "soundness.battery.0",
        "soundness.battery.1",
        "soundness.battery.2",
    ]


def test_an_instrument_that_returns_nothing_is_an_absence_naming_it(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty list is an instrument that measured nothing, not an IndexError about the grader."""
    panel = Panel(section="framing", instruments=(FakeInstrument("framing.context_leak", "framing", 0),))
    monkeypatch.setattr(registry, "discover", discovering(panel))
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    contracts.validate_record(record.to_dict())
    entry = by_id(record, "framing.context_leak")
    assert entry.kind == "absence"
    assert entry.value is None
    assert "framing.context_leak" in entry.absence.missing_access
    assert "could not be loaded" not in entry.absence.missing_access
    assert [e.entry_id for e in record.measurement.framing] == ["framing.context_leak"]


def test_a_discovered_panel_drives_the_plan_and_loses_its_absence(
    tmp_path: Path, monkeypatch
) -> None:
    """Discovery names the panels; a landed panel never gets both its entries and NOT_MEASURED."""
    panel = Panel(section="soundness", instruments=(FakeInstrument("soundness.battery", "soundness", 2),))
    monkeypatch.setattr(registry, "discover", discovering(panel))
    grader = fixture(tmp_path, "clean.py", CLEAN)

    planned = audit.plan(request_for(grader))
    filled = {"soundness", *BUILTIN_PANELS}
    assert planned.panels == tuple(section for section in SECTIONS if section in filled)
    assert "exploits" not in planned.panels

    record = audit_of(grader)
    soundness = [entry.entry_id for entry in record.measurement.soundness]
    assert soundness == ["soundness.battery.0", "soundness.battery.1"]
    assert "soundness.instrument_absent" not in soundness
    # every panel this build does not fill still says what it needed
    assert by_id(record, "exploits.instrument_absent").kind == "absence"
    assert by_id(record, "signal.instrument_absent").kind == "absence"
    assert by_id(record, "framing.instrument_absent").kind == "absence"


def test_a_panel_naming_a_section_the_record_does_not_have_is_a_hole(
    tmp_path: Path, monkeypatch
) -> None:
    """A section the record does not have is a plugin defect, recorded, never a KeyError crash."""
    panel = Panel(section="soundnes", instruments=(FakeInstrument("soundnes.typo", "soundnes", 1),))
    monkeypatch.setattr(registry, "discover", discovering(panel))
    record = audit_of(fixture(tmp_path, "clean.py", CLEAN))
    contracts.validate_record(record.to_dict())
    entry = by_id(record, "validity.panel_section")
    assert entry.absence.state == "COULD_NOT_CHECK"
    assert "soundnes" in entry.absence.missing_access
    assert by_id(record, "soundness.instrument_absent").kind == "absence"


# --- the subject's name is total ------------------------------------------------------------------


def embedded_record(page: bytes) -> dict:
    """The record the page carries, lifted out of the one JSON block the page holds."""
    text = page.decode("utf-8")
    opening = text.index('id="assay"')
    start = text.index(">", opening) + 1
    return json.loads(text[start : text.index("</script>", start)])


def test_a_project_opened_as_a_dot_still_names_its_reward_system(
    tmp_path: Path, monkeypatch
) -> None:
    """The whole flow, not a unit: `reward-lens audit` with no path used to be an RL0900.

    A project is opened at the root it was handed, so `Path(".").name` was the empty string and
    `RewardSystemRef.name` refused it with `string_too_short`. A directory name with letters,
    digits and a dot survives the cleaning intact.
    """
    root = tmp_path / "proj.42"
    write_example(root)
    monkeypatch.chdir(root)
    record = api.audit(api.AuditRequest(path="."))
    assert isinstance(record, contracts.Assay)
    contracts.validate_record(record.to_dict())


def test_the_declared_name_wins_and_the_directory_is_only_the_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """A-015: the reward system's name is what the project declares, if it declares one.

    Two projects in identically named directories, one declaring `name:` and one not. The declared
    name is the name; the directory's, resolved and then cleaned, is what answers only when there
    is nothing to declare. Both are audited from inside with no path, which is the case that used
    to have no directory name at all (`Path(".").name` is empty).
    """
    declared = tmp_path / "declared" / "proj.42"
    write_example(declared)
    monkeypatch.chdir(declared)
    assert api.audit(api.AuditRequest(path=".")).subject.reward_system.name == "code-reward"

    undeclared = tmp_path / "undeclared" / "proj.42"
    write_example(undeclared)
    config = undeclared / "rewardlens.yaml"
    config.write_text(
        "\n".join(
            line
            for line in config.read_text(encoding="utf-8").splitlines()
            if not line.startswith("name:")
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(undeclared)
    assert api.audit(api.AuditRequest(path=".")).subject.reward_system.name == "proj.42"


def test_the_cleaning_is_total_and_ends_on_a_stated_default() -> None:
    """Every string reaches a name `RewardSystemRef` accepts; none of them is the empty string."""
    assert _system_name(None, Path("/tmp/a b/c d")) == "c-d"
    assert _system_name(None, Path("/")) == DEFAULT_SYSTEM_NAME
    assert _system_name("Reward Model v2", Path("/tmp/ignored")) == "Reward-Model-v2"
    assert _system_name("---", Path("/")) == DEFAULT_SYSTEM_NAME


def test_the_same_project_audited_three_ways_is_one_reward_system(
    tmp_path: Path, monkeypatch
) -> None:
    """From inside with no path, by a relative path, and by an absolute one: one subject.

    Three copies rather than three audits of one, because the store refuses to record a subject
    version it has already recorded (RL0620).
    """
    roots = []
    for index in range(3):
        root = tmp_path / f"copy{index}" / "project"
        write_example(root)
        roots.append(root)

    monkeypatch.chdir(roots[0])
    inside = api.audit(api.AuditRequest(path="."))
    monkeypatch.chdir(roots[1].parent)
    relative = api.audit(api.AuditRequest(path="./project"))
    monkeypatch.chdir(tmp_path)
    absolute = api.audit(api.AuditRequest(path=roots[2].resolve()))
    three = (inside, relative, absolute)

    # A-015: the declared name, not the directory, so three copies in three directories are one
    # reward system rather than three named after where they happen to sit.
    assert {record.subject.reward_system.name for record in three} == {"code-reward"}
    assert {record.subject.reward_system.id for record in three} == {"code-reward-v1"}
    assert len({record.subject.version.digest for record in three}) == 1
    assert len({record.embedding.tier for record in three}) == 1
    assert len({tuple(entry.entry_id for entry in entries(record)) for record in three}) == 1

    # The three ids differ, and not only because of the path: `created` and every entry's
    # `provenance.started` are wall-clock, so two audits of one directory differ too. The path
    # itself reaches the record in exactly one field, the command that reproduces the run.
    assert len({record.assay_id for record in three}) == 3
    # `AuditRequest.path` normalises what it is given, so the relative way reproduces as
    # `project` rather than `./project`; the absolute way keeps its whole path.
    assert [record.provenance.reproduce[0].split()[2] for record in three] == [
        ".",
        "project",
        str(roots[2].resolve()),
    ]
    for record in three:
        carried = record.to_dict()
        carried.pop("provenance")
        assert str(tmp_path) not in json.dumps(carried)


# --- the embedding is planned before the seal -----------------------------------------------------


def test_the_record_declares_its_tier_before_it_is_sealed(tmp_path: Path) -> None:
    """D-80: one id over the record returned, the record on disk, and the record in the page.

    The plan is taken on the statement before the seal, so `assay_id` is the digest of a record
    that already declares how it will be carried. Nothing after the seal edits the record, which
    is what lets the store's acceptance check hold on the copy lifted back out of the HTML.
    """
    from reward_lens.render.report import render

    root = tmp_path / "sealed"
    write_example(root)
    record = api.audit(api.AuditRequest(path=root, name="sealed"))

    assert record.embedding.tier == "A"
    assert record.embedding.record_bytes > 0
    assert record.embedding.html_bytes > 0

    sealed = record.to_dict()
    assert contracts.digest(sealed) == record.assay_id

    on_disk = json.loads((root / "assays" / "sealed.assay.json").read_text(encoding="utf-8"))
    assert on_disk["assay_id"] == record.assay_id
    assert contracts.digest(on_disk) == record.assay_id

    # The plan is taken with the bundle manifest's digest, which the page carries in a `<meta>`,
    # so `html_bytes` is the length of the page that has it and a re-render has to be given the
    # same digest. What is being tested is unchanged: one id over the record returned, the record
    # on disk, and the record in the page, whichever page it is.
    claim = contracts.digest(manifest_beside(root / "assays" / "sealed.assay.json"))
    for page in (
        render(sealed, bundle_manifest_digest=claim),
        (root / "assays" / "sealed.assay.html").read_bytes(),
    ):
        embedded = embedded_record(page)
        assert embedded == sealed
        assert embedded["embedding"]["tier"] == "A"
        # the store's own acceptance check, on the record the page carries
        assert contracts.digest(embedded) == embedded["assay_id"] == record.assay_id
        assert meta_digest(page) == claim


# --- the bundle manifest, and the page's claim to it -----------------------------------------------


def manifest_beside(record_path: Path) -> dict:
    """The bundle manifest that names `record_path`, loaded from beside it."""
    return json.loads(
        record_path.with_name(MANIFEST_NAME).read_text(encoding="utf-8")
    )


def meta_digest(page: bytes) -> str:
    """The one bundle digest the page claims, read off the `<meta>` rather than the record."""
    found = re.findall(rb'<meta name="bundle-manifest-digest" content="([^"]*)">', page)
    assert len(found) == 1, f"expected one bundle-manifest-digest meta, found {len(found)}"
    return found[0].decode("utf-8")


def test_the_page_names_the_bundle_it_was_rendered_from(tmp_path: Path) -> None:
    """D-16: the page's `<meta>` is the digest of the manifest lying beside the record.

    This is the whole point of the manifest. A reader holding a report and a directory can decide
    whether the two belong together without trusting either: digest what is on disk, compare it
    with what the page claims, and a page from a different run does not match.
    """
    root = tmp_path / "claimed"
    write_example(root)
    record = api.audit(api.AuditRequest(path=root, name="claimed"))

    record_path = root / "assays" / "claimed.assay.json"
    page = record_path.with_suffix(".html").read_bytes()
    assert record_path.with_name(MANIFEST_NAME).exists()
    assert meta_digest(page) == contracts.digest(manifest_beside(record_path))
    assert record.assay_id == manifest_beside(record_path)["assay"]["id"]


def test_the_manifest_names_the_record_file_with_the_digest_of_its_bytes(
    tmp_path: Path,
) -> None:
    """The manifest's one artifact is the record: its path beside the manifest, and its sha256.

    The path is relative, so the check survives the bundle being copied somewhere else, and the
    digest is taken over the file as written rather than over the record in memory, because the
    file is what a reader has.
    """
    root = tmp_path / "named"
    write_example(root)
    record = api.audit(api.AuditRequest(path=root, name="named"))

    record_path = root / "assays" / "named.assay.json"
    manifest = manifest_beside(record_path)
    payload = record_path.read_bytes()

    assert [entry["path"] for entry in manifest["artifacts"]] == ["named.assay.json"]
    entry = manifest["artifacts"][0]
    assert entry["role"] == "record"
    assert entry["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert entry["bytes"] == len(payload)
    assert json.loads(payload)["assay_id"] == record.assay_id
    assert manifest["tool"]["name"] == "reward-lens" and manifest["tool"]["version"]


def test_the_manifest_names_the_bundle_and_not_the_directory(tmp_path: Path) -> None:
    """Three files in the directory are not in the manifest, each for its own reason.

    The report carries the manifest's digest and so cannot be digested by it. The manifest cannot
    name itself. And the store's `index.json` is the project's, not this bundle's: it spans every
    assay the project holds and is rewritten by the next audit, so a manifest that named it would
    stop describing what is on disk as soon as one more audit ran, which is the opposite of what a
    digest the page carries is for.
    """
    root = tmp_path / "excluded"
    write_example(root)
    api.audit(api.AuditRequest(path=root, name="excluded"))

    directory = root / "assays"
    manifest = manifest_beside(directory / "excluded.assay.json")
    named = {entry["path"] for entry in manifest["artifacts"]}
    written = {path.name for path in directory.iterdir()}

    assert named == {"excluded.assay.json"}
    assert written - named == {MANIFEST_NAME, "excluded.assay.html", "index.json"}


def test_the_fixed_manifest_name_means_the_latest_bundle_is_the_one_on_disk(
    tmp_path: Path,
) -> None:
    """The known cost of calling the file `manifest.json`, written down rather than discovered.

    Two audits of one subject version cannot both land: the store refuses the second as a rewrite.
    Two audits of two versions can, and then they share a directory and the second manifest
    replaces the first. The live report and the manifest agree; the superseded report claims a
    digest nothing on disk has, which is detectably superseded rather than quietly wrong, but it is
    a superseded report all the same. Naming the file `<name>.manifest.json` would remove the
    collision; the fixed name is what the packet was given.
    """
    root = tmp_path / "twice"
    write_example(root)
    first = api.audit(api.AuditRequest(path=root, name="first"))

    grader = root / "grader.py"
    grader.write_text(grader.read_text(encoding="utf-8") + "\n# a changed subject\n", "utf-8")
    second = api.audit(api.AuditRequest(path=root, name="second"))

    directory = root / "assays"
    manifest = manifest_beside(directory / "second.assay.json")
    assert manifest["assay"]["id"] == second.assay_id != first.assay_id
    assert manifest["artifacts"][0]["path"] == "second.assay.json"

    claim = contracts.digest(manifest)
    assert meta_digest((directory / "second.assay.html").read_bytes()) == claim
    assert meta_digest((directory / "first.assay.html").read_bytes()) != claim


def test_no_manifest_key_is_stripped_before_the_manifest_is_digested(tmp_path: Path) -> None:
    """`contracts.digest` drops `EXCLUDED_FROM_DIGEST` from whatever it hashes, manifests too.

    Those names exist for records, where the id and the attestation are made over the digest and
    so cannot be inside it. A manifest that reused one of them at the top level would put that
    field outside its own digest without saying so, which is why the assay id is nested.
    """
    root = tmp_path / "unstripped"
    write_example(root)
    api.audit(api.AuditRequest(path=root, name="unstripped"))

    manifest = manifest_beside(root / "assays" / "unstripped.assay.json")
    assert set(manifest) & set(contracts.EXCLUDED_FROM_DIGEST) == set()
    assert contracts.canonical_bytes(manifest) == manifest_bytes(manifest)


def test_a_bare_grader_audit_writes_its_manifest_beside_its_record(bare: Path) -> None:
    """No project, no store, same bundle: the manifest is written on the projectless path too."""
    record = audit_of(bare)

    written = sorted((bare.parent / "assays").glob("*.assay.json"))
    assert len(written) == 1
    manifest = manifest_beside(written[0])

    assert manifest["artifacts"][0]["path"] == written[0].name
    assert manifest["artifacts"][0]["sha256"] == (
        "sha256:" + hashlib.sha256(written[0].read_bytes()).hexdigest()
    )
    assert manifest["assay"]["id"] == record.assay_id
    page = written[0].with_suffix(".html").read_bytes()
    assert meta_digest(page) == contracts.digest(manifest)


# --- one subject version, one measurement: the pipeline honours the store's rule -------------------


def test_a_second_audit_of_an_unchanged_project_returns_the_stored_record(
    tmp_path: Path,
) -> None:
    """The bug: the second audit in a day used to die on RL0620 with nothing written.

    The store's rule is right and stays. A record is a measurement of a subject version, and there
    is only one measurement of a version to be had, so a second, different record of it is refused.
    What was wrong was the pipeline running ten instruments to build a record it could not store.
    It now asks first, and an unchanged subject gets the answer that already exists: the same id,
    the same file, no second record, nothing on disk touched.
    """
    root = tmp_path / "unchanged"
    write_example(root)

    first = api.audit(api.AuditRequest(path=root))
    project = Project.open(root)
    (name,) = project.names()
    before = {path.name: path.read_bytes() for path in (root / "assays").iterdir()}

    second = api.audit(api.AuditRequest(path=root))

    assert second.assay_id == first.assay_id
    assert project.names() == (name,)
    assert {path.name: path.read_bytes() for path in (root / "assays").iterdir()} == before
    # "byte for byte": what came back is not a re-serialisation that happens to agree, it is the
    # file, so a caller holding the record and a reader holding the bundle hold the same object.
    assert record_bytes(second) == project.path_for(name).read_bytes()
    assert second.to_dict() == first.to_dict()
    assert second.decision.state == first.decision.state


def test_the_reused_run_says_it_reused_and_the_measuring_run_says_nothing(
    tmp_path: Path,
) -> None:
    """Where the reuse is reported: on the run, through `audit.last_reuse()`, not in the record.

    It cannot be in the record. The record handed back is the stored one byte for byte, and a
    measurement that also carried which of the runs that fetched it was talking would not be that
    record any more. So the run says it, `Reused` is the sentence, and the record is untouched:
    `EXCLUDED_FROM_DIGEST` is not a loophole used here, and the id is the same either way.
    """
    root = tmp_path / "reported"
    write_example(root)

    first = api.audit(api.AuditRequest(path=root))
    assert audit.last_reuse() is None, "a run that measured must not claim reuse"

    second = api.audit(api.AuditRequest(path=root))
    reused = audit.last_reuse()

    assert isinstance(reused, audit.Reused)
    (name,) = Project.open(root).names()
    assert reused.name == name
    assert reused.assay_id == second.assay_id == first.assay_id
    assert reused.subject_version == second.subject.version.digest
    assert Path(reused.record_path) == Project.open(root).path_for(name)
    assert reused.rendered_report is False and reused.wrote_manifest is False
    assert name in reused.says and "no instrument ran" in reused.says
    # Nothing about the reuse reached the record, including the part of it outside the digest.
    assert second.to_dict() == first.to_dict()
    assert "reuse" not in json.dumps(second.environment_excluded_from_digest).lower()


def test_a_changed_grader_is_a_new_record_under_a_name_that_cannot_collide(
    tmp_path: Path,
) -> None:
    """The other half: a changed subject is measured, and the date alone was not a name.

    Two versions of one subject on one day both want `<subject>-<date>`, and the store will not
    let the second have it. The name carries the eight hex that tell the two versions apart, which
    is the pipeline's rule: `Project` offers `path_for`, `names` and `record`, and none of them
    proposes a next name.
    """
    root = tmp_path / "moved"
    write_example(root)
    first = api.audit(api.AuditRequest(path=root))

    grader = root / "grader.py"
    grader.write_text(grader.read_text(encoding="utf-8") + "\n# a changed subject\n", "utf-8")
    second = api.audit(api.AuditRequest(path=root))

    project = Project.open(root)
    assert audit.last_reuse() is None
    assert second.assay_id != first.assay_id
    assert second.subject.version.digest != first.subject.version.digest
    assert len(project.names()) == 2

    # A-015: the stem is the version the project declares, and the suffix appears only on the
    # record that collided. The first audit of the day is named plainly; the second, of a version
    # the store does not hold under that name, carries the eight hex that tell the two apart.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    short = second.subject.version.digest.rsplit(":", 1)[-1][:NAME_DIGEST_CHARS]
    assert set(project.names()) == {
        f"code-reward-v1-{stamp}",
        f"code-reward-v1-{stamp}-{short}",
    }
    assert len({path.name for path in (root / "assays").glob("*.assay.json")}) == 2


def test_two_projects_with_the_same_directory_name_do_not_collide(tmp_path: Path) -> None:
    """One name, two subjects, two stores: neither audit can see or answer for the other.

    Under A-015 the two derive the same name and keep it, because the name is the declared version
    and both declare `code-reward-v1`. Nothing has to keep the names apart: a name is a name inside
    one store. What must hold is that neither store answers for the other, and it does, because
    the reuse lookup is asked of a project and only ever sees that project's own `assays/`.
    """
    roots = []
    for index, tail in enumerate(("\n# left\n", "\n# right\n")):
        root = tmp_path / f"side{index}" / "demo"
        write_example(root)
        grader = root / "grader.py"
        grader.write_text(grader.read_text(encoding="utf-8") + tail, "utf-8")
        roots.append(root)

    left = api.audit(api.AuditRequest(path=roots[0]))
    right = api.audit(api.AuditRequest(path=roots[1]))

    assert audit.last_reuse() is None, "the second project is not a rerun of the first"
    assert left.assay_id != right.assay_id
    assert left.subject.version.digest != right.subject.version.digest

    names = [Project.open(root).names() for root in roots]
    assert [len(group) for group in names] == [1, 1]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for root, (name,) in zip(roots, names):
        assert name == f"code-reward-v1-{stamp}"
        assert (root / "assays" / f"{name}.assay.json").is_file()
    # The same name, and not the same record: the file each store holds under it is its own.
    assert (roots[0] / "assays" / f"{names[0][0]}.assay.json").read_bytes() != (
        roots[1] / "assays" / f"{names[1][0]}.assay.json"
    ).read_bytes()

    # And each project reuses its own record rather than the other's, which is the collision that
    # would matter: the same stem answering for a different subject.
    again = api.audit(api.AuditRequest(path=roots[0]))
    assert again.assay_id == left.assay_id
    assert audit.last_reuse().name == names[0][0]


def test_a_reused_bundle_that_lost_its_report_gets_the_same_report_back(
    tmp_path: Path,
) -> None:
    """Re-rendered only if absent, and what comes back is the same report of the same record.

    The report and the manifest are functions of the record and of nothing else, so a bundle that
    lost one can have it back without the record being measured or written again. The run says
    which of the two it composed, so a caller is never guessing.

    The manifest comes back byte for byte. The page comes back as the same page of the same record
    claiming the same bundle, but not byte for byte, and the reason is worth writing down rather
    than asserting around: `render` embeds the record by serialising the object it was handed, so
    the first page follows the order the pipeline built the record in and this one follows the
    canonical order it was stored in. Same keys, same values, same length, different order. That
    is `render`'s to fix, not this packet's, and the handoff proposes it.
    """
    root = tmp_path / "lost"
    write_example(root)
    api.audit(api.AuditRequest(path=root))

    project = Project.open(root)
    (name,) = project.names()
    record_path = project.path_for(name)
    page_path = record_path.with_name(f"{name}.assay.html")
    manifest_path = record_path.with_name(MANIFEST_NAME)
    page, manifest = page_path.read_bytes(), manifest_path.read_bytes()

    page_path.unlink()
    manifest_path.unlink()
    api.audit(api.AuditRequest(path=root))
    reused = audit.last_reuse()

    assert reused is not None
    assert reused.rendered_report is True and reused.wrote_manifest is True
    assert manifest_path.read_bytes() == manifest
    again = page_path.read_bytes()
    assert embedded_record(again) == embedded_record(page)
    assert len(again) == len(page)
    assert meta_digest(again) == meta_digest(page)
    assert meta_digest(again) == contracts.digest(manifest_beside(record_path))


def test_a_file_in_assays_that_is_not_a_record_is_stepped_over_not_raised_on(
    tmp_path: Path,
) -> None:
    """The reuse lookup shrugs at a stray file, because the write path already shrugs at one.

    `_check_not_a_rewrite` steps over anything in `assays/` it cannot read as a record, so a
    lookup that raised on the same file would make an audit fail on something the write would
    have ignored. `open_record` refuses a decoy with RL0604 and the lookup moves on; the real
    record is still found, and the second audit still reuses it.
    """
    root = tmp_path / "decoyed"
    write_example(root)
    (root / "assays").mkdir(parents=True, exist_ok=True)
    decoy = root / "assays" / "zz-decoy.assay.json"
    decoy.write_text(json.dumps({"assay_id": "sha256:" + "f" * 64}), encoding="utf-8")

    first = api.audit(api.AuditRequest(path=root))
    assert audit.last_reuse() is None
    with pytest.raises(RewardLensError) as refused:
        Project.open(root).open_record("zz-decoy")
    assert refused.value.code == "RL0604"

    second = api.audit(api.AuditRequest(path=root))
    reused = audit.last_reuse()
    assert reused is not None and reused.assay_id == first.assay_id
    assert reused.name != "zz-decoy"


def test_a_bare_grader_has_no_store_to_ask_and_measures_every_time(tmp_path: Path) -> None:
    """Reuse is the store's rule, so a run with no project keeps measuring, as it always did."""
    grader = fixture(tmp_path, "clean.py", CLEAN)

    first = audit_of(grader)
    assert audit.last_reuse() is None
    second = audit_of(grader)

    assert audit.last_reuse() is None
    assert second.assay_id != first.assay_id  # `created` is wall-clock; nothing refuses the write
    written = sorted((grader.parent / "assays").glob("*.assay.json"))
    assert len(written) == 1, [path.name for path in written]
    assert written[0].read_bytes() == record_bytes(second)


# --- what the record says of itself in two sentences (A-016) --------------------------------------


def transcript_of(record: contracts.Assay) -> dict:
    return record.extensions[absences.TRANSCRIPT_EXTENSION]


def test_the_demo_record_says_what_it_could_establish_and_what_it_could_not(
    demo_record: contracts.Assay,
) -> None:
    """Both sentences are read off this record, not off a copy of the transcript.

    Each assertion below names the thing in the record the sentence is a reading of: the replay
    check that passed, the failing finding that is counted, and the `needs` notes on the absences.
    A sentence that stopped agreeing with those would fail here rather than ship a record whose
    summary and whose entries say different things.
    """
    block = transcript_of(demo_record)
    replay = by_id(demo_record, "validity.replay_determinism")
    assert replay.check is not None and replay.check.passed
    assert "deterministic and replayable" in block["could_establish"]

    failing = [finding for finding in demo_record.findings if finding.kind == "fail"]
    assert [finding.code for finding in failing] == ["RL0201"]
    assert failing[0].entries == ["validity.writable_then_read"]
    assert "one isolation defect" in block["could_establish"]
    assert block["could_establish"].endswith(".")

    wanted = {
        (entry.extensions or {}).get(absences.TRANSCRIPT_EXTENSION, {}).get("needs")
        for entry in demo_record.entries()
    } - {None}
    assert wanted == {absences.RUN_RECORD}
    assert block["could_not"] == "everything that needs a run record."


def test_the_bare_graders_two_sentences_are_the_ones_the_transcript_prints(bare: Path) -> None:
    """A-020: the bare grader's block, verbatim, and the record it is a reading of.

    The replay check runs on a bare grader now, so the first sentence has something to establish;
    the defect clause is named from the finding's rule (RL0210 is input handling, and RL0201 would
    be isolation). The second names only the inputs nothing else is waiting on, which for a grader
    with neither is tasks and responses; the protected check and the run record stand on those and
    are carried by the remedy table instead, with all four `needs` intact.
    """
    record = audit_of(bare)
    block = transcript_of(record)

    replay = by_id(record, "validity.replay_determinism")
    assert replay.kind != "absence" and replay.check is not None and replay.check.passed
    failing = [finding for finding in record.findings if finding.kind == "fail"]
    assert [finding.code for finding in failing] == ["RL0210"]
    assert (
        block["could_establish"]
        == "the grader is deterministic and replayable, and it has one input-handling defect."
    )

    assert block["could_not"] == "everything that needs tasks or responses."
    needs = {
        (entry.extensions or {}).get(absences.TRANSCRIPT_EXTENSION, {}).get("needs")
        for entry in record.entries()
    } - {None}
    assert needs == {
        absences.TASK_SET,
        absences.SAMPLED_RESPONSES,
        absences.PROTECTED_CHECK,
        absences.RUN_RECORD,
    }


def test_the_record_carries_the_plan_it_ran_under(demo_record: contracts.Assay, demo: Path) -> None:
    plan = audit.plan(request_for(demo))
    assert transcript_of(demo_record)["plan"] == {
        "panels": len(plan.panels),
        "paid_calls": plan.paid_calls,
        "estimate_s": plan.estimate_s,
    }


def test_the_bare_grader_record_says_both_sentences_of_itself(bare: Path) -> None:
    """The same two sentences over a record with no task set: every clause moves with the record.

    Replay runs here on the pair the declaration asks for (A-020), so the sentence says so; the
    defect it did find is the input-handling one; and what it could not reach is named by the two
    inputs nothing else is waiting on, with the protected check and the run record left to the
    remedy table, which carries all four with what each unlocks.
    """
    record = audit_of(bare)
    block = transcript_of(record)

    replay = by_id(record, "validity.replay_determinism")
    assert replay.kind == "check" and replay.check is not None and replay.check.passed

    failing = [finding for finding in record.findings if finding.kind == "fail"]
    assert [finding.code for finding in failing] == ["RL0210"]
    assert failing[0].entries == ["validity.input_handling"]
    assert block["could_establish"] == (
        "the grader is deterministic and replayable, and it has one input-handling defect."
    )

    needs = [
        (entry.entry_id, (entry.extensions or {})[absences.TRANSCRIPT_EXTENSION]["needs"])
        for entry in record.entries()
        if (entry.extensions or {}).get(absences.TRANSCRIPT_EXTENSION, {}).get("needs")
    ]
    assert ("validity.task_validity", absences.TASK_SET) in needs
    assert ("signal.instrument_absent", absences.SAMPLED_RESPONSES) in needs
    assert ("soundness.instrument_absent", absences.PROTECTED_CHECK) in needs
    assert ("trace.instrument_absent", absences.RUN_RECORD) in needs
    assert block["could_not"] == "everything that needs tasks or responses."


def test_every_absence_carries_a_code_for_what_is_missing_and_the_shortest_remedy(
    bare: Path, demo_record: contracts.Assay
) -> None:
    """A-025 point 3: the hole's `missing` and `remedy`, as a machine can group and act on them.

    The bare grader is the record that exercises the whole vocabulary, because it is the one
    missing every input. The long `missing_access` sentence is untouched beside them: it is what
    the transcript prints, and the code is what a reader comparing two records groups on.
    """
    record = audit_of(bare)
    absent = [entry for entry in record.entries() if entry.kind == "absence"]
    stamped = {
        entry.entry_id: (
            entry.extensions[absences.TRANSCRIPT_EXTENSION]["missing_code"],
            entry.extensions[absences.TRANSCRIPT_EXTENSION]["remedy_short"],
        )
        for entry in absent
    }
    assert len(stamped) == len(absent)  # no absence goes out without one
    assert set(stamped.values()) == {
        ("instrument_not_in_this_build", "a later build"),
        ("task_set", "--tasks t.jsonl"),
        ("response_bank", "--responses r.jsonl"),
        ("protected_check", "--outcome ./tests"),
        ("run_record", "reward-lens trace <run>"),
        ("reference_material", "none for this substrate yet"),
    }
    assert stamped["validity.task_validity"] == ("task_set", "--tasks t.jsonl")
    assert stamped["calibration.instrument_absent"] == (
        "reference_material",
        "none for this substrate yet",
    )
    assert by_id(record, "validity.flakiness").absence.missing_access == (
        "no task set was supplied, and this check needs one"
    )  # the sentence is untouched beside the code

    # The demo supplies every input, so the only holes left are the ones no input lifts.
    supplied = {
        entry.extensions[absences.TRANSCRIPT_EXTENSION]["missing_code"]
        for entry in demo_record.entries()
        if entry.kind == "absence"
    }
    assert supplied == {"instrument_not_in_this_build", "run_record", "reference_material"}


def test_the_bare_graders_replay_is_the_pair_its_own_signature_asks_for(bare: Path) -> None:
    """A-020: the transcript prints `100 repeats, identical output, no hidden state`, so it ran.

    The pair is in the entry's assumptions, which is what makes the result readable: a determinism
    claim over an input nobody can see is a claim about nothing.
    """
    entry = by_id(audit_of(bare), "validity.replay_determinism")
    assert entry.result["repeats"] == 100
    assert entry.result["identical"] is True
    assert entry.result["hidden_state"] is False
    assert entry.result["summary"] == "100 repeats, identical output, no hidden state"
    assert any("score(task, response)" in note for note in entry.assumptions)
    assert any('"prompt"' in note and '"expected"' in note for note in entry.assumptions)


def test_a_grader_that_raises_on_its_own_pair_is_a_failed_check_and_not_an_absence(
    tmp_path: Path,
) -> None:
    """The raise is the result. An absence would say nothing was measured, and something was."""
    grader = tmp_path / "raiser.py"
    grader.write_text(
        "def score(task, response):\n    raise RuntimeError('no')\n", encoding="utf-8"
    )
    entry = by_id(audit_of(grader), "validity.replay_determinism")
    assert entry.kind == "check"
    assert entry.check is not None and entry.check.passed is False
    assert "RuntimeError" in entry.result["summary"]
    assert entry.result["raised"] == "RuntimeError"


def test_the_subject_clauses_the_closed_subject_cannot_carry(
    demo_record: contracts.Assay, demo: Path, bare: Path
) -> None:
    """A-019: the Subject line's clauses, at the record's top level, one shape per situation.

    A-025 point 4 adds the task file's name here: the closed `subject` carries the task set as a
    digest, and a digest does not tell a reader which file to open again. The envelope prints the
    two together as `<task file>@<digest>`, so the name has to be somewhere the record can reach.
    """
    project = transcript_of(demo_record)["subject"]
    assert project["grader"] == "grader.py"
    assert project["tasks"] > 0 and project["responses"] > 0
    assert project["task_file"] == "tasks.jsonl"  # as the project declares it, not a digest
    assert demo_record.subject.context.task_set is not None

    record = audit_of(bare)
    assert transcript_of(record)["subject"] == {
        "grader": "my_grader.py",
        "signature": "score(task, response)",
        "shape": "plain shape",
    }
    assert record.subject.context.task_set is None


def test_the_task_set_the_record_names_is_the_one_the_project_declares(
    demo_record: contracts.Assay, demo: Path, tmp_path: Path
) -> None:
    """A-019: the field names *this* task set, not merely that there was one.

    The record carries the digest the version already computed for the declared task set, so the
    field moves when the task file does. The path-qualified form the envelope prints
    (`tasks.jsonl@sha256:...`) is the envelope's: the assay schema constrains this field to a bare
    `sha256:<64 hex>` and rejects anything else, and the schema is frozen in this wave.
    """
    declared = Project.open(demo).version(methods=()).to_subject_digests().task_distribution
    assert demo_record.subject.context.task_set == declared

    other = tmp_path / "other"
    write_example(other)
    tasks = other / "tasks.jsonl"
    tasks.write_text(tasks.read_text(encoding="utf-8").replace("one", "two", 1), encoding="utf-8")
    moved = Project.open(other).version(methods=()).to_subject_digests().task_distribution
    assert moved != declared

    with pytest.raises(Exception):
        contracts.models.ContextRef(
            policy=None, task_set=f"tasks.jsonl@{declared}", configuration=None
        )


def test_the_decision_names_the_use_the_project_declares(
    demo_record: contracts.Assay, bare: Path
) -> None:
    """`unresolved` is a verdict about a use, so the record has to say which use."""
    assert demo_record.decision.action == "train_on_this_version"
    assert demo_record.decision.state == "unresolved"
    assert audit_of(bare).decision.action is None


def test_the_verdict_counts_one_blocking_finding_and_eight_required_panels(
    demo_record: contracts.Assay,
) -> None:
    """The transcript's `1 blocking finding, and 8 required panels could not run.` counts from here.

    One error-level finding is one reason. It used to be named twice, by rule and by finding id, so
    the reasons said two where the report prints one. `outcome_unqualified:` is not counted with
    them: it is a limit of the run, not a defect of the grader.
    """
    reasons = demo_record.decision.reasons
    blocking = [r for r in reasons if r.startswith("blocking_finding:")]
    missing = [r for r in reasons if r.startswith("required_missing:")]
    assert (len(blocking), len(missing)) == (1, 8)
    assert blocking == ["blocking_finding:RGX-local-0001"]
    assert [r for r in reasons if r.startswith("outcome_unqualified:")] == [
        "outcome_unqualified:protected_test_suite"
    ]
    assert len([f for f in demo_record.findings if f.level == "error"]) == 1


def test_eight_required_panels_could_not_run_and_calibration_is_one_of_them(
    demo_record: contracts.Assay,
) -> None:
    """The transcript counts 8 over 8 dark sections: no required panel is waived out of the count.

    Calibration was once counted out, on the reading that nothing in the remedy table brings a
    certified reference material for this substrate, so naming it would send a reader after what
    does not exist. A-023 withdrew that. A required panel the run could not fill costs the verdict
    whether or not the input is obtainable, and the remedy row is where the caveat belongs: it
    already says nothing here can be calibrated until such a material is published.
    """
    missing = [
        reason for reason in demo_record.decision.reasons if reason.startswith("required_missing:")
    ]
    assert len(missing) == 8
    assert "required_missing:calibration" in missing
    dark = {
        entry.section for entry in demo_record.entries() if entry.kind == "absence"
    } - {"validity", "reach"}
    assert len(dark) == 8 and "calibration" in dark
    named = {reason.split(":", 1)[1] for reason in missing}
    assert named == dark  # every dark section is named, none is held back
    assert named == {
        "soundness",
        "exploits",
        "framing",
        "reward_statistics",
        "signal",
        "trace",
        "forecast",
        "calibration",
    }


def test_the_reach_section_is_one_detection_and_the_witness_it_could_not_execute(
    demo_record: contracts.Assay,
) -> None:
    """A-014: the section is lit and carries a COULD_NOT_CHECK absence, which is what `!` is."""
    reach = [entry for entry in demo_record.entries() if entry.section == "reach"]
    assert [(entry.entry_id, entry.kind) for entry in reach] == [
        ("reach.attack_surface", "detection"),
        ("reach.executed_witness", "absence"),
    ]
    found, witness = reach
    assert found.check is None
    assert found.n == found.result["surfaces_exposed"] >= 1  # the count is on the detection
    assert "exposure_id" in found.result
    assert found.result["summary"].endswith("exposed, by static analysis only")
    assert found.result["note"] == "no executed witness: not in this build"
    assert witness.absence.state == "COULD_NOT_CHECK"
    assert witness.absence.missing_access == "not in this build"


def test_a_bare_grader_names_no_trainer_and_claims_nothing_about_one(bare: Path) -> None:
    """A-017: the transcript's sentence is a claim about a declared trainer, so it is not made."""
    record = audit_of(bare)
    message = next(finding.message for finding in record.findings if finding.code == "RL0210")
    assert "the trainer you named" not in message
    assert "no trainer is declared under `reward.trainer`" in message
    entry = by_id(record, "validity.input_handling")
    assert not any("declare a trainer to have it named here" in note for note in entry.limitations)


def declaring(root: Path, trainer: str) -> contracts.Assay:
    """A project with D-65's first defect in its grader, declaring `trainer` under `reward`."""
    write_example(root)
    (root / "grader.py").write_text(NONE_ON_MALFORMED.lstrip("\n"), encoding="utf-8")
    config = root / "rewardlens.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("trainer: trl", f"trainer: {trainer}"),
        encoding="utf-8",
    )
    return audit_of(root, project=Project.open(root))


def test_the_demo_declares_its_trainer_so_the_sentence_is_measured(tmp_path: Path) -> None:
    """A-025's fresh review: the sentence is selected by the trainer's name, not by there being one.

    `reward.trainer` admits four names and only trl's semantics were read for D-65. Picking the
    declared-trainer sentence by truthiness told a project declaring `verifiers` that the trainer
    it named turns None into NaN, which is trl's documented behaviour and nobody measured it here.
    The finding's message is what a reader sees, so that is what is read on both sides.
    """
    named = declaring(tmp_path / "trl", "trl")
    message = next(finding.message for finding in named.findings if finding.code == "RL0210")
    assert "the trainer you named turns None into NaN and drops the row" in message
    entry = by_id(named, "validity.input_handling")
    assert any("what trl does with the returned value" in note for note in entry.limitations)

    other = declaring(tmp_path / "verifiers", "verifiers")
    message = next(finding.message for finding in other.findings if finding.code == "RL0210")
    assert "the trainer you named" not in message
    assert "this build has not read the declared trainer's semantics" in message
    entry = by_id(other, "validity.input_handling")
    assert any(
        "`verifiers` is declared under `reward.trainer` and this build has read no semantics for it"
        in note
        for note in entry.limitations
    )


# --- reuse is scope-aware (A-016, section 6.1) ----------------------------------------------------


class CountingInstrument(FakeInstrument):
    """A `FakeInstrument` that says how many times it was asked to measure."""

    def __init__(self, instrument_id: str, section: str, count: int) -> None:
        super().__init__(instrument_id, section, count)
        self.calls = 0

    def run(self, subject, corpus, ctx):  # noqa: ANN001, ANN201
        self.calls += 1
        return super().run(subject, corpus, ctx)


def test_the_same_request_twice_runs_no_instrument_the_second_time(
    tmp_path: Path, monkeypatch
) -> None:
    """The reuse that has to survive the scope check: nothing about the plan moved."""
    instrument = CountingInstrument("soundness.battery", "soundness", 1)
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="soundness", instruments=(instrument,)))
    )
    root = tmp_path / "twice"
    write_example(root)

    first = api.audit(api.AuditRequest(path=root))
    assert instrument.calls == 1
    second = api.audit(api.AuditRequest(path=root))

    assert instrument.calls == 1, "the second audit measured again"
    assert audit.last_reuse() is not None
    assert api.last_reuse() is not None, "the SDK reports the reuse the runner recorded"
    assert second.assay_id == first.assay_id
    assert Project.open(root).names() == (Project.open(root).names()[0],)


def test_a_panel_that_lands_between_two_audits_is_measured_and_not_reused(
    tmp_path: Path, monkeypatch
) -> None:
    """Section 6.1, and the reproduction the principal's reviewer ran.

    The dependency set is unchanged, so the old check said reuse and the run reported that no
    instrument had run. But the request is wider than the record: the stored one carries
    `soundness.instrument_absent`, which is the runner saying that panel had nothing to run. So
    the panel runs, its entries are in a new record, and the record that could not answer is left
    exactly as it was.
    """
    root = tmp_path / "landed"
    write_example(root)
    first = api.audit(api.AuditRequest(path=root))
    (first_name,) = Project.open(root).names()
    first_bytes = Project.open(root).path_for(first_name).read_bytes()
    assert by_id(first, "soundness.instrument_absent").kind == "absence"

    instrument = CountingInstrument("soundness.battery", "soundness", 2)
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="soundness", instruments=(instrument,)))
    )
    second = api.audit(api.AuditRequest(path=root))

    assert audit.last_reuse() is None, "a wider plan may not be answered from a narrower record"
    assert instrument.calls == 1
    assert [entry.entry_id for entry in second.measurement.soundness] == [
        "soundness.battery.0",
        "soundness.battery.1",
    ]
    assert second.assay_id != first.assay_id
    # The old record is still on disk, sealed, and is the bytes it was.
    project = Project.open(root)
    assert first_name in project.names() and len(project.names()) == 2
    assert project.path_for(first_name).read_bytes() == first_bytes
    # A-016: same subject version, same day, so the name is told apart by what was measured.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (other,) = [name for name in project.names() if name != first_name]
    assert first_name == f"code-reward-v1-{stamp}"
    assert other.startswith(f"{first_name}-") and len(other) == len(first_name) + 9


def test_a_panel_that_was_already_there_does_not_stop_the_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    """The other side of the same check: coverage is about what ran, not about how much of it."""
    instrument = CountingInstrument("framing.context_leak", "framing", 1)
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="framing", instruments=(instrument,)))
    )
    root = tmp_path / "same-panel"
    write_example(root)
    first = api.audit(api.AuditRequest(path=root))
    second = api.audit(api.AuditRequest(path=root))
    assert instrument.calls == 1
    assert second.assay_id == first.assay_id


class DeclaringInstrument(CountingInstrument):
    """An instrument that declares the method it will write, so the plan can name it before it runs.

    A real wave-2 instrument's method is a constant of the instrument and of the parameters it was
    configured with, not of the subject, so it can be stated up front. That declaration is what
    lets the reuse check ask whether a stored record measured this entry the way this request
    would, rather than only whether the record holds the id at all.
    """

    def __init__(self, instrument_id: str, section: str, count: int, params: str) -> None:
        super().__init__(instrument_id, section, count)
        self.method = contracts.Method(
            id=instrument_id,
            version="1.0.0",
            params_digest=contracts.digest({"params": params}),
            procedure="a stand-in wave-2 procedure, run with the parameters its digest names",
        )

    def run(self, subject, corpus, ctx):  # noqa: ANN001, ANN201
        self.calls += 1
        subject_ref = contracts.digest({"source": subject.source})
        return [
            contracts.absence(
                self.section,
                f"{self.id}.{n}",
                f"what {self.id} measured, part {n}",
                "nothing: this instrument stands in for a wave-2 panel",
                "land the real panel",
                (f"no {self.section} claim from {self.id}",),
                subject_ref=subject_ref,
                provenance=ctx.provenance(),
                method=self.method,
            )
            for n in range(self.count)
        ]


def test_a_method_added_beside_one_that_already_ran_is_measured_and_not_reused(
    tmp_path: Path, monkeypatch
) -> None:
    """The principal's reviewer's reproduction, at the granularity that missed it.

    `reach` was already populated when the first audit ran, so every panel-name check said the
    record covered the request and the second audit returned the old assay with nothing run. The
    added method is a requested entry id the record does not hold, which is the same fact about a
    method that a dark panel is about a section, so it runs.
    """
    root = tmp_path / "added"
    write_example(root)
    ran = CountingInstrument("reach.oversight", "reach", 1)
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="reach", instruments=(ran,)))
    )
    first = api.audit(api.AuditRequest(path=root))
    assert ran.calls == 1
    project = Project.open(root)
    (first_name,) = project.names()
    first_bytes = project.path_for(first_name).read_bytes()

    added = CountingInstrument("reach.oversight_added", "reach", 1)
    monkeypatch.setattr(
        registry,
        "discover",
        discovering(
            Panel(
                section="reach",
                instruments=(CountingInstrument("reach.oversight", "reach", 1), added),
            )
        ),
    )
    second = api.audit(api.AuditRequest(path=root))

    assert api.last_reuse() is None, "a request naming a method the record lacks is not answered"
    assert added.calls == 1
    assert ran.calls == 1, "the instrument object of the first audit was not asked again"
    assert by_id(second, "reach.oversight_added.0").kind == "absence"
    assert second.assay_id != first.assay_id
    # The record that could not answer is left exactly as it was, under its own name.
    project = Project.open(root)
    assert first_name in project.names() and len(project.names()) == 2
    assert project.path_for(first_name).read_bytes() == first_bytes


def test_one_entry_id_measured_by_a_method_that_moved_is_measured_again(
    tmp_path: Path, monkeypatch
) -> None:
    """Same id, new parameters: the record holds the id and did not measure what is asked now."""
    root = tmp_path / "moved"
    write_example(root)
    before = DeclaringInstrument("reach.oversight", "reach", 1, "probes=4")
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="reach", instruments=(before,)))
    )
    first = api.audit(api.AuditRequest(path=root))
    assert before.calls == 1
    project = Project.open(root)
    (first_name,) = project.names()
    first_bytes = project.path_for(first_name).read_bytes()
    assert by_id(first, "reach.oversight.0").method.params_digest == before.method.params_digest

    after = DeclaringInstrument("reach.oversight", "reach", 1, "probes=16")
    assert after.method.id == before.method.id
    assert after.method.params_digest != before.method.params_digest
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="reach", instruments=(after,)))
    )
    second = api.audit(api.AuditRequest(path=root))

    assert api.last_reuse() is None, "the same id measured another way is another measurement"
    assert after.calls == 1
    assert second.assay_id != first.assay_id
    assert by_id(second, "reach.oversight.0").method.params_digest == after.method.params_digest
    project = Project.open(root)
    assert first_name in project.names() and len(project.names()) == 2
    assert project.path_for(first_name).read_bytes() == first_bytes


def test_an_unchanged_declared_method_is_reused_without_running_it(
    tmp_path: Path, monkeypatch
) -> None:
    """The control for the case above: the identity is compared, not merely recomputed."""
    root = tmp_path / "unmoved"
    write_example(root)
    monkeypatch.setattr(
        registry,
        "discover",
        discovering(
            Panel(
                section="reach",
                instruments=(DeclaringInstrument("reach.oversight", "reach", 1, "probes=4"),),
            )
        ),
    )
    first = api.audit(api.AuditRequest(path=root))

    again = DeclaringInstrument("reach.oversight", "reach", 1, "probes=4")
    monkeypatch.setattr(
        registry, "discover", discovering(Panel(section="reach", instruments=(again,)))
    )
    second = api.audit(api.AuditRequest(path=root))

    assert again.calls == 0
    assert api.last_reuse() is not None
    assert second.assay_id == first.assay_id
    assert Project.open(root).names() == (Project.open(root).names()[0],)


def test_a_method_the_request_does_not_ask_about_does_not_stop_the_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    """Containment one way: a record of a wider build still answers the narrower request.

    The panel is gone from this build, so what it measured is not part of what is being asked, and
    a record that holds more than the request names is not a record that measured something else.
    """
    root = tmp_path / "dropped"
    write_example(root)
    monkeypatch.setattr(
        registry,
        "discover",
        discovering(
            Panel(
                section="soundness",
                instruments=(DeclaringInstrument("soundness.battery", "soundness", 1, "n=3"),),
            )
        ),
    )
    first = api.audit(api.AuditRequest(path=root))

    monkeypatch.setattr(registry, "discover", discovering())
    second = api.audit(api.AuditRequest(path=root))

    assert api.last_reuse() is not None
    assert second.assay_id == first.assay_id
    assert Project.open(root).names() == (Project.open(root).names()[0],)


def test_a_changed_grader_leaves_the_record_of_the_old_one_alone(tmp_path: Path) -> None:
    """Re-measuring never edits what was measured: the superseded record is byte-identical."""
    root = tmp_path / "changed"
    write_example(root)
    api.audit(api.AuditRequest(path=root))
    project = Project.open(root)
    (name,) = project.names()
    before = project.path_for(name).read_bytes()

    grader = root / "grader.py"
    grader.write_text(grader.read_text(encoding="utf-8") + "\n# moved\n", encoding="utf-8")
    second = api.audit(api.AuditRequest(path=root))

    assert audit.last_reuse() is None
    assert Project.open(root).path_for(name).read_bytes() == before
    assert second.subject.version.digest != contracts.Assay.model_validate_json(before).subject.version.digest
