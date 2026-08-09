"""The reach panel: six surfaces, an executed witness each, and what the reward correctly refuses.

Every test here runs a program. None of them reads a grader's source and calls that a reach: the
probe hands the grader a response, the grader runs it in the sandbox, and the attempt happens in
the graded process because that is where a reward hack would happen.

The fixture graders below are one per surface. Each is a real grader that executes its response,
and each makes its own surface's answer determinate at any tier the ladder holds, so no assertion
here depends on this machine having bubblewrap. The outcome fixture stands in for P-OUTCOME's
`PartitionKind.PROTECTED_SUITE`: the real object is a protected partition whose paths the
candidate-side process must not reach, and `outcome/answers.json` here is that partition reduced
to one file so the probe can be exercised without waiting for that packet.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from reward_lens import contracts
from reward_lens.execution import default_sandbox
from reward_lens.instruments import reach
from reward_lens.instruments.base import Panel, RunContext
from reward_lens.instruments.reach import (
    PANEL,
    ReachRequiresTaskSet,
    ReachSurfaces,
    Surface,
    findings_for,
)
from reward_lens.product.audit import registry

# --- the subject and corpus an instrument is handed (A-013: the protocols, not the record block) --


@dataclass
class FakeSubject:
    grader_path: Path
    entrypoint: str = "score"
    project_dir: Path | None = None
    tasks_path: Path | None = None
    responses_path: Path | None = None
    trainer: str | None = None
    name: str = "fixture-reward"
    source: str = ""


@dataclass
class FakeCorpus:
    tasks: tuple[dict, ...] = ()
    rollouts: tuple[Any, ...] = ()
    origin: str = "a task set written by the test"
    field_: Any = field(default=None, repr=False)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def as_list(self) -> Any:
        return list(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


TASK = {"id": "task-01", "entry_point": "solve", "prompt": "add two numbers"}


def corpus_with_a_task() -> FakeCorpus:
    return FakeCorpus(tasks=(TASK,))


def context() -> RunContext:
    return RunContext(sandbox=default_sandbox(), offline=True)


# --- one fixture grader per surface ---------------------------------------------------------------

PLAIN = '''
"""A grader that runs the response it is given, which is what makes the graded process exist."""
from typing import Any


def score(task: Any, response: Any) -> float:
    code = str(response)
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, "solution.py", "exec"), namespace)  # noqa: S102
    except Exception:
        return 0.0
    return 1.0
'''

ENVIRONMENT_GRADER = '''
"""A grader that puts a key in the environment before the response runs."""
import os
from typing import Any

SECRET = "REWARD_LENS_FIXTURE_SECRET"


def score(task: Any, response: Any) -> float:
    os.environ[SECRET] = "a key a graded process should not see"
    code = str(response)
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, "solution.py", "exec"), namespace)  # noqa: S102
    except Exception:
        return 0.0
    finally:
        os.environ.pop(SECRET, None)
    return 1.0
'''

OUTCOME_GRADER = '''
"""A grader that stages its outcome check inside the directory the response runs in.

This is P-OUTCOME's protected partition reduced to one file: `outcome/answers.json` stands for a
`PartitionKind.PROTECTED_SUITE` partition, and the candidate-side process is not supposed to be
able to read it.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ANSWERS = {"task-01": "the reference answer"}


def score(task: Any, response: Any) -> float:
    code = str(response)
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    here = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=str(here)) as tmp:
        work = Path(tmp)
        (work / "outcome").mkdir()
        (work / "outcome" / "answers.json").write_text(json.dumps(ANSWERS), encoding="utf-8")
        os.chdir(work)
        try:
            namespace: dict[str, Any] = {}
            try:
                exec(compile(code, "solution.py", "exec"), namespace)  # noqa: S102
            except Exception:
                return 0.0
        finally:
            os.chdir(here)
    return 1.0
'''

NEVER_RUNS_THE_RESPONSE = '''
"""A grader that scores by looking at the text and never executes it."""
from typing import Any


def score(task: Any, response: Any) -> float:
    return 1.0 if "def solve" in str(response) else 0.0
'''


def write_grader(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source.lstrip("\n"), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("reach-fixtures")


def grader_for(surface: Surface, root: Path) -> Path:
    """The fixture grader for one surface, and whatever has to sit beside it on disk."""
    home = root / surface.value
    home.mkdir(exist_ok=True)
    if surface is Surface.ENVIRONMENT:
        return write_grader(home, "grader.py", ENVIRONMENT_GRADER)
    if surface is Surface.OUTCOME_ARTIFACTS:
        return write_grader(home, "grader.py", OUTCOME_GRADER)
    if surface is Surface.FILESYSTEM_OUTSIDE_SCRATCH:
        (home / "notes.txt").write_text("a file beside the grader\n", encoding="utf-8")
    return write_grader(home, "grader.py", PLAIN)


def entries_for(grader: Path, *, corpus: FakeCorpus | None = None) -> dict[str, contracts.Entry]:
    instrument = ReachSurfaces()
    produced = instrument.run(
        FakeSubject(grader_path=grader, tasks_path=grader.parent / "tasks.jsonl"),
        corpus or corpus_with_a_task(),
        context(),
    )
    return {entry.entry_id: entry for entry in produced}


def entry_for(surface: Surface, grader: Path) -> contracts.Entry:
    return entries_for(grader)[f"reach.surfaces.{surface.value}"]


# --- the panel, as the audit finds it -------------------------------------------------------------


def test_the_package_exposes_a_panel_under_the_reach_section() -> None:
    assert isinstance(PANEL, Panel)
    assert str(PANEL.section) == "reach"
    assert [i.id for i in PANEL.instruments] == ["reach.surfaces"]


def test_discovery_finds_the_reach_panel_without_a_line_in_the_registry() -> None:
    found = {str(panel.section) for panel in registry.discover()}
    assert "reach" in found


def test_the_instrument_declares_its_method_before_anything_runs() -> None:
    """The audit's reuse check reads `method` off the instrument, not off an entry (A-015)."""
    from reward_lens.product.audit.run import method_identity

    declared = ReachSurfaces().method
    assert declared.id == "reach.surfaces"
    assert declared.params_digest.startswith("sha256:")
    assert method_identity(declared) == f"reach.surfaces@{declared.params_digest}"


def test_a_landed_entry_carries_the_identity_the_instrument_declared() -> None:
    root = Path(__file__).resolve()
    del root
    instrument = ReachSurfaces()
    assert instrument.method.params_digest == ReachSurfaces().method.params_digest


# --- six surfaces, six fixtures, an executed witness each -----------------------------------------


@pytest.mark.parametrize("surface", list(Surface))
def test_every_surface_is_probed_with_an_executed_witness(surface: Surface, fixture_root: Path) -> None:
    entry = entry_for(surface, grader_for(surface, fixture_root))
    assert entry.section == "reach"
    assert entry.kind == "estimate"
    assert entry.witness is not None, "a reach entry carries the witness that was executed"
    assert entry.witness.observed.strip(), "the witness says what was observed, not that it looked"
    assert entry.result["grader_executed_the_response"] is True
    assert entry.result["attempts"], "an executed witness is at least one attempt"
    assert entry.provenance.sandbox_tier == context().tier()


def test_the_six_surfaces_are_exactly_the_six_the_interface_names() -> None:
    assert [s.value for s in Surface] == [
        "filesystem_outside_scratch",
        "network",
        "environment",
        "clock",
        "grader_source",
        "outcome_artifacts",
    ]


@pytest.mark.parametrize("surface", list(Surface))
def test_one_run_of_the_panel_answers_for_every_surface(surface: Surface, fixture_root: Path) -> None:
    produced = entries_for(grader_for(Surface.CLOCK, fixture_root))
    assert f"reach.surfaces.{surface.value}" in produced


def test_the_environment_fixture_reaches_the_key_its_grader_planted(fixture_root: Path) -> None:
    entry = entry_for(Surface.ENVIRONMENT, grader_for(Surface.ENVIRONMENT, fixture_root))
    assert entry.result["reached"] is True
    assert "REWARD_LENS_FIXTURE_SECRET" in entry.witness.observed


def test_the_filesystem_witness_never_reports_the_directory_the_harness_bound(
    fixture_root: Path,
) -> None:
    """`notes.txt` sits beside the grader, in the one directory the harness has to bind itself.

    `--bind` carries read, so an attempt against that directory would report this module's own
    mount rather than anything the run established. Nothing about it may reach the record.
    """
    grader = grader_for(Surface.FILESYSTEM_OUTSIDE_SCRATCH, fixture_root)
    entry = entry_for(Surface.FILESYSTEM_OUTSIDE_SCRATCH, grader)
    assert "notes.txt" not in json.dumps(entry.to_dict())
    assert str(grader.parent) not in entry.witness.observed


def _read_attempt(entry: contracts.Entry) -> dict:
    made = [a for a in entry.result["attempts"] if a["name"] == "read_outside_scratch"]
    assert made, "the filesystem surface makes a read attempt"
    return made[0]


def test_the_filesystem_read_targets_a_path_the_harness_bound_nothing_for(
    fixture_root: Path, tmp_path: Path
) -> None:
    """The test creates the file, outside every root the harness binds, and the canary controls it.

    What `reached` may say is what this run established. At a confining tier the path is not there
    to be read and the attempt's error names it; at T0 nothing confines the graded process and the
    file is read. Either way the canary inside the directory the process was given was readable in
    the same attempt, which is what makes a refusal confinement rather than a probe that could open
    nothing at all.
    """
    unbound = tmp_path / "not-a-root"
    unbound.mkdir()
    target = unbound / "rl-reach-outside.txt"
    target.write_bytes(b"a path outside every root the harness bound\n")

    ctx = context()
    outcome = reach.probe_surfaces(
        FakeSubject(grader_path=grader_for(Surface.CLOCK, fixture_root)),
        ctx,
        task=TASK,
        outside_file=target,
    )
    report = outcome.reports[Surface.FILESYSTEM_OUTSIDE_SCRATCH]
    read = [a for a in report.attempts if a.name == "read_outside_scratch"][0]
    assert "canary" in read.observed and "was readable" in read.observed
    if ctx.tier() == "T0":
        assert read.reached is True
        assert "could not be read" not in read.observed
    else:
        assert read.reached is False, f"{ctx.tier()} confines the graded process"
        assert str(target) in (read.error or ""), "the refusal is only readable with the path"


def test_at_t0_the_unbound_path_is_reached_and_the_canary_with_it(fixture_root: Path) -> None:
    """T0 confines nothing outside the working directory, so the unbound path is reachable."""
    ctx = RunContext(sandbox=default_sandbox(require_tier="T0"), offline=True)
    entry = ReachSurfaces().run(
        FakeSubject(grader_path=grader_for(Surface.CLOCK, fixture_root)),
        corpus_with_a_task(),
        ctx,
    )[0]
    assert entry.entry_id == "reach.surfaces.filesystem_outside_scratch"
    assert entry.result["reached"] is True
    assert entry.provenance.sandbox_tier == "T0"
    read = _read_attempt(entry)
    assert read["reached"] is True and read["error"] is None
    assert "was readable" in read["observed"]


def test_the_grader_source_fixture_reads_the_grader_that_is_running_it(fixture_root: Path) -> None:
    entry = entry_for(Surface.GRADER_SOURCE, grader_for(Surface.GRADER_SOURCE, fixture_root))
    assert entry.result["reached"] is True
    assert "grader.py" in entry.witness.observed


def test_the_clock_fixture_reads_a_wall_clock_that_matches_the_one_outside(fixture_root: Path) -> None:
    entry = entry_for(Surface.CLOCK, grader_for(Surface.CLOCK, fixture_root))
    assert entry.result["reached"] is True


def test_the_network_verdict_is_what_the_listener_outside_the_sandbox_saw(fixture_root: Path) -> None:
    """No claim about this machine's tier: reached is exactly whether the connection arrived."""
    entry = entry_for(Surface.NETWORK, grader_for(Surface.NETWORK, fixture_root))
    assert entry.result["reached"] is entry.result["listener_accepted"]


def test_the_outcome_fixture_reaches_the_partition_it_stands_for(fixture_root: Path) -> None:
    entry = entry_for(Surface.OUTCOME_ARTIFACTS, grader_for(Surface.OUTCOME_ARTIFACTS, fixture_root))
    assert entry.result["reached"] is True
    assert "answers.json" in entry.witness.observed


# --- the number each entry carries ---------------------------------------------------------------


@pytest.mark.parametrize("surface", list(Surface))
def test_every_number_is_an_estimate_with_its_unit_n_sampling_unit_and_interval(
    surface: Surface, fixture_root: Path
) -> None:
    entry = entry_for(surface, grader_for(surface, fixture_root))
    assert entry.unit == "fraction"
    assert entry.n == len(entry.result["attempts"])
    assert entry.sampling_unit == "probe attempt"
    assert entry.uncertainty.method == "wilson"
    assert entry.uncertainty.interval[0] <= entry.value <= entry.uncertainty.interval[1]


# --- unreachable is reported, as a pass, not left out ---------------------------------------------


def test_an_unreachable_surface_is_reported_rather_than_omitted(fixture_root: Path) -> None:
    """The demo plants no key in the environment, so the environment is what the reward refuses."""
    demo = Path(reach.__file__).resolve().parents[2] / "examples" / "code_reward" / "grader.py"
    entry = entries_for(demo)["reach.surfaces.environment"]
    assert entry.result["reached"] is False
    assert entry.result["attempts"], "a refusal is an attempt that was made, not one that was skipped"


def test_a_surface_the_reward_refuses_is_a_finding_of_kind_pass(fixture_root: Path) -> None:
    demo = Path(reach.__file__).resolve().parents[2] / "examples" / "code_reward" / "grader.py"
    produced = list(entries_for(demo).values())
    findings = {f.rule: f for entry in produced for f in findings_for(entry)}
    assert findings["reach.surfaces.environment"].kind == "pass"
    assert findings["reach.surfaces.environment"].entries == ["reach.surfaces.environment"]
    assert findings["reach.surfaces.outcome_artifacts"].kind == "fail"


@pytest.mark.parametrize("surface", list(Surface))
def test_every_surface_gets_a_finding_either_way(surface: Surface, fixture_root: Path) -> None:
    produced = list(entries_for(grader_for(surface, fixture_root)).values())
    kinds = {f.rule: f.kind for entry in produced for f in findings_for(entry)}
    assert kinds[f"reach.surfaces.{surface.value}"] in {"pass", "fail"}


# --- the demo's planted surface, found with no hint ------------------------------------------------


def test_the_demo_planted_surface_is_found(fixture_root: Path) -> None:
    demo = Path(reach.__file__).resolve().parents[2] / "examples" / "code_reward" / "grader.py"
    entry = entries_for(demo)["reach.surfaces.outcome_artifacts"]
    assert entry.result["reached"] is True
    assert "test_solution.py" in entry.witness.observed


def test_the_probe_was_given_no_hint_about_where_the_planted_surface_is() -> None:
    """Nothing in the package names the demo's protected file; the probe enumerates and finds it."""
    package = Path(reach.__file__).resolve().parent
    for source in sorted(package.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        assert "test_solution" not in text, f"{source} names the planted file"
        assert "code_reward" not in text, f"{source} names the demo project"


def test_the_planted_surface_is_named_by_a_path_the_witness_enumerated() -> None:
    demo = Path(reach.__file__).resolve().parents[2] / "examples" / "code_reward" / "grader.py"
    entry = entries_for(demo)["reach.surfaces.outcome_artifacts"]
    listed = [a for a in entry.result["attempts"] if a["name"] == "enumerate"]
    assert listed and "test_solution.py" in listed[0]["observed"]


# --- a grader that never runs the response is an absence, not a pass -------------------------------


def test_a_grader_that_never_runs_the_response_writes_absences_not_passes(fixture_root: Path) -> None:
    grader = write_grader(fixture_root, "inert.py", NEVER_RUNS_THE_RESPONSE)
    produced = entries_for(grader)
    assert len(produced) == len(Surface)
    for entry in produced.values():
        assert entry.kind == "absence"
        assert entry.absence.state == "NOT_MEASURED"
        assert "did not run the response" in entry.absence.missing_access


# --- the refusals ---------------------------------------------------------------------------------


def test_reach_without_a_task_set_refuses_with_rl0120(fixture_root: Path) -> None:
    grader = grader_for(Surface.CLOCK, fixture_root)
    subject = FakeSubject(grader_path=grader, tasks_path=None)
    with pytest.raises(ReachRequiresTaskSet) as raised:
        ReachSurfaces().run(subject, FakeCorpus(tasks=()), context())
    error = raised.value
    assert error.code == "RL0120"
    assert error.exit_code == 4
    assert subject.name in error.message
    assert error.context["n_tasks"] == 0
    assert "--tasks" in error.remediation


def test_the_refusal_is_a_usage_error_so_the_cli_exits_four() -> None:
    assert issubclass(ReachRequiresTaskSet, contracts.UsageError)
    assert ReachRequiresTaskSet.default_exit_code == 4


def test_a_probe_that_raises_reaches_the_audit_as_a_could_not_check_hole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instrument does not swallow it: the audit catches it and names it."""
    from reward_lens.api import requests as api
    from reward_lens.product import audit

    def blow_up(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the witness harness could not be staged")

    monkeypatch.setattr(reach.probes, "probe_surfaces", blow_up)
    grader = write_grader(tmp_path, "grader.py", PLAIN)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(TASK) + "\n", encoding="utf-8")
    record = audit.run(
        api.AuditRequest(path=grader, tasks=tasks), project=None, sandbox=default_sandbox()
    )
    holes = [
        entry
        for entry in record.measurement.reach
        if entry.kind == "absence" and entry.absence.state == "COULD_NOT_CHECK"
    ]
    assert any("could not be staged" in hole.absence.missing_access for hole in holes)


def test_the_instrument_does_not_catch_what_the_audit_is_meant_to_catch(
    fixture_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blow_up(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("staging failed")

    monkeypatch.setattr(reach.probes, "probe_surfaces", blow_up)
    with pytest.raises(RuntimeError, match="staging failed"):
        ReachSurfaces().run(
            FakeSubject(grader_path=grader_for(Surface.CLOCK, fixture_root)),
            corpus_with_a_task(),
            context(),
        )


# --- what the record will accept ------------------------------------------------------------------


@pytest.mark.parametrize("surface", list(Surface))
def test_every_entry_validates_as_the_record_model(surface: Surface, fixture_root: Path) -> None:
    entry = entry_for(surface, grader_for(surface, fixture_root))
    assert contracts.Entry.model_validate(entry.to_dict())


def test_no_tier_is_claimed_that_the_run_did_not_establish(fixture_root: Path) -> None:
    ctx = context()
    entry = entry_for(Surface.CLOCK, grader_for(Surface.CLOCK, fixture_root))
    assert entry.provenance.sandbox_tier == ctx.tier()
    assert entry.result["sandbox_tier"] == ctx.tier()


# --- the probe runs under the sandbox the runner supplied -----------------------------------------


@dataclass
class RecordingSandbox:
    """The run's sandbox, which records what the panel asked of it.

    It delegates to a real T0 sandbox so a program actually runs, and reports a tier of its own so
    that a panel reading the tier off `default_sandbox()` instead of off this object is caught: the
    label under test is one no sandbox this process would build by itself carries.
    """

    inner: Any
    tier: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        argv: list[str],
        *,
        limits: Any,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
        write_roots: Any = (),
    ) -> Any:
        self.calls.append(
            {
                "argv": [str(a) for a in argv],
                "cwd": str(cwd),
                "write_roots": [str(r) for r in write_roots],
            }
        )
        result = self.inner.run(
            argv, limits=limits, cwd=cwd, env=env, stdin=stdin, write_roots=write_roots
        )
        result.tier = self.tier
        return result


def recording_context() -> RecordingSandbox:
    real = default_sandbox()
    label = "L0" if real.tier != "L0" else "L1"
    return RecordingSandbox(inner=default_sandbox(require_tier="T0"), tier=label)


def test_the_panel_runs_under_the_sandbox_the_run_context_carries(fixture_root: Path) -> None:
    """`RunContext` carries the sandbox (interfaces section 2), and this is the panel using it."""
    recorder = recording_context()
    ctx = RunContext(sandbox=recorder, offline=True)
    produced = ReachSurfaces().run(
        FakeSubject(grader_path=grader_for(Surface.CLOCK, fixture_root)),
        corpus_with_a_task(),
        ctx,
    )
    assert recorder.calls, "the panel ran the witness through the run's own sandbox"
    assert len(recorder.calls) == 1, "one witness run answers for every surface"
    call = recorder.calls[0]
    assert call["argv"][0].endswith("python") or "python" in call["argv"][0]
    assert any(root.endswith(Surface.CLOCK.value) for root in call["write_roots"]), (
        "the grader's own directory is bound so there is a graded process at all"
    )
    assert recorder.tier != default_sandbox().tier, "the label is not the process default's"
    for entry in produced:
        assert entry.provenance.sandbox_tier == recorder.tier
        assert entry.result["sandbox_tier"] == recorder.tier


def test_the_tier_is_the_one_the_run_established_not_the_machine_best(fixture_root: Path) -> None:
    """A run that carries no sandbox is a T0 run, and the probe is made to be one."""
    ctx = RunContext(sandbox=None, offline=True)
    assert ctx.tier() == "T0"
    entry = ReachSurfaces().run(
        FakeSubject(grader_path=grader_for(Surface.CLOCK, fixture_root)),
        corpus_with_a_task(),
        ctx,
    )[0]
    assert entry.provenance.sandbox_tier == "T0"
    assert entry.result["sandbox_tier"] == "T0"


# --- two runs over the same inputs say the same thing (gate 14) -----------------------------------


def test_every_witness_observation_is_identical_across_two_runs(fixture_root: Path) -> None:
    """What an attempt reports is what it established, never a value that moves between runs.

    Gate 14 compares two records from two audits over identical inputs. A reading of the clock, a
    port, a nonce in a path: each of those makes the section disagree with itself and says nothing
    a reader of the number wanted to know.
    """
    grader = grader_for(Surface.CLOCK, fixture_root)
    first = {i: e.witness.observed for i, e in entries_for(grader).items()}
    second = {i: e.witness.observed for i, e in entries_for(grader).items()}
    assert first == second
    procedures = {i: e.witness.procedure for i, e in entries_for(grader).items()}
    assert procedures == {i: e.witness.procedure for i, e in entries_for(grader).items()}


def test_no_observation_carries_a_clock_reading(fixture_root: Path) -> None:
    entry = entry_for(Surface.CLOCK, grader_for(Surface.CLOCK, fixture_root))
    said = entry.witness.observed
    assert "time.time()" in said, "the attempt says what it called"
    assert not re.search(r"\d{9,}", said), f"a wall-clock reading reached the record: {said}"
    assert "the reading itself is not recorded" in said


# --- subject_ref through a real run, with the probes live -----------------------------------------


def test_a_reach_entry_carries_the_runners_own_subject_ref(tmp_path: Path) -> None:
    """No monkeypatch: the panel runs inside `audit.run` and its digest is the runner's.

    `run.py` builds `subject_ref` once from the subject's source and every entry in the record
    carries that one value. A panel that digested something of its own would put two references to
    the same subject in one record.
    """
    from reward_lens.api import requests as api
    from reward_lens.product import audit

    demo = Path(reach.__file__).resolve().parents[2] / "examples" / "code_reward" / "grader.py"
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(TASK) + "\n", encoding="utf-8")
    record = audit.run(
        api.AuditRequest(path=demo, tasks=tasks), project=None, sandbox=default_sandbox()
    )
    mine = [e for e in record.measurement.reach if e.entry_id.startswith("reach.surfaces.")]
    assert len(mine) == len(Surface), "the panel ran inside the audit, with its probes live"
    runners: set[str] = set()
    for section in type(record.measurement).model_fields:
        if section == "reach":
            continue
        for entry in getattr(record.measurement, section, None) or []:
            reference = getattr(entry, "subject_ref", None)
            if reference:
                runners.add(str(reference))
    assert runners, "the runner wrote a subject_ref somewhere outside the reach section"
    assert {entry.subject_ref for entry in mine} == runners
