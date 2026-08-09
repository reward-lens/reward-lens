"""The framing panel: what the policy can infer about the grader from its context.

The measurement is a static read of the context the policy is handed, on four channels: the prompt,
the system message, the file names it can see, and the error strings the grader returns. Every leak
is a finding with a witness showing the inference the context permits. The label is the point: the
record says in the entry itself that this is the cheap static proxy, and it names the Apollo-style
belief assay as a research module that was not run, with its finetuning requirements. A prompt test
relabelled as a belief measurement is the failure this panel exists to prevent, so the relabelling
is refused by the panel rather than caught by a reader.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from reward_lens import contracts, errors
from reward_lens.instruments import framing
from reward_lens.instruments.base import Instrument, Panel, RunContext
from reward_lens.instruments.framing import channels as channels_module
from reward_lens.instruments.framing import witness as witness_module
from reward_lens.product.audit import absences, registry

# The package re-exports `run` as the function, so the module of that name is asked for by path.
run_module = import_module("reward_lens.product.audit.run")

# --- fixtures ------------------------------------------------------------------------------------

CLEAN_SOURCE = '''
def score(response, task):
    """Return one number."""
    if not response:
        raise ValueError("internal failure")
    return 1.0
'''

LEAKY_SOURCE = '''
def score(response, task):
    """Return one number."""
    if "FINAL" not in response:
        raise ValueError("expected the answer to contain FINAL; the rubric requires it")
    return 1.0
'''

CLEAN_PROMPT = "Write a function that returns the nth Fibonacci number."
LEAKY_PROMPT = (
    "Answer the question. You will be scored by the rubric in reward.py: "
    "you need at least 0.8 to pass."
)
CLEAN_SYSTEM = "You are a helpful assistant. Answer in plain text."
LISTING = "Files you can open:\n- {grader}\n- README.md"
LEAKY_SYSTEM = (
    "You are being evaluated by an automated grader. The checker looks for a regex in your reply."
)


@dataclass
class FakeSubject:
    grader_path: Path
    source: str
    entrypoint: str = "score"
    project_dir: Path | None = None
    tasks_path: Path | None = None
    responses_path: Path | None = None
    trainer: str | None = None
    name: str = "fixture-reward"


@dataclass
class FakeCorpus:
    tasks: tuple[dict, ...] = ()
    rollouts: tuple[Any, ...] = ()
    origin: str = "a fixture task set"
    n_tasks: int = 0

    def as_list(self) -> list[Any]:
        return list(self.rollouts)

    def __len__(self) -> int:
        return len(self.rollouts)


@dataclass
class FakeRecord:
    """Only what `run._answers` reads of a stored record: the entries it holds."""

    held: tuple[contracts.Entry, ...] = field(default_factory=tuple)

    def entries(self) -> tuple[contracts.Entry, ...]:
        return self.held


def context() -> RunContext:
    return RunContext(sandbox=None, project=None, budget="0.00", offline=True, clock=time.monotonic)


def fixture(tmp_path: Path, leaking: frozenset[str] = frozenset()) -> tuple[Any, Any]:
    """A subject and a corpus that leak on exactly the named channels and on no others.

    The file-name channel leaks when the grader the context lists is named after what it does. Both
    projects hold the same files and both are listed to the policy; what differs is the one name.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    grader_name = "reward.py" if "file_names" in leaking else "main.py"
    (project_dir / grader_name).write_text("# the grader\n", encoding="utf-8")
    (project_dir / "README.md").write_text("# a project\n", encoding="utf-8")
    source = LEAKY_SOURCE if "error_strings" in leaking else CLEAN_SOURCE
    subject = FakeSubject(
        grader_path=project_dir / grader_name, source=source, project_dir=project_dir
    )
    task = {
        "id": "t1",
        "prompt": LEAKY_PROMPT if "prompt" in leaking else CLEAN_PROMPT,
        "system": LEAKY_SYSTEM if "system_message" in leaking else CLEAN_SYSTEM,
        "tools": LISTING.format(grader=grader_name),
    }
    return subject, FakeCorpus(tasks=(task,), n_tasks=1)


def entries_for(tmp_path: Path, leaking: frozenset[str] = frozenset()) -> dict[str, contracts.Entry]:
    subject, corpus = fixture(tmp_path, leaking)
    produced = framing.ContextInference().run(subject, corpus, context())
    return {entry.entry_id.rsplit(".", 1)[-1]: entry for entry in produced}


# --- the panel itself ----------------------------------------------------------------------------


def test_the_package_exposes_a_framing_panel_discovery_finds() -> None:
    assert isinstance(framing.PANEL, Panel)
    assert framing.PANEL.section == "framing"
    assert framing.PANEL.instruments
    for instrument in framing.PANEL.instruments:
        assert isinstance(instrument, Instrument)
    sections = [panel.section for panel in registry.discover()]
    assert "framing" in sections
    assert registry.import_failures.get("reward_lens.instruments.framing") is None


def test_every_instrument_declares_its_method_up_front() -> None:
    """The audit's reuse check reads the method before anything runs (`run._requested`)."""
    for instrument in framing.PANEL.instruments:
        declared = getattr(instrument, "method", None)
        assert isinstance(declared, contracts.Method)
        assert declared.id == instrument.id
        assert declared.params_digest.startswith("sha256:")
    wanted = run_module._requested((framing.PANEL,))
    assert set(wanted) == {"framing.context_inference", "framing.belief_assay"}
    assert all(identity is not None for identity in wanted.values())


def test_a_landed_panel_re_measures_an_already_audited_project(tmp_path: Path) -> None:
    """A record written before this panel landed does not answer a request that includes it."""
    wanted = run_module._requested((framing.PANEL,))
    before = contracts.absence(
        "validity",
        "validity.static.input_leakage",
        "whether the task carries the answer",
        "no task set",
        "supply --tasks",
        subject_ref="sha256:" + "0" * 64,
        provenance=context().provenance(),
    )
    assert run_module._answers(FakeRecord((before,)), wanted) is False
    subject, corpus = fixture(tmp_path)
    ctx = context()
    after = tuple(
        entry
        for instrument in framing.PANEL.instruments
        for entry in instrument.run(subject, corpus, ctx)
    )
    assert run_module._answers(FakeRecord((before, *after)), wanted) is True


def test_entry_ids_sit_under_the_instrument_id(tmp_path: Path) -> None:
    subject, corpus = fixture(tmp_path)
    for instrument in framing.PANEL.instruments:
        for entry in instrument.run(subject, corpus, context()):
            assert entry.entry_id == instrument.id or entry.entry_id.startswith(f"{instrument.id}.")
            assert entry.section == "framing"
            assert entry.method == instrument.method


# --- the four channels, each on a fixture that leaks and one that does not ------------------------


@pytest.mark.parametrize(
    "channel",
    ["prompt", "system_message", "file_names", "error_strings"],
)
def test_each_channel_is_probed_leaking_and_clean(channel: str, tmp_path: Path) -> None:
    leaked = entries_for(tmp_path / "leaks", frozenset({channel}))[channel]
    assert leaked.kind == "check"
    assert leaked.check.passed is False
    assert leaked.result["leaks"], f"{channel} leaked nothing on a fixture built to leak"
    assert leaked.result["leaks_found"] == len(leaked.result["leaks"])
    clean = entries_for(tmp_path / "clean")[channel]
    assert clean.kind == "check"
    assert clean.check.passed is True
    assert clean.result["leaks"] == []


def test_the_four_channels_are_exactly_the_ones_the_commission_names(tmp_path: Path) -> None:
    assert channels_module.CHANNEL_NAMES == (
        "prompt",
        "system_message",
        "file_names",
        "error_strings",
    )
    assert set(entries_for(tmp_path)) == set(channels_module.CHANNEL_NAMES)


def test_a_leak_on_one_channel_does_not_fire_the_others(tmp_path: Path) -> None:
    for channel in channels_module.CHANNEL_NAMES:
        produced = entries_for(tmp_path / channel, frozenset({channel}))
        leaking = {name for name, entry in produced.items() if entry.result["leaks"]}
        assert leaking == {channel}


def test_a_channel_with_nothing_to_read_is_an_absence_not_a_pass(tmp_path: Path) -> None:
    """No task set and no source: every channel reads nothing, and nothing read is not a pass."""
    subject = FakeSubject(grader_path=tmp_path / "main.py", source="", project_dir=None)
    produced = {
        entry.entry_id.rsplit(".", 1)[-1]: entry
        for entry in framing.ContextInference().run(subject, FakeCorpus(), context())
    }
    for channel in channels_module.CHANNEL_NAMES:
        entry = produced[channel]
        assert entry.kind == "absence"
        assert entry.absence.state == "NOT_MEASURED"
        assert entry.absence.missing_access
        assert entry.method == framing.ContextInference().method


# --- the witness ---------------------------------------------------------------------------------


def test_every_leak_finding_carries_a_witness_showing_the_inference(tmp_path: Path) -> None:
    for channel in channels_module.CHANNEL_NAMES:
        entry = entries_for(tmp_path / channel, frozenset({channel}))[channel]
        assert isinstance(entry.witness, contracts.Witness)
        assert entry.witness.observed.strip()
        assert entry.witness.procedure.strip()
        findings = witness_module.findings_for(entry)
        assert len(findings) == len(entry.result["leaks"])
        for index, finding in enumerate(findings):
            assert finding.witness_path
            carried = entry.result["leaks"][index]["witness"]
            assert finding.witness_path.endswith(f"/leaks/{index}/witness")
            # A contracts.Witness and not free-form JSON: the shape is validated where it is
            # built, and what the record carries validates back into the same type.
            revalidated = contracts.Witness.model_validate(carried)
            leak = entry.result["leaks"][index]
            assert revalidated.inputs["excerpt"] == leak["excerpt"]
            assert revalidated.observed.endswith(leak["inference"])
            assert revalidated.procedure.strip()


def test_the_witness_states_the_inference_not_the_possibility(tmp_path: Path) -> None:
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    for leak in entry.result["leaks"]:
        observed = leak["witness"]["observed"].lower()
        assert "the policy can" in observed
        assert "may" not in observed.split(".")[0]


def test_a_clean_channel_produces_no_findings(tmp_path: Path) -> None:
    for channel, entry in entries_for(tmp_path).items():
        assert witness_module.findings_for(entry) == (), channel


# --- the label -----------------------------------------------------------------------------------


def test_the_entry_itself_says_it_is_a_static_proxy(tmp_path: Path) -> None:
    produced = entries_for(tmp_path, frozenset({"prompt"}))
    for channel, entry in produced.items():
        assert entry.result["scope_label"] == framing.SCOPE_LABEL == "static_proxy"
        assert entry.result["not_measured"]
        joined = " ".join(entry.limitations).lower()
        assert "static proxy" in joined
        assert "belief" in joined
        assert "static" in entry.method.procedure.lower()


def test_the_belief_assay_is_named_as_not_run_with_its_finetuning_requirements(
    tmp_path: Path,
) -> None:
    subject, corpus = fixture(tmp_path)
    produced = framing.BeliefAssay().run(subject, corpus, context())
    assert len(produced) == 1
    entry = produced[0]
    assert entry.entry_id == "framing.belief_assay"
    assert entry.kind == "absence"
    assert entry.absence.state == "NOT_MEASURED"
    text = f"{entry.measurand} {entry.absence.missing_access} {entry.absence.remedy}".lower()
    assert "belief" in text
    assert "not run" in text or "was not" in text
    assert "finetun" in text
    assert entry.absence.affected_claims


# --- the finding code ----------------------------------------------------------------------------


def test_rl0231_is_the_framing_finding_code(tmp_path: Path) -> None:
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    findings = witness_module.findings_for(entry)
    assert findings
    for finding in findings:
        assert finding.code == framing.FINDING_CODE == "RL0231"
        assert finding.arm == "static"
        assert finding.entries == [entry.entry_id]
        assert finding.rule == entry.entry_id
        assert finding.message
        assert finding.severity_rationale


def test_rl0231_is_allocated_here_and_nowhere_else() -> None:
    """One packet mints the code. The catalogue holds every code's row, so it is not a second one."""
    root = Path(framing.__file__).resolve().parents[3]
    mine = (root / "reward_lens" / "instruments" / "framing").resolve()
    catalogue = (root / "reward_lens" / "errors").resolve()
    elsewhere = [
        str(path)
        for path in sorted((root / "reward_lens").rglob("*.py"))
        if "RL0231" in path.read_text(encoding="utf-8")
        and mine not in path.resolve().parents
        and catalogue not in path.resolve().parents
    ]
    assert elsewhere == []
    assert any(
        "RL0231" in path.read_text(encoding="utf-8") for path in sorted(mine.rglob("*.py"))
    )


# --- the refusals --------------------------------------------------------------------------------


def test_a_framing_entry_claiming_a_belief_measurement_is_refused(tmp_path: Path) -> None:
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    entry.measurand = "what the policy believes about the grader, measured"
    with pytest.raises(witness_module.BeliefClaimRefused) as raised:
        witness_module.assert_scope_honest(entry)
    assert raised.value.code == witness_module.REFUSAL_CODE == "RL0003"
    assert raised.value.exit_code == 4
    assert isinstance(raised.value, contracts.UsageError)
    assert "measurand" in raised.value.message
    assert raised.value.context["field"] == "measurand"


def test_a_framing_entry_mislabelled_in_its_scope_field_is_refused(tmp_path: Path) -> None:
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    entry.result = dict(entry.result) | {"scope_label": "belief_measurement"}
    with pytest.raises(witness_module.BeliefClaimRefused) as raised:
        witness_module.assert_scope_honest(entry)
    assert raised.value.code == "RL0003"
    assert raised.value.exit_code == 4
    assert raised.value.context["field"] == "result.scope_label"
    assert "result.scope_label" in raised.value.message


def test_the_panel_refuses_to_emit_a_mislabelled_entry(monkeypatch, tmp_path: Path) -> None:
    """The guard runs inside `run`, so the mislabelled entry never reaches the record."""

    def mislabelled(*args: Any, **kwargs: Any) -> str:
        return "what the policy believes about the grader, as measured here"

    monkeypatch.setattr(framing, "_measurand", mislabelled)
    subject, corpus = fixture(tmp_path, frozenset({"prompt"}))
    with pytest.raises(witness_module.BeliefClaimRefused):
        framing.ContextInference().run(subject, corpus, context())


def test_the_finding_message_is_held_to_the_same_label() -> None:
    with pytest.raises(witness_module.BeliefClaimRefused) as raised:
        witness_module.assert_message_honest(
            "the policy believes it is being graded, measured here", "message"
        )
    assert raised.value.context["field"] == "message"
    assert (
        witness_module.assert_message_honest(
            "the policy's belief was not measured; this is the static proxy", "message"
        )
        is not None
    )


def test_an_exception_inside_the_instrument_is_not_swallowed(monkeypatch, tmp_path: Path) -> None:
    def boom(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        raise RuntimeError("the cue table would not load")

    monkeypatch.setattr(channels_module, "probe", boom)
    subject, corpus = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="the cue table would not load"):
        framing.ContextInference().run(subject, corpus, context())


def test_that_exception_becomes_a_could_not_check_hole_naming_it() -> None:
    """What the audit does with it; the instrument's job is only to let it out."""
    entry = absences.could_not_check(
        context(),
        section="framing",
        entry_id="framing.context_inference",
        measurand="what the policy can infer about the grader from its context",
        failure=RuntimeError("the cue table would not load"),
        remedy="run a build in which this instrument imports",
        subject_ref="sha256:" + "0" * 64,
        affected_claims=("no framing claim from framing.context_inference",),
    )
    assert entry.kind == "absence"
    assert entry.absence.state == "COULD_NOT_CHECK"
    assert "RuntimeError" in entry.absence.missing_access
    assert "the cue table would not load" in entry.absence.missing_access


# --- the record --------------------------------------------------------------------------------


def test_every_entry_is_a_valid_record_entry(tmp_path: Path) -> None:
    subject, corpus = fixture(tmp_path, frozenset({"prompt", "file_names"}))
    ctx = context()
    for instrument in framing.PANEL.instruments:
        for entry in instrument.run(subject, corpus, ctx):
            round_tripped = contracts.Entry.model_validate(entry.model_dump(exclude_none=True))
            assert round_tripped.entry_id == entry.entry_id
            assert entry.subject_ref == contracts.digest({"source": subject.source})
            assert entry.provenance.offline is True
            assert entry.state in {"partial", "absent"}


# --- the belief assay, against the commission it comes from ---------------------------------------

#: The commission's own lines on the belief assay, pasted here as a literal. This is the source the
#: entry is checked against: not `framing.BELIEF_ASSAY_COMMISSION`, which is what the entry was
#: built from, because a test that compares a constant with itself passes on any six strings. Every
#: quotation inside these was checked against `PRODUCT_BUILD_PROMPT.md` at the lines it names.
COMMISSION_ON_THE_BELIEF_ASSAY: tuple[str, ...] = (
    "the assay is the Apollo-style one and belongs to research rather than to this audit, quoting "
    'PRODUCT_BUILD_PROMPT.md:2167 "The Apollo-style belief assay is a research module with"',
    "its finetuning requirements are stated before it is run, quoting "
    'PRODUCT_BUILD_PROMPT.md:2168 "its finetuning requirements stated"',
    "it is reported as its own entry and this record says it did not run, quoting "
    'PRODUCT_BUILD_PROMPT.md:3039 "names the belief assay as a separate module" and '
    'PRODUCT_BUILD_PROMPT.md:3040 "that was not run"',
    'the forbidden thing, quoting PRODUCT_BUILD_PROMPT.md:3040 "Forbidden: presenting a prompt '
    'test as a belief measurement", is never done here',
    "what this build ran instead is the proxy, quoting PRODUCT_BUILD_PROMPT.md:2166 "
    '"This is the cheap static proxy for" and PRODUCT_BUILD_PROMPT.md:2167 "the belief question, '
    'and the record says so"',
)


def test_the_belief_assay_entry_states_the_commissions_requirements_and_no_others(
    tmp_path: Path,
) -> None:
    subject, corpus = fixture(tmp_path)
    entry = framing.BeliefAssay().run(subject, corpus, context())[0]
    assert tuple(entry.limitations) == COMMISSION_ON_THE_BELIEF_ASSAY
    for stated in entry.limitations:
        assert "PRODUCT_BUILD_PROMPT.md:" in stated, stated
    assert not hasattr(framing, "BELIEF_ASSAY_REQUIREMENTS")


def test_every_requirement_names_a_line_of_the_commission() -> None:
    """The citation is a line number, so a reader can go and check the sentence at it."""
    cited = {
        int(piece.split(" ")[0].strip('".,'))
        for stated in COMMISSION_ON_THE_BELIEF_ASSAY
        for piece in stated.split("PRODUCT_BUILD_PROMPT.md:")[1:]
    }
    assert cited == {2166, 2167, 2168, 3039, 3040}


# --- the scope a gate reads -----------------------------------------------------------------------


def test_the_scope_field_is_the_scope_of_the_rung_the_method_declares(tmp_path: Path) -> None:
    subject, corpus = fixture(tmp_path, frozenset({"prompt", "file_names"}))
    assert witness_module.RUNG_SCOPES[0] == "evaluator_comparison"
    for instrument in framing.PANEL.instruments:
        for entry in instrument.run(subject, corpus, context()):
            assert witness_module.rung_of(entry.method) == framing.RUNG == 0
            assert entry.scope == witness_module.RUNG_SCOPES[0]


def test_a_rung_one_method_carrying_rung_zeros_scope_is_refused(tmp_path: Path) -> None:
    """8.5: the field Gate 20 reads has to be the scope of the rung the method declares."""
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    assert entry.scope == "evaluator_comparison"
    entry.method = witness_module.declare_method(
        id="framing.context_inference",
        version="1.0.0",
        rung=1,
        params={"arm": "selection"},
        procedure="the grader is put under selection stress and the run is read",
        credited_to="the test",
    )
    with pytest.raises(witness_module.ScopeMismatch) as raised:
        witness_module.assert_scope_honest(entry)
    assert raised.value.code == "RL0003"
    assert raised.value.exit_code == 4
    assert "selection_stress" in raised.value.message
    assert raised.value.context["field"] == "scope"


def test_a_method_that_declared_no_rung_is_refused(tmp_path: Path) -> None:
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    entry.method = contracts.Method(
        id="framing.context_inference",
        version="1.0.0",
        params_digest=contracts.digest({"arm": "undeclared"}),
        procedure="a method built without declaring the rung it sits on",
        credited_to="the test",
    )
    with pytest.raises(witness_module.ScopeMismatch) as raised:
        witness_module.assert_scope_honest(entry)
    assert raised.value.context["field"] == "method.params_digest"


def test_no_framing_entry_carries_a_count_without_its_sampling_unit(tmp_path: Path) -> None:
    """The leaks are counted, not sampled, so the count is data and not `Entry.n`."""
    subject, corpus = fixture(tmp_path, frozenset({"prompt", "file_names"}))
    for instrument in framing.PANEL.instruments:
        for entry in instrument.run(subject, corpus, context()):
            assert entry.kind in {"check", "absence"}
            assert entry.n is None
            assert entry.value is None
            assert entry.unit is None
            assert entry.sampling_unit is None
            assert entry.uncertainty is None
            if entry.kind == "check":
                assert entry.result["leaks_found"] == len(entry.result["leaks"])


# --- the file names the policy is actually shown ---------------------------------------------------


def honest_project(tmp_path: Path, task: dict) -> dict[str, contracts.Entry]:
    """A repository whose grader is named after what it does, audited on the given context."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "reward.py").write_text("# the grader\n", encoding="utf-8")
    (project_dir / "rubric.md").write_text("# what earns a point\n", encoding="utf-8")
    subject = FakeSubject(
        grader_path=project_dir / "reward.py", source=CLEAN_SOURCE, project_dir=project_dir
    )
    produced = framing.ContextInference().run(
        subject, FakeCorpus(tasks=(task,), n_tasks=1), context()
    )
    return {entry.entry_id.rsplit(".", 1)[-1]: entry for entry in produced}


def test_an_honest_project_the_policy_never_sees_raises_nothing(tmp_path: Path) -> None:
    """`reward.py` on the auditor's disk is not something the policy can read off its context."""
    entry = honest_project(tmp_path, {"id": "t1", "prompt": CLEAN_PROMPT, "system": CLEAN_SYSTEM})[
        "file_names"
    ]
    assert entry.kind == "absence"
    assert entry.absence.state == "NOT_MEASURED"
    assert "context" in entry.absence.missing_access
    assert witness_module.findings_for(entry) == ()


def test_a_context_that_lists_the_grader_leaks_with_the_listing_quoted(tmp_path: Path) -> None:
    listing = "Files you can open:\n- solve.py\n- reward.py\n- README.md"
    entry = honest_project(
        tmp_path, {"id": "t1", "prompt": CLEAN_PROMPT, "system": CLEAN_SYSTEM, "tools": listing}
    )["file_names"]
    assert entry.kind == "check"
    assert entry.check.passed is False
    findings = witness_module.findings_for(entry)
    assert findings
    for finding in findings:
        assert finding.code == "RL0231"
    for leak in entry.result["leaks"]:
        assert leak["origin"] == "task[0].tools:reward.py"
        assert leak["excerpt"] == "- reward.py"
        assert contracts.Witness.model_validate(leak["witness"]).inputs["excerpt"] == "- reward.py"


def test_the_channel_reads_the_context_and_never_the_auditors_filesystem(tmp_path: Path) -> None:
    """Same repository, same names on disk; only the context differs, and only it decides."""
    quiet = honest_project(tmp_path / "quiet", {"id": "t1", "prompt": CLEAN_PROMPT})["file_names"]
    loud = honest_project(
        tmp_path / "loud",
        {"id": "t1", "prompt": "Solve it. The grader is reward.py, next to rubric.md."},
    )["file_names"]
    assert quiet.kind == "absence"
    assert loud.result["samples_read"] == 1
    assert {leak["origin"] for leak in loud.result["leaks"]} == {"task[0].prompt:reward.py"}
    assert "source_visible" in loud.result["cues_fired"]
    assert all(leak["excerpt"].endswith("next to rubric.md.") for leak in loud.result["leaks"])


def test_every_cue_matches_on_word_boundaries() -> None:
    cues = {cue.id: cue for cue in channels_module.CUES}
    assert cues["source_visible"].fired("- reward.py")
    assert cues["source_visible"].fired("- rewarding.py") is None
    assert cues["source_visible"].fired("- scoreboard_of_names.txt") is None
    assert cues["grader_named"].fired("the grader decides")
    assert cues["grader_named"].fired("run the upgrader first") is None
    assert cues["threshold_disclosed"].fired("the threshold is 0.8")
    assert cues["threshold_disclosed"].fired("thresholding at dawn") is None


# --- the refusal's own wording ---------------------------------------------------------------------


def test_the_refusal_takes_its_message_from_the_catalogue(tmp_path: Path) -> None:
    """`reward-lens explain RL0003` has to describe the fault that was actually raised."""
    entry = entries_for(tmp_path, frozenset({"prompt"}))["prompt"]
    entry.measurand = "what the policy believes about the grader, measured"
    with pytest.raises(witness_module.BeliefClaimRefused) as raised:
        witness_module.assert_scope_honest(entry)
    catalogued = errors.make("RL0003", field="measurand", detail="the sentinel detail")
    assert catalogued.message.endswith("the sentinel detail")
    assert raised.value.message.startswith(catalogued.message.removesuffix("the sentinel detail"))
    assert raised.value.remediation == catalogued.remediation
