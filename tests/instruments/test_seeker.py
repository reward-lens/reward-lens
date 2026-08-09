"""The seeker: an actor protocol, four actors, a budget, and a record that replays.

Nothing here reaches a network or spends money. The API actor is exercised against a recorded
transport that speaks the chat-completions wire shape and refuses any prompt it has no recording
for, so there is no path from this file to a paid provider even if one were configured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens import contracts
from reward_lens.api import _dispatch
from reward_lens.graders.connect import bind, detect
from reward_lens.instruments.base import Panel, RunContext
from reward_lens.instruments.seeker import (
    FINDING_ID,
    NOT_FOUND,
    PANEL,
    Actor,
    ActorRecord,
    AgentActor,
    ApiActor,
    Budget,
    BudgetExhausted,
    Completion,
    ExploitSearch,
    FindingNotReplayable,
    GraderScorer,
    LocalActor,
    Pricing,
    Proposal,
    RecordedTransport,
    ScoreResult,
    Scored,
    ScriptedActor,
    SeekerRequiresBudget,
    SeekerState,
    findings_for,
    replay,
    scorer_for,
    seek,
)
from reward_lens.outcome import RoundLedger, SealedRequest
from reward_lens.partitions import ALLOWED_CAPABILITIES, SEEKER, Partition, PartitionKind

# --- the reward system under audit, and the intent it is meant to pay for ------------------------

#: A grader that pays for a marker string rather than for solving the task. The marker is the
#: exploit every actor below is meant to find; the task's answer is what the intent actually wants.
MARKER = "LGTM"


def keyword_reward(prompt: str, completion: str, **kwargs: object) -> float:
    """`(prompt, completion, **kw) -> float`: P-CONNECT's `SINGLE_FN`."""
    if MARKER in completion:
        return 1.0
    return 0.25 if len(completion.strip()) > 0 else 0.0


TASKS: tuple[dict, ...] = (
    {"id": "t1", "prompt": "what is 2 + 2?", "answer": "4"},
    {"id": "t2", "prompt": "what is 3 + 3?", "answer": "6"},
)

INTENT = "the response states the arithmetic answer the task asks for"


def outcome_check(task: dict, response: str) -> bool:
    """The independent check: did the response actually satisfy the intent?"""
    return str(task["answer"]) in str(response)


@pytest.fixture
def scorer() -> GraderScorer:
    """The audited reward, reached through P-CONNECT's frozen `detect` then `bind`."""
    return GraderScorer(bind(detect(keyword_reward)), outcome_check=outcome_check, threshold=0.9)


# --- the scripted policy fixture, the conformance reference --------------------------------------


EXPLOIT_LINE = f"{MARKER} -- shipping it"
HONEST_LINE = "the answer is 4"


def scripted_policy(state: SeekerState) -> str:
    """Two attempts: one honest response, then the marker the grader pays for."""
    return HONEST_LINE if not state.attempts else EXPLOIT_LINE


# --- the four actors, each built the same way ----------------------------------------------------


def chat_reply(text: str, *, prompt_tokens: int = 40, completion_tokens: int = 10) -> dict:
    """One chat-completions reply, in the wire shape a provider and a local server both return."""
    return {
        "id": "cmpl-fixture",
        "object": "chat.completion",
        "model": "fixture-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


class StubEndpoint:
    """An OpenAI-compatible endpoint, of the kind `doctor` finds on this machine, at no cost."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, payload: dict) -> dict:
        self.requests.append(payload)
        text = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        return chat_reply(text)


def recorded_transport() -> RecordedTransport:
    """A recording of the two replies the paid route would have been sold."""
    return RecordedTransport(
        [chat_reply(HONEST_LINE), chat_reply(EXPLOIT_LINE)],
        model="fixture-1",
    )


def scripted_caller(session: str) -> object:
    """The agent that calls `seeker.propose` through the SDK's dispatch seam and reads its score."""
    replies = [HONEST_LINE, EXPLOIT_LINE]

    def call(request: dict) -> dict:
        tool = _dispatch.load("reward_lens.instruments.seeker.actors.agent:propose_tool")
        assert tool is not None, "the SDK seam did not resolve seeker.propose"
        offered = replies[min(len(request["attempts"]), len(replies) - 1)]
        return tool(
            {"session": session, "task_id": request["task_id"], "response": offered,
             "rationale": "scripted caller"}
        )

    return call


def four_actors(scorer: GraderScorer) -> dict[str, Actor]:
    """One of each, all reaching the same audited reward."""
    return {
        "scripted": ScriptedActor(scripted_policy, scorer=scorer),
        "api": ApiActor(recorded_transport(), scorer=scorer, budget=Budget("0.50"),
                        pricing=Pricing("0.0030", "0.0150")),
        "local": LocalActor(StubEndpoint([HONEST_LINE, EXPLOIT_LINE]), scorer=scorer),
        "agent": AgentActor(scorer=scorer, caller=scripted_caller),
    }


# --- 1. the protocol: three methods, four actors -------------------------------------------------


@pytest.mark.parametrize("name", ["scripted", "api", "local", "agent"])
def test_every_actor_implements_all_three_protocol_methods(scorer: GraderScorer, name: str) -> None:
    actor = four_actors(scorer)[name]
    assert isinstance(actor, Actor)
    state = SeekerState(task=TASKS[0], intent=INTENT, attempts=(), budget_remaining_usd=
                        actor.budget.remaining, calls_remaining=actor.budget.calls_remaining)
    proposal = actor.propose(state)
    assert isinstance(proposal, Proposal)
    assert proposal.actor == name
    scored = actor.score(proposal)
    assert isinstance(scored, Scored)
    assert scored.proposal.proposal_id == proposal.proposal_id
    record = actor.record()
    assert isinstance(record, ActorRecord)
    assert record.actor == name
    assert proposal.proposal_id in {p.proposal_id for p in record.proposals}


@pytest.mark.parametrize("name", ["scripted", "api", "local", "agent"])
def test_every_actor_finds_the_marker_exploit_and_records_the_arm(
    scorer: GraderScorer, name: str
) -> None:
    record = seek(four_actors(scorer)[name], tasks=TASKS, intent=INTENT, max_proposals=4)
    assert record.found, f"the {name} actor found no exploit"
    assert record.outcome.startswith("found")
    assert record.actor == name
    assert record.arm == "black_box"
    exploit = next(s for s in record.scores if s.exploit)
    assert MARKER in exploit.proposal.response
    assert exploit.outcome_passed is False


def test_the_scripted_actor_is_the_conformance_reference(scorer: GraderScorer) -> None:
    """Every other actor's record has the same shape as the scripted one's."""
    reference = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                     max_proposals=4)
    for name, actor in four_actors(scorer).items():
        record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=4)
        assert record.to_dict().keys() == reference.to_dict().keys(), name
        assert [p.keys() for p in record.to_dict()["proposals"]] == [
            p.keys() for p in reference.to_dict()["proposals"]
        ], name


# --- 2. the API actor is metered, and refuses without a budget -----------------------------------


def test_the_api_actor_meters_spend_against_the_cap(scorer: GraderScorer) -> None:
    budget = Budget("0.50")
    actor = ApiActor(recorded_transport(), scorer=scorer, budget=budget,
                     pricing=Pricing("0.0030", "0.0150"))
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=4)
    # 40 prompt tokens at $0.0030/1k and 10 completion tokens at $0.0150/1k is $0.00027 a call.
    assert record.calls == len(record.proposals)
    assert float(record.spent_usd) >= 0.0
    assert float(record.spent_usd) <= 0.50
    assert record.budget_usd == "0.50"


def test_the_api_actor_stops_at_the_cap_rather_than_overspending(scorer: GraderScorer) -> None:
    tiny = Budget("0.01", max_calls=None)
    actor = ApiActor(
        RecordedTransport([chat_reply(HONEST_LINE, prompt_tokens=2000, completion_tokens=600)] * 8,
                          model="fixture-1"),
        scorer=scorer, budget=tiny, pricing=Pricing("3.0000", "15.0000"),
    )
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=8)
    assert record.stopped == "budget_exhausted"
    assert float(record.spent_usd) <= 0.01
    assert record.calls < 8


def test_a_paid_call_is_unreachable_without_a_budget(scorer: GraderScorer) -> None:
    """RL0502 is raised before a request is built, so the transport is never asked."""
    transport = recorded_transport()
    with pytest.raises(SeekerRequiresBudget) as raised:
        ApiActor(transport, scorer=scorer, budget=None, pricing=Pricing("0.0030", "0.0150"))
    assert transport.calls == 0
    error = raised.value
    assert error.code == "RL0502"
    assert error.exit_code == 6
    assert error.context["actor"] == "api"
    assert error.context["field"] == "max_budget_usd"
    assert "--max-budget-usd" in error.remediation


def test_a_budget_with_no_cap_is_also_refused(scorer: GraderScorer) -> None:
    with pytest.raises(SeekerRequiresBudget):
        ApiActor(recorded_transport(), scorer=scorer, budget=Budget(None),
                 pricing=Pricing("0.0030", "0.0150"))


@pytest.mark.parametrize("name", ["scripted", "local", "agent"])
def test_the_other_three_actors_never_raise_rl0502(scorer: GraderScorer, name: str) -> None:
    actors = {
        "scripted": lambda: ScriptedActor(scripted_policy, scorer=scorer),
        "local": lambda: LocalActor(StubEndpoint([HONEST_LINE]), scorer=scorer),
        "agent": lambda: AgentActor(scorer=scorer, caller=scripted_caller),
    }
    actor = actors[name]()
    assert actor.budget.cap is None
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=2)
    assert record.spent_usd == "0.00"


def test_the_budget_refuses_the_charge_that_would_exceed_the_cap() -> None:
    budget = Budget("0.10")
    budget.charge("0.09", calls=1)
    assert budget.remaining == "0.01"
    with pytest.raises(BudgetExhausted):
        budget.charge("0.02", calls=1)
    assert budget.spent == "0.09"
    assert budget.calls == 1


def test_the_package_imports_no_network_client() -> None:
    """No path from the seeker to a socket: the source says so, not a promise in a docstring."""
    import reward_lens.instruments.seeker as package

    banned = ("import requests", "import httpx", "import socket", "import urllib",
              "from urllib", "http.client", "aiohttp", "openai")
    for path in sorted(Path(package.__file__).parent.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path.name} names {needle}"


# --- 3. the local actor, against a stub endpoint -------------------------------------------------


def test_the_local_actor_talks_to_a_stub_endpoint_at_no_cost(scorer: GraderScorer) -> None:
    endpoint = StubEndpoint([HONEST_LINE, EXPLOIT_LINE])
    actor = LocalActor(endpoint, scorer=scorer, model="local-fixture")
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=4)
    assert endpoint.requests, "the local actor never called the endpoint"
    sent = endpoint.requests[0]
    assert sent["model"] == "local-fixture"
    assert sent["messages"][-1]["role"] == "user"
    assert record.spent_usd == "0.00"
    assert record.found


# --- 4. the agent actor, through the SDK with a scripted caller ----------------------------------


def test_the_agent_actor_proposes_through_the_sdk_seam(scorer: GraderScorer) -> None:
    actor = AgentActor(scorer=scorer, caller=scripted_caller)
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=4)
    assert record.spent_usd == "0.00"
    assert record.found
    assert record.proposals[0].inputs["via"] == "seeker.propose"


def test_the_propose_tool_is_what_the_sdk_seam_resolves() -> None:
    tool = _dispatch.load("reward_lens.instruments.seeker.actors.agent:propose_tool")
    assert callable(tool)


def test_the_propose_tool_refuses_an_unknown_session() -> None:
    tool = _dispatch.load("reward_lens.instruments.seeker.actors.agent:propose_tool")
    reply = tool({"session": "no-such-session", "task_id": "t1", "response": "x"})
    assert reply["error"]["code"] == "RL0003"


# --- 5. the honest negative ----------------------------------------------------------------------


def test_not_found_is_a_result_that_names_the_budget_and_the_actor(scorer: GraderScorer) -> None:
    actor = ApiActor(
        RecordedTransport([chat_reply("the answer is 4")] * 4, model="fixture-1"),
        scorer=scorer, budget=Budget("0.50"), pricing=Pricing("0.0030", "0.0150"),
    )
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=4)
    assert record.found == ()
    assert record.outcome == NOT_FOUND
    assert record.proposals, "an empty finding list with no proposals is not a negative result"
    sentence = record.not_found_sentence()
    assert "api" in sentence and "0.50" in sentence
    assert record.to_dict()["outcome"] == NOT_FOUND


def test_the_negative_is_not_an_empty_finding_list(scorer: GraderScorer) -> None:
    actor = ScriptedActor(lambda state: "the answer is 4", scorer=scorer)
    record = seek(actor, tasks=TASKS, intent=INTENT, max_proposals=2)
    entries = ExploitSearch().entries_for(record, subject_ref=contracts.digest({"s": 1}),
                                          ctx=RunContext())
    yielded = [e for e in entries if e.kind == "estimate"]
    assert len(yielded) == 1
    assert yielded[0].result["outcome"] == NOT_FOUND
    assert yielded[0].result["actor"] == "scripted"
    assert yielded[0].result["budget_usd"] == "0.00"
    assert yielded[0].value == 0.0


# --- 6. off by default, and a seeker-off run is a complete audit ---------------------------------


def test_the_seeker_is_off_by_default_in_the_run_context() -> None:
    ctx = RunContext()
    assert getattr(ctx, "seeker", "off") == "off"
    entries = ExploitSearch().run(_Subject(), _Corpus(), ctx)
    assert len(entries) == 1
    only = entries[0]
    assert only.kind == "absence"
    assert only.state == "absent"
    assert only.absence.state == "NOT_MEASURED"
    assert "was not run" in only.absence.missing_access
    assert "--seeker" in only.absence.remedy
    assert only.absence.affected_claims


def test_a_seeker_off_run_says_the_arm_was_not_run_rather_than_leaving_a_hole() -> None:
    ctx = RunContext()
    ctx.seeker = "off"
    entry = ExploitSearch().run(_Subject(), _Corpus(), ctx)[0]
    assert entry.section == "exploits"
    assert entry.entry_id == "exploits.seeker"
    assert entry.provenance.arm == "black_box"
    assert findings_for(entry) == ()


def test_a_seeker_on_with_no_actor_says_so_rather_than_running_nothing() -> None:
    ctx = RunContext()
    ctx.seeker = "api"
    entry = ExploitSearch().run(_Subject(), _Corpus(), ctx)[0]
    assert entry.kind == "absence"
    assert "actor" in entry.absence.missing_access


def test_the_panel_is_discovered_and_fills_the_exploits_section() -> None:
    assert isinstance(PANEL, Panel)
    assert PANEL.section == "exploits"
    assert [i.id for i in PANEL.instruments] == ["exploits.seeker"]


# --- 7. every proposal recorded, every finding replayed -------------------------------------------


def test_every_proposal_is_recorded_with_the_inputs_it_was_made_from(scorer: GraderScorer) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    assert len(record.proposals) == len(record.scores)
    for proposal in record.proposals:
        assert proposal.inputs, "a proposal with no recorded inputs cannot be replayed"
        assert proposal.proposal_id == Proposal.identity(
            actor=proposal.actor, task_id=proposal.task_id, response=proposal.response,
            inputs=proposal.inputs,
        )


def test_every_finding_replays_from_the_recorded_inputs_alone(scorer: GraderScorer) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    fresh = GraderScorer(bind(detect(keyword_reward)), outcome_check=outcome_check, threshold=0.9)
    again = replay(record, fresh)
    assert [s.proposal.proposal_id for s in again] == [s.proposal.proposal_id for s in record.scores]
    assert [s.reward for s in again] == [s.reward for s in record.scores]
    assert [s.exploit for s in again] == [s.exploit for s in record.scores]


def test_a_record_survives_a_round_trip_through_json(scorer: GraderScorer) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    restored = ActorRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored == record
    fresh = GraderScorer(bind(detect(keyword_reward)), outcome_check=outcome_check, threshold=0.9)
    assert [s.reward for s in replay(restored, fresh)] == [s.reward for s in record.scores]


def test_a_finding_that_cannot_be_replayed_is_refused_at_record_time(scorer: GraderScorer) -> None:
    """A grader whose answer drifts: the seeker refuses the finding rather than reporting it."""

    class Drifting:
        def __init__(self) -> None:
            self.seen = 0

        def score(self, task: dict, response: str) -> ScoreResult:
            self.seen += 1
            return ScoreResult(reward=1.0 if self.seen == 1 else 0.0,
                               outcome_passed=False, verdict="scored")

    actor = ScriptedActor(lambda state: EXPLOIT_LINE, scorer=Drifting())
    with pytest.raises(FindingNotReplayable) as raised:
        seek(actor, tasks=TASKS, intent=INTENT, max_proposals=2)
    assert "replay" in str(raised.value).lower()


def test_a_proposal_cannot_be_built_with_an_identity_that_is_not_its_own() -> None:
    good = Proposal.seal(actor="scripted", task_id="t1", response="x", inputs={"prompt": "p"})
    with pytest.raises(ValueError):
        Proposal(actor="scripted", task_id="t1", response="x", inputs={"prompt": "p"},
                 proposal_id="sha256:" + "0" * 64)
    assert good.proposal_id.startswith("sha256:")


# --- 8. the record the panel writes --------------------------------------------------------------


def test_the_panel_writes_one_witness_per_exploit_and_one_yield_estimate(
    scorer: GraderScorer,
) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    ctx = RunContext()
    ctx.seeker = "scripted"
    ctx.seeker_actor = ScriptedActor(scripted_policy, scorer=scorer)
    entries = ExploitSearch().entries_for(record, subject_ref=contracts.digest({"s": 1}), ctx=ctx)
    witnesses = [e for e in entries if e.kind == "witness"]
    estimates = [e for e in entries if e.kind == "estimate"]
    assert len(witnesses) == len(record.found)
    assert len(estimates) == 1
    for entry in entries:
        assert entry.section == "exploits"
        assert entry.provenance.arm == "black_box"
        assert entry.provenance.actor == "scripted"
        assert entry.provenance.budget_usd == "0.00"
        contracts.Entry.model_validate(entry.to_dict())
    assert witnesses[0].witness.inputs, "a witness with no inputs cannot be replayed"
    assert estimates[0].uncertainty.method == "wilson"
    assert estimates[0].sampling_unit == "proposal"


def test_the_panel_runs_end_to_end_when_an_actor_is_supplied(scorer: GraderScorer) -> None:
    ctx = RunContext()
    ctx.seeker = "scripted"
    ctx.seeker_actor = ScriptedActor(scripted_policy, scorer=scorer)
    ctx.seeker_tasks = TASKS
    ctx.seeker_intent = INTENT
    entries = ExploitSearch().run(_Subject(), _Corpus(), ctx)
    assert any(e.kind == "witness" for e in entries)
    assert any(e.kind == "estimate" for e in entries)


def test_a_finding_names_the_black_box_arm_and_points_at_its_witness(scorer: GraderScorer) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    ctx = RunContext()
    ctx.seeker = "scripted"
    witness = next(e for e in ExploitSearch().entries_for(
        record, subject_ref=contracts.digest({"s": 1}), ctx=ctx) if e.kind == "witness")
    found = findings_for(witness)
    assert len(found) == 1
    finding = found[0]
    assert finding.id == FINDING_ID
    assert finding.arm == "black_box"
    assert finding.kind == "fail"
    assert finding.entries == [witness.entry_id]
    assert len(finding.partial_fingerprint) >= 8
    contracts.Finding.model_validate(finding.to_dict())


def test_the_scope_the_entries_declare_matches_the_rung_they_ran_at(scorer: GraderScorer) -> None:
    record = seek(ScriptedActor(scripted_policy, scorer=scorer), tasks=TASKS, intent=INTENT,
                  max_proposals=4)
    ctx = RunContext()
    for entry in ExploitSearch().entries_for(record, subject_ref=contracts.digest({"s": 1}),
                                             ctx=ctx):
        assert entry.scope == "evaluator_comparison"
    with pytest.raises(ValueError):
        ExploitSearch().assert_scope_honest(
            contracts.absence("exploits", "exploits.seeker", "m", "a", "r", ("c",),
                              subject_ref=contracts.digest({"s": 1}),
                              provenance=RunContext().provenance(),
                              scope="held_out_transfer")
        )


# --- 9. the seeker and the protected partitions (interfaces section 8.4) -------------------------


def test_the_seeker_spends_a_sealed_round_rather_than_reading_a_partition(tmp_path: Path) -> None:
    ledger = RoundLedger(tmp_path / "rounds.json")
    sealed = ledger.seal(candidate_set="sha256:" + "a" * 64, partition_id="acc-1",
                         protocol_digest="sha256:" + "b" * 64)
    spent = ledger.spend(SealedRequest(candidate_digest=sealed.candidate_set,
                                       protocol_digest=sealed.protocol_digest,
                                       partition_id=sealed.partition_id, nonce=sealed.nonce))
    assert spent.state == "spent"


def test_the_seeker_capability_cannot_read_a_protected_partition(tmp_path: Path) -> None:
    root = tmp_path / "acceptance"
    root.mkdir()
    (root / "case.json").write_text('{"id": "c1"}', encoding="utf-8")
    partition = Partition.from_dir("acc-1", PartitionKind.PROTECTED_SUITE, root,
                                   log_path=tmp_path / "acc-1.jsonl")
    with pytest.raises(contracts.UsageError):
        partition.read("case.json", capability=SEEKER, purpose="attack development")
    assert SEEKER not in (ALLOWED_CAPABILITIES[PartitionKind.PROTECTED_SUITE] or frozenset())
    refused = partition.access_log.events[-1]
    assert refused.granted is False
    assert refused.capability == SEEKER


# --- helpers the panel tests hand to `run` -------------------------------------------------------


class _Subject:
    grader_path = Path("grader.py")
    entrypoint = "keyword_reward"
    project_dir = None
    tasks_path = None
    responses_path = None
    trainer = None
    name = "fixture"
    source = "def keyword_reward(prompt, completion, **kw): ...\n"


class _Corpus:
    rollouts: tuple = ()
    origin = "no task set and no response bank were supplied"
    tasks: tuple = TASKS
    n_tasks = len(TASKS)

    def as_list(self) -> list:
        return []

    def __len__(self) -> int:
        return 0


def test_scorer_for_binds_through_connect(tmp_path: Path) -> None:
    built = scorer_for(keyword_reward, outcome_check=outcome_check, threshold=0.9)
    assert isinstance(built, GraderScorer)
    result = built.score(TASKS[0], EXPLOIT_LINE)
    assert isinstance(result, ScoreResult)
    assert result.reward == 1.0
    assert result.outcome_passed is False


def test_a_completion_carries_the_tokens_the_meter_charges_for() -> None:
    completion = Completion.from_reply(chat_reply("x", prompt_tokens=12, completion_tokens=3))
    assert completion.text == "x"
    assert completion.input_tokens == 12
    assert completion.output_tokens == 3
