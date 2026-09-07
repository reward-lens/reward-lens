"""X1 acceptance: the grader card and the metrology report, released on real subjects.

The clause, in full:

    *On four real graders reachable from this repository, at least two of them the reward functions
    of real RL environments, `GraderCard` renders thirteen fields each, every one either a reading
    or a refusal carrying its remedy. The capability report and preflight make zero grader calls on
    all four. The dual-use fields are withheld from every published card. The series A metrology
    report is produced beside them. And the published findings section contains no number that the
    release's own evidence store cannot verify.*

Most of this file reads the released artifacts under `experiments/x1_release/` rather than
rebuilding them, because the release is the deliverable and a test that only checks a fresh rebuild
would pass while the published files said something else. One test does rebuild, on the subject
whose source and corpus are both on disk, so the pipeline is proved live rather than only proved to
have run once.

Producing the artifacts:

    python -m experiments.x1_card_release

It needs the network on a cold run, for one source file and four dataset slices, and nothing after
that: the slices are cached beside the release. It needs no GPU and no hosted model.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from experiments.x1_card_release import (
    FULL_ACCESS,
    LOG_ONLY_ACCESS,
    RECON,
    REPO,
    build_verl_gsm8k,
)
from reward_lens.artifacts.claims import check_files, find_unbound_numbers
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import Phase
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.card import CARD_FIELDS, GraderCard, card_context, render_card

RELEASE = REPO / "experiments" / "x1_release"

#: What the clause asks for by name. Two graders and two environment graders, and the environment
#: pair is what makes the second half of the clause a claim rather than a restatement of the first.
SUBJECTS = ("is_equiv", "swebench", "verl_gsm8k", "verl_search_r1")

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with `python -m experiments.x1_card_release`; "
        f"this file checks the published artifacts, and skipping it leaves them unchecked rather "
        f"than checked."
    ),
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence", readonly=True)


@pytest.fixture(scope="module")
def rows(store: EvidenceStore) -> dict[str, Any]:
    """Every row of the release, keyed by the observable name that wrote it."""
    return {store.get(i).observable: store.get(i) for i in list(store._index)}


def _card_text(key: str) -> str:
    return (RELEASE / "cards" / f"{key}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Clause 1: four real subjects, two of them environments
# ---------------------------------------------------------------------------


def test_the_release_is_four_real_subjects_and_two_are_environments(
    manifest: dict[str, Any],
) -> None:
    subjects = manifest["subjects"]
    assert set(subjects) == set(SUBJECTS)
    kinds = sorted(s["kind"] for s in subjects.values())
    assert kinds == ["environment", "environment", "grader", "grader"]
    for key, entry in subjects.items():
        assert entry["fingerprint"].startswith("verifier:"), key
        assert entry["n_rollouts"] > 0, key
        assert entry["corpus"], f"{key} does not say what corpus it was measured on"
    # The subject sources are content-addressed, so a card published today is about a program a
    # reader can identify tomorrow. That is what `STATIONARY_GRADER` is a condition on.
    assert len({s["fingerprint"] for s in subjects.values()}) == 4


# ---------------------------------------------------------------------------
# Clause 2: thirteen fields, each a reading or a refusal with a remedy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", SUBJECTS)
def test_every_published_card_carries_all_thirteen_fields(key: str) -> None:
    text = _card_text(key)
    for spec in CARD_FIELDS:
        assert spec.name in text, f"{key}: {spec.name} is not on the published page"
    assert "of 13 fields read and" in text


@pytest.mark.parametrize("key", SUBJECTS)
def test_every_refusal_on_every_published_card_carries_a_remedy(key: str) -> None:
    """The rule the card exists for, checked on the page rather than on the object.

    Counting `REFUSED` against `Remedy:` is deliberately a check of the rendering. A refusal that
    carries a remedy in memory and drops it on the way to the page is exactly the failure a reader
    would see and the object graph would not.
    """
    lines = _card_text(key).splitlines()
    refusals = [i for i, line in enumerate(lines) if "REFUSED" in line]
    remedies = [i for i, line in enumerate(lines) if line.strip().startswith("Remedy:")]
    assert refusals, f"{key}: a card with no refusals at all is a card nobody should believe"
    assert len(remedies) == len(refusals), (
        f"{key}: {len(refusals)} refusals, {len(remedies)} remedies"
    )
    for i in refusals:
        reason = lines[i].split("REFUSED")[1].strip()
        assert reason in RefusalReason.__members__, f"{key}: {reason} is not a refusal reason"


@pytest.mark.parametrize("key", SUBJECTS)
def test_the_read_and_refused_counts_on_the_page_add_to_thirteen(
    key: str, rows: dict[str, Any]
) -> None:
    value = rows[f"x1.card.{key}"].value
    assert value["n_fields"] == 13
    assert value["n_read"] + value["n_refused"] == 13
    assert value["n_read"] >= 1, f"{key}: a card that read nothing is a different result to check"
    assert f"{value['n_read']} of 13 fields read" in _card_text(key)
    assert value["lint_findings"] == 0, f"{key}: GraderCard did not lint clean on this subject"


def test_most_of_the_release_is_refusals_and_they_are_all_on_the_page(
    rows: dict[str, Any],
) -> None:
    """The headline shape. If this ever inverts, the claim in the findings section inverts with it,
    and the section is generated from these same rows so it cannot fail to."""
    release = rows["x1.release"].value
    assert release["total_fields"] == 4 * 13
    assert release["total_read"] + release["total_refused"] == release["total_fields"]
    assert release["refused_share"] == pytest.approx(
        release["total_refused"] / release["total_fields"]
    )
    assert release["total_refused"] > release["total_read"]


# ---------------------------------------------------------------------------
# Clause 3: the capability report costs nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", SUBJECTS)
def test_the_capability_report_and_preflight_made_no_grader_call(
    key: str, rows: dict[str, Any]
) -> None:
    value = rows[f"x1.card.{key}"].value
    assert value["plan_calls"] == 0, f"{key}: the capability report called the grader"
    assert value["preflight_calls"] == 0, f"{key}: preflight called the grader"
    assert value["priced_calls"] > 0, f"{key}: preflight returned a cost of nothing at all"
    plan = (RELEASE / "plans" / f"{key}.txt").read_text(encoding="utf-8")
    assert "grader calls" in plan
    for spec in CARD_FIELDS:
        assert spec.name in plan, f"{key}: {spec.name} is missing from the capability report"


@pytest.mark.parametrize("key", SUBJECTS)
def test_a_field_the_plan_called_unreachable_never_read(key: str, rows: dict[str, Any]) -> None:
    """The plan may promise more than the card delivers and it says so. The converse must not
    happen: a field the plan ruled out cannot turn up as a reading."""
    value = rows[f"x1.card.{key}"].value
    assert value["plan_available"] >= value["n_read"]
    assert value["plan_available"] + value["plan_refused"] == 13


def test_the_refusal_ledger_carries_every_refusal_with_its_remedy(rows: dict[str, Any]) -> None:
    """The refusals are published as an artifact in their own right, not only as card rows.

    One entry per field per subject, each with the instrument that would fill it, the reason, and
    the remedy in full. A ledger that dropped one would make the release's own headline count wrong,
    which is what the first assertion checks.
    """
    entries: list[dict[str, Any]] = []
    for key in SUBJECTS:
        path = RELEASE / "refusals" / f"{key}.json"
        assert path.exists(), f"{key} has no refusal ledger"
        entries.extend(json.loads(path.read_text(encoding="utf-8")))

    assert len(entries) == rows["x1.release"].value["total_refused"]
    for entry in entries:
        assert entry["reason"] in RefusalReason.__members__, entry
        assert entry["detail"].strip(), entry
        assert entry["remedy"].strip(), entry
        assert entry["quantity"], entry

    text = (RELEASE / "refusals.txt").read_text(encoding="utf-8")
    for entry in entries:
        assert entry["field"] in text
        assert entry["remedy"] in text, f"{entry['field']}: the remedy did not reach the ledger"


def test_the_commonest_refusal_reason_is_the_one_that_sends_the_reader_upstream(
    rows: dict[str, Any],
) -> None:
    """E30's distinction, checked on real output rather than on a unit fixture.

    `RECORD_INCOMPLETE` means the access was sufficient and the field was never written, so the fix
    is upstream. That is the majority case on a real grader nobody instrumented, and if it ever
    stops being the majority case here the likeliest cause is a refusal site reaching for
    `ACCESS_INSUFFICIENT` again, which is the confusion E30 exists to prevent.
    """
    release = rows["x1.release"].value
    assert release["n_record_incomplete"] > release["n_substrate_mismatch"]
    assert (
        release["n_record_incomplete"] + release["n_substrate_mismatch"] + release["n_other_reason"]
        == release["total_refused"]
    )
    assert "ACCESS_INSUFFICIENT" not in release["by_reason"], (
        "an access refusal on a card built with source, query, replicate and record access would "
        "mean the reason is being used for something that is not an access problem"
    )


# ---------------------------------------------------------------------------
# Clause 4: dual use, withheld
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", SUBJECTS)
def test_no_published_card_carries_a_reproducer(key: str, rows: dict[str, Any]) -> None:
    """D2's mutant list reads on all four subjects, and on all four it is redacted on the page.

    The check is the pair: the row says the field was sensitive, and the page says the reproducers
    were withheld. Either alone would pass on a card that quietly dropped the field.
    """
    value = rows[f"x1.card.{key}"].value
    assert value["n_sensitive"] >= 1, f"{key}: nothing declared itself sensitive"
    assert set(value["sensitive_fields"]) <= {"surviving mutants", "false-positive catalogue"}
    text = _card_text(key)
    assert "reproducers withheld" in text
    assert "withheld their reproducers" in text


def test_the_findings_section_says_the_reproducers_are_not_in_it() -> None:
    text = (RELEASE / "FINDINGS-x1.md").read_text(encoding="utf-8")
    assert "the redacted form" in text
    assert "not a command-line flag" in text


# ---------------------------------------------------------------------------
# Clause 5: the metrology report, beside the cards
# ---------------------------------------------------------------------------


def test_the_metrology_report_is_beside_the_cards(rows: dict[str, Any]) -> None:
    report = (RELEASE / "metrology.txt").read_text(encoding="utf-8")
    for heading in ("A2 variance components", "A1 effective group size", "A3 attenuation"):
        assert heading in report
    value = rows["x1.metrology"].value
    assert value["n_graders_rb2"] == 11
    assert value["n_responses"] == value["n_groups"] * value["k_nominal"]
    assert len(value["effective_group_size"]) == 11
    for grader, ladder in value["effective_group_size"].items():
        assert 0.0 < ladder["rung3"] < ladder["rung0"] <= value["k_nominal"], grader
        assert ladder["rung3_ci_low"] < ladder["rung3"] < ladder["rung3_ci_high"], grader
    # The gauge fixing is what makes the crossing legal and the report states its consequence.
    assert "sigma2(rater) is zero by construction" in report


# ---------------------------------------------------------------------------
# Clause 6: no number in the findings section that the store cannot verify
# ---------------------------------------------------------------------------


def test_the_findings_section_has_no_unbound_number_and_every_claim_resolves(
    store: EvidenceStore,
) -> None:
    """The clause that makes the rest of them worth reading.

    Run the way CI runs it, with no baseline, against the release's own store. A number here that
    no measurement produced is a failure; a claim tag whose stored value has drifted from the prose
    is a failure; a dangling evidence reference is a failure.
    """
    report = check_files([RELEASE / "FINDINGS-x1.md"], store=store)
    assert report.results, "the section carries no claims at all, which is its own failure"
    assert report.ok, report.render()
    assert not report.unbound, [str(u) for u in report.unbound]
    assert not report.unresolved_refs


def test_the_checker_would_have_caught_a_fabricated_number(store: EvidenceStore) -> None:
    """The gate, shown firing. A checker that has never rejected anything is decoration."""
    from reward_lens.artifacts.claims import check_text

    good = (RELEASE / "FINDINGS-x1.md").read_text(encoding="utf-8")
    assert find_unbound_numbers("The measured violation rate is 0.6852.\n")
    assert not check_text("A rate of 0.6852 [[claim value=0.6852 ev=ev:0000dead]].", store).ok
    assert check_text(good, store).ok


# ---------------------------------------------------------------------------
# Clause 7: the ordering, and the artifact that survives below the access minimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", SUBJECTS)
def test_a_reader_holding_only_a_log_still_gets_a_card(key: str, rows: dict[str, Any]) -> None:
    value = rows[f"x1.card.{key}"].value
    assert value["log_only_read"] < value["n_read"], f"{key}: less access did not cost a reading"
    assert value["log_only_read"] + value["log_only_refused"] == 13
    text = (RELEASE / "cards" / f"{key}-log-only.txt").read_text(encoding="utf-8")
    assert "short of D7's stated minimum" in text
    for spec in CARD_FIELDS:
        assert spec.name in text


# ---------------------------------------------------------------------------
# The pipeline, run live rather than read off disk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (RECON / "verl" / "verl" / "utils" / "reward_score" / "gsm8k.py").exists(),
    reason="verl's checkout is not on this machine, so the live rebuild has no subject",
)
@pytest.mark.skipif(
    not (RELEASE / "corpora" / "gsm8k.jsonl").exists(),
    reason="the GSM8K slice is not cached, and this test does not go to the network",
)
def test_the_pipeline_rebuilds_a_card_from_the_cached_corpus() -> None:
    """One subject, rebuilt end to end, with the zero-call claim re-established from outside.

    verl's GSM8K reward is the subject because both halves of it are on disk: the source is in the
    local checkout and the corpus slice is cached beside the release. So this proves the pipeline
    runs today without proving anything about the network.
    """
    subject = build_verl_gsm8k(RELEASE)
    assert lint_instrument(GraderCard(subject.inputs)) == []

    calls = {"n": 0}
    real = subject.inputs.grader

    def counted(**kw: Any) -> float:
        calls["n"] += 1
        return real(**kw)  # type: ignore[misc]

    watched = replace(subject.inputs, grader=counted)
    ctx = card_context(watched, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    plan = GraderCard(watched).capability_report(ctx)
    assert calls["n"] == 0, f"the capability report made {calls['n']} grader calls"
    assert len(plan.fields) == 13

    reading = GraderCard(subject.inputs).estimate(
        card_context(subject.inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    )
    assert not isinstance(reading, Refusal), render_card(reading)
    card = reading.value
    assert len(card.fields) == 13
    assert card.read_fields and card.refused_fields
    for f in card.refused_fields:
        assert f.remedy.strip(), f"{f.name} refuses with no remedy"
        assert f.reason in RefusalReason.__members__

    poor = GraderCard(subject.inputs).estimate(
        card_context(subject.inputs, access=LOG_ONLY_ACCESS, phase=Phase.PRE_RUN)
    )
    assert {f.name for f in poor.value.refused_fields} > {f.name for f in card.refused_fields}


def test_the_release_records_the_commit_it_was_measured_against(
    manifest: dict[str, Any], rows: dict[str, Any]
) -> None:
    """Numbers under this release move when the packages under it move, so the release says which
    tree it read. A published card with no commit behind it is a card about nothing in particular."""
    assert manifest["git_sha"]
    assert rows["x1.release"].value["git_sha"] == manifest["git_sha"]
    assert manifest["git_sha"] in (RELEASE / "FINDINGS-x1.md").read_text(encoding="utf-8")
