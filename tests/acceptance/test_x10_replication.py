"""X10 acceptance: the replication harness, its refusals, and the rate it will not report.

The clauses this file discharges:

  - every claim's citation resolves: the cited file still carries the quoted text on the recorded
    line, checked before anything is resolved, so a spec about a claim that has moved refuses
    instead of scoring;
  - `ReplicationSpec` refuses to be constructed when the claim cannot be adjudicated mechanically,
    which is the harness's most useful feature, and each refusal path is exercised separately;
  - the vacuity check is real: a rule that every reachable value passes, and a rule that every
    reachable value fails, are both rejected at construction;
  - the runner returns a `Reading` on every path and never raises for an anticipated condition,
    including the four ways an attempt can be stopped;
  - **no replication rate is returned.** `replication_rate` returns a `Refusal` whatever the
    corpus does, including a corpus in which everything replicated, and the bound it carries is a
    Wilson interval computed by `stats/sequential.py` rather than a sentence;
  - `pool_effects` refuses on units, through `Unit.compatible_with` rather than a string test,
    which is SPEC-ERRATA E15;
  - every instrument this package ships passes `lint_instrument`, which is SPEC-ERRATA E56, and
    the count of them is asserted so that adding one without linting it fails here;
  - the write-up carries no unbound number and every claim tag in it resolves against the store,
    which is SPEC-ERRATA E57, and this page is exactly the prose that erratum was about.

The regression pins on the two resolved effects and on the Wilson width are deliberate. They are
the numbers the write-up quotes, so a change to the extraction, the store, the tolerance or the
interval should break this file and be defended rather than pass quietly.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from experiments import x10_replication as x10
from experiments import x10_spec as x10_spec_module
from experiments.x10_spec import (
    Citation,
    ReplicationKind,
    ReplicationSpec,
    ReportedEffect,
    ResolutionRule,
    SpecLintError,
    Substrate,
    access_gap,
)
from reward_lens.artifacts.claims import check_text, find_unbound_numbers
from reward_lens.core.quantity import Unit
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import Access, Component
from reward_lens.measure.base import Instrument, lint_instrument
from reward_lens.studies.spec import Prediction

RELEASE = x10.REPO / "experiments" / "x10_replication"

pytestmark = pytest.mark.skipif(
    not x10.CAMPAIGN.exists() or not x10.FIELD_SCAN.exists(),
    reason=(
        f"X10 reads published claims from files outside this repository. The campaign store at "
        f"{x10.CAMPAIGN} or the field scan at {x10.FIELD_SCAN} is not on this disk."
    ),
)


@pytest.fixture(scope="module")
def result() -> x10.X10Result:
    return x10.run_all()


@pytest.fixture(scope="module")
def by_id(result: x10.X10Result) -> dict[str, x10.Attempt]:
    return {a.spec.id: a for a in result.attempts}


# ---------------------------------------------------------------------------
# Traceability: the claims are real and still say what the specs say they say
# ---------------------------------------------------------------------------


def test_every_citation_still_carries_its_quote_on_the_line_it_names():
    """The anti-fabrication check, run over all five claims plus the base-rate context line.

    A line number that has drifted, a quote that was transcribed rather than read, or an edit to a
    source all fail here rather than producing a replication verdict against a claim that has
    moved. This is the same discipline X8 applies to the six effect sizes it pools.
    """
    citations = [s.citation for s in x10.CLAIMS] + [x10.BASE_RATES_CITATION]
    bad = []
    for c in citations:
        found, actual = c.verify()
        if not found:
            bad.append(
                f"{c.source_path}:{c.source_line} does not carry {c.quote!r}; it reads {actual!r}"
            )
    assert not bad, "\n".join(bad)
    assert len(citations) == 6


def test_the_corpus_is_five_claims_spread_across_the_axes_the_row_asks_for():
    assert len(x10.CLAIMS) == 5
    ids = [s.id for s in x10.CLAIMS]
    assert ids == ["X10-1", "X10-2", "X10-3", "X10-4", "X10-5"]

    # At least one this build can resolve from disk, at least one that refuses for want of a
    # subject, at least one whose reported effect is an interval, and at least one of this
    # project's own published numbers.
    assert any(s.substrate.present for s in x10.CLAIMS)
    assert any(not s.substrate.present for s in x10.CLAIMS)
    assert any(s.reported.is_interval for s in x10.CLAIMS)
    assert sum(1 for s in x10.CLAIMS if s.original_effect_is_ours) >= 1
    assert any(not s.original_effect_is_ours for s in x10.CLAIMS)
    assert len({s.kind for s in x10.CLAIMS}) >= 2


def test_the_external_claim_says_its_number_came_from_a_summary_rather_than_the_paper():
    """The field a replication corpus goes wrong without, asserted on the claim that needs it.

    The dossier records the SAE numbers as read off a search-result summary with the PDF not
    fetched. A corpus that loses that distinction will eventually score a paper as unreplicated
    because a summary rounded, and the paper's authors will be right to object.
    """
    spec = next(s for s in x10.CLAIMS if s.id == "X10-5")
    assert "secondary summary" in spec.citation.provenance_of_effect
    assert spec.citation.arxiv == "2602.14111"
    assert not spec.original_effect_is_ours


def test_every_spec_hash_is_stable_and_distinct():
    """A frozen spec is a frozen spec. Two claims must not collide and one must not drift."""
    hashes = {s.id: s.spec_hash() for s in x10.CLAIMS}
    assert len(set(hashes.values())) == len(hashes)
    for s in x10.CLAIMS:
        assert s.spec_hash() == hashes[s.id]
        assert s.spec_hash().startswith("replication-spec:")

    # Editing a threshold produces a different spec, which is the whole point of hashing one.
    original = x10.CLAIMS[0]
    edited = dataclasses.replace(
        original,
        rule=ResolutionRule(
            metric=original.rule.metric,
            prediction=Prediction(
                metric=original.rule.metric,
                comparator=original.rule.prediction.comparator,
                threshold=original.rule.prediction.threshold * 2.0,
            ),
            reachable=original.rule.reachable,
            reachable_justification=original.rule.reachable_justification,
        ),
    )
    assert edited.spec_hash() != original.spec_hash()


# ---------------------------------------------------------------------------
# The lint: a claim that cannot be adjudicated cannot be written down
# ---------------------------------------------------------------------------


def _citation(**over: object) -> Citation:
    base = dict(
        authors="A. Author",
        year=2026,
        title="a paper",
        source_path=x10.REPO / "PREDICTIONS.md",
        source_line=1,
        quote="#",
        provenance_of_effect="the paper's own results table",
    )
    base.update(over)
    return Citation(**base)  # type: ignore[arg-type]


def _rule(**over: object) -> ResolutionRule:
    base = dict(
        metric="relative_gap",
        prediction=Prediction(metric="relative_gap", comparator="abs<", threshold=0.01),
        reachable=(0.0, 5.0),
        reachable_justification="a relative gap is non-negative and five times the effect is not a disagreement",
    )
    base.update(over)
    return ResolutionRule(**base)  # type: ignore[arg-type]


def _spec(**over: object) -> ReplicationSpec:
    base = dict(
        id="T-1",
        claim="A grader does a thing on a benchmark.",
        citation=_citation(),
        reported=ReportedEffect(
            unit=Unit("proportion", None, "raw", as_printed="fraction"), point=0.5
        ),
        substrate=Substrate(model="a model", dataset="a dataset"),
        requires={Component.RECORD: Access.RECORD},
        rule=_rule(),
        kind=ReplicationKind.DIRECT,
        resolver="experiments.x10_replication.resolve_stored_evidence",
    )
    base.update(over)
    return ReplicationSpec(**base)  # type: ignore[arg-type]


def test_the_baseline_spec_constructs_so_the_negative_cases_mean_something():
    assert _spec().id == "T-1"


def test_a_rule_every_reachable_value_passes_is_refused_at_construction():
    """The X7 defect, made impossible to write down.

    The campaign's TOPO-HODGE card registered `intransitive_mass > 0.03` on a design whose
    arithmetic floor is above 0.21, so no grader could have failed the prediction and nothing in
    the pipeline noticed. A comparator and a threshold are individually well formed however they
    are combined; only the reachable range makes the combination checkable.
    """
    with pytest.raises(SpecLintError, match="passed by every value"):
        _rule(
            prediction=Prediction(metric="relative_gap", comparator=">", threshold=-1.0),
            reachable=(0.0, 5.0),
        )


def test_a_rule_every_reachable_value_fails_is_refused_at_construction():
    with pytest.raises(SpecLintError, match="failed by every value"):
        _rule(
            prediction=Prediction(metric="relative_gap", comparator=">", threshold=10.0),
            reachable=(0.0, 5.0),
        )


def test_the_vacuity_check_catches_an_interior_boundary_and_not_only_the_endpoints():
    """`abs<` has an interior boundary, so endpoint-only checking would pass a vacuous rule.

    On a reachable range that is entirely inside the tolerance, every value passes and the rule
    decides nothing, but both endpoints look ordinary. The grid is what sees it.
    """
    with pytest.raises(SpecLintError, match="passed by every value"):
        _rule(
            prediction=Prediction(metric="relative_gap", comparator="abs<", threshold=1.0),
            reachable=(0.0, 0.5),
        )
    # And the same rule on a range that straddles the boundary is fine.
    assert _rule(
        prediction=Prediction(metric="relative_gap", comparator="abs<", threshold=1.0),
        reachable=(0.0, 5.0),
    ).check(0.5)


@pytest.mark.parametrize("comparator", ["==", "!="])
def test_an_equality_against_a_continuous_metric_is_refused(comparator: str):
    """Neither is a rule. One is satisfied by no measurement and the other by every measurement."""
    with pytest.raises(SpecLintError, match="tolerance"):
        _rule(prediction=Prediction(metric="relative_gap", comparator=comparator, threshold=0.5))


def test_a_reachable_range_with_no_justification_is_refused():
    """The range is the whole of the vacuity check, so a range nobody justifies makes it a formality."""
    with pytest.raises(SpecLintError, match="where it comes from"):
        _rule(reachable_justification="  ")


@pytest.mark.parametrize(
    "reachable, match",
    [
        ((0.0, float("inf")), "unbounded"),
        ((1.0, 1.0), "at most"),
        ((5.0, 0.0), "at most"),
    ],
)
def test_a_degenerate_reachable_range_is_refused(reachable: tuple[float, float], match: str):
    with pytest.raises(SpecLintError, match=match):
        _rule(reachable=reachable)


def test_a_rule_whose_metric_disagrees_with_its_prediction_is_refused():
    with pytest.raises(SpecLintError, match="disagree"):
        ResolutionRule(
            metric="relative_gap",
            prediction=Prediction(metric="interval_excess", comparator="<=", threshold=0.0),
            reachable=(0.0, 5.0),
            reachable_justification="justified",
        )


def test_an_effect_with_no_unit_token_is_refused_and_a_bare_string_is_not_a_unit():
    with pytest.raises(SpecLintError, match="as_printed"):
        ReportedEffect(unit=Unit("proportion", None, "raw"), point=0.5)
    with pytest.raises(SpecLintError, match="core.quantity.Unit"):
        ReportedEffect(unit="proportion", point=0.5)  # type: ignore[arg-type]


def test_an_effect_that_is_neither_a_point_nor_a_complete_interval_is_refused():
    unit = Unit("proportion", None, "raw", as_printed="fraction")
    with pytest.raises(SpecLintError, match="neither"):
        ReportedEffect(unit=unit, low=0.1)
    with pytest.raises(SpecLintError, match="backwards"):
        ReportedEffect(unit=unit, low=0.9, high=0.1)
    with pytest.raises(SpecLintError, match="not a measurement"):
        ReportedEffect(unit=unit, point=float("nan"))


def test_a_claim_that_restates_its_own_effect_is_refused():
    """Two copies of a number drift, and an embedded one cannot be bound to an evidence id.

    This is the lint that makes the write-up pass the claims gate by construction rather than by
    the author remembering, which is the lesson of SPEC-ERRATA E57.
    """
    with pytest.raises(SpecLintError, match="restates the effect"):
        _spec(claim="A grader scores 0.87 on a benchmark.")
    # An integer in a claim is a count, a year or a sample size, and is left alone.
    assert _spec(claim="A grader scores above chance on 500 prompts.").claim.endswith("prompts.")


def test_a_citation_with_no_quote_or_no_effect_provenance_is_refused():
    with pytest.raises(SpecLintError, match="carries no quoted text"):
        _citation(quote="")
    with pytest.raises(SpecLintError, match="where its reported effect was read from"):
        _citation(provenance_of_effect="")


def test_a_spec_with_no_access_requirement_or_no_resolver_is_refused():
    with pytest.raises(SpecLintError, match="no access requirement"):
        _spec(requires={})
    with pytest.raises(SpecLintError, match="not a dotted path"):
        _spec(resolver="resolve")
    with pytest.raises(SpecLintError, match="same as not naming it"):
        _spec(requires={Component.RECORD: Access.NONE})


def test_a_claim_in_two_sentences_is_refused():
    with pytest.raises(SpecLintError, match="more than one sentence"):
        _spec(claim="A grader does a thing. It does it on a benchmark.")


def test_a_substrate_with_no_model_or_no_dataset_is_refused():
    with pytest.raises(SpecLintError, match="no model named"):
        Substrate(model=" ", dataset="a dataset")
    with pytest.raises(SpecLintError, match="names no dataset"):
        Substrate(model="a model", dataset="")


# ---------------------------------------------------------------------------
# The runner: a Reading on every path, never an exception
# ---------------------------------------------------------------------------


def test_every_attempt_returns_a_reading_and_none_of_them_raise(result: x10.X10Result):
    """E27, one level up from `Instrument.estimate`.

    This is the entry point a corpus of fifty claims iterates over, so one raise in the middle of
    it loses the forty-nine readings either side. Every attempt is either an Evidence or a Refusal.
    """
    assert len(result.attempts) == 5
    for a in result.attempts:
        assert a.refused or getattr(a.reading, "id", "").startswith("ev:")
        if a.refused:
            assert isinstance(a.reading, Refusal)
            assert a.reading.remedy.strip()
            assert a.replicated is None
            assert a.metric_value is None
        else:
            assert a.replicated in (True, False)
            assert a.metric_value is not None


def test_a_refusal_never_becomes_a_failed_replication(result: x10.X10Result):
    """The single most damaging thing this harness could do is let those two share a column."""
    refused = [a for a in result.attempts if a.refused]
    assert len(refused) == 2
    assert {a.spec.id for a in refused} == {"X10-4", "X10-5"}
    for a in refused:
        assert a.replicated is None
        assert a.status == "refused"
        assert a.reading.reason is RefusalReason.ACCESS_INSUFFICIENT


def test_the_two_refusals_name_the_access_that_would_turn_them_into_attempts(
    by_id: dict[str, x10.Attempt],
):
    """A refusal that converts "somebody should check this" into a costed line item."""
    aisi = by_id["X10-4"]
    assert "POLICY: CONTROL" in aisi.reading.remedy
    assert "reward-hacking-olmo3.1-32b-kl0.0-seed2" in aisi.reading.remedy
    gap = access_gap(aisi.spec, x10.LOCAL_ACCESS)
    assert Component.POLICY in gap and Access.CONTROL & gap[Component.POLICY]

    sae = by_id["X10-5"]
    assert "POLICY: FORWARD" in sae.reading.remedy
    # The unstated fields travel with the remedy, because a disagreement against an underspecified
    # setup is unattributable between the finding and the setup.
    assert "revision" in sae.reading.remedy and "seed" in sae.reading.remedy
    assert set(sae.spec.substrate.missing_detail()) >= {"revision", "dtype", "engine", "n", "seed"}


def test_the_resolved_claims_reproduce_their_published_numbers(by_id: dict[str, x10.Attempt]):
    """The regression pins. These are the numbers the write-up quotes."""
    one = by_id["X10-1"]
    assert one.replicated is True
    assert one.replicated_effect == pytest.approx(0.2139761313773251, abs=5e-16)
    assert one.metric_value == pytest.approx(0.00011153583285, abs=5e-9)

    three = by_id["X10-3"]
    assert three.replicated is True
    assert three.replicated_effect == pytest.approx(0.4868605250149634, abs=5e-10)
    assert three.metric_value == 0.0  # inside the reported interval, so zero excess exactly
    assert three.spec.reported.is_interval


def test_the_harness_finds_one_of_this_project_s_own_numbers_disagreeing(
    by_id: dict[str, x10.Attempt],
):
    """X10-2, and it is the reason a harness that only points outward is not a harness.

    `PREDICTIONS.md` row R1 quotes a design floor of 0.238. X7's own enumeration of the fifteen
    designs in that corpus puts the floor at 0.2139761313773256, which is the Nectar-slice floor
    quoted for the corpus. Both numbers are on this disk and the harness resolves against the
    pooled one, because the same paragraph's other numbers are corpus-wide.
    """
    two = by_id["X10-2"]
    assert two.replicated is False
    assert two.spec.reported.point == 0.238
    assert two.replicated_effect == pytest.approx(0.21397613137732557, abs=5e-16)
    assert two.metric_value == pytest.approx(0.10094062446501856, abs=5e-15)
    assert two.metric_value > two.spec.rule.prediction.threshold

    # And the other reading of the sentence is a real number on this disk, which is why the spec
    # says the ambiguity is a citation to recheck rather than an arithmetic error.
    nectar = x10._store_row(two.spec.substrate.available_at, "x7.design_floor")
    assert nectar is not None
    assert nectar["by_slice"]["nectar-tournaments"]["floor"] == pytest.approx(0.238095238, abs=5e-9)


def test_a_citation_whose_line_has_moved_refuses_instead_of_scoring():
    spec = dataclasses.replace(
        x10.CLAIMS[0], citation=_citation(quote="a string that is not on line 1 of PREDICTIONS.md")
    )
    attempt = x10.run_replication(spec)
    assert attempt.refused
    assert attempt.reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "re-extract" in attempt.reading.remedy


def test_a_spec_naming_no_registered_resolver_refuses_rather_than_crashing():
    """The path the two refusing claims would reach if their access were satisfied.

    Two of the five name resolvers that are not registered, deliberately: a resolver for a
    substrate nobody standing here can reach would be code that has never run. This checks the
    refusal they would produce, by handing the runner an access matrix that satisfies them.
    """
    spec = next(s for s in x10.CLAIMS if s.id == "X10-5")
    assert spec.resolver not in x10.RESOLVERS
    generous = {c: Access(~0) for c in Component}
    attempt = x10.run_replication(spec, available=generous)
    assert attempt.refused
    # The substrate is still not on this disk, so that is what it refuses on first.
    assert attempt.reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "not on this disk" in attempt.reading.detail

    present = dataclasses.replace(
        spec, substrate=dataclasses.replace(spec.substrate, available_at=x10.REPO)
    )
    attempt = x10.run_replication(present, available=generous)
    assert attempt.refused
    assert attempt.reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert spec.resolver in attempt.reading.remedy
    assert "recorded and not attempted" in attempt.reading.remedy


def test_a_metric_outside_its_declared_reachable_range_refuses_rather_than_deciding():
    """A range asserted to make a rule pass its own lint is caught the first time it is wrong.

    The vacuity check is applied to the declared range, so a measurement outside that range means
    the check passed against the wrong interval and the rule has not been shown to decide anything.
    Returning a verdict there would be a confident wrong number produced by a lint that looked fine.
    """
    spec = dataclasses.replace(x10.CLAIMS[0], resolver="test.out_of_range")
    resolvers = {
        "test.out_of_range": lambda s: x10.Resolved(
            metric_value=99.0, replicated_effect=1.0, detail="deliberately out of range"
        )
    }
    attempt = x10.run_replication(spec, resolvers=resolvers)
    assert attempt.refused
    assert attempt.reading.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "outside the reachable range" in attempt.reading.detail
    assert spec.rule.reachable_justification in attempt.reading.remedy


def test_a_refusal_from_a_resolver_passes_straight_through():
    spec = dataclasses.replace(x10.CLAIMS[0], resolver="test.refuses")
    refusal = Refusal(
        instrument="test.refuses",
        reason=RefusalReason.LABEL_QUALITY_UNKNOWN,
        detail="the labels have no measured error rate",
        remedy="Measure the label error rate on a hand-audited sample and re-run.",
    )
    attempt = x10.run_replication(spec, resolvers={"test.refuses": lambda s: refusal})
    assert attempt.reading is refusal
    assert attempt.replicated is None


def test_the_stored_evidence_resolver_refuses_on_a_missing_row_and_a_missing_field():
    spec = x10.CLAIMS[0]
    missing_row = dataclasses.replace(
        spec, substrate=dataclasses.replace(spec.substrate, note="x7.not_a_row#curl_mass")
    )
    attempt = x10.run_replication(missing_row)
    assert attempt.refused
    assert attempt.reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "x7.not_a_row" in attempt.reading.detail

    missing_field = dataclasses.replace(
        spec, substrate=dataclasses.replace(spec.substrate, note="x7.reproduction#not_a_field")
    )
    attempt = x10.run_replication(missing_field)
    assert attempt.refused
    assert attempt.reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "not_a_field" in attempt.reading.detail


def test_an_interval_of_zero_width_raises_rather_than_dividing_by_an_epsilon():
    """A reported interval with no width is a point estimate dressed as one, and says so."""
    with pytest.raises(ValueError, match="no width"):
        x10._interval_excess(0.5, 0.5, 0.7)


def test_the_interval_excess_is_zero_inside_and_scaled_by_the_width_outside():
    assert x10._interval_excess(0.3, 0.7, 0.5) == 0.0
    assert x10._interval_excess(0.3, 0.7, 0.3) == 0.0
    assert x10._interval_excess(0.3, 0.7, 0.9) == pytest.approx(0.5)
    assert x10._interval_excess(0.3, 0.7, 0.1) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The two aggregations, and the rate that is never returned
# ---------------------------------------------------------------------------


def test_the_replication_rate_is_a_refusal_and_carries_a_measured_bound(result: x10.X10Result):
    """The row's central discipline, asserted at the type.

    "Three of five replicated" is the number this row exists not to publish. The refusal carries
    the Wilson interval on `partial` so the argument is a measured width rather than an assertion,
    and a reader who wants the ratio gets it with its interval attached instead of stripped off.
    """
    rate = result.rate
    assert isinstance(rate, Refusal)
    assert rate.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert rate.is_bounded

    stats = rate.statistics
    assert stats["specs"] == 5
    assert stats["attempted"] == 3
    assert stats["successes"] == 2
    assert stats["wilson_low"] == pytest.approx(0.20765960080204782, abs=5e-15)
    assert stats["wilson_high"] == pytest.approx(0.9385080552796038, abs=5e-15)

    width = stats["wilson_high"] - stats["wilson_low"]
    assert width == pytest.approx(0.730848454477556, abs=5e-15)
    # Wide enough to contain every published replication rate at once, which is the argument.
    assert all(
        stats["wilson_low"] <= v <= stats["wilson_high"] for v in x10.PUBLISHED_BASE_RATES.values()
    )

    bound = rate.partial
    assert bound is not None
    assert bound.value["point_if_you_insisted"] == pytest.approx(2 / 3)
    assert bound.value["kinds"] == ["REANALYSIS"]


def test_the_rate_refuses_even_when_every_claim_replicated():
    """There is no corpus on which this function starts returning a number.

    The decision not to report a rate over a demonstration corpus is not a small-sample accident
    that a happier result would fix. A corpus in which everything replicated is exactly the case
    where a point estimate is most tempting and least defensible.
    """
    everything = tuple(
        x10.Attempt(
            spec=s, reading=object(), metric_value=0.0, replicated_effect=1.0, replicated=True
        )
        for s in x10.CLAIMS
    )
    rate = x10.replication_rate(everything)
    assert isinstance(rate, Refusal)
    assert rate.statistics["successes"] == 5 and rate.statistics["attempted"] == 5
    # A perfect run still has an interval, and it still does not reach 1.
    assert rate.statistics["wilson_low"] < 1.0
    assert rate.is_bounded


def test_the_rate_refuses_differently_when_nothing_could_be_attempted():
    """Zero attempts is not a wide interval, it is no interval, and the reason has to differ."""
    nothing = tuple(
        x10.Attempt(
            spec=s, reading=Refusal("x", RefusalReason.ACCESS_INSUFFICIENT, "d", "Get more.")
        )
        for s in x10.CLAIMS
    )
    rate = x10.replication_rate(nothing)
    assert rate.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert not rate.is_bounded
    assert x10.pool_effects(nothing).reason is RefusalReason.ACCESS_INSUFFICIENT


def test_pooling_refuses_on_units_through_the_unit_type_rather_than_a_string(
    result: x10.X10Result,
):
    """SPEC-ERRATA E15, at the place a replication corpus would otherwise get it wrong.

    The resolved claims carry a mass fraction and a rank-pair fraction. Both are proportions and
    they are not the same quantity, which a string comparison of "proportion" would miss and
    `Unit.compatible_with` does not.
    """
    pooled = result.pooled
    assert isinstance(pooled, Refusal)
    assert pooled.reason is RefusalReason.UNIT_MISMATCH
    assert pooled.statistics["n_resolved"] == 3
    assert not x10.MASS_FRACTION.compatible_with(x10.RANK_PAIR_FRACTION)
    assert "Pool within a unit" in pooled.remedy


def test_two_undecided_units_are_not_thereby_compatible():
    """The 276-pair defect, checked on the unit this corpus actually carries one of."""
    assert not x10.INTERP_SCORE.is_decided
    other = Unit("OPEN", "OPEN", "OPEN", as_printed="some other score")
    assert not x10.INTERP_SCORE.compatible_with(other)
    assert not x10.INTERP_SCORE.compatible_with(x10.INTERP_SCORE)


def test_a_single_unit_corpus_hands_meta_analysis_something_it_can_consume():
    """The hand-off a real fifty-claim study would use, exercised on a corpus that can be pooled.

    `pool_effects` returns `(label, effect, variance)` triples in the shape `random_effects` takes.
    A claim reported as a bare point contributes no variance and is excluded rather than imputed,
    which is the discipline X8 applied when it refused two of its six sources for having no
    denominator.
    """
    unit = Unit("proportion", "group", "raw", as_printed="fraction")
    specs = [
        _spec(
            id=f"P-{i}",
            reported=ReportedEffect(unit=unit, point=0.5, low=0.4, high=0.6),
            substrate=Substrate(model="m", dataset="d"),
        )
        for i in range(3)
    ]
    attempts = tuple(
        x10.Attempt(
            spec=s,
            reading=object(),
            metric_value=0.0,
            replicated_effect=0.5 + 0.1 * i,
            replicated=True,
        )
        for i, s in enumerate(specs)
    )
    rows = x10.pool_effects(attempts)
    assert not isinstance(rows, Refusal)
    assert [label for label, _, _ in rows] == ["P-0", "P-1", "P-2"]
    for _, effect, variance in rows:
        assert math.isfinite(effect) and variance > 0.0

    from reward_lens.stats import meta

    fit = meta.random_effects(
        [e for _, e, _ in rows], [v for _, _, v in rows], labels=[n for n, _, _ in rows]
    )
    # k = 3 is the floor, so this is a real fit rather than a refusal, and its prediction interval
    # is wider than its confidence interval. That is the property the missing study would report.
    assert not isinstance(fit, Refusal)
    assert fit.k == 3
    assert fit.prediction_width > fit.ci_width


def test_a_point_reported_claim_contributes_no_variance_and_is_excluded_from_pooling():
    unit = Unit("proportion", "group", "raw", as_printed="fraction")
    attempts = tuple(
        x10.Attempt(
            spec=_spec(id=f"Q-{i}", reported=ReportedEffect(unit=unit, point=0.5)),
            reading=object(),
            metric_value=0.0,
            replicated_effect=0.5,
            replicated=True,
        )
        for i in range(3)
    )
    assert x10.pool_effects(attempts) == []


# ---------------------------------------------------------------------------
# Lint: SPEC-ERRATA E56
# ---------------------------------------------------------------------------


def test_every_instrument_this_package_ships_passes_lint():
    """E56: an acceptance test that renders a reading is not a substitute for one that lints.

    Four instruments shipped for two waves failing lint rule 1 while their package read `done`,
    because the clause tested the measurement and the lint tests the declaration. This package
    ships no `Instrument` today, and the count is asserted so that adding one without a registered
    quantity, a baseline, an envelope and an invariance group fails here rather than in wave eight.

    It ships none because a replication outcome has no registered quantity: `spec/QUANTITIES.yaml`
    carries nothing for a replication rate or a re-run's agreement with a published effect, and
    registering one is a decision about what the library claims to measure. The readings this
    harness produces are stamped with the quantity of the *claim* where the claim has one, which is
    the right place for it: the quantity is a property of what was measured, not of the harness.
    """
    # Classes as well as instances: a declaration carried on the class is what `lint_instrument`
    # reads, and checking only instances is how a shipped declaration goes unlinted.
    candidates = [getattr(x10, n) for n in dir(x10) if not n.startswith("_")]
    candidates += [
        getattr(x10_spec_module, n) for n in dir(x10_spec_module) if not n.startswith("_")
    ]
    shipped = [obj for obj in candidates if isinstance(obj, Instrument)]
    findings = [f for inst in shipped for f in lint_instrument(inst)]
    assert not findings, "\n".join(f.render() for f in findings)
    assert len(shipped) == 0

    # And the quantity travels on the reading from the claim, not from the harness.
    assert all(s.reported.quantity == "" for s in x10.CLAIMS)


# ---------------------------------------------------------------------------
# The write-up: SPEC-ERRATA E57, on exactly the prose that erratum was about
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered() -> tuple[str, EvidenceStore]:
    findings = RELEASE / "FINDINGS-x10.md"
    if not findings.exists():
        pytest.skip(
            f"{findings} has not been rendered; run `python -m experiments.x10_replication`"
        )
    return findings.read_text(encoding="utf-8"), EvidenceStore(RELEASE / "evidence", readonly=True)


def test_the_write_up_carries_no_unbound_number(rendered: tuple[str, EvidenceStore]):
    """A page whose whole subject is other people's numbers, held to the standing rule.

    E57 was found by the X7 write-up noticing its own section heading was being exempted, and this
    page is denser in quoted effects than that one was. Every decimal in it is either bound to an
    evidence id or is not there.
    """
    text, _ = rendered
    unbound = find_unbound_numbers(text)
    assert not unbound, "\n".join(str(u) for u in unbound)


def test_every_claim_tag_in_the_write_up_resolves_against_the_store(
    rendered: tuple[str, EvidenceStore],
):
    text, store = rendered
    report = check_text(text, store)
    assert report.ok, report.render()
    assert not report.unresolved_refs
    assert len(report.results) >= 25


def test_the_reported_effects_are_printed_through_the_rows_that_recorded_them(
    rendered: tuple[str, EvidenceStore],
):
    """Including the two that could not be attempted.

    A refusal row still carries what was claimed, because a corpus that stores only the readings it
    managed to take has a replication rate computed over whatever happened to work.

    **The row is found through the citation and not through the observable name**, and that is the
    repair rather than a detail. This test used to select by observable and assert there was one
    row, which encodes "this store was written once" as an invariant of an append-only store. The
    harness ran twice, the store correctly kept both runs, and the test failed on a store that was
    behaving exactly as designed while the write-up, which cites ids, was pointing at the right
    nine rows the whole time. An evidence id is a content hash, so resolving through it is
    unambiguous however many times the harness runs; `check_text` separately refuses a document
    that cites two runs at once.
    """
    text, store = rendered
    for spec in x10.CLAIMS:
        rows = [r for r in store if r.observable == f"x10.{spec.id}"]
        assert rows, f"{spec.id} has no row in the store"
        cited = [r for r in rows if f"ev={r.id}" in text]
        assert len(cited) == 1, (
            f"{spec.id}: {len(rows)} rows in the store and {len(cited)} of them cited by the "
            f"write-up. Exactly one reading is the reported one."
        )
        assert cited[0].value["reported_effect"] == spec.reported.point


def test_the_page_says_plainly_that_no_rate_is_reported(rendered: tuple[str, EvidenceStore]):
    text, _ = rendered
    assert "No replication rate is reported and none is registered" in text
    assert "Why no rate is reported" in text
    # And it does not smuggle one back in as a ratio in prose.
    assert "replication rate of" not in text.replace(
        "The replication rate of mechanistic interpretability findings is unknown.", ""
    )


def test_the_manifest_records_a_spec_hash_and_a_status_for_every_claim():
    import json

    manifest_path = RELEASE / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("the release directory has not been rendered")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_specs"] == 5
    assert set(manifest["specs"]) == {s.id for s in x10.CLAIMS}
    assert all(h.startswith("replication-spec:") for h in manifest["specs"].values())
    assert manifest["status"] == {
        "X10-1": "replicated",
        "X10-2": "not-replicated",
        "X10-3": "replicated",
        "X10-4": "refused",
        "X10-5": "refused",
    }
    assert manifest["git_sha"]


def test_the_run_is_cheap_enough_that_a_fifty_claim_corpus_is_not_the_hard_part(
    result: x10.X10Result,
):
    """The harness is not what makes the missing study expensive; the re-runs are.

    Ten seconds is a generous ceiling on a five-claim corpus that reads three files and one store.
    It is asserted so that a resolver quietly acquiring a model download shows up here.
    """
    assert result.seconds < 10.0
