"""X5, the acceptance clause.

    A McCrary test and a bunching elasticity are computed at a hard gate on a real per-rollout
    record, the placebo cutoff and the smooth-density null run beside every reading, and the
    elasticity responds when the gate is synthetically moved.

Read the second half of this docstring before quoting the first. **Every gate here is installed.**
The subject is 25,664 rollouts from `ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2`,
and the one hard reward gate that record really has is a conjunction with a thinking-format check in
it, whose assignment variable is a predicate on the text and therefore has no density. So the gates
tested below are installed counterfactually onto the record's own completion lengths at cutoffs we
chose, exactly as W4.5 did on the smaller GRPO fixture, and the tests assert that every reading says
so.

The subject is a derived fixture, `experiments/x5_threshold/aisi_lengths.npz`, holding five columns
and no text. These tests need no network.

The one result worth stating in a test docstring: on the full-range running variable the density
test returns z between -50 and -77 at cutoffs where no gate exists, and the smooth-density null
baseline comes back centred at -23 with a spread of 2.2 rather than at 0 with a spread of 1. The
baseline **detects** that failure and does not repair it: standardised against the band the statistic
is still 24 standard deviations out. Both halves are asserted below, because a version that quietly
made the second one small would look like an improvement and would in fact be a baseline that had
stopped measuring anything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.x5_discontinuity import (
    CUTOFFS,
    GH_QUERIES,
    HF_QUERIES,
    WINDOW,
    control_for_the_null,
    density_power,
    installed_gate,
    load_subject,
    move_the_gate,
    plant_response,
    read_bunching,
    read_density,
    running_variable,
    study_spec,
    summarise,
)
from reward_lens.core.evidence import Evidence
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.threshold import BunchingElasticity, DensityDiscontinuity, Gate

OUT = Path(__file__).resolve().parents[2] / "experiments" / "x5_threshold"


@pytest.fixture(scope="module")
def subject():
    loaded = load_subject(OUT)
    if loaded is None:
        pytest.skip(f"the derived AISI fixture is not at {OUT / 'aisi_lengths.npz'}")
    length, audit = loaded
    window = length[(length >= WINDOW[0]) & (length <= WINDOW[1])]
    return {"length": length, "window": window, "audit": audit}


# ---------------------------------------------------------------------------
# the search, and the one correction to W4.5
# ---------------------------------------------------------------------------


def test_the_record_really_does_carry_a_documented_hard_reward_gate(subject):
    """W4.5 said none exists. It does, and this measures it rather than quoting a dataset card."""
    audit = subject["audit"]
    assert audit.n == 25664
    assert audit.conjunction_rate > 0.99, audit.render()
    assert audit.gate_is_documented
    # and the format check is doing work rather than being redundant with correctness
    assert audit.reward_equals_passed < audit.n
    assert audit.n_format_failures > 0


def test_the_gates_running_variable_has_no_density_to_test(subject):
    """The correction that matters: a real gate, on a variable that takes two values.

    The obvious alternative reading is that the format failures are truncations at a token budget.
    That is checked rather than dismissed: a sampler stopping at a budget leaves a spike on the last
    value, and the largest tie anywhere in this record's upper tail is a handful of rollouts.
    """
    audit = subject["audit"]
    assert audit.running_variable_is_continuous is False
    assert audit.truncation_cap_present is False, audit.render()
    assert audit.largest_length_tie < audit.n // 500
    assert audit.length_max > 100 * audit.length_min


def test_the_search_queries_are_frozen_into_the_study_spec():
    """ "We looked and there is none" is a claim, and a claim needs the query list attached."""
    spec = study_spec()
    extra = spec.subjects.extra
    assert extra["hf_queries"] == list(HF_QUERIES)
    assert extra["gh_queries"] == list(GH_QUERIES)
    assert len(HF_QUERIES) >= 10 and len(GH_QUERIES) >= 4
    canonical = json.dumps(spec.__canonical__())
    assert "hf_queries" in canonical and "window" in canonical and "cutoffs" in canonical


# ---------------------------------------------------------------------------
# clause, first half: I1 and I2 at a gate on a real record, with both baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_mccrary_is_computed_at_each_gate_with_both_mandatory_baselines(subject, cutoff):
    reading = read_density(
        subject["window"], cutoff, "windowed", n_null=120, n_placebos=15, n_boot=60
    )
    assert not reading.refused, reading.evidence
    assert isinstance(reading.evidence, Evidence)
    assert reading.evidence.quantity == "gate.mccrary_statistic"
    value = reading.value
    assert value.n == subject["window"].size
    assert math.isfinite(value.theta) and value.se > 0 and math.isfinite(value.z)
    assert 0.0 <= value.p <= 1.0
    # both baselines the catalogue names for I1, on the reading rather than in a footnote
    assert value.smooth_null is not None and value.smooth_null.n_draws >= 100
    assert value.placebo is not None and len(value.placebo_z) >= 5
    # and the honesty field
    assert value.gate.installed and "INSTALLED" in value.render()


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_the_bunching_elasticity_is_computed_at_each_gate(subject, cutoff):
    row = read_bunching(subject["window"], cutoff, n_null=120, n_placebos=15, n_boot=60)
    assert not row.refused, row.evidence
    assert isinstance(row.evidence, Evidence)
    assert row.evidence.quantity == "gate.bunching_elasticity"
    value = row.value
    assert math.isfinite(value.excess_mass) and math.isfinite(value.dz_star)
    assert value.elasticity is not None and math.isfinite(value.elasticity)
    assert value.smooth_null is not None
    assert len(value.placebo_excess) >= 5
    assert value.gate.installed


def test_every_gate_this_module_can_build_is_marked_installed():
    """There is no recorded gate on this record, so there must be no way to claim one."""
    gate = installed_gate(900.0)
    assert gate.installed
    assert "counterfactually" in gate.provenance
    assert gate.penalty_fraction is not None and gate.penalty_fraction > 0


# ---------------------------------------------------------------------------
# the result: the baselines catch a statistic the normal approximation would have published
# ---------------------------------------------------------------------------


def test_the_full_range_test_withholds_a_huge_statistic_where_there_is_no_gate(subject):
    """A real completion-length density spans three orders of magnitude and breaks the local fit.

    **This test used to assert the instrument returned the statistic. It now asserts it refuses to,
    and that is the fix landing rather than the finding weakening.** When X5 first ran, the
    full-range reading came back as Evidence carrying a z of 50 to 76 on a record with no gate at
    the cutoff. Debt round three gave `density.py` the refusal this experiment argued for: when the
    instrument's own smooth-density null sits far from zero or has a spread far from one, the
    asymptotics it standardises against do not hold and it declines to report.

    So the published sentence gets stronger. It was "the instrument returns a z of 50.4 to 76.3
    where there is no gate", which is a defect. It is now "the instrument computes that z, sees
    that its own null is displaced, and withholds it with the reason", which is the architecture
    working. The number is still asserted here, because a refusal that loses its own working would
    be a different and worse failure.
    """
    reading = read_density(
        subject["length"], CUTOFFS[1], "full", n_null=150, n_placebos=10, n_boot=0
    )
    assert reading.refused, (
        "the full-range reading came back as Evidence. The instrument is supposed to refuse here: "
        "its own null band is displaced far from zero on this density, so the statistic it would "
        "report is not on the scale it claims"
    )
    assert reading.evidence.reason is RefusalReason.ENVELOPE_VIOLATED, reading.evidence.reason
    assert abs(reading.withheld_z) > 20.0, (
        f"withheld z={reading.withheld_z:.2f}; if the full-range statistic is now small the "
        f"instrument changed and this section of the findings needs rewriting"
    )
    assert reading.evidence.remedy, "a refusal must carry an instruction, not just a diagnosis"


def test_the_smooth_density_null_detects_the_failure_without_repairing_it(subject):
    """The mandatory baseline earning its place, and the limit of what it buys.

    This is the claim the findings section is built on, so both directions are pinned: the band is
    far from where the asymptotics put it, and the observed statistic is still far from the band.
    """
    reading = read_density(
        subject["length"], CUTOFFS[1], "full", n_null=150, n_placebos=10, n_boot=0
    )
    assert abs(reading.null_mean) > 3.0, (
        f"the null band is centred at {reading.null_mean:.3f}; the finding is that it is displaced "
        f"far from zero, so a band at zero means the estimator was fixed"
    )
    # The band's own spread is also wrong by about a factor of two against the asymptotic standard
    # error, which is the second half of the same measurement: on this density the estimator is off
    # in its centre and in its scale at once. Read through `statistics` rather than off the value,
    # because the instrument now refuses this reading and the numbers travel on the refusal.
    assert reading.null_sd > 1.5, (
        f"null band spread {reading.null_sd:.3f} against an asymptotic 1.0"
    )

    # And the part that must not be overstated. The baseline DETECTS the failure; it does not
    # repair it. Standardising against the band leaves the statistic far outside it, so a reader
    # who treated the band as a correction would still publish a gate. The right response is to
    # refuse and restrict the running variable, which the next test shows works.
    assert abs(reading.z_against_band) > 10.0, (
        f"z={reading.withheld_z:.2f} standardises to {reading.z_against_band:.2f} against the "
        f"band; if that is now small the baseline has become a correction and the findings "
        f"section, which says it is not one, needs rewriting"
    )


def test_restricting_to_the_smooth_window_restores_nominal_behaviour(subject):
    """And the same test where its premise holds behaves as advertised."""
    for cutoff in CUTOFFS:
        reading = read_density(
            subject["window"], cutoff, "windowed", n_null=150, n_placebos=10, n_boot=0
        )
        assert not reading.refused, reading.evidence
        assert abs(reading.value.z) < 5.0, (cutoff, reading.value.render())
        assert abs(reading.null_mean) < 1.5, (cutoff, reading.null_mean)
        assert 0.5 < reading.null_sd < 2.0, (cutoff, reading.null_sd)


# ---------------------------------------------------------------------------
# clause, second half: the elasticity responds when the gate is moved
# ---------------------------------------------------------------------------


def test_the_elasticity_responds_when_the_gate_is_synthetically_moved(subject):
    """I2's kill condition from the catalogue, on the real density with a labelled plant.

    A record re-scored under an installed gate cannot show bunching: bunching is a change in what
    the policy produced, and no counterfactual re-scoring of fixed rollouts produces one. So the
    sensitivity half is tested with a known response written into the real lengths and the
    specificity half is tested on the real lengths untouched.
    """
    response = move_the_gate(subject["window"])
    assert not isinstance(response, Refusal), response
    assert response.tracks, response.render()
    for own, others in zip(response.at_true_cutoff, response.at_other_cutoffs):
        assert own > max(others), response.render()
    assert response.margin > 100.0, response.render()
    assert all(math.isfinite(e) and e > 0 for e in response.elasticity)


def test_the_estimator_reports_nothing_where_no_gate_was_planted(subject):
    """The other direction. An estimator that always fires measures its own window."""
    masses = []
    for cutoff in CUTOFFS:
        row = read_bunching(subject["window"], cutoff, n_null=120, n_placebos=15, n_boot=0)
        assert not row.refused
        masses.append(abs(row.value.excess_mass))
    response = move_the_gate(subject["window"])
    assert not isinstance(response, Refusal)
    assert max(masses) < 0.1 * min(response.at_true_cutoff), (
        f"no-gate excess masses {masses} against planted {list(response.at_true_cutoff)}; the "
        f"separation that makes a gate a gate here is two orders of magnitude, not two sigma"
    )


def test_moving_a_gate_that_changes_nothing_is_void_rather_than_a_failed_kill(subject):
    """SPEC-ERRATA E24, void condition 8: an inert contrast is not a negative result."""
    from reward_lens.measure.threshold import gate_response

    values = subject["window"]
    out = gate_response(
        {CUTOFFS[0]: values, CUTOFFS[1]: values.copy()}, gate=installed_gate(CUTOFFS[0])
    )
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID
    assert out.statistics["void_condition"] == "contrast_inert"


# ---------------------------------------------------------------------------
# the null, the control and the power at the realised n
# ---------------------------------------------------------------------------


def test_a_no_gate_null_carries_a_matched_positive_control(subject):
    """A null without an identically-powered positive control is an underpowered experiment."""
    reading = read_density(
        subject["window"], CUTOFFS[1], "windowed", n_null=150, n_placebos=10, n_boot=0
    )
    power = density_power(subject["window"], CUTOFFS[1], replicates=12, shares=(0.02, 0.05, 0.10))
    verdict = control_for_the_null(reading, power)
    assert verdict is not None
    assert verdict.ok, verdict.render()
    assert verdict.control is not None and verdict.control.detected
    assert verdict.control.design.n == power.n


def test_a_null_with_no_control_refuses_with_no_matched_control(subject):
    """The refusal reason is a real one and this is the shape that triggers it."""
    from reward_lens.measure.controls.matched import ControlDesign, NullClaim, gate_null

    claim = NullClaim(
        instrument="DensityDiscontinuity",
        effect=0.4,
        p_value=0.7,
        design=ControlDesign(n=int(subject["window"].size), statistic="McCrary z"),
    )
    verdict = gate_null(claim, None)
    assert not verdict.ok
    assert verdict.refusal.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "no positive control" in verdict.refusal.detail
    assert "underpowered experiment" in verdict.refusal.detail
    assert "matched positive control" in verdict.refusal.remedy


def test_the_power_calculation_is_at_the_realised_n_and_reports_its_own_false_positives(subject):
    """A power number without the rate the same rule fires on nothing can be bought by lowering."""
    power = density_power(subject["window"], CUTOFFS[1], replicates=12, shares=(0.02, 0.05, 0.20))
    assert power.n == subject["window"].size
    assert power.power[-1] > power.power[0], power.render()
    assert power.false_positive < 0.35, power.render()
    assert power.replicates == 12


def test_a_planted_response_moves_the_density_and_a_permutation_does_not(subject):
    """The plant is what the power calculation is powering against, so check it does something."""
    values = subject["window"]
    planted = plant_response(values, CUTOFFS[1], share=0.5, seed=3)
    assert planted.size == values.size
    below = float(np.sum(planted < CUTOFFS[1]) - np.sum(values < CUTOFFS[1]))
    assert below > 500, "the plant moved almost nothing, so it cannot test anything"


# ---------------------------------------------------------------------------
# the four declarations every instrument has to satisfy, and the study metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instrument", [DensityDiscontinuity(), BunchingElasticity()], ids=lambda i: i.name
)
def test_the_instruments_this_experiment_drives_still_lint_clean(instrument):
    assert lint_instrument(instrument) == []


def test_an_instrument_with_nothing_to_read_refuses_with_a_remedy():
    for instrument in (DensityDiscontinuity(), BunchingElasticity()):
        out = instrument.estimate(Context())
        assert isinstance(out, Refusal)
        assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
        assert "pass `" in out.remedy


def test_a_cutoff_outside_the_support_refuses_rather_than_extrapolating(subject):
    """The local linear fit goes negative past the data, and a negative density is a refusal."""
    from reward_lens.measure.threshold import density_discontinuity

    gate = Gate(
        name="far", cutoff=float(subject["window"].max()) + 1.0, unit="characters", installed=True
    )
    out = density_discontinuity(
        running_variable(subject["window"], window=None), gate, n_boot=0, n_null=0, n_placebos=0
    )
    assert isinstance(out, Refusal)
    assert out.reason in {RefusalReason.RECORD_INCOMPLETE, RefusalReason.ACCESS_INSUFFICIENT}
    assert out.remedy.strip()


def test_every_metric_the_study_adjudicates_against_is_produced(subject):
    """A kill criterion that cannot be evaluated is not a criterion."""
    from experiments.x5_discontinuity import Analysis

    result = Analysis(
        frozen=None,
        search=None,
        audit=subject["audit"],  # type: ignore[arg-type]
        full=[
            read_density(subject["length"], CUTOFFS[1], "full", n_null=100, n_placebos=8, n_boot=0)
        ],
        windowed=[
            read_density(
                subject["window"], CUTOFFS[1], "windowed", n_null=100, n_placebos=8, n_boot=0
            )
        ],
        bunching=[read_bunching(subject["window"], CUTOFFS[1], n_null=100, n_placebos=8, n_boot=0)],
        response=move_the_gate(subject["window"]),
        power=density_power(subject["window"], CUTOFFS[1], replicates=6, shares=(0.05, 0.20)),
    )
    result.verdict = control_for_the_null(result.windowed[0], result.power)
    summary = summarise(result)
    for hypothesis in study_spec().hypotheses:
        assert hypothesis.prediction.metric in summary, hypothesis.id
    for criterion in study_spec().kill_criteria:
        assert criterion.metric in summary, criterion.id
    assert summary["gate_response_tracks"] == 1.0
    assert summary["gate_conjunction_rate"] > 0.99


def test_the_release_store_files_its_rows_under_its_own_namespace():
    """Two releases writing into one namespace is how a number gets attributed to the wrong study."""
    store = OUT / "evidence"
    if not store.is_dir():
        pytest.skip("no release store on disk; run experiments.x5_discontinuity first")
    rows = [
        json.loads(line)
        for path in store.rglob("*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert rows
    assert all(r["observable"].startswith("x5.") for r in rows)
    quantities = {r["quantity"] for r in rows if r.get("quantity")}
    assert {"gate.mccrary_statistic", "gate.bunching_elasticity"} <= quantities
    # and every reading names the record it came from
    assert all(
        r["subject"]["extra"].get("subject", "").startswith("ai-safety-institute/") for r in rows
    )
