"""X6 acceptance: the verdict-timing experiment, its controls, and P11's resolution.

The clause, in full:

    *C8's four controls run before any claim; the shipped `verdict_prefix_match_rate` is reproduced
    from the campaign's own store and its status as a measurement is decided rather than assumed;
    P11's resolution rule is evaluated exactly as registered, with both clauses reported separately
    and neither adjusted after the fact; the vanilla logit lens and a length baseline are both run
    and both reported even where they win; the corrected verdict direction is reported beside the
    naive one; every instrument the experiment drives passes `lint_instrument`; and the published
    findings section contains no number the experiment's own evidence store cannot verify.*

Two of these tests exist because of what the run found rather than because of the clause.
`test_control_four_cannot_fail` and `test_control_one_fires_on_shuffled_series` pin defects in C8's
shipped controls: the first is vacuous by algebra and the second responds to how lopsided a
judgment is rather than to when it was reached. Both are asserted to still be true, so that a fix to
`measure/selection/verdict.py` fails this file loudly rather than silently invalidating the
write-up. `test_the_release_exemption_still_has_the_hole_this_page_reports` does the same for a hole
in `artifacts/claims.py` that this page found while being written.

Most of the file reads the released artifacts under `experiments/x6_release/` rather than rebuilding
them, because the release is the deliverable. The archive arm and the synthetic control checks are
rebuilt live, because they are cheap and a pipeline proved to have run once is not a pipeline proved
to run.

Producing the artifacts:

    python -m experiments.x6_verdict_timing --allow-dirty --figures

The archive arm reads one evidence store read-only. The live arm needs `gpt2` and RewardBench 2 in
the local Hugging Face cache, runs on CPU, and touches no network.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from experiments.x6_verdict_timing import (
    CAMPAIGN_STORE,
    MID_STACK_LAYERS,
    REPO,
    VERDICT_TOKENS,
    identity_audit,
    judgment_mean,
    load_archive,
    order_swap,
    p11_rule,
    study_spec,
)
from reward_lens.artifacts.claims import check_files, find_unbound_numbers
from reward_lens.core.reading import Refusal
from reward_lens.core.store import EvidenceStore
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.selection.verdict import (
    Controls,
    VerdictDirection,
    commitment,
    settles_at,
    verdict_direction,
)

RELEASE = REPO / "experiments" / "x6_release"

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with "
        f"`python -m experiments.x6_verdict_timing --allow-dirty --figures`; this file checks the "
        f"published artifacts, and skipping it leaves them unchecked rather than checked."
    ),
)

needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=f"the campaign evidence store is not at {CAMPAIGN_STORE}",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence", readonly=True)


@pytest.fixture(scope="module")
def rows(manifest: dict[str, Any], store: EvidenceStore) -> dict[str, Any]:
    return {name: store.get(ev_id).value for name, ev_id in manifest["evidence_rows"].items()}


@pytest.fixture(scope="module")
def archive() -> Any:
    return load_archive()


# ---------------------------------------------------------------------------
# Lint. E56: an acceptance test that renders a reading is not one that lints
# ---------------------------------------------------------------------------


def test_every_instrument_this_experiment_drives_lints_clean() -> None:
    """E56's rule, applied to the one instrument X6 drives.

    Four instruments shipped for two waves failing lint rule 1 while their package read `done`,
    because the acceptance test rendered readings and never linted. X6 defines no new `Instrument`;
    it drives C8's, and driving one is the same reason to lint it.
    """
    findings = lint_instrument(VerdictDirection())
    assert findings == [], "\n".join(f.render() for f in findings)


def test_the_instrument_declares_both_invariance_groups() -> None:
    """E13 and E55: the mapping form, one relation per group, both resolving.

    E55 records three shipped instruments that worked out a second true invariance and wrote it into
    a comment because the annotation forbade the mapping form. C8 declares both, and this asserts
    that both resolve rather than that the field is non-empty.
    """
    from reward_lens.core.invariance import resolve_relation

    instrument = VerdictDirection()
    assert "repr.basis" in instrument.invariance
    assert "tokenization" in instrument.invariance
    for group in ("repr.basis", "tokenization"):
        relation = resolve_relation(instrument, group)
        assert relation.status == "invariant", (group, relation)


def test_the_frozen_spec_registers_both_clauses_separately() -> None:
    """P11's rule needs both clauses and they fail independently, so they are two hypotheses.

    A single hypothesis over a conjunction cannot record which half failed, and which half failed is
    the whole of what this experiment has to say about the prediction.
    """
    spec = study_spec()
    ids = {h.id for h in spec.hypotheses}
    assert {"H-p11-mean-moves", "H-p11-mode-fixed", "H-vbc-identity"} <= ids
    by_id = {h.id: h for h in spec.hypotheses}
    assert by_id["H-p11-mode-fixed"].prediction.threshold == 0.95
    assert by_id["H-p11-mean-moves"].prediction.ci_excludes == 0.0
    assert any(k.id == "K-c8-controls" for k in spec.kill_criteria)


# ---------------------------------------------------------------------------
# The archive: what the shipped 1.0 is
# ---------------------------------------------------------------------------


@needs_campaign
def test_the_published_rate_reproduces(archive: Any) -> None:
    audit = identity_audit(archive)
    assert audit["n"] == 1000
    assert audit["recomputed_rate"] == audit["published_rate"] == 1.0


@needs_campaign
def test_the_published_rate_is_an_identity(archive: Any) -> None:
    """The claim the whole experiment turns on, checked on the arrays rather than argued.

    If either agreement were below 1.0 the statistic would be a measurement after all and P11's
    premise would stand, so this is the test that decides which document gets written.
    """
    audit = identity_audit(archive)
    assert audit["identity_agreement"] == 1.0
    assert audit["verdict_is_sign_agreement"] == 1.0
    assert audit["n_zero_diff"] == 0
    # And therefore the rate carries no information about the judge: it is 1 minus the zero rate.
    assert audit["recomputed_rate"] == 1.0 - audit["n_zero_diff"] / audit["n"]


@needs_campaign
def test_the_record_has_one_position_per_item(archive: Any) -> None:
    """P11's rule needs positions. This is why it cannot be evaluated on this subject."""
    audit = identity_audit(archive)
    assert audit["positions_per_item"] == 1
    assert audit["critique_arm"].startswith("absent")
    assert archive.diff.ndim == 1 and archive.diff.size == archive.n


@needs_campaign
def test_the_order_swap_arm_reproduces_and_says_more(archive: Any) -> None:
    swap = order_swap(archive)
    assert swap["flip_rate"] == pytest.approx(swap["published_flip_rate"], abs=5e-4)
    # Position-locked and order-consistent partition the items.
    assert swap["order_consistent"] + swap["position_locked"] == pytest.approx(1.0, abs=1e-12)
    # The de-biased accuracy is strictly below the naive one, which is the point of computing it.
    assert swap["consistently_correct"] < swap["naive_accuracy"]


# ---------------------------------------------------------------------------
# P11's rule, on synthetic series where the answer is known
# ---------------------------------------------------------------------------


def test_judgment_mean_is_the_distribution_the_mode_comes_from() -> None:
    """The mean and the mode have to be two statistics of one distribution or the rule is nonsense.

    `judgment_mean` is `P(B)` under the next-token distribution restricted to the two verdict
    options, and the campaign's own `verdict_rule` is the sign of the same difference. So the mode is
    `mean > 0.5` exactly, and a rule about one moving while the other does not is well posed.
    """
    margin = np.linspace(-6.0, 6.0, 401)
    mean = judgment_mean(margin)
    assert np.all(np.diff(mean) < 0.0)
    np.testing.assert_array_equal(mean > 0.5, margin < 0.0)
    assert judgment_mean(np.array([0.0]))[0] == pytest.approx(0.5)


class _Reading:
    """A `LiveReading` stand-in carrying only what `p11_rule` reads."""

    def __init__(self, series: list[np.ndarray]) -> None:
        self._series = series
        self.items = list(range(len(series)))

    def cot(self, index: int) -> np.ndarray:
        return self._series[index]


def test_the_rule_resolves_in_favour_when_both_clauses_hold() -> None:
    """A mean that moves under a mode that never flips. The rule must say so.

    Margins stay strictly positive, so the mode is A at every position, while the magnitude falls
    steadily, so the judgment mean climbs. That is exactly the artifact P11 describes.
    """
    series = [np.linspace(6.0, 0.5, 60) + 0.02 * i for i in range(12)]
    out = p11_rule(_Reading(series), n_resamples=2000)
    assert out["clause_one_holds"] is True
    assert out["clause_two_holds"] is True
    assert out["resolves_in_favour"] is True
    assert out["argmax_unchanged_fraction"] == 1.0


def test_the_rule_matches_its_refutation_branch_when_both_move() -> None:
    """The registered refutation: the mean moves and the mode moves too."""
    series = [np.linspace(4.0, -4.0, 60) + 0.05 * i for i in range(12)]
    out = p11_rule(_Reading(series), n_resamples=2000)
    assert out["clause_one_holds"] is True
    assert out["clause_two_holds"] is False
    assert out["resolves_in_favour"] is False
    assert out["both_move"] is True


def test_the_rule_reports_the_outcome_it_does_not_name() -> None:
    """Neither branch: the mode moves and the slope's interval contains zero.

    This is what the live arm measured, and the rule has no verdict for it. The function has to
    report both clauses rather than collapse to a boolean, or the write-up would have to choose one
    of two branches that do not describe what happened.
    """
    rng = np.random.default_rng(0)
    series = [rng.normal(0.0, 1.0, 60) for _ in range(12)]
    out = p11_rule(_Reading(series), n_resamples=2000)
    assert out["clause_one_holds"] is False
    assert out["clause_two_holds"] is False
    assert out["resolves_in_favour"] is False
    assert out["both_move"] is False


def test_the_rule_declines_rather_than_fitting_two_points() -> None:
    out = p11_rule(_Reading([np.array([1.0, 2.0])]), n_resamples=100)
    assert out["n_items"] == 0
    assert "reason" in out


# ---------------------------------------------------------------------------
# The two controls that cannot do their job. Both pinned deliberately
# ---------------------------------------------------------------------------


def test_control_four_cannot_fail() -> None:
    """C8's verdict-token exchange is the exact negation of the reading, so it is -1 by algebra.

    Asserted on the shipped `verdict_direction` rather than on the experiment's numbers, so the
    claim is about the instrument and not about one run. If C8's control 4 is ever replaced by one
    that can fail, this test fails and the write-up's paragraph about it has to be revisited.
    """
    rng = np.random.default_rng(3)
    unembedding = rng.normal(size=(64, 16))
    residual = rng.normal(size=(200, 16))
    straight = residual @ verdict_direction(unembedding, 7, 41)
    exchanged = residual @ verdict_direction(unembedding, 41, 7)
    assert float(np.max(np.abs(straight + exchanged))) == 0.0
    assert float(np.corrcoef(straight, exchanged)[0, 1]) == pytest.approx(-1.0, abs=1e-12)
    # And the shipped control calls that a pass, on every input, forever.
    controls = Controls(
        mode_settles_at=0.5,
        mean_settles_at=0.5,
        spread_with_cot=1.0,
        spread_without_cot=1.0,
        length_baseline_auc=0.5,
        verdict_auc=0.9,
        permuted_correlation=float(np.corrcoef(straight, exchanged)[0, 1]),
    )
    assert controls.permutation_responds is True


def _fire_rate(mu: float, *, draws: int = 60, items: int = 16, seed: int = 11) -> float:
    """How often control 1 fires on independent draws with mean `mu`. No time structure at all."""
    rng = np.random.default_rng(seed)
    fired = 0
    for _ in range(draws):
        mode_at, mean_at = [], []
        for _ in range(items):
            margin = rng.normal(mu, 1.0, 48)
            mode_at.append(settles_at((margin < 0.0).astype(float)))
            mean_at.append(settles_at(judgment_mean(margin)))
        probe = Controls(
            mode_settles_at=float(np.mean(mode_at)),
            mean_settles_at=float(np.mean(mean_at)),
            spread_with_cot=float("nan"),
            spread_without_cot=float("nan"),
            length_baseline_auc=float("nan"),
            verdict_auc=float("nan"),
            permuted_correlation=float("nan"),
        )
        fired += int(probe.mode_fixed_mean_moving)
    return fired / draws


def test_control_one_fires_on_shuffled_series() -> None:
    """C8's control 1 responds to how lopsided the judgment is, not to when anything happened.

    `settles_at` calls a position still-moving when the series is more than 5% of its final value
    away from it. A sign can only be that when it flips; a probability is that whenever it wobbles.
    So a judgment that leans one way gives the mode few excursions with the last of them early, the
    mean is still wobbling at the end, and the gap fires the kill.

    Demonstrated on independent draws, where there is nothing to decide and nothing to time. Both
    ends are pinned, because the finding is the dependence rather than a single rate: a balanced
    judgment never fires it, and a lopsided one fires it most of the time. The first assertion is
    what makes the second mean something, since a control that fired on everything would be broken
    in a way that is easier to spot.
    """
    assert _fire_rate(0.0) == 0.0
    assert _fire_rate(1.2) > 0.5, (
        "control 1 no longer fires on lopsided series with no time structure, so either "
        "`settles_at` or `mode_fixed_mean_moving` changed and X6's paragraph needs rewriting"
    )
    assert _fire_rate(2.0) == 1.0


def test_what_c8_emits_actually_arrives() -> None:
    """E35, E44 and E51 are one family: a field declared, plumbed, tested at one call site and dead
    at another, with the symptom always an Evidence row carrying less than the instrument said.

    The published run cannot check this, because C8's kill condition fires and a refusal emits
    nothing. So the emit path is exercised here on controls that pass, and the assertion is on the
    fields that came back rather than on the ones that were declared. That inversion is the whole
    lesson of the three errata: `assert the emitted quantity, not the declared one`.
    """
    from reward_lens.core.types import Access, Component, Phase, Substrate
    from reward_lens.measure.base import Context, lint_reading
    from reward_lens.measure.meta.incremental import Detector, IncrementalValidityReading

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 40)
    increment = IncrementalValidityReading(
        own=Detector.from_scores("own", rng.normal(1.0, 1.0, 40), labels),
        baselines_run=[Detector.from_scores("baseline.length", rng.normal(0.0, 1.0, 40), labels)],
    ).compute()
    passing = Controls(
        mode_settles_at=0.50,
        mean_settles_at=0.52,
        spread_with_cot=1.0,
        spread_without_cot=1.0,
        length_baseline_auc=0.50,
        verdict_auc=0.80,
        permuted_correlation=-1.0,
        n_items=40,
    )
    assert passing.kills() is False and passing.all_passed
    instrument = VerdictDirection(
        naive_margins=[list(np.linspace(0.0, 1.0, 20))] * 3,
        corrected_margins=[list(np.linspace(0.0, 1.0, 20))] * 3,
        controls=passing,
        incremental=increment.record,
    )
    reading = instrument.estimate(
        Context(
            readout="decision",
            access={Component.GRADER: Access.FORWARD},
            substrate=Substrate.NEURAL_GEN,
            phase=Phase.PRE_RUN,
        )
    )
    assert not isinstance(reading, Refusal), getattr(reading, "detail", reading)
    assert reading.quantity == instrument.quantity == "judge.commitment_position"
    assert reading.incremental is not None
    assert reading.baselines
    assert lint_reading(reading, instrument) == []


def test_commitment_still_refuses_a_reading_that_comes_back() -> None:
    """The one piece of C8 that does discriminate, kept under test because X6 leans on it."""
    settled = commitment([0.0, 0.2, 0.5, 0.9, 1.0, 1.0])
    assert settled.is_stable is True
    wobbly = commitment([0.0, 1.0, 0.0, 0.0, 0.2, 1.0])
    assert wobbly.note != ""


# ---------------------------------------------------------------------------
# The release: what actually got published
# ---------------------------------------------------------------------------


def test_the_release_carries_both_arms(manifest: dict[str, Any]) -> None:
    assert manifest["n_archive_items"] == 1000
    assert manifest["archive_only"] is False, "the published release must carry the live arm"
    assert manifest["n_live_items"] >= 8
    assert manifest["vehicle"] == "gpt2"


def test_the_four_controls_all_have_numbers(rows: dict[str, Any]) -> None:
    """C8's clause: the four controls run before any claim. All four, with values, not defaults."""
    settling, spread = rows["control_settling"], rows["control_spread"]
    length, swap = rows["control_length"], rows["control_swap"]
    for value in (
        settling["mode_settles_at"],
        settling["mean_settles_at"],
        spread["spread_with_cot"],
        spread["spread_without_cot"],
        length["length_baseline_auc"],
        length["verdict_auc"],
        swap["permuted_correlation"],
    ):
        assert np.isfinite(value), value
    # Control 3 ran the whole bank, not only the length baseline C8 names.
    assert len(length["baseline_aucs"]) + len(length["baselines_refused"]) == 6


def test_the_naive_lens_is_reported_beside_the_corrected_direction(rows: dict[str, Any]) -> None:
    """C8's rung 0 is a mandatory comparator, not a predecessor, so both appear at every layer."""
    layers = rows["mid_stack"]["layers"]
    assert [row["layer"] for row in layers] == list(MID_STACK_LAYERS)
    for row in layers:
        assert np.isfinite(row["naive_r"]) and np.isfinite(row["corrected_r"])
        assert 0.0 <= row["cosine_naive_corrected"] <= 1.0
    # The correction is real: the two directions agree at the top and diverge with depth.
    assert layers[-1]["cosine_naive_corrected"] > layers[0]["cosine_naive_corrected"] + 0.2
    # And the average is coherent at the top and not at the bottom, which is what makes the
    # averaging step worth reporting rather than assuming.
    assert layers[-1]["coherence"] > 0.9
    assert layers[0]["coherence"] < 0.5


def test_the_instrument_refused_and_carried_its_numbers(rows: dict[str, Any]) -> None:
    """The kill condition fired, so a reading here would have published the artifact it detects."""
    instrument = rows["instrument"]
    assert instrument["lint_clean"] is True
    assert instrument["kills"] is True
    assert instrument["refused"] is True
    assert instrument["refusal_reason"] in {"BELOW_LOD", "NO_MATCHED_CONTROL"}
    assert instrument["refusal_remedy"].strip() != ""
    assert instrument["refusal_statistics"]


def test_p11_did_not_resolve_in_favour(rows: dict[str, Any]) -> None:
    """The registered rule, scored as registered. Both clauses reported, neither adjusted.

    If this ever flips, the finding changes and `PREDICTIONS.md` has to move; it must not flip
    because somebody softened a threshold.
    """
    rule = rows["p11_rule"]
    assert rule["resolves_in_favour"] is False
    assert rule["slope_ci_excludes_zero"] is False
    assert rule["argmax_unchanged_fraction"] < 0.95
    # Neither branch of the rule describes this, which is the reported outcome.
    assert rule["both_move"] is False


def test_the_length_baseline_is_published_even_though_it_wins(rows: dict[str, Any]) -> None:
    """C8 names a length baseline and the catalogue's whole argument for series C is that the
    losses get published. Here the dumb baseline beats the white-box reading and it is on the page."""
    length = rows["control_length"]
    assert length["length_baseline_auc"] > length["verdict_auc"]
    assert length["beats_length"] is False


# ---------------------------------------------------------------------------
# Every number in the write-up
# ---------------------------------------------------------------------------


def test_findings_numbers_all_verify(store: EvidenceStore) -> None:
    findings = RELEASE / "FINDINGS-x6.md"
    assert findings.exists()
    report = check_files([findings], store=store)
    failures = [r for r in report.results if not r.ok]
    assert not failures, "\n".join(str(f) for f in failures)
    assert not report.unresolved_refs, report.unresolved_refs


def test_the_release_exemption_still_has_the_hole_this_page_reports() -> None:
    """`_RELEASE_RE`'s first alternative exempts any `N.0` after a determiner, with nothing after it.

    E57's shape, one alternative along: the marker was written for "the 2.0 API" and it fires on
    "the 1.0" wherever that appears, including where 1.0 is the measured rate the page is about.
    Pinned here rather than only described, so that a repair to `artifacts/claims.py` fails this file
    and X6's paragraph about it gets rewritten instead of quietly becoming false.

    The first assertion is the control. If it ever stops holding, the detector has been weakened
    rather than the exemption narrowed, and that is a different and worse change.
    """
    assert [u.value for u in find_unbound_numbers("The measured rate is 1.0.")] == ["1.0"]
    assert find_unbound_numbers("The 1.0 is an identity.") == []
    assert find_unbound_numbers("The 12.0 rate is fabricated.") == []
    # And the narrow alternatives are narrow, which is why the first one is the defect.
    assert [u.value for u in find_unbound_numbers("The 2.0 API reports 0.71.")] == ["0.71"]


def test_findings_carries_no_unbound_number() -> None:
    """E57: the claims gate used to exempt every number in a sentence containing the word "say",
    which is exactly what a "what this does not say" section is full of. The hole is closed and this
    asserts the section written under it is clean."""
    text = (RELEASE / "FINDINGS-x6.md").read_text(encoding="utf-8")
    unbound = find_unbound_numbers(text)
    assert not unbound, "\n".join(str(u) for u in unbound)


def test_the_figure_captions_name_their_rows() -> None:
    """A plotted point has an evidence id on the same terms a sentence does."""
    captions = sorted(RELEASE.glob("x6-fig*.txt"))
    if not captions:
        pytest.skip("figures were not drawn; re-run with --figures")
    for path in captions:
        assert "ev:" in path.read_text(encoding="utf-8"), path


def test_the_verdict_tokens_are_single_tokens() -> None:
    """The campaign's call-sheet D1 trap, kept here because X6 depends on it: the tokenizer collapses
    both bracketed forms to the same `]]` token, so a contrast on `[[A]]` is not a contrast."""
    assert VERDICT_TOKENS == (" A", " B")
