"""The E1 preflight's measurement functions, on inputs whose answer is known by construction.

`tests/test_e1_preflight.py` covers the column enumerator, the configuration adjudicator and the
artifact bookkeeping. `tests/test_measure_reconstruct.py` covers the library primitives these
functions stand on. Between the two sat every function that turned a primitive into a number the
gate document printed, and none of them was tested at all.

Everything here is synthetic and every expected value is fixed by construction rather than read off
the published artifact. That is the whole point. The real parquet is 189 MB, the acceptance tests
already run against it, and a test whose expected value was copied out of the run it is meant to
check asserts only that the code still does whatever it did on the day somebody looked. A test that
knows the answer in advance can tell a defect in the instrument from a finding about the data, which
is the distinction a preflight exists to make.

Where a function documents a failure mode, the failure mode is exercised as well: a short row block,
an absent tokeniser, a trainer log with a hole in its step field, a candidate weighting that cannot
reach the observed maximum. Three of the constructions here describe situations the real run does
not contain, and those are the ones worth writing most: a coincidence that never fires on one
artifact is untested code, not a theorem.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence

import numpy as np
import pytest
from experiments.e1_preflight import preflight

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.ledger.imported import AlignmentStrength, StepAxis
from reward_lens.measure.ledger.price import StepSample
from reward_lens.measure.ledger.trainer_log import TrainerLog


def _log(steps: Sequence[float], **series: Sequence[float]) -> TrainerLog:
    """A `TrainerLog` whose entries carry the named series against the given steps.

    Built by hand rather than read from a file because the step field is the thing several of these
    tests are about, and a fixture that always wrote a dense index starting at one would quietly
    remove the case that matters.
    """
    entries: tuple[dict[str, Any], ...] = tuple(
        {"step": float(step), **{key: float(values[i]) for key, values in series.items()}}
        for i, step in enumerate(steps)
    )
    return TrainerLog(entries=entries, state={})


# ---------------------------------------------------------------------------
# P3. Recovering the optimiser's group from the row order
# ---------------------------------------------------------------------------


def test_blocks_numbers_the_row_runs_within_each_file_from_zero():
    """The group label is the row's ordinal inside its own file over the generation count.

    This is the one determination six of the seven sections are denominated in, and it is recovered
    from the row order rather than from a key, so nothing in the table itself would contradict a
    wrong answer. Pinning the labels for a two-block file followed by a one-block file fixes both
    halves of the arithmetic: that the counter divides by the group size, and that it restarts at
    zero when the file changes rather than running on across the whole table.
    """
    index = np.repeat([0.0, 1.0], [32, 16])
    got = preflight.blocks(index, size=16)
    assert not isinstance(got, Refusal)
    assert got.tolist() == [0] * 16 + [1] * 16 + [0] * 16


def test_blocks_refuses_when_a_file_holds_a_short_trailing_block():
    """A block shorter than the generation count displaces every boundary after it, so it refuses.

    The version this function replaced returned the displaced labels with no comment. That is the
    worst available behaviour here, because the labels are structurally plausible, the arithmetic
    downstream runs to completion, and the groups being normalised over are not the groups the
    trainer normalised over. The refusal has to name the offending block, so the count and its
    location are asserted rather than merely the type.
    """
    index = np.repeat([0.0, 1.0], [16, 8])
    got = preflight.blocks(index, size=16)
    assert isinstance(got, Refusal)
    assert got.instrument == "e1_preflight.blocks"
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "step 1 block 0 at 8 rows" in got.detail
    assert got.statistics["n_irregular"] == 1


def test_group_codes_are_assigned_by_sorted_order_and_not_by_hash():
    """The same labels give the same codes in a fresh process, which `hash` did not.

    Python seeds `hash` per process, so the codes moved between two runs of the same command and a
    collision would have merged two prompt groups differently on each. Checking determinism by
    running the function twice inside one process would pass under the old implementation as well,
    so the exact code vector is asserted instead: it is reproducible only if the assignment comes
    from the sorted order of the labels themselves.
    """
    codes = preflight.group_codes(["delta", "alpha", "charlie", "alpha", "bravo"])
    assert codes.tolist() == [3, 0, 2, 0, 1]
    assert codes.dtype == np.int64


def test_group_codes_sort_the_string_form_of_a_label_rather_than_its_value():
    """The order is lexicographic over `str(label)`, so a numeric label sorts as text.

    Worth pinning because the caller passes `problem_id` after casting it to string, so the
    behaviour is load-bearing on the real run, and because the alternative reading of "sorted order"
    would put 2.0 before 10.0 and produce a different partition of the same table. The partition is
    the same under either rule, but the codes are not, and the codes are what get compared.
    """
    assert preflight.group_codes([10.0, 2.0, 10.0]).tolist() == [0, 1, 0]


def test_per_file_means_exclude_nulls_per_column_rather_than_per_row():
    """A null in one column must not change the denominator of the columns beside it.

    A row-wise drop is the tempting implementation and it is wrong for a specific reason: a null in
    an adjudication column is an unscored rollout, and removing its whole row silently changes the
    per-step mean of every other column at that step, which is the quantity the closure test is
    computed against. The construction puts a null in `a` on a row where `b` is populated, and `b`'s
    mean at that file is 6.0 under the per-column rule against 8.0 under the per-row one.
    """
    index = np.asarray([0.0, 0.0, 1.0, 1.0])
    columns = {
        "a": np.asarray([1.0, 3.0, np.nan, 5.0]),
        "b": np.asarray([2.0, 2.0, 4.0, 8.0]),
    }
    files, means = preflight.per_file_means(index, columns)
    assert files.tolist() == [0.0, 1.0]
    assert means["a"].tolist() == [2.0, 5.0]
    assert means["b"].tolist() == [2.0, 6.0]


# ---------------------------------------------------------------------------
# P5. Step alignment
# ---------------------------------------------------------------------------


def test_token_lengths_refuses_by_name_when_the_tokeniser_package_is_absent(monkeypatch):
    """No character-count substitute, and the refusal says which package would fix it.

    `completions/mean_length` is logged in tokens. Comparing it against a character count and
    calling the ratio stable is exactly the plausible proxy the standing rule forbids, and the
    caller stops the whole run on this refusal rather than continuing on an assumed offset. Absence
    is simulated by putting `None` in `sys.modules`, which is what the import system treats as a
    halted import, so nothing is uninstalled and the test leaves the environment as it found it.
    """
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    got = preflight.token_lengths(["some completion"], "/nowhere/tokenizer.json")
    assert isinstance(got, Refusal)
    assert got.instrument == "e1_preflight.token_lengths"
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "`tokenizers` package is not installed" in got.detail
    assert "character-count comparison is not a substitute" in got.remedy


def _aligned_length_fixture(offset: int, template_tokens: int = 7):
    """Eight files of completion lengths, and a log carrying them displaced by a known offset.

    Each file gets its own range, so the identity that decides the alignment can only hold at the
    shift the fixture was built at: the test asks whether the logged minimum and the logged maximum
    sit the same distance above the file's own, and two files with different ranges cannot both
    satisfy that against each other's extremes.
    """
    index = np.repeat(np.arange(8, dtype=np.float64), 4)
    lengths: list[int] = []
    for f in range(8):
        low, high = 100 + 3 * f, 110 + 4 * f
        lengths.extend([low, low + 1, high - 1, high])
    log = _log(
        [float(f + offset) for f in range(8)],
        **{
            "completions/min_length": [100 + 3 * f + template_tokens for f in range(8)],
            "completions/max_length": [110 + 4 * f + template_tokens for f in range(8)],
        },
    )
    return index, np.asarray(lengths, dtype=np.int64), log


def test_alignment_evidence_picks_the_shift_where_the_offset_difference_vanishes():
    """The chosen shift is the one where the logged and stored extremes differ by one constant.

    The mean of a length series agrees under any nearby alignment and under a parallel sample too,
    which is why the mean is not the statistic used. The minimum and the maximum are order
    statistics of one particular set of completions, and the difference between the two offsets is
    zero exactly when both series describe the same completions. The fixture is built at an offset
    of one, so the shift is recovered rather than assumed, and the constant recovered as the median
    offset is the seven template tokens the fixture added.
    """
    index, lengths, log = _aligned_length_fixture(offset=1)
    got = preflight.alignment_evidence(index, lengths, log)

    assert got["best_shift"] == 1
    assert got["margin"] == 8
    at_best = got["shifts"]["1"]
    assert at_best["n_files"] == 8
    assert at_best["n_unclipped"] == 8
    assert at_best["n_offset_consistent"] == 8
    assert at_best["frac_offset_consistent"] == 1.0
    assert at_best["median_offset"] == 7.0
    assert at_best["mean_abs_offset"] == 7.0
    assert all(got["shifts"][str(s)]["n_offset_consistent"] == 0 for s in (-2, -1, 0, 2))


def test_alignment_evidence_reports_no_disagreement_when_every_file_agrees():
    """With the identity holding everywhere the disagreement report carries counts and nothing else.

    The early return is the shape a reader has to be able to tell apart from a report full of NaN
    quantiles, because a quantile of an empty list and a quantile of a list of zeros mean opposite
    things about the alignment. Asserting the exact key set is the only way to pin that.
    """
    index, lengths, log = _aligned_length_fixture(offset=1)
    got = preflight.alignment_evidence(index, lengths, log)
    assert got["disagreeing_files"] == {"n_unclipped": 8, "n_disagreeing": 0}


def _disagreement_fixture(cap: int = 1000):
    """Six unclipped files, three of which fail the offset identity, plus one clipped file.

    The gaps to the cap are chosen so the disagreeing files are not uniformly nearer it than the
    agreeing ones. That keeps the test from accidentally asserting the finding the function was
    written to look for, which is whether the disagreement concentrates near the generation cap.
    """
    rows = [
        (0, 990, 980),
        (1, 800, 790),
        (2, 400, 390),
        (3, 950, 900),
        (4, 900, 800),
        (5, 300, 100),
        (6, cap, 900),
    ]
    index = np.repeat(np.arange(len(rows), dtype=np.float64), 2)
    lengths = np.tile(np.asarray([10, 20]), len(rows))
    log = _log(
        [float(f) for f, _, _ in rows],
        **{
            "completions/min_length": [lmin for _, _, lmin in rows],
            "completions/max_length": [lmax for _, lmax, _ in rows],
        },
    )
    return index, lengths, log


def test_characterise_disagreement_counts_the_unclipped_files_and_the_failing_ones():
    """Files at the generation cap are excluded, and the rest are split by whether the identity held.

    A completion truncated at the cap is not an order statistic of the sampled lengths, it is the
    cap, so it cannot satisfy an identity between order statistics and it is filtered out by a
    strict inequality. The fixture includes one file exactly at the cap to check that the strict
    inequality drops it, and the reported medians are asserted because they are what decide whether
    the disagreement is a filter that is slightly too generous or an identity that is not one.
    """
    index, lengths, log = _disagreement_fixture()
    got = preflight._characterise_disagreement(index, lengths, log, 0, cap=1000)

    assert got["n_unclipped"] == 6
    assert got["n_disagreeing"] == 3
    assert got["median_gap_to_cap_disagreeing"] == 100.0
    assert got["median_gap_to_cap_agreeing"] == 200.0
    assert got["frac_disagreeing_within_128_of_cap"] == pytest.approx(2.0 / 3.0)
    assert got["frac_agreeing_within_128_of_cap"] == pytest.approx(1.0 / 3.0)
    assert got["median_absolute_offset_difference"] == 90.0
    assert got["max_absolute_offset_difference"] == 190.0


def test_characterise_disagreement_bands_keep_nested_subsets_as_the_margin_widens():
    """Each cap-margin band keeps a subset of the previous one, so both counts fall or hold.

    The bands are a tightening of the same filter, reported rather than applied, and the reason they
    are only reported is that moving a threshold after seeing the data is not a filter. What can be
    checked without deciding anything is the structure: the kept set shrinks with the margin and the
    consistent count cannot rise. The fraction is deliberately not asserted to be monotone, because
    it is not: it moves up between the 64 and 128 bands here, since a wider margin can drop an
    agreeing file and a disagreeing one at different rates.
    """
    index, lengths, log = _disagreement_fixture()
    bands = preflight._characterise_disagreement(index, lengths, log, 0, cap=1000)[
        "consistency_by_cap_margin"
    ]
    kept = [bands[str(w)]["n_kept"] for w in (0, 64, 128, 256, 512)]
    consistent = [bands[str(w)]["n_consistent"] for w in (0, 64, 128, 256, 512)]

    assert kept == [6, 4, 3, 2, 2]
    assert consistent == [3, 2, 2, 1, 1]
    assert all(b <= a for a, b in zip(kept, kept[1:]))
    assert all(b <= a for a, b in zip(consistent, consistent[1:]))


def test_align_pairs_on_the_step_value_and_skips_a_hole_in_the_logged_series():
    """A missing entry in the trainer log drops that one pair rather than sliding every later one.

    This is the repair that matters most in the module. The pairing used to be
    `observed[(files + offset) - 1]`, which indexes the logged series by position and is right only
    while that series is dense and starts at step one. The fixture removes step 3 from the log, so
    positional indexing would hand file 3 the value the trainer wrote at step 5 and every later
    quantity would be off by one. Pairing by value drops file 2, keeps file 3 against the value
    logged at step 4, and the two answers are different arrays rather than different tolerances.
    """
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    files = np.arange(5, dtype=np.float64)
    observed = np.asarray([10.0, 20.0, 40.0, 50.0])
    observed_steps = np.asarray([1.0, 2.0, 4.0, 5.0])

    aligned, obs = preflight._align(values, files, observed, 1, observed_steps)
    assert aligned.tolist() == [0.0, 1.0, 3.0, 4.0]
    assert obs.tolist() == [10.0, 20.0, 40.0, 50.0]


def test_align_falls_back_to_a_dense_series_numbered_from_step_one():
    """With no step field supplied the logged series is read as steps 1..n, and nothing else.

    The fallback is kept because the subject run's own log genuinely is dense from step one, and
    naming that assumption once here is better than leaving it unstated at nine call sites. Pinning
    it means the assumption is visible: at an offset of zero the first file asks for step 0, which
    a series numbered from one does not carry, and the pair is dropped rather than silently taken
    from position zero.
    """
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    files = np.arange(5, dtype=np.float64)
    observed = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])

    aligned, obs = preflight._align(values, files, observed, 1)
    assert aligned.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert obs.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    aligned_zero, obs_zero = preflight._align(values, files, observed, 0)
    assert aligned_zero.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert obs_zero.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_alignment_of_names_both_axes_rather_than_carrying_a_loose_integer():
    """The offset travels as a determination from the file index onto the trainer's log step.

    An integer in a function call is a number somebody typed, and the type exists so that a reader
    can ask which axis is displaced from which and what the evidence was. The source and target are
    asserted in that direction because inverting them inverts the sign of every later join, and the
    strength is asserted because an assumed alignment refuses at `join_on_axis` while an established
    one does not.
    """
    got = preflight.alignment_of(
        1, evidence="the order-statistic identity holds here and nowhere else"
    )
    assert got.source is StepAxis.ROLLOUT_FILE_INDEX
    assert got.target is StepAxis.TRAINER_LOG_STEP
    assert got.offset == 1
    assert got.strength is AlignmentStrength.IDENTITY
    assert not got.is_assumed
    assert got.to([0.0, 1.0, 2.0]).tolist() == [1.0, 2.0, 3.0]

    fitted = preflight.alignment_of(
        -2, evidence="the residual curve minimises here", strength="fitted"
    )
    assert fitted.strength is AlignmentStrength.FITTED
    assert fitted.offset == -2


def _degeneracy_shift_fixture():
    """A predicted degeneracy series and a logged one that reproduces it at a shift of two.

    Two adjacent pairs of the predicted series are deliberately equal, so a neighbouring shift picks
    up a couple of accidental matches. That is what makes the margin a quantity rather than the same
    number as the count: with every value distinct the runner-up would be zero and the test would
    not distinguish "the best shift wins" from "the best shift is the only one that scores".
    """
    predicted = np.asarray([0.0, 0.1, 0.1, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9])
    files = np.arange(10, dtype=np.float64)
    observed = np.full(11, -1.0)
    for f in range(10):
        observed[(f + 2) - 1] = predicted[f]
    return predicted, files, observed


def test_degeneracy_shift_curve_recovers_the_shift_the_data_was_built_at():
    """The exact-match count peaks at the true offset, and the margin is the gap to the runner-up.

    This is an alignment test the reward cannot spoil. Whether a group's reward is constant depends
    only on whether its components are constant, so the predicted series is exactly right even where
    the composition misses the reward's level by thirteen per cent, and it is a fraction rather than
    a scale so no rescaling can manufacture a match. The fixture puts two accidental matches at each
    neighbouring shift, so the asserted margin of eight is the real gap rather than the count again.
    """
    predicted, files, observed = _degeneracy_shift_fixture()
    got = preflight.degeneracy_shift_curve(predicted, files, observed)

    assert got["best_shift"] == 2
    assert got["best_n_exact"] == 10
    assert got["runner_up_n_exact"] == 2
    assert got["margin"] == 8
    at_best = got["shifts"].index(2)
    assert got["n_steps"][at_best] == 10
    assert got["absolute_rms"][at_best] == 0.0


# ---------------------------------------------------------------------------
# P3. The degeneracy structure, which settles the composition
# ---------------------------------------------------------------------------


def test_degeneracy_predicted_is_the_fraction_of_groups_with_every_component_constant():
    """Per file, the count of fully constant groups over the group count, and nothing else.

    A group's reward is constant exactly when every component entering it is constant, which is what
    makes this independent of the weights and therefore able to settle the composition where the
    weights are unidentified. The first file holds four groups of two, of which two are constant in
    both components, so the answer is a half by inspection; the second file is constant throughout,
    which pins that the denominator is the file's own group count rather than the table's.
    """
    index = np.repeat([0.0, 1.0], 8)
    group = np.tile(np.asarray([0, 0, 1, 1, 2, 2, 3, 3]), 2)
    components = {
        "a": np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float),
        "b": np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float),
    }
    files, predicted = preflight.degeneracy_predicted(index, group, components)
    assert files.tolist() == [0.0, 1.0]
    assert predicted.tolist() == [0.5, 1.0]


def _reproduction_fixture(second_file_constant: bool):
    """Three files of two groups, whose degenerate fraction is one, a half and zero in turn.

    The middle file is the interior step, and the flag decides whether the candidate component set
    reproduces it or overshoots to a fully degenerate step. A step where every group is degenerate
    or none is would be matched by almost any hypothesis, so the interior step is the whole test.
    """
    index = np.repeat([0.0, 1.0, 2.0], 4)
    group = np.tile(np.asarray([0, 0, 1, 1]), 3)
    middle = [1, 1, 2, 2] if second_file_constant else [1, 1, 2, 3]
    component = np.asarray([1, 1, 1, 1, *middle, 4, 5, 6, 7], dtype=float)
    observed = np.asarray([1.0, 0.5, 0.0])
    return index, group, {"a": component}, observed


def test_degeneracy_reproduction_counts_the_interior_step_it_matched():
    """An exact match on every step, and the interior step counted apart from the saturated ones.

    Counting four hundred agreements together overstates the evidence when most steps are saturated,
    because a step where every group is degenerate agrees with almost any candidate. The interior
    steps are the ones where the count could have come out several ways and did not, which is why
    they are reported separately and why the interior count is asserted here alongside the total.
    """
    index, group, components, observed = _reproduction_fixture(second_file_constant=False)
    got = preflight.degeneracy_reproduction(index, group, components, observed, 1)

    assert got["components"] == ["a"]
    assert got["n_steps"] == 3
    assert got["n_exact"] == 3
    assert got["frac_exact"] == 1.0
    assert got["n_interior_steps"] == 1
    assert got["n_exact_on_interior"] == 1
    assert got["absolute_rms"] == 0.0
    assert got["predicted"] == [1.0, 0.5, 0.0]


def test_degeneracy_reproduction_reports_an_interior_step_it_failed_to_match():
    """A candidate that predicts a degenerate step where the trainer logged a half is caught here.

    Under the failing candidate the two saturated steps still agree, so the headline exact count
    only falls from three to two and the fraction stays above two thirds. It is the interior count
    that goes to zero, which is the point of reporting it: a candidate can look almost right on the
    total and have got the only informative step wrong.
    """
    index, group, components, observed = _reproduction_fixture(second_file_constant=True)
    got = preflight.degeneracy_reproduction(index, group, components, observed, 1)

    assert got["n_exact"] == 2
    assert got["n_interior_steps"] == 1
    assert got["n_exact_on_interior"] == 0
    assert got["predicted"] == [1.0, 1.0, 0.0]
    assert got["absolute_rms"] == pytest.approx(math.sqrt(0.25 / 3.0))


# ---------------------------------------------------------------------------
# P2. The reward composition
# ---------------------------------------------------------------------------


def test_lattice_of_recovers_the_per_rollout_grid_from_the_logged_mean():
    """A mean of `n` rewards on a grid of `u` lands on `u / n`, so `n` times the grid recovers `u`.

    This is a constraint on the composition that costs nothing and refutes several otherwise
    feasible weightings before a residual is computed, so it has to be right in both directions.
    The values are exact multiples of one two-hundred-and-fifty-sixth and of no coarser grid, and at
    sixty-four rollouts a step the implied per-rollout lattice is a quarter, which is the resolution
    the format scorer's own source computes on.
    """
    values = np.asarray([k / 256 for k in (1, 2, 3, 5, 7)])
    got = preflight.lattice_of(values, 64)
    assert got["logged_grid"] == 1.0 / 256.0
    assert got["logged_grid_reciprocal"] == 256.0
    assert got["implied_per_rollout_grid"] == 0.25
    assert got["rounding_rms"] == (1.0 / 256.0) / math.sqrt(12.0)


def test_lattice_of_reports_no_grid_rather_than_a_very_fine_one():
    """A series the trainer did not round returns zero, and the implied quantities become None.

    The honest answer for a quantity with no grid is that there is no grid, and the failure to guard
    against it would be to divide by zero and report a reciprocal of infinity as if it were a
    resolution. Two values with no common rational step are enough to establish the case.
    """
    got = preflight.lattice_of(np.asarray([math.pi / 10.0, math.e / 10.0]), 64)
    assert got["logged_grid"] == 0.0
    assert got["logged_grid_reciprocal"] is None
    assert got["implied_per_rollout_grid"] is None
    assert got["rounding_rms"] == 0.0


def test_ceiling_test_refutes_exactly_when_the_observed_maximum_is_above_the_weight_sum():
    """Every component is bounded in the unit interval, so the weight sum is the largest attainable.

    A configuration that cannot reach the observed maximum is refuted by arithmetic, which is a
    stronger refutation than any residual and does not depend on a fit. Both sides of the boundary
    are asserted because the interesting case is the equality: a maximum that exactly equals the
    weight sum is reached rather than exceeded, and a test that refuted there would refute the
    correct configuration on the very step that confirms it.
    """
    reachable = preflight.ceiling_test(np.asarray([4.0, 5.0, 4.5]), [1.0, 4.0])
    assert reachable["max_attainable"] == 5.0
    assert reachable["max_observed"] == 5.0
    assert reachable["n_steps_exceeding"] == 0
    assert reachable["n_steps"] == 3
    assert reachable["refuted"] is False
    assert reachable["largest_excess"] == 0.0

    refuted = preflight.ceiling_test(np.asarray([4.0, 5.5]), [1.0, 4.0])
    assert refuted["n_steps_exceeding"] == 1
    assert refuted["refuted"] is True
    assert refuted["largest_excess"] == 0.5


def test_feasibility_counts_the_unit_interval_and_the_component_lattice_separately():
    """A residual has to lie in `[0, 1]` and on the component's own lattice, and the two differ.

    The format component's per-step mean is not observable, because the table carries the scorer's
    output only as a binary flag, so what is testable is whether it could exist at all. The lattice
    constraint is the one that decides it: a component taking quarter values over four rollouts has
    a per-step mean on a grid of one sixteenth and can land nowhere else, which refutes a candidate
    weight without any tolerance and without any fit. The seven residuals here include a value
    inside the interval but off the lattice and one on the lattice but outside the interval, so the
    two counts cannot be satisfied by the same set of steps.
    """
    implied = np.asarray([0.0, 0.0625, 0.5, 1.0, 1.5, -0.25, 0.03])
    got = preflight.feasibility(implied, np.zeros(7), n_per_step=4, increments=4)

    assert got["weight"] == 1.0
    assert got["n_steps"] == 7
    assert got["component_lattice"] == 0.0625
    assert got["implied_min"] == -0.25
    assert got["implied_max"] == 1.5
    assert got["n_inside_unit_interval"] == 5
    assert got["n_on_component_lattice"] == 6
    assert got["n_satisfying_both"] == 4
    assert "n_inside_with_lower_bound" not in got


def test_feasibility_applies_the_lower_bound_only_where_one_is_supplied():
    """The lower bound narrows the interval count and is reported under its own key.

    The bound is the binarisation's own mean, which the residual has to clear when the published
    flag is the indicator of a full score, and it is a separate count rather than a replacement
    because it rests on an extra assumption about what the flag means. One of the five steps inside
    the unit interval falls below the supplied bound, so the two counts have to differ by exactly
    one or the bound is not being applied.
    """
    implied = np.asarray([0.0, 0.0625, 0.5, 1.0, 1.5, -0.25, 0.03])
    lower = np.asarray([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0])
    got = preflight.feasibility(implied, np.zeros(7), lower=lower, n_per_step=4, increments=4)
    assert got["n_inside_unit_interval"] == 5
    assert got["n_inside_with_lower_bound"] == 4


def test_closure_reports_both_sign_counts_when_the_prediction_undershoots_at_every_step():
    """The residual is `predicted - observed`, so a reconstruction that can only lose reward is
    non-positive throughout.

    Naming only one of the two sign counts invites the write-up to describe the wrong direction and
    sound just as confident, which is why both are stored. Replacing a quarter-valued score by its
    own ceiling indicator can only undercount, so the direction to expect is non-positive, and the
    construction here undershoots by a quarter at every step so the two counts are the extreme
    opposite of each other rather than merely unequal.
    """
    observed = np.asarray([1.0, 1.25, 1.5, 1.75])
    files = np.arange(4, dtype=np.float64)
    got = preflight.closure(observed - 0.25, files, observed, 1, "reward")

    assert got["target"] == "reward"
    assert got["n"] == 4
    assert got["absolute_rms"] == 0.25
    assert got["max_absolute"] == 0.25
    assert got["mean_residual"] == -0.25
    assert got["n_non_positive"] == 4
    assert got["n_non_negative"] == 0
    assert got["n_exactly_zero"] == 0
    assert got["quantisation"] == 0.25
    assert got["at_floor"] is False
    assert got["floor_ratio"] == pytest.approx(math.sqrt(12.0))
    assert len(got["residuals"]) == 4


def test_closure_on_an_exact_reconstruction_sits_at_the_floor_with_both_signs_full():
    """A residual of zero is non-negative and non-positive at every step, and it is at the floor.

    The two sign counts both reaching `n` is the signature of an exact closure rather than a
    contradiction, and it is worth pinning because a reader who saw only one of them would read the
    same number as a one-sided bias. `at_floor` compares the residual against half the grid the
    target was logged on, so an exact reconstruction has to clear it for any positive grid.
    """
    observed = np.asarray([1.0, 1.25, 1.5, 1.75])
    files = np.arange(4, dtype=np.float64)
    got = preflight.closure(observed.copy(), files, observed, 1, "reward")

    assert got["absolute_rms"] == 0.0
    assert got["n_exactly_zero"] == 4
    assert got["n_non_negative"] == 4
    assert got["n_non_positive"] == 4
    assert got["at_floor"] is True
    assert got["floor_ratio"] == 0.0


def test_snap_report_returns_the_rational_label_where_one_fits_and_None_where_none_does():
    """A configured weight is a round number, so a fit that does not snap is a fit and not a recovery.

    The stored value is the label rather than the number, which is worth pinning because the field
    is called `snapped` and a reader would reasonably expect a float. The three cases are a value a
    hair off an integer, a truncated third, and a value with no simple rational within tolerance;
    the last one has to come back as None rather than as the nearest half, or the snap stops meaning
    anything. The integer case also pins the preference for the coarsest representation, since 2.0
    is written as `2` rather than as `4/2`.
    """
    got = preflight.snap_report(
        {"integer": 4.0000001, "third": 0.3333, "nothing": 0.4712389, "two": 2.0}, 1e-3
    )
    assert got["integer"] == {"fitted": 4.0000001, "snapped": "4"}
    assert got["third"]["snapped"] == "1/3"
    assert got["nothing"]["snapped"] is None
    assert got["two"]["snapped"] == "2"


# ---------------------------------------------------------------------------
# P4. The advantage convention
# ---------------------------------------------------------------------------


def test_advantage_sensitivity_is_a_perfect_rank_match_when_the_groups_share_a_spread():
    """Standardising by a spread that is the same in every group is a single positive rescaling.

    Within any one group the rank correlation is 1 by construction, so the pooled number is the only
    one that carries information and this is the case where it has to come back at 1 as well. The
    two groups here have different shapes and identical standard deviations, which keeps every
    pooled value distinct and rules out the correlation being 1 merely because the ties happened to
    line up.
    """
    index = np.zeros(6)
    group = np.asarray([0, 0, 0, 1, 1, 1])
    reward = np.asarray([0.0, 1.0, 5.0, 0.0, 4.0, 5.0])
    got = preflight.advantage_sensitivity(index, group, reward)

    assert got["n_steps_scored"] == 1
    assert got["pooled_rank_correlation_median"] == 1.0
    assert got["within_group_rank_correlation_median"] == 1.0
    assert got["sign_disagreement_max"] == 0.0
    assert got["std_epsilon"] == 1e-4
    assert got["affine_violation_relative"] == 1e-4


def test_advantage_sensitivity_falls_below_one_when_the_groups_have_different_spreads():
    """Dividing each group by its own spread reweights the groups against each other.

    The two conventions are not related by a monotone transformation across groups, and this is the
    measurement that says so. The expected value is the Spearman coefficient of the two rank vectors
    the construction produces, written as the arithmetic that generates it rather than as a decimal,
    so a reader can check it without running anything: the centred ranks are (2, 4, 5, 2, 2, 6) and
    the standardised ranks are (1, 4, 5, 2.5, 2.5, 6). The within-group correlation stays at 1,
    which is what makes the pooled drop attributable to the reweighting and to nothing else.
    """
    index = np.zeros(6)
    group = np.asarray([0, 0, 0, 1, 1, 1])
    reward = np.asarray([0.0, 1.0, 2.0, 0.0, 0.0, 3.0])
    got = preflight.advantage_sensitivity(index, group, reward)

    assert got["pooled_rank_correlation_median"] == pytest.approx(15.5 / math.sqrt(15.5 * 17.0))
    assert got["pooled_rank_correlation_median"] < 1.0
    assert got["within_group_rank_correlation_median"] == 1.0


def test_advantage_sensitivity_ignores_a_group_too_small_to_carry_a_contrast():
    """A group below `min_group` contributes no advantage, so adding one changes nothing.

    A single rollout has no within-group contrast to standardise, and the implementation leaves its
    advantage as NaN rather than as a zero. Zero would be a real number flowing into a rank
    correlation, so the check is that the whole reading is unchanged when a singleton group is
    appended to the step, not merely that it does not crash.
    """
    index = np.zeros(7)
    group = np.asarray([0, 0, 0, 1, 1, 1, 2])
    reward = np.asarray([0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 99.0])
    got = preflight.advantage_sensitivity(index, group, reward)
    assert got["pooled_rank_correlation_median"] == pytest.approx(15.5 / math.sqrt(15.5 * 17.0))
    assert got["n_steps_scored"] == 1


# ---------------------------------------------------------------------------
# P6. Clones, effective n and the usable window
# ---------------------------------------------------------------------------


def test_content_hashes_are_stable_across_versions_and_separate_different_texts():
    """The digest is pinned to a literal, because a hash that moves silently re-partitions the run.

    Clone detection, the informative-group count and the clone adjustment all key on these digests,
    so a change to the algorithm or the digest size would change every one of those numbers without
    changing a line of the code that computes them. Comparing against a recomputed `blake2b` would
    only restate the implementation; a literal is the only form of this test that would notice.
    """
    assert preflight.content_hashes(["the optimiser paid for this"]) == [
        "e568ede9a7bc856aa8b16eed4302b39a"
    ]
    assert preflight.content_hashes(["the optimiser paid for that"]) == [
        "303798ad3156da26ba7e5f26e2169e50"
    ]
    assert len(preflight.content_hashes(["x"])[0]) == 32


def test_content_hashes_coerce_a_non_string_through_str():
    """A numeric label and its own text form hash identically, which is the coercion at work.

    Worth pinning because the caller fills nulls with the empty string and hands the column across
    as a list of objects, so whatever pandas hands back has to hash the same way as the text it
    represents. It also names the collision this coercion accepts, which is a real one rather than a
    theoretical one on a table whose response column is not guaranteed to be typed.
    """
    assert preflight.content_hashes([1]) == preflight.content_hashes(["1"])


def test_clone_structure_reports_the_duplicate_fraction_and_the_modal_response():
    """Eight rows over five distinct responses, one of which appears four times.

    The duplicate fraction is the share of rows that repeat something already seen, so three of
    eight here, and the modal multiplicity and share describe the single response that dominates.
    Splitting the fraction by step matters as much as pooling it, because a corpus that is half
    duplicates spread evenly is a different object from one where a single step collapsed, and the
    fixture makes the two files differ so the per-step median cannot be inferred from the pooled
    number.
    """
    index = np.repeat([0.0, 1.0], 4)
    texts = ["A", "A", "A", "B", "A", "C", "D", "E"]
    got = preflight.clone_structure(index, preflight.content_hashes(texts), texts)

    assert got["n_rows"] == 8
    assert got["n_unique"] == 5
    assert got["duplicate_fraction"] == 0.375
    assert got["modal_multiplicity"] == 4
    assert got["modal_share"] == 0.5
    assert got["modal_response"] == "A"
    assert got["multiplicity_top10"] == [4, 1, 1, 1, 1]
    assert got["per_step_duplicate_fraction_median"] == 0.25


def _collapsing_fixture():
    """Two files of three groups, the first of which holds two groups with the same responses.

    The first file is the case the real run does not contain and the argument for the two counts
    being equal does not cover: its two informative groups hold the same pair of responses in
    different orders, so their sorted signatures match and the Kish effective size collapses them
    into one. The second file is the ordinary case, where the responses differ and nothing collapses.
    """
    index = np.repeat([0.0, 1.0], 6)
    group = np.tile(np.asarray([0, 0, 1, 1, 2, 2]), 2)
    component = np.asarray([0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    texts = ["p", "q", "q", "p", "z", "z", "p", "q", "r", "s", "z", "z"]
    return index, group, {"a": component}, preflight.content_hashes(texts)


def test_informative_groups_excludes_a_constant_group_and_collapses_two_identical_ones():
    """A group with no component variation carries no advantage, and two clone groups are one.

    Both halves are constructed rather than observed. The third group of each step is constant in
    its only component, so the raw count is two of three, and the first step's two informative
    groups hold the same responses in a different order, so the clone adjustment collapses them to
    one while the second step's stay at two. That second property is the one the real artifact never
    exercises, which is exactly why it needs a test: a floor met by one group copied twice is not a
    floor met, and nothing else in the suite would notice if the adjustment stopped collapsing.
    """
    index, group, components, hashes = _collapsing_fixture()
    got = preflight.informative_groups(index, group, components, hashes)

    assert got["files"] == [0.0, 1.0]
    assert got["groups_per_step"] == [3, 3]
    assert got["n_informative"] == [2, 2]
    assert got["n_informative_clone_adjusted"] == [1.0, 2.0]
    assert got["max_possible"] == 3
    assert got["mean_informative"] == 2.0
    assert got["mean_informative_clone_adjusted"] == 1.5


def test_clone_adjustment_check_names_the_step_where_the_two_counts_disagree():
    """The claim that the raw and adjusted counts are always equal is false, and this is the case.

    The argument for the equality is that a group is informative exactly when its responses differ,
    so clone adjustment cannot collapse an informative group. That does not follow: two different
    informative groups collapse when their sorted response multisets are equal, which internal
    variation does nothing to prevent. An exact coincidence between two quantities computed by
    different code paths is either a theorem or a bug, and this construction settles which by
    exhibiting a step where the two differ by one.
    """
    index, group, components, hashes = _collapsing_fixture()
    got = preflight.clone_adjustment_check(index, group, components, hashes)

    assert got["n_steps"] == 2
    assert got["n_steps_where_they_differ"] == 1
    assert got["max_absolute_difference"] == 1.0
    assert got["first_disagreements"] == [{"step": 0.0, "raw": 2.0, "adjusted": 1.0}]
    assert got["exactly_equal"] is False


def test_clone_adjustment_check_reports_equality_when_no_two_groups_share_a_response_multiset():
    """With every informative group holding distinct responses the two counts do agree exactly.

    The negative case has to be pinned beside the positive one, because a check that reported a
    disagreement on every input would pass the test above while telling a reader nothing. Here the
    responses are all distinct, so the Kish effective size equals the raw count and the report says
    so without qualification.
    """
    index = np.zeros(6)
    group = np.asarray([0, 0, 1, 1, 2, 2])
    components = {"a": np.asarray([0, 1, 0, 1, 1, 1], dtype=float)}
    hashes = preflight.content_hashes(["p", "q", "r", "s", "z", "z"])
    got = preflight.clone_adjustment_check(index, group, components, hashes)

    assert got["n_steps_where_they_differ"] == 0
    assert got["max_absolute_difference"] == 0.0
    assert got["first_disagreements"] == []
    assert got["exactly_equal"] is True


def test_floor_sensitivity_reports_the_run_length_and_both_ends_of_the_window():
    """How many steps clear a floor, where the window starts and ends, and the longest unbroken run.

    The run length is the quantity a usable window is actually decided on, because a floor met on
    scattered steps does not give a contiguous span to measure anything over. Both endpoints are
    reported one-based, which is a convention rather than an arithmetic fact and is therefore worth
    pinning: the counts here clear a floor of four at positions one, two, four, five and six of the
    zero-based array, and the report has to call the last of those step seven. The unreachable floor
    returns None at both ends rather than a zero, so an unmet floor cannot be read as a window that
    starts at the beginning.
    """
    got = preflight.floor_sensitivity([3, 5, 5, 2, 5, 5, 5, 1], (1, 4, 6))

    assert got["4"] == {
        "n_steps_above": 5,
        "last_step_above": 7,
        "longest_consecutive_run": 3,
        "first_step_above": 2,
    }
    assert got["1"]["longest_consecutive_run"] == 8
    assert got["1"]["first_step_above"] == 1
    assert got["6"] == {
        "n_steps_above": 0,
        "last_step_above": None,
        "longest_consecutive_run": 0,
        "first_step_above": None,
    }


# ---------------------------------------------------------------------------
# P7. Optimiser telemetry
# ---------------------------------------------------------------------------


def test_optimiser_telemetry_puts_an_absent_key_in_absent_rather_than_a_zero_in_present():
    """A quantity the trainer did not log has to read as missing and never as measured.

    The key names in `log_history` are a property of the trainer version that wrote the file rather
    than of any specification, so a module that reached for a plausible neighbour would report a
    series that is quietly the wrong quantity. Splitting the return into `present` and `absent` is
    what keeps that visible, and the test asserts the absent key is nowhere in `present` as well as
    being in `absent`, since a zero-filled entry in both would satisfy a weaker check.
    """
    log = _log([1.0, 2.0, 3.0], grad_norm=[1.0, 2.0, 6.0])
    got = preflight.optimiser_telemetry(log, ("grad_norm", "not_logged"))

    assert got["present"]["grad_norm"] == {
        "n": 3,
        "mean": 3.0,
        "min": 1.0,
        "max": 6.0,
        "median": 2.0,
    }
    assert "not_logged" not in got["present"]
    assert got["absent"]["not_logged"]["reason"] == "RECORD_INCOMPLETE"
    assert "carries no key 'not_logged'" in got["absent"]["not_logged"]["detail"]
    assert got["absent"]["not_logged"]["remedy"]


# ---------------------------------------------------------------------------
# The W4.3 re-adjudication, and what the missing resolution can move
# ---------------------------------------------------------------------------


def test_lambda_under_refuses_when_the_window_holds_fewer_than_three_step_pairs():
    """Three samples make two pairs, and two pairs are not a series to fit anything against.

    A through-origin fit over two points will return a slope and an uncentred R-squared, and both
    numbers would be reported with the same confidence as one fitted over four hundred. The refusal
    is what stops that, so it has to name the count rather than merely decline, and the label has to
    travel with it because the caller runs five reconstructions and needs to know which one refused.
    """
    samples = [
        StepSample(
            index=i,
            names=("f",),
            features=np.asarray([[1.0], [2.0]]),
            advantages=np.asarray([-1.0, 1.0]),
            group_ids=np.asarray([0, 0]),
            task_ids=("t", "t"),
            advantage_source="reconstructed",
        )
        for i in range(3)
    ]
    got = preflight.lambda_under(samples, [0, 1, 2], [0.0, 0.5, 1.0], label="short", n_bootstrap=0)

    assert isinstance(got, Refusal)
    assert got.instrument == "e1_preflight.lambda_under"
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert got.detail == "short: 2 step pairs is not a series"


def _readjudication_fixture():
    """Five steps of two groups of two, with one response the featuriser cannot read.

    The two groups have deliberately different reward spreads, which is what makes the centred and
    the standardised conventions disagree: standardising divides each group by its own spread, so it
    reweights the groups against each other even though it leaves the ordering inside either alone.
    The empty response is there to check that the shared featurisation drops it and counts it.
    """
    n_steps, per_step = 5, 4
    index = np.repeat(np.arange(n_steps, dtype=np.float64), per_step)
    group = np.tile(np.asarray(["A", "A", "B", "B"], dtype=object), n_steps)
    reward = np.tile(np.asarray([0.0, 4.0, 0.0, 0.5]), n_steps)
    hack = np.tile(np.asarray([0.0, 1.0, 0.0, 0.0]), n_steps)
    texts: list[str] = []
    for t in range(n_steps):
        texts.extend(
            [
                "alpha beta " * (t + 1),
                "gamma " * (t + 2),
                "delta epsilon zeta " * (t + 1),
                "eta theta " * (t + 3),
            ]
        )
    texts[7] = ""
    return index, texts, reward, group, hack


def test_readjudicate_w4_3_varies_the_advantage_and_nothing_else():
    """Two variants naming the same reward and grouping differ only if the convention differs.

    The featurisation is done once and shared, so a variant that re-featurised could differ from its
    neighbour for a reason that has nothing to do with the three things being varied. Two ways to
    check that: a variant repeated under a second label has to come back bit for bit identical, and
    a variant that changes only the convention has to move `eta_eff`, which is the scale-carrying
    quantity and the one standardisation is expected to shift. The featurised count is asserted too,
    because the single unreadable response has to be dropped and counted rather than silently
    featurised as an empty string.

    The repeated variant is compared field by field rather than as a whole dictionary, because five
    steps are too few for the bootstrap interval or the logistic transition, those fields come back
    NaN, and NaN does not equal itself. Matching them as NaN is stronger than skipping them: it
    holds the two variants to agreeing about which quantities this window cannot produce.
    """
    index, texts, reward, group, hack = _readjudication_fixture()
    got = preflight.readjudicate_w4_3(
        index,
        texts,
        {"r": reward},
        {"g": group},
        hack,
        (
            ("centred", "r", "g", False),
            ("standardised", "r", "g", True),
            ("standardised_again", "r", "g", True),
        ),
        n_bootstrap=0,
    )

    assert got["n_rollouts"] == 20
    assert got["n_featurised"] == 19

    centred = got["variants"]["centred"]
    standardised = got["variants"]["standardised"]
    repeated = got["variants"]["standardised_again"]

    assert centred["n_steps"] == 4
    assert centred["eta_eff"] != standardised["eta_eff"]
    assert set(repeated) == set(standardised)
    for key in set(standardised) - {"label"}:
        left, right = standardised[key], repeated[key]
        if isinstance(left, float) and math.isnan(left):
            assert isinstance(right, float) and math.isnan(right), key
        else:
            assert left == right, key


def test_resolution_sensitivity_counts_the_groups_the_indicator_calls_degenerate_at_zero():
    """Groups where the published flag is constant at zero are the ones the missing quarters can move.

    Everywhere else the two resolutions decide degeneracy identically, so this count is the whole
    bound. The four groups here are built so that each of the three ways a group can sit are present:
    the flag constant at zero, the flag constant at one, and the flag varying. Two of the four are
    constant at zero and both have a constant pass column, so both enter the at-risk count, which is
    an upper bound rather than an exact count: a group whose unobserved quarters happen to be
    constant is counted here and is not in fact at risk.
    """
    index = np.zeros(8)
    group = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    score = np.asarray([0.25, 0.75, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0])
    fmt_indicator = (score >= 1.0).astype(float)
    passed = np.ones(8)
    got = preflight.resolution_sensitivity(
        index, group, fmt_indicator, passed, preflight.content_hashes(["x"] * 8)
    )

    assert got["n_groups"] == 4
    assert got["n_steps"] == 1
    assert got["n_groups_both_components_constant"] == 3
    assert got["n_groups_format_indicator_constant_at_zero"] == 2
    assert got["n_groups_degeneracy_could_be_wrong"] == 2
    assert got["frac_groups_degeneracy_could_be_wrong"] == 0.5
    assert got["direction"].startswith("one-sided")


def test_resolution_sensitivity_bounds_the_error_in_the_direction_it_claims():
    """The indicator can call a group degenerate that was not, and never the other way round.

    The claim is asserted against a construction rather than read off the string, which is the only
    way to test it. The true quarter-valued score is built first and the published flag is derived
    from it, so both verdicts can be computed: no group whose true score is constant has a varying
    flag, which is the direction the claim rules out, and one group has a constant flag over a score
    that varies, which is the direction it allows. The reported count has to cover that group.
    """
    index = np.zeros(8)
    group = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    score = np.asarray([0.25, 0.75, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0])
    fmt_indicator = (score >= 1.0).astype(float)
    passed = np.ones(8)

    reverse_errors = 0
    forward_errors = 0
    for label in np.unique(group):
        sel = group == label
        score_constant = len(set(score[sel].tolist())) <= 1
        flag_constant = len(set(fmt_indicator[sel].tolist())) <= 1
        if score_constant and not flag_constant:
            reverse_errors += 1
        if flag_constant and not score_constant:
            forward_errors += 1

    assert reverse_errors == 0
    assert forward_errors == 1

    got = preflight.resolution_sensitivity(
        index, group, fmt_indicator, passed, preflight.content_hashes(["x"] * 8)
    )
    assert got["n_groups_degeneracy_could_be_wrong"] >= forward_errors
