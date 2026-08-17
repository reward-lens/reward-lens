"""The four determinations an imported run needs, on tables whose answers are known by construction.

Every fixture here is small enough to check by eye, and that is deliberate. The defects these types
exist to prevent were all invisible on the real artifact: a reward column that was one term of two,
a group key that merged two prompt groups on three files out of four hundred and one, a convention
that reordered almost nothing, and an alignment off by one step in four hundred. None of those
announces itself in a number. Each of them is obvious on six rows.

The one test that is not synthetic is the last: it asserts that the shipped AISI preset refuses to
hand out an operational reward, because the whole point of the rewrite is that a caller cannot get
`training_passed` back under that name, and a preset that quietly could would make every other test
here decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.ledger.imported import (
    AdvantageConvention,
    AlignmentStrength,
    AxisAlignment,
    ComponentResolution,
    ComposedReward,
    ConventionSource,
    DeclaredConvention,
    DeclaredFloor,
    ImportedRun,
    ProxyReward,
    RewardComponent,
    RolloutColumns,
    StepAxis,
    check_floors,
    identity_alignment,
    index_gaps,
    join_on_axis,
    recover_prompt_groups,
    semantic_collisions,
)

# ---------------------------------------------------------------------------
# Fixtures: two steps, two prompt groups of two, and one problem drawn twice
# ---------------------------------------------------------------------------


def _table() -> dict[str, list]:
    """Two eval files, four rollouts each, two prompt groups of two per file.

    File 1 draws problem `p` twice, which is the collision the semantic column cannot express and
    the row order can. Keyed on the problem, file 1 has one group of four; keyed on the row order it
    has two groups of two, which is what the trainer normalised over.
    """
    return {
        "rollout_index": [0, 0, 0, 0, 1, 1, 1, 1],
        "source_eval_file": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "problem_id": ["p", "p", "q", "q", "p", "p", "p", "p"],
        "fmt": [1, 0, 1, 1, 0, 0, 1, 1],
        "passed": [1, 1, 0, 0, 1, 0, 1, 1],
        "hacked": [0, 0, 1, 1, 1, 1, 0, 0],
        "response": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    }


def _composition(fmt_resolution: ComponentResolution) -> ComposedReward:
    return ComposedReward(
        components=(
            RewardComponent(
                name="fmt",
                weight=1.0,
                column="fmt",
                resolution=fmt_resolution,
                source="the fixture declares it",
                scorer_grid=0.25,
                published_grid=1.0,
                bounds=(0.0, 1.0),
            ),
            RewardComponent(
                name="passed",
                weight=4.0,
                column="passed",
                resolution=ComponentResolution.EXACT,
                source="the fixture declares it",
                bounds=(0.0, 1.0),
            ),
        ),
        source="the fixture declares it",
    )


CENTRED = DeclaredConvention(
    convention=AdvantageConvention.CENTRED,
    source=ConventionSource.CONFIGURATION_FAMILY,
    detail="the fixture's configuration says so",
)


def _run(fmt_resolution: ComponentResolution = ComponentResolution.EXACT) -> ImportedRun:
    return ImportedRun(
        name="fixture",
        columns=RolloutColumns(
            step="rollout_index",
            file="source_eval_file",
            label="hacked",
            text="response",
            semantic_id="problem_id",
        ),
        reward=_composition(fmt_resolution),
        convention=CENTRED,
        generations=2,
        axis=StepAxis.ROLLOUT_FILE_INDEX,
    )


# ---------------------------------------------------------------------------
# 1. The composed reward
# ---------------------------------------------------------------------------


def test_a_binarised_component_makes_the_operational_reward_refuse():
    """The refusal the whole rewrite exists for, and it names the column and the resolution."""
    got = _composition(ComponentResolution.BINARISED).operational(_table(), instrument="t")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "fmt" in got.detail and "binarised" in got.detail
    assert got.statistics["scorer_grid"] == {"fmt": 0.25}
    assert got.statistics["published_grid"] == {"fmt": 1.0}
    # The remedy is one column wide and says so, because a remedy a publisher cannot act on is a
    # sentence rather than a remedy.
    assert "resolution its own scorer computes it" in got.remedy


def test_an_exactly_published_composition_returns_the_reward_it_composes():
    got = _composition(ComponentResolution.EXACT).operational(_table(), instrument="t")
    assert not isinstance(got, Refusal)
    # 1*fmt + 4*passed, by hand: rows are (1,1) (0,1) (1,0) (1,0) (0,1) (0,0) (1,1) (1,1)
    assert got.tolist() == [5.0, 4.0, 1.0, 1.0, 4.0, 0.0, 5.0, 5.0]


def test_a_proxy_is_returned_as_an_object_that_says_it_is_not_the_reward():
    got = _composition(ComponentResolution.BINARISED).proxy(_table(), instrument="t")
    assert isinstance(got, ProxyReward)
    assert got.missing == ("fmt",)
    assert not got.is_exact
    assert "not the operational reward" in got.detail
    # An indicator of a full score can only undercount, and the type knows the direction.
    assert "only undercount" in got.detail


def test_a_component_with_no_column_at_all_cannot_even_be_proxied():
    composition = ComposedReward(
        components=(
            RewardComponent(
                name="secret",
                weight=1.0,
                column=None,
                resolution=ComponentResolution.ABSENT,
                source="the fixture declares it",
            ),
            RewardComponent(
                name="passed",
                weight=4.0,
                column="passed",
                resolution=ComponentResolution.EXACT,
                source="the fixture declares it",
            ),
        )
    )
    got = composition.proxy(_table(), instrument="t")
    assert isinstance(got, Refusal)
    assert "secret" in got.detail


def test_the_ceiling_is_the_arithmetic_that_refutes_a_weight_vector_without_a_fit():
    """Every bounded component contributes weight times its upper bound, so the sum is the ceiling."""
    assert _composition(ComponentResolution.EXACT).ceiling() == pytest.approx(5.0)
    unbounded = ComposedReward(
        components=(
            RewardComponent(
                name="x",
                weight=1.0,
                column="fmt",
                resolution=ComponentResolution.EXACT,
                source="s",
            ),
        )
    )
    assert unbounded.ceiling() is None


def test_a_component_weight_with_no_source_is_refused_at_construction():
    with pytest.raises(ValueError, match="no source"):
        RewardComponent(
            name="x",
            weight=1.0,
            column="fmt",
            resolution=ComponentResolution.EXACT,
            source="  ",
        )


# ---------------------------------------------------------------------------
# 2. Optimiser groups, and the semantic column that is not one
# ---------------------------------------------------------------------------


def test_row_order_recovers_the_groups_the_trainer_normalised_over():
    got = recover_prompt_groups(_table()["rollout_index"], generations=2)
    assert not isinstance(got, Refusal)
    assert got.group_id.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert got.generation_index.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert got.irregular_blocks == ()
    assert int(np.unique(got.keys()).size) == 4


def test_a_block_of_the_wrong_length_refuses_rather_than_producing_a_group():
    """The size invariant. A short block displaces every boundary after it, silently."""
    # Step 0 holds three rows, so its second block is one row long. Step 1's block is complete,
    # which is the point: the invariant fires on the block and not on the step.
    got = recover_prompt_groups([0, 0, 0, 1, 1], generations=2)
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert got.statistics["n_irregular"] == 1
    assert got.statistics["irregular"] == [[0.0, 1, 1]]
    assert "not 2 rows long" in got.detail


def test_the_invariant_can_be_relaxed_to_report_a_broken_batch_rather_than_refuse_it():
    got = recover_prompt_groups([0, 0, 0, 1, 1], generations=2, strict=False)
    assert not isinstance(got, Refusal)
    assert got.irregular_blocks == ((0.0, 1, 1),)


def test_a_repeated_problem_inside_a_step_is_reported_and_never_merged():
    groups = recover_prompt_groups(_table()["rollout_index"], generations=2)
    got = semantic_collisions(_table()["rollout_index"], _table()["problem_id"], groups)
    assert got["n_collisions"] == 1
    assert got["steps_affected"] == [1.0]
    assert got["collisions"][0]["semantic_id"] == "p"
    assert got["collisions"][0]["groups"] == [0, 1]
    # The two groups stay two. Merging them would build a group of four the trainer never saw.
    assert int(np.unique(groups.keys()).size) == 4


# ---------------------------------------------------------------------------
# 3. The advantage convention
# ---------------------------------------------------------------------------


def test_an_undetermined_convention_refuses_every_scale_dependent_quantity():
    from reward_lens.measure.ledger.imported import UNDETERMINED_CONVENTION

    got = UNDETERMINED_CONVENTION.requires(instrument="t")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "undetermined" in got.detail


def test_standardising_without_a_declared_ddof_refuses_and_says_what_it_costs():
    convention = DeclaredConvention(
        convention=AdvantageConvention.STD_NORMALISED,
        source=ConventionSource.ARTIFACT,
        detail="the artifact says so",
        std_epsilon=1e-4,
    )
    got = convention.requires(instrument="t")
    assert isinstance(got, Refusal)
    assert "3.3 per cent" in got.detail


def test_a_fully_declared_standardising_convention_passes():
    convention = DeclaredConvention(
        convention=AdvantageConvention.STD_NORMALISED,
        source=ConventionSource.ARTIFACT,
        detail="the artifact says so",
        std_epsilon=1e-4,
        std_ddof=1,
    )
    assert convention.requires(instrument="t") is None


def test_the_centred_convention_needs_no_epsilon_and_no_ddof():
    assert CENTRED.requires(instrument="t") is None


def test_amplification_is_undefined_rather_than_unavailable_on_a_centred_estimator():
    """The distinction E48 exists for: no access and no rewriting of the record gives it one."""
    got = CENTRED.amplification(instrument="t")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.QUANTITY_UNDEFINED
    assert got.statistics["instead"]


def test_an_unknown_convention_may_not_claim_a_source():
    with pytest.raises(ValueError, match="UNDETERMINED"):
        DeclaredConvention(
            convention=AdvantageConvention.UNKNOWN,
            source=ConventionSource.ARTIFACT,
            detail="contradictory",
        )


# ---------------------------------------------------------------------------
# 4. Step axes and the alignment between them
# ---------------------------------------------------------------------------


def _alignment(offset: int, strength: AlignmentStrength = AlignmentStrength.IDENTITY):
    return AxisAlignment(
        source=StepAxis.ROLLOUT_FILE_INDEX,
        target=StepAxis.TRAINER_LOG_STEP,
        offset=offset,
        strength=strength,
        evidence="the fixture declares it",
    )


def test_an_implicit_join_refuses_rather_than_pairing_by_position():
    got = join_on_axis([1.0, 2.0], [0, 1], [1.0, 2.0], [1, 2], alignment=None, instrument="t")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.UNIT_MISMATCH


def test_an_assumed_alignment_refuses_unless_the_caller_says_so_out_loud():
    args = ([1.0, 2.0], [0, 1], [1.0, 2.0], [1, 2])
    assumed = _alignment(1, AlignmentStrength.ASSUMED)
    assert isinstance(join_on_axis(*args, alignment=assumed, instrument="t"), Refusal)
    got = join_on_axis(*args, alignment=assumed, instrument="t", allow_assumed=True)
    assert not isinstance(got, Refusal)


def test_the_join_pairs_by_step_value_and_not_by_array_position():
    """The defect this replaces. A gap in the target shifts every pair under positional indexing.

    Source files 0, 1, 2 are steps 1, 2, 3. The target carries steps 1 and 3 and not 2, which is a
    trainer log with an entry missing. Pairing by position would put file 1 against step 3; pairing
    by value drops file 1 and says so.
    """
    got = join_on_axis(
        [10.0, 20.0, 30.0], [0, 1, 2], [1.0, 3.0], [1, 3], alignment=_alignment(1), instrument="t"
    )
    assert not isinstance(got, Refusal)
    assert got.source_values.tolist() == [10.0, 30.0]
    assert got.target_values.tolist() == [1.0, 3.0]
    assert got.target_steps.tolist() == [1.0, 3.0]
    assert got.n_source_unmatched == 1
    assert got.n == 2


def test_an_offset_of_one_is_what_separates_the_right_pairing_from_the_wrong_one():
    source, index = [10.0, 20.0, 30.0], [0, 1, 2]
    target, steps = [1.0, 2.0, 3.0], [1, 2, 3]
    right = join_on_axis(source, index, target, steps, alignment=_alignment(1), instrument="t")
    wrong = join_on_axis(source, index, target, steps, alignment=_alignment(0), instrument="t")
    assert right.source_values.tolist() == [10.0, 20.0, 30.0]
    assert right.target_values.tolist() == [1.0, 2.0, 3.0]
    assert wrong.source_values.tolist() == [20.0, 30.0]
    assert wrong.target_values.tolist() == [1.0, 2.0]


def test_an_alignment_with_no_evidence_is_refused_at_construction():
    with pytest.raises(ValueError, match="no evidence"):
        AxisAlignment(
            source=StepAxis.ROLLOUT_FILE_INDEX,
            target=StepAxis.TRAINER_LOG_STEP,
            offset=1,
            strength=AlignmentStrength.FITTED,
            evidence="",
        )


def test_an_axis_cannot_be_displaced_from_itself():
    with pytest.raises(ValueError, match="offset"):
        AxisAlignment(
            source=StepAxis.TRAINER_LOG_STEP,
            target=StepAxis.TRAINER_LOG_STEP,
            offset=1,
            strength=AlignmentStrength.RECORDED,
            evidence="e",
        )
    assert identity_alignment(StepAxis.TRAINER_LOG_STEP).offset == 0


def test_the_inverse_alignment_carries_the_same_evidence_at_the_same_strength():
    back = _alignment(1).inverse()
    assert back.offset == -1
    assert back.source is StepAxis.TRAINER_LOG_STEP
    assert back.strength is AlignmentStrength.IDENTITY


def test_a_missing_index_is_reported_by_value_and_not_only_as_a_count():
    """The companion run's defect in miniature: 403 distinct indices over a span of 404."""
    got = index_gaps([0, 1, 2, 4, 5])
    assert got["n_distinct"] == 5
    assert got["span"] == 6
    assert got["contiguous"] is False
    assert got["missing"] == [3]
    assert index_gaps([0, 1, 2])["contiguous"] is True


# ---------------------------------------------------------------------------
# 5. Declared floors, checked before the data is seen
# ---------------------------------------------------------------------------


def test_a_floor_above_what_the_artifact_can_produce_refuses_at_freeze_time():
    floor = DeclaredFloor(
        name="informative_groups",
        value=8.0,
        unit="groups per step",
        rule="eight independent groups per step, so a rank correlation has power",
    )
    got = floor.check(4.0, instrument="t")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.PLAN_NOT_CLOSED
    assert "unreachable" in got.detail
    # And the remedy forbids the repair that looks easiest.
    assert "Do not lower the floor to the observed value" in got.remedy


def test_a_reachable_floor_passes_and_says_by_how_much():
    floor = DeclaredFloor(name="f", value=4.0, unit="groups", rule="the batch holds four")
    assert floor.check(4.0, instrument="t") is None
    got = floor.reachability(4.0)
    assert got.reachable and got.maximum_attainable == 4.0


def test_a_floor_with_no_rule_cannot_be_declared():
    with pytest.raises(ValueError, match="no rule"):
        DeclaredFloor(name="f", value=1.0, unit="u", rule="")


def test_every_unreachable_floor_is_reported_rather_than_only_the_first():
    floors = [
        DeclaredFloor(name="a", value=8.0, unit="u", rule="r"),
        DeclaredFloor(name="b", value=20.0, unit="u", rule="r"),
        DeclaredFloor(name="c", value=1.0, unit="u", rule="r"),
    ]
    got = check_floors(floors, {"a": 4.0, "b": 10.0, "c": 4.0})
    assert len(got) == 2
    assert {r.statistics["name"] for r in got} == {"a", "b"}


def test_a_floor_with_no_supplied_maximum_is_an_unrun_check_rather_than_a_pass():
    got = check_floors([DeclaredFloor(name="a", value=1.0, unit="u", rule="r")], {})
    assert len(got) == 1
    assert "no maximum attainable value was supplied" in got[0].detail


@pytest.mark.parametrize("maximum", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_maximum_is_an_unestablished_check_rather_than_a_pass(maximum):
    """SPEC-ERRATA E67. This returned `reachable=True` and refused nothing.

    The failure is the quiet kind: the caller asks whether a declared threshold is a test at all,
    and gets back "yes" from an artifact whose maximum was never computed. A floor whose
    reachability is unknown has not been checked, and reporting it as reachable is how an
    unreachable floor reaches a freeze.
    """
    floor = DeclaredFloor(name="informative_groups", value=8.0, unit="groups per step", rule="r")
    got = floor.reachability(maximum)
    assert got.established is False
    assert got.reachable is False
    assert "never established" in got.detail

    refusal = floor.check(maximum, instrument="t")
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.RECORD_INCOMPLETE
    assert refusal.statistics["established"] is False


def test_a_finite_maximum_still_reports_established_so_the_two_states_stay_separable():
    floor = DeclaredFloor(name="f", value=4.0, unit="groups", rule="r")
    assert floor.reachability(4.0).established is True
    assert floor.reachability(1.0).established is True
    # Unreachable-and-established keeps its own refusal reason, distinct from unestablished.
    unreachable = floor.check(1.0, instrument="t")
    assert isinstance(unreachable, Refusal)
    assert unreachable.reason is RefusalReason.PLAN_NOT_CLOSED


# ---------------------------------------------------------------------------
# 6. The shipped preset, which is the one thing here that is not synthetic
# ---------------------------------------------------------------------------


def test_the_aisi_preset_refuses_to_hand_out_an_operational_reward():
    """No caller can obtain `training_passed` under the name of the reward on this artifact."""
    from reward_lens.measure.ledger.labelled import AISI_RUN

    got = AISI_RUN.operational_reward(
        {"thinking_format_ok": [1, 0], "training_passed": [1, 1], "response": ["a", "b"]}
    )
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert got.statistics["missing"] == ["thinking_format"]


def test_the_aisi_preset_carries_the_four_determinations_with_their_sources():
    from reward_lens.measure.ledger.labelled import AISI_RUN

    assert AISI_RUN.convention.convention is AdvantageConvention.CENTRED
    # A configuration-family determination, not an artifact one: no released field records it.
    assert AISI_RUN.convention.source is ConventionSource.CONFIGURATION_FAMILY
    assert AISI_RUN.generations == 16
    assert AISI_RUN.axis is StepAxis.ROLLOUT_FILE_INDEX
    assert AISI_RUN.alignment is not None
    assert AISI_RUN.alignment.offset == 1
    assert AISI_RUN.alignment.strength is AlignmentStrength.IDENTITY
    assert AISI_RUN.columns.semantic_id == "problem_id"
    # The ceiling every published configuration implies, which the logged mean exceeds.
    assert AISI_RUN.reward.ceiling() == pytest.approx(5.0)


def test_the_aisi_preset_has_no_column_named_the_reward():
    """`RolloutColumns` has no reward field, so the old mistake has nowhere to live."""
    from reward_lens.measure.ledger.labelled import AISI_RUN

    assert "reward" not in AISI_RUN.columns.as_json()
    assert "training_passed" not in AISI_RUN.columns.as_json().values()


# ---------------------------------------------------------------------------
# 7. The adapter, which is where the four determinations become a reading
# ---------------------------------------------------------------------------


def test_the_adapter_refuses_the_operational_reward_it_cannot_supply():
    from reward_lens.measure.ledger.labelled import steps_from_table

    got = steps_from_table(_table(), _run(ComponentResolution.BINARISED), reward="operational")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE


def test_the_adapter_builds_on_a_proxy_only_when_the_caller_names_it():
    from reward_lens.measure.ledger.labelled import steps_from_table

    got = steps_from_table(_table(), _run(ComponentResolution.BINARISED), reward="proxy")
    assert not isinstance(got, Refusal)
    assert len(got) == 2
    assert all("not the operational reward" in s.detail for s in got)


def test_there_is_no_third_option_and_no_default_for_which_reward_to_use():
    from reward_lens.measure.ledger.labelled import steps_from_table

    with pytest.raises(ValueError, match="no default"):
        steps_from_table(_table(), _run(), reward="whichever")


def test_the_detail_string_states_the_arithmetic_that_actually_ran():
    """The provenance string used to name a divisor the centred path never computes."""
    from reward_lens.measure.ledger.labelled import steps_from_table

    centred = steps_from_table(_table(), _run(), reward="operational")
    assert all("r - mean_g" in s.detail for s in centred)
    assert all("std_g" not in s.detail for s in centred)

    standardising = ImportedRun(
        name="fixture",
        columns=_run().columns,
        reward=_composition(ComponentResolution.EXACT),
        convention=DeclaredConvention(
            convention=AdvantageConvention.STD_NORMALISED,
            source=ConventionSource.ARTIFACT,
            detail="the fixture says so",
            std_epsilon=1e-4,
            std_ddof=1,
        ),
        generations=2,
        axis=StepAxis.ROLLOUT_FILE_INDEX,
    )
    got = steps_from_table(_table(), standardising, reward="operational")
    assert all("std_g[ddof=1] + 0.0001" in s.detail for s in got)


def test_the_adapter_groups_on_the_row_order_and_never_on_the_repeated_problem():
    """File 1 draws problem `p` twice. Keyed on the problem it would be one group of four."""
    from reward_lens.measure.ledger.labelled import steps_from_table

    got = steps_from_table(_table(), _run(), reward="operational")
    second = got[1]
    assert second.group_ids.tolist() == [0, 0, 1, 1]
    # The semantic problem is kept, under the name of the thing it is.
    assert second.task_ids == ("p", "p", "p", "p")


def test_the_adapter_refuses_when_the_convention_is_undetermined():
    from reward_lens.measure.ledger.imported import UNDETERMINED_CONVENTION
    from reward_lens.measure.ledger.labelled import steps_from_table

    run = ImportedRun(
        name="fixture",
        columns=_run().columns,
        reward=_composition(ComponentResolution.EXACT),
        convention=UNDETERMINED_CONVENTION,
        generations=2,
        axis=StepAxis.ROLLOUT_FILE_INDEX,
    )
    got = steps_from_table(_table(), run, reward="operational")
    assert isinstance(got, Refusal)
    assert "convention is undetermined" in got.detail
