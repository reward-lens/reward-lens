"""BLK-021: the tap's group statistics, at the denominator the same tap already records.

One record carried two answers to one question. `experiments/chain_c1/recorder/groupstats.py`
emits the contract fields `group_std` and `group_all_fail` from the group's own component scores
with the stdlib `statistics`; `Group.group_stats`, written by this adapter, sat beside them
computed a different way, because the construction site called

    GroupStats.from_scores(totals, std_epsilon=std_eps)

and passed neither of the two keyword arguments `from_scores` has always offered. So:

* ``std`` was the population form while ``EstimatorSpec.std_ddof`` **on the same record** said 1.
  The two differ by ``sqrt(K / (K - 1))``, which is 1.0690449676496976 at the ``K = 8`` this
  design runs. C1's tolerance on the advantage reconstruction is 20 percent, so a 6.9 percent
  convention error sits inside the band and C1 passes; C3 asserts a contrast below 1e-10 at equal
  grader output and a scale error leaves a zero a zero, so C3 passes too. Both instrument checks
  return their known answers while the quantity underneath is wrong by a factor nobody printed.
* ``all_fail`` was False on every group of every TRL-tapped record, which is exactly what a run in
  which no group ever failed also looks like. A value that is always the same is not a
  measurement.

**Why the repair is two arguments and not a second computation.** The failure this row names is
two producers of one number, so a repair that adds a second computation here and keeps it in step
with the recorder's is the same defect with a longer fuse. There is one producer of the
denominator in this adapter: ``_estimator_spec`` decides it once, off ``scale_rewards``, and it is
already on the ``EstimatorSpec`` this method attaches to the same ``Group``. The construction site
reads that object. ``test_there_is_one_producer_of_the_denominator_in_this_adapter`` is what keeps
a second literal from appearing beside it.

``failure_at`` has no such producer, because nothing in ``GRPOConfig`` says what failure is on a
task. It is declared by the caller, the way ``VerifiersAdapter.failure_at`` already is, and it
defaults to None: a tap that guessed the value would turn a gap into a claim, and ``from_scores``
reads None as "all-fail cannot be determined" rather than as "no group failed".

**What this file does not do**, because the row's `non_goal` refuses it: it does not make
``recorder/groupstats.py`` delegate to ``GroupStats.from_scores``, it does not touch the verifiers
adapter, which already passes ``failure_at``, it does not make ``std_ddof`` a default anywhere,
and it reconciles no record written before the field existed. The agreement asserted below is
between the tapped value and the recorder's **definition**, recomputed here with the recorder's
own function (``statistics.stdev``, ddof 1, `FAIL_AT = 0.0`), which is a second implementation
and therefore a real cross-check rather than a tautology. The library imports nothing from the
experiment tree and this test does not either.
"""

from __future__ import annotations

import inspect
import statistics

import numpy as np
import pytest

from reward_lens.record.schema import GroupStats
from reward_lens.tap.adapters.trl import TRLTap
from reward_lens.tap.contract import TapBudget

GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=16 * 1024 * 1024,
    max_added_alloc_bytes_per_step=16 * 1024 * 1024,
)

#: The design's group size. The whole row is about a factor that is 6.9% here and 41.4% at K = 2,
#: so the fixture runs at the K the design runs at rather than at the one that flatters it.
K = 8

#: Eight distinct scores, one prompt, one group. Distinct so the standard deviation is far from
#: both the epsilon and zero, and so the two denominators give visibly different answers.
SCORES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

#: Every rollout at the failure value. `recorder/groupstats.py` fixes that value at exact zero
#: and says why: a group in which every rollout scored 1e-9 is not a group that all failed.
ALL_FAIL_SCORES = (0.0,) * K

#: `recorder/groupstats.py`'s FAIL_AT, restated rather than imported: the library does not import
#: the experiment tree, and CB-1010's invariant is that no experiment file lives in it.
RECORDER_FAIL_AT = 0.0

#: sqrt(K / (K - 1)) at K = 8, to sixteen digits. The factor the two conventions differ by.
BESSEL_AT_K8 = 1.0690449676496976

PROMPT = "one prompt, eight rollouts"
COMPLETIONS = tuple(f"c{i}" for i in range(K))
COMPLETION_IDS = tuple(tuple(range(100 * i, 100 * i + 3)) for i in range(K))


class FakeArgs:
    """``GRPOConfig`` cut to what ``_read_config`` reads, as in ``tests/test_tap_trl.py``."""

    beta = 0.0
    temperature = 0.9
    top_p = 1.0
    epsilon = 0.2
    epsilon_high = None
    delta = None
    seed = 11
    loss_type = "grpo"
    scale_rewards = "group"
    multi_objective_aggregation = "sum_then_normalize"
    importance_sampling_level = "token"
    mask_truncated_completions = False
    num_iterations = 1
    steps_per_generation = 1
    generation_batch_size = K
    max_completion_length = 512
    num_generations = K
    reward_weights = None
    log_completions = True
    use_vllm = False
    optim = "adamw_torch"
    per_device_train_batch_size = K


class FakeState:
    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step


class FakeTrainer:
    def __init__(self, scale_rewards: str = "group") -> None:
        self.args = FakeArgs()
        self.args.scale_rewards = scale_rewards
        self.num_generations = K
        self.model = None
        self.tools = None
        self.callbacks: list = []
        self._logs = {"prompt": [], "completion": [], "advantages": [], "rewards": {}}
        self._metrics = {"train": {}}

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)


def drive(tap: TRLTap, *, scores=SCORES, scale_rewards: str = "group"):
    """One step of one group of K, through the adapter's own seams."""
    trainer = FakeTrainer(scale_rewards=scale_rewards)

    def grader(prompts, completions, completion_ids, **kwargs):
        by_completion = dict(zip(COMPLETIONS, scores))
        return [by_completion[c] for c in completions]

    wrapped = tap.wrap(grader)
    tap.attach(trainer)
    callback = trainer.callbacks[0]
    prompts = [PROMPT] * K
    wrapped(
        prompts=prompts,
        completions=list(COMPLETIONS),
        completion_ids=[list(x) for x in COMPLETION_IDS],
        trainer_state=FakeState(0),
        log_metric=lambda name, value: None,
        log_extra=lambda column, values: None,
    )
    trainer._logs["prompt"] = prompts
    trainer._logs["completion"] = list(COMPLETIONS)
    trainer._logs["advantages"] = [0.0] * K
    callback.on_step_end(trainer.args, FakeState(1), None)
    callback.on_train_end(trainer.args, FakeState(1), None)
    return tap.finish()


def one_group(run):
    """The single group this fixture produces, with the assertion that it is single."""
    groups = [g for s in run.steps for g in s.groups]
    assert len(groups) == 1, f"the fixture is one group of {K}; got {len(groups)}"
    assert len(groups[0].trajectories) == K
    return groups[0]


# ---------------------------------------------------------------------------
# the fixture's own controls: without these every assertion below is vacuous
# ---------------------------------------------------------------------------


def test_the_fixture_separates_the_two_denominators():
    population = float(np.std(np.asarray(SCORES), ddof=0))
    sample = float(np.std(np.asarray(SCORES), ddof=1))
    assert population == pytest.approx(2.29128784747792, abs=1e-12)
    assert sample == pytest.approx(2.449489742783178, abs=1e-12)
    assert sample / population == pytest.approx(BESSEL_AT_K8, abs=1e-15)
    assert sample / population == pytest.approx(float(np.sqrt(K / (K - 1))), abs=1e-15)


def test_the_old_call_shape_is_the_defect_this_row_names():
    """`must_fail_on`, reproduced in process: the call as it stood, on this fixture.

    Not a paraphrase of it. This is the exact expression that was at the construction site, and
    it is kept here so the repair below cannot be read as an improvement to a call that was fine.
    """
    was = GroupStats.from_scores(SCORES, std_epsilon=1e-4)
    assert was.all_fail is False
    assert was.std_ddof == 0
    assert was.std == pytest.approx(float(np.std(np.asarray(SCORES), ddof=0)), abs=1e-12)

    all_fail_was = GroupStats.from_scores(ALL_FAIL_SCORES, std_epsilon=1e-4)
    assert all_fail_was.all_fail is False, "the flag was False even on a group that all failed"

    now = GroupStats.from_scores(SCORES, std_epsilon=1e-4, std_ddof=1, failure_at=0.0)
    assert now.std / was.std == pytest.approx(BESSEL_AT_K8, abs=1e-12)


# ---------------------------------------------------------------------------
# all_fail: a measurement, in both directions
# ---------------------------------------------------------------------------


def test_a_tapped_group_that_all_failed_carries_all_fail_true():
    run = drive(
        TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT),
        scores=ALL_FAIL_SCORES,
    )
    group = one_group(run)
    assert group.group_stats.k == K
    assert group.group_stats.all_fail is True


def test_a_tapped_group_with_one_survivor_is_not_all_fail():
    """The control. Without it the flag could be a constant True and every assertion still pass."""
    survivor = (0.0,) * (K - 1) + (1.0,)
    run = drive(
        TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT), scores=survivor
    )
    assert one_group(run).group_stats.all_fail is False


def test_all_fail_stays_false_when_no_failure_value_was_declared():
    """None is an abstention and it reads as one.

    Nothing in `GRPOConfig` says what failure is on a task, so the tap cannot derive this value
    and does not invent it. A caller who declares nothing gets the flag `from_scores` documents
    for that case, and the record is no worse than it was; a caller who declares 0.0, as
    `recorder/groupstats.py` does, gets a measurement.
    """
    run = drive(TRLTap(run_id="blk021", budget=GENEROUS), scores=ALL_FAIL_SCORES)
    assert one_group(run).group_stats.all_fail is False


# ---------------------------------------------------------------------------
# std: the denominator the same record declares
# ---------------------------------------------------------------------------


def test_the_tapped_std_is_taken_at_the_denominator_the_same_record_declares():
    run = drive(TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT))
    group = one_group(run)
    spec = group.estimator

    assert spec.std_normalised is True
    assert spec.std_ddof == 1
    assert group.group_stats.std_ddof == spec.std_ddof
    assert group.group_stats.std == pytest.approx(
        float(np.std(np.asarray(SCORES), ddof=spec.std_ddof)), abs=1e-12
    )
    # and it is not the value the old call shape produced, by the factor the row measured.
    population = float(np.std(np.asarray(SCORES), ddof=0))
    assert group.group_stats.std / population == pytest.approx(BESSEL_AT_K8, abs=1e-12)


def test_the_tapped_std_agrees_with_the_contract_field_the_recorder_emits():
    """The third clause: one group, two producers, one answer.

    `recorder/groupstats.py` computes `group_std` with the stdlib `statistics`, choosing
    `pstdev` at ddof 0 and `stdev` at ddof 1 from the ddof the same record carries. That
    definition is recomputed here rather than imported, because the library does not import the
    experiment tree; being a second implementation over a different library is what makes the
    agreement evidence.
    """
    run = drive(TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT))
    group = one_group(run)
    recorded_ddof = group.estimator.std_ddof
    recorder_group_std = (
        statistics.pstdev(SCORES) if recorded_ddof == 0 else statistics.stdev(SCORES)
    )

    assert group.group_stats.std == pytest.approx(recorder_group_std, abs=1e-12)
    # the disagreement the row measured, kept so the agreement above is not read as trivial.
    assert group.group_stats.std != pytest.approx(statistics.pstdev(SCORES), abs=1e-12)

    # and the other contract field, computed the recorder's way: the weighted objective at or
    # below FAIL_AT on every rollout.
    fail_run = drive(
        TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT),
        scores=ALL_FAIL_SCORES,
    )
    recorder_group_all_fail = all(v <= RECORDER_FAIL_AT for v in ALL_FAIL_SCORES)
    assert one_group(fail_run).group_stats.all_fail is recorder_group_all_fail


def test_under_scale_rewards_none_the_record_claims_no_denominator_it_did_not_use():
    """`scale_rewards="none"` means the trainer takes no standard deviation at all.

    `_estimator_spec` writes `std_ddof=None` there, and the same one producer therefore hands
    `from_scores` None, which computes the population form and records that it did. That is the
    case `recorder/groupstats.py` refuses outright when the recorded ddof is not 0, and the two
    now agree instead of contradicting each other.
    """
    run = drive(
        TRLTap(run_id="blk021", budget=GENEROUS, failure_at=RECORDER_FAIL_AT),
        scale_rewards="none",
    )
    group = one_group(run)
    assert group.estimator.std_normalised is False
    assert group.estimator.std_ddof is None
    assert group.group_stats.std_ddof == 0
    assert group.group_stats.std == pytest.approx(
        float(np.std(np.asarray(SCORES), ddof=0)), abs=1e-12
    )
    assert group.group_stats.std == pytest.approx(statistics.pstdev(SCORES), abs=1e-12)


# ---------------------------------------------------------------------------
# one definition, one producer
# ---------------------------------------------------------------------------


def test_there_is_one_producer_of_the_denominator_in_this_adapter():
    """The archetype guard. Two files disagreeing is what this row is; two *lines* disagreeing in
    one file is the same failure at a shorter range, and it is what a repair that hard-codes the
    denominator a second time at the construction site would leave behind."""
    from reward_lens.tap.adapters import trl as trl_module

    source = inspect.getsource(trl_module)
    assert source.count("std_ddof=1 ") + source.count("std_ddof=1,") == 1, (
        "the denominator is decided in exactly one place in this adapter, `_estimator_spec`. "
        "A second literal is the drift this row exists to remove."
    )
    site = inspect.getsource(trl_module.TRLTap._groups_for)
    assert "std_ddof=estimator.std_ddof" in site
    assert "failure_at=self.failure_at" in site


def test_the_contract_subclass_builds_no_group_stats_of_its_own():
    """`ContractTRLTap` is the tap the C1 recorder drives. It must inherit the construction site
    rather than own a second one, or the repair is true of the base class and false of the run."""
    from reward_lens.tap.adapters import trl_contract

    source = inspect.getsource(trl_contract)
    assert "from_scores" not in source
    assert "GroupStats" not in source
    assert "super()._groups_for(b)" in inspect.getsource(trl_contract.ContractTRLTap._groups_for)
    assert trl_contract.ContractTRLTap._estimator_spec is TRLTap._estimator_spec
