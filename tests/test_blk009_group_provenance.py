"""BLK-009: the prompt-group id and the generation index survive the shuffle.

`grpo_trainer.py:1589` is `generation_batch = shuffle_sequence_dict(generation_batch)`, inside
`_prepare_inputs` and applied to whatever `_generate_and_score_completions` returned at `:1587`.
After that line the row order carries no grouping at all, so anything that reconstructs a group
from row order downstream of it reconstructs the wrong one.

Everything the tap reads is upstream of that line and this file is what says so rather than
assuming it. The reward functions are called at `:1673`, inside
`_generate_and_score_completions`. `_logs["prompt"]`, `["completion"]`, `["rewards"]` and
`["advantages"]` are extended at `:2823-2827`, also inside it. The shuffle happens to the return
value, one frame up.

**Why a permutation has to be injected.** The tap groups by consecutive runs of
`num_generations` rows. On an unshuffled fixture that assumption is true whatever the tap does,
so a test built on one passes without touching the defect and fails silently on a real batch. The
fixture here applies a real permutation and the two arms differ in one thing only: which order the
tap is handed. `test_the_injected_permutation_actually_changes_the_answer` is the control that
keeps the main assertion from being vacuous.

**The comparison quantity.** TRL's own `frac_reward_zero_std` at `:2818` is
`is_std_zero.float().mean()`, where `is_std_zero` is `torch.isclose(std_rewards, 0)` on the
row-expanded per-group standard deviation. The tap's counterpart is
`GroupStats.degenerate`, `std <= std_epsilon`. TRL's `nanstd` (`trl/trainer/utils.py:859`) carries
Bessel's correction and `GroupStats` uses `ddof=0`, so the two standard deviations differ by
`sqrt(K/(K-1))` -- that is BLK-021's half of this and not repaired here. The zero/non-zero
decision is identical under both conventions, which is why the fraction is comparable and the
fixture keeps its live groups far from either threshold.
"""

from __future__ import annotations

import pytest

from reward_lens.tap.adapters.trl import TRLTap
from reward_lens.tap.contract import TapBudget

GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=16 * 1024 * 1024,
    max_added_alloc_bytes_per_step=16 * 1024 * 1024,
)

K = 2

#: Four groups of two in generation order. Groups 0 and 1 are degenerate (no spread inside the
#: group); groups 2 and 3 are live and far from any epsilon. The trainer's degeneracy fraction is
#: therefore exactly 0.5.
PROMPTS = ("p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3")
REWARDS = (1.0, 1.0, 2.0, 2.0, 3.0, 5.0, 7.0, 11.0)
COMPLETIONS = tuple(f"c{i}" for i in range(8))
COMPLETION_IDS = tuple(tuple(range(100 * i, 100 * i + 3)) for i in range(8))

#: The injected permutation: post-shuffle row `i` is generation-order row `PERMUTATION[i]`. Chosen
#: so that every post-shuffle pair straddles two different prompts, which is what a real
#: `shuffle_sequence_dict` does to a batch and what makes the two arms disagree.
PERMUTATION = (0, 2, 1, 3, 4, 6, 5, 7)

EXPECTED_GROUP_IDS = (0, 0, 1, 1, 2, 2, 3, 3)
EXPECTED_GENERATION_INDEX = (0, 1, 0, 1, 0, 1, 0, 1)


def degeneracy_fraction(rewards, k: int) -> float:
    """`frac_reward_zero_std` reproduced from `grpo_trainer.py:2760-2777` and `:2818`.

    `nanstd(rewards.view(-1, num_generations), dim=1)`, `repeat_interleave`d back over the rows,
    then `torch.isclose(std, 0).float().mean()`. Written out in plain Python so the comparison
    quantity is derived here rather than read off a metric the fixture itself supplied.
    """
    rows = [rewards[i : i + k] for i in range(0, len(rewards), k)]
    flags = []
    for group in rows:
        mean = sum(group) / len(group)
        var = sum((x - mean) ** 2 for x in group) / len(group)
        correction = len(group) / (len(group) - 1) if len(group) > 1 else 1.0
        std = (var * correction) ** 0.5
        flags.extend([abs(std) <= 1e-8] * len(group))
    return sum(1 for f in flags if f) / len(flags)


class FakeArgs:
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
    generation_batch_size = 8
    max_completion_length = 512
    num_generations = K
    reward_weights = None
    log_completions = True
    use_vllm = False
    optim = "adamw_torch"
    per_device_train_batch_size = 8


class FakeState:
    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step


class FakeTrainer:
    def __init__(self) -> None:
        self.args = FakeArgs()
        self.num_generations = K
        self.model = None
        self.tools = None
        self.callbacks: list = []
        self._logs = {"prompt": [], "completion": [], "advantages": [], "rewards": {}}
        self._metrics = {"train": {}}

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)


def permute(seq, perm=PERMUTATION):
    return [seq[i] for i in perm]


def drive(tap: TRLTap, *, order: str = "generation", steps: int = 1):
    """One or more steps, with the row order the tap is handed under the test's control.

    ``order="generation"`` is where the seams actually are: the reward function is called and
    ``_logs`` is written inside ``_generate_and_score_completions``, upstream of the shuffle.

    ``order="shuffled"`` is the named failure mode -- a tap that captured downstream of
    ``shuffle_sequence_dict``. Both arms compute the trainer's own degeneracy fraction from the
    generation order, because that is where `:2818` computes it.
    """
    trainer = FakeTrainer()

    def grader(prompts, completions, completion_ids, **kwargs):
        by_completion = dict(zip(COMPLETIONS, REWARDS))
        return [by_completion[c] for c in completions]

    wrapped = tap.wrap(grader)
    tap.attach(trainer)
    callback = trainer.callbacks[0]

    seen_prompts = list(PROMPTS) if order == "generation" else permute(PROMPTS)
    seen_completions = list(COMPLETIONS) if order == "generation" else permute(COMPLETIONS)
    seen_ids = (
        [list(x) for x in COMPLETION_IDS]
        if order == "generation"
        else [list(x) for x in permute(COMPLETION_IDS)]
    )

    for step in range(steps):
        wrapped(
            prompts=seen_prompts,
            completions=seen_completions,
            completion_ids=seen_ids,
            trainer_state=FakeState(step),
            log_metric=lambda name, value: None,
            log_extra=lambda column, values: None,
        )
        trainer._logs["prompt"] = list(seen_prompts)
        trainer._logs["completion"] = list(seen_completions)
        trainer._logs["advantages"] = [0.0] * len(seen_completions)
        trainer._metrics["train"] = {
            "frac_reward_zero_std": [degeneracy_fraction(REWARDS, K)],
        }
        callback.on_step_end(trainer.args, FakeState(step + 1), None)
    callback.on_train_end(trainer.args, FakeState(steps), None)
    return tap.finish()


def rollouts(run):
    return [t for s in run.steps for g in s.groups for t in g.trajectories]


# ---------------------------------------------------------------------------
# the fixture's own controls
# ---------------------------------------------------------------------------


def test_the_trainers_degeneracy_fraction_on_this_fixture_is_a_half():
    assert degeneracy_fraction(REWARDS, K) == 0.5


def test_the_injected_permutation_actually_changes_the_answer():
    """Without this the main assertion is satisfied by any fixture at all.

    The permutation is not a relabelling: read in the post-shuffle order the same eight rewards
    give a different degeneracy fraction, so a tap that grouped downstream of the shuffle would
    record a number that is wrong rather than merely differently ordered.
    """
    shuffled_rewards = permute(REWARDS)
    assert degeneracy_fraction(shuffled_rewards, K) == 0.0
    assert degeneracy_fraction(shuffled_rewards, K) != degeneracy_fraction(REWARDS, K)
    # and it scrambles the prompts inside every group, which is the signal the guard reads.
    shuffled_prompts = permute(PROMPTS)
    assert [shuffled_prompts[i : i + K] for i in range(0, 8, K)] == [
        ["p0", "p1"],
        ["p0", "p1"],
        ["p2", "p3"],
        ["p2", "p3"],
    ]


# ---------------------------------------------------------------------------
# the row: two fields, captured upstream
# ---------------------------------------------------------------------------


def test_the_prompt_group_id_is_a_field_on_every_rollout():
    run = drive(TRLTap(run_id="blk009", budget=GENEROUS))
    got = [int(t.features["prompt_group_id"]) for t in rollouts(run)]
    assert got == list(EXPECTED_GROUP_IDS)


def test_the_generation_index_is_a_field_and_not_only_a_suffix_of_the_id():
    """`:804` put it inside `trajectory_id(group=..., ordinal=j - lo)`. A field a reader has to
    recover by parsing an opaque id is not a recorded field, and the id's construction is free to
    change without anybody noticing what it took with it."""
    run = drive(TRLTap(run_id="blk009", budget=GENEROUS))
    trajectories = rollouts(run)
    got = [int(t.features["generation_index"]) for t in trajectories]
    assert got == list(EXPECTED_GENERATION_INDEX)
    # the id is still opaque; nothing here parsed it.
    assert all(str(t.id) != str(i) for i, t in enumerate(trajectories))


def test_the_recorded_grouping_reproduces_the_trainers_degeneracy_fraction():
    """The row's closure proof, on every step.

    **This one passes at `59a5f5a` and it is recorded here that it does.** The tap already read
    upstream of the shuffle before this repair, so the fraction already came out right; what did
    not exist was the two fields and any check that the reading was upstream. The tests that carry
    the row are `test_the_prompt_group_id_is_a_field_on_every_rollout`,
    `test_the_generation_index_is_a_field_and_not_only_a_suffix_of_the_id` and
    `test_a_post_shuffle_order_is_detected_and_the_two_fields_are_withheld`, all three of which
    fail at the baseline. This one is kept because it must keep holding, not because it discovered
    anything.
    """
    run = drive(TRLTap(run_id="blk009", budget=GENEROUS), steps=3)
    expected = degeneracy_fraction(REWARDS, K)
    steps = list(run.steps)
    assert len(steps) == 3
    for step in steps:
        recorded = sum(1 for g in step.groups if g.group_stats.degenerate) / len(step.groups)
        assert recorded == pytest.approx(expected), (
            f"step {step.index}: tap grouping gives {recorded}, trainer reported {expected}"
        )


def test_every_group_holds_one_prompt_and_the_rows_that_belong_to_it():
    run = drive(TRLTap(run_id="blk009", budget=GENEROUS))
    step = next(iter(run.steps))
    assert len(step.groups) == 4
    for ordinal, group in enumerate(step.groups):
        prompts = {t.turns[0].text for t in group.trajectories}
        assert len(prompts) == 1, f"group {ordinal} straddles prompts {sorted(prompts)}"
        assert {int(t.features["prompt_group_id"]) for t in group.trajectories} == {ordinal}
        assert sorted(int(t.features["generation_index"]) for t in group.trajectories) == [0, 1]


# ---------------------------------------------------------------------------
# the guard, demonstrated firing on a broken fixture
# ---------------------------------------------------------------------------


def test_a_post_shuffle_order_is_detected_and_the_two_fields_are_withheld():
    """The named failure mode: capturing the group id downstream of `shuffle_sequence_dict`.

    A tap reading that order cannot recover the grouping from row order and must not pretend to.
    The check is the one signal that survives the permutation: the K rows of a group are one
    prompt's rollouts, so a candidate group whose rows carry different prompts is not a group.
    """
    tap = TRLTap(run_id="blk009", budget=GENEROUS)
    run = drive(tap, order="shuffled")
    trajectories = rollouts(run)
    assert len(trajectories) == 8
    assert all("prompt_group_id" not in t.features for t in trajectories)
    assert all("generation_index" not in t.features for t in trajectories)
    assert tap.unverified_grouping_steps == 1
    assert "prompt" in tap.grouping_refusal_reason


def test_the_wrong_grouping_would_have_recorded_the_wrong_fraction():
    """What the withheld fields would have said. This is the number the row exists to stop being
    written, and it is recorded here so the guard's value is visible rather than asserted."""
    tap = TRLTap(run_id="blk009", budget=GENEROUS)
    run = drive(tap, order="shuffled")
    step = next(iter(run.steps))
    recorded = sum(1 for g in step.groups if g.group_stats.degenerate) / len(step.groups)
    assert recorded == 0.0
    assert recorded != degeneracy_fraction(REWARDS, K)


def test_the_guard_does_not_fire_on_the_order_the_tap_actually_sees():
    tap = TRLTap(run_id="blk009", budget=GENEROUS)
    drive(tap, order="generation", steps=3)
    assert tap.unverified_grouping_steps == 0
    assert tap.grouping_refusal_reason == ""
