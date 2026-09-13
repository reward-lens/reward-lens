"""BLK-010: the per-rollout completion length in tokens reaches the record.

TRL builds the per-sequence vector at ``grpo_trainer.py:2289`` and ``:2291``, reduces it to
mean, min and max at ``:2302-2304``, and keeps nothing per row. The tap never emitted it either,
so a downstream analysis that needs a length has only a character count to reach for, and the
design's own ledger records that a character proxy already produced one wrong published negative
in this project: "the largest upper-tail tie is 2" was in characters and is **211 in tokens**.

The fixture below is built around that number on purpose. Row 1 is forty characters and three
tokens; row 2 is one character and two hundred and eleven tokens. Any implementation that reaches
for ``len(text)`` gets 40 and 1, and every assertion here fails.

**The field-shape convention this row establishes**, which BLK-009 and BLK-005 follow:

1. A per-rollout quantity is emitted **on the rollout's own row** and never as a step aggregate.
2. Its name is the recording contract's name
   (``chain/repair/schema/recording_contract.json`` -> ``completion_length_tokens``), so one
   quantity has one spelling across the corpus.
3. It is read off the object the trainer handed the grader, which is **upstream** of any
   reordering the trainer does afterwards.
4. When the source is absent the key is **absent**. Not a zero, not a proxy, and never a
   reconstruction from a logged aggregate.
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

#: The incident this row exists to prevent, reproduced as a fixture. Index 1 is the row whose
#: character count (1) and token count (211) disagree by two orders of magnitude.
COMPLETION_TEXTS = ("a" * 40, "x", "abc", "d" * 7)
COMPLETION_IDS = (
    tuple(range(9000, 9003)),  # 3 tokens, 40 characters
    tuple(range(9100, 9311)),  # 211 tokens, 1 character
    tuple(range(9400, 9402)),  # 2 tokens, 3 characters
    tuple(range(9500, 9505)),  # 5 tokens, 7 characters
)
CHAR_LENGTHS = tuple(len(t) for t in COMPLETION_TEXTS)
TOKEN_LENGTHS = tuple(len(ids) for ids in COMPLETION_IDS)


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
    generation_batch_size = 4
    max_completion_length = 512
    num_generations = 2
    reward_weights = None
    log_completions = True
    use_vllm = False
    optim = "adamw_torch"
    per_device_train_batch_size = 4


class FakeState:
    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step


class FakeTrainer:
    """Only the attributes the callback reads, as in ``tests/test_tap_trl.py``."""

    def __init__(self) -> None:
        self.args = FakeArgs()
        self.num_generations = 2
        self.model = None
        self.tools = None
        self.callbacks: list = []
        self._logs = {"prompt": [], "completion": [], "advantages": [], "rewards": {}}
        self._metrics = {"train": {}}

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)


def grader(prompts, completions, completion_ids, **kwargs):
    return [1.0 * i for i in range(len(completions))]


def drive(tap: TRLTap, *, completion_ids=COMPLETION_IDS):
    """One step through the whole seam: wrap, attach, call the grader, fire the boundary."""
    trainer = FakeTrainer()
    wrapped = tap.wrap(grader)
    tap.attach(trainer)
    callback = trainer.callbacks[0]
    prompts = ["p0", "p0", "p1", "p1"]
    call_kwargs = {
        "trainer_state": FakeState(0),
        "log_metric": lambda name, value: None,
        "log_extra": lambda column, values: None,
        "completion_ids": [list(ids) for ids in completion_ids],
    }
    wrapped(prompts=prompts, completions=list(COMPLETION_TEXTS), **call_kwargs)
    trainer._logs["prompt"] = prompts
    trainer._logs["completion"] = list(COMPLETION_TEXTS)
    trainer._logs["advantages"] = [0.5, -0.5, 0.5, -0.5]
    callback.on_step_end(trainer.args, FakeState(1), None)
    callback.on_train_end(trainer.args, FakeState(1), None)
    return tap.finish()


def rollouts(run):
    return [t for s in run.steps for g in s.groups for t in g.trajectories]


# ---------------------------------------------------------------------------
# the fixture is the test's own control: chars and tokens must disagree
# ---------------------------------------------------------------------------


def test_the_fixture_discriminates_between_the_proxy_and_the_count():
    """If these agreed, every assertion below would pass under a character proxy."""
    assert CHAR_LENGTHS == (40, 1, 3, 7)
    assert TOKEN_LENGTHS == (3, 211, 2, 5)
    assert all(c != t for c, t in zip(CHAR_LENGTHS, TOKEN_LENGTHS))
    assert 211 in TOKEN_LENGTHS and 211 not in CHAR_LENGTHS


# ---------------------------------------------------------------------------
# the row
# ---------------------------------------------------------------------------


def test_the_token_count_appears_on_the_row_beside_the_reward():
    """The contract row: "per-rollout completion length **in tokens**, on the row beside the
    reward". Beside the reward means on the same `Trajectory`, not in a step-level metric."""
    run = drive(TRLTap(run_id="blk010", budget=GENEROUS))
    trajectories = rollouts(run)
    assert len(trajectories) == 4
    got = [t.features.get("completion_length_tokens") for t in trajectories]
    assert got == [3.0, 211.0, 2.0, 5.0]
    # beside the reward: the same object carries both.
    assert all(t.scores is not None for t in trajectories)


def test_the_emitted_count_is_trls_own_per_sequence_vector():
    """`grpo_trainer.py:2291` is `torch.tensor([len(ids) for ids in completion_ids])`.

    Recomputed here from the same input the trainer used, which is the one the tap was handed.
    """
    trl_completion_lengths = [len(ids) for ids in COMPLETION_IDS]  # grpo_trainer.py:2291
    run = drive(TRLTap(run_id="blk010", budget=GENEROUS))
    got = [int(t.features["completion_length_tokens"]) for t in rollouts(run)]
    assert got == trl_completion_lengths


def test_the_character_proxy_is_emitted_too_and_is_not_the_token_field():
    """The row's closure proof wants both, so a reader can see the substitution rather than
    inherit it. The two keys are named apart and the token one is the one an analysis reads."""
    run = drive(TRLTap(run_id="blk010", budget=GENEROUS))
    trajectories = rollouts(run)
    chars = [t.features.get("completion_length_chars") for t in trajectories]
    tokens = [t.features.get("completion_length_tokens") for t in trajectories]
    assert chars == [40.0, 1.0, 3.0, 7.0]
    assert tokens == [3.0, 211.0, 2.0, 5.0]
    assert chars != tokens


def test_no_token_count_is_invented_when_the_reference_was_dropped():
    """Absent is absent. `retain_args=False` is the real degraded path: the tap keeps the shapes
    and drops the host's objects, so the ids are gone and the completions come back off
    `trainer._logs`. A zero here reads as an empty completion, and a character count read as a
    token count is the published-negative incident this row is named for.

    Paired with the retaining tap in the same test so it cannot pass by emitting nothing ever.
    """
    kept = rollouts(drive(TRLTap(run_id="blk010", budget=GENEROUS, retain_args=True)))
    assert [int(t.features["completion_length_tokens"]) for t in kept] == [3, 211, 2, 5]

    dropped = rollouts(drive(TRLTap(run_id="blk010", budget=GENEROUS, retain_args=False)))
    assert len(dropped) == 4
    assert all("completion_length_tokens" not in t.features for t in dropped)
    # the proxy is still emitted, still labelled as characters, and is never promoted.
    assert [t.features.get("completion_length_chars") for t in dropped] == [40.0, 1.0, 3.0, 7.0]


def test_a_truncated_id_list_is_refused_rather_than_padded():
    """A shorter `completion_ids` than `completions` means the tap is looking at a different
    batch. Emitting the rows it can match would put a length on a row it does not belong to.

    Paired with the well-formed batch so the assertion cannot be satisfied by never emitting.
    """
    whole = rollouts(drive(TRLTap(run_id="blk010", budget=GENEROUS)))
    assert [int(t.features["completion_length_tokens"]) for t in whole] == [3, 211, 2, 5]

    ragged = rollouts(
        drive(TRLTap(run_id="blk010", budget=GENEROUS), completion_ids=COMPLETION_IDS[:2])
    )
    assert len(ragged) == 4
    assert all("completion_length_tokens" not in t.features for t in ragged)


@pytest.mark.parametrize("index,expected", [(0, 3), (1, 211), (2, 2), (3, 5)])
def test_every_row_carries_its_own_length_and_not_the_batch_mean(index, expected):
    """The whole defect is a reduction. A per-row field that happens to hold the mean on every
    row would satisfy a laxer test; these four values are distinct and the mean is 55.25."""
    run = drive(TRLTap(run_id="blk010", budget=GENEROUS))
    trajectories = rollouts(run)
    assert int(trajectories[index].features["completion_length_tokens"]) == expected
    mean = sum(TOKEN_LENGTHS) / len(TOKEN_LENGTHS)
    assert mean == pytest.approx(55.25)
    assert int(trajectories[index].features["completion_length_tokens"]) != int(mean)
