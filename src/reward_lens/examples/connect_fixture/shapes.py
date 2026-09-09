"""One callable per shape of the frozen `Shape` enum. D-65 forbids a shape recognised without one.

Each of these is the thing the connector's test actually runs `inspect.signature` over, so the
four enum members are established by four separate objects rather than by four branches of one.
The `PreTrainedModel` branch is a stub: it carries the `config`, the `forward(input_ids, ...)`
signature and the `.logits` output that D-65's `.logits[:, 0]` branch needs, and it downloads
nothing.

Owned by P-CONNECT.
"""

from dataclasses import dataclass

__all__ = [
    "Score",
    "ScoringHeadOutput",
    "ScoringHeadStub",
    "StubConfig",
    "async_trl_fn",
    "batch_fn",
    "inspect_scorer",
    "scoring_head_stub",
    "single_fn",
    "trl_declares_only_what_it_uses",
    "trl_in_any_order",
    "trl_keyword_only",
    "var_keyword_reward",
]


def single_fn(prompt, completion, **kw) -> float:
    """`SINGLE_FN`: one item in, one number out. Interfaces section 1's signature exactly."""
    if not completion:
        return 0.0
    return min(1.0, len(prompt) / len(completion))


def batch_fn(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """`BATCH_FN`: how TRL calls a reward function (`grpo_trainer.py:1659-1661`, TRL 1.9.2)."""
    return [1.0 if completion else 0.0 for completion in completions]


async def async_trl_fn(prompts, completions, completion_ids, **kwargs) -> list[float]:
    """`ASYNC_TRL`: the same call, awaited. Told apart by `inspect.iscoroutinefunction`."""
    return [0.5 if completion else 0.0 for completion in completions]


def var_keyword_reward(**kwargs) -> float:
    """A `**kwargs`-only reward, which `verifiers` binds by handing it everything it has."""
    return 1.0 if kwargs.get("completion") else 0.0


@dataclass(frozen=True)
class StubConfig:
    """The `config` a transformers model carries. Two fields, and no framework behind them."""

    model_type: str = "reward_lens_stub"
    num_labels: int = 1


@dataclass(frozen=True)
class ScoringHeadOutput:
    """What `forward` returns: the `logits` D-65's `.logits[:, 0]` branch reads."""

    logits: list[list[float]]


class ScoringHeadStub:
    """`PRETRAINED_MODEL`: a scoring head with the transformers calling convention, stubbed."""

    base_model_prefix = "stub"

    def __init__(self, config: StubConfig | None = None) -> None:
        self.config = config or StubConfig()

    def forward(self, input_ids, attention_mask=None) -> ScoringHeadOutput:
        return ScoringHeadOutput(logits=[[float(len(row)) / 100.0] for row in input_ids])

    def __call__(self, input_ids, attention_mask=None) -> ScoringHeadOutput:
        return self.forward(input_ids, attention_mask=attention_mask)


def scoring_head_stub() -> ScoringHeadStub:
    """The fixture instance. A factory, so every test gets its own object."""
    return ScoringHeadStub()


# --- TRL's keyword-bound forms, one fixture each ---------------------------------------------------
#
# TRL calls a reward function with `prompts=`, `completions=`, `completion_ids=` and the dataset's
# extra columns, and never positionally (`trl/trainer/grpo_trainer.py:1683-1685`, TRL 1.13.0). So a
# reward function that declares only what it uses is the common real form, and all three of these
# are the same convention written three ways.


def trl_declares_only_what_it_uses(completions, **kwargs) -> list[float]:
    """`def f(completions, **kwargs)`: the form the docs use, and the one attempt 1 read as plain."""
    return [1.0 if completion else 0.0 for completion in completions]


def trl_in_any_order(completions, prompts, **kwargs) -> list[float]:
    """Order is not part of the convention, because nothing is passed positionally."""
    return [float(len(prompt) > len(completion)) for prompt, completion in zip(prompts, completions)]


def trl_keyword_only(*, prompts, completions, completion_ids, **kwargs) -> list[float]:
    """Keyword-only parameters, which is how D-65 writes the TRL shape down."""
    return [float(len(ids)) / 100.0 for ids in completion_ids]


class Score:
    """What an inspect scorer returns. A value and an explanation, and no framework behind it."""

    __slots__ = ("value", "explanation")

    def __init__(self, value: float, explanation: str = "") -> None:
        self.value = value
        self.explanation = explanation


async def inspect_scorer(state, target) -> "Score":
    """`inspect`: two positional arguments, `state` then `target`, awaited, a `Score` back.

    Written without the `TaskState` and `Target` annotations on purpose: the annotated form and
    this one are the same scorer, and the connector has to recognise both.
    """
    text = str(getattr(state, "output", state))
    return Score(1.0 if str(target) in text else 0.0, explanation="target in output")
