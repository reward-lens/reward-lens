"""The frozen `Shape` enum, D-65's declared vocabulary, and what a detection carries.

Two axes, kept apart on purpose.

`Shape` is the calling convention, and it is the frozen five of interfaces section 1. Its *values*
are the tokens that go into a record, so a shape survives serialisation without a second table.

`Detection.declared` is D-65's `reward.shape` vocabulary: `plain`, `trl`, `verifiers`, `inspect`.
That is the family, it is what an override is checked against, and it is what the connect
transcript prints in its shape column. A `verifiers` callable is `SINGLE_FN` when individual and
`BATCH_FN` when group; both are still `verifiers` in the project file.

Owned by P-CONNECT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from reward_lens.graders.base import ValidityFinding

__all__ = [
    "DECLARED_SHAPES",
    "Detection",
    "Finding",
    "Shape",
    "shape_for_declared",
]

#: Interfaces section 1 calls the warnings list `list[Finding]`. The wave-1 type it means is
#: `graders.base.ValidityFinding`, which is the only finding shape that carries a witness.
Finding = ValidityFinding


class Shape(str, Enum):
    """The four callable shapes 4.0 recognises, plus the refusal. Frozen by interfaces section 1."""

    #: `(prompt, completion, **kw) -> float`. One item in, one number out.
    SINGLE_FN = "plain"
    #: Lists in, a list out: how TRL calls a reward function, and how a group rubric is called.
    BATCH_FN = "trl"
    #: The same call, awaited. Told apart by `inspect.iscoroutinefunction`, never by a name.
    ASYNC_TRL = "trl_async"
    #: A `PreTrainedModel` with a scoring head, read through the `.logits[:, 0]` branch.
    PRETRAINED_MODEL = "pretrained_model"
    #: Recognised, and refused. Carries the reason and the docs page that says why.
    UNSUPPORTED = "unsupported"


#: D-65's four values for `reward.shape`. An override is checked against exactly this tuple.
DECLARED_SHAPES: tuple[str, ...] = ("plain", "trl", "verifiers", "inspect")


def shape_for_declared(declared: str, *, awaits: bool, group: bool) -> Shape:
    """The calling convention a declared family resolves to, given what the object turned out to be.

    D-65 puts the async variant and the `PreTrainedModel` branch inside `trl` rather than beside
    it, so the family alone never fixes the convention: the object still has to be looked at.
    """
    if declared == "plain":
        return Shape.SINGLE_FN
    if declared == "trl":
        return Shape.ASYNC_TRL if awaits else Shape.BATCH_FN
    if declared == "verifiers":
        return Shape.BATCH_FN if group else Shape.SINGLE_FN
    if declared == "inspect":
        return Shape.SINGLE_FN
    return Shape.UNSUPPORTED


@dataclass
class Detection:
    """What `detect` returns. `shape`, `entry`, `evidence` and `warnings` are frozen by section 1.

    The rest is additive and stated in the handoff: `declared` is the family token written to
    `rewardlens.yaml`, `awaits` says whether the entry has to be awaited, and `reason` and
    `docs_page` are filled only on `UNSUPPORTED`.
    """

    shape: Shape
    entry: str
    evidence: list[str] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    declared: str = "plain"
    awaits: bool = False
    reason: str = ""
    docs_page: str = ""
    #: The live object the entry resolved to. Not serialised; `bind` scores through it.
    target: object = None

    @property
    def signature_line(self) -> str:
        """The `name(params) -> return` line the transcript prints, taken off the evidence."""
        for line in self.evidence:
            if line.startswith("signature: "):
                return line[len("signature: ") :]
        return self.entry.rpartition(":")[2]

    @property
    def unsupported_label(self) -> str:
        """What an RL0710 refusal calls the shape it found: `step()`, or the callable's own name."""
        line = self.signature_line
        return "step()" if line.startswith("step(") else (line.partition("(")[0] or "unsupported")
