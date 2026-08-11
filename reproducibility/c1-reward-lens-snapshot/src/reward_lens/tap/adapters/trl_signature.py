"""The TRL tap's public signature, declared and checked. BLK-048.

The blocker's own words: "The tap's public interface is cited by file and line
and never by signature, so a builder cannot tell whether it accepts new
per-rollout scalars without a schema change." `CHAIN_GAPS.md` cites
`tap/adapters/trl.py:796`, `:804` and `:975` and gives no signature. A builder
without a signature writes a parallel parquet writer, which is what the standing
rule "fix the library and never the caller" exists to stop.

So the signature is written down here, **and checked**, because a documented
signature that drifts is worse than none: it is a wrong answer with a citation.
`verify()` introspects the live `TRLTap` and raises on any divergence. Run it
from a test and the documentation cannot go stale without something going red.

Line numbers are not used anywhere in this module. The three the audit cites had
already moved by the time it was read: at the pinned SHA the group id went in at
`:796` and the trajectory id at `:804`, and they are now elsewhere. The module's
own docstring in `trl.py` still cites the old numbers, which is the same defect
one level down.

## The public surface, in the order a caller uses it

    tap = TRLTap(run_id=..., budget=..., emit_metrics=..., emit_extra=...,
                 retain_args=..., record_grad_presence=..., max_steps=...,
                 ring=..., name=...)            # all keyword-only
    funcs = tap.wrap(reward_funcs)              # BEFORE the trainer is built
    trainer = tap.attach(GRPOTrainer(...))      # AFTER; returns the trainer
    ...                                         # train
    run = tap.finish(kind="train", access=None) # the whole Run record
    RecordWriter(root).write(run)               # persistence is the caller's

## The three seams, which are where an emitter attaches

  1. **the reward functions.** `wrap` instruments each callable and the
     instrumented callable carries `.instrumented`, `.guard`, `.ring`,
     `.effect`.
  2. **the trainer.** `attach` installs a `TrainerCallback` with four hooks:
     `on_step_end`, `on_pre_optimizer_step`, `on_log`, `on_train_end`. Every
     one is routed through `_guarded`, which swallows exceptions so the tap
     cannot break training.
  3. **the record.** `steps()` and `finish()`. This is the boundary: everything
     before it swallows exceptions and these two raise. An emitter that needs a
     field to be present rather than best-effort puts its refusal here.

## The extension points an emitter overrides, and what each returns

`_features_for` is the one that matters, because it is the only per-rollout
scalar map on the trajectory and it is `Mapping[str, float]`. A contract field
that is not a float does not go here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

__all__ = ["EXPECTED", "SignatureDrift", "declared", "verify"]


class SignatureDrift(RuntimeError):
    """The live tap does not match the signature this module documents."""


@dataclass(frozen=True)
class Declared:
    name: str
    kind: str  # "method" | "property" | "attribute"
    params: tuple[str, ...] = ()
    keyword_only: bool = False
    note: str = ""


# The public surface. `params` excludes `self`.
EXPECTED: tuple[Declared, ...] = (
    Declared(
        "__init__",
        "method",
        (
            "run_id",
            "budget",
            "emit_metrics",
            "emit_extra",
            "retain_args",
            "record_grad_presence",
            "max_steps",
            "ring",
            "name",
        ),
        keyword_only=True,
        note="all keyword-only; raises ValueError on max_steps < 1",
    ),
    Declared(
        "wrap",
        "method",
        ("reward_funcs",),
        note="call BEFORE GRPOTrainer(...); accepts one callable or a "
        "sequence and returns the same shape",
    ),
    Declared(
        "attach", "method", ("trainer",), note="call AFTER the trainer exists; returns the trainer"
    ),
    Declared(
        "effects", "method", (), note="one InstrumentEffect per wrapped function, in wrap order"
    ),
    Declared(
        "metric_names",
        "method",
        (),
        note="invariant to tap state on purpose: a rank-varying key set "
        "hangs TRL's per-name gather",
    ),
    Declared("steps", "method", (), note="builds the Step records. The boundary: this raises"),
    Declared("sampling_policy", "method", ()),
    Declared(
        "finish",
        "method",
        ("kind", "access"),
        keyword_only=True,
        note="the whole Run. Persistence is the caller's, through "
        "reward_lens.record.writer.RecordWriter",
    ),
    Declared("instrument_effect", "property", ()),
)

# The seams an emitter is expected to override. Named here so that a rename in
# the tap breaks the emitter loudly rather than silently disabling a field.
SEAMS: tuple[Declared, ...] = (
    Declared(
        "_features_for",
        "method",
        (
            "row",
            "weights",
            "completion_text",
            "completion_tokens",
            "group_ordinal",
            "generation_index",
        ),
        note="Mapping[str, float] per rollout. The ONLY per-rollout scalar "
        "map on the trajectory, and it is float-typed, so a contract "
        "field that is not a float does not go through it",
    ),
    Declared(
        "_turns_for",
        "method",
        ("prompt", "completion"),
        note="returns (user Turn, assistant Turn). The tap sets index, "
        "role and text only, though Turn already declares token_ids, "
        "logprobs_sampling, logprobs_train and loss_mask",
    ),
    Declared(
        "_groups_for", "method", ("b",), note="builds the Group and Trajectory objects for one step"
    ),
    Declared("_schedule_for", "method", ("b",), note="Mapping[str, float] per step"),
)


def declared() -> str:
    """The signature as text, for a proof directory or a design appendix."""
    out = ["TRLTap, public surface:"]
    for d in EXPECTED:
        star = "*, " if d.keyword_only and d.params else ""
        out.append(
            f"  {d.name}({star}{', '.join(d.params)})" + (f"    # {d.note}" if d.note else "")
        )
    out.append("")
    out.append("Seams an emitter overrides:")
    for d in SEAMS:
        out.append(f"  {d.name}({', '.join(d.params)})" + (f"    # {d.note}" if d.note else ""))
    return "\n".join(out)


def verify(cls=None) -> None:
    """Raise if the live tap has drifted from what this module documents."""
    if cls is None:
        from reward_lens.tap.adapters.trl import TRLTap as cls  # noqa: N813

    problems: list[str] = []
    for d in EXPECTED + SEAMS:
        obj = getattr(cls, d.name, None)
        if obj is None:
            problems.append(f"{d.name}: documented and absent from {cls.__name__}")
            continue
        if d.kind == "property":
            if not isinstance(inspect.getattr_static(cls, d.name), property):
                problems.append(f"{d.name}: documented as a property and is not one")
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            problems.append(f"{d.name}: has no inspectable signature")
            continue
        got = [p for p in sig.parameters if p != "self"]
        missing = [p for p in d.params if p not in got]
        if missing:
            problems.append(
                f"{d.name}: documented parameters {missing} are not in the live signature {got}"
            )
        if d.keyword_only and d.params:
            positional = [
                p.name
                for p in sig.parameters.values()
                if p.name != "self" and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            shared = [p for p in positional if p in d.params]
            if shared:
                problems.append(f"{d.name}: documented keyword-only and {shared} are positional")
    if problems:
        raise SignatureDrift(
            "the TRL tap's public signature has drifted from "
            "reward_lens.tap.adapters.trl_signature:\n  " + "\n  ".join(problems)
        )
