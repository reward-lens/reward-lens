"""A published per-rollout table, read as ledger steps, so `Λ` runs on a labelled series.

The instruments read `StepSample`, which is deliberately not a `Run`: the arithmetic is the same
whether the rows came from a record this library wrote or from a table somebody else published, and
putting the record type in the middle of it would make the second case a rewrite rather than an
adapter. This module is that adapter, and it is what puts the acceptance clause's labelled series
inside reach on a laptop.

**What changed here, and why it is a rewrite rather than a patch.** The first version of this
adapter took four keyword arguments with plausible defaults: which column was the reward, which was
the group, which was the step, and whether advantages were standardised. All four defaults were
wrong on the first published artifact it was pointed at, and one number this project had already
published was computed through all four of them. The defaults were not wrong because somebody chose
badly. They were wrong because a default is a determination nobody had to write down, so nobody
could audit it and nobody did.

So the four are gone. Reading an imported table now needs an `ImportedRun`, which carries a
composed reward with a resolution per component, an explicit advantage convention with the
provenance of the determination, an explicit generation count from which the optimiser's groups are
recovered, and a typed step axis. Those types live in `measure.ledger.imported` and none of them
knows about any publisher.

**Two of the four were already known elsewhere in this repository, which is the part worth sitting
with.** `experiments/x5_discontinuity.py` records that `training_passed = passed AND
thinking_format_ok` holds on 25,620 of 25,664 rows: it found the format scorer inside the reward
column months before anything went looking for a second component. And both `x4_exploit_growth.py`
and `x5_discontinuity.py` number the eval files from one when they build a step axis, which is
exactly the `f + 1` alignment this module got wrong, verified here: the rank of `source_eval_file`
in name order equals `rollout_index` on every row, so their `i + 1` is the trainer's step and this
module's `i` was not. Neither fact reached this file. A default is not only unaudited, it is
unconnectable: there is nowhere for a finding in another module to attach itself to a keyword
argument, and there is somewhere for it to attach to a `RewardComponent` with a `source` field.

**The one thing this adapter still has to reconstruct.** A rollout table carries rewards and labels
and no advantages, because the advantage is an artefact of the trainer rather than of the rollout.
So the advantage is recomputed from the group's own rewards under the declared convention, and
every reading built this way carries ``advantage_source="reconstructed"`` and a ``detail`` written
from the arithmetic that actually ran rather than from a template.

**Traps this adapter is written against, each a real property of the AISI series.**

1. The step index is **per eval file, not per row**. `rollout_index` there is the chronological
   index of the source `.eval` file. `check_step_axis` answers that question against the number of
   eval files the repository publishes, which is outside the table; comparing the table's step
   column against the table's own file column is an internal consistency check that passes on a run
   whose index is missing an entry, which is exactly the run where the question is being asked.
2. The label column is `int64` with **1, 0 or null**. A null is unscored, not a negative, so a naive
   `.mean()` over a NaN-filled float cast silently drops rows from the denominator without saying
   so. `label_rate` returns the rate, the denominator and the null count together.
3. The two published series are **different lengths**: one has 401 eval logs and the other 404.
   Nothing here hard-codes a length.
4. `hack_config` is a **JSON string**, not a struct, and is left as a string unless asked for.
5. Groups are the optimiser's, recovered from contiguous row blocks, and are keyed on `(step,
   group)`. The publisher's `problem_id` is the semantic problem and is **not** the group: three of
   the 401 files draw the same problem twice, and keying on it fuses two prompt groups into one
   group of thirty-two that no optimiser step ever normalised over.
6. The reward is a composition, not a column. `training_passed` is one component of two and the
   other is published as an indicator of itself, so asking this adapter for the operational reward
   on the AISI series returns `RECORD_INCOMPLETE` and says which column at which resolution closes
   it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.ledger.features import SURFACE_NAMES, surface_features
from reward_lens.measure.ledger.imported import (
    AdvantageConvention,
    AlignmentStrength,
    AxisAlignment,
    ComponentResolution,
    ComposedReward,
    ConventionSource,
    DeclaredConvention,
    ImportedRun,
    PromptGroups,
    ProxyReward,
    RewardComponent,
    RolloutColumns,
    StepAxis,
    index_gaps,
)
from reward_lens.measure.ledger.price import StepSample, advantages_from_rewards

#: The AISI reward-hacking rollout series, as the four determinations plus the columns that are not
#: the reward. Named as a preset rather than hard-coded into the loader, because the adapter is
#: about the shape and not about one publisher.
#:
#: Every field carries the source of its own determination and two of them are worth reading before
#: using this: the format component is published as an indicator of a quarter-valued score, so the
#: operational reward refuses; and the convention is a configuration-family determination rather
#: than an artifact one, because no released field records `scale_rewards`.
AISI_RUN = ImportedRun(
    name="ai-safety-institute/reward-hacking-olmo3.1-32b",
    columns=RolloutColumns(
        step="rollout_index",
        file="source_eval_file",
        label="reward_hacked",
        text="response",
        semantic_id="problem_id",
    ),
    reward=ComposedReward(
        components=(
            RewardComponent(
                name="thinking_format",
                weight=1.0,
                column="thinking_format_ok",
                resolution=ComponentResolution.BINARISED,
                source=(
                    "the weight is the first entry of `reward_weights` in the publisher's training "
                    "configurations, and the composition is identified independently by the "
                    "degeneracy structure reproducing exactly on every step"
                ),
                scorer_grid=0.25,
                published_grid=1.0,
                bounds=(0.0, 1.0),
            ),
            RewardComponent(
                name="training_passed",
                weight=4.0,
                column="training_passed",
                resolution=ComponentResolution.EXACT,
                source=(
                    "the weight is the second entry of `reward_weights` in the publisher's "
                    "training configurations; the column is the scorer's own binary output and is "
                    "published at the resolution the scorer computes it"
                ),
                bounds=(0.0, 1.0),
            ),
        ),
        source=(
            "identified by which columns the logged `frac_reward_zero_std` is a function of, which "
            "settles the composition without needing the weights"
        ),
    ),
    convention=DeclaredConvention(
        convention=AdvantageConvention.CENTRED,
        source=ConventionSource.CONFIGURATION_FAMILY,
        detail=(
            "`scale_rewards: none` appears in every one of the publisher's forty-one training "
            "configurations with no other value anywhere in the repository, and one of them "
            "carries the comment that Dr. GRPO recommends no scaling. No field of any released "
            "artifact records it, so this is a determination about the configuration family the "
            "run is presumed to belong to rather than about the run"
        ),
    ),
    generations=16,
    axis=StepAxis.ROLLOUT_FILE_INDEX,
    alignment=AxisAlignment(
        source=StepAxis.ROLLOUT_FILE_INDEX,
        target=StepAxis.TRAINER_LOG_STEP,
        offset=1,
        strength=AlignmentStrength.IDENTITY,
        evidence=(
            "the difference between the logged-minus-parquet offset at the completion-length "
            "minimum and the same offset at the maximum is zero at this shift and at no other. "
            "The difference is an identity rather than a fit: it is zero exactly when the two "
            "series describe the same sixty-four completions, and it does not degrade when the "
            "reward reconstruction is incomplete"
        ),
        statistics={"n_offset_consistent": 166, "n_unclipped": 237, "next_best": 1},
    ),
    note=(
        "The eval files carry no step field and are ordered by their ISO timestamp. Index i was "
        "read as step i for most of this project's life; it is step i + 1, with i = 0 the batch "
        "scored at step 1."
    ),
)


@dataclass(frozen=True)
class LabelRate:
    """One step's labelled rate, with the two numbers a rate is meaningless without."""

    step: int
    rate: float
    n_labelled: int
    n_null: int

    @property
    def n_total(self) -> int:
        return self.n_labelled + self.n_null


@dataclass(frozen=True)
class StepAxisReport:
    """What a step column is, checked against a count that does not come from the table."""

    n_distinct_steps: int
    n_distinct_files_in_table: int
    n_published_files: int | None
    contiguous: bool
    missing: tuple[int, ...]
    low: int | None
    high: int | None

    @property
    def agrees_with_publication(self) -> bool | None:
        if self.n_published_files is None:
            return None
        return self.n_distinct_steps == self.n_published_files


def _column(table: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in table:
        raise KeyError(
            f"the table has no column {name!r}; it carries {sorted(table)[:12]}"
            + ("..." if len(table) > 12 else "")
        )
    return np.asarray(table[name], dtype=object)


def _as_float(values: np.ndarray) -> np.ndarray:
    """An object column of ints, floats and Nones, as float64 with None becoming NaN.

    Explicit rather than `astype(float)`, because a column of Python `None` casts to the string
    ``'None'`` under some dtypes and to `nan` under others, and one of those two silently becomes a
    number.
    """
    out = np.empty(values.shape[0], dtype=np.float64)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = np.nan
        else:
            out[i] = float(v)
    return out


def check_step_axis(
    table: Mapping[str, Any],
    columns: RolloutColumns,
    *,
    n_published_files: int | None = None,
    instrument: str = "check_step_axis",
) -> StepAxisReport | Refusal:
    """Whether the step column is the training-step axis, checked against the publication.

    The check this replaces compared the table's step column against the table's file column, both
    read from the same table. That is an internal consistency check wearing a step-axis name, and it
    has a failure mode worse than a false negative: on a run whose index is missing an entry, the
    table has as many distinct indices as it has distinct file names, so the counts agree, the check
    passes, and the defect is invisible. The companion AISI run is that case. It publishes four
    hundred and four eval files, its table carries four hundred and three distinct indices with one
    value absent from the middle of an otherwise contiguous range, and the old check saw four
    hundred and three against four hundred and three.

    So this one takes the count of files the **repository** publishes and compares against that,
    and it reports the gaps by value rather than as a count, because which index is missing decides
    whether the gap displaces every step after it or merely puts a hole in the series.
    """
    steps = _as_float(_column(table, columns.step))
    files = np.unique(_column(table, columns.file))
    gaps = index_gaps(steps)
    report = StepAxisReport(
        n_distinct_steps=int(gaps["n_distinct"]),
        n_distinct_files_in_table=int(files.size),
        n_published_files=n_published_files,
        contiguous=bool(gaps["contiguous"]),
        missing=tuple(int(v) for v in gaps["missing"]),
        low=gaps["low"],
        high=gaps["high"],
    )
    if report.n_distinct_steps != report.n_distinct_files_in_table:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"the step column {columns.step!r} has {report.n_distinct_steps} distinct values "
                f"and the file column {columns.file!r} has {report.n_distinct_files_in_table}. In "
                f"a published rollout table the step index is per eval file, so this column is a "
                f"row counter or the files are not one per step."
            ),
            remedy=(
                f"Build the series from {columns.file!r} sorted by name instead, and state that "
                f"the step axis is inferred from file order."
            ),
            statistics={
                "steps": report.n_distinct_steps,
                "files": report.n_distinct_files_in_table,
            },
        )
    if n_published_files is not None and report.n_distinct_steps != n_published_files:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the repository publishes {n_published_files} eval files and the table carries "
                f"{report.n_distinct_steps} distinct step indices"
                + (
                    f", with {len(report.missing)} absent from the range "
                    f"{report.low} to {report.high}: {list(report.missing)[:16]}"
                    if report.missing
                    else " over a contiguous range"
                )
                + ". One published file has no rows in the table, so the file-to-step mapping "
                "cannot be right everywhere on this run."
            ),
            remedy=(
                "Establish which file is unrepresented and whether the index was assigned before "
                "or after it was dropped. Until that is settled, anything that assumes a "
                "contiguous index on this run is wrong, and the alignment established on the head "
                "of the series does not extend past the gap."
            ),
            statistics={
                "n_published_files": n_published_files,
                "n_distinct_steps": report.n_distinct_steps,
                "missing": list(report.missing),
                "low": report.low,
                "high": report.high,
            },
        )
    return report


def label_rate(
    table: Mapping[str, Any],
    columns: RolloutColumns,
    label: str | None = None,
) -> list[LabelRate]:
    """The labelled positive rate per step, with the denominator and the null count beside it.

    Nulls are excluded from the denominator and counted, never read as zeros. A step where every
    label is null has a rate of NaN and a denominator of zero, which is different from a step where
    every rollout was scored and none was positive, and the two must not render the same.
    """
    key = label or columns.label
    steps = _as_float(_column(table, columns.step))
    values = _as_float(_column(table, key))
    out: list[LabelRate] = []
    for step in np.unique(steps[np.isfinite(steps)]):
        mask = steps == step
        block = values[mask]
        scored = block[np.isfinite(block)]
        out.append(
            LabelRate(
                step=int(step),
                rate=float(scored.mean()) if scored.size else float("nan"),
                n_labelled=int(scored.size),
                n_null=int(block.size - scored.size),
            )
        )
    return out


def steps_from_table(
    table: Mapping[str, Any],
    run: ImportedRun,
    *,
    reward: str = "operational",
    min_group: int = 2,
    groups: PromptGroups | None = None,
    strict_groups: bool = True,
) -> list[StepSample] | Refusal:
    """Every step of a per-rollout table, featurised, with advantages reconstructed per group.

    ``run`` is required and carries the four determinations. There is no column-mapping argument and
    no ``std_normalised`` flag, because both were ways of stating a determination without recording
    where it came from.

    ``reward`` is ``"operational"`` or ``"proxy"`` and it is the second required decision. Under
    ``"operational"`` this returns whatever `ComposedReward.operational` returns, which on a run
    with a binarised component is a `RECORD_INCOMPLETE` refusal naming the column and the
    resolution that would close it. Under ``"proxy"`` it builds on the best composite the published
    columns support and every `StepSample` carries, in its own ``detail``, the sentence saying so.
    A caller writing ``reward="proxy"`` has said out loud that the numbers downstream are not
    denominated in the trainer's reward, which is the whole difference between this and a default.

    Groups are the optimiser's, recovered from contiguous row blocks under `run.generations` and
    keyed on ``(step, group)``. Rollouts whose text produces no features are dropped and counted;
    rollouts in a group too small to carry within-group spread keep their row and carry a NaN
    advantage, because they were still sampled from the policy and still belong in `z`. Those are
    two different numbers and ``detail`` reports both.
    """
    if reward not in ("operational", "proxy"):
        raise ValueError(
            f"reward must be 'operational' or 'proxy', not {reward!r}. There is no third option "
            f"and there is deliberately no default that picks one."
        )

    convention = run.convention
    blocked = convention.requires(instrument=f"{run.name}.steps_from_table")
    if blocked is not None:
        return blocked

    got_groups = groups if groups is not None else run.prompt_groups(table, strict=strict_groups)
    if isinstance(got_groups, Refusal):
        return got_groups

    if reward == "operational":
        rewards_or_refusal = run.operational_reward(table)
        if isinstance(rewards_or_refusal, Refusal):
            return rewards_or_refusal
        rewards = rewards_or_refusal
        reward_detail = "the operational reward, every component at the resolution the scorer used"
    else:
        proxy = run.proxy_reward(table)
        if isinstance(proxy, Refusal):
            return proxy
        assert isinstance(proxy, ProxyReward)
        rewards = proxy.values
        reward_detail = proxy.detail

    steps = _as_float(_column(table, run.columns.step))
    texts = _column(table, run.columns.text)
    semantic = (
        _column(table, run.columns.semantic_id).astype(str)
        if run.columns.semantic_id
        else np.asarray([""] * steps.size, dtype=object)
    )
    std_normalised = convention.std_normalised
    ddof = convention.std_ddof if std_normalised else None
    epsilon = convention.std_epsilon if std_normalised else 0.0

    out: list[StepSample] = []
    for step in np.unique(steps[np.isfinite(steps)]):
        mask = steps == step
        idx = np.flatnonzero(mask)
        rows: list[list[float]] = []
        keep: list[int] = []
        for i in idx:
            text = texts[i]
            values = surface_features("" if text is None else str(text), 1)
            if values is None:
                continue
            rows.append([values[n] for n in SURFACE_NAMES])
            keep.append(int(i))
        dropped = int(idx.size - len(keep))
        if not keep:
            continue
        kept = np.asarray(keep, dtype=np.intp)
        # The group code is the optimiser's own block ordinal, so it is an integer that already
        # means something rather than a rank over sorted strings. Sorting string labels was safe
        # while the group key was a column value and is not safe once the group is a run of rows.
        group_ids = np.asarray(got_groups.group_id[kept], dtype=np.int64)
        sizes = {int(g): int(np.sum(group_ids == g)) for g in np.unique(group_ids)}
        big = np.asarray([sizes[int(g)] >= min_group for g in group_ids], dtype=bool)
        advantages = advantages_from_rewards(
            rewards[kept],
            group_ids,
            std_epsilon=epsilon,
            std_normalised=std_normalised,
            std_ddof=ddof,
        )
        advantages = np.where(big, advantages, np.nan)
        n_nan = int(np.count_nonzero(~np.isfinite(advantages)))
        arithmetic = (
            f"(r - mean_g) / (std_g[ddof={ddof}] + {epsilon:g})" if std_normalised else "r - mean_g"
        )
        out.append(
            StepSample(
                index=int(step),
                names=SURFACE_NAMES,
                features=np.asarray(rows, dtype=np.float64),
                advantages=advantages,
                group_ids=group_ids,
                # The semantic problem, kept as the task identity it is. The optimiser's group is
                # `group_ids` and the two are different fields on purpose: task overlap between
                # steps is a question about which problems were drawn, and the group is a question
                # about what the trainer normalised within.
                task_ids=tuple(str(g) for g in semantic[kept]),
                advantage_source="reconstructed",
                n_dropped=dropped,
                detail=(
                    f"{len(keep)} rollouts over {len(sizes)} optimiser groups; advantages "
                    f"reconstructed as {arithmetic} on {reward_detail}"
                    + (f"; {dropped} carried no response text" if dropped else "")
                    + (
                        f"; {n_nan} carry no advantage because their group holds fewer than "
                        f"{min_group} rollouts"
                        if n_nan
                        else ""
                    )
                ),
            )
        )
    return out


def read_parquet(path: str) -> dict[str, np.ndarray]:
    """A parquet file as a column mapping, through pandas, which is already a core dependency.

    Kept to one function so that the acceptance test's skip condition is "this file is not here"
    rather than "this environment cannot read parquet". `pyarrow` ships in the ``[record]`` extra and
    pandas needs it for this call; nothing else in the ledger touches either.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    return {name: frame[name].to_numpy() for name in frame.columns}


def parse_hack_config(value: Any) -> dict[str, Any] | None:
    """`hack_config` is a JSON string, not a struct. Parsed only when asked for, never implicitly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def rate_series(rates: Sequence[LabelRate]) -> tuple[list[int], list[float]]:
    """``(steps, rates)`` for the steps that carried at least one scored label."""
    usable = [r for r in rates if r.n_labelled > 0]
    return [r.step for r in usable], [r.rate for r in usable]


__all__ = [
    "AISI_RUN",
    "LabelRate",
    "StepAxisReport",
    "check_step_axis",
    "label_rate",
    "parse_hack_config",
    "rate_series",
    "read_parquet",
    "steps_from_table",
]
