"""A published `trainer_state.json`, read as named series without assuming what the names are.

A checkpoint's `log_history` is the only authority for what a trainer actually logged, and it is the
only place a reconstruction of the reward can be closed against. Reading it looks trivial and is
not, for one reason: the key names are a property of the trainer version that wrote the file, not of
the specification anybody read. TRL has renamed reward metrics more than once, and a module that
reaches for ``entry["reward"]`` and falls back to something plausible when it is absent will report
a series that is quietly the wrong quantity.

So this module inverts the usual order. It records the keys verbatim first, with how many entries
carry each, and every series is fetched by a key the caller has seen in that listing. A key that is
not there produces a `Refusal` with `RECORD_INCOMPLETE`, never a substitute.

**Generally useful beyond the run it was written for.** Nothing here knows about the AI Safety
Institute's artifacts, about GRPO, or about reward hacking. Any Hugging Face `Trainer` checkpoint
has this file with this shape, so anyone holding one can get its telemetry out under the same rule.

`quantisation_step` is the other half of the same care. A logged scalar usually lives on a grid,
either because the trainer rounded it or because it is a mean of a bounded discrete quantity over a
fixed batch size, and a residual judged against a tolerance finer than that grid fails a
reconstruction that in fact closed exactly. The grid is a measurable property of the series and it
is cheaper to measure than to argue about.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason

#: The step key every Hugging Face `Trainer` writes. Unlike the metric keys it has not moved, and a
#: file that lacks it is not a `trainer_state.json`, so this one is allowed to be a constant.
STEP_KEY = "step"


@dataclass(frozen=True)
class KeyCensus:
    """Which keys `log_history` carries, verbatim, and how many entries carry each.

    The count matters as much as the name. A `Trainer` interleaves training entries with evaluation
    entries and a final summary entry, so a key present in 400 of 404 entries is a training metric
    with four foreign rows beside it, and a key present in 1 of 404 is a summary. Reading the union
    of keys without the counts makes those three look like the same thing.
    """

    counts: Mapping[str, int]
    n_entries: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts))

    def carried_by_all(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, n in self.counts.items() if n == self.n_entries))

    def rare(self, threshold: float = 0.5) -> tuple[str, ...]:
        """Keys on fewer than `threshold` of the entries, which is where summary rows show up."""
        limit = threshold * self.n_entries
        return tuple(sorted(k for k, n in self.counts.items() if n < limit))


@dataclass(frozen=True)
class Series:
    """One logged scalar as `(steps, values)`, with the key it was read under kept beside it."""

    key: str
    steps: np.ndarray
    values: np.ndarray

    @property
    def n(self) -> int:
        return int(self.steps.size)


@dataclass(frozen=True)
class TrainerLog:
    """`log_history` from a checkpoint, plus the rest of the state file's scalars.

    Holds the entries as read. Every accessor below is explicit about what it assumes, and the two
    that assume anything at all say so in their own docstrings.
    """

    entries: tuple[Mapping[str, Any], ...]
    state: Mapping[str, Any]
    path: str = ""

    @property
    def census(self) -> KeyCensus:
        counter: Counter[str] = Counter()
        for entry in self.entries:
            counter.update(entry.keys())
        return KeyCensus(counts=dict(counter), n_entries=len(self.entries))

    def first(self) -> Mapping[str, Any]:
        """`log_history[0]` verbatim. The gate document asks for this by name and it is why."""
        if not self.entries:
            raise ValueError(f"{self.path or 'this trainer state'} has an empty log_history")
        return dict(self.entries[0])

    def series(self, key: str) -> Series | Refusal:
        """The `(step, value)` pairs for one key, in step order, or a refusal naming what is there.

        Entries that do not carry the key are skipped rather than filled, and entries that carry it
        as a non-number are skipped and counted, because a string in a numeric series is a different
        failure from an absent one and coercing both to NaN would render them the same.
        """
        census = self.census
        if key not in census.counts:
            return Refusal(
                instrument="trainer_log.series",
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"log_history carries no key {key!r}. It carries {census.names}, of which "
                    f"{census.carried_by_all()} appear on every entry."
                ),
                remedy=(
                    f"Read the series under one of the keys this file actually has, or re-export "
                    f"the checkpoint from a trainer version that logs {key!r}. Do not substitute a "
                    f"neighbouring key: the closure targets are only meaningful against the "
                    f"quantity the trainer named."
                ),
                statistics={"requested": key, "available": list(census.names)},
            )
        steps: list[float] = []
        values: list[float] = []
        skipped_no_step = 0
        skipped_not_a_number = 0
        for entry in self.entries:
            if key not in entry:
                continue
            if STEP_KEY not in entry:
                skipped_no_step += 1
                continue
            raw = entry[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                skipped_not_a_number += 1
                continue
            if not math.isfinite(float(raw)):
                skipped_not_a_number += 1
                continue
            steps.append(float(entry[STEP_KEY]))
            values.append(float(raw))
        if not steps:
            return Refusal(
                instrument="trainer_log.series",
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"key {key!r} appears on {census.counts[key]} entries and none of them "
                    f"produced a usable (step, value) pair: {skipped_no_step} carried no "
                    f"{STEP_KEY!r} and {skipped_not_a_number} carried a non-finite or non-numeric "
                    f"value."
                ),
                remedy=(
                    "Check whether this key is a summary field rather than a per-step metric. A "
                    "summary belongs in `state`, not in a series."
                ),
                statistics={
                    "key": key,
                    "entries_with_key": census.counts[key],
                    "skipped_no_step": skipped_no_step,
                    "skipped_not_a_number": skipped_not_a_number,
                },
            )
        order = np.argsort(np.asarray(steps, dtype=np.float64), kind="stable")
        return Series(
            key=key,
            steps=np.asarray(steps, dtype=np.float64)[order],
            values=np.asarray(values, dtype=np.float64)[order],
        )

    def require(self, key: str) -> Series:
        """`series`, raising instead of refusing. For callers that have already checked the census."""
        got = self.series(key)
        if isinstance(got, Refusal):
            raise KeyError(got.detail)
        return got


def read_trainer_state(path: str | Path) -> TrainerLog:
    """A `trainer_state.json` from disk. Nothing is renamed, coerced or dropped on the way in."""
    text = Path(path).read_text(encoding="utf-8")
    state = json.loads(text)
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} does not hold a JSON object")
    history = state.get("log_history", [])
    if not isinstance(history, Sequence):
        raise ValueError(f"{path} has a log_history that is not a list")
    entries = tuple(dict(e) for e in history if isinstance(e, Mapping))
    scalars = {k: v for k, v in state.items() if k != "log_history"}
    return TrainerLog(entries=entries, state=scalars, path=str(path))


def quantisation_step(
    values: Sequence[float] | np.ndarray,
    *,
    max_denominator: int = 4096,
    atol: float = 1e-9,
) -> float:
    """The coarsest grid `q` on which every value lands, or 0.0 when there is no such grid.

    Searched over `1/d` for integer `d` up to `max_denominator`, coarsest first, so the answer is
    the largest step that explains the data rather than the smallest one that trivially does. A
    series of arbitrary floats returns 0.0, which reads as "no grid" and is the honest answer for a
    quantity the trainer did not round.

    Why this is worth measuring rather than assuming. A mean of `n` rollout rewards, each of which
    is a multiple of `u`, lands on a grid of `u / n` exactly, with no rounding anywhere. So the grid
    of a logged mean carries information about the per-rollout reward's own lattice, and that is a
    constraint on the composition before a single weight is fitted.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    for denominator in range(1, max_denominator + 1):
        q = 1.0 / denominator
        scaled = arr / q
        if np.all(np.abs(scaled - np.round(scaled)) <= atol * np.maximum(1.0, np.abs(scaled))):
            return q
    return 0.0


def rounding_rms(q: float) -> float:
    """The RMS residual a pure rounding onto a grid of `q` produces, which is `q / sqrt(12)`.

    The number a closure residual is compared against when the question is whether a reconstruction
    failed or merely met the floor the log was written at.
    """
    return float(q) / math.sqrt(12.0) if q > 0 else 0.0


__all__ = [
    "KeyCensus",
    "STEP_KEY",
    "Series",
    "TrainerLog",
    "quantisation_step",
    "read_trainer_state",
    "rounding_rms",
]
