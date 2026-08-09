"""Reading an Inspect AI `.eval` archive, so a parsed convenience table is never the only source.

This module exists because of a specific failure, and the failure is worth stating because it is
what the module is shaped to prevent.

A published reinforcement-learning artifact shipped two things: a parquet of one row per rollout,
and a directory of Inspect `.eval` archives holding the raw evaluation logs. The dataset card
documented both, under a heading reading "Two ways to use this dataset". The parquet carried a
column named `thinking_format_ok` as a binary `int64`. The underlying scorer returned a sum of four
quarter increments, so the parquet's column was the indicator of a component rather than the
component. Every analysis this project ran on that artifact inherited the binarisation, four
documents were written around the loss, a letter was drafted asking the publisher for the missing
field, and the field was in the second directory the whole time.

The parquet was not wrong. `thinking_format_ok` reads exactly as its name says. What was missing was
anybody opening the other documented way to use the data.

So this reader is deliberately not a convenience wrapper over a convenience table. It reads the
archive, it reports what the archive carries rather than what the caller hoped for, and where a
field is absent it returns a typed refusal naming the field rather than a default.

**Why it does not import `inspect_ai`.** An `.eval` file is a ZIP of JSON. Reading it directly costs
about eighty lines and adds no dependency, and more importantly it means the reader still works when
the archive was written by a version of Inspect that the installed one cannot load. The archives this
was built against were written by `0.3.190.dev92+ge1b3ff76`, a development build. A reader that
refuses to open an artifact because it cannot resolve a pinned dependency is a reader that sends
somebody back to the parsed table.

**The grade mapping is verified rather than assumed, and that is the load-bearing part.** Inspect
scores may be numeric or may be single-character grades, where `C` conventionally means correct and
`I` incorrect. Mapping `C` to 1 and `I` to 0 is the obvious choice and an obvious choice is not a
checked one. Every complete archive carries a `reductions.json` holding Inspect's own reduction of
each metric across epochs, computed by the harness that wrote the grades. `verify_reductions`
recomputes those reductions from the per-rollout values under the assumed mapping and compares. If
the mapping were wrong the reductions would not reproduce. On the artifact this was built against
they reproduce bit-exactly on all 972 reductions of one archive, which turns the mapping from a
convention into a measurement against the file's own arithmetic.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reward_lens.core.reading import Refusal, refuse_incomplete

#: The Inspect grade alphabet. `C` correct, `I` incorrect, `P` partial, `N` no credit.
#:
#: `P` is mapped to 0.5 because that is Inspect's own convention for a partial grade, and it is
#: recorded here rather than inline so a caller who disagrees can pass their own map and so the
#: disagreement is visible in a diff. Verify it with `verify_reductions` before relying on it: on an
#: archive whose scorers never emit `P`, this entry is untested by construction.
GRADE_VALUES: Mapping[str, float] = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}

_HEADER = "header.json"
_REDUCTIONS = "reductions.json"
_SUMMARIES = "summaries.json"
_SAMPLES = "samples/"


@dataclass(frozen=True)
class InspectHeader:
    """The provenance an archive carries about the run that produced it.

    Every field here is optional because an archive may be truncated. The companion run of the
    artifact this was built against publishes one archive of 1,186 bytes containing nothing but
    `_journal/start.json`: an evaluation that started and wrote no samples. A reader that assumed
    `header.json` exists would raise on it, and the right behaviour is to report an archive with no
    header as exactly that, because it is a fact about the run and not an error in the reader.
    """

    eval_id: str | None = None
    run_id: str | None = None
    created: str | None = None
    task: str | None = None
    model: str | None = None
    #: The publisher's source repository and commit, when the harness recorded one. This is often
    #: the only pointer from a published artifact back to the code that produced it.
    revision: Mapping[str, Any] | None = None
    task_args: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    model_generate_config: Mapping[str, Any] = field(default_factory=dict)
    packages: Mapping[str, Any] = field(default_factory=dict)
    #: One entry per scorer, each declaring the metric names it reports.
    scorers: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    @property
    def epochs(self) -> int | None:
        """Generations per prompt, when the config declares it."""
        value = self.config.get("epochs")
        return int(value) if isinstance(value, int) else None

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """The prompt identifiers, when the dataset block declares them."""
        ids = self.dataset.get("sample_ids")
        return tuple(str(i) for i in ids) if isinstance(ids, (list, tuple)) else ()

    @property
    def scorer_metrics(self) -> tuple[tuple[str, str], ...]:
        """Every `(scorer, metric)` pair the header declares.

        A scorer reporting a single scalar declares its metrics as a list; one reporting several
        named values declares them as a mapping keyed by metric name. Both shapes appear in the same
        header, and flattening them here is what lets a caller count the reward functions without
        opening a sample. On the artifact this was built against, four scorers declare nine metrics
        between them, which is the number a trainer log's ratio turned out to be denominated in.
        """
        out: list[tuple[str, str]] = []
        for scorer in self.scorers:
            name = str(scorer.get("name", ""))
            metrics = scorer.get("metrics")
            if isinstance(metrics, Mapping):
                out.extend((name, str(k)) for k in metrics)
            else:
                out.append((name, name))
        return tuple(out)


@dataclass(frozen=True)
class InspectSample:
    """One rollout: its scores, its problem metadata, and the generation the trainer saw."""

    sample_id: str
    epoch: int
    #: Flattened `{metric_name: value}` across every scorer, grades already mapped to numbers.
    scores: Mapping[str, float]
    #: Per-scorer free text, when the scorer wrote any. The static hack detection of the artifact
    #: this was built against is published here and nowhere else.
    explanations: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Token ids the trainer itself held, when the generation stack recorded them.
    prompt_ids: tuple[int, ...] = ()
    completion_ids: tuple[int, ...] = ()
    #: Sampling-stack log probabilities, one per completion token. These are the sampling stack's,
    #: not the training stack's, and the distinction matters: an importance ratio needs both.
    sampling_logprobs: tuple[float, ...] = ()
    completion_text: str | None = None
    stop_reason: str | None = None

    @property
    def n_completion_tokens(self) -> int | None:
        """Completion length in the trainer's own tokens, or `None` if the archive lacks ids.

        Length in characters is not a substitute. Measuring completion length by character count
        rather than by token count has already produced one wrong published negative in this
        project, so the property returns `None` rather than falling back.
        """
        return len(self.completion_ids) or None


@dataclass(frozen=True)
class InspectEvalLog:
    """A whole archive: its header, its rollouts, and what it was missing."""

    path: Path
    header: InspectHeader | None
    samples: tuple[InspectSample, ...]
    #: Non-fatal absences, so a caller can tell a truncated archive from a complete one without
    #: re-opening it.
    missing: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.header is not None and bool(self.samples)


def _grade(value: Any, grades: Mapping[str, float]) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value in grades:
            return grades[value]
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def _flatten_scores(
    block: Mapping[str, Any], grades: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    """Flatten Inspect's `scores` block to `{metric: number}` plus `{scorer: explanation}`.

    A scorer's `value` is either a scalar or a mapping of named sub-metrics. The flattened key is
    the scorer name in the first case and the sub-metric name in the second, which is what makes the
    result join cleanly against a configuration that lists reward functions in one flat vector.
    """
    values: dict[str, float] = {}
    explanations: dict[str, str] = {}
    for scorer_name, entry in block.items():
        if not isinstance(entry, Mapping):
            continue
        raw = entry.get("value")
        if isinstance(raw, Mapping):
            for metric_name, metric_value in raw.items():
                values[str(metric_name)] = _grade(metric_value, grades)
        else:
            values[str(scorer_name)] = _grade(raw, grades)
        explanation = entry.get("explanation")
        if isinstance(explanation, str) and explanation:
            explanations[str(scorer_name)] = explanation
    return values, explanations


def read_header(
    path: str | Path, *, grades: Mapping[str, float] = GRADE_VALUES
) -> InspectHeader | None:
    """The archive's header, or `None` if it has none. Cheap: opens one entry."""
    del grades
    with zipfile.ZipFile(Path(path)) as archive:
        if _HEADER not in archive.namelist():
            return None
        raw = json.loads(archive.read(_HEADER))
    block = raw.get("eval", {}) if isinstance(raw, Mapping) else {}
    return InspectHeader(
        eval_id=block.get("eval_id"),
        run_id=block.get("run_id"),
        created=block.get("created"),
        task=block.get("task"),
        model=block.get("model"),
        revision=block.get("revision"),
        task_args=block.get("task_args") or {},
        dataset=block.get("dataset") or {},
        config=block.get("config") or {},
        model_generate_config=block.get("model_generate_config") or {},
        packages=block.get("packages") or {},
        scorers=tuple(block.get("scorers") or ()),
    )


def read_eval(
    path: str | Path,
    *,
    grades: Mapping[str, float] = GRADE_VALUES,
    with_logprobs: bool = True,
    with_text: bool = True,
) -> InspectEvalLog:
    """Read one archive whole.

    `with_logprobs` and `with_text` are off-switches for the two fields that dominate the memory
    cost. A run of four hundred archives at sixty-four rollouts each carries roughly ten million
    log probabilities and the same number of token ids, and a caller who wants the scores does not
    want to pay for them.
    """
    path = Path(path)
    missing: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        header = read_header(path) if _HEADER in names else None
        if header is None:
            missing.append(_HEADER)
        for expected in (_REDUCTIONS, _SUMMARIES):
            if expected not in names:
                missing.append(expected)
        sample_names = sorted(n for n in names if n.startswith(_SAMPLES) and n.endswith(".json"))
        if not sample_names:
            missing.append(_SAMPLES)
        samples: list[InspectSample] = []
        for name in sample_names:
            raw = json.loads(archive.read(name))
            values, explanations = _flatten_scores(raw.get("scores") or {}, grades)
            output = raw.get("output") or {}
            choices = output.get("choices") or [{}]
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            completion = ((output.get("metadata") or {}).get("trl_completion_data") or [{}])[0]
            logprobs: tuple[float, ...] = ()
            if with_logprobs:
                raw_lp = completion.get("logprobs")
                if isinstance(raw_lp, list):
                    logprobs = tuple(float(x) for x in raw_lp)
                else:
                    content = (choice.get("logprobs") or {}).get("content") or []
                    logprobs = tuple(float(c["logprob"]) for c in content if "logprob" in c)
            text = message.get("content") if with_text else None
            samples.append(
                InspectSample(
                    sample_id=str(raw.get("id")),
                    epoch=int(raw.get("epoch", 0)),
                    scores=values,
                    explanations=explanations,
                    metadata=raw.get("metadata") or {},
                    prompt_ids=tuple(completion.get("prompt_ids") or ()) if with_logprobs else (),
                    completion_ids=tuple(completion.get("completion_ids") or ()),
                    sampling_logprobs=logprobs,
                    completion_text=text if isinstance(text, str) else None,
                    stop_reason=choice.get("stop_reason"),
                )
            )
    return InspectEvalLog(path=path, header=header, samples=tuple(samples), missing=tuple(missing))


def verify_reductions(
    path: str | Path,
    *,
    grades: Mapping[str, float] = GRADE_VALUES,
    tolerance: float = 1e-12,
) -> dict[str, Any] | Refusal:
    """Recompute the archive's own reductions from the per-rollout values, and compare.

    This is the check that turns the grade mapping from a convention into a measurement. Inspect
    writes `reductions.json` itself, from the same grades, so recomputing it under an assumed
    mapping and getting the same numbers is evidence the mapping is the one the harness used. A
    mismatch means the mapping is wrong, or the reducer is not the mean, and either way the caller
    should know before treating a grade as a number.

    Returns a report with the comparison counts, or a `RECORD_INCOMPLETE` refusal if the archive
    carries no reductions to check against. It never returns a pass by default: an archive without
    reductions is unverified, not verified.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if _REDUCTIONS not in archive.namelist():
            return refuse_incomplete(
                "inspect_eval.verify_reductions",
                field=_REDUCTIONS,
                subject=str(path),
                remedy=(
                    "This archive carries no reductions block, so the grade mapping cannot be "
                    "checked against the harness's own arithmetic. Verify the mapping on a complete "
                    "archive from the same run and pass the verified map explicitly."
                ),
            )
        reductions = json.loads(archive.read(_REDUCTIONS))
    log = read_eval(path, grades=grades, with_logprobs=False, with_text=False)
    observed: dict[tuple[str, str], list[float]] = {}
    for sample in log.samples:
        for metric, value in sample.scores.items():
            observed.setdefault((metric, sample.sample_id), []).append(value)

    entries = reductions if isinstance(reductions, list) else reductions.get("reductions", [])
    checked = 0
    mismatches: list[dict[str, Any]] = []
    max_abs = 0.0
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        metric = str(entry.get("scorer") or entry.get("name") or "")
        reducer = str(entry.get("reducer") or "mean")
        for item in entry.get("samples") or ():
            if not isinstance(item, Mapping):
                continue
            key = (metric, str(item.get("sample_id")))
            values = observed.get(key)
            if values is None:
                inner = item.get("value")
                if isinstance(inner, Mapping):
                    for sub, sub_value in inner.items():
                        sub_values = observed.get((str(sub), str(item.get("sample_id"))))
                        if sub_values is None:
                            continue
                        checked += 1
                        delta = abs(sum(sub_values) / len(sub_values) - _grade(sub_value, grades))
                        max_abs = max(max_abs, delta)
                        if delta > tolerance:
                            mismatches.append(
                                {"metric": sub, "sample_id": item.get("sample_id"), "delta": delta}
                            )
                continue
            if reducer != "mean":
                continue
            checked += 1
            delta = abs(sum(values) / len(values) - _grade(item.get("value"), grades))
            max_abs = max(max_abs, delta)
            if delta > tolerance:
                mismatches.append(
                    {"metric": metric, "sample_id": item.get("sample_id"), "delta": delta}
                )
    return {
        "path": str(path),
        "n_checked": checked,
        "n_mismatch": len(mismatches),
        "max_abs_difference": max_abs,
        "grades": dict(grades),
        "mismatches": mismatches[:20],
        "verified": checked > 0 and not mismatches,
    }


def iter_scores(
    paths: Sequence[str | Path],
    *,
    grades: Mapping[str, float] = GRADE_VALUES,
) -> Iterator[dict[str, Any]]:
    """One flat row per rollout across many archives, for building a table.

    Rows carry `source_eval_file` as a basename rather than a path, because that is how published
    rollout tables tend to store it and a join that needs a string transform on one side is a join
    somebody gets wrong once.
    """
    for index, path in enumerate(paths):
        path = Path(path)
        log = read_eval(path, grades=grades, with_logprobs=True, with_text=False)
        commit = (log.header.revision or {}).get("commit") if log.header else None
        for sample in log.samples:
            row: dict[str, Any] = {
                "file_index": index,
                "source_eval_file": path.name,
                "sample_id": sample.sample_id,
                "epoch": sample.epoch,
                "n_prompt_tokens": len(sample.prompt_ids) or None,
                "n_completion_tokens": sample.n_completion_tokens,
                "stop_reason": sample.stop_reason,
                "git_commit": commit,
            }
            row.update(sample.scores)
            for key in ("problem_id", "difficulty", "cf_rating", "test_count", "hack_group"):
                if key in sample.metadata:
                    row[key] = sample.metadata[key]
            yield row
