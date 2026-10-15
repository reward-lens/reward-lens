"""TRL's completions parquet as a `TraceRecord`, second in D-42's support order.

The primary path needs no instrumentation of the user's trainer: TRL already writes this file when
`log_completions` is on. What it does not write is a group column, and consecutive rows of the file
are the generations of one prompt, which is exactly the adjacency a reader must not read as a
grouping. So a run whose groups were never logged reads with `group_id=None` on every row, the
`trace.group_not_declared` rule fires, and `strict` refuses while `permissive` carries the
violation. A user who wants the grouping logs it with `log_extra` and names the column here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from reward_lens.record.convert import trl_parquet

from .integrity import CheckInput, enforce
from .record import (
    SERIES_NAMES,
    DeclaredReward,
    Policy,
    Series,
    TraceMetadata,
    TraceRecord,
    TraceRow,
)

__all__ = ["READER", "TrlCompletionsReader"]

_STATEMENT = (
    "RECORD: the values the trainer logged, and nothing else. That supports a reconstruction of "
    "what the optimiser saw and the selection evidence in it; it cannot support anything that "
    "needs the grader called again"
)


class TrlCompletionsReader:
    """One run directory of `completions_*.parquet`, read without importing `trl`."""

    name = "trl_completions"

    def can_read(self, path: Any) -> bool:
        try:
            return bool(trl_parquet.completions_files(Path(path)))
        except OSError:  # pragma: no cover - an unreadable directory has no opinion
            return False

    def read(
        self,
        path: Any,
        *,
        policy: Policy = "strict",
        reward_functions: Sequence[str] | None = None,
        extra_columns: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
        group_column: str | None = None,
        group_size: int | None = None,
    ) -> TraceRecord:
        root = Path(path)
        tables = trl_parquet.read_directory(
            root,
            reward_functions=reward_functions,
            extra_columns=extra_columns,
            weights=weights,
        )
        if not tables:
            raise FileNotFoundError(f"no completions parquet under {root}")
        names: list[str] = []
        for table in tables:
            for one in table.reward_functions:
                if one not in names:
                    names.append(one)
        per_component: dict[str, list[float | None]] = {name: [] for name in names}
        rows: list[TraceRow] = []
        index = 0
        for table in tables:
            totals = table.totals()
            advantages = table.advantages()
            for position, raw in enumerate(table.rows):
                values = {name: trl_parquet._number(raw.get(name)) for name in names}
                for name in names:
                    per_component[name].append(values[name])
                group = raw.get(group_column) if group_column else None
                rows.append(
                    TraceRow(
                        index=index,
                        step=int(raw.get("step", 0)),
                        group_id=None if group is None else str(group),
                        row_id=f"{Path(table.path).name}:{position}",
                        components=values,
                        reward_recorded=totals[position],
                        reward_applied=totals[position],
                        advantage=advantages[position],
                        selection=None,
                        task_id=(
                            _identity("prompt", raw.get("prompt"))
                            if raw.get("prompt") is not None
                            else None
                        ),
                        response_digest=(
                            _identity("completion", raw.get("completion"))
                            if raw.get("completion") is not None
                            else None
                        ),
                        response_codec=(
                            "rfc8785-json/1" if raw.get("completion") is not None else None
                        ),
                    )
                )
                index += 1

        vector = tables[0].weight_vector()
        components = {
            name: Series(
                name=name,
                values=tuple(per_component[name]),
                definition=f"the column TRL wrote under the reward function's own name, {name!r}",
                provenance={
                    "kind": "raw_native_component_score",
                    "source": "TRL completions parquet column",
                    "reader": self.name,
                    "column": name,
                },
            )
            for name in names
        }
        series = _series(rows, names, vector, bool(weights), group_column)
        results = enforce(
            CheckInput(
                rows=tuple(rows),
                series=series,
                components=components,
                declared_rows=len(rows),
                declared_group_size=group_size,
                source=str(root),
            ),
            policy=policy,
        )
        return TraceRecord(
            reader=self.name,
            source=str(root),
            run_id=str(root.name),
            rows=tuple(rows),
            series=series,
            components=components,
            access_level="RECORD",
            access_statement=_STATEMENT,
            policy=policy,
            complete=True,
            completeness_statement=(
                "every completions file present when the directory was listed was read; a run "
                "still writing will have more"
            ),
            subject_ref=_subject_ref(tables),
            depends_on=("digest:training_semantics", "digest:samples", "digest:policy"),
            framework="trl",
            framework_version=trl_parquet.TRL_VERSION,
            integrity=results,
            limitations=(
                ()
                if group_column
                else (
                    "the file logs no grouping, so no group-level quantity is available; "
                    "adjacency is not a grouping",
                )
            ),
            convenience={
                "declared_weighted_component_sum": Series(
                    name="declared_weighted_component_sum",
                    values=tuple(table_value for table in tables for table_value in table.totals()),
                    weights=vector,
                    components=tuple(names),
                    definition=(
                        "a convenience aggregate reconstructed from the caller-declared vector; "
                        "TRL does not write an aggregate column"
                    ),
                    provenance={
                        "kind": "derived_convenience_aggregate",
                        "source": "TRL completions parquet columns",
                        "reader": self.name,
                    },
                ),
                **_raw_reward_convenience(tables),
            },
            metadata=TraceMetadata(
                declared_reward=DeclaredReward(
                    partition=tuple(names),
                    weights=dict(weights) if weights is not None else None,
                    grouping=(
                        {"column": group_column, "declared_size": group_size}
                        if group_column is not None
                        else None
                    ),
                    provenance={
                        "reader": self.name,
                        "partition": "caller-declared reward_functions",
                        "weights": "caller-declared weights"
                        if weights is not None
                        else "not declared",
                        "grouping": "caller-declared group_column"
                        if group_column
                        else "not declared",
                    },
                )
            ),
        )


def _series(
    rows: list[TraceRow],
    names: list[str],
    vector: tuple[float, ...],
    declared: bool,
    group_column: str | None,
) -> dict[str, Series]:
    ordered = {
        "reward_recorded": Series(
            name="reward_recorded",
            values=tuple(None for _ in rows),
            components=tuple(names),
            definition=(
                "the displayed aggregate is unavailable: TRL writes no aggregate column, and "
                "the batch means in its metric stream are not per-completion totals"
            ),
            present=False,
            absent_reason=(
                "the TRL completions parquet has no per-completion displayed aggregate; its "
                "metric-stream batch means cannot be substituted"
            ),
        ),
        "reward_applied": Series(
            name="reward_applied",
            values=tuple(row.reward_applied if declared else None for row in rows),
            weights=vector if declared else None,
            components=tuple(names),
            definition=(
                "the optimised composite under the declared weight vector"
                if declared
                else "the optimised composite is unavailable because this input records no weight vector"
            ),
            present=declared,
            absent_reason=(
                "the TRL completions parquet records no applied weight vector, so it cannot "
                "establish an optimised composite"
                if not declared
                else ""
            ),
            provenance=(
                {
                    "kind": "reconstructed_optimised_composite",
                    "source": "caller-declared weight vector",
                    "reader": "trl_completions",
                }
                if declared
                else {}
            ),
        ),
        "advantage": Series(
            name="advantage",
            values=tuple(row.advantage for row in rows),
            definition="the advantage column TRL wrote, one per completion",
        ),
        "selection": Series(
            name="selection",
            values=tuple(row.selection for row in rows),
            present=False,
            absent_reason=(
                "the completions parquet does not record which rows entered the update"
                + ("" if group_column else ", and no grouping was logged to derive it from")
            ),
            definition="whether the row contributed to the update",
        ),
    }
    return {name: ordered[name] for name in SERIES_NAMES}


def _subject_ref(tables: tuple[Any, ...]) -> str:
    from reward_lens.contracts import digest

    return digest(
        {
            "framework": "trl",
            "framework_version": trl_parquet.TRL_VERSION,
            "files": [
                {
                    "name": Path(table.path).name,
                    "columns": list(table.columns),
                    "reward_functions": list(table.reward_functions),
                    "rows": len(table.rows),
                }
                for table in tables
            ],
        }
    )


def _identity(kind: str, value: Any) -> str:
    from reward_lens.contracts import digest

    return digest({kind: value})


def _raw_reward_convenience(tables: tuple[Any, ...]) -> dict[str, Series]:
    values: list[float | None] = []
    recorded = False
    for table in tables:
        raw = table.raw_reward()
        if raw is None:
            values.extend(None for _ in table.rows)
            continue
        recorded = True
        values.extend(raw)
    if not recorded:
        return {}
    return {
        "recorded_raw_reward": Series(
            name="recorded_raw_reward",
            values=tuple(values),
            definition=(
                "raw imported reward retained only as a convenience aggregate, never an "
                "optimised composite; missing source columns remain null in row position"
            ),
            provenance={
                "kind": "raw_imported_reward_convenience",
                "source": "TRL completions raw reward column",
                "reader": "trl_completions",
            },
        )
    }


READER = TrlCompletionsReader()
