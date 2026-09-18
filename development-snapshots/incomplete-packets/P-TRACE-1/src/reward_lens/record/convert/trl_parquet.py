"""TRL's completions parquet, read without importing `trl` (D-42).

Verified against the released 1.13.0 rather than `main`. The writer is one dict splatted into a
table:

    table = {
        "step": [...], "prompt": [...], "completion": [...],
        **self._logs["rewards"],      # one column per reward function, under that function's name
        **self._logs["extra"],        # whatever the user passed to log_extra
        "advantage": [...],
    }

There is no column named for the total. The total is the weighted sum of the per-function columns,
and the batch means that a dashboard shows live are in the metric stream, not in this file. That
distinction is what E1 paid for, and it is the reason nothing here reads a total off a column.

The one thing the file cannot tell you is where the reward functions stop and the user's extra
columns start, because the writer splats two dicts into one flat table and keeps no boundary. A
reader that guessed would put an extra column into the weighted sum. So the partition is declared
by the caller, and a table read with neither half declared refuses with RL0610.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from reward_lens.core.extras import ExtraRequiredError, require_extra
from reward_lens.product.trace.errors import RecordIncomplete, RecordIntegrity

__all__ = [
    "COMPLETIONS_GLOB",
    "CompletionsTable",
    "RESERVED_COLUMNS",
    "TRL_TAG",
    "TRL_VERIFIED_COMMAND",
    "TRL_VERSION",
    "WRITER_LOCATION",
    "WRITER_LOCATION_1_9_2",
    "completions_files",
    "read_completions",
    "read_directory",
]

#: The release this layout was read off, and the command that fetched it. Recorded rather than
#: remembered: a layout checked against a memory of a tag is not checked.
TRL_VERSION = "1.13.0"
TRL_TAG = "v1.13.0"
TRL_VERIFIED_COMMAND = "uv pip install trl==1.13.0 --no-deps --target <scratch>"
WRITER_LOCATION = "trl/trainer/grpo_trainer.py:3443-3450"
#: The vendored 1.9.2 checkout writes the same table from the same dict, at its own lines.
WRITER_LOCATION_1_9_2 = "trl/trainer/grpo_trainer.py:3314-3325"

#: The columns the writer names itself. Everything between `completion` and `advantage` is the
#: two splatted dicts, in that order, with no recorded boundary between them.
RESERVED_COLUMNS: tuple[str, ...] = ("step", "prompt", "completion", "advantage")

COMPLETIONS_GLOB = "completions_*.parquet"

#: The name no column may carry, under the first rule of the run-record standard (section 7.4).
_REFUSED_COLUMN = "reward"  # forbidden-column-name


#: What a caller is doing when the `[trace]` extra turns out to be missing.
_SUBSYSTEM = "reading a TRL completions parquet"


def _pyarrow_present() -> bool:
    """Whether the `[trace]` extra's marker distribution is importable here."""
    try:
        importlib.import_module("pyarrow")
    except Exception:  # noqa: BLE001 - a pyarrow that will not import is a pyarrow that is not here
        return False
    return True


def _require_pyarrow() -> None:
    """Refuse, naming the `[trace]` extra, when pyarrow will not import here.

    `require_extra` decides with `importlib.util.find_spec`, which answers yes for a distribution
    that is on the path but will not import: a half-removed install, a wheel built for another
    ABI, a namespace package left behind. `_pyarrow_present` decides by importing, which is the
    question this module actually asks, so when the two disagree the import is the one to believe.
    Without this second raise the caller would fall through to `importlib.import_module` below and
    get a bare `ModuleNotFoundError`, which is the outcome `require_extra` exists to prevent.
    """
    if _pyarrow_present():
        return
    require_extra("trace", subsystem=_SUBSYSTEM)
    raise ExtraRequiredError(
        f"{_SUBSYSTEM} needs the optional 'trace' extra. pyarrow is on the path here but will "
        f"not import, so the extra is not usable. Reinstall it with:  "
        f"pip install --force-reinstall 'reward-lens[trace]'"
    )


@dataclass(frozen=True)
class CompletionsTable:
    """One completions file, with its columns partitioned as the caller declared them."""

    path: str
    columns: tuple[str, ...]
    reward_functions: tuple[str, ...]
    extra_columns: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    weights: Mapping[str, float] = field(default_factory=dict)
    engine: str = "pyarrow"
    trl_version: str = TRL_VERSION

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted({int(row["step"]) for row in self.rows}))

    def weight_vector(self) -> tuple[float, ...]:
        return tuple(float(self.weights.get(name, 1.0)) for name in self.reward_functions)

    def component(self, name: str) -> tuple[float | None, ...]:
        return tuple(_number(row.get(name)) for row in self.rows)

    def totals(self) -> tuple[float | None, ...]:
        """The weighted sum, one per row, with the trainer's missing values preserved."""
        vector = self.weight_vector()
        out: list[float | None] = []
        for row in self.rows:
            total = 0.0
            seen = False
            for weight, name in zip(vector, self.reward_functions):
                value = _number(row.get(name))
                if value is None:
                    continue
                total += weight * value
                seen = True
            out.append(total if seen else None)
        return tuple(out)

    def advantages(self) -> tuple[float | None, ...]:
        return tuple(_number(row.get("advantage")) for row in self.rows)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def completions_files(directory: Path | str) -> tuple[Path, ...]:
    """Every completions parquet under a run directory, in step order."""
    root = Path(directory)
    if root.is_file():
        return (root,)
    inner = root / "completions"
    base = inner if inner.is_dir() else root
    return tuple(sorted(base.glob(COMPLETIONS_GLOB)))


def read_completions(
    path: Path | str,
    *,
    reward_functions: Sequence[str] | None = None,
    extra_columns: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> CompletionsTable:
    """One completions parquet, read through pyarrow, with the column partition declared."""
    _require_pyarrow()
    pq = importlib.import_module("pyarrow.parquet")
    target = Path(path)
    table = pq.read_table(target)
    columns = tuple(str(name) for name in table.column_names)
    if _REFUSED_COLUMN in columns:
        raise RecordIntegrity(
            message=(
                f"{target} carries a column named {_REFUSED_COLUMN!r}; no column anywhere is "
                f"named that, because a single column with that name is how a displayed batch "
                f"mean gets read as a per-completion total"
            ),
            remediation=(
                "name the per-function columns after their functions and let the total be the "
                "weighted sum, which is what TRL's own writer does"
            ),
            context={"rule": "trace.refused_column_name", "row": columns[0], "column": columns[0]},
        )
    middle = tuple(name for name in columns if name not in RESERVED_COLUMNS)
    if reward_functions is None and extra_columns is None:
        raise RecordIncomplete(
            message=(
                f"{target} does not record which of its columns ({', '.join(middle)}) are reward "
                f"functions and which are the user's extra columns, so the series "
                f"reward_recorded cannot be built: TRL splats both dicts into one table and "
                f"keeps no boundary"
            ),
            remediation=(
                "pass reward_functions= or extra_columns= to declare the partition; the order of "
                f"the columns is the writer's own ({WRITER_LOCATION}), rewards then extras, but "
                "the boundary is not in the file"
            ),
            context={"series": ["reward_recorded"], "columns": list(middle), "source": str(target)},
        )
    if reward_functions is None:
        declared_extra = tuple(extra_columns or ())
        functions = tuple(name for name in middle if name not in declared_extra)
    else:
        functions = tuple(reward_functions)
        declared_extra = (
            tuple(extra_columns)
            if extra_columns is not None
            else tuple(name for name in middle if name not in functions)
        )
    unknown = [name for name in functions if name not in columns]
    if unknown:
        raise RecordIncomplete(
            message=(
                f"{target} has no column for the declared reward function(s) "
                f"{', '.join(unknown)}, so the series reward_recorded cannot be built"
            ),
            remediation="declare the reward functions this file actually carries: " + ", ".join(middle),
            context={"series": ["reward_recorded"], "columns": list(middle), "source": str(target)},
        )
    rows = tuple(table.to_pylist())
    return CompletionsTable(
        path=str(target),
        columns=columns,
        reward_functions=functions,
        extra_columns=declared_extra,
        rows=rows,
        weights=dict(weights or {}),
        engine="pyarrow",
    )


def read_directory(
    directory: Path | str,
    *,
    reward_functions: Sequence[str] | None = None,
    extra_columns: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[CompletionsTable, ...]:
    """Every completions file of one run, in step order."""
    return tuple(
        read_completions(
            one,
            reward_functions=reward_functions,
            extra_columns=extra_columns,
            weights=weights,
        )
        for one in completions_files(directory)
    )
