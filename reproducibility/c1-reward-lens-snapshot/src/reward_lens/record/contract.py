"""The recording contract: a schema entry, a writer, and a refusal, per field.

BLK-048. The contract is roughly thirty fields across twenty-three rows in the
design's Part 14.2 and nothing emits any of them. The field is named in frozen
prose, the schema has no entry for it, no writer populates it, and nothing
refuses when it is missing.

**The rule this module enforces mechanically, rather than restating.**

    Three edits per field, always all three: a schema entry with a real type, a
    writer that populates it, and a refusal that fires when it is absent or
    wrong. A schema entry with no writer is a promise; a writer with no refusal
    is a field that silently stays null in the one run that matters.

`ContractSchema` refuses to construct if any required field is missing its
writer or its refusal, so the two-out-of-three state is not representable. That
is the whole point of putting this in the library: a field list in a document
can be two thirds of a contract and look complete.

**Three naming rules, and all three exist because of one dictionary.** On the
comparable run the trainer wrote `rewards/{name}/mean` in a loop keyed on the
reward function's name, all nine functions were registered under one name, nine
appends landed in one list, and the ratio of the logged composite to that list's
mean is exactly 9.000000000 on every step. So:

  * no column anywhere is named `reward`,
  * one series per component keyed on an **index** rather than a name, with
    uniqueness asserted at write time,
  * every emitted series carries **its own** attainable maximum as metadata,
    because a two-component ceiling was compared against a nine-component series
    for months.

`set_component_series` is the only way to put a component series into a record
and it enforces all three.

**What this module is not.** It is not a parquet writer. The rollout records go
where the tap already puts them, through `RecordWriter`, and `ContractWriter`
writes the contract's own run-level and artifact-level fields into the same run
directory, beside the record, through the same root. A second writer in an
experiment directory would satisfy a closure proof and leave the next run with
the same gap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.core.errors import RewardLensError

__all__ = [
    "GRAINS",
    "ContractField",
    "ContractRecord",
    "ContractRefusal",
    "ContractSchema",
    "ContractWriter",
    "SchemaIncomplete",
]

GRAINS = ("run", "step", "group", "rollout", "token", "artifact", "quantity")

# Part 14's contract line. The name is banned as a column, not as a word.
BANNED_COLUMN_NAMES = frozenset({"reward", "rewards"})


class ContractRefusal(RewardLensError):
    """A contract field is absent, null, or of the wrong type.

    This is a refusal and not a warning. A field that is silently absent
    produces an artifact that looks careful and is not, which is worse than one
    that is obviously missing.
    """


class SchemaIncomplete(RewardLensError):
    """A required field has fewer than three of its three edits."""


# --------------------------------------------------------------------- types


def _is_array(spec: str) -> str | None:
    if spec.startswith("array<") and spec.endswith(">"):
        return spec[len("array<") : -1]
    return None


def _scalar_ok(kind: str, v: Any) -> bool:
    if kind == "float":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if kind == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if kind == "bool":
        return isinstance(v, bool)
    if kind == "str":
        return isinstance(v, str)
    if kind == "object":
        return isinstance(v, Mapping)
    if kind.startswith("enum"):
        return isinstance(v, str) and v != ""
    return False


def type_ok(spec: str, v: Any) -> bool:
    inner = _is_array(spec)
    if inner is not None:
        if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
            return False
        return all(_scalar_ok(inner, x) for x in v)
    return _scalar_ok(spec, v)


# -------------------------------------------------------------------- fields


@dataclass(frozen=True)
class ContractField:
    """One field, with all three of its edits or none.

    `writer` names the object that populates it and `refusal` says what fires
    when it is absent or wrong. Both are strings on purpose: they are read back
    out in the refusal message, so a reader of a failed rehearsal is told which
    writer did not run rather than only which key is missing.

    `pending_row` is the blocker that owns the field's **value**, where the
    upstream row has not landed. The schema entry, the writer and the refusal
    are built anyway; what is recorded is that the value's provenance is not yet
    this packet's to give. A field is never dropped for this reason.
    """

    name: str
    type: str
    grain: str
    writer: str = ""
    refusal: str = ""
    required: bool = True
    pending_row: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.name in BANNED_COLUMN_NAMES:
            raise SchemaIncomplete(
                f"no column anywhere is named `{self.name}` (Part 14's contract "
                "line). The comparable run logged an unweighted nine-metric sum "
                "and a weighted two-metric objective under one name and the "
                "confusion cost a year"
            )
        if self.grain not in GRAINS:
            raise SchemaIncomplete(f"{self.name}: grain {self.grain!r} not in {GRAINS}")
        if _is_array(self.type) is None and not _scalar_kind_known(self.type):
            raise SchemaIncomplete(f"{self.name}: type {self.type!r} is not a real type")
        if self.required and not self.writer:
            raise SchemaIncomplete(f"{self.name}: a schema entry with no writer is a promise")
        if self.required and not self.refusal:
            raise SchemaIncomplete(
                f"{self.name}: a writer with no refusal is a field that silently "
                "stays null in the one run that matters"
            )


def _scalar_kind_known(kind: str) -> bool:
    return kind in {"float", "int", "bool", "str", "object"} or kind.startswith("enum")


# -------------------------------------------------------------------- schema


@dataclass(frozen=True)
class ContractSchema:
    """Every contract field, indexed by name and by grain."""

    fields: tuple[ContractField, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise SchemaIncomplete(f"a field named twice is two fields: {dupes}")

    @classmethod
    def from_obj(cls, obj: Mapping[str, Any]) -> "ContractSchema":
        """Build from the corpus's `recording_contract.json` shape.

        The run's field list belongs to the run, so it is passed in rather than
        vendored here. What is in the library is the machinery.
        """
        out: list[ContractField] = []
        for row in obj.get("contract_rows", ()):
            for f in row.get("fields", ()):
                out.append(
                    ContractField(
                        name=f["name"],
                        type=f["type"],
                        grain=f["granularity"],
                        writer=f.get("writer", ""),
                        refusal=f.get("refusal", ""),
                        required=bool(f.get("required", True)),
                        pending_row=f.get("pending_row", ""),
                        note=f.get("note", ""),
                    )
                )
        return cls(tuple(out))

    def by_grain(self, grain: str) -> tuple[ContractField, ...]:
        return tuple(f for f in self.fields if f.grain == grain)

    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def pending(self) -> tuple[ContractField, ...]:
        """Fields built in full whose VALUE waits on an upstream row."""
        return tuple(f for f in self.fields if f.pending_row)

    def __len__(self) -> int:
        return len(self.fields)


# -------------------------------------------------------------------- record


@dataclass
class ContractRecord:
    """The contract's values, at their own grains.

    `values[grain][key][field]`. `key` is the run id at run grain, the step
    index at step grain, the rollout id at rollout grain, and the artifact path
    at artifact grain.
    """

    schema: ContractSchema
    values: dict = field(default_factory=dict)

    # ------------------------------------------------------------- writing
    def put(self, grain: str, key: Any, name: str, value: Any) -> None:
        if grain not in GRAINS:
            raise ContractRefusal(f"grain {grain!r} is not one of {GRAINS}")
        if name in BANNED_COLUMN_NAMES:
            raise ContractRefusal(f"no column anywhere is named `{name}` (Part 14's contract line)")
        spec = next((f for f in self.schema.fields if f.name == name), None)
        if spec is None:
            raise ContractRefusal(
                f"`{name}` is not in the recording contract. A field written but "
                "not registered is a field nothing refuses without"
            )
        if spec.grain != grain:
            raise ContractRefusal(
                f"`{name}` is a {spec.grain}-grain field and was written at {grain} grain"
            )
        self.values.setdefault(grain, {}).setdefault(str(key), {})[name] = value

    def set_component_series(
        self, key: Any, values: Sequence[float], index: Sequence[int], maxima: Sequence[float]
    ) -> None:
        """The only door a component series comes through.

        Enforces all three of Part 14's naming rules at write time: keyed on an
        index rather than a name, the index unique, and every series carrying
        its own attainable maximum.
        """
        if len(values) != len(index) or len(values) != len(maxima):
            raise ContractRefusal(
                f"the component series has {len(values)} values, {len(index)} "
                f"index entries and {len(maxima)} maxima. A series whose index "
                f"is shorter than the series is a series keyed on something "
                f"else"
            )
        if len(set(index)) != len(index):
            raise ContractRefusal(
                f"the component index {list(index)} is not unique. This is the "
                f"exact defect: nine functions registered under one key put "
                f"nine appends in one list and produced a ratio of exactly "
                f"9.000000000 on every step"
            )
        if any(not isinstance(i, int) or isinstance(i, bool) for i in index):
            raise ContractRefusal(
                "the component index is not integral, so the series is keyed on a name after all"
            )
        for i, m in zip(index, maxima):
            if not isinstance(m, (int, float)) or isinstance(m, bool):
                raise ContractRefusal(
                    f"component {i} carries no attainable maximum. A ceiling "
                    f"belongs to a series and not to a run"
                )
        self.put("rollout", key, "reward_components", [float(v) for v in values])
        self.put("run", "run", "reward_component_index", [int(i) for i in index])
        self.put("run", "run", "reward_component_max", [float(m) for m in maxima])

    # ------------------------------------------------------------- reading
    def get(self, grain: str, key: Any, name: str) -> Any:
        return self.values.get(grain, {}).get(str(key), {}).get(name)

    def keys(self, grain: str) -> tuple[str, ...]:
        return tuple(self.values.get(grain, {}))

    # ------------------------------------------------------------- refusing
    def verify(
        self, *, expect: Mapping[str, Iterable[Any]] | None = None, allow_pending: bool = False
    ) -> None:
        """Field by field, at every key of its grain. Raises on the first hole.

        `expect` names the keys that must exist at each grain, so the check is
        "every rollout carries every rollout field" rather than "the row count
        looks right". A count is what a rehearsal reports when a writer never
        ran on the last group.
        """
        expect = {k: list(v) for k, v in (expect or {}).items()}
        problems: list[str] = []
        for spec in self.schema.fields:
            if not spec.required:
                continue
            if spec.pending_row and allow_pending:
                continue
            keys = expect.get(spec.grain)
            if keys is None:
                keys = list(self.values.get(spec.grain, {}))
            if not keys:
                problems.append(
                    f"`{spec.name}` ({spec.grain}): no {spec.grain} was recorded "
                    f"at all, so the field cannot be present on any of them. "
                    f"Writer: {spec.writer}"
                )
                continue
            for k in keys:
                v = self.get(spec.grain, k, spec.name)
                if v is None:
                    problems.append(
                        f"`{spec.name}` absent on {spec.grain} {k}. "
                        f"{spec.refusal} Writer: {spec.writer}"
                        + (f" [value pending {spec.pending_row}]" if spec.pending_row else "")
                    )
                elif not type_ok(spec.type, v):
                    problems.append(
                        f"`{spec.name}` on {spec.grain} {k} is {type(v).__name__} "
                        f"and the contract declares {spec.type}. {spec.refusal}"
                    )
        if problems:
            raise ContractRefusal(
                f"{len(problems)} contract field(s) absent or wrong:\n  "
                + "\n  ".join(problems[:40])
                + ("\n  ..." if len(problems) > 40 else "")
            )


# -------------------------------------------------------------------- writer


@dataclass
class ContractWriter:
    """Writes the contract beside the record, in the record's own run directory.

    `RecordWriter` owns the rollout tables. This owns the contract's run-level
    and artifact-level fields and the per-rollout contract columns that the
    record model has no slot for, and it puts them under the same root so there
    is one artifact tree and not two.
    """

    root: str | Path
    run_id: str

    def run_dir(self) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.run_id)
        return Path(self.root) / "runs" / safe

    def write(self, record: ContractRecord) -> Path:
        d = self.run_dir() / "contract"
        d.mkdir(parents=True, exist_ok=True)
        for grain in GRAINS:
            block = record.values.get(grain)
            if not block:
                continue
            for key_name in block.values():
                for col in key_name:
                    if col in BANNED_COLUMN_NAMES:
                        raise ContractRefusal(f"no column anywhere is named `{col}`")
            (d / f"{grain}.json").write_text(
                json.dumps(block, indent=1, sort_keys=True, default=_json_default), encoding="utf-8"
            )
        (d / "schema.json").write_text(
            json.dumps(
                {
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            "grain": f.grain,
                            "writer": f.writer,
                            "refusal": f.refusal,
                            "pending_row": f.pending_row,
                        }
                        for f in record.schema.fields
                    ]
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return d


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)
