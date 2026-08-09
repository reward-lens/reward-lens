"""`rewardlens.yaml` as a typed contract.

pydantic carries this one rather than the JSON Schema doing it alone, because a person edits this
file by hand and the error they need says which key, on which line of their own vocabulary, was
wrong. The checked-in `schema/project/1.0/rewardlens.schema.json` is the same contract for readers
that are not Python, and a test refuses any project file the two layers disagree about.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_serializer

from .serialise import bind

__all__ = [
    "ProjectConfig",
    "RewardSpec",
    "TasksSpec",
    "ResponsesSpec",
    "OutcomeSpec",
    "SCHEMA_PATH",
    "project_schema",
]

#: The project schema ships beside the package in the wheel and beside the repo root in a checkout.
_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "schema" / "project" / "1.0" / "rewardlens.schema.json",
    Path(__file__).resolve().parents[3] / "schema" / "project" / "1.0" / "rewardlens.schema.json",
)
SCHEMA_PATH = _CANDIDATES[1]


@lru_cache(maxsize=1)
def project_schema() -> dict:
    """`schema/project/1.0/rewardlens.schema.json`, read once.

    `rewardlens.yaml` has a schema of its own, and it governs this module's serialisation the way
    the assay schema governs the record's: a property it requires is always written, one it leaves
    optional is omitted when None.
    """
    import json

    for candidate in _CANDIDATES:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "rewardlens.schema.json was not found beside the package or at the repo root; "
        f"looked in {', '.join(str(c) for c in _CANDIDATES)}"
    )

Ident = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")]
Text = Annotated[str, Field(min_length=1)]

#: What the reward system is called, in the words its authors use. The record carries this as
#: `subject.reward_system.name`, and the assay schema constrains that to a non-empty string of at
#: most 200 characters with no character rules beyond it: it is prose, not an identifier, so
#: `Ident` is deliberately not applied. The one constraint the assay schema does not spell out is
#: `\S`, which refuses a name that is whitespace alone; `minLength: 1` alone would admit `"  "`,
#: and a Subject line printed from that reads as though the system had no name. The hand-authored
#: project schema carries the same three constraints, so both layers refuse the same files.
SystemName = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]

#: The three ways a grader can mislead a trainer without raising, each named so a finding can say
#: what the named trainer would do with it (D-65).
Watch = Literal["none_to_nan", "exception_to_zero", "bool_as_score"]
Shape = Literal["plain", "trl", "verifiers", "inspect"]


class Base(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)

    @model_serializer(mode="wrap")
    def _drop_absent_optionals(self, handler: Any) -> dict[str, Any]:
        """The one rule: what the project schema requires stays, even as null; the rest goes."""
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        omit = bind(project_schema(), {ProjectConfig: project_schema()}).omitted(type(self))
        return {k: v for k, v in dumped.items() if v is not None or k not in omit}

    def to_dict(self) -> dict[str, Any]:
        """A pure function of the values, the same rule `models.Base` follows.

        `exclude_unset` was the old rule here, which made the output depend on how the object was
        built: a config loaded from `rewardlens.yaml` and the same config built in code wrote
        different dicts, and `runtime/fingerprint.py` reads this one.
        """
        return self.model_dump(mode="json", by_alias=True)


class ComponentSpec(Base):
    name: Text
    entry: Text
    weight: float = None  # type: ignore[assignment]


class RewardSpec(Base):
    entry: Text
    kind: Literal["plain", "composite"] = "plain"
    shape: Shape | None = None
    components: list[ComponentSpec] = Field(default_factory=list)
    watch: list[Watch] = Field(default_factory=list)
    trainer: Shape | None = None


class TasksSpec(Base):
    path: Text
    prompt: Text
    reference: str = None  # type: ignore[assignment]
    tests: str = None  # type: ignore[assignment]


class ResponsesSpec(Base):
    path: Text


class OutcomeSpec(Base):
    kind: Literal["protected_test_suite"]
    path: Text


class VersionSpec(Base):
    id: Ident
    parents: list[Ident] = Field(default_factory=list)


class ProjectConfig(Base):
    """What a project declares about its reward system, before anything is measured.

    `name` is the system's own name and nothing is derived from it. Before this field existed the
    record's `subject.reward_system.name` was taken from the project directory, which named the
    folder someone happened to check the project out into rather than the reward system, and which
    changed under a rename. A project that does not declare one is unchanged: the field stays None,
    the serialiser drops it, and whatever assembles the record falls back as it did before.
    """

    name: SystemName | None = None
    reward: RewardSpec
    tasks: TasksSpec
    outcome: OutcomeSpec | None
    responses: ResponsesSpec | None = None
    success: str | None = None
    constraints: list[str] = Field(default_factory=list)
    version: VersionSpec | None = None

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "ProjectConfig":
        """A person edits `rewardlens.yaml` by hand, so a bad field is RL0003 and names itself.

        pydantic's own `ValidationError` is kept as the cause, because it carries every location
        and this carries only the first: a caller who wants them all reads `__cause__.errors()`.
        """
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as invalid:
            raise _bad_project_field(invalid) from invalid


def _bad_project_field(invalid: ValidationError) -> Exception:
    from reward_lens.errors import make

    first = invalid.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "(the file itself)"
    return make("RL0003", field=field, detail=first["msg"], context={"field": field})
