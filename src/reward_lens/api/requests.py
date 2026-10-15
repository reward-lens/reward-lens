"""The typed requests and the two results that are not records (D-67, section 8.0).

Every option the CLI exposes as a flag is a field here, so a notebook cell and a terminal ask the
same question. The models are strict and closed: a misspelt field is refused rather than ignored,
and a float is never money. They are frozen, because a request is a value, not a workspace.

One convenience the frozen line does not forbid: a field typed `Path` also accepts the string a
terminal or a JSON payload hands it, converted before strict validation sees it. Nothing else is
coerced.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Iterator, Literal, Mapping

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from reward_lens import errors
from reward_lens.contracts import RewardLensError

if TYPE_CHECKING:  # the name only: the execution package is never imported at import time
    from reward_lens.execution import SandboxProbe

__all__ = [
    "AuditRequest",
    "Capabilities",
    "Capability",
    "CompareRequest",
    "DEFAULT_SEED",
    "DoctorRequest",
    "ExportRequest",
    "ForecastIssueRequest",
    "ForecastLedgerRequest",
    "ForecastResolveRequest",
    "ImproveRequest",
    "ImportRequest",
    "ImportResult",
    "InstallInfo",
    "Plan",
    "SeekerProposalRequest",
    "TraceRequest",
]

#: The fleet's fixed seed: every default run is reproducible without saying so.
DEFAULT_SEED = 20260911


def _as_path(value: Any) -> Any:
    return Path(value) if isinstance(value, str) else value


PathField = Annotated[Path, BeforeValidator(_as_path)]

Seeker = Literal["off", "api", "local", "agent"]
Money = str


def _validate_weight_declarations(declarations: tuple[str, ...]) -> tuple[str, ...]:
    names: set[str] = set()
    for declaration in declarations:
        name, separator, value = declaration.partition("=")
        if not separator or not name or not value:
            raise ValueError("each weights declaration must have the form NAME=VALUE")
        if name in names:
            raise ValueError(f"weights declares {name!r} more than once")
        try:
            number = float(value)
        except ValueError as failure:
            raise ValueError(f"weights value for {name!r} is not a number") from failure
        if not math.isfinite(number):
            raise ValueError(f"weights value for {name!r} must be finite")
        names.add(name)
    return declarations


def _validate_group_size(value: int | None) -> int | None:
    if value is not None and value <= 0:
        raise ValueError("group_size must be a positive integer")
    return value


def _validate_trl_declarations(
    reward_functions: tuple[str, ...],
    extra_columns: tuple[str, ...],
    weights: tuple[str, ...],
    group_column: str | None,
    group_size: int | None,
) -> None:
    reward_names = set(reward_functions)
    extra_names = set(extra_columns)
    if len(reward_names) != len(reward_functions):
        raise ValueError("reward_functions names a function more than once")
    if len(extra_names) != len(extra_columns):
        raise ValueError("extra_columns names a column more than once")
    overlap = reward_names & extra_names
    if overlap:
        raise ValueError(
            "a column cannot be both a reward function and an extra column: "
            + ", ".join(sorted(overlap))
        )
    weight_names = {declaration.partition("=")[0] for declaration in weights}
    if weight_names and not reward_names:
        raise ValueError("weights needs reward_functions so each component is named")
    if weight_names and weight_names != reward_names:
        missing = sorted(reward_names - weight_names)
        unknown = sorted(weight_names - reward_names)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("not declared as reward functions: " + ", ".join(unknown))
        raise ValueError(
            "weights must name every declared reward function exactly once ("
            + "; ".join(detail)
            + ")"
        )
    if group_column is not None and group_column not in extra_names:
        raise ValueError("group_column must also be named in extra_columns")
    if group_size is not None and group_column is None:
        raise ValueError(
            "group_size needs group_column, because a cardinality without a source column is not evidence"
        )


def _weight_map(declarations: tuple[str, ...]) -> dict[str, float]:
    return {
        declaration.partition("=")[0]: float(declaration.partition("=")[2])
        for declaration in declarations
    }


def _as_sandbox_probe(value: Any) -> "SandboxProbe | None":
    """Accept the probe result `reward_lens.execution.SandboxProbe`, and nothing else.

    What a `Capabilities` carries is the measurement, not the instrument: the interface freezes
    `sandbox: SandboxProbe`, which is what `probe()` returns and what `doctor` reports. A runner
    is refused here rather than quietly reported, because a reader of the field would take its
    `tier` for a measured one.

    The type is imported inside the function and not at module import, so `import
    reward_lens.api` still costs the contracts and the standard library: the execution package is
    reached only when a probe is actually handed over, and a build without it can still report
    its capabilities.
    """
    if value is None:
        return None
    try:
        from reward_lens.execution import SandboxProbe as _SandboxProbe
    except ImportError:  # a build without the execution package: nothing to check against
        return value
    if not isinstance(value, _SandboxProbe):
        raise ValueError(
            "the sandbox of a Capabilities is the probe result "
            "reward_lens.execution.SandboxProbe, as `probe()` returns it, and not a sandbox runner"
        )
    return value


#: `reward_lens.execution.SandboxProbe`, checked on the value rather than imported at import time.
SandboxProbeField = Annotated[Any, BeforeValidator(_as_sandbox_probe)]

#: The fields that carry an identifier rather than a setting: a run, a version, a forecast. A bad
#: value in one of these is RL0002, a bad identifier; a bad value anywhere else is RL0003, a bad
#: config field. Both carry exit 4, which is what a caller returns for an input it cannot use.
IDENTIFIER_FIELDS = frozenset({"run", "resume", "baseline", "candidate", "forecast_id"})


def _detail(problem: dict[str, Any]) -> str:
    """What is wrong with the value, in pydantic's own words and with the value it was given."""
    kind = problem.get("type", "")
    said = str(problem.get("msg", "")).rstrip(".")
    if kind == "extra_forbidden":
        return "this request has no field of that name"
    if kind == "missing":
        return "the request cannot be built without it"
    if "input" in problem:
        return f"{said} (the value given was {problem['input']!r})"
    return said or "the value is not one this field takes"


def _refusal(model: type[BaseModel], invalid: PydanticValidationError) -> RewardLensError:
    """Turn one pydantic failure into the reserved code-4 refusal.

    Built through `reward_lens.errors.make`, which fits: RL0002 and RL0003 both take `{field}`
    and `{detail}`, and both carry exit 4, so what the caller sees is the catalogue's own text and
    the catalogue's own remedy. Nothing here writes a message the catalogue does not hold.
    """
    problems = invalid.errors()
    first = problems[0] if problems else {}
    where = ".".join(str(part) for part in first.get("loc", ()))
    field = where or model.__name__
    code = "RL0002" if where in IDENTIFIER_FIELDS else "RL0003"
    refusal = errors.make(code, field=field, detail=_detail(first))
    refusal.context.update({"request": model.__name__, "field": field, "problems": len(problems)})
    return refusal


class Request(BaseModel):
    """Strict, closed and frozen: the shared configuration of every request and result here.

    One validation path, and it is this class's. A value a model refuses leaves the SDK as the
    reserved code-4 refusal (RL0003 for a bad config field, RL0002 for a bad identifier) and never
    as a `pydantic.ValidationError`, which is what lets a caller print two lines and exit 4 without
    error logic of its own, and what keeps pydantic off the public surface.

    One pydantic error still surfaces: an assignment to a request already built. A frozen violation
    is a mistake in the calling code rather than an input, and no command line can reach it.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except PydanticValidationError as invalid:
            raise _refusal(type(self), invalid) from invalid

    @classmethod
    def model_validate(cls, obj: Any, **rest: Any) -> Any:
        """The payload path a notebook or a JSON body takes, through the same refusal."""
        try:
            return super().model_validate(obj, **rest)
        except PydanticValidationError as invalid:
            raise _refusal(cls, invalid) from invalid


class AuditRequest(Request):
    """One audit of one reward system, with the options the CLI exposes as flags."""

    path: PathField
    tasks: PathField | None = None
    responses: PathField | None = None
    outcome: PathField | None = None
    seeker: Seeker = "off"
    seeker_config: PathField | None = None
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    dry_run: bool = False
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    detection_reference: PathField | None = None
    reward_functions: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    group_column: str | None = None
    group_size: int | None = None
    name: str | None = None
    resume: str | None = None
    continue_run: bool = False
    only: tuple[str, ...] = ()
    policy: PathField | str | None = None
    non_interactive: bool = False

    _weight_declarations_are_finite_named_pairs = field_validator("weights")(
        lambda declarations: _validate_weight_declarations(declarations)
    )
    _group_size_is_positive = field_validator("group_size")(
        lambda value: _validate_group_size(value)
    )

    @model_validator(mode="after")
    def _trl_declarations_describe_one_unambiguous_partition(self) -> "AuditRequest":
        _validate_trl_declarations(
            self.reward_functions,
            self.extra_columns,
            self.weights,
            self.group_column,
            self.group_size,
        )
        return self

    @property
    def weight_map(self) -> dict[str, float]:
        return _weight_map(self.weights)


class TraceRequest(Request):
    """What a training run actually optimised, read back from the run record.

    A TRL completions parquet deliberately flattens the reward-function and ``log_extra``
    dictionaries into one table. These declarations preserve the boundary the file omitted;
    they are caller statements, never columns inferred from position or a default weight vector.
    Other readers receive no interpretation of them.
    """

    run: str | None = None
    project: PathField | None = None
    only: tuple[str, ...] = ()
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    name: str | None = None
    non_interactive: bool = False
    reward_functions: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    weights: tuple[str, ...] = ()
    group_column: str | None = None
    group_size: int | None = None

    @field_validator("weights")
    @classmethod
    def _weight_declarations_are_finite_named_pairs(
        cls, declarations: tuple[str, ...]
    ) -> tuple[str, ...]:
        names: set[str] = set()
        for declaration in declarations:
            name, separator, value = declaration.partition("=")
            if not separator or not name or not value:
                raise ValueError("each weights declaration must have the form NAME=VALUE")
            if name in names:
                raise ValueError(f"weights declares {name!r} more than once")
            try:
                number = float(value)
            except ValueError as failure:
                raise ValueError(f"weights value for {name!r} is not a number") from failure
            if not math.isfinite(number):
                raise ValueError(f"weights value for {name!r} must be finite")
            names.add(name)
        return declarations

    @field_validator("group_size")
    @classmethod
    def _group_size_is_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("group_size must be a positive integer")
        return value

    @model_validator(mode="after")
    def _trl_declarations_describe_one_unambiguous_partition(self) -> "TraceRequest":
        reward_names = set(self.reward_functions)
        extra_names = set(self.extra_columns)
        if len(reward_names) != len(self.reward_functions):
            raise ValueError("reward_functions names a function more than once")
        if len(extra_names) != len(self.extra_columns):
            raise ValueError("extra_columns names a column more than once")
        overlap = reward_names & extra_names
        if overlap:
            raise ValueError(
                "a column cannot be both a reward function and an extra column: "
                + ", ".join(sorted(overlap))
            )
        weight_names = {declaration.partition("=")[0] for declaration in self.weights}
        if weight_names and not reward_names:
            raise ValueError("weights needs reward_functions so each component is named")
        if weight_names and weight_names != reward_names:
            missing = sorted(reward_names - weight_names)
            unknown = sorted(weight_names - reward_names)
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("not declared as reward functions: " + ", ".join(unknown))
            raise ValueError(
                "weights must name every declared reward function exactly once ("
                + "; ".join(detail)
                + ")"
            )
        if self.group_column is not None and self.group_column not in extra_names:
            raise ValueError("group_column must also be named in extra_columns")
        if self.group_size is not None and self.group_column is None:
            raise ValueError(
                "group_size needs group_column, because a cardinality without a source column is not evidence"
            )
        return self

    @property
    def weight_map(self) -> dict[str, float]:
        """The validated declaration map handed to the TRL reader, with no implicit component."""
        return _weight_map(self.weights)


class CompareRequest(Request):
    """Two reward versions, with finite-bank, replay, and executable-rung inputs."""

    baseline: str
    candidate: str
    project: PathField | None = None
    bank: PathField | None = None
    rungs: tuple[int, ...] = ()
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    baseline_config: PathField | None = None
    candidate_config: PathField | None = None
    selection_config: PathField | None = None
    controlled_update_config: PathField | None = None
    short_fork_config: PathField | None = None
    transfer_config: PathField | None = None
    name: str | None = None
    non_interactive: bool = False


class ForecastIssueRequest(Request):
    """A claim written down before the evidence, so that the ledger can score it later."""

    claim: str
    record: PathField | None = None
    project: PathField | None = None
    resolves_by: str | None = None
    horizon: str | None = None
    seed: int = DEFAULT_SEED
    offline: bool = True
    name: str | None = None
    non_interactive: bool = False


class ForecastResolveRequest(Request):
    """The outcome of a claim already on the ledger."""

    forecast_id: str
    project: PathField | None = None
    outcome: str | None = None
    seed: int = DEFAULT_SEED
    offline: bool = True
    name: str | None = None
    non_interactive: bool = False


class ForecastLedgerRequest(Request):
    """The standing ledger of issued and resolved claims.

    Additive to the request list the wave-1 interfaces spell out; section 8.0 names
    `forecast_ledger` as a verb, and a verb here takes a typed request.
    """

    project: PathField | None = None
    since: str | None = None
    seed: int = DEFAULT_SEED
    offline: bool = True
    name: str | None = None
    non_interactive: bool = False


class ImproveRequest(Request):
    """What to change about a reward system, and what changing it would cost.

    The loop admits four equal candidate origins.  The origin says how a
    candidate arrived; it does not confer a different evaluation path.
    """

    path: PathField
    only: tuple[str, ...] = ()
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    name: str | None = None
    non_interactive: bool = False
    describe: str | None = None
    candidate_source: (
        Literal["owner_edit", "agent_draft", "generator_output", "supported_intervention"] | None
    ) = None
    candidate: Mapping[str, Any] | None = None
    forecast: Mapping[str, Any] | None = None
    resume: str | None = None
    mark_pair: tuple[PathField, PathField] | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _improve_payload_is_a_complete_interaction(self) -> "ImproveRequest":
        """Refuse incomplete paired inputs rather than silently dropping them."""
        if (self.candidate_source is None) != (self.candidate is None):
            raise ValueError("candidate_source and candidate must be provided together")
        if self.mark_pair is None and self.why is not None:
            raise ValueError("why requires mark_pair")
        if self.mark_pair is not None and not self.why:
            raise ValueError("mark_pair requires a non-empty why")
        if self.describe is not None and not self.describe.strip():
            raise ValueError("describe must be non-empty when provided")
        return self


class SeekerProposalRequest(Request):
    """One explicit, zero-spend caller proposal against a qualified project target.

    This is deliberately neutral about transport: MCP translates its camel-case
    arguments here, while Python callers use the same immutable request.  The
    response is a candidate answer, never a score, outcome, actor receipt, or
    caller-supplied evidence.
    """

    project_path: PathField
    task_id: str
    response: str
    rationale: str = ""
    spend: Literal["none"] = "none"
    run_dir: PathField | None = None

    @field_validator("task_id", "response")
    @classmethod
    def _required_proposal_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must be a non-empty string")
        return value


class ExportRequest(Request):
    """A record out of the store and into a file a reader can open."""

    record: PathField | None = None
    project: PathField | None = None
    format: str = "html"
    dest: PathField | None = None
    offline: bool = True
    non_interactive: bool = False
    tier: Literal["A", "B", "C"] = "A"
    backend: Literal["ed25519", "sigstore"] = "ed25519"
    #: The single measured Reach entry a badge may project.  It is meaningful only
    #: for ``format="badge"``; omission is accepted only when exactly one exists.
    badge_reach_entry_id: str | None = None


class ImportRequest(Request):
    """One received record or Export bundle, optionally verified and durably retained."""

    path: PathField
    verify: bool = False
    dest: PathField | None = None


class ImportResult(Request):
    """The parsed record and the durable evidence paths created by an import."""

    record: Any
    record_path: PathField
    receipt_path: PathField
    unknown_extensions: tuple[str, ...] = ()
    signer: str | None = None
    verification_message: str | None = None


class DoctorRequest(Request):
    """What this installation can and cannot do."""

    project: PathField | None = None


class Plan(Request):
    """What a run would do before it does it (section 8.0, `dry_run`)."""

    command: str
    panels: tuple[str, ...] = ()
    paid_calls: int = 0
    estimate_usd: Money = "0.00"
    #: The seconds the run is expected to take, before it runs (A-016). Zero is the default a
    #: planner that cannot estimate leaves standing, not a claim that the run is instant.
    estimate_s: int = 0
    notes: tuple[str, ...] = ()


class Capability(Request):
    """One thing this installation can or cannot do, and what would unlock it."""

    id: str
    status: Literal["available", "unavailable", "pending"]
    reason: str
    unlocks: list[str] = []
    cost: Money = "0.00"


class InstallInfo(Request):
    """The installation itself: what is installed, on what interpreter, with which extras."""

    version: str
    python: str
    platform: str
    extras: tuple[str, ...] = ()


class Capabilities(Request):
    """What `doctor` reports: the capabilities, the sandbox probe and the installation.

    `sandbox` is the `reward_lens.execution.SandboxProbe` the interface freezes: what this machine
    was measured to hold, not the runner that would use it.

    The frozen line is `list[Capability] plus sandbox, install`, so the capabilities are a `list`
    and this object behaves as that list: a consumer iterates it, measures it, indexes it and
    appends to it. The list is the model's own, so an append through the object and an append
    through the field are one append.
    """

    capabilities: list[Capability] = []
    # Captured local adapter conformance, keyed by family. This is separate from the access
    # ladder above, whose list and iteration contract remain unchanged.
    compatibility_matrix: dict[str, dict[str, str]] = Field(default_factory=dict)
    sandbox: SandboxProbeField | None = None
    install: InstallInfo

    def __iter__(self) -> Iterator[Capability]:  # type: ignore[override]
        return iter(self.capabilities)

    def __len__(self) -> int:
        return len(self.capabilities)

    def __getitem__(self, index: int) -> Capability:
        return self.capabilities[index]

    def append(self, capability: Capability) -> None:
        """Add one capability, as a consumer holding the frozen `list[Capability]` would."""
        self.capabilities.append(capability)
