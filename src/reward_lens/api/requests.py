"""The typed requests and the two results that are not records (D-67, section 8.0).

Every option the CLI exposes as a flag is a field here, so a notebook cell and a terminal ask the
same question. The models are strict and closed: a misspelt field is refused rather than ignored,
and a float is never money. They are frozen, because a request is a value, not a workspace.

One convenience the frozen line does not forbid: a field typed `Path` also accepts the string a
terminal or a JSON payload hands it, converted before strict validation sees it. Nothing else is
coerced.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Iterator, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict
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
    "InstallInfo",
    "Plan",
    "TraceRequest",
]

#: The fleet's fixed seed: every default run is reproducible without saying so.
DEFAULT_SEED = 20260911


def _as_path(value: Any) -> Any:
    return Path(value) if isinstance(value, str) else value


PathField = Annotated[Path, BeforeValidator(_as_path)]

Seeker = Literal["off", "api", "local", "agent"]
Money = str


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
    refusal.context.update(
        {"request": model.__name__, "field": field, "problems": len(problems)}
    )
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
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    dry_run: bool = False
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    name: str | None = None
    resume: str | None = None
    only: tuple[str, ...] = ()
    policy: PathField | str | None = None
    non_interactive: bool = False


class TraceRequest(Request):
    """What a training run actually optimised, read back from the run record."""

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


class CompareRequest(Request):
    """Two versions of one reward system, on as many rungs of the ladder as the build holds."""

    baseline: str
    candidate: str
    project: PathField | None = None
    rungs: tuple[int, ...] = ()
    max_budget_usd: Money | None = None
    offline: bool = True
    sandbox: str = "auto"
    require_tier: str | None = None
    seed: int = DEFAULT_SEED
    run_dir: PathField | None = None
    name: str | None = None
    non_interactive: bool = False


class ForecastIssueRequest(Request):
    """A claim written down before the evidence, so that the ledger can score it later."""

    claim: str
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
    """What to change about a reward system, and what changing it would cost."""

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


class ExportRequest(Request):
    """A record out of the store and into a file a reader can open."""

    record: PathField | None = None
    project: PathField | None = None
    format: str = "html"
    dest: PathField | None = None
    offline: bool = True
    non_interactive: bool = False


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
