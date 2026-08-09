"""The adapter protocol, the manifest and the evidence envelope (D-34, D-36, section 7.1).

Three rules hold this module together.

The manifest is immutable and content addressed, and a capability in it is an observation. The
existence of a method establishes nothing: `CapabilityVector` refuses any explicit boolean set true
whose name is not proven true in `probed`, and it refuses it at construction, so an unproven claim
cannot be stored and read back later as if it had been checked. Nothing here calls `hasattr` on a
grader, and `Grader` is deliberately not `runtime_checkable`, because an `isinstance` against a
runtime-checkable protocol is exactly the `hasattr` sweep D-34 forbids.

A verdict is decoupled from a score (D-36). Nine values, of which only `scored` carries a number;
a timeout is a timeout and never a reward of zero.

The supervisor writes provenance, counters and timing. The graded code never does: the adapter
quarantines its stdout and stamps the envelope from what the sandbox reported.

Owned by P-ADAPT-PY.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from reward_lens.contracts import CapabilityUnavailable, Counters, Verdict, digest

__all__ = [
    "CAPABILITY_KEYS",
    "EXPLICIT_CAPABILITIES",
    "VERDICTS",
    "AdapterCapabilityUnproven",
    "CapabilityVector",
    "EvidenceEnvelope",
    "Grader",
    "InvalidInput",
    "Limits",
    "Manifest",
    "Result",
    "Sandbox",
    "ValidityFinding",
    "Verdict",
    "source_digest_of",
]


# --- the sandbox seam, interfaces section 3 ------------------------------------------------------
#
# P-EXEC owns these. It runs beside this packet and had not landed when this was written, so the
# frozen shapes are defined here behind the import and become P-EXEC's the moment P-EXEC lands.

try:  # pragma: no cover - which branch runs depends on whether P-EXEC has landed
    from reward_lens.execution import Limits, Result, Sandbox  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - the shim, interfaces section 3

    @dataclass(frozen=True)
    class Limits:  # type: ignore[no-redef]
        """Interfaces section 3. Defined here only until P-EXEC lands."""

        wall_s: float = 30
        cpu_s: float = 30
        memory_bytes: int = 2 * 2**30
        processes: int = 64
        network: bool = False
        stdout_bytes: int = 1 << 20
        artifact_bytes: int = 64 << 20

    @dataclass(frozen=True)
    class Result:  # type: ignore[no-redef]
        """Interfaces section 3. Defined here only until P-EXEC lands."""

        exit_code: int
        stdout: bytes
        stderr: bytes
        counters: Counters
        breach: Literal[
            "wall", "cpu", "memory", "processes", "stdout", "artifact"
        ] | None
        tier: str

    class Sandbox(Protocol):  # type: ignore[no-redef]
        """Interfaces section 3. Defined here only until P-EXEC lands."""

        def run(
            self,
            argv: list[str],
            *,
            limits: Limits,
            cwd: Path,
            env: dict[str, str],
            stdin: bytes | None = None,
        ) -> Result: ...


# --- verdicts (D-36 plus `scored`) ---------------------------------------------------------------

VERDICTS: tuple[str, ...] = (
    "scored",
    "invalid_input",
    "unscored",
    "grader_error",
    "timeout",
    "resource_exhausted",
    "provider_unavailable",
    "parse_error",
    "cancelled",
)

#: The verdicts that may carry a number. Everything else carries `None`, never zero.
SCORING_VERDICTS: frozenset[str] = frozenset({"scored"})


# --- refusals ------------------------------------------------------------------------------------


class AdapterCapabilityUnproven(CapabilityUnavailable):
    """RL0702, ADAPTER_CAPABILITY_UNPROVEN, exit 5.

    Raised when a manifest claims a capability the probe did not establish, and when the adapter
    itself cannot be established: a grader file that is not there, or that has no entrypoint of the
    declared shape, is a `plain`-shape capability that nothing has proven.
    """

    default_exit_code = 5

    def __init__(self, *, capability: str, message: str, remediation: str = "") -> None:
        super().__init__(
            code="RL0702",
            message=message,
            remediation=remediation or f"probe the grader before claiming {capability}",
            context={"capability": capability},
        )
        self.capability = capability


# --- the capability vector (section 7.1) ---------------------------------------------------------

#: The explicit booleans section 7.1 names. Each is a claim; each needs a proof in `probed`.
EXPLICIT_CAPABILITIES: tuple[str, ...] = (
    "component_dag",
    "token_quantities",
    "checkpoint_capture",
    "controlled_updates",
    "cost_metering",
)

#: The probe's own vocabulary, research note 03's capability vector. A key missing from `probed`
#: is not a false claim, it is no claim at all, and the adapter fills every one of these.
CAPABILITY_KEYS: tuple[str, ...] = (
    "source_visible",
    "locally_replayable",
    "controls_execution",
    "returns_components",
    "has_mutation_surface",
    "outcome_independent",
    "state_observable",
    "protected_partition",
    "cost_metered",
)


@dataclass(frozen=True)
class CapabilityVector:
    """Explicit booleans plus the probe that has to have proven each true one."""

    component_dag: bool = False
    token_quantities: bool = False
    checkpoint_capture: bool = False
    controlled_updates: bool = False
    cost_metering: bool = False
    probed: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probed", {str(k): bool(v) for k, v in self.probed.items()})
        for name in EXPLICIT_CAPABILITIES:
            if getattr(self, name) is not True:
                continue
            if self.probed.get(name) is not True:
                raise AdapterCapabilityUnproven(
                    capability=name,
                    message=(
                        f"the manifest claims {name}, and the probe did not establish it; "
                        "a capability absent from the probe cannot be claimed"
                    ),
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in EXPLICIT_CAPABILITIES},
            "probed": dict(sorted(self.probed.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityVector:
        return cls(
            **{name: bool(data.get(name, False)) for name in EXPLICIT_CAPABILITIES},
            probed=dict(data.get("probed") or {}),
        )


# --- the manifest (D-34) -------------------------------------------------------------------------


def source_digest_of(source: str | None) -> str:
    """The SHA-256 a manifest records for a grader's source, through `canonical` alone (D-66).

    A consumer that builds a manifest from interfaces section 4's fields hands over no source, and
    this is what such a manifest records: the digest of the empty source, computed exactly the way
    a real one is. It is a digest of what the manifest was given, never a stand-in for a digest of
    something it was not.
    """
    return digest({"source": source or ""})


@dataclass(frozen=True)
class Manifest:
    """D-34's manifest. Immutable, content addressed, and digested only through `canonical`."""

    family: str
    implementation_revision: str
    input_schema_digest: str
    output_schema_digest: str
    #: Computed from `source` when it is not given. `source_digest` precedes `source`: a caller that
    #: already holds the digest passes it, and a caller that holds the text passes the text.
    source_digest: str = ""
    fixture_digest: str | None = None
    image_digest: str | None = None
    shape: str = "plain"
    score_domain: str = "[0,1]"
    score_direction: str = "higher_is_better"
    aggregation: str | None = None
    determinism_class: str = "declared_deterministic"
    state_model: str = "stateless"
    access_level: str = "source_visible"
    declared_inputs: tuple[str, ...] = ()
    declared_outputs: tuple[str, ...] = ()
    network_policy: str = "deny"
    secret_policy: str = "none"
    resource_needs: dict[str, Any] = field(default_factory=dict)
    replay_mode: str = "local_replay_unproven"
    capabilities: CapabilityVector = field(default_factory=CapabilityVector)
    #: Init-only: the grader's text, used to compute `source_digest` and never stored in the record.
    source: dataclasses.InitVar[str | None] = None

    def __post_init__(self, source: str | None = None) -> None:
        # Kept for `dataclasses.replace`, which passes every init argument through. It is not a
        # field: `fields()`, `to_dict()`, `digest()` and `__eq__` never see it.
        object.__setattr__(self, "source", source)
        if not self.source_digest:
            object.__setattr__(self, "source_digest", source_digest_of(source))
        object.__setattr__(self, "declared_inputs", tuple(self.declared_inputs))
        object.__setattr__(self, "declared_outputs", tuple(self.declared_outputs))
        object.__setattr__(self, "resource_needs", dict(self.resource_needs))
        # Re-run the capability rule: a manifest built from a dict must refuse the same claims.
        CapabilityVector.from_dict(self.capabilities.to_dict())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "capabilities":
                out[f.name] = value.to_dict()
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        kwargs = dict(data)
        kwargs.pop("source", None)  # never a stored field; a dict that carries one is not a record
        caps = kwargs.pop("capabilities", None)
        return cls(
            **kwargs,
            capabilities=CapabilityVector.from_dict(caps or {}),
        )

    def digest(self) -> str:
        """`sha256:<hex>` over RFC 8785 bytes of `to_dict()`. The only digest path (D-66)."""
        return digest(self.to_dict())


# --- input validation ----------------------------------------------------------------------------


@dataclass(frozen=True)
class InvalidInput:
    """What `validate_input` returns instead of raising: the field and why it was refused."""

    field: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "reason": self.reason}


# --- the three validity findings of D-65 ---------------------------------------------------------


@dataclass(frozen=True)
class ValidityFinding:
    """One of RL0210, RL0211 or RL0212, with the witness that shows it happened.

    `trainer_behaviour` is the point: the finding says what the named framework would have done
    with this return, because silently absorbing it is the failure being reported.
    """

    code: str
    rule: str
    level: str
    trainer: str
    trainer_behaviour: str
    message: str
    witness: dict[str, Any]
    scope: str = "grader"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidityFinding:
        return cls(**dict(data))


# --- the evidence envelope (D-34) ----------------------------------------------------------------

#: `to_dict()` writes exactly these keys, in this order: interfaces section 4's frozen list.
ENVELOPE_KEYS: tuple[str, ...] = (
    "run_id",
    "subject_digest",
    "manifest_digest",
    "request_digest",
    "start",
    "end",
    "seed",
    "verdict",
    "score",
    "components",
    "observations",
    "artifacts",
    "counters",
    "provenance",
    "errors",
    "limitations",
)

#: Dropped by `normalised()`: everything that changes between two honest runs of the same input.
_VOLATILE = ("run_id", "start", "end", "counters")


@dataclass(frozen=True)
class EvidenceEnvelope:
    """What `score` returns. The supervisor fills it; the graded code fills none of it."""

    run_id: str
    subject_digest: str
    manifest_digest: str
    request_digest: str
    start: str
    end: str
    seed: int | None
    verdict: str
    score: float | None
    components: dict[str, float] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Any, ...] = ()
    counters: Counters = field(default_factory=Counters)
    provenance: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    #: Additive to the frozen field list and not serialised under its own key: `to_dict()` puts
    #: these under `observations["validity_findings"]`, so the wire shape is the frozen one.
    findings: tuple[ValidityFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "findings", tuple(self.findings))
        if self.verdict not in VERDICTS:
            raise ValueError(f"{self.verdict!r} is not one of the nine verdicts")
        if self.verdict not in SCORING_VERDICTS and self.score is not None:
            raise ValueError(
                f"a {self.verdict} verdict carries no score; D-36 keeps the two decoupled"
            )

    def to_dict(self) -> dict[str, Any]:
        observations = dict(self.observations)
        observations["validity_findings"] = [f.to_dict() for f in self.findings]
        out: dict[str, Any] = {}
        for key in ENVELOPE_KEYS:
            if key == "observations":
                out[key] = observations
            elif key == "counters":
                out[key] = self.counters.model_dump(exclude_none=True)
            elif key == "artifacts":
                out[key] = list(self.artifacts)
            elif key in ("errors", "limitations"):
                out[key] = list(getattr(self, key))
            else:
                out[key] = getattr(self, key)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceEnvelope:
        observations = dict(data.get("observations") or {})
        findings = tuple(
            ValidityFinding.from_dict(f) for f in observations.pop("validity_findings", [])
        )
        return cls(
            run_id=data["run_id"],
            subject_digest=data["subject_digest"],
            manifest_digest=data["manifest_digest"],
            request_digest=data["request_digest"],
            start=data["start"],
            end=data["end"],
            seed=data["seed"],
            verdict=data["verdict"],
            score=data["score"],
            components=dict(data.get("components") or {}),
            observations=observations,
            artifacts=tuple(data.get("artifacts") or ()),
            counters=Counters.model_validate(data.get("counters") or {}),
            provenance=dict(data.get("provenance") or {}),
            errors=tuple(data.get("errors") or ()),
            limitations=tuple(data.get("limitations") or ()),
            findings=findings,
        )

    def normalised(self) -> dict[str, Any]:
        """The envelope with the parts that honestly differ between runs removed.

        Two clean replays of a deterministic grader must emit identical normalised evidence
        (research note 03). Run id, wall clock and counters are not evidence of disagreement.
        """
        out = self.to_dict()
        for key in _VOLATILE:
            out.pop(key, None)
        provenance = dict(out.get("provenance") or {})
        for key in ("started", "duration_s", "counters"):
            provenance.pop(key, None)
        out["provenance"] = provenance
        return out

    def digest(self) -> str:
        """`sha256:<hex>` over the normalised envelope, through `canonical` alone (D-66)."""
        return digest(self.normalised())


# --- the protocol (D-34) -------------------------------------------------------------------------


class Grader(Protocol):
    """Every family of D-35 implements this. Not `runtime_checkable`, by D-34.

    The optional members below are optional in the sense that a family may not define them, and a
    caller learns which from the manifest's capability vector, never from `hasattr`.
    """

    def describe(self) -> Manifest: ...

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput: ...

    def score(
        self,
        task: dict,
        response: str | dict,
        *,
        sandbox: Sandbox,
        limits: Limits,
    ) -> EvidenceEnvelope: ...

    # Optional, and declared in the manifest rather than discovered:
    def source(self) -> str | None: ...

    def reset(self) -> None: ...

    def replay(
        self,
        envelope: EvidenceEnvelope,
        *,
        sandbox: Sandbox,
        limits: Limits,
    ) -> EvidenceEnvelope: ...


def verdict_of(name: str) -> Verdict:
    """Narrow a string to the schema's `Verdict`, refusing anything outside the nine."""
    if name not in VERDICTS:
        raise ValueError(f"{name!r} is not one of the nine verdicts")
    return name  # type: ignore[return-value]


def sequence_of_str(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(values or ())
