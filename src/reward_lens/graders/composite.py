"""A record-preserving weighted composite grader.

The aggregate is deliberately the last field calculated.  Each child envelope and
its raw score are recorded first, then the declared transformation is applied and
recorded separately.  This makes the result usable by trace and statistics
readers without letting a dashboard scalar erase the component evidence.
"""

from __future__ import annotations

import datetime as _datetime
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

from reward_lens.contracts import Counters, digest
from reward_lens.graders.base import (
    AdapterCapabilityUnproven,
    CapabilityVector,
    EvidenceEnvelope,
    Grader,
    InvalidInput,
    Limits,
    Manifest,
    Sandbox,
)

__all__ = [
    "Component",
    "CompositeGrader",
    "ComponentValue",
    "Reconstruction",
    "reconstruct",
]


MissingPolicy = Literal["zero", "exclude", "refuse"]
Normalisation = Literal["identity", "binary", "clamp_0_1"]
Gate = Callable[[Mapping[str, "ComponentValue"]], bool]


@dataclass(frozen=True)
class Component:
    """One named node in the ordered component DAG.

    ``missing`` is intentionally a visible policy.  Its safe default is
    ``refuse``; passing ``None`` is an undeclared policy and is rejected before
    any grader is called.  ``exclude`` also produces no scalar aggregate: it is
    a recorded hole, not permission to manufacture a partial total.
    """

    name: str
    grader: Grader
    weight: float = 1.0
    depends_on: tuple[str, ...] = ()
    gate: Gate | None = None
    normalisation: Normalisation = "identity"
    clip: tuple[float, float] | None = None
    missing: MissingPolicy | None = "refuse"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a component name is required")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if not math.isfinite(float(self.weight)):
            raise ValueError(f"component {self.name!r} has a non-finite weight")
        if self.normalisation not in ("identity", "binary", "clamp_0_1"):
            raise ValueError(
                f"component {self.name!r} has unknown normalisation {self.normalisation!r}"
            )
        if self.missing not in ("zero", "exclude", "refuse"):
            raise ValueError(f"component {self.name!r} has no declared missing-value behaviour")
        _validate_clip(self.clip, f"component {self.name!r}")


@dataclass(frozen=True)
class ComponentValue:
    """The actual outcome of one component, available to later gates only."""

    name: str
    raw_score: float | None
    verdict: str | None
    gate_passed: bool
    missing_action: str | None


@dataclass(frozen=True)
class Reconstruction:
    """The independent algebraic check of an already emitted component record."""

    recorded_aggregate: float | None
    reconstructed_aggregate: float | None
    matches: bool
    finding: dict[str, Any] | None = None


def _utc_now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="microseconds")


def _validate_clip(clip: tuple[float, float] | None, subject: str) -> None:
    if clip is None:
        return
    if len(clip) != 2 or not all(math.isfinite(float(value)) for value in clip):
        raise ValueError(f"{subject} clip must contain two finite bounds")
    if float(clip[0]) > float(clip[1]):
        raise ValueError(f"{subject} clip lower bound exceeds upper bound")


def _normalise(value: float, name: str) -> float:
    if name == "identity":
        return value
    if name == "binary":
        return 1.0 if value else 0.0
    if name == "clamp_0_1":
        return min(1.0, max(0.0, value))
    raise ValueError(f"unknown normalisation {name!r}")


def _clip(value: float, bounds: Sequence[float] | None) -> float:
    if bounds is None:
        return value
    return min(float(bounds[1]), max(float(bounds[0]), value))


def _integrity_material(
    *,
    declared: Mapping[str, Any],
    actual: Mapping[str, Any],
    score: float | None,
    verdict: str,
    components: Mapping[str, float],
) -> dict[str, Any]:
    """Everything the composite record uses as evidence, in one hashed payload."""

    return {
        "declared": _thaw(declared),
        "actual": _thaw(actual),
        "envelope_score": score,
        "verdict": verdict,
        "raw_components": _thaw(components),
    }


class _FrozenMapping(MappingABC[str, Any]):
    """A non-``dict`` immutable JSON-value mapping for issued public evidence."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = MappingProxyType({key: _freeze(value) for key, value in values.items()})

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):  # noqa: ANN204 - inherited Mapping iterator protocol
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"_FrozenMapping({dict(self._values)!r})"


def _freeze(value: Any) -> Any:
    """Recursively make the public component evidence immutable."""

    if isinstance(value, MappingABC):
        return _FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """The explicit serialization/canonicalization boundary for frozen evidence."""

    if isinstance(value, MappingABC):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw(item) for item in sorted(value, key=repr)]
    return value


class _CompositeEvidenceEnvelope(EvidenceEnvelope):
    """An envelope whose public values are frozen and whose wire values are plain JSON."""

    def to_dict(self) -> dict[str, Any]:
        wire = super().to_dict()
        for name in ("components", "observations", "provenance"):
            wire[name] = _thaw(getattr(self, name))
        return wire


class CompositeGrader:
    """A wave-1 ``Grader`` over an ordered DAG of child graders.

    Child identities and capability vectors are copied as records.  They are not
    promoted into claims about the composite: in particular, a remote child does
    not give this wrapper source visibility, containment, or local replay.
    """

    def __init__(
        self,
        *,
        components: Sequence[Component],
        final_clip: tuple[float, float] | None = None,
        probed: Mapping[str, bool] | None = None,
        implementation_revision: str = "composite/v1",
    ) -> None:
        self.components = tuple(components)
        if not self.components:
            raise ValueError("a composite needs at least one component")
        _validate_clip(final_clip, "composite")
        self.final_clip = final_clip
        self._validate_dag()
        probe_record = {str(name): bool(value) for name, value in (probed or {}).items()}
        if probe_record.get("component_dag") is not True:
            raise AdapterCapabilityUnproven(
                capability="component_dag",
                message=(
                    "the composite manifest would claim component_dag, but no passing "
                    "component-DAG probe was recorded"
                ),
            )
        self._manifest = Manifest(
            family="COMPOSITE",
            implementation_revision=implementation_revision,
            input_schema_digest=digest({"task": "mapping", "response": ["str", "mapping"]}),
            output_schema_digest=digest(
                {"components": "component-record/v1", "aggregate": "float"}
            ),
            aggregation="weighted_sum",
            score_domain="unbounded"
            if final_clip is None
            else f"[{final_clip[0]},{final_clip[1]}]",
            determinism_class="inherits_component_determinism",
            state_model="component_declared",
            access_level="component_access_recorded",
            declared_inputs=("task", "response"),
            declared_outputs=("component_records", "aggregate"),
            replay_mode="record_reconstruction",
            capabilities=CapabilityVector(component_dag=True, probed=probe_record),
        )

    def _validate_dag(self) -> None:
        seen: set[str] = set()
        for component in self.components:
            if component.name in seen:
                raise ValueError(f"component {component.name!r} appears more than once")
            absent = set(component.depends_on) - seen
            if absent:
                if set(component.depends_on) - {node.name for node in self.components}:
                    raise ValueError(
                        f"component {component.name!r} depends on unknown components {sorted(absent)!r}"
                    )
                raise ValueError(
                    f"component {component.name!r} must appear after dependencies {sorted(absent)!r}"
                )
            seen.add(component.name)

    def describe(self) -> Manifest:
        return self._manifest

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput:
        if not isinstance(task, dict):
            return InvalidInput(
                field="task", reason="a composite passes a mapping to each component"
            )
        if not isinstance(response, (str, dict)):
            return InvalidInput(field="response", reason="a response is text or a mapping")
        return None

    def source(self) -> str | None:
        # A graph may include remote children. Returning the wrapper source would
        # invite a consumer to read it as source access to every child.
        return None

    def reset(self) -> None:
        for component in self.components:
            component.grader.reset()

    def score(
        self, task: dict, response: str | dict, *, sandbox: Sandbox, limits: Limits
    ) -> EvidenceEnvelope:
        started = _utc_now()
        invalid = self.validate_input(task, response)
        if invalid is not None:
            return self._envelope(
                task, response, started, "invalid_input", None, (), (), errors=(invalid.reason,)
            )

        values: dict[str, ComponentValue] = {}
        rows: list[dict[str, Any]] = []
        numeric_components: dict[str, float] = {}
        incomplete: list[str] = []
        for component in self.components:
            gate_passed = component.gate(values) if component.gate is not None else True
            declared = self._declared_component(component)
            if not gate_passed:
                values[component.name] = ComponentValue(
                    component.name, None, None, False, "excluded_by_gate"
                )
                rows.append(
                    {
                        "name": component.name,
                        "identity": declared["identity"],
                        "declared": declared,
                        "gate_passed": False,
                        "raw_score": None,
                        "verdict": None,
                        "normalised_score": None,
                        "weighted_score": 0.0,
                        "missing_action": "excluded_by_gate",
                        "evidence": None,
                    }
                )
                continue

            child = component.grader.score(task, response, sandbox=sandbox, limits=limits)
            raw = child.score if child.verdict == "scored" and child.score is not None else None
            if raw is not None and math.isfinite(float(raw)):
                numeric_components[component.name] = float(raw)
                normalised = _clip(_normalise(float(raw), component.normalisation), component.clip)
                weighted = component.weight * normalised
                action: str | None = None
            elif raw is not None:
                raw = None
                normalised = None
                weighted = None
                action = "non_finite_refused"
                incomplete.append(component.name)
            elif component.missing == "zero":
                normalised = 0.0
                weighted = 0.0
                action = "zero"
            else:
                normalised = None
                weighted = None
                action = str(component.missing)
                incomplete.append(component.name)

            values[component.name] = ComponentValue(
                component.name, raw, child.verdict, True, action
            )
            rows.append(
                {
                    "name": component.name,
                    "identity": declared["identity"],
                    "declared": declared,
                    "gate_passed": True,
                    "raw_score": raw,
                    "verdict": child.verdict,
                    "normalised_score": normalised,
                    "weighted_score": weighted,
                    "missing_action": action,
                    "evidence": child.to_dict(),
                }
            )

        if incomplete:
            aggregate = None
            pre_clip = None
            verdict = "unscored"
            errors = tuple(
                f"component {name!r} has no score under its declared missing-value behaviour"
                for name in incomplete
            )
        else:
            pre_clip = sum(float(row["weighted_score"]) for row in rows)
            aggregate = _clip(pre_clip, self.final_clip)
            verdict = "scored"
            errors = ()
        actual = {
            "component_order": [component.name for component in self.components],
            "components": rows,
            "pre_clip": pre_clip,
            "aggregate": aggregate,
        }
        return self._envelope(
            task, response, started, verdict, aggregate, numeric_components, actual, errors=errors
        )

    def replay(
        self, envelope: EvidenceEnvelope, *, sandbox: Sandbox, limits: Limits
    ) -> EvidenceEnvelope:
        """Replay from recorded request inputs when they are available."""
        task = envelope.provenance.get("task")
        response = envelope.provenance.get("response")
        if isinstance(task, dict) and isinstance(response, (str, dict)):
            return self.score(task, response, sandbox=sandbox, limits=limits)
        return envelope

    def reconstruct(self, envelope: EvidenceEnvelope) -> Reconstruction:
        return reconstruct(envelope)

    def _declared_component(self, component: Component) -> dict[str, Any]:
        manifest = component.grader.describe()
        return {
            "identity": {
                "name": component.name,
                "manifest_digest": manifest.digest(),
                "family": manifest.family,
            },
            "weight": component.weight,
            "depends_on": list(component.depends_on),
            "normalisation": component.normalisation,
            "clip": list(component.clip) if component.clip is not None else None,
            "missing": component.missing,
            # This is evidence copied from the child manifest, never a composite
            # claim that the wrapper has the child's source or containment.
            "child_capabilities": manifest.capabilities.to_dict(),
        }

    def _envelope(
        self,
        task: dict,
        response: str | dict,
        started: str,
        verdict: str,
        score: float | None,
        components: Mapping[str, float],
        actual: Mapping[str, Any],
        *,
        errors: tuple[str, ...] = (),
    ) -> EvidenceEnvelope:
        declared = {
            "schema": "composite/v1",
            "aggregation": "weighted_sum",
            "final_clip": list(self.final_clip) if self.final_clip is not None else None,
            "components": [self._declared_component(component) for component in self.components],
        }
        record = {"declared": declared, "actual": dict(actual)}
        record["integrity_digest"] = digest(
            _integrity_material(
                declared=declared,
                actual=actual,
                score=score,
                verdict=verdict,
                components=components,
            )
        )
        return _CompositeEvidenceEnvelope(
            run_id=digest({"manifest": self._manifest.digest(), "start": started}),
            subject_digest=digest({"response": response}),
            manifest_digest=self._manifest.digest(),
            request_digest=digest({"task": task, "response": response}),
            start=started,
            end=_utc_now(),
            seed=None,
            verdict=verdict,
            score=score,
            components=_freeze(dict(components)),
            observations=_freeze({"composite": record}),
            counters=Counters(),
            provenance=_freeze(
                {
                    "task": dict(task),
                    "response": response,
                    "execution": "component_grader_records",
                }
            ),
            errors=errors,
        )


def reconstruct(envelope: EvidenceEnvelope) -> Reconstruction:
    """Recompute a composite aggregate from its emitted component records.

    This function does not call graders, read configuration, or use the stored
    weighted terms.  It re-applies only the declared transformation to the raw
    values that were emitted in the envelope, then compares that value with the
    aggregate the envelope carries.
    """

    try:
        composite = envelope.observations["composite"]
        declared = composite["declared"]
        actual = composite["actual"]
        recorded = actual.get("aggregate")
        expected_integrity = digest(
            _integrity_material(
                declared=declared,
                actual=actual,
                score=envelope.score,
                verdict=envelope.verdict,
                components=envelope.components,
            )
        )
        if composite.get("integrity_digest") != expected_integrity:
            return Reconstruction(
                recorded_aggregate=recorded,
                reconstructed_aggregate=None,
                matches=False,
                finding={
                    "kind": "composite_record_integrity",
                    "recorded_aggregate": recorded,
                    "reconstructed_aggregate": None,
                    "reason": "the composite record digest does not match its emitted evidence",
                },
            )
        declared_by_name = {item["identity"]["name"]: item for item in declared["components"]}
        terms: list[float] = []
        incomplete = False
        for row in actual["components"]:
            config = declared_by_name[row["name"]]
            if row["gate_passed"] is False:
                terms.append(0.0)
                continue
            raw = row["raw_score"]
            if raw is None:
                if config["missing"] == "zero":
                    terms.append(0.0)
                    continue
                incomplete = True
                continue
            normalised = _normalise(float(raw), config["normalisation"])
            normalised = _clip(normalised, config.get("clip"))
            terms.append(float(config["weight"]) * normalised)
        reconstructed = None if incomplete else _clip(sum(terms), declared.get("final_clip"))
    except (KeyError, TypeError, ValueError) as exc:
        return Reconstruction(
            recorded_aggregate=actual.get("aggregate") if "actual" in locals() else envelope.score,
            reconstructed_aggregate=None,
            matches=False,
            finding={
                "kind": "composite_reconstruction_unavailable",
                "recorded_aggregate": actual.get("aggregate")
                if "actual" in locals()
                else envelope.score,
                "reason": f"{type(exc).__name__}: {exc}",
            },
        )

    recorded = actual["aggregate"]
    matches = (
        recorded is None
        and reconstructed is None
        or recorded is not None
        and reconstructed is not None
        and math.isclose(recorded, reconstructed, rel_tol=1e-12, abs_tol=1e-12)
    )
    if matches:
        return Reconstruction(recorded, reconstructed, True)
    return Reconstruction(
        recorded_aggregate=recorded,
        reconstructed_aggregate=reconstructed,
        matches=False,
        finding={
            "kind": "composite_reconstruction_mismatch",
            "recorded_aggregate": recorded,
            "reconstructed_aggregate": reconstructed,
        },
    )
