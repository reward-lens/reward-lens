"""Turning a detection into a wave-1 `Grader`, and refusing an override that does not hold up.

`bind` is where `rewardlens.yaml` gets its say. `reward.shape` and `reward.entry` override what
detection found, in that they replace it rather than argue with it, and the evidence list records
that they did so the audit can show which line of the project file moved the result. An override
that names a family outside D-65's four, or an entry that does not resolve, refuses at exit 4 and
names the field, because a silently ignored override is how a project file stops describing the
run it configures.

The grader `bind` returns calls the entry in this process. A live Python object cannot be handed
to the sandbox, so the envelope says so in `limitations` rather than implying an isolation this
path does not have; pointing `reward.entry` at a file gets P-ADAPT-PY's sandboxed `PythonGrader`.

Owned by P-CONNECT.
"""

from __future__ import annotations

import asyncio
import datetime as _datetime
import inspect
from typing import Any, Mapping

from reward_lens.contracts import digest
from reward_lens.graders.base import (
    EvidenceEnvelope,
    InvalidInput,
    Manifest,
    ValidityFinding,
)

from .detect import _BATCH_PROBE, _call, detect, probe
from .errors import ShapeUnsupported, bad_override
from .shapes import DECLARED_SHAPES, Detection, Shape, shape_for_declared
from .verifiers_discriminator import is_group_rubric

__all__ = ["ConnectedGrader", "bind"]

_IN_PROCESS = (
    "the connected entry ran in this process: a live Python object cannot be handed to the "
    "sandbox. Point reward.entry at a file to get the sandboxed grader instead."
)


def _now() -> str:
    return _datetime.datetime.now(tz=_datetime.timezone.utc).isoformat(timespec="microseconds")


def _apply_overrides(detection: Detection, overrides: Mapping[str, Any] | None) -> Detection:
    if not overrides:
        return detection
    unknown = set(overrides) - {"shape", "entry"}
    if unknown:
        raise bad_override(
            "reward." + sorted(unknown)[0],
            f"{sorted(unknown)[0]} is not a key the connector reads; the two it reads are shape and entry",
        )

    if "entry" in overrides:
        value = overrides["entry"]
        if not isinstance(value, str) or ":" not in value:
            raise bad_override(
                "reward.entry",
                f"{value!r} is not module:callable and not a path to a reward file",
            )
        try:
            detection = detect(value, probe_entry=False)
        except Exception as caught:
            raise bad_override(
                "reward.entry", f"{value} does not resolve to a callable: {caught}"
            ) from caught
        detection.evidence.append(f"reward.entry: overridden to {value} by the project file")

    if "shape" in overrides:
        value = overrides["shape"]
        if not isinstance(value, str) or value not in DECLARED_SHAPES:
            raise bad_override(
                "reward.shape",
                f"{value!r} is not one of {', '.join(DECLARED_SHAPES)}",
            )
        target = detection.target
        awaits = inspect.iscoroutinefunction(target)
        group = is_group_rubric(target) if callable(target) else False
        detection.declared = value
        detection.shape = shape_for_declared(value, awaits=awaits, group=group)
        detection.awaits = awaits
        detection.evidence.append(f"reward.shape: overridden to {value} by the project file")
    return detection


class ConnectedGrader:
    """A `Grader` over a connected callable. D-34's manifest, D-36's verdicts, and no claims beyond."""

    def __init__(self, detection: Detection) -> None:
        self.detection = detection
        self._source = _source_of(detection.target)
        self._manifest = Manifest(
            family="connected",
            implementation_revision=f"connect/{detection.shape.value}",
            input_schema_digest=digest({"task": ["prompt"], "response": "str"}),
            output_schema_digest=digest({"score": "float"}),
            shape=detection.declared,
            score_domain="unbounded",
            determinism_class="declared_deterministic",
            state_model="stateless",
            access_level="source_visible",
            declared_inputs=("prompt", "completion"),
            declared_outputs=("score",),
            replay_mode="local_replay_unproven",
            source=self._source,
        )

    # --- the protocol ----------------------------------------------------------------------------

    def describe(self) -> Manifest:
        return self._manifest

    def validate_input(self, task: dict, response: str | dict) -> None | InvalidInput:
        if not isinstance(task, dict) or not task.get("prompt"):
            return InvalidInput(field="prompt", reason="the task carries no prompt to score against")
        if not isinstance(response, (str, dict)):
            return InvalidInput(field="response", reason="a response is text or a mapping")
        return None

    def source(self) -> str | None:
        return self._source

    def reset(self) -> None:
        return None

    def score(self, task: dict, response: str | dict, *, sandbox=None, limits=None) -> EvidenceEnvelope:
        refused = self.validate_input(task, response)
        start = _now()
        if refused is not None:
            return self._envelope(task, response, start, "invalid_input", None, [], errors=(refused.reason,))
        prompt = str(task.get("prompt", ""))
        text = response if isinstance(response, str) else str(response)
        row = self._row(prompt, text)
        try:
            returned = _call(self.detection, row)
        except Exception as caught:
            finding = probe_finding_for_exception(self.detection.entry, caught)
            return self._envelope(task, response, start, "grader_error", None, [finding],
                                  errors=(f"{type(caught).__name__}: {caught}",))
        value, findings = _one_value(self.detection.entry, returned)
        verdict = "scored" if value is not None else "unscored"
        return self._envelope(task, response, start, verdict, value, findings)

    def replay(self, envelope: EvidenceEnvelope, *, sandbox=None, limits=None) -> EvidenceEnvelope:
        task = dict(envelope.provenance.get("task", {}))
        response = envelope.provenance.get("response", "")
        return self.score(task, response, sandbox=sandbox, limits=limits)

    # --- the parts that are not the protocol -----------------------------------------------------

    def _row(self, prompt: str, text: str) -> dict[str, Any]:
        if self.detection.shape in (Shape.BATCH_FN, Shape.ASYNC_TRL):
            row = dict(_BATCH_PROBE)
            row.update(
                prompts=[prompt],
                completions=[text],
                completion_ids=[[0]],
                answer=[""],
                answers=[""],
                states=[{}],
                tasks=[prompt],
                infos=[{}],
            )
            return row
        return {"prompt": prompt, "completion": text, "target": "", "answer": "", "state": {},
                "task": prompt, "info": {}}

    def _envelope(self, task, response, start, verdict, score, findings, errors=()) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            run_id=digest({"entry": self.detection.entry, "start": start}),
            subject_digest=digest({"response": response if isinstance(response, str) else dict(response)}),
            manifest_digest=self._manifest.digest(),
            request_digest=digest({"task": task, "response": response if isinstance(response, str) else dict(response)}),
            start=start,
            end=_now(),
            seed=None,
            verdict=verdict,
            score=score,
            provenance={"entry": self.detection.entry, "shape": self.detection.shape.value,
                        "declared": self.detection.declared, "execution": "in_process",
                        "task": dict(task) if isinstance(task, dict) else {}, "response": response},
            errors=tuple(errors),
            limitations=(_IN_PROCESS,),
            findings=tuple(findings),
        )


def probe_finding_for_exception(entry: str, caught: BaseException) -> ValidityFinding:
    return ValidityFinding(
        code="RL0214",
        rule="exception_to_zero",
        level="warning",
        trainer="verifiers",
        trainer_behaviour="Rubric logs the exception and records 0.0 for that reward function",
        message=f"{entry} raised {type(caught).__name__} while scoring",
        witness={"called": True, "entry": entry, "exception": type(caught).__name__,
                 "detail": str(caught), "would_have_scored": 0.0},
    )


def _one_value(entry: str, returned: Any) -> tuple[float | None, list[ValidityFinding]]:
    value = returned[0] if isinstance(returned, list) and returned else returned
    findings: list[ValidityFinding] = []
    if value is None:
        findings.append(
            ValidityFinding(
                code="RL0213", rule="none_to_nan", level="warning", trainer="trl",
                trainer_behaviour="GRPOTrainer turns the None into NaN and drops that reward function for the row",
                message=f"{entry} returned None for this row",
                witness={"called": True, "entry": entry, "returned": None, "index": 0,
                         "would_have_become": "nan"},
            )
        )
        return None, findings
    if isinstance(value, bool):
        findings.append(
            ValidityFinding(
                code="RL0215", rule="bool_as_score", level="warning", trainer="trl",
                trainer_behaviour="float() accepts the boolean, so 1.0 and 0.0 are recorded as though measured",
                message=f"{entry} returned a bool for this row",
                witness={"called": True, "entry": entry, "returned": value, "index": 0,
                         "float_of_it": float(value)},
            )
        )
    return float(value), findings


def _source_of(target: Any) -> str | None:
    try:
        return inspect.getsource(target if callable(target) else type(target))
    except (OSError, TypeError):
        return None


def bind(detection: Detection, *, overrides: Mapping[str, Any] | None = None):
    """Return the wave-1 `Grader` for a detection, after the project file has had its say."""
    resolved = _apply_overrides(detection, overrides)
    if resolved.shape is Shape.UNSUPPORTED:
        raise ShapeUnsupported(
            shape=resolved.unsupported_label,
            entry=resolved.entry,
            detail=resolved.reason,
        )
    return ConnectedGrader(resolved)


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = (asyncio, probe)
