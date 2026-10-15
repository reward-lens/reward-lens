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
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from reward_lens import errors
from reward_lens.contracts import digest
from reward_lens.graders.base import (
    EvidenceEnvelope,
    InvalidInput,
    Manifest,
    ValidityFinding,
)

from .detect import _BATCH_PROBE, _call, detect, probe
from .errors import bad_override
from .shapes import DECLARED_SHAPES, Detection, Shape, shape_for_declared
from .verifiers_discriminator import is_group_rubric

if TYPE_CHECKING:
    from reward_lens.graders import reward_model

__all__ = [
    "ConnectedGrader",
    "ConnectorCapability",
    "ConnectorCapabilityReport",
    "ConnectorCapabilityState",
    "ConnectorSelectionRanking",
    "PretrainedModelBinding",
    "bind",
    "overrides_from_project_file",
]

_IN_PROCESS = (
    "the connected entry ran in this process: a live Python object cannot be handed to the "
    "sandbox. Point reward.entry at a file to get the sandboxed grader instead."
)

# The reward-model module must stay out of the connector's base import graph.  Its adapter accepts
# this serialized spelling and resolves it to ``ScoreDirection.HIGHER_IS_BETTER`` at construction.
_DEFAULT_PRETRAINED_DIRECTION: Any = "higher_is_better"


class ConnectorCapabilityState(StrEnum):
    """Whether this connector executed a particular measurement check."""

    AVAILABLE = "available"
    ABSENT = "absent"


class ConnectorSelectionRanking(StrEnum):
    """Observed score ordering, deliberately separate from an outcome judgment."""

    BASELINE_HIGHER = "baseline_higher"
    CANDIDATE_HIGHER = "candidate_higher"
    TIE = "tie"


@dataclass(frozen=True)
class PretrainedModelBinding:
    """The explicit tokenizer and scoring policy for an already-loaded reward head.

    A ``PRETRAINED_MODEL`` detection contains the head itself, but no trustworthy way to infer its
    tokenizer or score interpretation.  The caller therefore supplies this binding directly;
    this connector deliberately does not add a configuration or model-loading policy.
    """

    tokenizer: object
    template: str
    preprocessing: reward_model.Preprocessing
    transform: Callable[[float], float] | None = None
    direction: reward_model.ScoreDirection = _DEFAULT_PRETRAINED_DIRECTION


@dataclass(frozen=True)
class ConnectorCapability:
    """One adapter check, with executed evidence or the reason its scope is absent."""

    name: str
    state: ConnectorCapabilityState
    reason: str
    evidence: tuple[EvidenceEnvelope, ...] = ()
    repeatable: bool | None = None
    ranking: ConnectorSelectionRanking | None = None


@dataclass(frozen=True)
class ConnectorCapabilityReport:
    """Independent capability checks; an absence never darkens another check."""

    checks: tuple[ConnectorCapability, ...]

    def status(self, name: str) -> ConnectorCapability:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


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


def overrides_from_project_file(path: str | Path) -> dict[str, Any]:
    """Read the two connector-owned overrides from a ``rewardlens.yaml`` file.

    The complete project configuration has other owners.  This reader deliberately extracts
    only ``reward.shape`` and ``reward.entry`` so that the existing frozen ``bind`` boundary
    remains the place that validates and applies those overrides.
    """
    import yaml

    project_file = Path(path)
    try:
        document = yaml.safe_load(project_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as caught:
        raise bad_override(
            "rewardlens.yaml", f"could not read {project_file}: {caught}"
        ) from caught
    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise bad_override("rewardlens.yaml", "the project file must contain a mapping")
    reward = document.get("reward")
    if reward is None:
        return {}
    if not isinstance(reward, Mapping):
        raise bad_override("reward", "reward must contain a mapping")
    return {key: reward[key] for key in ("shape", "entry") if key in reward}


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
            return InvalidInput(
                field="prompt", reason="the task carries no prompt to score against"
            )
        if not isinstance(response, (str, dict)):
            return InvalidInput(field="response", reason="a response is text or a mapping")
        return None

    def source(self) -> str | None:
        return self._source

    def reset(self) -> None:
        return None

    def score(
        self, task: dict, response: str | dict, *, sandbox=None, limits=None
    ) -> EvidenceEnvelope:
        refused = self.validate_input(task, response)
        start = _now()
        if refused is not None:
            return self._envelope(
                task, response, start, "invalid_input", None, [], errors=(refused.reason,)
            )
        prompt = str(task.get("prompt", ""))
        text = response if isinstance(response, str) else str(response)
        row = self._row(prompt, text)
        try:
            returned = _call(self.detection, row)
        except Exception as caught:
            finding = probe_finding_for_exception(self.detection.entry, caught)
            return self._envelope(
                task,
                response,
                start,
                "grader_error",
                None,
                [finding],
                errors=(f"{type(caught).__name__}: {caught}",),
            )
        value, findings = _one_value(self.detection.entry, returned)
        verdict = "scored" if value is not None else "unscored"
        return self._envelope(task, response, start, verdict, value, findings)

    def replay(self, envelope: EvidenceEnvelope, *, sandbox=None, limits=None) -> EvidenceEnvelope:
        task = dict(envelope.provenance.get("task", {}))
        response = envelope.provenance.get("response", "")
        return self.score(task, response, sandbox=sandbox, limits=limits)

    def attach_outcome(self, envelope: EvidenceEnvelope, outcome: object) -> EvidenceEnvelope:
        """Bind an explicit evaluator result to this score without converting unknown to false."""
        from reward_lens.outcome import Outcome

        if type(outcome) is not Outcome:
            raise TypeError("outcome must be an explicit Outcome, never a boolean-like payload")
        observations = dict(envelope.observations)
        if "outcome" in observations:
            raise ValueError(
                "the envelope already carries an outcome; replacing it would hide its provenance"
            )
        observations["outcome"] = {
            "state": outcome.state.value,
            "reason": outcome.reason,
            "protocol_digest": outcome.protocol_digest,
            "task_validity": outcome.task_validity.value,
            "scope": outcome.scope,
        }
        return replace(envelope, observations=observations)

    def probe_capabilities(
        self,
        *,
        task: dict,
        baseline_response: str | dict,
        candidate_response: str | dict,
    ) -> ConnectorCapabilityReport:
        """Execute black-box checks and report the source-mutation hole only at its own scope."""
        first = self.score(task, baseline_response)
        repeated = self.score(task, baseline_response)
        candidate = self.score(task, candidate_response)
        repeat_evidence = (first, repeated)
        selection_evidence = (first, candidate)
        repeatability_ran = _comparison_completed(first, repeated)
        repeatable = _repeatability_result(first, repeated) if repeatability_ran else None
        selection_ran = first.verdict == "scored" and candidate.verdict == "scored"
        ranking = _selection_ranking(first, candidate) if selection_ran else None
        return ConnectorCapabilityReport(
            checks=(
                ConnectorCapability(
                    name="source_mutation",
                    state=ConnectorCapabilityState.ABSENT,
                    reason="the connected callable has no supplied source-mutation runner",
                ),
                ConnectorCapability(
                    name="black_box_repeatability",
                    state=(
                        ConnectorCapabilityState.AVAILABLE
                        if repeatability_ran
                        else ConnectorCapabilityState.ABSENT
                    ),
                    reason=(
                        "same-input scorer envelopes agree on verdict, score, and findings"
                        if repeatable is True
                        else "same-input scorer envelopes disagree on verdict, score, or findings"
                        if repeatable is False
                        else "a repeated scorer comparison could not be measured"
                    ),
                    evidence=repeat_evidence if repeatability_ran else (),
                    repeatable=repeatable,
                ),
                ConnectorCapability(
                    name="selection",
                    state=(
                        ConnectorCapabilityState.AVAILABLE
                        if selection_ran
                        else ConnectorCapabilityState.ABSENT
                    ),
                    reason=(
                        "baseline and candidate scores were ranked without an outcome claim"
                        if selection_ran
                        else "baseline or candidate scorer call did not produce a scored envelope"
                    ),
                    evidence=selection_evidence if selection_ran else (),
                    ranking=ranking,
                ),
            )
        )

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
        return {
            "prompt": prompt,
            "completion": text,
            "target": "",
            "answer": "",
            "state": {},
            "task": prompt,
            "info": {},
        }

    def _envelope(
        self, task, response, start, verdict, score, findings, errors=()
    ) -> EvidenceEnvelope:
        return EvidenceEnvelope(
            run_id=digest({"entry": self.detection.entry, "start": start}),
            subject_digest=digest(
                {"response": response if isinstance(response, str) else dict(response)}
            ),
            manifest_digest=self._manifest.digest(),
            request_digest=digest(
                {
                    "task": task,
                    "response": response if isinstance(response, str) else dict(response),
                }
            ),
            start=start,
            end=_now(),
            seed=None,
            verdict=verdict,
            score=score,
            provenance={
                "entry": self.detection.entry,
                "shape": self.detection.declared,
                "convention": self.detection.shape.value,
                "execution": "in_process",
                "task": dict(task) if isinstance(task, dict) else {},
                "response": response,
            },
            errors=tuple(errors),
            limitations=(_IN_PROCESS,),
            findings=tuple(findings),
        )


def _comparison_completed(first: EvidenceEnvelope, second: EvidenceEnvelope) -> bool:
    """Invalid inputs never reach a scorer, so they are not a repeatability measurement."""
    return first.verdict != "invalid_input" and second.verdict != "invalid_input"


def _repeatability_result(first: EvidenceEnvelope, second: EvidenceEnvelope) -> bool:
    """Compare only measured result payloads, never volatile run identifiers or timestamps."""
    return _result_payload(first) == _result_payload(second)


def _result_payload(envelope: EvidenceEnvelope) -> tuple[Any, ...]:
    return (
        envelope.verdict,
        envelope.score,
        tuple(finding.to_dict() for finding in envelope.findings),
    )


def _selection_ranking(
    baseline: EvidenceEnvelope, candidate: EvidenceEnvelope
) -> ConnectorSelectionRanking:
    assert baseline.score is not None and candidate.score is not None
    if baseline.score > candidate.score:
        return ConnectorSelectionRanking.BASELINE_HIGHER
    if candidate.score > baseline.score:
        return ConnectorSelectionRanking.CANDIDATE_HIGHER
    return ConnectorSelectionRanking.TIE


def probe_finding_for_exception(entry: str, caught: BaseException) -> ValidityFinding:
    return ValidityFinding(
        code="RL0214",
        rule="exception_to_zero",
        level="warning",
        trainer="verifiers",
        trainer_behaviour="Rubric logs the exception and records 0.0 for that reward function",
        message=errors.make("RL0214", entry=entry, exception=type(caught).__name__).message,
        witness={
            "called": True,
            "entry": entry,
            "exception": type(caught).__name__,
            "detail": str(caught),
            "would_have_scored": 0.0,
        },
    )


def _one_value(entry: str, returned: Any) -> tuple[float | None, list[ValidityFinding]]:
    value = returned[0] if isinstance(returned, list) and returned else returned
    findings: list[ValidityFinding] = []
    if value is None:
        findings.append(
            ValidityFinding(
                code="RL0213",
                rule="none_to_nan",
                level="warning",
                trainer="trl",
                trainer_behaviour="GRPOTrainer turns the None into NaN and drops that reward function for the row",
                message=errors.make("RL0213", entry=entry).message,
                witness={
                    "called": True,
                    "entry": entry,
                    "returned": None,
                    "index": 0,
                    "would_have_become": "nan",
                },
            )
        )
        return None, findings
    if isinstance(value, bool):
        findings.append(
            ValidityFinding(
                code="RL0215",
                rule="bool_as_score",
                level="warning",
                trainer="trl",
                trainer_behaviour="float() accepts the boolean, so 1.0 and 0.0 are recorded as though measured",
                message=errors.make("RL0215", entry=entry).message,
                witness={
                    "called": True,
                    "entry": entry,
                    "returned": value,
                    "index": 0,
                    "float_of_it": float(value),
                },
            )
        )
    return float(value), findings


def _source_of(target: Any) -> str | None:
    try:
        return inspect.getsource(target if callable(target) else type(target))
    except (OSError, TypeError):
        return None


def bind(
    detection: Detection,
    *,
    overrides: Mapping[str, Any] | None = None,
    pretrained: PretrainedModelBinding | None = None,
):
    """Return the wave-1 `Grader` for a detection, after the project file has had its say."""
    resolved = _apply_overrides(detection, overrides)
    if resolved.shape is Shape.UNSUPPORTED:
        raise errors.make(
            "RL0710",
            shape=resolved.unsupported_label,
            entry=resolved.entry,
            detail=resolved.reason,
        )
    if resolved.shape is Shape.PRETRAINED_MODEL:
        if pretrained is None:
            raise bad_override(
                "reward.pretrained",
                "a PRETRAINED_MODEL needs an explicit PretrainedModelBinding with its matching "
                "tokenizer, template, preprocessing, and score direction",
            )
        # Kept inside this branch so detection and every callable connector family retain their
        # framework-free base import path.  The adapter receives an already-loaded model; it does
        # not resolve a model name or make a provider request.
        from reward_lens.graders import reward_model

        return reward_model.RewardModel.from_pretrained(
            resolved.target,
            pretrained.tokenizer,
            template=pretrained.template,
            preprocessing=pretrained.preprocessing,
            transform=pretrained.transform,
            direction=pretrained.direction,
        )
    return ConnectedGrader(resolved)


def _unused() -> None:  # pragma: no cover - keeps the import list honest for linters
    _ = (asyncio, probe)
