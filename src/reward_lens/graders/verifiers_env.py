"""Execution-backed adapter for mixed verifiers rubrics."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import subprocess
import sys
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from reward_lens.contracts import digest
from reward_lens.graders.base import (
    AdapterCapabilityUnproven,
    CapabilityVector,
    EvidenceEnvelope,
    InvalidInput,
    Limits,
    Manifest,
)
from reward_lens.graders.connect.verifiers_discriminator import bound_names, is_group_rubric
from reward_lens.tap.adapters.verifiers import VERIFIERS_COMMIT, VerifiersConverter

__all__ = [
    "CapabilitySet",
    "DeterministicProbe",
    "EvidencePlane",
    "JudgeProbe",
    "RubricComponent",
    "StatefulProbe",
    "VerifiedWriterArtifact",
    "VerifiersEnvGrader",
]

_WRITER = "verifiers.utils.save_utils.save_outputs"
_RESULTS_FILE = "results.jsonl"
_VENDORED_WRITER = (
    Path(__file__).resolve().parents[1] / "_vendor" / "verifiers_writer" / "save_utils.py"
)
_PINNED_WRITER_SHA256 = "07cc179170e73d0b9587f58f7782d4865afdd235db89289ac21cdd5474d98820"


class EvidencePlane(str, Enum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    STATEFUL = "stateful"


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"probe values must be JSON-like, not {type(value).__name__}")


def _thawed(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) for v in value):
            return {k: _thawed(v) for k, v in value}
        return [_thawed(v) for v in value]
    return value


@dataclass(frozen=True)
class _Probe:
    inputs: Mapping[str, Any]
    expected_score: float

    def __post_init__(self) -> None:
        if isinstance(self.expected_score, bool) or not isinstance(
            self.expected_score, (int, float)
        ):
            raise TypeError("expected_score must be numeric")
        object.__setattr__(self, "inputs", _frozen(dict(self.inputs)))
        object.__setattr__(self, "expected_score", float(self.expected_score))

    def inputs_for_execution(self) -> dict[str, Any]:
        inputs = _thawed(self.inputs)
        if not isinstance(inputs, dict):
            raise TypeError("probe inputs did not retain mapping shape")
        return inputs


@dataclass(frozen=True)
class DeterministicProbe(_Probe):
    pass


@dataclass(frozen=True)
class JudgeProbe(_Probe):
    """A request for a real JudgeAdapter fixture exchange, never caller-owned judge metadata."""

    judge_adapter: object
    task: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "task", _frozen(dict(self.task)))

    def task_for_execution(self) -> dict[str, Any]:
        task = _thawed(self.task)
        if not isinstance(task, dict):
            raise TypeError("judge task did not retain mapping shape")
        return task


@dataclass(frozen=True)
class StatefulProbe(_Probe):
    terminal_key: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.terminal_key:
            raise ValueError("terminal_key must be nonempty")


@dataclass(frozen=True)
class RubricComponent:
    name: str
    function: Callable[..., Any]
    probe: DeterministicProbe | JudgeProbe | StatefulProbe


@dataclass(frozen=True)
class ProbeWitness:
    evidence_plane: EvidencePlane
    input_digest: str
    expected_score: float
    observed_score: float
    observed_digest: str
    evaluator_id: str | None = None
    request_digest: str | None = None
    response_digest: str | None = None
    state_before_digest: str | None = None
    state_after_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_plane": self.evidence_plane.value,
            "input_digest": self.input_digest,
            "expected_score": self.expected_score,
            "observed_score": self.observed_score,
            "observed_digest": self.observed_digest,
            "evaluator_id": self.evaluator_id,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "state_before_digest": self.state_before_digest,
            "state_after_digest": self.state_after_digest,
        }


@dataclass(frozen=True)
class CapabilitySet:
    name: str
    evidence_plane: EvidencePlane
    group_rubric: bool
    bound_inputs: frozenset[str]
    probe: ProbeWitness
    capabilities: CapabilityVector

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evidence_plane": self.evidence_plane.value,
            "group_rubric": self.group_rubric,
            "bound_inputs": sorted(self.bound_inputs),
            "probe": self.probe.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass(frozen=True, eq=False)
class VerifiedWriterArtifact:
    results_path: Path
    framework_revision: str
    writer: str
    command: tuple[str, ...]
    output_sha256: str
    writer_source_sha256: str
    trap_dependency_calls: int


@dataclass(frozen=True)
class _ObservedWriterExecution:
    artifact: VerifiedWriterArtifact
    completed_at: str


_OBSERVED_WRITERS: weakref.WeakKeyDictionary[VerifiedWriterArtifact, _ObservedWriterExecution] = (
    weakref.WeakKeyDictionary()
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterCapabilityUnproven(capability=field, message=f"{field} must be numeric")
    return float(value)


def _source(components: Iterable[RubricComponent]) -> str:
    found = []
    for component in components:
        try:
            value = inspect.getsource(component.function)
        except (OSError, TypeError):
            value = repr(component.function)
        found.append(f"{component.name}:{value}")
    return "\n".join(found)


_WRITER_PROGRAM = r"""
import importlib.util, json, sys, types
from pathlib import Path
traps = []
def trap(*args, **kwargs):
    traps.append("called")
    raise RuntimeError("stub dependency called")
def module(name, **attrs):
    item = types.ModuleType(name)
    for key, value in attrs.items(): setattr(item, key, value)
    sys.modules[name] = item
    return item
module("datasets", Dataset=type("Dataset", (), {"from_list": staticmethod(trap)}))
module("openai", AsyncOpenAI=type("AsyncOpenAI", (), {}))
module("pydantic", BaseModel=type("BaseModel", (), {}))
module("verifiers"); module("verifiers.utils")
module("verifiers.types", **{n: dict for n in ("ClientConfig", "GenerateMetadata", "GenerateOutputs", "Response", "RolloutOutput", "SamplingArgs", "State", "TokenUsage", "Tool")})
module("verifiers.utils.error_utils", error_data=trap, validate_error_data=trap)
module("verifiers.utils.message_utils", sanitize_tool_calls=trap, serialize_messages_for_output=trap)
module("verifiers.utils.metric_utils", **{n: type(n, (), {}) for n in ("EnvMetrics", "ErrorRateMetric", "FinalInputTokensMetric", "FinalOutputTokensMetric", "InputTokensMetric", "OutputTokensMetric", "PassAtKMetric", "RewardMetric")})
module("verifiers.utils.path_utils", get_results_path=trap)
module("verifiers.utils.usage_utils", StateUsageTracker=type("StateUsageTracker", (), {}), response_usage_tokens=trap)
module("verifiers.utils.version_utils", get_version_info=trap)
output, source, rows = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
spec = importlib.util.spec_from_file_location("verifiers.utils.save_utils", source)
writer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = writer
spec.loader.exec_module(writer)
writer.save_outputs(rows, output, mode="w")
print(json.dumps({"trap_dependency_calls": len(traps)}))
"""


class VerifiersEnvGrader:
    def __init__(
        self,
        *,
        manifest: Manifest,
        capability_sets: tuple[CapabilitySet, ...],
        converter: VerifiersConverter,
    ) -> None:
        self._manifest = manifest
        self.capability_sets = capability_sets
        self._converter = converter

    @staticmethod
    def _call(
        component: RubricComponent, inputs: dict[str, Any]
    ) -> tuple[float, frozenset[str], bool]:
        group = is_group_rubric(component.function)
        bound = frozenset(bound_names(component.function, inputs))
        try:
            value = component.function(**{name: inputs[name] for name in bound})
        except Exception as exc:
            raise AdapterCapabilityUnproven(
                capability=f"component:{component.name}.probe_execution",
                message=f"component:{component.name} raised {type(exc).__name__} during its probe",
            ) from exc
        return _number(value, f"component:{component.name}.probe_return"), bound, group

    @classmethod
    def _observe(cls, component: RubricComponent) -> CapabilitySet:
        probe = component.probe
        inputs = probe.inputs_for_execution()
        if isinstance(probe, DeterministicProbe):
            score, bound, group = cls._call(component, inputs)
            replay, _, _ = cls._call(component, probe.inputs_for_execution())
            if "completion" not in bound or not math.isclose(score, replay):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.deterministic_replay",
                    message=f"component:{component.name} did not produce a repeatable completion probe",
                )
            witness = ProbeWitness(
                EvidencePlane.DETERMINISTIC,
                digest(inputs),
                probe.expected_score,
                score,
                digest({"score": score, "replay": replay}),
            )
        elif isinstance(probe, JudgeProbe):
            try:
                judge_module = importlib.import_module("reward_lens.graders.judge")
                judge_type = getattr(judge_module, "JudgeAdapter")
            except (ImportError, AttributeError) as exc:
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.adapter_binding",
                    message="JUDGE requires the accepted reward_lens.graders.judge.JudgeAdapter seam",
                ) from exc
            if not isinstance(probe.judge_adapter, judge_type):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.adapter",
                    message=(
                        f"component:{component.name} needs an actual JudgeAdapter execution, "
                        "not evaluator/request/response metadata"
                    ),
                )
            prompt, completion = inputs.get("prompt"), inputs.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.inputs",
                    message=f"component:{component.name} judge probe needs string prompt and completion",
                )
            exchange = probe.judge_adapter.score_pair(
                probe.task_for_execution(), prompt, completion, sandbox=None, limits=Limits()
            )
            if exchange.verdict != "scored" or exchange.score is None:
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.verdict",
                    message=f"component:{component.name} judge exchange did not produce a scored verdict",
                )
            observations = exchange.observations
            identity = observations.get("judge_identity")
            raw_exchange = observations.get("raw_exchange")
            if not isinstance(identity, Mapping) or not isinstance(raw_exchange, Mapping):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.exchange",
                    message=f"component:{component.name} judge exchange omitted identity or raw exchange references",
                )
            request_digest, response_digest = (
                raw_exchange.get("request_digest"),
                raw_exchange.get("response_digest"),
            )
            if not isinstance(request_digest, str) or not isinstance(response_digest, str):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.exchange",
                    message=f"component:{component.name} judge exchange has no retained request/response digests",
                )
            try:
                raw_response = probe.judge_adapter.raw_artifact(response_digest)
                parsed_wrapper = json.loads(raw_response.decode("utf-8"))
                parsed = parsed_wrapper["judge_raw"]
            except (
                KeyError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                AttributeError,
            ) as exc:
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.parser",
                    message=f"component:{component.name} could not recover the JudgeAdapter response for parsing",
                ) from exc
            if not isinstance(parsed, Mapping) or parsed.get("verdict") not in {
                "left",
                "right",
                "tie",
            }:
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.parser",
                    message=f"component:{component.name} JudgeAdapter response has no valid verdict",
                )
            parsed_score = _number(parsed.get("score"), f"component:{component.name}.judge.parser")
            if not math.isclose(parsed_score, float(exchange.score)):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.score_link",
                    message=f"component:{component.name} parsed judge score differs from the adapter envelope",
                )
            score, bound, group = cls._call(component, {**inputs, "judge_score": parsed_score})
            if not {"prompt", "completion", "judge_score"}.issubset(bound) or not math.isclose(
                score, parsed_score
            ):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.judge.score_link",
                    message=(
                        f"component:{component.name} must bind judge_score and return the parsed "
                        "JudgeAdapter score"
                    ),
                )
            witness = ProbeWitness(
                EvidencePlane.JUDGE,
                digest(inputs),
                probe.expected_score,
                score,
                digest({"score": score}),
                evaluator_id=digest(dict(identity)),
                request_digest=request_digest,
                response_digest=response_digest,
            )
        elif isinstance(probe, StatefulProbe):
            state, task = inputs.get("state"), inputs.get("task")
            if not isinstance(state, dict) or not isinstance(task, dict):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.state_input",
                    message=f"component:{component.name} stateful probe needs mutable state and task objects",
                )
            before = digest(state)
            score, bound, group = cls._call(component, inputs)
            after = digest(state)
            if (
                not {"state", "task"}.issubset(bound)
                or before == after
                or state.get(probe.terminal_key) is not True
            ):
                raise AdapterCapabilityUnproven(
                    capability=f"component:{component.name}.state_transition",
                    message=f"component:{component.name} did not execute the required terminal state transition",
                )
            witness = ProbeWitness(
                EvidencePlane.STATEFUL,
                digest(inputs),
                probe.expected_score,
                score,
                digest({"score": score, "state": state}),
                state_before_digest=before,
                state_after_digest=after,
            )
        else:
            raise TypeError(f"unsupported probe {type(probe).__name__}")
        if not math.isclose(witness.observed_score, witness.expected_score):
            raise AdapterCapabilityUnproven(
                capability=f"component:{component.name}.expected_score",
                message=f"component:{component.name} observed {witness.observed_score}, not expected {witness.expected_score}",
            )
        return CapabilitySet(
            component.name,
            witness.evidence_plane,
            group,
            bound,
            witness,
            CapabilityVector(probed={"returns_components": True}),
        )

    @classmethod
    def from_rubric(cls, components: Iterable[RubricComponent]) -> "VerifiersEnvGrader":
        component_tuple = tuple(components)
        if not component_tuple:
            raise AdapterCapabilityUnproven(
                capability="rubric.components", message="VERIFIERS_ENV needs an observed component"
            )
        names = [component.name for component in component_tuple]
        if len(names) != len(set(names)):
            raise AdapterCapabilityUnproven(
                capability="rubric.components", message="components need distinct names"
            )
        sets = tuple(cls._observe(component) for component in component_tuple)
        manifest = Manifest(
            family="VERIFIERS_ENV",
            implementation_revision="reward-lens/graders/verifiers_env/2",
            source_digest=digest(
                {"rubric": _source(component_tuple), "framework": VERIFIERS_COMMIT}
            ),
            input_schema_digest=digest({"type": "verifiers.Rubric", "components": names}),
            output_schema_digest=digest({"type": "verifiers.results.jsonl", "metrics": names}),
            shape="verifiers",
            aggregation="framework_weighted_rubric",
            determinism_class="mixed_by_observed_capability_set",
            state_model="mixed_by_observed_capability_set",
            declared_inputs=("prompt", "completion", "state", "task"),
            declared_outputs=tuple(names),
            network_policy="deny",
            capabilities=CapabilityVector(probed={"returns_components": bool(sets)}),
        )
        return cls(
            manifest=manifest,
            capability_sets=sets,
            converter=VerifiersConverter(framework_version=VERIFIERS_COMMIT),
        )

    def describe(self) -> Manifest:
        return self._manifest

    def source(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def validate_input(self, task: Any, response: Any) -> None | InvalidInput:
        if not isinstance(task, Mapping):
            return InvalidInput(field="task", reason="a verifiers task must be an object")
        if not isinstance(response, (str, Mapping, list)):
            return InvalidInput(
                field="response", reason="a response must be text, an object, or messages"
            )
        return None

    def produce_verified_output(
        self, results_path: Path | str, *, rows: list[Mapping[str, Any]]
    ) -> VerifiedWriterArtifact:
        if not _VENDORED_WRITER.is_file():
            raise AdapterCapabilityUnproven(
                capability="writer.source", message=f"pinned writer missing: {_VENDORED_WRITER}"
            )
        if _sha256(_VENDORED_WRITER) != _PINNED_WRITER_SHA256:
            raise AdapterCapabilityUnproven(
                capability="writer.source_digest",
                message="pinned verifiers writer source digest differs",
            )
        root = Path(results_path)
        root.mkdir(parents=True, exist_ok=True)
        output = root / _RESULTS_FILE
        try:
            row_data = json.dumps(list(rows), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise AdapterCapabilityUnproven(
                capability="writer.rows", message="writer rows must be JSON serializable"
            ) from exc
        command = ("save_outputs", str(_VENDORED_WRITER), str(root), "mode=w")
        run = subprocess.run(
            [sys.executable, "-c", _WRITER_PROGRAM, str(root), str(_VENDORED_WRITER), row_data],
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode:
            raise AdapterCapabilityUnproven(
                capability="writer.execution",
                message=f"pinned save_outputs failed: {run.stderr.strip()[:300]}",
            )
        try:
            traps = int(json.loads(run.stdout.strip().splitlines()[-1])["trap_dependency_calls"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AdapterCapabilityUnproven(
                capability="writer.execution_receipt",
                message="writer did not return an execution observation",
            ) from exc
        if traps or not output.is_file():
            raise AdapterCapabilityUnproven(
                capability="writer.execution",
                message="writer called a trap dependency or did not create results.jsonl",
            )
        artifact = VerifiedWriterArtifact(
            root,
            VERIFIERS_COMMIT,
            _WRITER,
            command,
            _sha256(output),
            _sha256(_VENDORED_WRITER),
            traps,
        )
        _OBSERVED_WRITERS[artifact] = _ObservedWriterExecution(
            artifact, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        return artifact

    def _output_for(self, artifact: VerifiedWriterArtifact) -> Path:
        observed = _OBSERVED_WRITERS.get(artifact)
        if observed is None:
            raise AdapterCapabilityUnproven(
                capability="provenance.execution_receipt",
                message="caller-supplied writer metadata and hashes are not a machine-observed writer execution receipt",
            )
        output = observed.artifact.results_path / _RESULTS_FILE
        if (
            observed.artifact.framework_revision != VERIFIERS_COMMIT
            or observed.artifact.writer != _WRITER
            or observed.artifact.trap_dependency_calls != 0
            or not output.is_file()
            or not _VENDORED_WRITER.is_file()
            or observed.artifact.output_sha256 != _sha256(output)
            or observed.artifact.writer_source_sha256 != _sha256(_VENDORED_WRITER)
        ):
            raise AdapterCapabilityUnproven(
                capability="provenance.execution_receipt",
                message="observed writer artifact changed after its receipt",
            )
        return output

    def read_output(self, artifact: VerifiedWriterArtifact) -> EvidenceEnvelope:
        output = self._output_for(artifact)
        try:
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as exc:
            raise AdapterCapabilityUnproven(
                capability="results.jsonl", message="observed artifact is not JSONL"
            ) from exc
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise AdapterCapabilityUnproven(
                capability="results.jsonl", message="adapter requires exactly one object row"
            )
        row = rows[0]
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise AdapterCapabilityUnproven(
                capability="results.metrics", message="results.metrics is absent"
            )
        components = {
            item.name: _number(metrics.get(item.name), f"results.metrics.{item.name}")
            for item in self.capability_sets
        }
        reward = _number(row.get("reward"), "results.reward")
        trajectory = self._converter.rollout(row)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return EvidenceEnvelope(
            run_id=f"verifiers:{row.get('example_id', 'unknown')}",
            subject_digest=digest({"example_id": row.get("example_id")}),
            manifest_digest=self._manifest.digest(),
            request_digest="sha256:" + _sha256(output),
            start=now,
            end=now,
            seed=None,
            verdict="scored",
            score=reward,
            components=components,
            observations={
                "capability_sets": [item.to_dict() for item in self.capability_sets],
                "tap_converter": {
                    "trajectory_id": str(trajectory.id),
                    "task_ref": str(trajectory.task_ref),
                    "features": {str(key): value for key, value in trajectory.features.items()},
                },
            },
            artifacts=(str(output),),
            provenance={
                "framework_revision": artifact.framework_revision,
                "writer": artifact.writer,
                "command": list(artifact.command),
                "output_sha256": artifact.output_sha256,
                "receipt_scope": "local_process_execution",
                "origin_authenticated": False,
            },
            limitations=(
                "The receipt proves this process executed pinned writer source; it does not cryptographically authenticate external artifact origin.",
            ),
        )
