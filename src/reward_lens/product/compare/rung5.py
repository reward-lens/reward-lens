"""Rung five's generated CPU short-fork conformance record."""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from reward_lens import contracts
from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.intervention.clamp import (
    ClampSpecification,
    PinnedShortForkExecutor,
    ShortForkResult,
    build_clamp_callback,
    fixture_digest,
    run_short_fork,
)
from reward_lens.learning.intervention.swap import (
    Intervention,
    MaterializedBatch,
    TRL110StateAdapter,
)

from .registry import CompareInputs, CompareResult, Rung, register
from .scorecard import ROWS, Absence, Scorecard, ScorecardRow

__all__ = [
    "CPUConformanceRecordInvalid",
    "CPUShortForkConfig",
    "R5_ENTRY_ID",
    "Rung5Input",
    "append_short_fork_assay",
    "build_cpu_short_fork",
    "compare_verified_short_fork",
    "compare_short_fork",
    "default_cpu_short_fork_config",
    "load_cpu_conformance_record",
    "run",
    "write_cpu_conformance_record",
]


R5_ENTRY_ID = "reward_statistics.short_fork_response"
_CONFIG_VERSION = 1
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "profile",
        "vocab",
        "prompts",
        "model",
        "training",
        "target",
        "useful_task",
        "interventions",
        "limits",
        "subject_ref",
        "sandbox_tier",
        "offline",
    }
)
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")


def _config_error(field: str, detail: str) -> ValueError:
    return ValueError(f"CPU_SHORT_FORK_CONFIG_INVALID: {field} {detail}")


def _mapping(value: Any, field: str, *, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _config_error(field, "must be an object")
    found = set(value)
    if found != keys:
        missing = sorted(keys - found)
        extra = sorted(found - keys)
        raise _config_error(field, f"has missing={missing!r} extra={extra!r}")
    return dict(value)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _config_error(field, f"must be an integer at least {minimum}")
    return value


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive finite number" if positive else "finite number"
        raise _config_error(field, f"must be a {qualifier}")
    return float(value)


@dataclass(frozen=True)
class CPUShortForkConfig:
    """Explicit bounded inputs for one actual pinned CPU short-fork execution.

    The mapping form is the public JSON contract. It records only execution
    inputs; the factory creates provenance at execution time and returns the
    resulting retained ``Rung5Input``.
    """

    seed: int
    profile: TrainerProfile
    vocab: tuple[str, ...]
    prompts: tuple[str, ...]
    model: Mapping[str, int]
    training: Mapping[str, int | float]
    target: Mapping[str, Any]
    useful_task: Mapping[str, str]
    interventions: Mapping[str, tuple[float, ...]]
    limits: Mapping[str, int | float]
    subject_ref: str
    sandbox_tier: str = "T0"
    offline: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CPUShortForkConfig":
        data = _mapping(raw, "config", keys=set(_CONFIG_KEYS))
        if _integer(data["schema_version"], "schema_version", minimum=1) != _CONFIG_VERSION:
            raise _config_error("schema_version", f"must equal {_CONFIG_VERSION}")
        seed = _integer(data["seed"], "seed")
        profile_raw = _mapping(
            data["profile"], "profile", keys={"version", "mutable_objects", "fingerprint"}
        )
        if profile_raw["version"] != "1.10.0":
            raise _config_error("profile.version", "must equal '1.10.0'")
        if profile_raw["mutable_objects"] != list(EIGHT_OBJECTS):
            raise _config_error("profile.mutable_objects", "must equal the pinned ordered profile")
        if not isinstance(profile_raw["fingerprint"], str) or not profile_raw["fingerprint"]:
            raise _config_error("profile.fingerprint", "must be a non-empty string")
        profile = TrainerProfile(
            profile_raw["version"],
            tuple(profile_raw["mutable_objects"]),
            profile_raw["fingerprint"],
        )
        vocab = data["vocab"]
        if (
            not isinstance(vocab, list)
            or len(vocab) < 4
            or any(not isinstance(token, str) or not token for token in vocab)
            or len(set(vocab)) != len(vocab)
        ):
            raise _config_error("vocab", "must be a unique non-empty token list")
        if not {"<pad>", "<unk>", "<eos>"}.issubset(vocab):
            raise _config_error("vocab", "must include <pad>, <unk>, and <eos>")
        prompts = data["prompts"]
        if (
            not isinstance(prompts, list)
            or len(prompts) < 2
            or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
        ):
            raise _config_error("prompts", "must contain at least two non-empty prompts")
        model = _mapping(
            data["model"], "model", keys={"n_positions", "n_embd", "n_layer", "n_head"}
        )
        for key in model:
            model[key] = _integer(model[key], f"model.{key}", minimum=1)
        training = _mapping(
            data["training"],
            "training",
            keys={"batch_size", "generations", "completion_length", "learning_rate"},
        )
        for key in ("batch_size", "generations", "completion_length"):
            training[key] = _integer(training[key], f"training.{key}", minimum=1)
        training["learning_rate"] = _finite(
            training["learning_rate"], "training.learning_rate", positive=True
        )
        if training["batch_size"] != len(prompts) or training["generations"] != len(prompts):
            raise _config_error(
                "training", "must use the prompt count for batch_size and generations"
            )
        target = _mapping(
            data["target"], "target", keys={"name", "embedding_index", "value", "tolerance"}
        )
        if not isinstance(target["name"], str) or not target["name"]:
            raise _config_error("target.name", "must be a non-empty string")
        index = target["embedding_index"]
        if not isinstance(index, list) or len(index) != 2:
            raise _config_error("target.embedding_index", "must be a two-element integer list")
        row, column = (
            _integer(index[0], "target.embedding_index[0]"),
            _integer(index[1], "target.embedding_index[1]"),
        )
        if row >= len(vocab) or column >= model["n_embd"]:
            raise _config_error(
                "target.embedding_index", "must address the configured embedding matrix"
            )
        target["embedding_index"] = [row, column]
        target["value"] = _finite(target["value"], "target.value")
        target["tolerance"] = _finite(target["tolerance"], "target.tolerance", positive=True)
        useful_task = _mapping(data["useful_task"], "useful_task", keys={"name", "prompt", "token"})
        if any(
            not isinstance(useful_task[key], str) or not useful_task[key] for key in useful_task
        ):
            raise _config_error("useful_task", "values must be non-empty strings")
        if useful_task["token"] not in vocab:
            raise _config_error("useful_task.token", "must be in vocab")
        interventions_raw = _mapping(
            data["interventions"], "interventions", keys={"null", "candidate", "placebo"}
        )
        interventions: dict[str, tuple[float, ...]] = {}
        for name, offsets in interventions_raw.items():
            if not isinstance(offsets, list) or len(offsets) != len(prompts):
                raise _config_error(
                    f"interventions.{name}", "must contain one finite offset per prompt"
                )
            interventions[name] = tuple(
                _finite(offset, f"interventions.{name}") for offset in offsets
            )
        limits = _mapping(
            data["limits"],
            "limits",
            keys={"steps", "cpu_threads", "max_correction", "max_iterations"},
        )
        if _integer(limits["steps"], "limits.steps", minimum=1) != 1:
            raise _config_error("limits.steps", "must equal one")
        if _integer(limits["cpu_threads"], "limits.cpu_threads", minimum=1) != 1:
            raise _config_error("limits.cpu_threads", "must equal one")
        limits["max_correction"] = _finite(
            limits["max_correction"], "limits.max_correction", positive=True
        )
        limits["max_iterations"] = _integer(
            limits["max_iterations"], "limits.max_iterations", minimum=1
        )
        subject_ref = data["subject_ref"]
        if not isinstance(subject_ref, str) or not _SHA256_REF.fullmatch(subject_ref):
            raise _config_error("subject_ref", "must be a sha256 reference")
        if data["sandbox_tier"] != "T0" or data["offline"] is not True:
            raise _config_error("execution", "must remain offline T0 CPU conformance")
        return cls(
            seed=seed,
            profile=profile,
            vocab=tuple(vocab),
            prompts=tuple(prompts),
            model=model,
            training=training,
            target=target,
            useful_task=useful_task,
            interventions=interventions,
            limits=limits,
            subject_ref=subject_ref,
            sandbox_tier=data["sandbox_tier"],
            offline=data["offline"],
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the stable JSON-ready public execution contract."""
        return {
            "schema_version": _CONFIG_VERSION,
            "seed": self.seed,
            "profile": {
                "version": self.profile.version,
                "mutable_objects": list(self.profile.mutable_objects),
                "fingerprint": self.profile.fingerprint,
            },
            "vocab": list(self.vocab),
            "prompts": list(self.prompts),
            "model": dict(self.model),
            "training": dict(self.training),
            "target": dict(self.target),
            "useful_task": dict(self.useful_task),
            "interventions": {name: list(offsets) for name, offsets in self.interventions.items()},
            "limits": dict(self.limits),
            "subject_ref": self.subject_ref,
            "sandbox_tier": self.sandbox_tier,
            "offline": self.offline,
        }

    def fixture_mapping(self) -> dict[str, Any]:
        """The execution-relevant contract retained beside the generated record."""
        data = self.to_mapping()
        return {
            "fixture": "rung-five-public-cpu-v1",
            "seed": data["seed"],
            "profile": data["profile"],
            "vocab": data["vocab"],
            "prompts": data["prompts"],
            "model": data["model"],
            "training": data["training"],
            "target": data["target"],
            "useful_task": data["useful_task"],
            "interventions": data["interventions"],
            "limits": data["limits"],
            "subject_ref": data["subject_ref"],
        }


def default_cpu_short_fork_config(*, subject_ref: str) -> CPUShortForkConfig:
    """Return the package-owned, explicit tiny pinned-TRL CPU configuration."""
    return CPUShortForkConfig.from_mapping(
        {
            "schema_version": _CONFIG_VERSION,
            "seed": 31,
            "profile": {
                "version": "1.10.0",
                "mutable_objects": list(EIGHT_OBJECTS),
                "fingerprint": "rung-five-public-cpu-v1",
            },
            "vocab": ["<pad>", "<unk>", "<eos>", "say", "one", "two"],
            "prompts": ["say one", "say two"],
            "model": {"n_positions": 32, "n_embd": 16, "n_layer": 1, "n_head": 1},
            "training": {
                "batch_size": 2,
                "generations": 2,
                "completion_length": 3,
                "learning_rate": 0.0001,
            },
            "target": {
                "name": "embedding_0_0",
                "embedding_index": [0, 0],
                "value": 0.5,
                "tolerance": 0.000001,
            },
            "useful_task": {"name": "next_token_logit", "prompt": "say one", "token": "two"},
            "interventions": {
                "null": [0.0, 0.0],
                "candidate": [0.25, -0.25],
                "placebo": [-0.25, 0.25],
            },
            "limits": {"steps": 1, "cpu_threads": 1, "max_correction": 2.0, "max_iterations": 4},
            "subject_ref": subject_ref,
            "sandbox_tier": "T0",
            "offline": True,
        }
    )


class CPUConformanceRecordInvalid(ValueError):
    """CPU_CONFORMANCE_RECORD_INVALID: a generated record is absent or changed."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"CPU_CONFORMANCE_RECORD_INVALID: {detail}")


@dataclass(frozen=True)
class Rung5Input:
    """The retained result and verified temporary artifact needed to produce R5 evidence."""

    result: ShortForkResult
    record_path: Path | str
    fixture_root: Path | str
    subject_ref: str
    provenance: contracts.EntryProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.result, ShortForkResult):
            raise TypeError("rung 5 canonical producer requires a ShortForkResult")
        if not isinstance(self.provenance, contracts.EntryProvenance):
            raise TypeError("rung 5 canonical producer requires actual EntryProvenance")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_fixture(config: CPUShortForkConfig, artifact_dir: Path) -> Path:
    """Materialize the explicit execution contract, never a test-owned source fixture."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not artifact_dir.is_dir():
        raise ValueError("CPU_SHORT_FORK_ARTIFACT_INVALID: artifact_dir is not a directory")
    _write_json(artifact_dir / "runtime-config.json", config.to_mapping())
    fixture_root = artifact_dir / "runtime-fixture"
    fixture_root.mkdir(exist_ok=True)
    _write_json(fixture_root / "fork.json", config.fixture_mapping())
    return fixture_root


def build_cpu_short_fork(config: CPUShortForkConfig, artifact_dir: Path | str) -> Rung5Input:
    """Execute the configured tiny pinned CPU fork and retain its verified artifact.

    This is the public R5 producer. It does not accept a saved result, use test
    fixtures, download models, or choose accelerator execution. Every arm runs
    one actual optimizer update through the accepted pinned replay adapter.
    """
    if not isinstance(config, CPUShortForkConfig):
        raise TypeError("rung 5 CPU producer requires CPUShortForkConfig")
    import importlib.metadata

    from reward_lens.learning.contracts import TrainerUnsupported

    try:
        version = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    if version != "1.10.0":
        raise TrainerUnsupported(version)
    # A caller may instantiate the dataclass directly. Reparse its public
    # representation so the same bounded schema governs every entry path.
    config = CPUShortForkConfig.from_mapping(config.to_mapping())
    destination = Path(artifact_dir)
    fixture_root = _runtime_fixture(config, destination)
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    import numpy as np
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    torch.set_num_threads(int(config.limits["cpu_threads"]))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    vocab = {token: index for index, token in enumerate(config.vocab)}
    raw_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=int(config.model["n_positions"]),
            n_ctx=int(config.model["n_positions"]),
            n_embd=int(config.model["n_embd"]),
            n_layer=int(config.model["n_layer"]),
            n_head=int(config.model["n_head"]),
            bos_token_id=vocab["<eos>"],
            eos_token_id=vocab["<eos>"],
            pad_token_id=vocab["<pad>"],
        )
    )
    row, column = config.target["embedding_index"]
    with torch.no_grad():
        model.get_input_embeddings().weight[int(row), int(column)] = float(config.target["value"])
    source_scorer_calls = 0

    def reward(prompts: Any, completions: Any, completion_ids: Any, **kwargs: Any) -> list[float]:
        del prompts, completion_ids, kwargs
        nonlocal source_scorer_calls
        source_scorer_calls += 1
        return [float(index % 2 + source_scorer_calls - 1) for index, _ in enumerate(completions)]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=str(destination / "trainer-output"),
            per_device_train_batch_size=int(config.training["batch_size"]),
            num_generations=int(config.training["generations"]),
            max_completion_length=int(config.training["completion_length"]),
            max_steps=int(config.limits["steps"]),
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            optim="adamw_torch",
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            learning_rate=float(config.training["learning_rate"]),
        ),
        train_dataset=Dataset.from_dict({"prompt": list(config.prompts)}),
        processing_class=tokenizer,
    )
    trainer.create_optimizer_and_scheduler(num_training_steps=int(config.limits["steps"]))
    adapter = TRL110StateAdapter(trainer)
    callback = build_clamp_callback(
        config.profile,
        ClampSpecification(
            target_name=str(config.target["name"]),
            target=lambda current: current.get_input_embeddings().weight[int(row), int(column)],
            target_value=float(config.target["value"]),
            tolerance=float(config.target["tolerance"]),
            max_correction=float(config.limits["max_correction"]),
            max_iterations=int(config.limits["max_iterations"]),
        ),
    )
    callback.attach(trainer.optimizer, trainer.model)
    batch = MaterializedBatch(tuple({"prompt": prompt} for prompt in config.prompts))
    executor = PinnedShortForkExecutor(adapter, batch, config.profile, callback)
    useful_inputs = tokenizer(str(config.useful_task["prompt"]), return_tensors="pt")["input_ids"]
    useful_token = vocab[str(config.useful_task["token"])]

    def useful_task(current: Any) -> float:
        with torch.no_grad():
            return float(current(useful_inputs).logits[0, -1, useful_token].cpu())

    try:
        result = run_short_fork(
            (
                executor.arm(
                    "null",
                    Intervention("null", config.interventions["null"]),
                    {str(config.useful_task["name"]): useful_task},
                ),
                executor.arm(
                    "null",
                    Intervention("null", config.interventions["null"]),
                    {str(config.useful_task["name"]): useful_task},
                ),
                executor.arm(
                    "candidate",
                    Intervention("candidate", config.interventions["candidate"]),
                    {str(config.useful_task["name"]): useful_task},
                ),
                executor.arm(
                    "placebo",
                    Intervention("placebo", config.interventions["placebo"]),
                    {str(config.useful_task["name"]): useful_task},
                ),
            ),
            profile=config.profile,
            clamp=callback,
            fixture_root=fixture_root,
            limits=config.limits,
        )
    finally:
        callback.detach()
    if (
        adapter.generation_count != 1
        or adapter.source_scorer_count != 1
        or source_scorer_calls != 1
    ):
        raise RuntimeError(
            "CPU_SHORT_FORK_EXECUTION_REFUSAL: source replay was not frozen exactly once"
        )
    if len(callback.readings) != 4 or len(executor.witnesses) != 4:
        raise RuntimeError(
            "CPU_SHORT_FORK_EXECUTION_REFUSAL: every configured arm must execute one clamped update"
        )
    record_path = write_cpu_conformance_record(
        result, fixture_root=fixture_root, path=destination / "cpu-conformance.json"
    )
    load_cpu_conformance_record(record_path, fixture_root=fixture_root)
    return Rung5Input(
        result=result,
        record_path=record_path,
        fixture_root=fixture_root,
        subject_ref=config.subject_ref,
        provenance=contracts.EntryProvenance(
            started=started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_s=time.monotonic() - started_clock,
            sandbox_tier=config.sandbox_tier,
            offline=config.offline,
        ),
    )


_REQUIRED = (
    "fixture_digest",
    "profile",
    "clamp",
    "target_measure",
    "useful_task_controls",
    "null_spread",
    "limits",
    "scope",
    "claim_boundary",
    "arm_names",
)


def _record(result: ShortForkResult) -> dict[str, Any]:
    return {
        "fixture_digest": result.fixture_digest,
        "profile": {
            "version": result.profile.version,
            "mutable_objects": list(result.profile.mutable_objects),
            "fingerprint": result.profile.fingerprint,
        },
        "clamp": {"method": result.clamp_method},
        "target_measure": {
            "name": result.target_measure_name,
            "values": list(result.target_values),
        },
        "useful_task_controls": {
            name: list(values) for name, values in result.useful_task_controls.items()
        },
        "null_spread": {"value": result.null_spread_value, "method": result.null_spread_method},
        "limits": dict(result.limits),
        "scope": result.scope,
        "claim_boundary": result.statement,
        "arm_names": list(result.arm_names),
    }


def _fixture_contract(
    root: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    source = Path(root) / "fork.json"
    try:
        fixture = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CPUConformanceRecordInvalid("fixture/fork.json is missing") from error
    except json.JSONDecodeError as error:
        raise CPUConformanceRecordInvalid("fixture/fork.json is invalid JSON") from error
    if not isinstance(fixture, dict) or not all(
        isinstance(fixture.get(name), dict)
        for name in ("profile", "limits", "target", "useful_task")
    ):
        raise CPUConformanceRecordInvalid(
            "fixture profile, limits, target, and useful_task are required"
        )
    if not isinstance(fixture["target"].get("name"), str) or not fixture["target"]["name"]:
        raise CPUConformanceRecordInvalid("fixture target.name is required")
    if (
        not isinstance(fixture["useful_task"].get("name"), str)
        or not fixture["useful_task"]["name"]
    ):
        raise CPUConformanceRecordInvalid("fixture useful_task.name is required")
    subject_ref = fixture.get("subject_ref")
    if subject_ref is not None and (
        not isinstance(subject_ref, str) or not _SHA256_REF.fullmatch(subject_ref)
    ):
        raise CPUConformanceRecordInvalid(
            "fixture subject_ref must be a sha256 reference when present"
        )
    return (
        fixture["profile"],
        fixture["limits"],
        fixture["target"],
        fixture["useful_task"],
        subject_ref,
    )


def _finite_arm_values(value: Any, *, field: str, arm_count: int) -> None:
    if not isinstance(value, list) or len(value) != arm_count:
        raise CPUConformanceRecordInvalid(f"{field} must contain one value for every arm")
    if any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in value):
        raise CPUConformanceRecordInvalid(f"{field} must contain only finite numeric values")


def _validate_measurement_identity(
    record: Mapping[str, Any],
    *,
    expected_target: Mapping[str, Any],
    expected_useful_task: Mapping[str, Any],
) -> None:
    arms = _required(record, "arm_names")
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(name, str) or not name for name in arms)
    ):
        raise CPUConformanceRecordInvalid("arm_names must be a non-empty list of arm names")
    target = _required(record, "target_measure")
    if not isinstance(target, dict) or target.get("name") != expected_target["name"]:
        raise CPUConformanceRecordInvalid(
            "target_measure.name does not match the named fixture target"
        )
    _finite_arm_values(target.get("values"), field="target_measure.values", arm_count=len(arms))
    controls = _required(record, "useful_task_controls")
    if not isinstance(controls, dict) or set(controls) != {expected_useful_task["name"]}:
        raise CPUConformanceRecordInvalid(
            "useful_task_controls keys do not match the named fixture control"
        )
    _finite_arm_values(
        controls[expected_useful_task["name"]],
        field=f"useful_task_controls.{expected_useful_task['name']}",
        arm_count=len(arms),
    )


def write_cpu_conformance_record(
    result: ShortForkResult, *, fixture_root: Path | str, path: Path | str
) -> Path:
    """Write only a test-local generated record after verifying its fixture identity."""
    if not isinstance(result, ShortForkResult):
        raise TypeError("rung 5 requires a ShortForkResult")
    expected = fixture_digest(fixture_root)
    if result.fixture_digest != expected:
        raise CPUConformanceRecordInvalid("fixture_digest does not match the named fixture")
    expected_profile, expected_limits, expected_target, expected_useful_task, _subject_ref = (
        _fixture_contract(fixture_root)
    )
    actual_profile = {
        "version": result.profile.version,
        "mutable_objects": list(result.profile.mutable_objects),
        "fingerprint": result.profile.fingerprint,
    }
    if actual_profile != expected_profile:
        raise CPUConformanceRecordInvalid("profile does not match the named fixture")
    if dict(result.limits) != expected_limits:
        raise CPUConformanceRecordInvalid("limits do not match the named fixture")
    payload = _record(result)
    _validate_measurement_identity(
        payload, expected_target=expected_target, expected_useful_task=expected_useful_task
    )
    destination = Path(path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _required(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise CPUConformanceRecordInvalid(f"missing field {name!r}")
    return record[name]


def load_cpu_conformance_record(path: Path | str, *, fixture_root: Path | str) -> dict[str, Any]:
    """Re-read a generated record and fail closed on any required integrity mismatch."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CPUConformanceRecordInvalid("missing generated record") from error
    except json.JSONDecodeError as error:
        raise CPUConformanceRecordInvalid("invalid JSON record") from error
    if not isinstance(payload, dict):
        raise CPUConformanceRecordInvalid("record root must be an object")
    for name in _REQUIRED:
        _required(payload, name)
    if payload["fixture_digest"] != fixture_digest(fixture_root):
        raise CPUConformanceRecordInvalid("fixture_digest does not match the named fixture")
    expected_profile, expected_limits, expected_target, expected_useful_task, _subject_ref = (
        _fixture_contract(fixture_root)
    )
    profile = _required(payload, "profile")
    if not isinstance(profile, dict) or profile.get("version") != "1.10.0":
        raise CPUConformanceRecordInvalid("profile.version must be '1.10.0'")
    if (
        profile.get("mutable_objects") != list(EIGHT_OBJECTS)
        or profile.get("mutable_objects") != expected_profile["mutable_objects"]
    ):
        raise CPUConformanceRecordInvalid(
            "profile.mutable_objects does not match the named fixture"
        )
    if profile.get("fingerprint") != expected_profile["fingerprint"]:
        raise CPUConformanceRecordInvalid("profile.fingerprint does not match the named fixture")
    clamp = _required(payload, "clamp")
    if not isinstance(clamp, dict) or clamp.get("method") != "optimizer_step_projection":
        raise CPUConformanceRecordInvalid("clamp.method must be 'optimizer_step_projection'")
    _validate_measurement_identity(
        payload, expected_target=expected_target, expected_useful_task=expected_useful_task
    )
    spread = _required(payload, "null_spread")
    if not isinstance(spread, dict) or spread.get("method") != "sample_standard_deviation":
        raise CPUConformanceRecordInvalid("null_spread.method must be 'sample_standard_deviation'")
    if not isinstance(spread.get("value"), (int, float)) or spread["value"] < 0:
        raise CPUConformanceRecordInvalid("null_spread.value must be a non-negative computed value")
    if not isinstance(payload["limits"], dict) or payload["limits"] != expected_limits:
        raise CPUConformanceRecordInvalid("limits do not match the named fixture")
    if payload["scope"] != "short_fork_behavior":
        raise CPUConformanceRecordInvalid("scope must be 'short_fork_behavior'")
    boundary = payload["claim_boundary"]
    if not isinstance(boundary, str) or "not a behavioural result" not in boundary.lower():
        raise CPUConformanceRecordInvalid("claim_boundary must state 'not a behavioural result'")
    return payload


def _control_evidence(
    record: Mapping[str, Any], *, target: Mapping[str, Any], useful_task: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Derive the CPU-control predicate from retained arm witnesses, never a constant."""
    arm_names = list(record["arm_names"])
    targets = list(record["target_measure"]["values"])
    controls = list(record["useful_task_controls"][useful_task["name"]])
    target_value = float(target["value"])
    tolerance = float(target["tolerance"])

    def arm_evidence(name: str) -> dict[str, Any]:
        indexes = [index for index, arm in enumerate(arm_names) if arm == name]
        readings = [targets[index] for index in indexes]
        useful = [controls[index] for index in indexes]
        target_held = bool(indexes) and all(
            abs(value - target_value) <= tolerance for value in readings
        )
        useful_task_held = bool(useful) and all(math.isfinite(float(value)) for value in useful)
        return {
            "passed": target_held and useful_task_held,
            "arm_indexes": indexes,
            "post_step_target_values": readings,
            "post_step_useful_task_values": useful,
            "target_held": target_held,
            "useful_task_held": useful_task_held,
        }

    null = arm_evidence("null")
    null["passed"] = bool(null["passed"]) and len(null["arm_indexes"]) >= 2
    null["null_spread"] = record["null_spread"]
    candidate = arm_evidence("candidate")
    placebo = arm_evidence("placebo")
    passed = bool(null["passed"] and candidate["passed"] and placebo["passed"])
    return passed, {"null_replay": null, "candidate": candidate, "placebo": placebo}


def compare_verified_short_fork(input: Rung5Input) -> CompareResult:
    """Turn the actual short-fork result and its verified record into one canonical R5 entry."""
    if not isinstance(input, Rung5Input):
        raise TypeError("rung 5 canonical producer requires Rung5Input")
    record = load_cpu_conformance_record(input.record_path, fixture_root=input.fixture_root)
    if record != _record(input.result):
        raise CPUConformanceRecordInvalid(
            "verified record does not match the supplied actual result"
        )
    _profile, _limits, target, useful_task, bound_subject_ref = _fixture_contract(
        input.fixture_root
    )
    if bound_subject_ref is None:
        raise CPUConformanceRecordInvalid(
            "fixture subject_ref is absent: legacy unbound fixtures cannot produce canonical external identity"
        )
    if input.subject_ref != bound_subject_ref:
        raise CPUConformanceRecordInvalid(
            "subject_ref does not match the fixture-bound execution identity"
        )
    passed, controls = _control_evidence(record, target=target, useful_task=useful_task)
    entry = contracts.Entry(
        entry_id=R5_ENTRY_ID,
        section="reward_statistics",
        kind="check",
        measurand="whether a bounded pinned-TRL short fork retains target and useful-task control witnesses",
        method=contracts.Method(
            id=R5_ENTRY_ID,
            version="3.1.0",
            params_digest=contracts.digest(
                {
                    "fixture_digest": record["fixture_digest"],
                    "profile": record["profile"],
                    "limits": record["limits"],
                }
            ),
            procedure="verify the named CPU fixture record, then derive null, candidate, and placebo control predicates from its retained post-step measurements",
        ),
        scope="short_fork_behavior",
        subject_ref=input.subject_ref,
        depends_on=["digest:training_semantics", "digest:environment"],
        state="complete",
        provenance=input.provenance,
        limitations=[
            "CPU conformance only; this is not a behavioural result, transfer result, or qualified outcome effect."
        ],
        result={"cpu_conformance": record, **controls},
        check=contracts.Check(
            predicate="two null arms plus candidate and placebo retain fixture-bound post-step target and useful-task witnesses",
            passed=passed,
            scope_tested="one pinned CPU fixture and one optimizer step per retained arm",
        ),
    )
    rows = [
        ScorecardRow(
            row_id,
            row_id.replace("_", " "),
            Absence(
                "not measured: CPU short-fork conformance has no qualified outcome interval",
                "run the registered task and held-out outcome measurements before making a behavioural claim",
            ),
            "short_fork_behavior",
        )
        for row_id in ROWS
    ]
    return CompareResult(
        "short_fork_behavior",
        Scorecard.render(rows, rung=Rung.SHORT_FORK),
        [entry],
        "a verified CPU short-fork conformance check with retained control witnesses",
        "behaviour after training, transfer, or a qualified outcome effect",
        record,
    )


def append_short_fork_assay(assay: contracts.Assay, comparison: CompareResult) -> contracts.Assay:
    """Append the one canonical R5 entry to an assay before the shared store seals it."""
    if not isinstance(assay, contracts.Assay) or not isinstance(comparison, CompareResult):
        raise TypeError("rung 5 assay producer requires an Assay and CompareResult")
    entries = [entry for entry in comparison.entries if entry.entry_id == R5_ENTRY_ID]
    if len(entries) != 1:
        raise ValueError("rung 5 comparison must carry exactly one canonical entry")
    data = assay.to_dict()
    existing = data["measurement"]["reward_statistics"]
    if any(entry["entry_id"] == R5_ENTRY_ID for entry in existing):
        raise ValueError("rung 5 canonical entry is already present in this assay")
    existing.append(entries[0].model_dump(mode="json", by_alias=True, exclude_unset=True))
    data["assay_id"] = contracts.digest(data)
    return contracts.Assay.model_validate(data)


def compare_short_fork(result: ShortForkResult) -> CompareResult:
    """Expose a real rung-five record while preserving its conformance-only limit."""
    if not isinstance(result, ShortForkResult):
        raise TypeError("rung 5 requires a ShortForkResult")
    rows = [
        ScorecardRow(
            row_id,
            row_id.replace("_", " "),
            Absence(
                "not measured: CPU short-fork conformance has no qualified outcome interval",
                "run the registered task and held-out outcome measurements before making a behavioural claim",
            ),
            result.scope,
        )
        for row_id in ROWS
    ]
    return CompareResult(
        result.scope,
        Scorecard.render(rows, rung=Rung.SHORT_FORK),
        [],
        "an integrity-checked CPU short-fork clamp conformance record",
        "behaviour after training, transfer, or a qualified outcome effect",
        result,
    )


# The scientific trial route is separate from the accepted CPU conformance wire shape.
CODING_SHORT_FORK_PROFILE = "m3-qwen2-text-penalty-short-fork-v1"


@dataclass(frozen=True)
class CodingShortForkConfig:
    """One prepared checkpoint/reward pair and one fixed optimizer-step horizon."""

    prepared_manifest: Path
    prepared_manifest_digest: str
    subject_ref: str
    profile: str = CODING_SHORT_FORK_PROFILE

    def __post_init__(self) -> None:
        if (
            self.profile != CODING_SHORT_FORK_PROFILE
            or not isinstance(self.prepared_manifest, Path)
            or not self.prepared_manifest.is_absolute()
        ):
            raise ValueError(
                "coding short fork requires its exact profile and absolute prepared manifest"
            )
        if any(
            not isinstance(value, str) or not _SHA256_REF.fullmatch(value)
            for value in (self.subject_ref, self.prepared_manifest_digest)
        ):
            raise ValueError("coding short fork requires manifest and subject digests")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CodingShortForkConfig":
        if set(raw) != {"profile", "prepared_manifest", "prepared_manifest_digest", "subject_ref"}:
            raise ValueError("coding short fork accepts only its exact prepared declaration")
        return cls(
            Path(raw["prepared_manifest"]),
            raw["prepared_manifest_digest"],
            raw["subject_ref"],
            raw["profile"],
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "profile": self.profile,
            "prepared_manifest": str(self.prepared_manifest),
            "prepared_manifest_digest": self.prepared_manifest_digest,
            "subject_ref": self.subject_ref,
        }

    @property
    def digest(self) -> str:
        return contracts.digest(self.to_mapping())


@dataclass(frozen=True)
class CodingShortForkInput:
    """Retained actual update/outcome receipts, bound to the producer module."""

    payload: Mapping[str, Any]
    artifact_path: Path
    artifact_digest: str
    subject_ref: str
    provenance: contracts.EntryProvenance


def _coding_project_subject(project, config_path: Path) -> str:
    from reward_lens.api._record import absent_record
    from reward_lens.store.digests import method_set

    provisional = absent_record(command="compare", reproduce=[], path=config_path)
    methods = set(method_set(provisional.to_dict()))
    absence = provisional.measurement.reward_statistics[0].method
    methods.discard(f"{absence.id}@{absence.version}")
    methods.add(f"{R5_ENTRY_ID}@3.1.0")
    return project.version(methods=tuple(sorted(methods))).digest()


def validate_coding_short_fork_preparation(
    config: CodingShortForkConfig,
) -> tuple[dict[str, Any], Any]:
    """Reuse exact R4 byte/checkpoint qualification while binding the R5 method subject."""
    from reward_lens.store import Project

    from . import rung4

    if rung4._file_digest(config.prepared_manifest) != config.prepared_manifest_digest:
        raise ValueError("coding short-fork preparation digest mismatch")
    manifest = json.loads(config.prepared_manifest.read_text())
    r4_config = rung4.CodingControlledUpdateConfig(
        config.prepared_manifest, config.prepared_manifest_digest, manifest["candidate_subject_ref"]
    )
    manifest = rung4.validate_coding_preparation(r4_config)
    project = Project.open(config.prepared_manifest.parent / "inputs/text-penalty")
    if _coding_project_subject(project, project.root / "short-fork.json") != config.subject_ref:
        raise ValueError(
            "coding short-fork prepared project differs from the authoritative R5 subject"
        )
    return manifest, project


def _coding_floor_context(project, *, subject_ref: str, sandbox, limits):
    """Read the project's existing declaration; never select a default floor partition."""
    from reward_lens.outcome.project_execution import (
        ProjectOutcomeExecution,
        load_project_outcome_execution,
    )

    declaration = (
        None if project.config.outcome is None else project.config.outcome.held_out_capability
    )
    if declaration is None:
        return None
    if declaration.floor is None:
        raise ValueError("declared coding capability set has no threshold")
    execution = load_project_outcome_execution(
        project, expected_subject_ref=subject_ref, sandbox=sandbox, limits=limits
    )
    if (
        not isinstance(execution, ProjectOutcomeExecution)
        or not execution.qualified
        or execution.capability_floor is None
    ):
        raise ValueError("declared coding capability floor lacks qualified supervised execution")
    if not 1 <= len(execution.capability_floor.cases) <= 40:
        raise ValueError("coding capability floor supports one to forty explicitly declared cases")
    return execution


def _coding_floor_payload(execution, *, generate, device: str) -> dict[str, Any]:
    """Measure fresh post-update responses through the existing supervised resolver."""
    if execution is None:
        return {
            "state": "NOT_MEASURED",
            "reason": "no held-out capability set declared",
            "held": None,
            "repair_claim_qualified": False,
        }
    rows = []
    declared = execution.capability_floor
    for case in declared.cases:
        response = generate(dict(case.task))
        resolved = execution.resolve_for_actual(
            task=case.task, response=response, item_id=case.item_id
        )
        from reward_lens.outcome import OutcomeState

        if resolved.outcome.state not in {OutcomeState.PASS, OutcomeState.FAIL}:
            raise ValueError("coding capability response could not be independently resolved")
        rows.append(
            {
                "item_id": case.item_id,
                "task": dict(case.task),
                "response": response,
                "passed": resolved.outcome.state is OutcomeState.PASS,
                "outcome": {
                    "item_id": resolved.outcome.item_id,
                    "state": resolved.outcome.state.value,
                    "reason": resolved.outcome.reason,
                    "protocol_digest": resolved.outcome.protocol_digest,
                    "evaluated_on_partition_id": resolved.outcome.evaluated_on_partition_id,
                    "evaluated_item_digest": resolved.outcome.evaluated_item_digest,
                },
                "supervisor_receipt": resolved.receipt.to_record(),
                "response_digest": resolved.response_digest,
                "partition_digest": resolved.target_partition_digest,
            }
        )
    if not rows:
        raise ValueError("declared coding capability floor contains no cases")
    value = sum(row["passed"] for row in rows) / len(rows)
    return {
        "state": "MEASURED",
        "partition_id": declared.partition.id,
        "partition_digest": declared.partition.digest,
        "floor": float(declared.floor),
        "value": value,
        "n": len(rows),
        "held": value >= float(declared.floor),
        "rows": rows,
        "repair_claim_qualified": False,
    }


def _run_coding_short_fork(
    config: CodingShortForkConfig, manifest: Mapping[str, Any], project, artifact_dir: Path
) -> dict[str, Any]:
    """One real short fork per arm; no clamp coordinate or generic trainer is invented."""
    from reward_lens.learning.intervention.swap import capture_state, restore_state, same_start

    from . import rung4

    adapter_type = getattr(rung4, "_CodingTRL110StateAdapter", None)
    if adapter_type is None:
        raise contracts.CapabilityUnavailable(
            code="RL0702",
            message="coding short fork requires the reviewed CUDA RNG-complete R4 adapter",
            remediation="integrate and rebind the finite R4 CUDA-state repair first",
        )
    try:
        torch, sandbox = rung4._coding_backend(manifest)
    except contracts.CapabilityUnavailable as error:
        raise contracts.CapabilityUnavailable(
            code=error.code,
            message="coding short-fork backend unavailable: " + error.message,
            remediation=error.remediation,
            context={**error.context, "profile": CODING_SHORT_FORK_PROFILE},
        ) from error
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from reward_lens.execution import Limits
    from reward_lens.graders.python_grader import PythonGrader

    limits = Limits(
        wall_s=30,
        cpu_s=30,
        memory_bytes=2 * 1024**3,
        processes=64,
        stdout_bytes=65536,
        artifact_bytes=65536,
    )
    execution = _coding_floor_context(
        project, subject_ref=config.subject_ref, sandbox=sandbox, limits=limits
    )
    roots = {
        name: config.prepared_manifest.parent / "inputs" / name
        for name in ("baseline", "text-penalty")
    }
    graders = {name: PythonGrader.from_path(root / "grader.py") for name, root in roots.items()}
    independent = PythonGrader.from_path(roots["baseline"] / "independent-outcome.py")
    tasks = [
        json.loads(line)
        for line in (roots["baseline"] / "tasks.jsonl").read_text().splitlines()
        if line
    ]
    chosen = [
        next(task for task in tasks if task["task_id"] == name) for name in ("task-01", "task-02")
    ]
    task_by_prompt = {
        task["prompt"] + "\nReturn one fenced Python implementation.": dict(
            task, id=task["task_id"]
        )
        for task in chosen
    }
    training_receipts = []
    offsets = []

    def reward(prompts, completions, **kwargs):
        del kwargs
        scores = []
        offsets.clear()
        for prompt, completion in zip(prompts, completions, strict=True):
            task = task_by_prompt[prompt]
            scored = {
                name: grader.score(
                    task,
                    completion,
                    sandbox=sandbox,
                    limits=limits,
                    project_dir=roots[name],
                    seed=37,
                )
                for name, grader in graders.items()
            }
            if any(env.verdict != "scored" or env.score is None for env in scored.values()):
                raise ValueError("coding short-fork source scoring refused")
            base, candidate = float(scored["baseline"].score), float(scored["text-penalty"].score)
            scores.append(base)
            offsets.append(candidate - base)
            training_receipts.append(
                {
                    "task_id": task["task_id"],
                    "completion": completion,
                    "baseline": base,
                    "candidate": candidate,
                    "envelopes": {name: env.to_dict() for name, env in scored.items()},
                }
            )
        return scores

    checkpoint = Path(manifest["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cuda")
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("coding short-fork model did not load on CUDA")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=str(artifact_dir / "trainer"),
            per_device_train_batch_size=4,
            num_generations=2,
            max_completion_length=128,
            max_steps=1,
            learning_rate=1e-5,
            logging_strategy="no",
            report_to=[],
            optim="adamw_torch",
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            bf16=False,
            fp16=False,
            seed=37,
        ),
        train_dataset=Dataset.from_dict({"prompt": list(task_by_prompt)}),
        processing_class=tokenizer,
    )
    trainer.create_optimizer_and_scheduler(1)
    adapter = adapter_type(trainer)
    profile = TrainerProfile("1.10.0", EIGHT_OBJECTS, CODING_SHORT_FORK_PROFILE)
    baseline = capture_state(adapter, profile)
    batch = MaterializedBatch(
        tuple({"prompt": prompt} for prompt in task_by_prompt for _ in range(2))
    )
    replay = adapter.freeze_scored_replay(batch)
    if len(offsets) != 4 or not any(offsets):
        raise ValueError("coding revision has no measured effect on the frozen generated batch")
    restore_state(adapter, baseline, profile)
    same_start(capture_state(adapter, profile), baseline).require()

    def generate(task):
        prompt = task["prompt"] + "\nReturn one fenced Python implementation."
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        was_training = trainer.model.training
        trainer.model.eval()
        try:
            with torch.no_grad():
                output = trainer.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        finally:
            trainer.model.train(was_training)

    arms = []
    for name, factor in (("null", 0), ("null", 0), ("candidate", 1), ("placebo", -1)):
        restore_state(adapter, baseline, profile)
        same_start(capture_state(adapter, profile), baseline).require()
        try:
            value = adapter.run_one_update(
                replay=replay,
                intervention=Intervention(
                    name, tuple(float(factor * offset) for offset in offsets)
                ),
            )
            during = capture_state(adapter, profile).digest()
            outcomes = []
            for task in chosen:
                actual_task = dict(task, id=task["task_id"])
                response = generate(actual_task)
                envelope = independent.score(
                    actual_task,
                    response,
                    sandbox=sandbox,
                    limits=limits,
                    project_dir=roots["baseline"],
                    seed=37,
                )
                if envelope.verdict != "scored" or envelope.score not in (0.0, 1.0):
                    raise ValueError("post-update protected coding outcome refused")
                outcomes.append(
                    {
                        "task_id": task["task_id"],
                        "response": response,
                        "envelope": envelope.to_dict(),
                    }
                )
            floor = _coding_floor_payload(execution, generate=generate, device="cuda")
        finally:
            restore_state(adapter, baseline, profile)
            same_start(capture_state(adapter, profile), baseline).require()
        update = {
            key: value[key]
            for key in (
                "batch_digest",
                "scored_replay_digest",
                "loss_input_digest",
                "loss",
                "gradient_norm",
                "optimizer_update_norm",
            )
        }
        update["batch_bytes_hex"] = value["batch_bytes"].hex()
        update["reward_offsets"] = list(value["intervention"].reward_offsets)
        update["reward_witness"] = [
            {
                "source_values": list(call["source_values"]),
                "returned_values": list(call["returned_values"]),
            }
            for call in value["reward_witness"]
        ]
        arms.append(
            {
                "name": name,
                "start_digest": baseline.digest(),
                "during_digest": during,
                "restored_digest": capture_state(adapter, profile).digest(),
                "update": update,
                "outcomes": outcomes,
                "capability_floor": floor,
            }
        )
    null_values = [
        sum(row["envelope"]["score"] for row in arm["outcomes"]) / len(arm["outcomes"])
        for arm in arms[:2]
    ]
    from statistics import stdev

    return {
        "device": "cuda",
        "execution_class": "actual_cuda_short_fork",
        "horizon_steps": 1,
        "arms": arms,
        "training_receipts": training_receipts,
        "null_spread": float(stdev(null_values)),
        "null_spread_method": "sample_standard_deviation",
    }


def _validate_coding_short_fork_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload["profile"] != CODING_SHORT_FORK_PROFILE
        or payload["horizon_steps"] != 1
        or payload["device"] not in {"cuda", "controlled-test-loader"}
    ):
        raise ValueError("coding short-fork profile, horizon or device is unsupported")
    arms = payload["arms"]
    if [arm["name"] for arm in arms] != ["null", "null", "candidate", "placebo"] or len(
        payload["training_receipts"]
    ) != 4:
        raise ValueError(
            "coding short fork requires two null arms and actual candidate/placebo evidence"
        )
    wanted_class = (
        "actual_cuda_short_fork" if payload["device"] == "cuda" else "controlled_test_loader"
    )
    if payload["execution_class"] != wanted_class or not _SHA256_REF.fullmatch(
        payload["subject_ref"]
    ):
        raise ValueError("coding short-fork execution classification or subject is invalid")
    starts = {arm["start_digest"] for arm in arms}
    if len(starts) != 1 or any(
        arm["restored_digest"] != arm["start_digest"] or arm["during_digest"] == arm["start_digest"]
        for arm in arms
    ):
        raise ValueError("coding short-fork state restoration failed")
    for key in ("batch_digest", "batch_bytes_hex", "scored_replay_digest", "loss_input_digest"):
        if len({arm["update"][key] for arm in arms}) != 1 or not arms[0]["update"][key]:
            raise ValueError("coding short-fork arms do not share the frozen replay")
    offsets = [float(row["candidate"] - row["baseline"]) for row in payload["training_receipts"]]
    if not any(offsets) or any(not math.isfinite(value) for value in offsets):
        raise ValueError("coding short-fork revision has no finite measured effect")
    if [row["task_id"] for row in payload["training_receipts"]] != [
        "task-01",
        "task-01",
        "task-02",
        "task-02",
    ]:
        raise ValueError("coding short-fork training receipt grouping changed")
    if payload["device"] == "cuda" and any(
        not _SHA256_REF.fullmatch(arm[key])
        for arm in arms
        for key in ("start_digest", "during_digest", "restored_digest")
    ):
        raise ValueError("coding short-fork CUDA state identities are not canonical")
    for row in payload["training_receipts"]:
        if not row["completion"]:
            raise ValueError("coding short-fork training generation is absent")
        for name, score in (("baseline", row["baseline"]), ("text-penalty", row["candidate"])):
            envelope = row["envelopes"][name]
            if envelope["verdict"] != "scored" or envelope["score"] != score:
                raise ValueError("coding short-fork training score receipt differs from the update")
            if payload["device"] == "cuda":
                request = envelope["observations"]["request"]
                if (
                    request["task"]["task_id"] != row["task_id"]
                    or request["response"] != row["completion"]
                    or envelope["request_digest"] != contracts.digest(request)
                    or envelope["provenance"]["sandbox_tier"] != "L3"
                ):
                    raise ValueError(
                        "coding short-fork training receipt is not actual protected execution"
                    )
    for arm, factor in zip(arms, (0, 0, 1, -1), strict=True):
        update = arm["update"]
        if not math.isfinite(float(update["loss"])):
            raise ValueError("coding short-fork loss is not finite")
        if update["reward_offsets"] != [factor * value for value in offsets] or any(
            not math.isfinite(float(update[key])) or update[key] <= 0
            for key in ("gradient_norm", "optimizer_update_norm")
        ):
            raise ValueError("coding short-fork null/candidate/placebo update evidence failed")
        source = [
            float(value) for call in update["reward_witness"] for value in call["source_values"]
        ]
        returned = [
            float(value) for call in update["reward_witness"] for value in call["returned_values"]
        ]
        if source != [row["baseline"] for row in payload["training_receipts"]] or returned != [
            base + factor * offset for base, offset in zip(source, offsets, strict=True)
        ]:
            raise ValueError("coding short-fork applied reward does not bind actual scores")
        if [row["task_id"] for row in arm["outcomes"]] != ["task-01", "task-02"] or any(
            not row["response"]
            or row["envelope"]["verdict"] != "scored"
            or row["envelope"]["score"] not in (0.0, 1.0)
            for row in arm["outcomes"]
        ):
            raise ValueError("coding short-fork independent outcomes are unresolved")
        if payload["device"] == "cuda":
            for row in arm["outcomes"]:
                envelope = row["envelope"]
                request = envelope["observations"]["request"]
                if (
                    request["task"]["task_id"] != row["task_id"]
                    or request["response"] != row["response"]
                    or envelope["request_digest"] != contracts.digest(request)
                    or envelope["provenance"]["sandbox_tier"] != "L3"
                ):
                    raise ValueError(
                        "coding short-fork post-update receipt is not actual protected execution"
                    )
        floor = arm["capability_floor"]
        if floor["state"] == "NOT_MEASURED":
            if floor.get("held") is not None or floor.get("repair_claim_qualified") is not False:
                raise ValueError("coding short-fork missing capability floor was relabeled")
        elif floor["state"] == "MEASURED":
            rows = floor["rows"]
            if not rows:
                raise ValueError("coding short-fork declared capability floor has no observations")
            value = sum(row["passed"] for row in rows) / len(rows)
            if (
                floor.get("repair_claim_qualified") is not False
                or not rows
                or any(type(row["passed"]) is not bool for row in rows)
                or not 0 <= floor["floor"] <= 1
                or floor["n"] != len(rows)
                or floor["value"] != value
                or floor["held"] is not (value >= floor["floor"])
            ):
                raise ValueError(
                    "coding short-fork capability floor differs from measured outcomes"
                )
        else:
            raise ValueError("coding short-fork capability state is invalid")
    if arms[0]["outcomes"] != arms[1]["outcomes"]:
        # Supervisor run IDs/timing differ; compare actual tasks/responses/verdicts instead.
        def null_projection(arm):
            return [
                (row["task_id"], row["response"], row["envelope"]["score"])
                for row in arm["outcomes"]
            ]

        if null_projection(arms[0]) != null_projection(arms[1]):
            raise ValueError("coding short-fork null replay diverged")
    null_values = [sum(row["envelope"]["score"] for row in arm["outcomes"]) / 2 for arm in arms[:2]]
    from statistics import stdev

    if payload["null_spread_method"] != "sample_standard_deviation" or payload[
        "null_spread"
    ] != float(stdev(null_values)):
        raise ValueError("coding short-fork null spread was not measured")


def build_coding_short_fork(
    config: CodingShortForkConfig, artifact_dir: Path | str
) -> CodingShortForkInput:
    """Prepare/execute the finite coding route and retain its result for canonical R5."""
    from . import rung4

    destination = Path(artifact_dir)
    started = datetime.now(UTC)
    clock = time.monotonic()
    try:
        manifest, project = validate_coding_short_fork_preparation(config)
        execution = _run_coding_short_fork(config, manifest, project, destination)
        payload = {
            **execution,
            "schema": "reward-lens.coding-short-fork-execution/v1",
            "profile": config.profile,
            "config_digest": config.digest,
            "subject_ref": config.subject_ref,
            "prepared_manifest_digest": config.prepared_manifest_digest,
            "checkpoint": manifest["checkpoint"],
            "checkpoint_files": manifest["checkpoint_files"],
            "input_files": manifest["input_files"],
            "module_source_digest": rung4._file_digest(Path(__file__)),
            "claim_boundary": "bounded short-fork protocol observations; no behavioral repair claim without independent review and a measured capability floor",
        }
        _validate_coding_short_fork_payload(payload)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise contracts.UsageError(
            code="RL0131",
            message="coding short fork refused: " + str(error),
            remediation="supply the exact prepared coding profile and intact outcome, null/placebo and restoration controls",
        ) from error
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "coding-short-fork-execution.json"
    path.write_bytes(contracts.canonical_bytes(payload))
    return CodingShortForkInput(
        payload,
        path,
        rung4._file_digest(path),
        config.subject_ref,
        contracts.EntryProvenance(
            started=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_s=time.monotonic() - clock,
            sandbox_tier="L3" if payload["device"] == "cuda" else "T0",
            offline=True,
        ),
    )


def compare_coding_short_fork(input: CodingShortForkInput) -> CompareResult:
    from . import rung4

    payload = dict(input.payload)
    if (
        input.subject_ref != payload["subject_ref"]
        or rung4._file_digest(input.artifact_path) != input.artifact_digest
        or input.artifact_path.read_bytes() != contracts.canonical_bytes(payload)
        or payload["module_source_digest"] != rung4._file_digest(Path(__file__))
    ):
        raise ValueError("coding short-fork artifact or source binding changed")
    _validate_coding_short_fork_payload(payload)
    floor = payload["arms"][2]["capability_floor"]
    entry = contracts.Entry(
        entry_id=R5_ENTRY_ID,
        section="reward_statistics",
        kind="check",
        measurand="bounded coding short-fork updates and independently scored post-update responses",
        method=contracts.Method(
            id=R5_ENTRY_ID,
            version="3.1.0",
            params_digest=contracts.digest(payload),
            procedure="execute two nulls, candidate and placebo from one restored checkpoint, score fresh post-update responses and preserve the project's capability declaration or absence",
        ),
        scope="short_fork_behavior",
        subject_ref=input.subject_ref,
        depends_on=["digest:training_semantics", "digest:environment"],
        state="complete",
        provenance=input.provenance,
        limitations=[
            payload["claim_boundary"],
            "no held-out transfer or full M3 acceptance follows from this one profile",
        ],
        result={
            "coding_short_fork": payload,
            "capability_floor": floor,
            "claim_boundary": payload["claim_boundary"],
        },
        check=contracts.Check(
            predicate="four arms share restored state and frozen training replay; actual post-update outcome receipts and null spread are retained",
            passed=True,
            scope_tested="one declared checkpoint, two coding prompts and one optimizer step per arm",
        ),
    )
    rows = [
        ScorecardRow(
            name,
            name.replace("_", " "),
            Absence(
                "not measured: qualified behavioral repair gate remains closed"
                if name != "legitimate_solutions_kept"
                else (
                    "not measured: no held-out capability set declared"
                    if floor["state"] == "NOT_MEASURED"
                    else "raw declared capability observations retained; qualified scorecard promotion awaits independent trial review"
                ),
                "inspect the retained coding trial and complete independent outcome/capability qualification",
            ),
            "short_fork_behavior",
        )
        for name in ROWS
    ]
    return CompareResult(
        "short_fork_behavior",
        Scorecard.render(rows, rung=Rung.SHORT_FORK),
        [entry],
        "retained coding short-fork protocol observations and capability-floor disposition",
        "a qualified behavioral repair, transfer or full M3 result",
        payload,
    )


def run(inputs: CompareInputs) -> CompareResult:
    """Registry entrypoint: callers supply the retained result/record typed seam."""
    if isinstance(inputs.run_record, CodingShortForkInput):
        return compare_coding_short_fork(inputs.run_record)
    if not isinstance(inputs.run_record, Rung5Input):
        raise TypeError("rung 5 requires a Rung5Input with result and verified record")
    return compare_verified_short_fork(inputs.run_record)


register(Rung.SHORT_FORK, run)
