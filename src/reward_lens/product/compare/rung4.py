"""Persistence-ready evidence for one actual rung-four controlled update."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from reward_lens import contracts
from reward_lens.learning.contracts import EIGHT_OBJECTS, ArmResult, TrainerProfile
from reward_lens.learning.intervention.swap import (
    ControlledUpdateResult,
    Intervention,
    MaterializedBatch,
    SwapArm,
    TRL110StateAdapter,
    run_swap,
)
from reward_lens.product.trace.record import Series, TraceRecord, TraceRow

from .registry import CompareResult
from .scorecard import ROWS, Absence, Scorecard, ScorecardRow

__all__ = [
    "CpuControlledUpdateConfig",
    "ControlledUpdateEvidence",
    "ControlledUpdateRecord",
    "build_cpu_controlled_update",
    "controlled_update_compare",
    "controlled_update_record",
    "CODING_UPDATE_PROFILE",
    "CodingControlledUpdateConfig",
    "validate_coding_preparation",
    "build_coding_controlled_update",
]
_SCOPE = "controlled_update_response"
_ENTRY_ID = "reward_statistics.controlled_update_response"
_LOCAL_CONFORMANCE_SUBJECT = contracts.digest(
    {"subject": "reward-lens local CPU controlled-update conformance fixture"}
)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CpuControlledUpdateConfig:
    """Bounded local-only execution inputs for the public rung-four factory."""

    seed: int = 31
    batch_size: int = 2
    num_generations: int = 2
    max_completion_length: int = 3
    candidate_offsets: tuple[float, float] = (0.25, -0.25)
    subject_ref: str = _LOCAL_CONFORMANCE_SUBJECT

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("CPU controlled update seed must be a bounded integer")
        if (
            self.batch_size != 2
            or self.num_generations != 2
            or not 1 <= self.max_completion_length <= 8
        ):
            raise ValueError(
                "CPU controlled update supports only the bounded two-by-two local fixture"
            )
        if len(self.candidate_offsets) != 2 or any(
            type(x) is not float for x in self.candidate_offsets
        ):
            raise TypeError("CPU controlled update candidate_offsets must be two floats")
        if not all(math.isfinite(offset) for offset in self.candidate_offsets):
            raise ValueError("CPU controlled update candidate_offsets must be finite")
        if not isinstance(self.subject_ref, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.subject_ref
        ):
            raise ValueError(
                "CPU controlled update subject_ref must be an authoritative sha256 digest"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CpuControlledUpdateConfig":
        fields = {
            "seed",
            "batch_size",
            "num_generations",
            "max_completion_length",
            "candidate_offsets",
            "subject_ref",
        }
        if not isinstance(value, Mapping) or set(value) - fields:
            raise ValueError("CPU controlled update config has undeclared fields")
        if "subject_ref" not in value:
            raise ValueError("CPU controlled update public config requires subject_ref")
        offsets = value.get("candidate_offsets", cls.candidate_offsets)
        return cls(
            value.get("seed", 31),
            value.get("batch_size", 2),
            value.get("num_generations", 2),
            value.get("max_completion_length", 3),
            tuple(offsets),
            value["subject_ref"],
        )

    def canonical_mapping(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
            "candidate_offsets": list(self.candidate_offsets),
            "subject_ref": self.subject_ref,
        }

    @property
    def digest(self) -> str:
        return contracts.digest(self.canonical_mapping())


CODING_UPDATE_PROFILE = "m3-qwen2-text-penalty-controlled-update-v1"
_CODING_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_CODING_WEIGHTS = "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
_CODING_CHECKPOINT_PINS = {
    ".gitattributes": "sha256:11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE": "sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "sha256:b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
    "config.json": "sha256:18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "sha256:e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "sha256:599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "sha256:c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "sha256:5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
_CODING_RUNTIME_PINS = {
    "accelerate": "1.15.0",
    "datasets": "5.0.1",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.6.0",
    "transformers": "4.57.6",
    "trl": "1.10.0",
}
_CODING_INPUT_PINS = {
    "inputs/baseline/grader.py": "sha256:09deeaa0c6e63db5cb1e2442004969c853396eb81fd151c1d650620b00cff7b3",
    "inputs/text-penalty/grader.py": "sha256:0a1a0d2e99d3d1c1e675b64ef692bd33c29d1b91957ac9da4a0973e7bff0b2b5",
    "inputs/baseline/tasks.jsonl": "sha256:c7236beb22736461004935e3aec62fb4b09ee3ab4d3fa6a6978e83bb0fc7cb5a",
    "inputs/baseline/responses.jsonl": "sha256:6c5512f5e141bf477788afc441424a862dd0b6116ce7a77d6d4162ac6831ddcd",
    "inputs/baseline/outcome/check.py": "sha256:cf79bc6e6cca1755ad6fa91185db164309a57ff2ee0f5f3b838da778f33960aa",
    "inputs/baseline/outcome/test_solution.py": "sha256:6752b946590539dd40859b217448f4c4c18a6b37c9afa8346c6b3c94620eb94e",
    "inputs/baseline/independent-outcome.py": "sha256:d1bb69a4ad1cc28c8118601311916838efe4e709de2f6c7935d5132373c56c6e",
    "inputs/ACTUAL_PROTECTED_CONTROL_QUALIFICATION.json": "sha256:ba80f009078bcf6e932595464582e284d23bfb202551c923d9706542932093d4",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class CodingControlledUpdateConfig:
    """One fixed offline coding trial; this declaration is not a general trainer API."""

    prepared_manifest: Path
    prepared_manifest_digest: str
    subject_ref: str
    profile: str = CODING_UPDATE_PROFILE

    def __post_init__(self) -> None:
        if (
            self.profile != CODING_UPDATE_PROFILE
            or not isinstance(self.prepared_manifest, Path)
            or not self.prepared_manifest.is_absolute()
        ):
            raise ValueError(
                "coding controlled update requires its declared profile and absolute prepared path"
            )
        if any(
            not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
            for value in (self.prepared_manifest_digest, self.subject_ref)
        ):
            raise ValueError(
                "coding controlled update requires manifest and subject byte identities"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodingControlledUpdateConfig":
        if set(value) != {
            "profile",
            "prepared_manifest",
            "prepared_manifest_digest",
            "subject_ref",
        }:
            raise ValueError("coding controlled update requires the exact declared profile fields")
        if value["profile"] != CODING_UPDATE_PROFILE:
            raise ValueError("unsupported coding controlled-update profile")
        if (
            not isinstance(value["prepared_manifest"], str)
            or not Path(value["prepared_manifest"]).is_absolute()
        ):
            raise ValueError("coding prepared_manifest must be an absolute local path")
        for name in ("prepared_manifest_digest", "subject_ref"):
            if not isinstance(value[name], str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value[name]
            ):
                raise ValueError("coding controlled update requires an authoritative " + name)
        return cls(
            Path(value["prepared_manifest"]),
            value["prepared_manifest_digest"],
            value["subject_ref"],
        )

    def canonical_mapping(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "prepared_manifest": str(self.prepared_manifest),
            "prepared_manifest_digest": self.prepared_manifest_digest,
            "subject_ref": self.subject_ref,
        }

    @property
    def digest(self) -> str:
        return contracts.digest(self.canonical_mapping())


def validate_coding_preparation(config: CodingControlledUpdateConfig) -> dict[str, Any]:
    """Read and bind the finite prepared inputs before importing or loading a model."""
    if (
        config.profile != CODING_UPDATE_PROFILE
        or _file_digest(config.prepared_manifest) != config.prepared_manifest_digest
    ):
        raise ValueError("coding preparation manifest digest or profile mismatch")
    manifest = json.loads(config.prepared_manifest.read_text())
    if (
        manifest["schema"] != "reward-lens.m3-local-coding-trial-preparation/1"
        or manifest["candidate"] != "text-penalty"
    ):
        raise ValueError("unsupported coding preparation")
    if manifest["candidate_subject_ref"] != config.subject_ref:
        raise ValueError("coding preparation subject differs from public subject")
    checkpoint = Path(manifest["checkpoint"])
    if (
        checkpoint.name != _CODING_REVISION
        or manifest["checkpoint_files"].get("model.safetensors") != _CODING_WEIGHTS
    ):
        raise ValueError(
            "unsupported coding checkpoint; requires the pinned complete Qwen checkpoint"
        )
    if (
        manifest["checkpoint_files"] != _CODING_CHECKPOINT_PINS
        or {key: manifest["runtime"].get(key) for key in _CODING_RUNTIME_PINS}
        != _CODING_RUNTIME_PINS
    ):
        raise ValueError("unsupported coding checkpoint or runtime preparation pins")
    if not {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    } <= set(manifest["checkpoint_files"]):
        raise ValueError("coding checkpoint declaration is incomplete")
    for directory, files in (
        (config.prepared_manifest.parent, manifest["input_files"]),
        (checkpoint, manifest["checkpoint_files"]),
        (Path(manifest["source_root"]) / "src/reward_lens", manifest["source_files"]),
    ):
        for relative, expected in files.items():
            path = directory / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or (
                    directory != checkpoint
                    and not path.resolve().is_relative_to(directory.resolve())
                )
                or _file_digest(path) != expected
            ):
                raise ValueError("coding preparation byte binding mismatch: " + relative)
    if (
        not Path(__file__)
        .resolve()
        .is_relative_to(Path(manifest["source_root"]).resolve() / "src/reward_lens")
    ):
        raise ValueError("coding preparation source import origin mismatch")
    required_sources = {
        "learning/intervention/swap.py",
        "learning/intervention/clamp.py",
        "product/compare/rung4.py",
        "product/compare/verb.py",
        "api/requests.py",
        "graders/python_grader.py",
    }
    if set(manifest["source_files"]) != required_sources:
        raise ValueError("coding preparation source binding is incomplete")
    if any(
        manifest["input_files"].get(path) != digest for path, digest in _CODING_INPUT_PINS.items()
    ):
        raise ValueError("unsupported coding grader, tasks, responses or protected qualification")
    if (
        manifest["input_files"].get("inputs/text-penalty/tasks.jsonl")
        != _CODING_INPUT_PINS["inputs/baseline/tasks.jsonl"]
        or manifest["input_files"].get("inputs/text-penalty/baseline-grader.py")
        != _CODING_INPUT_PINS["inputs/baseline/grader.py"]
    ):
        raise ValueError("coding baseline and candidate do not share correctness tests")
    for name in ("check.py", "test_solution.py"):
        if (
            manifest["input_files"].get("inputs/text-penalty/outcome/" + name)
            != _CODING_INPUT_PINS["inputs/baseline/outcome/" + name]
        ):
            raise ValueError("coding revisions changed the shared protected correctness suite")
    from reward_lens.api._record import absent_record
    from reward_lens.store import Project
    from reward_lens.store.digests import method_set

    project_path = config.prepared_manifest.parent / "inputs/text-penalty"
    provisional = absent_record(
        command="compare", reproduce=[], path=project_path / "controlled-update.json"
    )
    methods = set(method_set(provisional.to_dict()))
    absence = provisional.measurement.reward_statistics[0].method
    methods.discard(f"{absence.id}@{absence.version}")
    methods.add("reward_statistics.controlled_update_response@1.0.0")
    if (
        Project.open(project_path).version(methods=tuple(sorted(methods))).digest()
        != config.subject_ref
    ):
        raise ValueError("coding prepared grader project does not match the public subject")
    if manifest["domain"] != {
        "task_count": 40,
        "reference_response_count": 120,
        "one_update_task_ids": ["task-01", "task-02"],
        "generations_per_prompt": 2,
        "materialized_rows": 4,
        "baseline_and_candidate_task_tests_equal": True,
        "new_generation_not_reference_bank_replay": True,
    }:
        raise ValueError("unsupported coding task and completion protocol")
    if manifest["limits"] != {
        "one_step_per_arm": 1,
        "prompt_count": 2,
        "max_completion_length": 128,
        "learning_rate": 0.00001,
        "sandbox_required": "L3",
        "scoring_wall_s": 30,
    }:
        raise ValueError("unsupported coding update limits")
    model_config = json.loads((checkpoint / "config.json").read_text())
    if model_config.get("model_type") != "qwen2" or model_config.get("architectures") != [
        "Qwen2ForCausalLM"
    ]:
        raise ValueError("unsupported coding checkpoint architecture")
    if (Path(manifest["source_root"]) / "fleet/KILL_SWITCH").exists():
        raise ValueError("coding update kill switch is present")
    spec = importlib.util.spec_from_file_location(
        "reward_lens_coding_protected_controls",
        config.prepared_manifest.parent / "inputs/baseline/outcome/check.py",
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    qualification = evaluator.qualify()
    if (
        not qualification.get("qualified")
        or qualification.get("known_good") != 12
        or qualification.get("known_wrong") != 12
    ):
        raise ValueError("protected coding evaluator controls did not qualify")
    return manifest


def _coding_backend(manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    """Require the pinned trainer and real local CUDA access without a toy fallback."""
    try:
        if importlib.metadata.version("trl") != "1.10.0":
            raise ValueError("requires trl==1.10.0")
        for name in (
            "torch",
            "transformers",
            "datasets",
            "accelerate",
            "numpy",
            "tokenizers",
            "safetensors",
        ):
            if importlib.metadata.version(name) != manifest["runtime"][name]:
                raise ValueError(
                    "coding runtime byte preparation has a different " + name + " version"
                )
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        from reward_lens.execution.probe import default_sandbox

        return torch, default_sandbox(require_tier="L3")
    except contracts.CapabilityUnavailable:
        raise
    except (ImportError, ValueError) as error:
        raise contracts.CapabilityUnavailable(
            code="RL0702",
            message="coding controlled-update backend unavailable: " + str(error),
            remediation="provide the pinned offline trl==1.10.0 CUDA backend and an L3 sandbox in a separately authorized execution slot",
            context={
                "profile": CODING_UPDATE_PROFILE,
                "family": "trl==1.10.0",
                "capability": "offline CUDA controlled coding update",
            },
        ) from error


class _CodingTRL110StateAdapter(TRL110StateAdapter):
    """Bind all visible CUDA generators within the existing RNG state object."""

    def __init__(self, trainer: Any) -> None:
        super().__init__(trainer)
        self._cuda_rng_inventory = self._cuda_inventory()

    @staticmethod
    def _cuda_inventory() -> dict[str, Any]:
        import torch

        count = torch.cuda.device_count()
        if not torch.cuda.is_available() or count < 1:
            raise ValueError("TRAINER_STATE_INCOMPLETE: coding CUDA RNG devices unavailable")
        devices = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            uuid = getattr(properties, "uuid", None)
            if not isinstance(uuid, str) or not uuid:
                raise ValueError(
                    "TRAINER_STATE_INCOMPLETE: CUDA RNG device UUID provenance unavailable"
                )
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": uuid,
                    "compute_capability": (properties.major, properties.minor),
                    "total_memory": properties.total_memory,
                }
            )
        return {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": tuple(devices),
        }

    def _require_cuda_inventory(self, inventory: Any) -> None:
        if (
            inventory != self._cuda_rng_inventory
            or self._cuda_inventory() != self._cuda_rng_inventory
        ):
            raise ValueError("SAME_START_REFUSAL: divergent CUDA RNG device provenance")

    def _require_cuda_states(self, states: Any) -> None:
        import torch

        if not isinstance(states, (tuple, list)) or len(states) != len(
            self._cuda_rng_inventory["devices"]
        ):
            raise ValueError("TRAINER_STATE_INCOMPLETE: CUDA RNG device cardinality mismatch")
        if any(
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
            for state in states
        ):
            raise ValueError("TRAINER_STATE_INCOMPLETE: malformed CUDA RNG device state")

    @property
    def rng_states(self) -> Mapping[str, Any]:
        import torch

        self._require_cuda_inventory(self._cuda_rng_inventory)
        states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        self._require_cuda_states(states)
        return {
            **super().rng_states,
            "cuda_inventory": copy.deepcopy(self._cuda_rng_inventory),
            "torch_cuda": states,
        }

    @rng_states.setter
    def rng_states(self, snapshot: Mapping[str, Any]) -> None:
        import torch

        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "python",
            "numpy",
            "torch_cpu",
            "cuda_inventory",
            "torch_cuda",
        }:
            raise ValueError("TRAINER_STATE_INCOMPLETE: coding CUDA RNG snapshot is incomplete")
        # Validate every CUDA device before changing any of the host RNG state.
        self._require_cuda_inventory(snapshot["cuda_inventory"])
        self._require_cuda_states(snapshot["torch_cuda"])
        TRL110StateAdapter.rng_states.fset(self, snapshot)
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _run_coding_update(
    config: CodingControlledUpdateConfig, manifest: Mapping[str, Any], artifact_dir: Path
) -> tuple[ControlledUpdateResult, list[dict[str, Any]], str]:
    torch, sandbox = _coding_backend(manifest)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from reward_lens.execution import Limits
    from reward_lens.graders.python_grader import PythonGrader

    root = config.prepared_manifest.parent / "inputs"
    projects = {name: root / name for name in ("baseline", "text-penalty")}
    graders = {name: PythonGrader.from_path(path / "grader.py") for name, path in projects.items()}
    independent = PythonGrader.from_path(projects["baseline"] / "independent-outcome.py")
    limits = Limits(
        wall_s=30,
        cpu_s=30,
        memory_bytes=2 * 1024**3,
        processes=64,
        stdout_bytes=65536,
        artifact_bytes=65536,
    )
    tasks = [
        json.loads(line)
        for line in (projects["baseline"] / "tasks.jsonl").read_text().splitlines()
        if line
    ]
    chosen = [
        next(task for task in tasks if task["task_id"] == ident)
        for ident in manifest["domain"]["one_update_task_ids"]
    ]
    prompt_tasks = {
        task["prompt"] + "\nReturn one fenced Python implementation.": dict(
            task, id=task["task_id"]
        )
        for task in chosen
    }
    observations: list[dict[str, Any]] = []
    measured_offsets: list[float] = []

    def reward(prompts, completions, **kwargs):
        del kwargs
        base_values = []
        measured_offsets.clear()
        for prompt, response in zip(prompts, completions, strict=True):
            task = prompt_tasks[prompt]
            envelopes = {
                name: grader.score(
                    task,
                    response,
                    sandbox=sandbox,
                    limits=limits,
                    project_dir=projects[name],
                    seed=37,
                )
                for name, grader in graders.items()
            }
            if any(env.verdict != "scored" or env.score is None for env in envelopes.values()):
                raise ValueError("coding reward execution refused")
            base = float(envelopes["baseline"].score)
            candidate = float(envelopes["text-penalty"].score)
            outcome = independent.score(
                task,
                response,
                sandbox=sandbox,
                limits=limits,
                project_dir=projects["baseline"],
                seed=37,
            )
            if outcome.verdict != "scored" or outcome.score not in (0.0, 1.0):
                raise ValueError("protected coding outcome execution refused")
            base_values.append(base)
            measured_offsets.append(candidate - base)
            observations.append(
                {
                    "task_id": task["task_id"],
                    "completion": response,
                    "baseline": base,
                    "candidate": candidate,
                    "envelopes": {name: env.to_dict() for name, env in envelopes.items()},
                    "independent_frozen_response_outcome": outcome.to_dict(),
                }
            )
        return base_values

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
    device = next(model.parameters()).device.type
    if device != "cuda":
        raise ValueError("coding model did not load on CUDA")
    args = GRPOConfig(
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
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=args,
        train_dataset=Dataset.from_dict({"prompt": list(prompt_tasks)}),
        processing_class=tokenizer,
    )
    trainer.create_optimizer_and_scheduler(1)
    adapter = _CodingTRL110StateAdapter(trainer)

    def arm(name, factor):
        def apply(host, replay):
            if len(measured_offsets) != sum(
                len(scores) for scores in replay.base_scores
            ) or not any(abs(value) > 0 for value in measured_offsets):
                raise ValueError(
                    "coding revision has no measured nonzero effect on the frozen generated batch"
                )
            return host.run_one_update(
                replay=replay,
                intervention=Intervention(
                    name, tuple(float(factor * value) for value in measured_offsets)
                ),
            )

        return SwapArm(name, apply)

    rows = tuple({"prompt": prompt} for prompt in prompt_tasks for _ in range(2))
    result = run_swap(
        MaterializedBatch(rows),
        [arm("candidate", 1), arm("placebo", -1)],
        TrainerProfile("1.10.0", EIGHT_OBJECTS, CODING_UPDATE_PROFILE),
        trainer=adapter,
        null=arm("null", 0),
    )
    return result, observations, device


def _validate_coding_execution(result: ControlledUpdateResult, profile: Mapping[str, Any]) -> None:
    values = _values(result)
    observations = profile["reward_observations"]
    if (
        profile["profile"] != CODING_UPDATE_PROFILE
        or profile["device"] not in {"cuda", "controlled-test-loader"}
        or len(observations) != 4
    ):
        raise ValueError("coding execution lacks the bounded four-completion evidence")
    if [row["task_id"] for row in observations] != ["task-01", "task-01", "task-02", "task-02"]:
        raise ValueError("coding execution task grouping differs from the declared protocol")
    offsets = tuple(float(row["candidate"] - row["baseline"]) for row in observations)
    if not any(offsets):
        raise ValueError("coding candidate has no measured reward effect")
    for row in observations:
        if not isinstance(row["completion"], str) or not row["completion"]:
            raise ValueError("coding execution lost an actual generated completion")
        if profile["device"] == "cuda":
            for env in (*row["envelopes"].values(), row["independent_frozen_response_outcome"]):
                request = env["observations"]["request"]
                if (
                    request["response"] != row["completion"]
                    or request["task"]["task_id"] != row["task_id"]
                    or env["request_digest"] != contracts.digest(request)
                    or env["provenance"]["sandbox_tier"] != "L3"
                ):
                    raise ValueError(
                        "coding scoring receipt does not bind actual generation and protected access"
                    )
        for name, score in (("baseline", row["baseline"]), ("text-penalty", row["candidate"])):
            env = row["envelopes"][name]
            if env["verdict"] != "scored" or env["score"] != score:
                raise ValueError("coding reward receipt does not bind the measured score")
        outcome = row["independent_frozen_response_outcome"]
        if outcome["verdict"] != "scored" or outcome["score"] not in (0.0, 1.0):
            raise ValueError("protected coding outcome receipt is absent or refused")
    for name, factor in (("null", 0), ("candidate", 1), ("placebo", -1)):
        value = values[name]
        wanted = tuple(factor * offset for offset in offsets)
        if value["intervention"].reward_offsets != wanted:
            raise ValueError("coding null, candidate or placebo offsets do not bind actual scores")
        sources = tuple(
            float(score) for call in value["reward_witness"] for score in call["source_values"]
        )
        returned = tuple(
            float(score) for call in value["reward_witness"] for score in call["returned_values"]
        )
        if sources != tuple(row["baseline"] for row in observations) or returned != tuple(
            base + offset for base, offset in zip(sources, wanted, strict=True)
        ):
            raise ValueError("coding reward witness does not bind actual scores and intervention")


def build_coding_controlled_update(
    config: CodingControlledUpdateConfig, artifact_dir: Path
) -> tuple[ControlledUpdateResult, "ControlledUpdateEvidence"]:
    """Execute one declared coding profile and retain it through the public R4 consumer."""
    if not isinstance(config, CodingControlledUpdateConfig) or not isinstance(artifact_dir, Path):
        raise TypeError(
            "build_coding_controlled_update requires CodingControlledUpdateConfig and Path"
        )
    started = datetime.now(timezone.utc)
    timer = time.perf_counter()
    try:
        manifest = validate_coding_preparation(config)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise contracts.UsageError(
            code="RL0131",
            message="coding preparation refused: " + str(error),
            remediation="rebuild and bind the exact supported offline coding preparation",
            context={"profile": config.profile},
        ) from error
    try:
        result, observations, device = _run_coding_update(config, manifest, artifact_dir)
    except ValueError as error:
        raise contracts.UsageError(
            code="RL0131",
            message="coding update control refused: " + str(error),
            remediation="inspect the actual frozen generation, reward and restoration controls before making a controlled-update claim",
            context={"profile": config.profile},
        ) from error
    profile = {
        "profile": config.profile,
        "device": device,
        "execution_class": "actual_cuda_update" if device == "cuda" else "controlled_test_loader",
        "prepared_manifest_digest": config.prepared_manifest_digest,
        "checkpoint": manifest["checkpoint"],
        "checkpoint_files": manifest["checkpoint_files"],
        "input_files": manifest["input_files"],
        "source_files": manifest["source_files"],
        "domain": manifest["domain"],
        "runtime": manifest["runtime"],
        "reward_observations": observations,
        "claim_boundary": "one controlled update and frozen-response outcomes only; no behavioral or capability-retention result",
    }
    try:
        _validate_coding_execution(result, profile)
    except (KeyError, TypeError, ValueError) as error:
        raise contracts.UsageError(
            code="RL0131",
            message="coding update evidence refused: " + str(error),
            remediation="restore the exact actual scoring, null, placebo and state witnesses",
            context={"profile": config.profile},
        ) from error
    identities = _execution_identities(result, config.digest, config.subject_ref)
    module_digest = _sha256(Path(__file__).read_bytes())
    payload = _execution_artifact_payload(
        result, config.digest, config.subject_ref, module_digest, profile
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / "controlled-update-execution.json"
    artifact.write_bytes(contracts.canonical_bytes(payload))
    binding = {
        "execution": "in-memory local pinned GRPOTrainer",
        "artifact_path": str(artifact),
        "artifact_digest": _file_digest(artifact),
        "module_source_digest": module_digest,
        "trace_projection_digest": "",
    }
    trace = _execution_trace(result, identities, binding, profile)
    binding["trace_projection_digest"] = contracts.digest(_trace_projection(trace))
    evidence = ControlledUpdateEvidence(
        trace,
        contracts.EntryProvenance(
            started=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_s=time.perf_counter() - timer,
            sandbox_tier="L3" if device == "cuda" else "T0",
            offline=True,
        ),
        identities,
        binding,
        profile,
    )
    evidence.require_result(result)
    return result, evidence


@dataclass(frozen=True)
class ControlledUpdateEvidence:
    """Measured execution identities and provenance retained beside one update."""

    trace: TraceRecord
    provenance: contracts.EntryProvenance
    identities: Mapping[str, str] = field(default_factory=dict)
    source_binding: Mapping[str, str] = field(default_factory=dict)
    execution_profile: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trace, TraceRecord):
            raise TypeError("controlled update requires the actual TraceRecord input")
        if not isinstance(self.provenance, contracts.EntryProvenance):
            raise TypeError("controlled update requires actual EntryProvenance")
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        object.__setattr__(self, "source_binding", MappingProxyType(dict(self.source_binding)))
        if self.identities:
            required = {
                "batch_digest",
                "scored_replay_digest",
                "loss_input_digest",
                "config_digest",
                "declared_subject_ref",
                "execution_digest",
            }
            if set(self.identities) != required or any(
                not value.startswith("sha256:") for value in self.identities.values()
            ):
                raise ValueError("controlled update requires canonical execution identities")
            if self.trace.reader != "controlled_update_execution" or self.trace.complete:
                raise ValueError("controlled update trace must be an incomplete execution record")
            if self.trace.subject_ref != self.identities["declared_subject_ref"]:
                raise ValueError("controlled update trace subject must bind the declared subject")
            expected_dependencies = tuple(
                self.identities[name]
                for name in (
                    "batch_digest",
                    "scored_replay_digest",
                    "loss_input_digest",
                    "config_digest",
                )
            )
            if self.trace.depends_on != expected_dependencies or not self.source_binding:
                raise ValueError("controlled update trace must bind actual inputs and source")
        elif (
            not self.trace.complete
            or not self.trace.subject_ref.startswith("sha256:")
            or not self.trace.depends_on
        ):
            raise ValueError(
                "controlled update requires a complete TraceRecord with subject and dependency identities"
            )

    def require_result(self, result: ControlledUpdateResult) -> None:
        """Refuse a result whose retained execution identity changed after the run."""
        if not self.identities:
            return
        try:
            actual = _execution_identities(
                result, self.identities["config_digest"], self.identities["declared_subject_ref"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "controlled update execution identities do not match the retained run"
            ) from error
        if actual != dict(self.identities):
            raise ValueError("controlled update execution identities do not match the retained run")
        if self.execution_profile:
            _validate_coding_execution(result, self.execution_profile)
        _validate_execution_source(
            result, actual, self.trace, self.source_binding, self.execution_profile
        )


def build_cpu_controlled_update(
    config: CpuControlledUpdateConfig, artifact_dir: Path
) -> tuple[ControlledUpdateResult, ControlledUpdateEvidence]:
    """Execute the bounded, random-weight, CPU-only pinned TRL conformance run."""
    if not isinstance(config, CpuControlledUpdateConfig) or not isinstance(artifact_dir, Path):
        raise TypeError("build_cpu_controlled_update requires CpuControlledUpdateConfig and Path")
    import importlib.metadata

    from reward_lens.learning.contracts import TrainerUnsupported

    try:
        version = importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    if version != "1.10.0":
        raise TrainerUnsupported(version)
    import numpy as np
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ACCELERATE_USE_CPU"] = "true"
    torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    raw = Tokenizer(
        WordLevel(
            vocab={"<pad>": 0, "<unk>": 1, "<eos>": 2, "say": 3, "one": 4, "two": 5},
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    )

    def reward(prompts, completions, completion_ids, **kwargs):
        return [float(i % 2) for i, _ in enumerate(completions)]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=str(artifact_dir),
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=config.max_completion_length,
            max_steps=1,
            logging_strategy="no",
            report_to=[],
            use_cpu=True,
            optim="adamw_torch",
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            learning_rate=1e-4,
        ),
        train_dataset=Dataset.from_dict({"prompt": ["say one", "say two"]}),
        processing_class=tokenizer,
    )
    trainer.create_optimizer_and_scheduler(1)
    host = TRL110StateAdapter(trainer)
    batch = MaterializedBatch(({"prompt": "say one"}, {"prompt": "say two"}))
    offsets = {
        "null": (0.0, 0.0),
        "candidate": config.candidate_offsets,
        "placebo": (-config.candidate_offsets[0], -config.candidate_offsets[1]),
    }

    def arm(name: str) -> SwapArm:
        return SwapArm(
            name,
            lambda h, replay: h.run_one_update(
                replay=replay, intervention=Intervention(name, offsets[name])
            ),
        )

    result = run_swap(
        batch,
        [arm("candidate"), arm("placebo")],
        TrainerProfile("1.10.0", EIGHT_OBJECTS, "cpu-controlled-update-v1"),
        trainer=host,
        null=arm("null"),
    )
    identities = _execution_identities(result, config.digest, config.subject_ref)
    module_source_digest = _sha256(Path(__file__).read_bytes())
    artifact_payload = _execution_artifact_payload(
        result, config.digest, config.subject_ref, module_source_digest
    )
    artifact_bytes = contracts.canonical_bytes(artifact_payload)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "controlled-update-execution.json"
    artifact_path.write_bytes(artifact_bytes)
    source_binding = {
        "execution": "in-memory local pinned GRPOTrainer",
        "artifact_path": str(artifact_path.resolve()),
        "artifact_digest": _sha256(artifact_bytes),
        "module_source_digest": module_source_digest,
    }
    trace = _execution_trace(result, identities, source_binding)
    source_binding["trace_projection_digest"] = contracts.digest(_trace_projection(trace))
    evidence = ControlledUpdateEvidence(
        trace,
        contracts.EntryProvenance(
            started=started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_s=time.perf_counter() - started_timer,
            sandbox_tier="T0",
            offline=True,
        ),
        identities=identities,
        source_binding=source_binding,
    )
    return result, evidence


@dataclass(frozen=True)
class ControlledUpdateRecord:
    scope: str
    result: ControlledUpdateResult
    evidence: ControlledUpdateEvidence
    arm_results: Mapping[str, ArmResult]
    claim_boundary: str

    def __post_init__(self) -> None:
        if self.scope != _SCOPE or self.result.scope != _SCOPE:
            raise ValueError("rung 4 record and controlled result scope must agree")
        if "mechanism conformance, not a behavioural result" not in self.claim_boundary.lower():
            raise ValueError("rung 4 must preserve its mechanism-only boundary")
        if tuple(self.arm_results) != ("base", "null", "candidate", "placebo") or not all(
            isinstance(a, ArmResult) for a in self.arm_results.values()
        ):
            raise ValueError("rung 4 requires strict base, null, candidate, and placebo ArmResults")
        object.__setattr__(self, "arm_results", MappingProxyType(dict(self.arm_results)))


def _profile(result: ControlledUpdateResult) -> dict[str, Any]:
    return {
        "version": result.profile.version,
        "mutable_objects": list(result.profile.mutable_objects),
        "fingerprint": result.profile.fingerprint,
    }


def _values(result: ControlledUpdateResult) -> dict[str, Mapping[str, Any]]:
    expected = ("null", "candidate", "placebo")
    if result.arm_order != expected or tuple(name for name, _ in result.arms) != expected:
        raise ValueError("controlled update requires null, candidate, and placebo in that order")
    output: dict[str, Mapping[str, Any]] = {}
    required = {
        "batch_digest",
        "batch_bytes",
        "scored_replay_digest",
        "loss_input_digest",
        "intervention",
        "reward_witness",
        "loss",
        "gradient_norm",
        "optimizer_update_norm",
        "advantages",
        "changed_private_fields",
    }
    for name, witness in result.arms:
        if not isinstance(witness, Mapping) or not isinstance(witness.get("value"), Mapping):
            raise ValueError(f"controlled update arm {name!r} has no retained actual result")
        value = witness["value"]
        if required - set(value) or not isinstance(value["intervention"], Intervention):
            raise ValueError(f"controlled update arm {name!r} lacks typed retained evidence")
        during = witness.get("during_digest")
        restored = witness.get("restored_digest")
        if not isinstance(during, str) or not during or during == result.start_digest:
            raise ValueError(f"controlled update arm {name!r} lacks a present changed-state digest")
        if (
            not isinstance(restored, str)
            or not restored
            or restored not in {result.start_digest, result.final_digest}
        ):
            raise ValueError(
                f"controlled update arm {name!r} lacks changed-and-restored state evidence"
            )
        if (
            value["intervention"].name != name
            or value["gradient_norm"] <= 0
            or value["optimizer_update_norm"] <= 0
        ):
            raise ValueError(f"controlled update arm {name!r} lacks a typed nonzero update witness")
        output[name] = value
    if result.final_digest != result.start_digest:
        raise ValueError("controlled update final state does not equal its shared baseline")
    for field_name in ("batch_digest", "batch_bytes", "scored_replay_digest", "loss_input_digest"):
        if any(
            not isinstance(value[field_name], (str, bytes)) or not value[field_name]
            for value in output.values()
        ):
            raise ValueError(f"controlled update arms lack a present {field_name}")
        if len({value[field_name] for value in output.values()}) != 1:
            raise ValueError(f"controlled update arms do not share one {field_name}")
    return output


def _execution_payload(result: ControlledUpdateResult, subject_ref: str) -> dict[str, Any]:
    """The exact executed facts whose JCS identity admits a factory result."""
    values = _values(result)
    arms: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        intervention = value["intervention"]
        assert isinstance(intervention, Intervention)
        reward_witness = tuple(
            {
                "source_values": list(call["source_values"]),
                "returned_values": list(call["returned_values"]),
            }
            for call in value["reward_witness"]
        )
        arms[name] = {
            "batch_digest": value["batch_digest"],
            "batch_bytes_hex": value["batch_bytes"].hex(),
            "scored_replay_digest": value["scored_replay_digest"],
            "loss_input_digest": value["loss_input_digest"],
            "reward_witness": reward_witness,
            "advantages": list(value["advantages"]),
            "loss": value["loss"],
            "gradient_norm": value["gradient_norm"],
            "optimizer_update_norm": value["optimizer_update_norm"],
            "intervention": {
                "name": intervention.name,
                "reward_offsets": list(intervention.reward_offsets),
            },
        }
    return {
        "declared_subject_ref": subject_ref,
        "start_digest": result.start_digest,
        "final_digest": result.final_digest,
        "profile": _profile(result),
        "arms": arms,
    }


def _execution_artifact_payload(
    result: ControlledUpdateResult,
    config_digest: str,
    subject_ref: str,
    module_source_digest: str,
    execution_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The retained JCS artifact: actual update facts plus the executing module bytes."""
    return {
        **({"coding_execution": dict(execution_profile)} if execution_profile else {}),
        "schema": "reward-lens.controlled-update-execution/v1",
        "config_digest": config_digest,
        "declared_subject_ref": subject_ref,
        "module_source_digest": module_source_digest,
        "execution": _execution_payload(result, subject_ref),
    }


def _execution_identities(
    result: ControlledUpdateResult, config_digest: str, subject_ref: str
) -> dict[str, str]:
    """RFC 8785 identities derived from actual replay and update witnesses."""
    values = _values(result)
    shared = values["null"]
    batch_digest = contracts.digest(
        {
            "batch_digest": shared["batch_digest"],
            "batch_bytes_hex": shared["batch_bytes"].hex(),
        }
    )
    replay_digest = contracts.digest({"scored_replay_digest": shared["scored_replay_digest"]})
    loss_input_digest = contracts.digest({"loss_input_digest": shared["loss_input_digest"]})
    payload = _execution_payload(result, subject_ref)
    return {
        "batch_digest": batch_digest,
        "scored_replay_digest": replay_digest,
        "loss_input_digest": loss_input_digest,
        "config_digest": config_digest,
        "declared_subject_ref": subject_ref,
        "execution_digest": contracts.digest(
            {
                "config_digest": config_digest,
                "declared_subject_ref": subject_ref,
                "execution": payload,
            }
        ),
    }


def _trace_projection(trace: TraceRecord) -> dict[str, Any]:
    """Canonical, complete projection of every trace fact admitted as evidence."""
    return {
        "reader": trace.reader,
        "source": trace.source,
        "run_id": trace.run_id,
        "rows": [
            {
                "index": row.index,
                "step": row.step,
                "group_id": row.group_id,
                "row_id": row.row_id,
                "components": dict(row.components),
                "reward_recorded": row.reward_recorded,
                "reward_applied": row.reward_applied,
                "advantage": row.advantage,
                "selection": row.selection,
            }
            for row in trace.rows
        ],
        "series": {
            name: {
                "name": series.name,
                "values": list(series.values),
                "components": list(series.components),
                "definition": series.definition,
                "present": series.present,
                "absent_reason": series.absent_reason,
            }
            for name, series in trace.series.items()
        },
        "components": {
            name: {
                "name": series.name,
                "values": list(series.values),
                "definition": series.definition,
                "present": series.present,
            }
            for name, series in trace.components.items()
        },
        "access_level": trace.access_level,
        "access_statement": trace.access_statement,
        "policy": trace.policy,
        "complete": trace.complete,
        "completeness_statement": trace.completeness_statement,
        "subject_ref": trace.subject_ref,
        "depends_on": list(trace.depends_on),
        "framework": trace.framework,
        "framework_version": trace.framework_version,
        "limitations": list(trace.limitations),
    }


def _validate_execution_source(
    result: ControlledUpdateResult,
    identities: Mapping[str, str],
    trace: TraceRecord,
    source_binding: Mapping[str, str],
    execution_profile: Mapping[str, Any] | None = None,
) -> None:
    """Require the trace and retained JCS artifact to equal this executed result."""
    required = {
        "execution",
        "artifact_path",
        "artifact_digest",
        "module_source_digest",
        "trace_projection_digest",
    }
    if set(source_binding) != required:
        raise ValueError("controlled update source binding is incomplete")
    if source_binding["execution"] != "in-memory local pinned GRPOTrainer":
        raise ValueError("controlled update source binding does not name the executed trainer")
    module_source_digest = _sha256(Path(__file__).read_bytes())
    if source_binding["module_source_digest"] != module_source_digest:
        raise ValueError("controlled update source binding module bytes changed")
    artifact_path = Path(source_binding["artifact_path"])
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("controlled update retained execution artifact is unavailable") from error
    expected_artifact = contracts.canonical_bytes(
        _execution_artifact_payload(
            result,
            identities["config_digest"],
            identities["declared_subject_ref"],
            module_source_digest,
            execution_profile,
        )
    )
    if artifact_bytes != expected_artifact or source_binding["artifact_digest"] != _sha256(
        artifact_bytes
    ):
        raise ValueError("controlled update retained execution artifact does not match the run")
    expected_trace = _execution_trace(result, identities, source_binding, execution_profile)
    if trace != expected_trace:
        raise ValueError("controlled update trace does not match the retained execution projection")
    if source_binding["trace_projection_digest"] != contracts.digest(
        _trace_projection(expected_trace)
    ):
        raise ValueError("controlled update trace projection digest does not match the run")


def _execution_trace(
    result: ControlledUpdateResult,
    identities: Mapping[str, str],
    source_binding: Mapping[str, str],
    execution_profile: Mapping[str, Any] | None = None,
) -> TraceRecord:
    """Make a partial trace from returned candidate reward, advantage, and loss inputs."""
    candidate = _values(result)["candidate"]
    reward_calls = tuple(candidate["reward_witness"])
    source_rewards = tuple(float(value) for call in reward_calls for value in call["source_values"])
    applied_rewards = tuple(
        float(value) for call in reward_calls for value in call["returned_values"]
    )
    advantages = tuple(float(value) for value in candidate["advantages"])
    if (
        not source_rewards
        or len(source_rewards) != len(applied_rewards)
        or len(source_rewards) != len(advantages)
    ):
        raise ValueError(
            "controlled update cannot trace unmatched actual reward and advantage values"
        )
    rows = tuple(
        TraceRow(
            index=index,
            step=1,
            group_id=(
                execution_profile["reward_observations"][index]["task_id"]
                if execution_profile
                else identities["batch_digest"]
            ),
            row_id=contracts.digest(
                {"execution_digest": identities["execution_digest"], "index": index}
            ),
            components={"source_reward": source_rewards[index]},
            reward_recorded=source_rewards[index],
            reward_applied=applied_rewards[index],
            advantage=advantages[index],
        )
        for index in range(len(source_rewards))
    )
    return TraceRecord(
        reader="controlled_update_execution",
        source=source_binding["execution"],
        run_id="controlled-update-" + identities["execution_digest"][-16:],
        rows=rows,
        series={
            "reward_recorded": Series(
                "reward_recorded", source_rewards, definition="returned source reward values"
            ),
            "reward_applied": Series(
                "reward_applied", applied_rewards, definition="intervention-adjusted reward values"
            ),
            "advantage": Series("advantage", advantages, definition="pinned GRPO advantages"),
            "selection": Series(
                "selection", present=False, absent_reason="no selection measurement in one update"
            ),
        },
        components={
            "source_reward": Series(
                "source_reward", source_rewards, definition="source scorer output"
            ),
        },
        access_level="local",
        access_statement=(
            "in-memory local pinned " + execution_profile["device"] + " GRPOTrainer execution"
            if execution_profile
            else "in-memory local pinned CPU GRPOTrainer execution"
        ),
        policy="strict",
        complete=False,
        completeness_statement="partial controlled-update execution evidence; not a generic run record",
        subject_ref=identities["declared_subject_ref"],
        depends_on=tuple(
            identities[name]
            for name in (
                "batch_digest",
                "scored_replay_digest",
                "loss_input_digest",
                "config_digest",
            )
        ),
        framework="trl",
        framework_version="1.10.0",
        limitations=("mechanism conformance only; not a behavioural result",),
    )


def _base(result: ControlledUpdateResult, value: Mapping[str, Any]) -> ArmResult:
    return ArmResult.from_mapping(
        {
            "adapter_path": f"trl=={result.profile.version}",
            "rollouts": (
                {
                    "batch_digest": value["batch_digest"],
                    "batch_bytes": value["batch_bytes"],
                    "scored_replay_digest": value["scored_replay_digest"],
                },
            ),
            "arm": "base",
            "arm_kind": "frozen_scored_replay",
            "grader": "base_reward_scorer",
            "resumed_from": result.start_digest,
            "resume_restored": {"restored_digest": result.start_digest},
            "group_witness": {
                "batch_digest": value["batch_digest"],
                "scored_replay_digest": value["scored_replay_digest"],
            },
            "a2_step0": {"state_digest": result.start_digest},
            "parameters": {"trainer_profile": _profile(result)},
            "instance": {"scope": result.scope, "arm_order": list(result.arm_order)},
            "telemetry": {"final_restored_digest": result.final_digest},
        }
    )


def _updated(name: str, result: ControlledUpdateResult, value: Mapping[str, Any]) -> ArmResult:
    intervention = value["intervention"]
    assert isinstance(intervention, Intervention)
    return ArmResult.from_mapping(
        {
            "adapter_path": f"trl=={result.profile.version}",
            "rollouts": (
                {
                    "batch_digest": value["batch_digest"],
                    "batch_bytes": value["batch_bytes"],
                    "scored_replay_digest": value["scored_replay_digest"],
                    "loss_input_digest": value["loss_input_digest"],
                },
            ),
            "arm": name,
            "arm_kind": name,
            "grader": intervention.name,
            "steps_run": 1,
            "resumed_from": result.start_digest,
            "resume_restored": {"restored_digest": result.start_digest},
            "group_witness": {
                "batch_digest": value["batch_digest"],
                "scored_replay_digest": value["scored_replay_digest"],
                "loss_input_digest": value["loss_input_digest"],
            },
            "a2_step0": {
                "advantages": list(value["advantages"]),
                "reward_witness": list(value["reward_witness"]),
            },
            "parameters": {
                "trainer_profile": _profile(result),
                "intervention": {
                    "name": intervention.name,
                    "reward_offsets": list(intervention.reward_offsets),
                },
            },
            "instance": {"scope": result.scope, "arm_order": list(result.arm_order)},
            "telemetry": {
                "loss": value["loss"],
                "gradient_norm": value["gradient_norm"],
                "optimizer_update_norm": value["optimizer_update_norm"],
                "changed_private_fields": list(value["changed_private_fields"]),
            },
        }
    )


def controlled_update_record(
    result: ControlledUpdateResult, evidence: ControlledUpdateEvidence
) -> ControlledUpdateRecord:
    if not isinstance(result, ControlledUpdateResult) or not isinstance(
        evidence, ControlledUpdateEvidence
    ):
        raise TypeError("rung 4 requires actual run_swap result and retained TraceRecord evidence")
    evidence.require_result(result)
    values = _values(result)
    return ControlledUpdateRecord(
        _SCOPE,
        result,
        evidence,
        {
            "base": _base(result, values["null"]),
            **{
                name: _updated(name, result, values[name])
                for name in ("null", "candidate", "placebo")
            },
        },
        "Controlled-update mechanism conformance, not a behavioural result.",
    )


def _scorecard() -> Scorecard:
    return Scorecard.render(
        tuple(
            ScorecardRow(
                row,
                row.replace("_", " "),
                Absence(
                    "not measured: rung 4 establishes controlled-update mechanism conformance only",
                    "run a qualified short fork or held-out transfer protocol for this scorecard row",
                ),
                _SCOPE,
            )
            for row in ROWS
        ),
        rung="same_batch_swap",
    )


def _entry_dependencies(evidence: ControlledUpdateEvidence) -> list[str]:
    """Map retained execution facts to the frozen contract's dependency categories."""
    if evidence.identities:
        return [
            "digest:samples",
            "digest:scorer_config",
            "digest:training_semantics",
            "digest:instrument_method",
        ]
    return list(evidence.trace.depends_on)


def _absence(evidence: ControlledUpdateEvidence, reason: str) -> contracts.Entry:
    return contracts.Entry(
        entry_id=_ENTRY_ID,
        section="reward_statistics",
        kind="absence",
        measurand="controlled same-batch update response",
        method=contracts.Method(
            id=_ENTRY_ID,
            version="1.0.0",
            params_digest=contracts.digest(
                {
                    "reason": reason,
                    "trace": evidence.trace.source,
                    "identities": dict(evidence.identities),
                }
            ),
            procedure="require retained base, null, candidate, and placebo run_swap evidence",
        ),
        scope=_SCOPE,
        subject_ref=evidence.trace.subject_ref,
        depends_on=_entry_dependencies(evidence),
        state="absent",
        provenance=evidence.provenance,
        limitations=[
            "no controlled-update or behavioural claim is made when required arm evidence is absent"
        ],
        absence=contracts.Absence(
            state="COULD_NOT_CHECK",
            missing_access=reason,
            affected_claims=[
                "controlled update response",
                "null replay control",
                "placebo control",
            ],
            remedy="supply an actual run_swap result with all required arm evidence",
        ),
    )


def _arm_mapping(arm: ArmResult) -> dict[str, Any]:
    def json_value(value: Any) -> Any:
        if is_dataclass(value):
            return json_value(asdict(value))
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, Mapping):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        return value

    return {field.name: json_value(getattr(arm, field.name)) for field in fields(ArmResult)}


def _entry(record: ControlledUpdateRecord) -> contracts.Entry:
    arms = {name: _arm_mapping(arm) for name, arm in record.arm_results.items()}
    result = {
        "trace_record": {
            "source": record.evidence.trace.source,
            "run_id": record.evidence.trace.run_id,
            "reader": record.evidence.trace.reader,
            "subject_ref": record.evidence.trace.subject_ref,
            "depends_on": list(record.evidence.trace.depends_on),
            "source_binding": dict(record.evidence.source_binding),
            "execution_identities": dict(record.evidence.identities),
        },
        "controlled_result": {
            "start_digest": record.result.start_digest,
            "final_digest": record.result.final_digest,
            "arm_order": list(record.result.arm_order),
            "trainer_profile": _profile(record.result),
        },
        "arm_results": arms,
        "null_replay": {"passed": True, "arm": arms["null"]},
        "placebo": {"passed": True, "arm": arms["placebo"]},
        "claim_boundary": record.claim_boundary,
    }
    if record.evidence.execution_profile:
        result["coding_execution"] = dict(record.evidence.execution_profile)
    return contracts.Entry(
        entry_id=_ENTRY_ID,
        section="reward_statistics",
        kind="check",
        measurand="whether one frozen scored batch reaches every controlled update arm from the same restorable state",
        method=contracts.Method(
            id=_ENTRY_ID,
            version="1.0.0",
            params_digest=contracts.digest(
                {
                    "trace_subject_ref": record.evidence.trace.subject_ref,
                    "trace_execution_identities": dict(record.evidence.identities),
                    "controlled_result": result["controlled_result"],
                    "arm_results": arms,
                    **(
                        {"coding_execution": result["coding_execution"]}
                        if "coding_execution" in result
                        else {}
                    ),
                }
            ),
            procedure="freeze one actual scored replay, restore the declared state before each null, candidate, and placebo update, then retain strict product ArmResults",
        ),
        scope=_SCOPE,
        subject_ref=record.evidence.trace.subject_ref,
        depends_on=_entry_dependencies(record.evidence),
        state="complete",
        provenance=record.evidence.provenance,
        limitations=[
            "this establishes controlled-update mechanism conformance, not a behavioural result",
            "no capability retention, short-fork behaviour, held-out transfer, or causal outcome claim follows from one update",
        ],
        result=result,
        check=contracts.Check(
            predicate="all four required arm records retain one frozen base and passed null and placebo controls",
            passed=True,
            scope_tested="one pinned trainer profile and one retained same-batch controlled update",
        ),
    )


def controlled_update_compare(
    result: ControlledUpdateResult, evidence: ControlledUpdateEvidence
) -> CompareResult:
    """Return an entry the accepted Compare/Audit assembler can persist unchanged."""
    if not isinstance(evidence, ControlledUpdateEvidence):
        raise TypeError("rung 4 requires retained TraceRecord evidence")
    try:
        record = controlled_update_record(result, evidence)
    except (TypeError, ValueError) as error:
        return CompareResult(
            _SCOPE,
            _scorecard(),
            [_absence(evidence, str(error))],
            "no controlled-update claim because required arm evidence is absent or invalid",
            "a behavioural, capability-retention, short-fork, held-out transfer, or causal outcome result",
        )
    return CompareResult(
        _SCOPE,
        _scorecard(),
        [_entry(record)],
        "same-state, same-scored-batch controlled-update mechanism conformance with retained null and placebo controls",
        "a behavioural, capability-retention, short-fork, held-out transfer, or causal outcome result",
        record,
    )
