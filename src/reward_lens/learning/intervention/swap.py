"""Controlled same-batch update conformance for compare rung four.

The adapter deliberately knows only a small trainer protocol.  A supported
caller presents all eight mutable objects named by :class:`TrainerProfile` and
an ``apply_controlled_update`` method.  The CPU toy exercises that protocol;
it is mechanism conformance, never evidence of behaviour.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reward_lens.learning.contracts import EIGHT_OBJECTS, TrainerProfile
from reward_lens.learning.contracts.errors import TrainerStateIncomplete, TrainerUnsupported

__all__ = [
    "ControlledUpdateResult",
    "Intervention",
    "MaterializedBatch",
    "PINNED_TRAINER_MUTABLE_FIELDS",
    "ScoredReplayBundle",
    "SameStart",
    "SwapArm",
    "TRL110StateAdapter",
    "TrainerState",
    "capture_state",
    "restore_state",
    "run_swap",
    "same_start",
]


PINNED_TRAINER_MUTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "_buffered_inputs",
        "_step",
        "_current_train_step_time",
        "_metrics",
        "_logs",
        "_pending_extra_logs",
        "_pending_metrics",
    }
)


def _canonical(value: Any) -> Any:
    """Return a JSON-safe immutable projection without a repr fallback."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [_canonical(key), _canonical(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        }
    if isinstance(value, (list, tuple, deque)):
        return {"sequence": [_canonical(item) for item in value]}
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor = detach().cpu().contiguous()
        return {
            "tensor": {
                "dtype": str(tensor.dtype),
                "shape": tuple(int(part) for part in tensor.shape),
                "bytes": tensor.numpy().tobytes().hex(),
            }
        }
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return {
            "array": {
                "dtype": str(value.dtype),
                "shape": tuple(int(part) for part in value.shape),
                "bytes": value.tobytes().hex(),
            }
        }
    if hasattr(value, "__dict__"):
        return {"object": type(value).__qualname__, "fields": _canonical(vars(value))}
    raise TypeError(f"controlled state has no canonical encoding: {type(value)!r}")


def _digest(values: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical(values), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _can_digest(value: Any) -> bool:
    try:
        _digest({"value": value})
    except (TypeError, ValueError):
        return False
    return True


def _read_object(trainer: Any, name: str) -> Any:
    if name == "state_global_step":
        state = getattr(trainer, "state", None)
        if state is None or not hasattr(state, "global_step"):
            raise AttributeError(name)
        return state.global_step
    if not hasattr(trainer, name):
        raise AttributeError(name)
    return getattr(trainer, name)


def _write_object(trainer: Any, name: str, value: Any) -> None:
    if name == "state_global_step":
        trainer.state.global_step = value
    else:
        setattr(trainer, name, value)


def _learning_rate(trainer: Any) -> float:
    if hasattr(trainer, "learning_rate"):
        return float(trainer.learning_rate)
    optimizer = getattr(trainer, "optimizer", None)
    groups = getattr(optimizer, "param_groups", ())
    if groups:
        return float(groups[0]["lr"])
    raise AttributeError("learning_rate")


def _parameter_shape(weights: Any) -> tuple[int, ...]:
    shape = getattr(weights, "shape", None)
    if shape is not None:
        return tuple(int(part) for part in shape)
    if isinstance(weights, Mapping):
        return (len(weights), *sum((_parameter_shape(item) for item in weights.values()), ()))
    if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        return (len(weights),)
    return ()


def _optimizer_step_count(moments: Any) -> int:
    if isinstance(moments, Mapping):
        if "step" in moments:
            return int(moments["step"])
        state = moments.get("state")
        if isinstance(state, Mapping):
            steps = [
                int(value["step"])
                for value in state.values()
                if isinstance(value, Mapping) and "step" in value
            ]
            if steps:
                if len(set(steps)) != 1:
                    return -1
                return steps[0]
            return 0
    raise AttributeError("adam_moments.step")


def _fused_loss(trainer: Any) -> bool:
    if not hasattr(trainer, "fused_loss"):
        raise AttributeError("fused_loss")
    return bool(trainer.fused_loss)


@dataclass(frozen=True)
class TrainerState:
    """An immutable copy of every state object that one controlled step mutates."""

    values: Mapping[str, Any]
    learning_rate: float
    parameter_shape: tuple[int, ...]
    optimizer_step_count: int
    fused_loss: bool

    def digest(self) -> str:
        return _digest(
            {
                "objects": self.values,
                "learning_rate": self.learning_rate,
                "parameter_shape": self.parameter_shape,
                "optimizer_step_count": self.optimizer_step_count,
                "fused_loss": self.fused_loss,
            }
        )


@dataclass(frozen=True)
class SameStart:
    """A precise refusal reason when two controlled arms do not share a start."""

    valid: bool
    field: str | None = None

    def require(self) -> None:
        if not self.valid:
            raise ValueError(f"SAME_START_REFUSAL: divergent {self.field}")


@dataclass(frozen=True)
class SwapArm:
    """One named supplied update. `placebo` is supplied just like any other arm."""

    name: str
    apply: Callable[[Any, Any], Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("swap arm name must be non-empty")
        if not callable(self.apply):
            raise TypeError("swap arm apply must be callable")


@dataclass(frozen=True)
class MaterializedBatch:
    """Typed local GRPO rows with one canonical byte identity.

    The cursor consumes these rows directly. This makes the byte identity part
    of the controlled-update witness instead of a sidecar label.
    """

    rows: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        if not self.rows or any(type(row) is not dict for row in self.rows):
            raise TypeError("materialized batch requires non-empty plain row mappings")
        if any(set(row) != {"prompt"} or not isinstance(row["prompt"], str) for row in self.rows):
            raise ValueError("materialized GRPO rows must contain exactly a string prompt")

    @property
    def bytes(self) -> bytes:
        return json.dumps(self.rows, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.bytes).hexdigest()


@dataclass(frozen=True)
class Intervention:
    """Explicit per-completion values which enter the GRPO reward calculation."""

    name: str
    reward_offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.reward_offsets
            or any(type(value) is not float for value in self.reward_offsets)
        ):
            raise TypeError("intervention requires a name and one or more float reward offsets")


@dataclass(frozen=True)
class ScoredReplayBundle:
    """One frozen GRPO generation whose scored loss inputs are replayed by each arm."""

    batch: MaterializedBatch
    generation_payload: Any
    base_scores: tuple[tuple[float, ...], ...]
    base_reward_witness: tuple[Mapping[str, Any], ...]

    @property
    def digest(self) -> str:
        return _digest(
            {
                "batch_bytes": self.batch.bytes,
                "generation_payload": self.generation_payload,
                "base_scores": self.base_scores,
                "base_reward_witness": self.base_reward_witness,
            }
        )


@dataclass(frozen=True)
class ControlledUpdateResult:
    """The rung-four result, scoped to mechanism conformance only."""

    scope: str
    statement: str
    start_digest: str
    final_digest: str
    arm_order: tuple[str, ...]
    arms: tuple[tuple[str, Any], ...]
    profile: TrainerProfile

    def __post_init__(self) -> None:
        if self.scope != "controlled_update_response":
            raise ValueError("controlled update result scope must be controlled_update_response")
        if (
            "mechanism conformance" not in self.statement.lower()
            or "not a behavioural result" not in self.statement.lower()
        ):
            raise ValueError(
                "controlled update result must state its non-behavioural conformance limit"
            )
        if not self.arm_order or self.arm_order[0] != "null":
            raise ValueError("null replay must run before comparison arms")
        if "placebo" not in self.arm_order:
            raise ValueError("placebo must be a first-class controlled arm")
        if self.start_digest != self.final_digest:
            raise ValueError("controlled arms must restore the original state digest")


class TRL110StateAdapter:
    """One explicit pinned-GRPO update over a restorable materialized cursor.

    ``GRPOTrainer.train`` owns an opaque dataloader and may choose a different
    batch for each invocation. Rung four instead consumes ``MaterializedBatch``
    rows through this adapter's cursor, calls the pinned GRPO generation, reward,
    loss, backward, optimiser, and scheduler path exactly once, and snapshots
    every mutable trainer and loop field that path touches.
    """

    def __init__(self, trainer: Any) -> None:
        try:
            version = importlib.metadata.version("trl")
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        if version != "1.10.0":
            raise TrainerUnsupported(version)
        if bool(getattr(trainer, "fused_loss", False)) or bool(
            getattr(getattr(trainer, "args", None), "use_liger_kernel", False)
        ):
            raise ValueError("SAME_START_REFUSAL: divergent fused loss")
        if (
            getattr(trainer, "optimizer", None) is None
            or getattr(trainer, "lr_scheduler", None) is None
        ):
            raise ValueError(
                "TRAINER_STATE_INCOMPLETE: optimizer and scheduler must be initialized"
            )
        self.trainer = trainer
        self._fused_loss = False
        self._materialized_digest: str | None = None
        self._materialized_position = 0
        self._active_intervention: Intervention | None = None
        self._reward_witness: list[Mapping[str, Any]] = []
        self._score_mode: str | None = None
        self._frozen_base_scores: tuple[tuple[float, ...], ...] = ()
        self._replay_score_cursor = 0
        self._source_scorer_count = 0
        self._generation_count = 0
        # ``Trainer.train`` initializes this loop value before its first
        # training step. The explicit one-step route performs that same setup.
        self.trainer.current_gradient_accumulation_steps = int(
            getattr(self.trainer.args, "gradient_accumulation_steps", 1)
        )
        self._original_reward_funcs = tuple(trainer.reward_funcs)
        trainer.reward_funcs = [self._intervening_reward(fn) for fn in self._original_reward_funcs]

    def _intervening_reward(self, reward: Callable[..., Any]) -> Callable[..., Any]:
        def controlled_reward(*args: Any, **kwargs: Any) -> Any:
            if self._score_mode == "freeze":
                values = reward(*args, **kwargs)
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    raise TypeError("controlled GRPO reward must return a sequence")
                source_values = tuple(float(value) for value in values)
                self._source_scorer_count += 1
                self._frozen_base_scores = (*self._frozen_base_scores, source_values)
                self._reward_witness.append(
                    {
                        "intervention": Intervention("pre_intervention", (0.0,)),
                        "source_values": source_values,
                        "returned_values": source_values,
                    }
                )
                return list(source_values)
            if self._score_mode != "replay" or self._active_intervention is None:
                raise RuntimeError("controlled GRPO reward ran outside scored replay")
            if self._replay_score_cursor >= len(self._frozen_base_scores):
                raise ValueError(
                    "TRAINER_STATE_INCOMPLETE: replay called more reward functions than freeze"
                )
            source_values = self._frozen_base_scores[self._replay_score_cursor]
            self._replay_score_cursor += 1
            adjusted = [
                value
                + self._active_intervention.reward_offsets[
                    index % len(self._active_intervention.reward_offsets)
                ]
                for index, value in enumerate(source_values)
            ]
            self._reward_witness.append(
                {
                    "intervention": self._active_intervention,
                    "source_values": source_values,
                    "returned_values": tuple(adjusted),
                }
            )
            return adjusted

        return controlled_reward

    @property
    def adapter_weights(self) -> dict[str, Any]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.trainer.model.named_parameters()
        }

    @adapter_weights.setter
    def adapter_weights(self, snapshot: Mapping[str, Any]) -> None:
        current = dict(self.trainer.model.named_parameters())
        if set(snapshot) != set(current):
            raise ValueError("TRAINER_STATE_INCOMPLETE: model parameter names changed")
        for name, value in snapshot.items():
            parameter = current[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError("SAME_START_REFUSAL: divergent parameter shape")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    @property
    def adam_moments(self) -> Mapping[str, Any]:
        return self.trainer.optimizer.state_dict()

    @adam_moments.setter
    def adam_moments(self, snapshot: Mapping[str, Any]) -> None:
        self.trainer.optimizer.load_state_dict(snapshot)

    @property
    def scheduler_counter(self) -> Mapping[str, Any]:
        return self.trainer.lr_scheduler.state_dict()

    @scheduler_counter.setter
    def scheduler_counter(self, snapshot: Mapping[str, Any]) -> None:
        self.trainer.lr_scheduler.load_state_dict(snapshot)

    @property
    def rng_states(self) -> Mapping[str, Any]:
        import numpy as np
        import torch

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }

    @rng_states.setter
    def rng_states(self, snapshot: Mapping[str, Any]) -> None:
        import numpy as np
        import torch

        random.setstate(snapshot["python"])
        np.random.set_state(snapshot["numpy"])
        torch.set_rng_state(snapshot["torch_cpu"])

    @property
    def dataloader_position(self) -> Mapping[str, Any]:
        """The actual cursor that supplies the explicit GRPO generation rows."""
        return {"batch_digest": self._materialized_digest, "position": self._materialized_position}

    @dataloader_position.setter
    def dataloader_position(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping) or set(value) != {"batch_digest", "position"}:
            raise ValueError("TRAINER_STATE_INCOMPLETE: materialized cursor state is malformed")
        self._materialized_digest = value["batch_digest"]
        self._materialized_position = int(value["position"])

    @property
    def buffered_inputs(self) -> Mapping[str, Any]:
        return self.trainer._buffered_inputs

    @buffered_inputs.setter
    def buffered_inputs(self, value: Mapping[str, Any]) -> None:
        self.trainer._buffered_inputs = value

    @property
    def trl_step(self) -> Mapping[str, Any]:
        """Every private loop field the direct single-update path mutates."""
        return {
            "step": self.trainer._step,
            "current_train_step_time": self.trainer._current_train_step_time,
            "metrics": copy.deepcopy(self.trainer._metrics),
            "logs": copy.deepcopy(self.trainer._logs),
            "pending_extra_logs": copy.deepcopy(self.trainer._pending_extra_logs),
            "pending_metrics": copy.deepcopy(self.trainer._pending_metrics),
        }

    @trl_step.setter
    def trl_step(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping) or set(value) != {
            "step",
            "current_train_step_time",
            "metrics",
            "logs",
            "pending_extra_logs",
            "pending_metrics",
        }:
            raise ValueError("TRAINER_STATE_INCOMPLETE: GRPO loop state is malformed")
        self.trainer._step = int(value["step"])
        self.trainer._current_train_step_time = float(value["current_train_step_time"])
        self.trainer._metrics = copy.deepcopy(value["metrics"])
        self.trainer._logs = copy.deepcopy(value["logs"])
        self.trainer._pending_extra_logs = copy.deepcopy(value["pending_extra_logs"])
        self.trainer._pending_metrics = copy.deepcopy(value["pending_metrics"])

    @property
    def state(self) -> Any:
        return self

    @property
    def global_step(self) -> Mapping[str, Any]:
        """Full TrainerState, rather than the earlier global-step-only proxy."""
        return copy.deepcopy(vars(self.trainer.state))

    @global_step.setter
    def global_step(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise ValueError("TRAINER_STATE_INCOMPLETE: TrainerState snapshot is malformed")
        current = vars(self.trainer.state)
        if set(value) != set(current):
            raise ValueError("TRAINER_STATE_INCOMPLETE: TrainerState fields changed")
        current.update(copy.deepcopy(value))

    @property
    def learning_rate(self) -> float:
        return float(self.trainer.optimizer.param_groups[0]["lr"])

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.trainer.optimizer.param_groups[0]["lr"] = float(value)

    @property
    def fused_loss(self) -> bool:
        return self._fused_loss

    @fused_loss.setter
    def fused_loss(self, value: bool) -> None:
        if value:
            raise ValueError("SAME_START_REFUSAL: divergent fused loss")
        self._fused_loss = False

    @property
    def generation_count(self) -> int:
        return self._generation_count

    @property
    def source_scorer_count(self) -> int:
        """Calls to the source scorer, which must occur only during freeze."""
        return self._source_scorer_count

    def private_profile_digest(self) -> str:
        """Digest the exact private pinned-path fields this adapter permits to mutate."""
        return _digest(
            {name: getattr(self.trainer, name) for name in sorted(PINNED_TRAINER_MUTABLE_FIELDS)}
        )

    def _changed_private_fields(self, before: Mapping[str, str]) -> tuple[str, ...]:
        after = {
            name: _digest({"value": getattr(self.trainer, name)})
            for name in vars(self.trainer)
            if name.startswith("_")
            and not name.startswith("__")
            and _can_digest(getattr(self.trainer, name))
        }
        changed = tuple(
            sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        )
        unexpected = tuple(name for name in changed if name not in PINNED_TRAINER_MUTABLE_FIELDS)
        if unexpected:
            raise ValueError(
                f"TRAINER_STATE_INCOMPLETE: unexpected pinned-path mutation {unexpected!r}"
            )
        return changed

    def _private_inventory(self) -> dict[str, str]:
        return {
            name: _digest({"value": getattr(self.trainer, name)})
            for name in vars(self.trainer)
            if name.startswith("_")
            and not name.startswith("__")
            and _can_digest(getattr(self.trainer, name))
        }

    def freeze_scored_replay(self, batch: MaterializedBatch) -> ScoredReplayBundle:
        """Generate exactly once, preserving the complete pre-intervention replay payload."""
        if not isinstance(batch, MaterializedBatch):
            raise TypeError("scored replay requires a MaterializedBatch")
        original_generate = self.trainer._generate
        had_generate_override = "_generate" in vars(self.trainer)
        captured: Any = None
        private_before = self._private_inventory()

        def capture_generation(prompts: Any) -> Any:
            nonlocal captured
            self._generation_count += 1
            if self._generation_count != 1:
                raise ValueError("SCORED_REPLAY_GENERATION_REPEATED")
            captured = original_generate(prompts)
            return captured

        self._reward_witness = []
        self._frozen_base_scores = ()
        self._replay_score_cursor = 0
        self._score_mode = "freeze"
        self.trainer._generate = capture_generation
        try:
            self.trainer._generate_and_score_completions([dict(row) for row in batch.rows])
        finally:
            if had_generate_override:
                self.trainer._generate = original_generate
            else:
                del self.trainer._generate
            self._score_mode = None
        self._changed_private_fields(private_before)
        if captured is None:
            raise ValueError(
                "TRAINER_STATE_INCOMPLETE: pinned GRPO generation returned no replay payload"
            )
        if not self._frozen_base_scores:
            raise ValueError(
                "TRAINER_STATE_INCOMPLETE: pinned GRPO freeze returned no base scorer outputs"
            )
        return ScoredReplayBundle(
            batch=batch,
            generation_payload=copy.deepcopy(captured),
            base_scores=copy.deepcopy(self._frozen_base_scores),
            base_reward_witness=tuple(copy.deepcopy(self._reward_witness)),
        )

    def run_one_update(
        self, *, replay: ScoredReplayBundle, intervention: Intervention
    ) -> Mapping[str, Any]:
        """Score one frozen generation under an intervention and take one real GRPO step."""
        if not isinstance(replay, ScoredReplayBundle) or not isinstance(intervention, Intervention):
            raise TypeError("controlled update requires ScoredReplayBundle and Intervention")
        if (
            self._materialized_digest not in (None, replay.batch.digest)
            or self._materialized_position != 0
        ):
            raise ValueError("SAME_START_REFUSAL: divergent materialized iterator state")
        self._materialized_digest = replay.batch.digest
        self._materialized_position = len(replay.batch.rows)
        self._active_intervention = intervention
        self._reward_witness = []
        self._frozen_base_scores = tuple(copy.deepcopy(replay.base_scores))
        self._replay_score_cursor = 0
        self._score_mode = "replay"
        original_generate = self.trainer._generate
        had_generate_override = "_generate" in vars(self.trainer)
        private_before = self._private_inventory()

        def replay_generation(prompts: Any) -> Any:
            return copy.deepcopy(replay.generation_payload)

        self.trainer._generate = replay_generation
        try:
            inputs = self.trainer._generate_and_score_completions(
                [dict(row) for row in replay.batch.rows]
            )
            if self._replay_score_cursor != len(self._frozen_base_scores):
                raise ValueError(
                    "TRAINER_STATE_INCOMPLETE: replay did not consume every frozen base score"
                )
            # ``training_step`` ordinarily calls ``_prepare_inputs`` to draw from
            # its private dataloader. The explicit generation above is already
            # the typed, materialized batch, so replace that one call with an
            # identity bridge while retaining GRPO's real loss, backward, and
            # private-step bookkeeping.
            original_prepare = self.trainer._prepare_inputs
            had_prepare_override = "_prepare_inputs" in vars(self.trainer)
            self.trainer._prepare_inputs = lambda prepared: prepared
            try:
                loss = self.trainer.training_step(
                    self.trainer.model, inputs, num_items_in_batch=None
                )
            finally:
                if had_prepare_override:
                    self.trainer._prepare_inputs = original_prepare
                else:
                    del self.trainer._prepare_inputs
            gradient_norm = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in self.trainer.model.parameters()
                if parameter.grad is not None
            )
            self.trainer.optimizer.step()
            self.trainer.lr_scheduler.step()
            optimizer_update_norm = 0.0
            for state in self.trainer.optimizer.state.values():
                moment = state.get("exp_avg") if isinstance(state, Mapping) else None
                if moment is not None:
                    optimizer_update_norm += float(moment.detach().abs().sum().cpu())
            self.trainer.optimizer.zero_grad(set_to_none=True)
            self.trainer.model.zero_grad(set_to_none=True)
            self.trainer.state.global_step += 1
            if had_generate_override:
                self.trainer._generate = original_generate
            else:
                del self.trainer._generate
            changed_private = self._changed_private_fields(private_before)
            return {
                "batch_digest": replay.batch.digest,
                "batch_bytes": replay.batch.bytes,
                "scored_replay_digest": replay.digest,
                "loss_input_digest": _digest(
                    {name: value for name, value in inputs.items() if name != "advantages"}
                ),
                "intervention": intervention,
                "reward_witness": tuple(self._reward_witness),
                "loss": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm,
                "optimizer_update_norm": optimizer_update_norm,
                "advantages": tuple(float(value) for value in inputs["advantages"].detach().cpu()),
                "changed_private_fields": changed_private,
            }
        finally:
            if had_generate_override:
                self.trainer._generate = original_generate
            elif "_generate" in vars(self.trainer):
                del self.trainer._generate
            self._active_intervention = None
            self._score_mode = None


def capture_state(trainer: Any, profile: TrainerProfile) -> TrainerState:
    """Copy the declared full mutable state and its same-start fingerprints."""
    missing = tuple(name for name in EIGHT_OBJECTS if name not in profile.mutable_objects)
    if missing:
        raise TrainerStateIncomplete(missing)
    absent: list[str] = []
    values: dict[str, Any] = {}
    for name in EIGHT_OBJECTS:
        try:
            values[name] = copy.deepcopy(_read_object(trainer, name))
        except AttributeError:
            absent.append(name)
    if absent:
        raise TrainerStateIncomplete(tuple(absent))
    try:
        return TrainerState(
            values=values,
            learning_rate=_learning_rate(trainer),
            parameter_shape=_parameter_shape(values["adapter_weights"]),
            optimizer_step_count=_optimizer_step_count(values["adam_moments"]),
            fused_loss=_fused_loss(trainer),
        )
    except AttributeError as exc:
        raise TrainerStateIncomplete((str(exc),)) from exc


def restore_state(trainer: Any, state: TrainerState, profile: TrainerProfile) -> None:
    """Restore all eight objects and the scalar state that makes a replay comparable."""
    missing = tuple(name for name in EIGHT_OBJECTS if name not in state.values)
    if missing:
        raise TrainerStateIncomplete(missing)
    for name in EIGHT_OBJECTS:
        _write_object(trainer, name, copy.deepcopy(state.values[name]))
    if hasattr(trainer, "learning_rate"):
        trainer.learning_rate = state.learning_rate
    else:
        trainer.optimizer.param_groups[0]["lr"] = state.learning_rate
    trainer.fused_loss = state.fused_loss
    after = capture_state(trainer, profile)
    same_start(after, state).require()


def same_start(left: TrainerState, right: TrainerState) -> SameStart:
    """Compare the five start invariants in the frozen refusal order."""
    if left.learning_rate != right.learning_rate:
        return SameStart(False, "learning rate")
    if left.parameter_shape != right.parameter_shape:
        return SameStart(False, "parameter shape")
    if left.optimizer_step_count != right.optimizer_step_count:
        return SameStart(False, "optimizer step count")
    if left.fused_loss != right.fused_loss:
        return SameStart(False, "fused loss")
    if left.digest() != right.digest():
        return SameStart(False, "digest")
    return SameStart(True)


def run_swap(
    batch: Any,
    candidates: Sequence[SwapArm],
    profile: TrainerProfile,
    *,
    trainer: Any,
    null: SwapArm,
) -> ControlledUpdateResult:
    """Run one null replay then each supplied candidate exactly once from one start."""
    if null.name != "null":
        raise ValueError("the null arm must be named 'null'")
    arms = (null, *tuple(candidates))
    names = tuple(arm.name for arm in arms)
    if len(names) != len(set(names)):
        raise ValueError("every controlled arm must have a unique name")
    if "placebo" not in names:
        raise ValueError("placebo must be supplied as a first-class arm")
    baseline = capture_state(trainer, profile)
    supplied_batch = batch
    if isinstance(trainer, TRL110StateAdapter) and isinstance(batch, MaterializedBatch):
        supplied_batch = trainer.freeze_scored_replay(batch)
        restore_state(trainer, baseline, profile)
        same_start(capture_state(trainer, profile), baseline).require()
    results: list[tuple[str, Any]] = []
    for arm in arms:
        restore_state(trainer, baseline, profile)
        same_start(capture_state(trainer, profile), baseline).require()
        value = arm.apply(trainer, supplied_batch)
        during = capture_state(trainer, profile)
        if during.digest() == baseline.digest():
            raise ValueError(f"CONTROLLED_UPDATE_VACUOUS: {arm.name} changed no controlled state")
        changed_objects = tuple(
            name
            for name in EIGHT_OBJECTS
            if _digest({"value": during.values[name]}) != _digest({"value": baseline.values[name]})
        )
        required_update_evidence = {
            "adam_moments",
            "scheduler_counter",
            "rng_states",
            "dataloader_position",
            "trl_step",
            "state_global_step",
        }
        missing = tuple(sorted(required_update_evidence - set(changed_objects)))
        if missing:
            raise ValueError(f"CONTROLLED_UPDATE_INCOMPLETE: {arm.name} left {missing!r} unchanged")
        restore_state(trainer, baseline, profile)
        same_start(capture_state(trainer, profile), baseline).require()
        results.append(
            (
                arm.name,
                {
                    "value": value,
                    "during_digest": during.digest(),
                    "restored_digest": baseline.digest(),
                    "changed_objects": changed_objects,
                },
            )
        )
    final = capture_state(trainer, profile)
    return ControlledUpdateResult(
        scope="controlled_update_response",
        statement=(
            "This CPU toy run establishes mechanism conformance for a controlled update response; "
            "it is not a behavioural result."
        ),
        start_digest=baseline.digest(),
        final_digest=final.digest(),
        arm_order=names,
        arms=tuple(results),
        profile=profile,
    )
