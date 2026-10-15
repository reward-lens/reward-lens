"""Optimizer-step projection clamp and bounded CPU short-fork conformance."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Any

from reward_lens.learning.contracts import TrainerProfile
from reward_lens.learning.intervention.swap import (
    Intervention,
    MaterializedBatch,
    TrainerState,
    TRL110StateAdapter,
    capture_state,
    restore_state,
    same_start,
)

__all__ = [
    "ClampCallback",
    "ClampFailed",
    "ClampReading",
    "ClampSpecification",
    "PinnedShortForkExecutor",
    "ShortForkArm",
    "ShortForkResult",
    "build_clamp_callback",
    "fixture_digest",
    "run_short_fork",
]


def fixture_digest(root: Path | str) -> str:
    """Hash the complete named fixture, including relative filenames."""
    fixture_root = Path(root)
    if not fixture_root.is_dir():
        raise ValueError(f"CPU fixture root does not exist: {fixture_root}")
    digest = hashlib.sha256()
    for path in sorted(item for item in fixture_root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(fixture_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ClampSpecification:
    """The declared coordinate and numerical limits of one optimizer-step clamp."""

    target_name: str
    target: Callable[[Any], Any]
    target_value: float
    tolerance: float
    max_correction: float
    max_iterations: int

    def __post_init__(self) -> None:
        if not self.target_name or not callable(self.target):
            raise TypeError("clamp requires a named callable target measure")
        if not math.isfinite(self.target_value):
            raise ValueError("clamp target_value must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("clamp tolerance must be positive and finite")
        if not math.isfinite(self.max_correction) or self.max_correction <= 0:
            raise ValueError("clamp max_correction must be positive and finite")
        if self.max_iterations < 1:
            raise ValueError("clamp max_iterations must be at least one")


@dataclass(frozen=True)
class ClampReading:
    """A measurement made after an actual optimizer step and its correction."""

    step: int
    before: float
    after: float
    target: float
    tolerance: float
    iterations: int
    correction_norm: float
    method: str = "optimizer_step_projection"

    @property
    def held(self) -> bool:
        return abs(self.after - self.target) <= self.tolerance


class ClampFailed(RuntimeError):
    """The declared coordinate could not be returned to its declared tolerance."""

    def __init__(self, reading: ClampReading) -> None:
        self.reading = reading
        super().__init__(
            "CLAMP_NOT_HELD: optimizer step "
            f"{reading.step} left target at {reading.after:.8g}; expected {reading.target:.8g} "
            f"within {reading.tolerance:.8g}"
        )


class ClampCallback:
    """A real PyTorch optimizer-hook clamp, not a trainer or reward wrapper."""

    method = "optimizer_step_projection"

    def __init__(self, profile: TrainerProfile, specification: ClampSpecification) -> None:
        if not isinstance(profile, TrainerProfile):
            raise TypeError("clamp requires a pinned TrainerProfile")
        if not isinstance(specification, ClampSpecification):
            raise TypeError("clamp requires a ClampSpecification")
        self.profile = profile
        self.specification = specification
        self.readings: list[ClampReading] = []
        self._model: Any | None = None
        self._step = 0
        self._handles: tuple[Any, Any] = ()

    def attach(self, optimizer: Any, model: Any) -> None:
        """Attach to optimizer pre/post-step hooks before the fork begins."""
        if self._handles:
            raise RuntimeError("clamp callback is already attached")
        if not callable(getattr(optimizer, "register_step_pre_hook", None)) or not callable(
            getattr(optimizer, "register_step_post_hook", None)
        ):
            raise TypeError("optimizer does not expose PyTorch step hooks")
        self._model = model
        self._handles = (
            optimizer.register_step_pre_hook(self._on_pre_optimizer_step),
            optimizer.register_step_post_hook(self._on_optimizer_step),
        )

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = ()
        self._model = None

    def _on_pre_optimizer_step(
        self, optimizer: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        del optimizer, args, kwargs

    def _on_optimizer_step(
        self, optimizer: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        del optimizer, args, kwargs
        if self._model is None:
            raise RuntimeError("clamp post-step hook has no model")
        self._step += 1
        self.readings.append(self.apply(self._model, step=self._step))

    def _parameters(self, model: Any) -> list[Any]:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("CLAMP_TARGET_UNREACHABLE: no trainable model parameters")
        return parameters

    def _scalar_target(self, model: Any) -> Any:
        value = self.specification.target(model)
        if getattr(value, "ndim", None) != 0:
            raise TypeError("clamp target measure must return a scalar tensor")
        return value

    def _value(self, model: Any) -> float:
        value = self._scalar_target(model)
        return float(value.detach().cpu().item())

    def apply(self, model: Any, *, step: int) -> ClampReading:
        """Project trainable parameters after a real optimizer update."""
        import torch

        parameters = self._parameters(model)
        before = self._value(model)
        current = before
        correction_norm = 0.0
        used = 0
        for used in range(1, self.specification.max_iterations + 1):
            if abs(current - self.specification.target_value) <= self.specification.tolerance:
                break
            target = self._scalar_target(model)
            gradients = torch.autograd.grad(target, parameters, allow_unused=True)
            usable = [
                torch.zeros_like(parameter) if gradient is None else gradient.detach()
                for parameter, gradient in zip(parameters, gradients)
            ]
            squared_norm = float(sum((gradient.double() ** 2).sum() for gradient in usable).item())
            if not math.isfinite(squared_norm) or squared_norm <= 0:
                raise ValueError("CLAMP_TARGET_UNREACHABLE: target gradient is zero or non-finite")
            coefficient = (self.specification.target_value - current) / squared_norm
            correction = abs(coefficient) * math.sqrt(squared_norm)
            if correction > self.specification.max_correction:
                coefficient *= self.specification.max_correction / correction
                correction = self.specification.max_correction
            with torch.no_grad():
                for parameter, gradient in zip(parameters, usable):
                    parameter.add_(gradient, alpha=coefficient)
            correction_norm += correction
            current = self._value(model)
        reading = ClampReading(
            step=step,
            before=before,
            after=current,
            target=self.specification.target_value,
            tolerance=self.specification.tolerance,
            iterations=used,
            correction_norm=correction_norm,
        )
        if not reading.held:
            raise ClampFailed(reading)
        return reading


def build_clamp_callback(
    profile: TrainerProfile, specification: ClampSpecification
) -> ClampCallback:
    """Build the rung-five callback against the pinned trainer profile."""
    return ClampCallback(profile, specification)


class PinnedShortForkExecutor:
    """Execute short-fork arms through the accepted pinned GRPO replay adapter.

    The executor is deliberately a thin composition of rung four's state and
    replay primitives. It neither wraps trainer values nor re-implements the
    adapter: every arm restores the captured eight-object state, replays one
    frozen generation, takes one real optimizer step, then restores that state.
    """

    def __init__(
        self,
        adapter: TRL110StateAdapter,
        batch: MaterializedBatch,
        profile: TrainerProfile,
        clamp: ClampCallback,
    ) -> None:
        if not isinstance(adapter, TRL110StateAdapter) or not isinstance(batch, MaterializedBatch):
            raise TypeError("pinned short fork requires a TRL110StateAdapter and MaterializedBatch")
        if clamp.profile != profile:
            raise ValueError("SAME_START_REFUSAL: divergent trainer profile")
        self.adapter = adapter
        self.profile = profile
        self.clamp = clamp
        self.baseline: TrainerState = capture_state(adapter, profile)
        self.replay = adapter.freeze_scored_replay(batch)
        restore_state(adapter, self.baseline, profile)
        same_start(capture_state(adapter, profile), self.baseline).require()
        self.witnesses: list[Mapping[str, Any]] = []

    def arm(
        self,
        name: str,
        intervention: Intervention,
        useful_task_controls: Mapping[str, Callable[[Any], float]],
    ) -> "ShortForkArm":
        """Return a state-restoring arm whose measures read the post-step model."""
        if not isinstance(intervention, Intervention):
            raise TypeError("pinned short fork arm requires an Intervention")
        if not useful_task_controls or any(
            not callable(measure) for measure in useful_task_controls.values()
        ):
            raise TypeError("pinned short fork arm requires callable useful-task controls")
        observed: dict[str, Any] = {}

        def run() -> None:
            restore_state(self.adapter, self.baseline, self.profile)
            same_start(capture_state(self.adapter, self.profile), self.baseline).require()
            before_readings = len(self.clamp.readings)
            value = self.adapter.run_one_update(replay=self.replay, intervention=intervention)
            if len(self.clamp.readings) != before_readings + 1:
                raise RuntimeError(
                    "CLAMP_OPTIMIZER_HOOK_MISSING: no clamp reading followed optimizer.step"
                )
            during = capture_state(self.adapter, self.profile)
            if during.digest() == self.baseline.digest():
                raise RuntimeError(
                    "CONTROLLED_UPDATE_VACUOUS: pinned fork arm changed no controlled state"
                )
            witness = {
                "arm": name,
                "value": value,
                "during_digest": during.digest(),
                "start_digest": self.baseline.digest(),
                "clamp_reading": self.clamp.readings[-1],
            }
            observed["witness"] = witness
            self.witnesses.append(witness)

        def target_measure() -> float:
            if "witness" not in observed:
                raise RuntimeError("short-fork target measure was read before its arm ran")
            return self.clamp._value(self.adapter.trainer.model)

        def control(measure: Callable[[Any], float]) -> Callable[[], float]:
            def read() -> float:
                if "witness" not in observed:
                    raise RuntimeError("short-fork useful-task control was read before its arm ran")
                return float(measure(self.adapter.trainer.model))

            return read

        def cleanup() -> None:
            restore_state(self.adapter, self.baseline, self.profile)
            same_start(capture_state(self.adapter, self.profile), self.baseline).require()

        return ShortForkArm(
            name,
            run,
            target_measure,
            {key: control(value) for key, value in useful_task_controls.items()},
            cleanup,
        )


@dataclass(frozen=True)
class ShortForkArm:
    """One executed fork arm and the target/useful-task measures read after it."""

    name: str
    run: Callable[[], None]
    target_measure: Callable[[], float]
    useful_task_controls: Mapping[str, Callable[[], float]]
    cleanup: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.name or not callable(self.run) or not callable(self.target_measure):
            raise TypeError("short fork arm requires a name, run function, and target measure")
        if not self.useful_task_controls or any(
            not name or not callable(measure) for name, measure in self.useful_task_controls.items()
        ):
            raise TypeError("short fork arm requires named callable useful-task controls")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("short fork cleanup must be callable when supplied")


@dataclass(frozen=True)
class ShortForkResult:
    """CPU conformance observations, explicitly limited to a short fork."""

    scope: str
    statement: str
    profile: TrainerProfile
    clamp_method: str
    target_measure_name: str
    target_values: tuple[float, ...]
    useful_task_controls: Mapping[str, tuple[float, ...]]
    null_spread_value: float
    null_spread_method: str
    fixture_digest: str
    limits: Mapping[str, int | float]
    arm_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.scope != "short_fork_behavior":
            raise ValueError("short fork scope must be short_fork_behavior")
        if "not a behavioural result" not in self.statement.lower():
            raise ValueError("short fork result must preserve its non-behavioural claim boundary")
        if self.clamp_method != "optimizer_step_projection":
            raise ValueError("short fork requires the optimizer_step_projection clamp method")
        if not self.target_measure_name or not self.fixture_digest.startswith("sha256:"):
            raise ValueError("short fork requires a named target and fixture digest")
        if self.null_spread_method != "sample_standard_deviation":
            raise ValueError("short fork null spread must be computed by sample_standard_deviation")
        if len(self.target_values) != len(self.arm_names) or len(self.arm_names) < 3:
            raise ValueError("short fork requires observations for each executed arm")
        if not self.limits or not self.useful_task_controls:
            raise ValueError("short fork requires declared limits and useful-task controls")


def run_short_fork(
    arms: Sequence[ShortForkArm],
    *,
    profile: TrainerProfile,
    clamp: ClampCallback,
    fixture_root: Path | str,
    limits: Mapping[str, int | float],
) -> ShortForkResult:
    """Execute each bounded arm, then derive controls and null spread from those observations."""
    if not isinstance(profile, TrainerProfile) or not isinstance(clamp, ClampCallback):
        raise TypeError("short fork requires a pinned profile and ClampCallback")
    if clamp.profile != profile:
        raise ValueError("SAME_START_REFUSAL: divergent trainer profile")
    selected = tuple(arms)
    if len(selected) < 3:
        raise ValueError("short fork requires at least two null arms and one comparison arm")
    controls = tuple(selected[0].useful_task_controls)
    if any(tuple(arm.useful_task_controls) != controls for arm in selected):
        raise ValueError("short fork useful-task controls must agree across arms")
    names: list[str] = []
    target_values: list[float] = []
    measured_controls = {name: [] for name in controls}
    null_values: list[float] = []
    for arm in selected:
        try:
            arm.run()
            target = float(arm.target_measure())
            names.append(arm.name)
            target_values.append(target)
            if arm.name == "null":
                null_values.append(target)
            for name in controls:
                measured_controls[name].append(float(arm.useful_task_controls[name]()))
        finally:
            if arm.cleanup is not None:
                arm.cleanup()
    if len(null_values) < 2:
        raise ValueError("short fork requires at least two executed null-arm terminal measures")
    return ShortForkResult(
        scope="short_fork_behavior",
        statement="CPU short-fork mechanism conformance, not a behavioural result.",
        profile=profile,
        clamp_method=clamp.method,
        target_measure_name=clamp.specification.target_name,
        target_values=tuple(target_values),
        useful_task_controls={name: tuple(values) for name, values in measured_controls.items()},
        null_spread_value=float(stdev(null_values)),
        null_spread_method="sample_standard_deviation",
        fixture_digest=fixture_digest(fixture_root),
        limits=dict(limits),
        arm_names=tuple(names),
    )
