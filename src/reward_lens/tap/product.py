"""Product-facing, value-preserving wrappers for trainer reward callables.

The generic tap already owns the difficult hot-path rules: it preserves the original return
object, re-raises a grader exception unchanged, preserves coroutine-ness, and retains a bounded
call record.  This module binds that seam to a small run-record context without importing or
patching TRL or ``verifiers``.  The context receives one event per call; a record writer can turn
those events into its framework-specific run hierarchy after the trainer reaches a step boundary.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reward_lens.tap.contract import (
    DEFAULT,
    CallOutcome,
    SimpleRun,
    TapBudget,
    classify_return,
    is_sentinel,
)
from reward_lens.tap.grader_wrap import instrument_grader

__all__ = [
    "TAP_VALUE_CHANGED",
    "TRAINER_UNSUPPORTED",
    "TappedReward",
    "TappedRubric",
    "ProductTapEffect",
    "assert_value_unchanged",
    "tap_reward",
    "tap_rubric",
    "verify_cpu_conformance_receipt",
    "write_cpu_conformance_receipt",
]


class TRAINER_UNSUPPORTED(ValueError):
    """The declared TRL version is outside the product tap's supported reader range."""


class TAP_VALUE_CHANGED(AssertionError):
    """A conformance witness observed a wrapper returning a different object."""


@runtime_checkable
class TapRunContext(Protocol):
    """The narrow run-record seam used by both framework wrappers.

    The context owns persistence.  The tap does not write a file, patch a trainer, or choose a
    step boundary.  It only calls this append method after the original callable has returned or
    raised, so an ordinary caller can supply a small in-memory context during integration tests.
    """

    def append_tap_event(self, event: Mapping[str, Any]) -> None: ...


@dataclass
class ProductTapEffect:
    """Measured product-layer recording work, including event creation and sink append.

    ``instrument_grader.effect()`` deliberately measures its own hot path.  The product layer
    does additional work after that seam: it projects the run-record event and calls its declared
    sink.  Those costs are recorded separately so a slow sink cannot disappear from an overhead
    report.
    """

    calls: int = 0
    added_ns_total: int = 0
    added_ns_max: int = 0
    _samples: list[int] = field(default_factory=list, repr=False)

    def observe(self, elapsed_ns: int) -> None:
        self.calls += 1
        self.added_ns_total += elapsed_ns
        self.added_ns_max = max(self.added_ns_max, elapsed_ns)
        if len(self._samples) < 1024:
            self._samples.append(elapsed_ns)

    def as_record(self) -> dict[str, Any]:
        samples = sorted(self._samples)
        p50 = samples[(len(samples) - 1) // 2] if samples else 0
        p99 = samples[min(len(samples) - 1, (99 * len(samples) + 99) // 100)] if samples else 0
        return {
            "calls": self.calls,
            "added_ns_total": self.added_ns_total,
            "added_ns_p50": p50,
            "added_ns_p99": p99,
            "added_ns_max": self.added_ns_max,
        }


@runtime_checkable
class _TappedBase(Protocol):
    """Structural callable result shared by the frozen public wrapper interfaces."""

    adapter: str
    callable: Callable[..., Any]
    run_context: TapRunContext
    instrumented: Callable[..., Any]
    product_effect: Callable[[], ProductTapEffect]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TappedReward(_TappedBase, Protocol):
    """The callable TRL reward result of :func:`tap_reward`."""


@runtime_checkable
class TappedRubric(_TappedBase, Protocol):
    """The callable verifiers rubric result of :func:`tap_rubric`."""


def _context_field(context: object, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _require_context(context: object) -> TapRunContext:
    append = getattr(context, "append_tap_event", None)
    if not callable(append):
        raise TypeError(
            "run_context must provide append_tap_event(event) so the tap can append run-record fields"
        )
    return context  # type: ignore[return-value]


def _check_trl_version(context: object) -> None:
    """Accept the tap/trace reader range frozen by D-48, or abstain when no version was declared."""
    version = _context_field(context, "trainer_version", _context_field(context, "trl_version"))
    if version is None:
        return
    parts = str(version).split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        raise TRAINER_UNSUPPORTED(
            f"TRAINER_UNSUPPORTED trainer_version={version!r}; expected trl>=1.9,<2.0"
        ) from None
    if major != 1 or minor < 9:
        raise TRAINER_UNSUPPORTED(
            f"TRAINER_UNSUPPORTED trainer_version={version!r}; expected trl>=1.9,<2.0"
        )


def _effect_fields(effect: Any) -> dict[str, Any]:
    """A stable scalar projection of the existing Plane-A effect object."""
    return {
        "invocations": effect.calls,
        "added_ns_total": effect.added_ns_total,
        "added_ns_p50": effect.added_ns_p50,
        "added_ns_p99": effect.added_ns_p99,
        "resident_bytes": effect.resident_bytes,
        "recorder_exceptions": effect.recorder_exceptions,
        "enabled": effect.enabled,
        "breaches": tuple(b.__canonical__() for b in effect.breaches),
        "unchecked": effect.unchecked,
    }


def _event(
    *,
    adapter: str,
    context: object,
    value: Any,
    outcome: CallOutcome,
    instrumented: Callable[..., Any],
    exception: BaseException | None = None,
) -> dict[str, Any]:
    effect = instrumented.effect()  # type: ignore[attr-defined]
    return {
        "adapter": adapter,
        "run_id": effect.run_id,
        "step": _context_field(context, "step"),
        "outcome": outcome.value,
        # Never give the arbitrary sink a mutable trainer return object. `instrument_grader` owns
        # its bounded private reference; a later record writer can drain that after the trainer
        # boundary. The product event carries only a safe, constant-time shape projection.
        "value_shape": dict(classify_return(value)._asdict()) if exception is None else None,
        "exception_type": None if exception is None else type(exception).__name__,
        "exception_message": None if exception is None else str(exception).split("\n", 1)[0],
        "effect": _effect_fields(effect),
    }


def _tap(
    fn: Callable[..., Any],
    *,
    adapter: str,
    run_context: object,
    budget: TapBudget,
) -> Callable[..., Any]:
    context = _require_context(run_context)
    handle = SimpleRun(
        run_id=str(_context_field(context, "run_id", adapter)),
        step=_context_field(context, "step"),
    )
    instrumented = instrument_grader(
        fn, run=handle, budget=budget, name=getattr(fn, "__name__", adapter)
    )
    product_effect = ProductTapEffect()

    def append(value: Any, outcome: CallOutcome, exception: BaseException | None = None) -> None:
        # `instrument_grader` has already protected its own recorder. Context persistence happens
        # after the host callable and must not reconstruct or substitute its result.
        started_ns = time.perf_counter_ns()
        try:
            context.append_tap_event(
                _event(
                    adapter=adapter,
                    context=context,
                    value=value,
                    outcome=outcome,
                    instrumented=instrumented,
                    exception=exception,
                )
            )
        except Exception as recorder_exc:
            # This append is part of Plane A too. A record sink that is full or malformed must not
            # replace a trainer reward or an exception from the wrapped grader.
            instrumented.guard.note_recorder_exception(recorder_exc)  # type: ignore[attr-defined]
        finally:
            product_effect.observe(time.perf_counter_ns() - started_ns)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            handle.step = _context_field(context, "step")
            try:
                value = await instrumented(*args, **kwargs)
            except BaseException as exc:
                append(None, CallOutcome.RAISED, exc)
                raise
            outcome = (
                CallOutcome.SENTINEL
                if is_sentinel(value, classify_return(value))
                else CallOutcome.RETURNED
            )
            append(value, outcome)
            return value

        wrapper: Callable[..., Any] = async_wrapper
    else:

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            handle.step = _context_field(context, "step")
            try:
                value = instrumented(*args, **kwargs)
            except BaseException as exc:
                append(None, CallOutcome.RAISED, exc)
                raise
            outcome = (
                CallOutcome.SENTINEL
                if is_sentinel(value, classify_return(value))
                else CallOutcome.RETURNED
            )
            append(value, outcome)
            return value

        wrapper = sync_wrapper

    wrapper.instrumented = instrumented  # type: ignore[attr-defined]
    wrapper.adapter = adapter  # type: ignore[attr-defined]
    wrapper.callable = wrapper  # type: ignore[attr-defined]
    wrapper.run_context = context  # type: ignore[attr-defined]
    wrapper.product_effect = lambda: product_effect  # type: ignore[attr-defined]
    return wrapper


def tap_reward(
    fn: Callable[..., Any], *, run_context: TapRunContext, budget: TapBudget = DEFAULT
) -> TappedReward:
    """Wrap one callable TRL reward function without changing its call or return semantics."""
    _check_trl_version(run_context)
    return _tap(
        fn,
        adapter="trl_reward",
        run_context=run_context,
        budget=budget,
    )  # type: ignore[return-value]


def tap_rubric(
    rubric: Callable[..., Any], *, run_context: TapRunContext, budget: TapBudget = DEFAULT
) -> TappedRubric:
    """Wrap one verifiers rubric upstream of its own zero-substitution boundary."""
    return _tap(
        rubric,
        adapter="verifiers_rubric",
        run_context=run_context,
        budget=budget,
    )  # type: ignore[return-value]


def assert_value_unchanged(original: Any, observed: Any, *, wrapper: str, fixture: str) -> None:
    """Raise the named refusal when a conformance fixture sees a replacement object."""
    if observed is not original:
        raise TAP_VALUE_CHANGED(
            f"TAP_VALUE_CHANGED wrapper={wrapper} fixture={fixture} "
            f"original_type={type(original).__name__} observed_type={type(observed).__name__}"
        )


_RECEIPT_SCHEMA = "reward-lens/tap-cpu-conformance/1"
_MEASUREMENT_FIELDS = (
    "product_calls",
    "product_added_ns_total",
    "product_added_ns_p50",
    "product_added_ns_p99",
    "product_added_ns_max",
    "inner_added_ns_total",
    "wall_bare_median_ns",
    "wall_tapped_median_ns",
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _receipt_error(field: str) -> ValueError:
    return ValueError(f"TAP_RECEIPT_TAMPERED field={field}")


def _validate_receipt_body(body: Mapping[str, Any], *, fixture_id: str | None = None) -> None:
    if body.get("schema") != _RECEIPT_SCHEMA:
        raise _receipt_error("schema")
    fixture = body.get("fixture")
    if not isinstance(fixture, Mapping) or not isinstance(fixture.get("id"), str):
        raise _receipt_error("fixture.id")
    if fixture_id is not None and fixture["id"] != fixture_id:
        raise _receipt_error("fixture.id")
    expected_fixture_hash = hashlib.sha256(fixture["id"].encode()).hexdigest()
    if fixture.get("sha256") != expected_fixture_hash:
        raise _receipt_error("fixture.sha256")
    runtime = body.get("runtime")
    if not isinstance(runtime, Mapping) or any(
        not runtime.get(k) for k in ("trl", "torch", "transformers")
    ):
        raise _receipt_error("runtime")
    measurement = body.get("measurement")
    if not isinstance(measurement, Mapping):
        raise _receipt_error("measurement")
    for name in _MEASUREMENT_FIELDS:
        value = measurement.get(name)
        if not isinstance(value, (int, float)) or value < 0:
            raise _receipt_error(f"measurement.{name}")
    if measurement["product_calls"] < 1 or measurement["product_added_ns_total"] < 1:
        raise _receipt_error("measurement.product_effect")
    constraints = body.get("constraints")
    if (
        not isinstance(constraints, Mapping)
        or constraints.get("device") != "cpu"
        or constraints.get("threads") != 1
    ):
        raise _receipt_error("constraints")


def write_cpu_conformance_receipt(
    path: Path,
    *,
    fixture_id: str,
    runtime: Mapping[str, str],
    measurement: Mapping[str, int | float],
    constraints: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate a self-digesting receipt for one local CPU tap-conformance fixture."""
    body: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "fixture": {"id": fixture_id, "sha256": hashlib.sha256(fixture_id.encode()).hexdigest()},
        "runtime": dict(runtime),
        "measurement": dict(measurement),
        "constraints": dict(constraints),
    }
    _validate_receipt_body(body, fixture_id=fixture_id)
    receipt = {**body, "sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest()}
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def verify_cpu_conformance_receipt(path: Path, *, fixture_id: str) -> dict[str, Any]:
    """Read a generated receipt and reject missing, altered, or wrong-fixture fields."""
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _receipt_error("document") from exc
    if not isinstance(receipt, dict):
        raise _receipt_error("document")
    digest = receipt.pop("sha256", None)
    _validate_receipt_body(receipt, fixture_id=fixture_id)
    if digest != hashlib.sha256(_canonical_bytes(receipt)).hexdigest():
        raise _receipt_error("sha256")
    receipt["sha256"] = digest
    return receipt
