"""The TRL tap: bind ``instrument_grader`` to a ``GRPOTrainer`` and come back with a record.

Verified against TRL 1.9.2 (the current release on 2026-08-01) and transformers 5.14.1. Every line
number below was re-located by content in the installed source rather than carried over from a
survey, because the survey's numbers had already drifted once.

Three seams, and the whole adapter is those three seams.

**The reward function.** ``_calculate_rewards`` calls it as
``reward_func(prompts=, completions=, completion_ids=, **reward_kwargs)`` at ``grpo_trainer.py``
1659-1661, and again identically under ``await`` at 1671-1673. Which branch a function takes is
decided by ``inspect.iscoroutinefunction(reward_func)`` at 1654, so wrapping an ``async def`` grader
in a synchronous wrapper does not just miss the async call site, it moves the grader onto the *sync*
site and hands TRL a coroutine where it iterates a list. ``wrap`` mirrors the coroutine-ness of what
it wraps for exactly that reason.

**The instrumentation channel.** ``reward_kwargs`` is built at 1618-1631 and carries
``trainer_state`` (so ``global_step`` reaches the grader), ``log_extra``, ``log_metric``, and
``environments`` when there are any. ``log_metric`` and ``log_extra`` are live bound methods of the
trainer, ``_log_metric`` at 1598 and ``_log_completion_extra`` at 1586. A reward function can push
its own scalars into TRL's metric stream and its own columns into the completions table with no
subclassing and no patching, and that is the channel this adapter uses. Line 1618 also splats every
remaining dataset column into the same dict, so the key set is open-ended and nothing here assumes a
closed one.

**The step boundary.** ``tap/`` measures and buffers and never emits, because it does not know
where a step ends. A ``TrainerCallback`` does. ``on_step_end`` drains the ring into per-step
buckets, ``on_log`` picks up the optimizer telemetry that only exists after a log, and ``finish()``
turns the buckets into ``Run -> Step -> Group -> Trajectory -> Turn``. Building the frozen record
objects happens in ``finish()``, after training, so the only thing the hot path does is append.

----

Two hazards in that channel that are not obvious from reading the call site, both found by reading
what TRL does with the values afterwards, and both of which would let a Plane A component take down
a host.

**``log_metric`` participates in a collective.** The flush at 2769-2774 iterates
``sorted(self._pending_metrics)`` and calls ``self.accelerator.gather`` once per name. TRL's own
comment there says the keys must be sorted so that all ranks call it in the same order. Sorting
fixes the *order*; it does not fix the *set*. A tap that pushes a metric only while it is enabled
has made its key set a function of its own local state, and Plane A's disable decision is
per-process by construction: one rank breaches its latency budget, stops pushing, and now that rank
enters the gather loop with three names where the others have four. That is a hang, not an
exception, and it is the worst failure this file could have. So **the emission here is never gated
on tap state.** The names are fixed at construction and every one of them is pushed on every call,
with a zero standing in for a measurement the tap could not take. ``metric_names`` exposes the set
and ``test_metric_key_set_is_invariant_to_tap_state`` asserts it.

**``log_extra`` builds a DataFrame column.** ``log()`` at 3314-3321 assembles
``{"step": ..., "prompt": ..., "completion": ..., **rewards, **extra, "advantage": ...}`` and hands
it to ``pd.DataFrame``, which raises on ragged columns. A tap that pushed a partial column, or
pushed on some batches and not others, crashes the host inside its own logging. Same fix, plus one
more: the column is emitted opt-in (``emit_extra=False``), because unlike a metric it costs a list
allocation of ``B x G`` on the hot path and it writes into an artifact the host owns.

----

One asymmetry worth stating, because it is the package. Plane A fails open: nothing in here raises
into TRL, the guard disables itself rather than misbehave, and a grader that raises has its
exception recorded and re-raised untouched. Plane B fails closed and loud. The boundary is
``finish()``: everything before it swallows, everything after it is ordinary library code that
raises when it is asked for something it does not have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from reward_lens.tap.contract import (
    DEFAULT,
    CallOutcome,
    GraderCall,
    InstrumentEffect,
    TapBudget,
)
from reward_lens.tap.grader_wrap import instrument_grader
from reward_lens.tap.ring import TapRing

if TYPE_CHECKING:  # the framework is never imported at module scope
    from reward_lens.record.schema import Run, Step

#: The metric names this adapter pushes through ``log_metric``. Fixed at import, pushed on every
#: call whatever the tap is doing, because the flush at ``grpo_trainer.py:2769-2774`` is a
#: per-name collective and a key set that varies by rank is a deadlock. See the module docstring.
METRIC_NAMES: tuple[str, ...] = (
    "reward_lens/tap_added_us",
    "reward_lens/grader_ms",
    "reward_lens/tap_enabled",
)

#: The column ``emit_extra=True`` adds to TRL's completions table. One value per completion, always,
#: for the DataFrame reason in the module docstring. It is the join key from a parquet row back to
#: this tap's own ``GraderCall.seq``.
EXTRA_COLUMN = "reward_lens_seq"

#: How many steps the in-process buffer holds before it stops taking new ones. A step of a small
#: GRPO run is a few hundred bytes of Python objects plus references to the host's prompt and
#: completion strings, so 2048 is tens of megabytes at a realistic batch size and it is a cap
#: rather than a target. Reaching it is recorded on the run's ``RecordSamplingPolicy`` as a
#: SELECTIVE sample with the reason written out, never as a full one.
DEFAULT_MAX_STEPS = 2048


# ---------------------------------------------------------------------------
# What could not be instrumented, and why
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skipped:
    """A reward function this adapter declined to wrap, with the reason it declined.

    There is exactly one reason today and it is a real one. TRL branches on
    ``isinstance(reward_func, nn.Module)`` at ``grpo_trainer.py:1638``, *before* it considers
    calling anything, and a reward model that takes that branch is invoked as
    ``reward_func(**reward_inputs).logits[:, 0]`` with tokenised text. Wrapping it in a plain
    function moves it onto the callable branch, where TRL would call it with ``prompts=``,
    ``completions=`` and ``completion_ids=`` and it would raise. So a reward *model* is passed
    through untouched and the record says so, rather than the tap silently breaking the run it was
    supposed to observe.

    This is the honest half of the coverage claim. A run whose reward is a model is a run this
    adapter records nothing about at the grader, and an analysis that does not know that would
    read the absence as an absence of calls.
    """

    name: str
    reason: str
    kind: str = ""

    def render(self) -> str:
        return f"{self.name} ({self.kind}): {self.reason}"


# ---------------------------------------------------------------------------
# The run handle
# ---------------------------------------------------------------------------


@dataclass
class TRLRunHandle:
    """The three attributes ``tap/`` asks for, with ``step`` written from ``trainer_state``.

    ``RunHandle`` is deliberately attributes and no methods, so the hot path reads rather than
    calls. ``step`` is the one that moves: the adapter sets it from
    ``reward_kwargs["trainer_state"].global_step`` immediately before each grader call, so every
    ``GraderCall`` carries the step index TRL itself was on when the reward was computed.

    That index is worth pinning down because it is not the one on the completions table. The
    grader sees ``global_step`` *before* the optimizer step increments it, so the first batch is
    step 0; ``log()`` runs after the increment and stamps the same rows ``completions_00001``.
    Recording the grader's own view keeps the tap's records aligned with the thing that produced
    them, and the difference is asserted in ``test_step_index_seen_by_grader_is_pre_increment``.
    """

    run_id: str = "trl"
    ring: TapRing = field(default_factory=TapRing)
    step: int | None = None


# ---------------------------------------------------------------------------
# The per-step bucket
# ---------------------------------------------------------------------------


@dataclass
class StepBucket:
    """Everything gathered for one optimizer step, before any of it becomes a record.

    A mutable pile on purpose. It is filled from three places that fire at different moments
    (the grader call, ``on_step_end``, ``on_log``) and it is read once, in ``finish()``, which is
    where the frozen objects get built. Freezing it earlier would mean either building a ``Step``
    three times or holding a half-built one, and both are worse than a dict.
    """

    index: int
    calls: list[GraderCall] = field(default_factory=list)
    #: Advantages as TRL computed them, read from ``trainer._logs["advantages"]`` at the boundary.
    advantages: tuple[float, ...] | None = None
    #: Per-reward-function rewards after the cross-process gather, from ``trainer._logs["rewards"]``.
    rewards: dict[str, tuple[float, ...]] = field(default_factory=dict)
    prompts: tuple[str, ...] | None = None
    completions: tuple[str, ...] | None = None
    #: Last value appended to ``trainer._metrics["train"][k]``, not a mean over the step.
    metrics: dict[str, float] = field(default_factory=dict)
    #: What ``on_pre_optimizer_step`` saw. Empty when the hook did not fire.
    grads: dict[str, Any] = field(default_factory=dict)
    #: ``InstrumentEffect`` from ``tap/``, taken at this step's boundary; cumulative by design.
    effect: InstrumentEffect | None = None
    boundary_global_step: int | None = None
    lr: float | None = None


# ---------------------------------------------------------------------------
# The tap
# ---------------------------------------------------------------------------


class TRLTap:
    """Wrap a TRL GRPO run's reward functions and collect the record they produce.

    Three calls, in this order::

        tap = TRLTap(run_id="grpo-demo")
        trainer = GRPOTrainer(model=..., reward_funcs=tap.wrap(reward_funcs), args=..., ...)
        tap.attach(trainer)
        trainer.train()
        run = tap.finish()

    ``wrap`` has to come before the trainer is built, because ``GRPOTrainer.__init__`` resolves
    ``reward_func_names`` from the callables at ``grpo_trainer.py:510-525`` and stores the list.
    ``attach`` has to come after, because the callback needs the trainer to read ``num_generations``,
    the estimator configuration and ``_logs``. Doing them in the other order is not an error this
    class can detect, so ``attach`` records what it found and ``finish`` reports it.

    Nothing between ``wrap`` and ``finish`` can raise into TRL. That is not a promise about this
    code being correct; it is enforced. The grader wrapper is ``tap.instrument_grader``, whose
    recorder segment is caught and counted, and every callback body here is inside
    ``_guarded``, which catches, counts and keeps going. The deliberately-failing-tap test drives
    the whole thing with a recorder that raises on every single call and asserts the training loop
    produces byte-identical weights.
    """

    def __init__(
        self,
        *,
        run_id: str = "trl-grpo",
        budget: TapBudget = DEFAULT,
        emit_metrics: bool = True,
        emit_extra: bool = False,
        retain_args: bool = True,
        record_grad_presence: bool = True,
        record_token_ids: bool = True,
        failure_at: float | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        ring: TapRing | None = None,
        name: str = "trl",
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1; got {max_steps}")
        self.run_id = run_id
        self.budget = budget
        self.emit_metrics = emit_metrics
        self.emit_extra = emit_extra
        self.retain_args = retain_args
        self.record_grad_presence = record_grad_presence
        #: BLK-027. `Turn.token_ids` is a declared field and this adapter left it
        #: None, so the default is True. False restores the previous behaviour
        #: exactly, which is an abstention rather than an empty tuple.
        self.record_token_ids = record_token_ids
        #: BLK-021. The score at or below which a rollout counts as a failure on the
        #: optimised objective, so `GroupStats.all_fail` is a measurement rather than
        #: False on every group of every record. None is an abstention and
        #: `from_scores` reads it as one: nothing in `GRPOConfig` says what failure is on
        #: a task, so a value invented here would turn a gap into a claim. The caller
        #: declares it, as `VerifiersAdapter.failure_at` already is declared.
        self.failure_at = failure_at
        self.max_steps = max_steps
        self.name = name

        self.handle = TRLRunHandle(
            run_id=run_id,
            ring=ring if ring is not None else TapRing.for_bytes(budget.max_resident_bytes),
        )
        self.wrapped: list[Any] = []
        self.skipped: list[Skipped] = []

        self._buckets: dict[int, StepBucket] = {}
        self._order: list[int] = []
        self._dropped_steps = 0
        #: The last step index the full buffer turned away, so a refusal counts once per step
        #: rather than once per asker. See ``_bucket``.
        self._last_refused: int | None = None
        self._trainer: Any = None
        self._attached = False
        #: Counted, never raised. A non-zero value is reported on the run and is the thing that
        #: says a record is thinner than it looks.
        self.adapter_exceptions = 0
        self.adapter_exception_keys: list[str] = []
        #: Steps whose row order could not be shown to be the generation order, so the group id
        #: and generation index were withheld rather than guessed. See ``_grouping_provenance``.
        #: BLK-009.
        self.unverified_grouping_steps = 0
        #: Why the last such step failed the check, empty while nothing has failed it.
        self.grouping_refusal_reason = ""
        self._prev_added_ns = 0
        self._prev_inner_ns = 0
        self._prev_calls = 0
        #: Monotone counter for the ``log_extra`` column. A counter rather than a read off the
        #: ring, because ``RingStats`` is O(size) and this is the hot path. See ``_after``.
        self._extra_seq = 0
        self._config: dict[str, Any] = {}
        self._started_at = ""

    # -- seam 1: the reward functions ---------------------------------------

    def wrap(self, reward_funcs: Any) -> Any:
        """Instrument every reward function TRL will call, and pass through the ones it will not.

        Accepts what ``GRPOTrainer`` accepts: one callable, or a list of them. Returns the same
        shape, so this drops into the constructor call without changing anything else.

        Three properties this has to preserve, each of which TRL reads off the object.

        ``__name__`` decides the metric keys. ``get_callable_name`` (``trainer/utils.py:224``) is
        what fills ``reward_func_names`` at ``grpo_trainer.py:524``, and those names become
        ``rewards/{name}/mean`` and the reward column of the completions table.
        ``functools.wraps`` inside ``instrument_grader`` carries it over, so wrapping does not
        rename anybody's metrics.

        Coroutine-ness decides the call site. ``instrument_grader`` produces an ``async def``
        wrapper for an ``async def`` grader, and the small adapter layer added here mirrors that,
        so an async grader stays async and keeps taking the ``await`` branch at 1671-1673.

        Being an ``nn.Module`` decides whether TRL calls it as a function at all. It is checked
        first, at 1638, and a wrapped module is no longer a module, so modules are passed through
        and recorded in ``skipped``.
        """
        single = not isinstance(reward_funcs, (list, tuple))
        funcs = [reward_funcs] if single else list(reward_funcs)
        out: list[Any] = []
        for f in funcs:
            out.append(self._wrap_one(f))
        self.wrapped = out
        return out[0] if single else out

    def _wrap_one(self, func: Any) -> Any:
        import inspect

        if isinstance(func, str) or _is_torch_module(func):
            self.skipped.append(
                Skipped(
                    name=func
                    if isinstance(func, str)
                    else getattr(func, "__name__", type(func).__name__),
                    kind="model id" if isinstance(func, str) else "reward model",
                    reason=(
                        "TRL dispatches on isinstance(reward_func, nn.Module) at "
                        "grpo_trainer.py:1638 before it considers calling anything, and it invokes "
                        "that branch with tokenised text rather than with prompts= and "
                        "completions=. A string is the same case one step earlier: it is loaded "
                        "into an AutoModelForSequenceClassification at 512-518 and becomes that "
                        "branch. Wrapping either would move it onto the callable branch and the "
                        "call would raise, so it is passed through and its calls are not recorded."
                    ),
                )
            )
            return func

        instrumented = instrument_grader(
            func,
            run=self.handle,
            budget=self.budget,
            retain_args=self.retain_args,
            facet_keys=(),
            name=getattr(func, "__name__", repr(func)),
        )
        guard = instrumented.guard  # type: ignore[attr-defined]

        if inspect.iscoroutinefunction(func):
            import functools

            @functools.wraps(func)
            async def async_adapted(*args: Any, **kwargs: Any) -> Any:
                self._before(kwargs)
                out = await instrumented(*args, **kwargs)
                self._after(kwargs, guard)
                return out

            adapted: Any = async_adapted
        else:
            import functools

            @functools.wraps(func)
            def sync_adapted(*args: Any, **kwargs: Any) -> Any:
                self._before(kwargs)
                out = instrumented(*args, **kwargs)
                self._after(kwargs, guard)
                return out

            adapted = sync_adapted

        adapted.instrumented = instrumented  # type: ignore[attr-defined]
        adapted.guard = guard  # type: ignore[attr-defined]
        adapted.ring = self.handle.ring  # type: ignore[attr-defined]
        adapted.effect = instrumented.effect  # type: ignore[attr-defined]
        return adapted

    def _before(self, kwargs: Mapping[str, Any]) -> None:
        """Put the step index on the handle so the ``GraderCall`` carries it. Two attribute reads.

        Everything here is inside a bare ``except``. It runs before the grader, on the host's
        thread, and the one behaviour it may not have is to stop the grader from running.
        """
        try:
            state = kwargs.get("trainer_state")
            if state is not None:
                self.handle.step = int(state.global_step)
        except Exception as exc:  # pragma: no cover - defensive; see the failing-tap test
            self._note(exc)

    def _after(self, kwargs: Mapping[str, Any], guard: Any) -> None:
        """Push this call's cost back into TRL's own metric stream, unconditionally.

        Unconditionally is the load-bearing word and the module docstring has the reason: the
        flush is a per-name collective and a key set that depends on local tap state deadlocks a
        multi-process run. So the same three names go in on every call, and a tap that is switched
        off contributes zeros rather than nothing.

        Everything read here is O(1) in the size of the ring and of the batch, and that is a
        constraint rather than an observation. The first version of this method took its sequence
        number from ``ring.stats().offered``, and ``stats()`` walks every record it holds to count
        retained references, so the tap's own cost grew linearly with the number of calls it had
        already recorded: 120 microseconds per call after four thousand calls against 1.6
        microseconds for the grader itself. A counter fixed it. The lesson is worth keeping, since
        every accessor on a Plane A object that looks free has to be checked rather than assumed.

        Walking the return value to count abstentions would be O(B x G) for the same reason, so
        abstention counting happens at drain time from the retained value instead.
        """
        try:
            log_metric = kwargs.get("log_metric")
            if self.emit_metrics and log_metric is not None:
                added = guard.added_ns_total
                inner = guard.inner_ns_total
                calls = guard.calls
                d_added = added - self._prev_added_ns
                d_inner = inner - self._prev_inner_ns
                moved = calls - self._prev_calls
                self._prev_added_ns = added
                self._prev_inner_ns = inner
                self._prev_calls = calls
                log_metric(METRIC_NAMES[0], d_added / 1e3)
                log_metric(METRIC_NAMES[1], d_inner / 1e6)
                log_metric(METRIC_NAMES[2], 1.0 if moved else 0.0)
            if self.emit_extra:
                log_extra = kwargs.get("log_extra")
                completions = kwargs.get("completions")
                if log_extra is not None and completions is not None:
                    self._extra_seq += 1
                    log_extra(EXTRA_COLUMN, [self._extra_seq] * len(completions))
        except Exception as exc:  # pragma: no cover - defensive; see the failing-tap test
            self._note(exc)

    # -- seam 2: the trainer -------------------------------------------------

    def attach(self, trainer: Any) -> Any:
        """Add the step-boundary callback and read the estimator configuration off the trainer.

        Reading the configuration here rather than at ``finish()`` is deliberate: ``EstimatorSpec``
        is supposed to say how scores became advantages *exactly*, and the fields it needs are all
        on ``GRPOConfig``, which is fixed for the run and cheap to read once. Doing it at the end
        would read the same object, but doing it here means a run that crashes still has it.

        Returns the trainer, so ``tap.attach(GRPOTrainer(...))`` works as one expression.
        """
        self._trainer = trainer
        try:
            trainer.add_callback(self._build_callback())
            self._attached = True
            self._config = _read_config(trainer)
            self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        except Exception as exc:
            self._note(exc)
        return trainer

    def _build_callback(self) -> Any:
        """Construct the ``TrainerCallback`` subclass, importing transformers only now."""
        TrainerCallback = _callback_base()

        outer = self

        class RewardLensCallback(TrainerCallback):
            """The step boundary. Every body is guarded; none of them can raise into the loop."""

            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                outer._guarded(outer._on_step_end, state)

            def on_pre_optimizer_step(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> None:
                if outer.record_grad_presence:
                    outer._guarded(outer._on_pre_optimizer_step, state, kwargs)

            def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                outer._guarded(outer._on_log, state, kwargs.get("logs"))

            def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                outer._guarded(outer._on_train_end, state)

        return RewardLensCallback()

    def _guarded(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run a callback body so that nothing in it reaches TRL. The Plane A contract, applied.

        ``BaseException`` is not caught, for the reason ``tap.grader_wrap`` gives: a
        ``KeyboardInterrupt`` inside somebody else's training loop is control flow and swallowing
        it would be a worse breach than any exception this could hide.
        """
        try:
            fn(*args)
        except Exception as exc:
            self._note(exc)

    def _note(self, exc: BaseException) -> None:
        """Count an adapter failure without allocating an unbounded ledger."""
        self.adapter_exceptions += 1
        key = f"{type(exc).__name__}: {str(exc).split(chr(10), 1)[0][:160]}"
        if key not in self.adapter_exception_keys and len(self.adapter_exception_keys) < 16:
            self.adapter_exception_keys.append(key)

    def _bucket(self, index: int) -> StepBucket | None:
        """The bucket for a step, or ``None`` once the buffer is full.

        The refusal counts *distinct* steps, not calls. Three places ask for a bucket per step
        (the ring drain, the boundary, and the gradient probe), so counting every miss would put a
        population of eight on a five-step run and the sampling policy's rate would be fiction.
        A single latch is enough because a training run's step indices arrive monotonically.

        A negative index is refused rather than stored. ``on_step_end`` asks for
        ``global_step - 1``, and a framework that fires it before the increment would ask for -1 on
        the first step; ``Step.__post_init__`` rejects a negative index, so keeping one here would
        turn a harmless hook-ordering difference into an exception in ``finish()``.
        """
        if index < 0:
            return None
        b = self._buckets.get(index)
        if b is not None:
            return b
        if len(self._buckets) >= self.max_steps:
            if index != self._last_refused:
                self._last_refused = index
                self._dropped_steps += 1
            return None
        b = StepBucket(index=index)
        self._buckets[index] = b
        self._order.append(index)
        return b

    def _on_step_end(self, state: Any) -> None:
        """Drain the ring into per-step buckets and pick up what only exists at a boundary.

        The bucketing key is the step index the *grader* recorded, not ``state.global_step`` here,
        which is the same number plus one. That makes the rule robust to which side of the
        increment this hook lands on and to ``steps_per_generation > 1``, where one generation
        feeds several optimizer steps: a call lands in the bucket for the step it was computed
        during, and a step that consumed a previous generation gets an empty bucket rather than a
        wrong one.
        """
        boundary = int(state.global_step)
        for call in self.handle.ring.drain():
            idx = call.step if call.step is not None else boundary
            b = self._bucket(idx)
            if b is not None:
                b.calls.append(call)

        b = self._bucket(boundary - 1)
        if b is None:
            return
        b.boundary_global_step = boundary
        trainer = self._trainer
        if trainer is None:
            return
        logs = getattr(trainer, "_logs", None)
        if isinstance(logs, dict):
            adv = logs.get("advantages")
            if adv is not None:
                b.advantages = tuple(float(x) for x in adv)
            for key, dest in (("prompt", "prompts"), ("completion", "completions")):
                seq = logs.get(key)
                if seq is not None:
                    setattr(b, dest, tuple(str(x) for x in seq))
            rewards = logs.get("rewards")
            if isinstance(rewards, dict):
                for name, values in rewards.items():
                    b.rewards[str(name)] = tuple(float(x) for x in values)
        metrics = getattr(trainer, "_metrics", None)
        if isinstance(metrics, dict):
            train = metrics.get("train") or {}
            for key, values in train.items():
                if values:
                    try:
                        b.metrics[str(key)] = float(values[-1])
                    except (TypeError, ValueError):
                        continue
        b.effect = self._first_effect()

    def _on_pre_optimizer_step(self, state: Any, kwargs: Mapping[str, Any]) -> None:
        """Record that the one moment with live per-parameter gradients exists, and that it fired.

        ``transformers/trainer.py`` calls this at 1772, between ``_clip_grad_norm`` at 1769 and
        ``optimizer.step()`` at 1773, so every parameter's ``.grad`` is populated and already
        clipped. ``callback_handler.call_event`` (``trainer_callback.py:543-556``) injects
        ``model``, ``processing_class``, ``optimizer``, ``lr_scheduler`` and both dataloaders into
        every hook, so the model arrives here without anybody holding a reference to it.

        This adapter takes nothing off those gradients. The credit measure that wants them is G1
        and it is a different package; what is recorded here is that the seam is real, that it
        fires once per optimizer step, and how many parameter tensors carried a gradient when it
        did. That is a loop over tensors, not over elements: 148 iterations for a 124M-parameter
        model, and it reads ``.grad is not None`` without touching any data.
        """
        b = self._bucket(int(state.global_step))
        if b is None:
            return
        model = kwargs.get("model")
        total = 0
        with_grad = 0
        if model is not None and hasattr(model, "parameters"):
            for p in model.parameters():
                total += 1
                if getattr(p, "grad", None) is not None:
                    with_grad += 1
        b.grads = {
            "hook_fired": True,
            "model_in_kwargs": model is not None,
            "optimizer_in_kwargs": kwargs.get("optimizer") is not None,
            "param_tensors": total,
            "param_tensors_with_grad": with_grad,
            "grads_are_clipped": True,
        }

    def _on_log(self, state: Any, logs: Mapping[str, Any] | None) -> None:
        """Pick up the fields that only exist after a log, notably the clipped gradient norm.

        ``_clip_grad_norm`` returns the norm at ``trainer.py:1769`` and
        ``on_pre_optimizer_step`` is not given it, so the only place it is readable without
        patching is the log payload. It lands under ``grad_norm`` and it is already clipped.
        """
        if not logs:
            return
        idx = int(state.global_step) - 1
        b = self._buckets.get(idx)
        if b is None:
            return
        for key, dest in (("grad_norm", "grad_norm"), ("learning_rate", "lr")):
            if key in logs:
                try:
                    value = float(logs[key])
                except (TypeError, ValueError):
                    continue
                if dest == "lr":
                    b.lr = value
                else:
                    b.metrics.setdefault(dest, value)

    def _on_train_end(self, state: Any) -> None:
        """Take whatever is still in the ring. Nothing is left behind because training stopped."""
        boundary = int(state.global_step)
        for call in self.handle.ring.drain():
            idx = call.step if call.step is not None else boundary
            b = self._bucket(idx)
            if b is not None:
                b.calls.append(call)
        last = self._buckets.get(boundary - 1)
        if last is not None and last.effect is None:
            last.effect = self._first_effect()

    def _first_effect(self) -> InstrumentEffect | None:
        """The effect from the first wrapped grader, or ``None`` if there is not one.

        First rather than all, because a ``Step`` carries one ``InstrumentEffect`` and the common
        case is one reward function. A run with several gets the first one on the step and all of
        them from ``effects()``, which is a gap the flat per-step form has rather than one this
        adapter can close on its own. It is named in the report.
        """
        for func in self.wrapped:
            effect = getattr(func, "effect", None)
            if effect is not None:
                return effect()
        return None

    # -- what it cost --------------------------------------------------------

    @property
    def instrument_effect(self) -> InstrumentEffect:
        """The cumulative instrument effect across every wrapped grader.

        One effect per wrapped callable, and this returns the first one because the common case is
        one reward function. ``effects()`` gives them all.
        """
        effects = self.effects()
        if not effects:
            return InstrumentEffect(tap_name=self.name, run_id=self.run_id, calls=0)
        return effects[0]

    def effects(self) -> tuple[InstrumentEffect, ...]:
        """One ``InstrumentEffect`` per wrapped reward function, in the order they were wrapped."""
        out = []
        for func in self.wrapped:
            effect = getattr(func, "effect", None)
            if effect is not None:
                out.append(effect())
        return tuple(out)

    def metric_names(self) -> tuple[str, ...]:
        """The metric keys this tap pushes, which do not depend on whether it is working.

        Exposed so the invariance can be asserted from outside rather than promised in a comment.
        """
        return METRIC_NAMES if self.emit_metrics else ()

    # -- seam 3: the record --------------------------------------------------

    def steps(self) -> tuple["Step", ...]:
        """Build the ``Step`` records. Off the hot path: this runs after training.

        Everything before this line swallows exceptions because it runs inside somebody else's
        loop. This line is where that stops. From here on it is ordinary library code and a
        malformed record raises, because a wrong record is worse than no record.
        """
        from reward_lens.record.schema import (
            InstrumentEffect as StepEffect,
        )
        from reward_lens.record.schema import (
            OptimizerTelemetry,
            Step,
        )

        out: list[Step] = []
        for index in sorted(self._order):
            b = self._buckets[index]
            groups = self._groups_for(b)
            effect = (
                StepEffect.from_canonical(b.effect.as_step_record())
                if b.effect is not None
                else StepEffect()
            )
            out.append(
                Step(
                    index=index,
                    groups=groups,
                    schedule=self._schedule_for(b),
                    optimizer=OptimizerTelemetry(
                        grad_norm_clipped=b.metrics.get("grad_norm"),
                        grad_norm_unclipped=None,
                        clip_fraction=b.metrics.get("clip_ratio"),
                        kl_to_ref=b.metrics.get("kl"),
                        kl_to_previous=None,
                        entropy=b.metrics.get("entropy"),
                        update_norm=None,
                        extra={
                            k: v
                            for k, v in b.metrics.items()
                            if k not in ("grad_norm", "clip_ratio", "kl", "entropy")
                        },
                    ),
                    instrument=effect,
                )
            )
        return tuple(out)

    def _schedule_for(self, b: StepBucket) -> dict[str, float]:
        """The knobs whose value at this step an estimator has to know to be reproducible."""
        cfg = self._config
        out: dict[str, float] = {}
        for key in ("beta", "temperature", "top_p", "epsilon", "epsilon_high", "delta"):
            value = cfg.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        k = cfg.get("num_generations")
        if isinstance(k, int):
            out["num_generations"] = float(k)
        if b.lr is not None:
            out["learning_rate"] = b.lr
        if b.grads:
            out["param_tensors_with_grad"] = float(b.grads.get("param_tensors_with_grad", 0))
        return out

    def _groups_for(self, b: StepBucket) -> tuple[Any, ...]:
        """Reconstruct the K-rollout groups from the flat ``B x G`` batch one call received.

        The group is recoverable because ``_calculate_rewards`` gets every row at once and TRL's
        own ``.view(-1, num_generations, ...)`` is what reconstructs it, at
        ``grpo_trainer.py:2686`` and ``:2712``. Consecutive runs of ``num_generations`` rows are
        one prompt's rollouts.

        **That assumption is true of one row order and false of another**, which is BLK-009.
        ``shuffle_sequence_dict`` at ``:1589`` permutes the generation batch, and after it row
        order carries no grouping at all. Both sources this method reads are upstream of it: the
        reward functions are called at ``:1673`` inside ``_generate_and_score_completions``, and
        ``_logs`` is extended at ``:2823-2827`` inside the same function, while the shuffle is
        applied one frame up to that function's return value. So the assumption holds today.
        ``_grouping_provenance`` is what says so on each step instead of leaving it inherited: it
        checks the one signal that survives a permutation, that the K rows of a group carry one
        prompt, and the group id and generation index are emitted only where it holds.

        Scores come from the wrapped call's own return value where there is one, so the record
        holds what the grader actually returned rather than what survived the aggregation. Where
        the tap was disabled or the ring dropped, they come from ``trainer._logs["rewards"]``,
        which is post-gather and therefore the same numbers by a different route. Where neither
        exists the score is ``None``, which is an abstention and not a zero.
        """
        from reward_lens.record.schema import (
            Group,
            GroupStats,
            group_id,
            make_trajectory,
            task_id,
            trajectory_id,
        )

        k = int(self._config.get("num_generations") or 1)
        prompts, completions = self._texts_for(b)
        n = len(completions) if completions is not None else 0
        if n == 0:
            return ()
        completion_tokens = self._completion_lengths_for(b, n)
        completion_ids = self._completion_ids_for(b, n)
        grouping_verified, refusal = self._grouping_provenance(prompts, n, k)
        if not grouping_verified:
            self.unverified_grouping_steps += 1
            self.grouping_refusal_reason = refusal
        per_func = self._scores_for(b, n)
        refs = self._call_refs(b)
        weights = self._weights_for(tuple(per_func))
        estimator = self._estimator_spec()
        std_eps = float(self._config.get("std_epsilon") or 1e-4)

        groups: list[Group] = []
        n_groups = max(1, (n + k - 1) // k)
        for g in range(n_groups):
            lo, hi = g * k, min((g + 1) * k, n)
            if lo >= hi:
                continue
            prompt_text = prompts[lo] if prompts is not None and lo < len(prompts) else ""
            task = task_id(dataset=f"{self.run_id}:step{b.index}", index=_short(prompt_text))
            gid = group_id(run=self.run_id, step=b.index, task=str(task), ordinal=g)
            totals: list[float | None] = []
            trajectories = []
            for j in range(lo, hi):
                row = {name: values[j] for name, values in per_func.items()}
                totals.append(_total(row, weights))
                trajectories.append(
                    make_trajectory(
                        id=str(trajectory_id(group=str(gid), ordinal=j - lo)),
                        task_ref=str(task),
                        turns=self._turns_for(
                            prompt_text,
                            completions[j],
                            completion_ids[j] if completion_ids is not None else None,
                        ),
                        scores=self._score_tree(row, refs, weights),
                        advantage=(
                            b.advantages[j]
                            if b.advantages is not None and j < len(b.advantages)
                            else None
                        ),
                        provenance=self._provenance_for(b, n_turns=2),
                        features=self._features_for(
                            row,
                            weights,
                            completion_text=completions[j],
                            completion_tokens=(
                                completion_tokens[j] if completion_tokens is not None else None
                            ),
                            group_ordinal=g if grouping_verified else None,
                            generation_index=(j - lo) if grouping_verified else None,
                        ),
                    )
                )
            groups.append(
                Group(
                    id=gid,
                    task_ref=task,
                    trajectories=tuple(trajectories),
                    estimator=estimator,
                    group_stats=GroupStats.from_scores(
                        totals,
                        std_epsilon=std_eps,
                        # BLK-021, one producer for the denominator. `_estimator_spec`
                        # decides it once, off `scale_rewards`, and it is already on the
                        # `EstimatorSpec` attached to this same `Group`; the statistics
                        # read that decision rather than repeating it. A second literal
                        # here is the drift this row is about: the population standard
                        # deviation sitting beside a recorded ddof of 1, which is 6.9%
                        # apart at K = 8 and 41.4% apart at K = 2.
                        std_ddof=estimator.std_ddof,
                        # BLK-021, the other half. Without a failure value `all_fail` is
                        # False on every group of every TRL-tapped record, which is what
                        # a run in which no group ever failed also looks like.
                        failure_at=self.failure_at,
                    ),
                )
            )
        return tuple(groups)

    def _weights_for(self, names: tuple[str, ...]) -> tuple[float, ...]:
        """``reward_weights``, defaulting to one per function, which is what TRL defaults to.

        ``GRPOConfig.reward_weights`` is ``None`` unless it is set, and the trainer then uses a
        vector of ones (``grpo_trainer.py:528-536``). The weights belong on the tree rather than
        folded into the leaves, because a composition whose weights are visible is one that F3 and
        the E series can sweep and ablate; a pre-multiplied leaf is a number nobody can take apart.
        """
        declared = self._config.get("reward_weights")
        if isinstance(declared, (list, tuple)) and len(declared) == len(names):
            return tuple(float(w) for w in declared)
        return tuple(1.0 for _ in names)

    def _call_refs(self, b: StepBucket) -> dict[str, Any]:
        """One ``GraderCallRef`` per reward function for this step, from the tap's own records.

        ``latency_s`` is deliberately left ``None`` and the call's latency goes in ``facets``
        instead. TRL scores a whole ``B x G`` batch in one call, so there is no per-row timing to
        be had, and putting the batch's latency on every row would let anybody who sums the column
        get a number ``B x G`` times too large. The facet says what it is.
        """
        from reward_lens.record.scores import GraderCallRef

        out: dict[str, Any] = {}
        for call in b.calls:
            out[call.grader] = GraderCallRef(
                grader=call.grader,
                outcome=call.outcome.value,
                facets={
                    "call_latency_s": call.inner_ns / 1e9,
                    "rows_in_call": call.shape.length if call.shape is not None else None,
                    "note": "one TRL call scores the whole B x G batch; this is that call",
                },
                seq=call.seq,
                step=call.step,
                error_type=call.error_type,
                error_message=call.error_message,
            )
        return out

    def _score_tree(
        self, row: Mapping[str, float | None], refs: Mapping[str, Any], weights: tuple[float, ...]
    ) -> Any:
        """One rollout's score as a ``ScoreTree``: a weighted sum of one leaf per reward function.

        That is exactly the operator TRL applies at ``grpo_trainer.py:2683``,
        ``(rewards_per_func * reward_weights).nansum(dim=1)``, and recording it as a tree rather
        than as a total is what lets the E series ablate a component without re-running anything.

        **The tree and TRL disagree about a partial abstention, on purpose, and the disagreement
        is the measurement.** ``nansum`` treats a NaN as a zero, so a row where one of three
        graders abstained is summed over the other two and the missing component is silently worth
        nothing. ``evaluate`` on this tree returns NaN instead, because a sum with a missing term
        is not a number. TRL only refuses when *every* function abstained (``unscorable_mask`` at
        2679). So on a multi-grader run, ``evaluate(trajectory.scores)`` is NaN on exactly the rows
        where TRL substituted a zero, and comparing it against ``features["trl_realised_reward"]``
        finds them. With one reward function the two agree by construction, which is why this needs
        saying rather than showing.
        """
        from reward_lens.record.scores import Leaf, WeightedSum

        names = tuple(row)
        leaves = tuple(
            Leaf(
                name=name,
                value=row[name],
                grader_call=refs.get(name),
                abstained=row[name] is None,
            )
            for name in names
        )
        if len(leaves) == 1 and weights == (1.0,):
            # An unweighted single grader is its own composition. Wrapping it in a sum of one would
            # put a node in the tree that the run does not have, and node_names is part of the
            # record's identity.
            return leaves[0]
        return WeightedSum(name="reward", children=leaves, weights=weights)

    def _features_for(
        self,
        row: Mapping[str, float | None],
        weights: tuple[float, ...],
        *,
        completion_text: str = "",
        completion_tokens: int | None = None,
        group_ordinal: int | None = None,
        generation_index: int | None = None,
    ) -> dict[str, float]:
        """The per-rollout scalars that belong on the row rather than in a step aggregate.

        ``trl_realised_reward`` is TRL's own realised reward for this row, kept because it is what
        produced the advantage. This is ``nansum`` reproduced: present components are weighted and
        summed, absent ones contribute nothing rather than refusing. It is not the metrologically
        right answer and it is not meant to be. It is the number the optimizer actually used, and a
        record that holds only the right answer cannot show that the run used a different one.
        Absent entirely when every component abstained, because TRL marks that row unscorable at
        2679 and its advantage is forced to zero at 2730. A zero here would be the exact confusion
        the field is missing.

        ``completion_length_tokens`` is BLK-010. TRL builds the per-sequence vector at
        ``grpo_trainer.py:2289``/``:2291``, reduces it to mean, min and max at ``:2302-2304`` and
        keeps nothing per row, so the length the length-matched comparators and the bunching
        estimator need never existed downstream. It is recovered from ``completion_ids``, which
        TRL hands every callable reward function at ``:1673``; ``len(ids)`` here is the same
        expression the trainer evaluates at ``:2291``. **Absent when the ids are absent.** A
        character count promoted to a token count is not a degraded reading of this quantity, it
        is a different quantity, and this project has already published one wrong negative on that
        substitution: "the largest upper-tail tie is 2" was in characters and is 211 in tokens.

        ``completion_length_chars`` is that character proxy, emitted **beside** the token count and
        named for what it is. Both are written so a reader can see the two disagree instead of
        inheriting one for the other; nothing downstream may read the chars field as a length in
        tokens, and the pair is what makes the substitution refutable rather than invisible.

        ``prompt_group_id`` and ``generation_index`` are BLK-009. ``:796`` already puts the group
        into ``Group.id`` and ``:804`` puts the generation index inside the trajectory id, and a
        value a reader has to recover by parsing an opaque identifier is not a recorded field: the
        id's construction is free to change without anybody noticing what it took with it, and
        every within-group estimand depends on the grouping being readable. ``prompt_group_id`` is
        the ordinal within the step, which is what pairs with ``generation_index`` to address a
        rollout; ``Group.id`` stays the globally unique handle. **Both are absent when
        ``_grouping_provenance`` could not show the row order is the generation order**, because
        the alternative is a grouping recovered from a permuted order, and BLK-009's own
        consequence line is that such a grouping either fails C3's theorem check at the instrument
        or, worse, passes on the wrong grouping.

        The keys are ``FeatureID``s, a plain string namespace rather than registered quantities, so
        naming them is not a quantity-id decision. ``trl_realised_reward`` keeps its framework
        prefix because it is a TRL-specific number; the other four do not, because they are the
        recording contract's own cross-framework field names and one quantity gets one spelling.
        """
        out: dict[str, float] = {}
        total = _total(row, weights)
        if total is not None:
            out["trl_realised_reward"] = total
        out["completion_length_chars"] = float(len(completion_text))
        if completion_tokens is not None:
            out["completion_length_tokens"] = float(completion_tokens)
        if group_ordinal is not None:
            out["prompt_group_id"] = float(group_ordinal)
        if generation_index is not None:
            out["generation_index"] = float(generation_index)
        return out

    def _grouping_provenance(
        self, prompts: tuple[str, ...] | None, n: int, k: int
    ) -> tuple[bool, str]:
        """Whether this step's row order is still the generation order, and why not when it is not.

        BLK-009. ``shuffle_sequence_dict`` at ``grpo_trainer.py:1589`` permutes the generation
        batch, and downstream of it consecutive runs of ``num_generations`` rows are not one
        prompt's rollouts. Everything this adapter reads is upstream of that line, but "upstream"
        is a claim about TRL's control flow at one version, and a claim nothing checks is the kind
        that survives the release that falsifies it.

        The check is the one signal a permutation cannot preserve: a group is one prompt scored K
        times, so its K rows carry the same prompt. Shuffled rows do not, and the probability that
        a permutation leaves every group prompt-homogeneous by accident falls off fast in the
        number of groups. It costs one pass over the prompt list per step, at finish time.

        ``k <= 1`` is verified trivially and honestly: with one rollout per prompt there is no
        grouping to get wrong. No prompts means not verified, because the check cannot run, and
        an unrunnable check is not a passing one.
        """
        if k <= 1:
            return True, ""
        if prompts is None or len(prompts) < n:
            return False, (
                "no prompt column for this step: the K rows of a group are one prompt's rollouts, "
                "so without the prompts there is nothing to check the row order against"
            )
        for g in range((n + k - 1) // k):
            lo, hi = g * k, min((g + 1) * k, n)
            if hi - lo < 2:
                continue
            distinct = {prompts[j] for j in range(lo, hi)}
            if len(distinct) != 1:
                return False, (
                    f"candidate group {g} spans {len(distinct)} distinct prompts across "
                    f"{hi - lo} rows, so the row order is not the generation order. Downstream of "
                    f"shuffle_sequence_dict (grpo_trainer.py:1589) the grouping is not recoverable "
                    f"from row order and it is not guessed here"
                )
        return True, ""

    def _completion_lengths_for(self, b: StepBucket, n: int) -> tuple[int, ...] | None:
        """Per-rollout completion length in tokens, or ``None`` when it was not recoverable.

        ``completion_ids`` is the third positional keyword TRL passes every callable reward
        function (``grpo_trainer.py:1673`` for the sync branch, ``:1691`` for the async one): a
        list of per-sequence token id lists, unpadded, one per rollout. It arrives **inside**
        ``_generate_and_score_completions``, so it is upstream of ``shuffle_sequence_dict`` at
        ``:1589`` and the row order is the generation order. ``len(ids)`` is the same expression
        the trainer evaluates at ``:2291`` before reducing it away.

        Three things this deliberately does not do.

        It does not fall back to ``trainer._logs``, because the completion length in tokens is not
        in ``_logs`` at all: only the text is. Reconstructing it by re-tokenising that text would
        be a second tokenizer's opinion about a batch the first one already counted.

        It does not keep the ids. One int per rollout is what the contract row asks for and what
        the downstream estimands consume; retaining the ids for a 401-step run at 128 rollouts and
        about 1,240 tokens each is tens of millions of Python ints for a number already computed.
        That is why ``Turn.token_ids`` stays ``None`` on this adapter's records and
        ``Trajectory.n_tokens`` stays 0, which was already true before this change.

        It does not pad or truncate. A ``completion_ids`` shorter or longer than ``completions``
        means the two came from different batches, and matching the rows it can would put a length
        on a row it does not belong to. The whole step abstains instead.

        **The one case where this is not TRL's logged number.** With tools or a rollout-func
        ``env_mask``, ``:2289`` counts only model-generated tokens (``sum(mask)``) while this
        counts every completion token. The design's run has no tools; a run that does gets the
        completion length rather than the model-token length and this docstring is the notice.
        """
        for call in b.calls:
            kwargs = call.kwargs
            if not kwargs:
                continue
            ids = kwargs.get("completion_ids")
            if ids is None:
                continue
            try:
                lengths = tuple(len(seq) for seq in ids)
            except TypeError:
                return None
            return lengths if len(lengths) == n else None
        return None

    def _completion_ids_for(self, b: StepBucket, n: int) -> tuple[tuple[int, ...], ...] | None:
        """The per-rollout completion token ids themselves, not their lengths. BLK-027.

        ``_completion_lengths_for`` above deliberately discards these and its docstring says
        why: one int per rollout is what the downstream estimands consume, and keeping the ids
        for a 401-step run is tens of millions of Python ints for a number already computed.
        That reasoning is right about the estimands and wrong about the contract.
        ``record/turns.py`` **declares** ``Turn.token_ids``, ``worlds/RECORDING.md``'s §3.4
        lists the per-token fields as contract rows, and a declared field that nothing writes is
        the defect BLK-027 names. So the ids are kept when asked for and not otherwise.

        ``TRLTap(record_token_ids=False)`` is the switch. It defaults to True because the field
        is declared; a caller who wants the old behaviour gets the same ``None`` as before, which
        is an abstention rather than an empty tuple.

        **The cost, stated rather than implied.** At the design's geometry, 128 rollouts of about
        1,240 tokens each, one step is roughly 159,000 ints and 401 steps is about 64 million.
        Records are written per step rather than accumulated, so the resident cost is one step;
        the cost that lands is on disk, and `worlds/RECORDING.md` already budgets the per-token
        fields at 1.6 MB per rollout-set at ``L = 32``.

        Same three refusals as its sibling: no fall back to ``trainer._logs``, which does not
        carry ids at all; no re-tokenising of the text, which is a second tokenizer's opinion; and
        no padding or truncation, because a length mismatch means two different batches and the
        whole step abstains.
        """
        if not self.record_token_ids:
            return None
        for call in b.calls:
            kwargs = call.kwargs
            if not kwargs:
                continue
            ids = kwargs.get("completion_ids")
            if ids is None:
                continue
            try:
                out = tuple(tuple(int(t) for t in seq) for seq in ids)
            except TypeError:
                return None
            return out if len(out) == n else None
        return None

    def _texts_for(self, b: StepBucket) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
        """Prompts and completions, preferring the grader's own arguments over TRL's log deques.

        The wrapped call retained the host's objects by reference, so where the tap was working
        these are the exact lists the grader was handed. ``trainer._logs`` is the fallback and it
        is post-gather, which is the same content on one process and a superset on several.
        """
        for call in b.calls:
            kwargs = call.kwargs
            if not kwargs:
                continue
            completions = kwargs.get("completions")
            prompts = kwargs.get("prompts")
            if completions is None:
                continue
            return (
                tuple(str(p) for p in prompts) if prompts is not None else None,
                tuple(str(c) for c in completions),
            )
        return (b.prompts, b.completions)

    def _scores_for(self, b: StepBucket, n: int) -> dict[str, list[float | None]]:
        """Per-reward-function scores for this step, one entry per row, ``None`` for an abstention.

        Reading the return value here rather than in the wrapper is the point of
        ``GraderCall.components()``: walking a ``B x G`` list costs more than everything else the
        tap does and Plane A may not spend it, so the walk happens now, after training.
        """
        out: dict[str, list[float | None]] = {}
        for call in b.calls:
            if call.outcome in (CallOutcome.RAISED, CallOutcome.TIMED_OUT):
                out[call.grader] = [None] * n
                continue
            values: list[float | None] = [None] * n
            for key, value in call.components():
                try:
                    idx = int(key)
                except ValueError:
                    continue
                if 0 <= idx < n:
                    values[idx] = _as_float(value)
            out[call.grader] = values
        for name, gathered in b.rewards.items():
            if name in out:
                continue
            out[name] = [_as_float(v) for v in gathered[:n]] + [None] * max(0, n - len(gathered))
        return out

    def _turns_for(
        self,
        prompt: str,
        completion: str,
        completion_ids: tuple[int, ...] | None = None,
    ) -> tuple[Any, ...]:
        """One user turn and one assistant turn. Single-turn GRPO is two turns, not one.

        Recording the prompt as a turn rather than as a field on the trajectory is what makes a
        multi-turn record and a single-turn record the same shape, which is the whole reason the
        hierarchy has a turn level.

        **BLK-027, and what is filled here and what is not.** ``Turn`` declares four per-token
        fields and this adapter set none of them. ``token_ids`` is now populated on the assistant
        turn from ``completion_ids``, which arrives as the third keyword TRL passes every reward
        function and is upstream of ``shuffle_sequence_dict``, so its row order is the generation
        order. The other three are not reachable from here and the reason differs per field:

        ``loss_mask``       TRL's ``completion_mask`` is built in
                            ``_generate_and_score_completions`` and is not passed to a reward
                            function. It is reachable from ``trainer._buffered_inputs``, which is
                            assigned AFTER ``shuffle_sequence_dict`` permutes the batch and the
                            permutation is never returned, so the rows cannot be mapped back
                            except by content-matching completions, which is ambiguous on
                            duplicates. Part 4.2's override at the point before the shuffle is
                            what makes it reachable, and that is BLK-009's build item.
        ``logprobs_sampling`` same route, same obstacle, and additionally absent entirely unless
                            ``use_vllm=True``: on the transformers generate path
                            ``_generate_single_turn`` sets ``logprobs = None``. The design sets
                            ``use_vllm`` to colocate for exactly this reason.
        ``logprobs_train``  not reachable from ANY callback. It is consumed into ``log_ratio``
                            inside ``_compute_loss`` and dies there. The ``_compute_loss``
                            override is BLK-004's, in ``P2-LOSS``.

        The prompt turn's ``token_ids`` stays ``None``: ``prompt_ids`` is not among the keywords
        a reward function receives, and re-tokenising the prompt text would be a second
        tokenizer's opinion about a batch the first one already counted.
        """
        from reward_lens.record.turns import Turn

        return (
            Turn(index=0, role="user", text=prompt),
            Turn(index=1, role="assistant", text=completion, token_ids=completion_ids),
        )

    def _provenance_for(self, b: StepBucket, *, n_turns: int) -> tuple[Any, ...]:
        """Segment provenance, mandatory and covering every turn.

        One segment here, and that is a claim rather than a default: it holds because this run
        generated with the training process's own weights at this step and consumed them at this
        step. ``num_iterations > 1`` or ``steps_per_generation > 1`` breaks it, because a
        generation is then reused across optimizer steps and the second consumer is off-policy by
        the number of steps in between. ``attach`` reads both, ``_regime`` refuses to declare
        ``NEAR_POLICY`` when either is not 1, and the staleness recorded here follows the same
        rule rather than being hardcoded to zero.
        """
        from reward_lens.record.provenance import SamplingMeta, SegmentProvenance
        from reward_lens.record.tensors import Engine

        cfg = self._config
        reuse = int(cfg.get("num_iterations") or 1) * int(cfg.get("steps_per_generation") or 1)
        return (
            SegmentProvenance(
                turn_range=(0, n_turns),
                policy_version=f"{self.run_id}@step{b.index}",  # type: ignore[arg-type]
                staleness_steps=0 if reuse == 1 else max(0, reuse - 1),
                engine=Engine(
                    name=str(cfg.get("generation_engine") or "transformers"),
                    revision=str(cfg.get("transformers_version") or "unknown"),
                    attention_impl=str(cfg.get("attn_implementation") or "unknown"),
                    dtype=str(cfg.get("dtype") or "unknown"),
                ),
                sampling=SamplingMeta(
                    temperature=_opt_float(cfg.get("temperature")),
                    top_p=_opt_float(cfg.get("top_p")),
                    top_k=_opt_int(cfg.get("top_k")),
                    seed=_opt_int(cfg.get("seed")),
                    max_tokens=_opt_int(cfg.get("max_completion_length")),
                    batch_composition=(
                        f"{cfg.get('num_generations')} generations per prompt, "
                        f"generation batch {cfg.get('generation_batch_size')}"
                    ),
                ),
            ),
        )

    def _estimator_spec(self) -> Any:
        """How scores became advantages, read off ``GRPOConfig`` rather than inferred.

        Section 2.2 asks for this exactly, and GRPO is one of the few estimators where exactly is
        achievable, because every branch in ``_generate_and_score_completions`` (2681-2725) is
        selected by a config field.

        ``std_normalised`` is ``scale_rewards != "none"``, and the epsilon is the literal ``1e-4``
        added to the denominator at 2707 and 2721. ``clip_low``/``clip_high`` are ``epsilon`` and
        ``epsilon_high or epsilon``, the PPO-style ratio clip. ``advantage_whitening`` is
        ``normalize_then_sum``, which standardises each reward function within the group before
        weighting, and it is a different operator from ``sum_then_normalize``: the record has to
        say which one ran or the advantage is not reproducible from it.
        """
        from reward_lens.record.schema import EstimatorSpec

        cfg = self._config
        scale = str(cfg.get("scale_rewards") or "group")
        agg = str(cfg.get("multi_objective_aggregation") or "sum_then_normalize")
        loss_type = str(cfg.get("loss_type") or "unknown")
        aggregation = {
            "grpo": "sequence",
            "bnpo": "batch",
            "dr_grpo": "token",
            "dapo": "token",
        }.get(loss_type, "unknown")
        return EstimatorSpec(
            family=f"grpo/{loss_type}",
            group_centred=True,
            std_normalised=scale != "none",
            std_epsilon=1e-4 if scale != "none" else None,
            # TRL's `nanstd` applies Bessel's correction explicitly, multiplying the variance by
            # `count / (count - 1)` at `trl/trainer/utils.py:877-879`, so the divisor is the
            # sample standard deviation and not the population one. Read from the installed
            # source rather than assumed: the two differ by 15.5% at the `num_generations = 4`
            # this adapter is most often pointed at.
            std_ddof=1 if scale != "none" else None,
            degenerate_policy=(
                "advantage is zero when the group std is zero, because the numerator is zero; "
                "TRL does not drop the group. Rows where every reward function abstained are "
                "masked to NaN at grpo_trainer.py:2679 and their advantage forced to 0 at 2730."
            ),
            clip_low=_opt_float(cfg.get("epsilon")),
            clip_high=_opt_float(cfg.get("epsilon_high") or cfg.get("epsilon")),
            clip_ratio_c=_opt_float(cfg.get("delta")),
            aggregation=aggregation,  # type: ignore[arg-type]
            loss_mask_policy=(
                "truncated completions masked" if cfg.get("mask_truncated_completions") else "none"
            ),
            off_policy_correction=str(cfg.get("importance_sampling_level") or "none"),
            kl_penalty="k3" if _opt_float(cfg.get("beta")) else None,
            kl_coefficient=_opt_float(cfg.get("beta")),
            advantage_whitening=agg == "normalize_then_sum",
            extra={
                "scale_rewards": scale,
                "multi_objective_aggregation": agg,
                "loss_type": loss_type,
                "num_iterations": cfg.get("num_iterations"),
                "steps_per_generation": cfg.get("steps_per_generation"),
                "reward_weights": cfg.get("reward_weights"),
            },
        )

    def sampling_policy(self) -> Any:
        """What fraction of the run this record holds, and how the fraction was chosen.

        Three ways this record can be less than the whole run, and each has to show up here
        rather than as a silent hole. The step buffer can fill, which keeps the first
        ``max_steps`` and is a SELECTIVE sample with the reason written out. The ring can drop
        under load, which loses individual grader calls. And a reward *model* is never wrapped at
        all, so a run whose reward is a model has no grader calls to record.
        """
        from reward_lens.record.schema import RecordSamplingPolicy, SamplingScheme

        recorded = len(self._buckets)
        population = recorded + self._dropped_steps
        ring = self.handle.ring.stats()
        notes = [f"ring: {ring.render()}"]
        if self.skipped:
            notes.append("not instrumented: " + "; ".join(s.render() for s in self.skipped))
        if self.adapter_exceptions:
            notes.append(
                f"adapter caught {self.adapter_exceptions} exceptions, "
                f"{len(self.adapter_exception_keys)} distinct: "
                + "; ".join(self.adapter_exception_keys)
            )
        if self._dropped_steps:
            return RecordSamplingPolicy(
                scheme=SamplingScheme.SELECTIVE,
                rate=recorded / population if population else 1.0,
                selected_by=(
                    f"the first {recorded} optimizer steps; the in-process step buffer reached its "
                    f"cap of {self.max_steps} and {self._dropped_steps} later steps were not "
                    f"recorded. The sample is a prefix of the run, so any statistic over it is a "
                    f"statistic about early training and no arithmetic correction recovers the rest."
                ),
                population=population,
                recorded=recorded,
                unit="step",
                notes=" | ".join(notes),
            )
        return RecordSamplingPolicy(
            scheme=SamplingScheme.FULL,
            rate=1.0,
            population=population,
            recorded=recorded,
            unit="step",
            notes=" | ".join(notes),
        )

    def finish(self, *, kind: str = "train", access: Mapping[Any, Any] | None = None) -> "Run":
        """The whole record: ``Run -> Step -> Group -> Trajectory -> Turn``, built after training.

        ``access`` defaults to ``RECORD`` on every component and nothing above it. That is not
        modesty, it is what a record supports: ``Access.RECORD`` is "read logged values that
        already exist" and that is exactly and only what this produces. Whether the grader can
        still be called, whether the policy can still be run forwards, and whether either can be
        differentiated are facts about what the analyst holds after the run, which this adapter
        cannot check and will not assert. Pass ``access=`` to say so when it is true.
        """
        from reward_lens.core.types import Access, Component
        from reward_lens.record.schema import (
            ComponentRef,
            InMemoryStepStream,
            Run,
            RunLineage,
            run_id,
        )

        cfg = self._config
        steps = self.steps()
        components = {
            Component.POLICY: ComponentRef(
                name=str(cfg.get("model") or "unknown"),
                kind="policy",
                version=str(cfg.get("transformers_version") or "unknown"),
            ),
            Component.GRADER: ComponentRef(
                name=", ".join(getattr(f, "__name__", repr(f)) for f in self.wrapped) or "none",
                kind="reward_functions",
                extra={
                    "n_wrapped": len(self.wrapped) - len(self.skipped),
                    "n_skipped": len(self.skipped),
                    "skipped": [s.render() for s in self.skipped],
                },
            ),
            Component.ESTIMATOR: ComponentRef(name="grpo", kind="estimator"),
            Component.OPTIMIZER: ComponentRef(
                name=str(cfg.get("optim") or "unknown"), kind="optimizer"
            ),
        }
        return Run(
            id=run_id(name=self.run_id, seed=_opt_int(cfg.get("seed"))),
            kind=kind,  # type: ignore[arg-type]
            components=components,
            access=(
                dict(access)
                if access is not None
                else {
                    c: Access.RECORD
                    for c in (
                        Component.POLICY,
                        Component.GRADER,
                        Component.ESTIMATOR,
                        Component.OPTIMIZER,
                        Component.RECORD,
                    )
                }
            ),
            regime=self._regime(),
            steps=InMemoryStepStream(steps),
            lineage=RunLineage(
                framework="trl",
                framework_version=str(cfg.get("trl_version") or "unknown"),
                created_at=self._started_at,
                extra={
                    "transformers_version": cfg.get("transformers_version"),
                    "attached": self._attached,
                    "device": cfg.get("device"),
                    "config": {k: v for k, v in cfg.items() if k != "device"},
                },
            ),
            sampling_policy=self.sampling_policy(),
        )

    def _regime(self) -> Any:
        """Declare only the conditions ``GRPOConfig`` settles, and say who declared them.

        Three are settled by configuration alone and the rest are not.

        ``NEAR_POLICY`` holds when ``num_iterations`` and ``steps_per_generation`` are both 1, so
        that every rollout is consumed by the optimizer step that generated it and the segment
        provenance is singular. Either one above 1 and a generation is reused, so it is declared
        False rather than omitted, because "we know it is violated" is different from "we did not
        look".

        ``NO_COMPACTION`` holds because this path never rewrites a prefix: TRL's GRPO generates a
        completion per prompt and scores it, and there is nowhere for a rewrite to happen.

        ``MASK_STABLE`` holds because the loss-mask policy is ``mask_truncated_completions`` and
        ``loss_type``, both fixed on the config for the whole run.

        ``STATIONARY_GRADER`` is deliberately absent, and it is the interesting omission. TRL
        hands the reward function ``trainer_state`` at 1621 specifically so it can shape reward by
        training progress, and its own comment says so. A grader that reads ``global_step`` is
        non-stationary by design and nothing on the config reveals whether this one does, so
        declaring it either way would be a guess. The measured conditions are Plane B's.
        """
        from reward_lens.core.envelope import RegimeCondition
        from reward_lens.record.schema import RegimeDeclaration

        cfg = self._config
        iters = int(cfg.get("num_iterations") or 1)
        per_gen = int(cfg.get("steps_per_generation") or 1)
        on_policy = iters == 1 and per_gen == 1
        return RegimeDeclaration(
            declared={
                RegimeCondition.NEAR_POLICY: on_policy,
                RegimeCondition.NO_COMPACTION: True,
                RegimeCondition.MASK_STABLE: True,
            },
            notes={
                RegimeCondition.NEAR_POLICY: (
                    f"num_iterations={iters}, steps_per_generation={per_gen}; a generation is "
                    f"consumed by {iters * per_gen} optimizer step(s)"
                ),
                RegimeCondition.NO_COMPACTION: (
                    "TRL's GRPO path generates one completion per prompt and never rewrites a "
                    "prefix, so no importance ratio in this run crosses a compaction boundary"
                ),
                RegimeCondition.MASK_STABLE: (
                    f"loss_type={cfg.get('loss_type')}, "
                    f"mask_truncated_completions={cfg.get('mask_truncated_completions')}, "
                    f"both fixed on GRPOConfig for the whole run"
                ),
            },
            declared_by="reward_lens.tap.adapters.trl, read from GRPOConfig",
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _callback_base() -> type:
    """``TrainerCallback`` when transformers is installed, and a bare stand-in when it is not.

    ``Trainer.add_callback`` type-checks what it is handed, so a real run needs the real base class
    and gets it. But a process without transformers is a process without a ``Trainer``, so the only
    caller that can reach the fallback is a test driving the step boundary over a trainer-shaped
    object, which is the half of this adapter that is meant to work on a base install.

    Importing unconditionally was the same shape of bug the guard in ``attach`` is there to absorb,
    and the two combined to hide it: the import raised, ``attach`` filed the ImportError as a note
    and returned, no callback was ever registered, and the tap then recorded nothing for the whole
    run without ever saying that transformers was what it wanted.
    """
    try:
        from transformers import TrainerCallback
    except ImportError:
        return object
    return TrainerCallback


def _is_torch_module(obj: Any) -> bool:
    """Whether TRL will take the ``nn.Module`` branch, without importing torch to find out.

    The MRO walk comes first and the ``isinstance`` second, and the order is the point. A process
    with no torch in it cannot be holding a torch module, so importing torch to answer the
    question would be a heavier side effect than the answer is worth; but doing the ``isinstance``
    first *when torch happens to be loaded* makes the function's answer depend on whether some
    unrelated code has imported torch yet, which is exactly the kind of order-dependence that
    turns into a flaky test and then into a wrong record. Walking the MRO for a class defined
    under ``torch.nn`` is import-free, catches a user's own ``nn.Module`` subclass through its
    bases, and gives the same answer either way.
    """
    import sys

    if any(c.__module__.startswith("torch.nn") for c in type(obj).__mro__):
        return True
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            return isinstance(obj, torch.nn.Module)
        except Exception:
            return False
    return False


def _read_config(trainer: Any) -> dict[str, Any]:
    """Everything about the run that is fixed at construction, read once.

    Deliberately tolerant. Every field is fetched with a default because TRL moves weekly and a
    field that vanished should leave a gap in the record, not an exception in a training loop:
    ``max_prompt_length`` was on ``GRPOConfig`` when the specification was written and is not on
    1.9.2, which is exactly the failure this shape prevents.
    """
    args = getattr(trainer, "args", None)
    out: dict[str, Any] = {}
    for key in (
        "beta",
        "temperature",
        "top_p",
        "top_k",
        "epsilon",
        "epsilon_high",
        "delta",
        "seed",
        "loss_type",
        "scale_rewards",
        "multi_objective_aggregation",
        "importance_sampling_level",
        "mask_truncated_completions",
        "num_iterations",
        "steps_per_generation",
        "generation_batch_size",
        "max_completion_length",
        "num_generations",
        "reward_weights",
        "log_completions",
        "use_vllm",
        "optim",
        "learning_rate",
        "gradient_accumulation_steps",
        "per_device_train_batch_size",
        "max_steps",
        "output_dir",
    ):
        value = getattr(args, key, None)
        if value is not None and not isinstance(value, (str, int, float, bool, list, tuple)):
            value = str(value)
        out[key] = list(value) if isinstance(value, tuple) else value

    # num_generations lives on the trainer as well as the config, and the trainer's is the one
    # _generate_and_score_completions uses for the group view, so prefer it.
    k = getattr(trainer, "num_generations", None)
    if isinstance(k, int):
        out["num_generations"] = k

    model = getattr(trainer, "model", None)
    if model is not None:
        cfg = getattr(model, "config", None)
        out["model"] = getattr(cfg, "name_or_path", None) or type(model).__name__
        out["dtype"] = str(getattr(model, "dtype", "unknown"))
        out["attn_implementation"] = str(
            getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
        )
        try:
            out["device"] = str(next(model.parameters()).device)
        except Exception:
            out["device"] = "unknown"

    import sys

    for mod, key in (("trl", "trl_version"), ("transformers", "transformers_version")):
        m = sys.modules.get(mod)
        out[key] = getattr(m, "__version__", "unknown") if m is not None else "unknown"
    out["generation_engine"] = "vllm" if out.get("use_vllm") else "transformers"
    return out


def _as_float(value: Any) -> float | None:
    """A score, or ``None`` for an abstention. Never a zero standing in for either."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _total(
    scores: Mapping[str, float | None], weights: Sequence[float] | None = None
) -> float | None:
    """The realised total, or ``None`` when every component abstained.

    TRL's own rule at ``grpo_trainer.py:2679-2684``, reproduced rather than improved on. A row
    where every reward function returned ``None`` is unscorable and is marked NaN; a row where
    *some* abstained is summed over the ones that did not, because ``nansum`` treats a NaN as a
    zero. The second half is a silent behaviour, and the group statistics on the record have to be
    computed from the same numbers the group baseline was, or the record describes a run that did
    not happen. ``evaluate(trajectory.scores)`` is where the metrologically correct refusal lives.
    """
    values = list(scores.values())
    ws = (
        list(weights)
        if weights is not None and len(weights) == len(values)
        else [1.0] * len(values)
    )
    present = [(w, v) for w, v in zip(ws, values) if v is not None]
    if not present:
        return None
    return float(sum(w * v for w, v in present))


def _short(text: str, n: int = 48) -> str:
    """A stable, printable stand-in for a prompt, for use inside a task id."""
    flat = " ".join(text.split())
    return flat[:n] if flat else "empty"


__all__ = [
    "DEFAULT_MAX_STEPS",
    "EXTRA_COLUMN",
    "METRIC_NAMES",
    "Skipped",
    "StepBucket",
    "TRLRunHandle",
    "TRLTap",
]
