"""Executed grader scores for a supplied response bank.

The carrier is deliberately narrower than a training trace.  A response bank
can show variation in what the supplied grader returns for repeated declared
prompt groups; it cannot establish the reward that was applied during an
optimiser update, its components, or an optimiser's grouping semantics.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from reward_lens.execution import Limits
from reward_lens.graders.python_grader import PythonGrader

ResponseScoreStatus = Literal["measured", "invalid", "error"]

# Each row is an independent, contained PythonGrader subprocess with its own
# fresh scratch directory.  Six concurrent rows leave headroom for the Audit
# process and keep the real response-bank MCP route inside its fixed request
# deadline without changing any individual score's limits or evidence.
# This is a process-wide admission bound, shared by ordinary thread workers
# and by each request's reaped process workers.  Twelve cuts the 120-row
# fixture to ten waves while leaving explicit headroom for the MCP server.
_MAX_RESPONSE_SCORE_WORKERS = 12
_RESPONSE_SCORE_CAPACITY = threading.BoundedSemaphore(_MAX_RESPONSE_SCORE_WORKERS)
_CANCELLATION_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "response_score_cancellation_event", default=None
)


class ResponseScoreCancelled(RuntimeError):
    """The caller cancelled a response-bank run after its active children were reaped."""


def _profile(event: str) -> None:
    path = os.environ.get("REWARD_LENS_MCP_PROFILE_PATH")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"{event} {time.monotonic():.9f}\n")
        except OSError:
            pass


@contextmanager
def response_score_cancellation(event: threading.Event):
    """Bind a caller's cancellation event to one synchronous audit invocation."""
    token = _CANCELLATION_EVENT.set(event)
    try:
        yield
    finally:
        _CANCELLATION_EVENT.reset(token)


@dataclass(frozen=True)
class ResponseScore:
    """One actual grader invocation over one joined response-bank rollout."""

    response_id: str
    task_id: str
    score: float | None
    status: ResponseScoreStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.response_id or not self.task_id:
            raise ValueError("a response score needs its response and declared task identities")
        if self.status == "measured":
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("a measured response score must be numeric")
            if not math.isfinite(float(self.score)):
                raise ValueError("a measured response score must be finite")
        elif self.score is not None:
            raise ValueError("an invalid or errored response score remains absent")


def _score_rollout(
    rollout: Any,
    *,
    grader: PythonGrader,
    sandbox: Any,
    project_dir: Path | None,
    capacity_reserved: bool = False,
) -> ResponseScore:
    """Score one independent rollout with the caller's containment capability."""
    inputs = getattr(rollout, "inputs", {})
    task = inputs.get("task") if isinstance(inputs, dict) else None
    response = inputs.get("response") if isinstance(inputs, dict) else None
    response_id = str(getattr(rollout, "id", ""))
    task_id = task.get("id") if isinstance(task, dict) else None
    if not response_id or not isinstance(task_id, str) or not task_id:
        return ResponseScore(
            response_id=response_id or "unidentified-response",
            task_id=str(task_id or "unidentified-task"),
            score=None,
            status="invalid",
            reason="the joined rollout does not carry a declared task_id and response_id",
        )

    # PythonGrader holds only immutable source/manifest configuration while
    # score() stages a distinct temporary tree for every invocation.  The
    # sandbox is the caller-supplied capability, so this helper cannot weaken
    # containment or substitute a shared artifact directory.
    try:
        if capacity_reserved:
            envelope = grader.score(
                task,
                response,
                sandbox=sandbox,
                limits=Limits(),
                project_dir=project_dir,
            )
        else:
            with _RESPONSE_SCORE_CAPACITY:
                envelope = grader.score(
                    task,
                    response,
                    sandbox=sandbox,
                    limits=Limits(),
                    project_dir=project_dir,
                )
    except Exception as error:
        return ResponseScore(
            response_id=response_id,
            task_id=task_id,
            score=None,
            status="error",
            reason=f"operational scoring failure: {type(error).__name__}: {error}",
        )
    raw = envelope.score
    if (
        str(envelope.verdict) == "scored"
        and not isinstance(raw, bool)
        and isinstance(raw, (int, float))
        and math.isfinite(float(raw))
    ):
        return ResponseScore(
            response_id=response_id,
            task_id=task_id,
            score=float(raw),
            status="measured",
        )

    verdict = str(envelope.verdict)
    status: ResponseScoreStatus = "invalid" if verdict == "invalid_input" else "error"
    detail = "; ".join(str(error) for error in (envelope.errors or ()))
    if not detail:
        detail = f"grader returned {verdict} without a finite numeric score"
    return ResponseScore(
        response_id=response_id,
        task_id=task_id,
        score=None,
        status=status,
        reason=detail,
    )


def _score_in_child(
    receive: object,
    send: object,
    grader: PythonGrader,
    sandbox: Any,
    project_dir: Path | None,
) -> None:
    """Run assigned scores in a killable parent of their contained grader groups."""
    if sys.platform == "linux":
        # launch blocks SIGINT while registering a worker. Leave the terminal's
        # process group before unblocking it, so Ctrl+C reaches the CLI, which
        # can reap this worker and its contained grader groups. Drop a signal
        # that was pending during spawn; it addressed the former group.
        os.setsid()
        if signal.SIGINT in signal.sigpending():
            signal.sigwait({signal.SIGINT})
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    try:
        while True:
            try:
                index, rollout = receive.recv()  # type: ignore[union-attr]
            except EOFError:
                return
            send.send(  # type: ignore[union-attr]
                (
                    index,
                    _score_rollout(
                        rollout,
                        grader=grader,
                        sandbox=sandbox,
                        project_dir=project_dir,
                        capacity_reserved=True,
                    ),
                )
            )
    finally:
        receive.close()  # type: ignore[union-attr]
        send.close()  # type: ignore[union-attr]


def _error_for_lost_child(rollout: Any, detail: str) -> ResponseScore:
    inputs = getattr(rollout, "inputs", {})
    task = inputs.get("task") if isinstance(inputs, dict) else None
    return ResponseScore(
        response_id=str(getattr(rollout, "id", "")) or "unidentified-response",
        task_id=str(task.get("id", "")) if isinstance(task, dict) else "unidentified-task",
        score=None,
        status="error",
        reason=f"operational scoring failure: {detail}",
    )


def _reap_contained_child_groups(worker: multiprocessing.Process) -> None:
    """Kill this score worker's separate contained process groups before reaping it."""
    if not worker.is_alive():
        worker.join()
        return
    pending = [worker.pid]
    descendants: set[int] = set()
    while pending:
        pid = pending.pop()
        try:
            task_ids = tuple(Path(f"/proc/{pid}/task").iterdir())
        except OSError:
            continue
        for task in task_ids:
            try:
                children = (task / "children").read_text().split()
            except OSError:
                continue
            for child_text in children:
                child_pid = int(child_text)
                if child_pid not in descendants:
                    descendants.add(child_pid)
                    pending.append(child_pid)
    try:
        worker_group = os.getpgid(worker.pid)
    except ProcessLookupError:
        worker_group = None
    for child_pid in descendants:
        try:
            group = os.getpgid(child_pid)
        except ProcessLookupError:
            continue
        # A worker may share the caller's group during startup; a contained
        # subprocess starts a distinct session. Never signal either caller or
        # worker group; the worker's own SIGKILL handles that branch.
        if group not in {os.getpgrp(), worker_group}:
            try:
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    worker.kill()
    worker.join()


def _score_with_killable_children(
    rollouts: tuple[Any, ...],
    *,
    grader: PythonGrader,
    sandbox: Any,
    project_dir: Path | None,
    cancellation: threading.Event,
) -> tuple[ResponseScore, ...]:
    """Keep a fixed process pool so cancellation reaps active contained child groups."""
    # Spawn keeps the server's transport locks out of a score worker.  The
    # guarded package command module leaves the importable worker target alone.
    context = multiprocessing.get_context("spawn")
    results: list[ResponseScore | None] = [None] * len(rollouts)
    active: dict[int, tuple[multiprocessing.Process, Any, Any, int | None]] = {}
    next_index = 0

    def launch(worker: int) -> bool:
        if cancellation.is_set():
            raise ResponseScoreCancelled("request cancellation preempted response scorer admission")
        # Another audit may own the remaining permits. Keep polling this
        # request's active workers instead of waiting here while they hold slots.
        if not _RESPONSE_SCORE_CAPACITY.acquire(blocking=False):
            return False
        receive = work_send = result_receive = send = None
        child = None
        previous_mask = None
        try:
            receive, work_send = context.Pipe(duplex=False)
            result_receive, send = context.Pipe(duplex=False)
            child = context.Process(
                target=_score_in_child,
                args=(receive, send, grader, sandbox, project_dir),
                daemon=False,
            )
            if sys.platform == "linux":
                # Register the child before an interrupt may reach this thread.
                # The worker inherits the blocked mask until it leaves the
                # terminal group, so Ctrl+C cannot strand a contained grader.
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            child.start()
            active[worker] = (child, work_send, result_receive, None)
            receive.close()
            send.close()
            return True
        except BaseException:
            active.pop(worker, None)
            try:
                if child is not None and child.pid is not None:
                    _reap_contained_child_groups(child)
            finally:
                for endpoint in (receive, work_send, result_receive, send):
                    if endpoint is not None:
                        endpoint.close()
                _RESPONSE_SCORE_CAPACITY.release()
            raise
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def assign(worker: int) -> None:
        nonlocal next_index
        child, work_send, result_receive, _inflight = active[worker]
        if next_index >= len(rollouts):
            work_send.close()
            active[worker] = (child, work_send, result_receive, None)
            return
        work_send.send((next_index, rollouts[next_index]))
        active[worker] = (child, work_send, result_receive, next_index)
        next_index += 1

    pool_size = min(_MAX_RESPONSE_SCORE_WORKERS, len(rollouts))
    try:
        while active or next_index < len(rollouts):
            if cancellation.is_set():
                for child, _work_send, _result_receive, _inflight in active.values():
                    if child.is_alive():
                        _reap_contained_child_groups(child)
                for child, work_send, result_receive, _inflight in active.values():
                    child.join()
                    work_send.close()
                    result_receive.close()
                    _RESPONSE_SCORE_CAPACITY.release()
                active.clear()
                raise ResponseScoreCancelled("request cancellation reaped active response scorers")

            # Admit only workers that can immediately receive a row. A caller
            # with no permit still reaches the cancellation-responsive wait;
            # one with active workers always reaches the result poll below.
            for worker in range(pool_size):
                if next_index >= len(rollouts):
                    break
                if worker not in active:
                    if not launch(worker):
                        break
                    assign(worker)

            settled: list[int] = []
            for worker, (child, work_send, result_receive, inflight) in active.items():
                if result_receive.poll():
                    try:
                        index, row = result_receive.recv()
                        results[index] = row
                    except EOFError:
                        if inflight is not None:
                            results[inflight] = _error_for_lost_child(
                                rollouts[inflight], "score worker exited"
                            )
                        settled.append(worker)
                    else:
                        assign(worker)
                elif not child.is_alive():
                    if inflight is not None:
                        results[inflight] = _error_for_lost_child(
                            rollouts[inflight], f"score worker exited with code {child.exitcode}"
                        )
                    settled.append(worker)
            for worker in settled:
                child, work_send, result_receive, _inflight = active[worker]
                child.join()
                work_send.close()
                result_receive.close()
                del active[worker]
                _RESPONSE_SCORE_CAPACITY.release()
            if not settled:
                # This is only the parent scheduler's observation interval; it
                # does not alter any contained grader limit.  A short wait
                # keeps a completed row from idling an entire fixed pool slot.
                cancellation.wait(0.005)
    finally:
        for child, work_send, result_receive, _inflight in active.values():
            if child.is_alive():
                _reap_contained_child_groups(child)
            child.join()
            work_send.close()
            result_receive.close()
            _RESPONSE_SCORE_CAPACITY.release()

    assert all(row is not None for row in results)
    return tuple(results)  # type: ignore[return-value]


def score_response_bank(
    *,
    grader_path: Path,
    entrypoint: str,
    corpus: Any,
    sandbox: Any,
    project_dir: Path | None,
) -> tuple[ResponseScore, ...]:
    """Run each joined bank row through the actual local Python grader once.

    The caller supplies the audit's sandbox and project root so this helper
    cannot silently choose a weaker execution route.  Every joined rollout is
    represented in the result; unscored, non-finite and boolean outputs are
    retained as precise non-measurements rather than converted into a numeric
    contrast input.
    """

    grader = PythonGrader.from_path(grader_path, entrypoint)
    rollouts = tuple(getattr(corpus, "rollouts", ()))
    if not rollouts:
        return ()
    _profile("response_score_bank_start")
    score_one = partial(
        _score_rollout,
        grader=grader,
        sandbox=sandbox,
        project_dir=project_dir,
    )
    cancellation = _CANCELLATION_EVENT.get()
    try:
        if cancellation is not None:
            return _score_with_killable_children(
                rollouts,
                grader=grader,
                sandbox=sandbox,
                project_dir=project_dir,
                cancellation=cancellation,
            )
        with ThreadPoolExecutor(
            max_workers=min(_MAX_RESPONSE_SCORE_WORKERS, len(rollouts)),
            thread_name_prefix="reward-lens-response-score",
        ) as workers:
            # Executor.map yields in input order even when contained subprocesses
            # finish out of order, so the saved response evidence remains exactly
            # aligned with the joined corpus.
            return tuple(workers.map(score_one, rollouts))
    finally:
        _profile("response_score_bank_end")
