"""One supervisor for every operating system: the parent's clock, the parent's counters.

Section 7.2 puts the counters on the supervisor rather than the grader, and records them before
teardown, because a killed process reports nothing. So the loop below samples while the child is
alive: the size of its process group, the summed resident set, the summed CPU. A run that ends at
`SIGKILL` still comes back with counters.

Every dimension is enforced twice. The loop is the rule the parent applies; a kernel mechanism set
on the child is the backstop, and each backstop has a test that exceeds it. Neither is claimed
without the other.
"""

from __future__ import annotations

import os
import resource
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from reward_lens.contracts import Counters

from .errors import ExecutionLimitBreached
from .limits import Limits

__all__ = ["Result", "supervise"]

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_SAMPLE_S = 0.03
_DISK_EVERY = 8  # sample the working directory's size about four times a second

#: `ru_maxrss` is kilobytes on Linux and bytes on Darwin, and getting that wrong is a factor of
#: 1024 in the dimension a run is judged on.
_MAXRSS_UNIT = 1 if sys.platform == "darwin" else 1024


@dataclass
class Result:
    """What one sandboxed run produced, including the tier and operating system that held."""

    exit_code: int
    stdout: bytes
    stderr: bytes
    counters: Counters
    breach: str | None
    tier: str
    sandbox_os: str = "linux"
    probe: object | None = None
    argv: tuple[str, ...] = ()
    detail: dict[str, object] = field(default_factory=dict)

    def raise_for_breach(self) -> None:
        """RL0410 naming the dimension, for a caller that wants the refusal rather than the field."""
        if self.breach is None:
            return
        raise ExecutionLimitBreached(
            dimension=self.breach,
            limit=self.detail.get("limit"),
            observed=self.detail.get("observed"),
        )


def _sample_group(root_pid: int) -> tuple[int, int, float, int]:
    """(process count, RSS bytes, CPU seconds, address space bytes) over one grader's whole tree.

    Membership is the descendant tree as well as the process group, because bubblewrap's PID
    namespace puts the grader's children in a group of their own while leaving the parent link
    visible on the host. Counting only the group would undercount exactly at the tier that
    contains most.
    """
    rows: dict[int, tuple[int, int, int, float, int]] = {}
    try:
        entries = sorted(os.listdir("/proc"))
    except OSError:  # pragma: no cover - /proc is always there on Linux
        return 0, 0, 0.0, 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        close = data.rfind(b")")
        if close < 0:
            continue
        fields = data[close + 2 :].split()
        if len(fields) < 22:
            continue
        try:
            rows[int(entry)] = (
                int(fields[1]),
                int(fields[2]),
                int(fields[21]) * _PAGE,
                (int(fields[11]) + int(fields[12])) / _TICKS,
                int(fields[20]),
            )
        except ValueError:  # pragma: no cover - a process exiting mid-read
            continue
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgrp, _rss, _cpu, _vsz) in rows.items():
        children.setdefault(ppid, []).append(pid)
    members = {pid for pid, (_ppid, pgrp, _rss, _cpu, _vsz) in rows.items() if pgrp == root_pid}
    if root_pid in rows:
        members.add(root_pid)
    stack = sorted(members)
    while stack:
        pid = stack.pop()
        for kid in children.get(pid, ()):
            if kid not in members:
                members.add(kid)
                stack.append(kid)
    count = len(members)
    rss = sum(rows[pid][2] for pid in members if pid in rows)
    cpu = sum(rows[pid][3] for pid in members if pid in rows)
    vsz = sum(rows[pid][4] for pid in members if pid in rows)
    return count, rss, cpu, vsz


class _Reaper:
    """Reap the child here, so the kernel's own accounting for it survives the wait.

    `subprocess` reaps with `waitpid`, which throws the child's rusage away; what is left is
    `RUSAGE_CHILDREN`, a running total over every child this process has ever waited for, and a
    high-water mark set by an earlier run hides this one's. `os.wait4` returns `ru_maxrss` for
    this child and no other. That is what makes the memory rule a fact about the run rather than
    about whether a 30 ms sampler happened to look at the right moment, so this class takes the
    reaping over from `Popen` entirely: nothing here calls `proc.poll()` or `proc.wait()`.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self.rusage: resource.struct_rusage | None = None

    def poll(self, *, block: bool = False) -> int | None:
        """The exit code if the child has been reaped, else `None`. Sets `Popen.returncode`."""
        proc = self._proc
        if proc.returncode is not None:
            return proc.returncode
        try:
            pid, status, rusage = os.wait4(proc.pid, 0 if block else os.WNOHANG)
        except ChildProcessError:  # pragma: no cover - only if something else reaped it
            proc.returncode = -signal.SIGKILL
            return proc.returncode
        if pid == 0:
            return None
        self.rusage = rusage
        proc.returncode = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
        return proc.returncode

    @property
    def peak_rss_bytes(self) -> int:
        """The kernel's high-water mark for this child's resident set, zero if it was not reaped."""
        if self.rusage is None:  # pragma: no cover - the child is always reaped below
            return 0
        return int(self.rusage.ru_maxrss) * _MAXRSS_UNIT


def _disk_bytes(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def supervise(
    argv: list[str],
    *,
    limits: Limits,
    cwd: Path,
    env: dict[str, str],
    stdin: bytes | None,
    tier: str,
    sandbox_os: str,
    child_setup: Callable[[], None] | None,
    on_spawn: Callable[[int], None] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> Result:
    """Spawn, watch, and account for one child process group."""
    cwd = Path(cwd)
    # The artifact budget is what the run writes, not what it was handed: a project staged as the
    # working directory (A-005) is the baseline, and growth over it is the dimension.
    disk_baseline = _disk_bytes(cwd)
    started = time.monotonic()

    proc = subprocess.Popen(  # noqa: S603 - the argv is built by this package
        argv,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=pass_fds,
        start_new_session=True,
        restore_signals=True,
        preexec_fn=child_setup,  # noqa: PLW1509 - the supervisor is single threaded by design
    )
    pgid = proc.pid
    reaper = _Reaper(proc)
    if on_spawn is not None:
        on_spawn(proc.pid)

    if stdin is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            proc.stdin.close()

    peak_processes, peak_rss, cpu_s, peak_vsz = _sample_group(pgid)
    peak_processes = max(peak_processes, 1)
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    sizes = {"stdout": 0, "stderr": 0}
    breach: str | None = None
    detail: dict[str, object] = {}

    def record(dimension: str, limit: object, observed: object) -> None:
        nonlocal breach
        if breach is None:
            breach = dimension
            detail["limit"] = limit
            detail["observed"] = observed

    sel = selectors.DefaultSelector()
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is not None:
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, selectors.EVENT_READ, name)

    deadline = started + limits.wall_s
    ticks = 0
    while True:
        if reaper.poll() is not None and not sel.get_map():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            record("wall", limits.wall_s, round(time.monotonic() - started, 3))
            break
        for key, _mask in sel.select(timeout=min(_SAMPLE_S, remaining)):
            name = key.data
            try:
                data = key.fileobj.read()  # type: ignore[union-attr]
            except (BlockingIOError, InterruptedError):  # pragma: no cover - rare
                continue
            except OSError:  # pragma: no cover - the pipe went away
                data = b""
            if not data:
                sel.unregister(key.fileobj)
                continue
            room = limits.stdout_bytes - sizes[name]
            if room > 0:
                chunks[name].append(data[:room])
                sizes[name] += min(room, len(data))
            if sizes[name] + max(0, len(data) - max(room, 0)) > limits.stdout_bytes or len(
                data
            ) > room:
                record("stdout", limits.stdout_bytes, sizes[name] + len(data))

        ticks += 1
        count, rss, cpu, vsz = _sample_group(pgid)
        if count:
            peak_processes = max(peak_processes, count)
            peak_rss = max(peak_rss, rss)
            cpu_s = max(cpu_s, cpu)
            peak_vsz = max(peak_vsz, vsz)
        # One rule, on the dimension the counters carry: a resident set over budget. It is
        # reachable because RLIMIT_AS is set above the budget (`address_space_ceiling`), so the
        # allocator no longer ends the run before the supervisor can see it. This is the early
        # stop; the same rule is applied again after the wait to the kernel's own high-water mark
        # for the child, which no sampler can miss.
        if peak_rss > limits.memory_bytes:
            record("memory", limits.memory_bytes, peak_rss)
        if peak_processes > limits.processes:
            record("processes", limits.processes, peak_processes)
        if cpu_s > limits.cpu_s:
            record("cpu", limits.cpu_s, round(cpu_s, 3))
        if ticks % _DISK_EVERY == 0:
            grown = _disk_bytes(cwd) - disk_baseline
            if grown > limits.artifact_bytes:
                record("artifact", limits.artifact_bytes, grown)
        if breach is not None:
            break
        if reaper.poll() is not None and not sel.get_map():
            break

    # Teardown: the counters above are already recorded, which is the point of sampling.
    if reaper.poll() is None or breach is not None:
        _killpg(pgid, signal.SIGKILL)
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is None:
            continue
        try:
            rest = stream.read() or b""
        except (OSError, ValueError):  # pragma: no cover - already closed
            rest = b""
        room = limits.stdout_bytes - sizes[name]
        if room > 0 and rest:
            chunks[name].append(rest[:room])
            sizes[name] += min(room, len(rest))
        try:
            stream.close()
        except OSError:  # pragma: no cover
            pass
    sel.close()
    reaper.poll(block=True)  # SIGKILL is not refusable, so this returns
    # The direct child is reaped; anything it forked is still in the group and is not left behind.
    _killpg(pgid, signal.SIGKILL)

    wall_s = time.monotonic() - started
    child = reaper.rusage
    if child is not None:
        cpu_s = max(cpu_s, child.ru_utime + child.ru_stime)
        peak_rss = max(peak_rss, reaper.peak_rss_bytes)

    exit_code = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    # A wrapper such as bubblewrap reports its child's fatal signal as 128 + N rather than dying
    # of it, so both spellings of the same event name the same dimension.
    if exit_code in (-signal.SIGXCPU, 128 + int(signal.SIGXCPU)):
        record("cpu", limits.cpu_s, round(cpu_s, 3))
    if exit_code in (-signal.SIGXFSZ, 128 + int(signal.SIGXFSZ)):
        record("artifact", limits.artifact_bytes, "one file exceeded RLIMIT_FSIZE")

    disk = max(0, _disk_bytes(cwd) - disk_baseline)
    if disk > limits.artifact_bytes:
        record("artifact", limits.artifact_bytes, disk)

    stdout = b"".join(chunks["stdout"])
    stderr = b"".join(chunks["stderr"])

    # The outcome, not the sampler. `peak_rss` now carries the kernel's own high-water mark for
    # this child, so a grader that grew past the budget is named here whether or not a sample
    # caught it in the act. The second rule is the address-space backstop's only visible outcome:
    # `RLIMIT_AS` is refused mappings rather than delivering a signal, and a Python grader turns
    # that refusal into `MemoryError` and a non-zero exit. Both spellings name `memory`, so the
    # result is the same whichever fired.
    if peak_rss > limits.memory_bytes:
        record("memory", limits.memory_bytes, int(peak_rss))
    if exit_code != 0 and b"MemoryError" in stderr:
        record("memory", limits.memory_bytes, "the child hit the RLIMIT_AS backstop (MemoryError)")
    # The same shape for the file-size backstop: `RLIMIT_FSIZE` refuses the write with EFBIG as
    # well as raising SIGXFSZ, and a grader that catches the error exits with it in the traceback
    # having written the budget exactly, which growth alone does not exceed.
    if exit_code != 0 and b"File too large" in stderr:
        record("artifact", limits.artifact_bytes, "one file exceeded RLIMIT_FSIZE (EFBIG)")

    detail.setdefault("peak_vsz_bytes", int(peak_vsz))
    detail.setdefault("disk_baseline_bytes", int(disk_baseline))
    counters = Counters(
        calls=1,
        cpu_s=float(round(cpu_s, 3)),
        wall_s=float(round(wall_s, 3)),
        peak_rss_bytes=int(peak_rss),
        processes=int(peak_processes),
        disk_bytes=int(disk),
        output_bytes=len(stdout) + len(stderr),
        api_calls=0,
        input_tokens=0,
        output_tokens=0,
        usd="0.00",
    )
    return Result(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        counters=counters,
        breach=breach,
        tier=tier,
        sandbox_os=sandbox_os,
        argv=tuple(argv),
        detail=detail,
    )
