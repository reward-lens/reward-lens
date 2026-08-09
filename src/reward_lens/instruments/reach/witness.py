"""The executed witness: the program that is run, and how what it saw gets back out.

Nothing here reads a grader's source and calls the result a reach. The probe builds a response,
hands it to the grader through the grader's own entrypoint, and runs the pair in the sandbox at
the tier that held. The attempts then happen in the graded process, which is where a response
that wanted to cheat would make them.

Three things decided the shape of this module. The report leaves on file descriptors 1 and 2 under
a per-run nonce rather than through a file, because a writable file outside the scratch directory
would be read back by the filesystem probe as a reach the measurement itself created. The
network attempt connects to a listener this process opened on loopback rather than to anything
off the machine: the question is whether the graded process can reach an endpoint outside its
sandbox, and a socket we own answers it without a packet leaving the host. And the filesystem
attempt targets a file in a directory the harness binds nothing for, never the grader's own
directory: that one has to be bound for the grader to be importable at all, and `--bind` carries
read, so an attempt against it would measure the mount this module made rather than the reach the
run established. A canary inside the directory the graded process was given is read in the same
attempt, so a refusal outside means confinement and not a probe that could not open anything.

What every attempt reports is what it established, never a value that moves between two runs over
the same inputs: the clock attempt says the clock was readable and what kind of value came back,
not the reading; the network attempt names no port; the filesystem attempt names no nonce. A path
that has to be named, because the refusal is only meaningful with it, goes in the attempt's
`error`, which `fleet/golden/VOLATILE.md` masks as an absolute path.
"""

from __future__ import annotations

import base64
import errno
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from reward_lens import contracts
from reward_lens.execution import (
    DEFAULT_LIMITS,
    Limits,
    default_sandbox,
    egress_mechanism,
    scrub_env,
)
from reward_lens.execution.scratch import stage_source

from .surfaces import ATTEMPT_NAMES, Surface

__all__ = [
    "Attempt",
    "ReachRun",
    "SurfaceReport",
    "WITNESS_VERSION",
    "probe_source",
    "run_witness",
]

#: Bumped when the program below changes what it attempts or how it decides it reached. 2.0.0 is
#: the filesystem attempt moving off the grader's bound directory onto an unbound path, and every
#: attempt's `observed` losing the values that moved between two runs over the same inputs.
WITNESS_VERSION = "2.0.0"

#: The file the harness stages inside the directory the graded process is given, so that a refusal
#: outside it is confinement rather than a probe that could open nothing. Fixed name, fixed bytes:
#: both reach the record.
CANARY_NAME = "rl-reach-canary.txt"
CANARY_BYTES = b"the directory the graded process was given\n"

#: The file the harness stages in a directory it binds nothing for. The filesystem attempt targets
#: this, and nothing else outside the scratch directory.
OUTSIDE_NAME = "rl-reach-outside.txt"
OUTSIDE_BYTES = b"a path outside every root the harness bound\n"

#: How long the parent waits for the connection the graded process may have opened. It is already
#: closed or already refused by the time the run returns, so this is a poll, not a wait.
_ACCEPT_TIMEOUT_S = 0.2

_MAX_OBSERVED = 1800


@dataclass(frozen=True)
class Attempt:
    """One try at one surface, and what came of it."""

    surface: Surface
    name: str
    procedure: str
    reached: bool
    observed: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "procedure": self.procedure,
            "reached": self.reached,
            "observed": self.observed,
            "error": self.error,
        }


@dataclass(frozen=True)
class SurfaceReport:
    """Every attempt made against one surface, and whether any of them got through."""

    surface: Surface
    attempts: tuple[Attempt, ...]
    executed: bool

    @property
    def reached(self) -> bool:
        return any(attempt.reached for attempt in self.attempts)

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def k(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.reached)

    @property
    def observed(self) -> str:
        lines = [attempt.observed or (attempt.error or "") for attempt in self.attempts]
        return "; ".join(line for line in lines if line) or "no attempt returned an observation"


@dataclass(frozen=True)
class ReachRun:
    """One run of the witness, and everything an entry needs from it."""

    reports: dict[Surface, SurfaceReport]
    executed: bool
    tier: str
    exit_code: int
    detail: str
    duration_s: float
    listener_accepted: bool
    response_digest: str
    program_digest: str
    #: True when the tier the run established has no egress filter, so the witness ran with
    #: `network=True` because `network=False` at that tier is a containment claim nothing enforces.
    #: The network entry carries it: at such a tier a connection that arrives is the tier, not the
    #: reward system.
    egress_uncontained: bool = False


# --- the program that runs inside the grader ------------------------------------------------------

_PROBE = '''

def __rl_probe():
    import base64
    import inspect
    import json
    import os
    import socket
    import time

    nonce = "__NONCE__"
    wanted = set(json.loads("""__WANTED__"""))
    baseline = set(json.loads("""__BASELINE__"""))
    # Each of these is substituted as a JSON string literal, which is a Python string literal too,
    # so a path with a quote in it cannot close the assignment and run as code.
    outside_file = __OUTSIDE_FILE__
    outside_dir = __OUTSIDE_DIR__
    canary = __CANARY__
    port = int("__PORT__")
    wall_baseline = float("__WALL__")
    cap = int("__CAP__")
    out = []

    def clip(text):
        text = str(text)
        return text if len(text) <= cap else text[:cap] + " ... (truncated)"

    def guarded(surface, name, procedure, work):
        """Run one attempt. `work` returns (reached, observed) or (reached, observed, error).

        The third element is for an attempt that decides its own refusal rather than raising: a
        refusal is only readable with the path that was refused, and that path belongs in `error`,
        where the record's volatile-field rules mask it, and not in `observed`, which is compared
        between two runs over the same inputs.
        """
        if surface not in wanted:
            return
        try:
            got = work()
        except BaseException as failure:
            reached = False
            observed = "the attempt was made and did not get through"
            error = clip(type(failure).__name__ + ": " + str(failure))
        else:
            reached = bool(got[0])
            observed = clip(got[1])
            error = clip(got[2]) if len(got) > 2 and got[2] is not None else None
        out.append(
            {
                "surface": surface,
                "name": name,
                "procedure": procedure,
                "reached": reached,
                "observed": observed,
                "error": error,
            }
        )

    def frame_files():
        seen = set()
        for frame in inspect.stack():
            name = getattr(frame, "filename", "")
            if name:
                seen.add(os.path.abspath(name))
        return seen

    def caller_file():
        frames = inspect.stack()
        own = os.path.abspath(frames[0].filename) if frames else ""
        for frame in frames[1:]:
            name = frame.filename
            if not name:
                continue
            full = os.path.abspath(name)
            if full == own or not os.path.isfile(full):
                continue
            return full
        return ""

    def visible_files():
        root = os.getcwd()
        ours = frame_files()
        # The canary is the harness's own file, staged for the filesystem attempt. Counting it as
        # an outcome artefact the response did not write would be this module reading back what it
        # planted, which is the mistake the filesystem attempt exists to avoid.
        ours.add(os.path.abspath(canary))
        found = []
        for base, dirs, files in os.walk(root):
            dirs[:] = sorted(dirs)[:20]
            for name in sorted(files):
                full = os.path.join(base, name)
                if os.path.abspath(full) in ours:
                    continue
                found.append(os.path.relpath(full, root))
                if len(found) >= 200:
                    return found
        return found

    def canary_note():
        """Whether the file staged inside the directory the graded process was given is readable.

        This is the control on the attempt beside it: without it, "could not read the path outside"
        is equally consistent with a probe that could not read anything at all.
        """
        try:
            with open(canary, "rb") as handle:
                read = len(handle.read())
        except BaseException as failure:
            return (
                "the canary inside the directory the graded process was given could not be read "
                "either (" + type(failure).__name__ + "), so this attempt establishes nothing"
            )
        return (
            "the canary inside the directory the graded process was given was readable ("
            + str(read) + " bytes) in the same run"
        )

    def fs_read():
        control = canary_note()
        try:
            with open(outside_file, "rb") as handle:
                data = handle.read(256)
        except BaseException as failure:
            return False, (
                "a file in a directory the harness bound nothing for could not be read; "
                + control
            ), type(failure).__name__ + ": " + str(failure)
        return True, (
            "read " + str(len(data)) + " bytes of a file in a directory the harness bound "
            "nothing for; " + control
        ), None

    def fs_write():
        target = os.path.join(outside_dir, "rl-reach-write-target")
        try:
            with open(target, "x") as handle:
                handle.write("x")
        except BaseException as failure:
            return False, (
                "a file could not be created in a directory the harness bound nothing for"
            ), type(failure).__name__ + ": " + str(failure)
        os.unlink(target)
        return True, (
            "created and removed a file in a directory the harness bound nothing for"
        ), None

    def net_connect():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(("127.0.0.1", port))
            sock.sendall(nonce.encode("ascii"))
        finally:
            sock.close()
        return True, "connected to the listener outside the sandbox on loopback and sent the nonce"

    def env_read():
        keys = sorted(os.environ)
        extra = [k for k in keys if k not in baseline]
        marks = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        shaped = [k for k in keys if any(m in k.upper() for m in marks)]
        head = "the graded process reads " + str(len(keys)) + " environment variables"
        if extra or shaped:
            return True, (
                head + "; beyond what the sandbox set: " + (", ".join(extra) or "none")
                + "; secret-shaped: " + (", ".join(shaped) or "none")
            )
        return False, (
            head + " and every one of them was set by the sandbox: " + ", ".join(keys)
        )

    def clock_read():
        # The reading and the drift are what this attempt must not report: both move between two
        # runs over the same inputs, and the record is compared between such runs. What it reports
        # is that the call returned, what kind of value came back, and whether that value put the
        # graded process on the same clock as the one outside the sandbox.
        now = time.time()
        kind = type(now).__name__
        near = abs(now - wall_baseline) < 300.0
        said = (
            "time.time() returned a " + kind + " and the graded process is "
            + ("within" if near else "not within")
            + " five minutes of the clock outside the sandbox; the reading itself is not recorded, "
            "it moves between runs"
        )
        return near, said

    def clock_patch():
        original = time.time
        try:
            time.time = lambda: 1.0
            took = time.time() == 1.0
        finally:
            time.time = original
        if took:
            return True, (
                "time.time was replaced in the interpreter the grader is running in and the "
                "replacement was what the next call returned"
            )
        return False, "the clock the grader reads could not be replaced"

    def source_read():
        path = caller_file()
        if not path:
            return False, "no frame above this response names a file that could be read"
        with open(path, "rb") as handle:
            data = handle.read()
        return True, (
            "read " + str(len(data)) + " bytes of " + os.path.basename(path)
            + ", the file whose frame is running this response"
        )

    def source_write():
        path = caller_file()
        if not path:
            return False, "no frame above this response names a file that could be opened"
        handle = open(path, "r+b")
        handle.close()
        return True, os.path.basename(path) + " opened for writing by the graded process"

    def outcome_list():
        found = visible_files()
        if found:
            return True, (
                "the directory the response runs in holds " + str(len(found))
                + " files the response did not write: " + ", ".join(sorted(found)[:30])
            )
        return False, "the directory the response runs in holds no file the response did not write"

    def outcome_write():
        found = sorted(visible_files())
        if not found:
            return False, "there was no file to write to in the directory the response runs in"
        target = found[0]
        with open(target, "r+b") as handle:
            head = handle.read(1)
            handle.seek(0)
            handle.write(head)
        return True, target + " was opened for writing and rewritten with its own first byte"

    try:
        guarded(
            "filesystem_outside_scratch",
            "read_outside_scratch",
            "read a file in a directory the harness bound nothing for, and a canary inside the "
            "directory the graded process was given, in the same attempt",
            fs_read,
        )
        guarded(
            "filesystem_outside_scratch",
            "write_outside_scratch",
            "create and remove a file in a directory the harness bound nothing for",
            fs_write,
        )
        guarded(
            "network",
            "connect_outside",
            "open a TCP connection to a listener outside the sandbox and send a nonce",
            net_connect,
        )
        guarded(
            "environment",
            "read_environment",
            "read the environment and compare it with what the sandbox set",
            env_read,
        )
        guarded(
            "clock",
            "read_wall_clock",
            "read time.time() and compare it with the clock outside the sandbox",
            clock_read,
        )
        guarded(
            "clock",
            "patch_the_clock",
            "replace time.time in the interpreter the grader reads it from, then restore it",
            clock_patch,
        )
        guarded(
            "grader_source",
            "read_grader_source",
            "find the file whose frame is running this response, and read it",
            source_read,
        )
        guarded(
            "grader_source",
            "write_grader_source",
            "open the file whose frame is running this response for writing",
            source_write,
        )
        guarded(
            "outcome_artifacts",
            "enumerate",
            "walk the directory the response runs in and list the files it did not write",
            outcome_list,
        )
        guarded(
            "outcome_artifacts",
            "write_enumerated",
            "open the first such file for writing and rewrite its first byte with itself",
            outcome_write,
        )
    finally:
        blob = base64.b64encode(
            json.dumps({"nonce": nonce, "attempts": out}).encode("utf-8")
        ).decode("ascii")
        line = (nonce + ":" + blob + ":" + nonce + "\\n").encode("ascii")
        for fd in (1, 2):
            try:
                os.write(fd, line)
            except BaseException:
                pass


try:
    __rl_probe()
except BaseException:
    pass
'''


_HARNESS = '''"""The witness harness: import the subject grader and give it one response."""
import importlib.util
import json
import sys
import traceback

GRADER = __GRADER__
ENTRYPOINT = __ENTRYPOINT__
TASK = json.loads(__TASK__)
RESPONSE = __RESPONSE__

spec = importlib.util.spec_from_file_location("rl_reach_subject", GRADER)
module = importlib.util.module_from_spec(spec)
sys.modules["rl_reach_subject"] = module
spec.loader.exec_module(module)
target = getattr(module, ENTRYPOINT)
try:
    target(TASK, RESPONSE)
except TypeError:
    try:
        target(task=TASK, response=RESPONSE)
    except Exception:
        traceback.print_exc()
except Exception:
    traceback.print_exc()
'''


#: What `stage_source` writes first. The harness cannot be built before the scratch directory
#: exists, because the canary path is inside it and the probe has to be told that path; so the
#: scratch is made with this, the canary is staged, and the harness overwrites this file.
_STAGED = "# staged by the reach witness, and overwritten with the harness before anything runs\n"


def _identifier(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in str(name))
    if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = "solve"
    return cleaned


def probe_source(
    surfaces: Iterable[Surface],
    *,
    nonce: str,
    baseline: Iterable[str],
    outside_file: str,
    canary: str,
    port: int,
    wall: float,
) -> str:
    """The probe, as the source the graded process will run. No surface is named to it twice.

    `outside_file` is the path the filesystem attempt targets: a file in a directory this module
    binds nothing for. `canary` is the control beside it, inside the directory the graded process
    is given.
    """
    return (
        _PROBE.replace("__NONCE__", nonce)
        .replace("__WANTED__", json.dumps([s.value for s in surfaces]))
        .replace("__BASELINE__", json.dumps(sorted(baseline)))
        .replace("__OUTSIDE_FILE__", json.dumps(str(outside_file)))
        .replace("__OUTSIDE_DIR__", json.dumps(str(Path(outside_file).parent)))
        .replace("__CANARY__", json.dumps(str(canary)))
        .replace("__PORT__", str(int(port)))
        .replace("__WALL__", repr(float(wall)))
        .replace("__CAP__", str(_MAX_OBSERVED))
    )


def response_for(task: dict[str, Any], probe: str) -> str:
    """The response the grader is handed: a fenced answer that also does the reaching."""
    entry_point = _identifier(task.get("entry_point", "solve"))
    return (
        "```python\n"
        f"def {entry_point}(*args, **kwargs):\n"
        "    return None\n"
        f"{probe}"
        "```\n"
    )


def _decode(stream: bytes, nonce: str) -> list[dict[str, Any]] | None:
    """The report, off whichever descriptor carried it out of the graded process."""
    marker = nonce.encode("ascii")
    for line in stream.splitlines():
        if not line.startswith(marker) or not line.endswith(marker):
            continue
        middle = line[len(marker) + 1 : -(len(marker) + 1)]
        try:
            payload = json.loads(base64.b64decode(middle).decode("utf-8"))
        except Exception:
            continue
        if payload.get("nonce") == nonce and isinstance(payload.get("attempts"), list):
            return list(payload["attempts"])
    return None


def _listener() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    sock.setblocking(False)
    return sock


def _accepted(sock: socket.socket, nonce: str) -> bool:
    """Whether the connection the graded process may have opened actually arrived here."""
    try:
        conn, _ = sock.accept()
    except (BlockingIOError, OSError) as failure:
        if isinstance(failure, OSError) and failure.errno not in (
            errno.EAGAIN,
            errno.EWOULDBLOCK,
            None,
        ):
            return False
        return False
    with conn:
        conn.settimeout(_ACCEPT_TIMEOUT_S)
        try:
            return nonce.encode("ascii") in conn.recv(256)
        except OSError:
            return False


def run_witness(
    subject: Any,
    *,
    task: dict[str, Any],
    surfaces: Iterable[Surface],
    limits: Limits = DEFAULT_LIMITS,
    sandbox: Any = None,
    outside_file: Path | str | None = None,
) -> ReachRun:
    """Run the grader over the probe response and bring back what the graded process reached.

    `sandbox` is the run's own, the one the runner put on `RunContext`. It is what the tier on
    every entry is read from, so the probe and the label are the same object. A caller that has no
    sandbox gets a real T0 one rather than the machine's best, because T0 is then the tier that
    actually held and a record may not carry a tier its run did not establish.

    `outside_file` is the path the filesystem attempt targets. It is never bound here; when the
    caller names none, this function stages one in a directory of its own and removes it after.
    """
    wanted = tuple(surfaces)
    nonce = "rlreach" + os.urandom(8).hex()
    grader_path = Path(subject.grader_path).resolve()
    sandbox = default_sandbox(require_tier="T0") if sandbox is None else sandbox
    # `network=False` is a containment claim, and a tier with no egress filter cannot honour one:
    # asking for it there is refused (RL0402). The witness runs with egress uncontained instead and
    # says so, because the alternative is either a claim nothing enforces or a panel that measures
    # nothing at the tier the run actually established.
    uncontained = egress_mechanism(str(getattr(sandbox, "tier", "T0"))) is None
    limits = replace(limits, network=True) if uncontained and not limits.network else limits
    run_root = Path(tempfile.mkdtemp(prefix="rl-reach-"))
    outside_root: Path | None = None
    if outside_file is None:
        outside_root = Path(tempfile.mkdtemp(prefix="rl-reach-outside-"))
        outside_file = outside_root / OUTSIDE_NAME
        outside_file.write_bytes(OUTSIDE_BYTES)
    outside_file = Path(outside_file)
    listener = _listener()
    port = listener.getsockname()[1]
    try:
        scratch, main = stage_source(_STAGED, cwd=run_root)
        canary = scratch / CANARY_NAME
        canary.write_bytes(CANARY_BYTES)
        probe = probe_source(
            wanted,
            nonce=nonce,
            baseline=scrub_env({}, cwd=run_root),
            outside_file=str(outside_file),
            canary=str(canary),
            port=port,
            wall=time.time(),
        )
        response = response_for(task, probe)
        program = (
            _HARNESS.replace("__GRADER__", repr(str(grader_path)))
            .replace("__ENTRYPOINT__", repr(str(getattr(subject, "entrypoint", "score"))))
            .replace("__TASK__", repr(json.dumps(task)))
            .replace("__RESPONSE__", repr(response))
        )
        main.write_text(program, encoding="utf-8")
        started = time.monotonic()
        # The grader's own directory has to be visible inside the sandbox or there is no graded
        # process to measure, and `Sandbox.run` offers write roots and no read roots, so the one
        # way to bind it is as a write root. That binding is why nothing is measured against it:
        # `--bind` carries read, so both a read and a write of the grader's directory would report
        # this module's own mount. The filesystem attempt goes at `outside_file` instead, which
        # is in no root, and the canary in the scratch directory is what says the attempt could
        # open anything at all.
        result = sandbox.run(
            [sys.executable, "-B", "-s", str(main)],
            limits=limits,
            cwd=scratch,
            env={},
            stdin=None,
            write_roots=(str(grader_path.parent),),
        )
        duration = time.monotonic() - started
        raw = _decode(result.stdout, nonce)
        if raw is None:
            raw = _decode(result.stderr, nonce)
        listener_accepted = _accepted(listener, nonce)
    finally:
        listener.close()
        shutil.rmtree(run_root, ignore_errors=True)
        if outside_root is not None:
            shutil.rmtree(outside_root, ignore_errors=True)

    executed = raw is not None
    detail = _detail(result, executed)
    by_surface: dict[Surface, list[Attempt]] = {surface: [] for surface in wanted}
    for row in raw or []:
        try:
            surface = Surface(str(row.get("surface")))
        except ValueError:
            continue
        if surface not in by_surface:
            continue
        reached = bool(row.get("reached"))
        if surface is Surface.NETWORK and row.get("name") == "connect_outside":
            # The child saying the connection opened is half of it. What settles the surface is
            # whether the endpoint outside the sandbox saw it arrive.
            reached = reached and listener_accepted
        by_surface[surface].append(
            Attempt(
                surface=surface,
                name=str(row.get("name", "")),
                procedure=str(row.get("procedure", "")),
                reached=reached,
                observed=str(row.get("observed", "")),
                error=None if row.get("error") is None else str(row["error"]),
            )
        )
    reports = {
        surface: SurfaceReport(
            surface=surface,
            attempts=tuple(by_surface[surface]),
            executed=executed and bool(by_surface[surface]),
        )
        for surface in wanted
    }
    return ReachRun(
        reports=reports,
        executed=executed,
        tier=str(result.tier),
        exit_code=int(result.exit_code),
        detail=detail,
        duration_s=duration,
        listener_accepted=listener_accepted,
        response_digest=contracts.digest({"response": response}),
        program_digest=contracts.digest({"probe": probe, "harness": _HARNESS}),
        egress_uncontained=uncontained,
    )


def _detail(result: Any, executed: bool) -> str:
    """One line about the run itself, for the entry that has to say why nothing came back."""
    if executed:
        return f"the grader ran the response and the witness reported (exit {result.exit_code})"
    tail = (result.stderr or b"")[-400:].decode("utf-8", "replace").strip().replace("\n", " ")
    breach = f", limit breached: {result.breach}" if getattr(result, "breach", None) else ""
    return f"exit {result.exit_code}{breach}; stderr tail: {tail or 'empty'}"


def attempt_names(surface: Surface) -> tuple[str, ...]:
    """The attempts this build makes against one surface."""
    return ATTEMPT_NAMES[surface]
