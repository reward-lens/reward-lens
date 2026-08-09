"""The Linux ladder, T0 through L3, behind one `Sandbox`.

The child side is a `preexec_fn` that runs in the forked child between `fork` and `exec` and applies
the tier's mechanisms in the only order that works: rlimits, then `PR_SET_NO_NEW_PRIVS` (Landlock
refuses to attach without it), then the filesystem and network ruleset, then the seccomp filter.
`RLIMIT_NPROC` is the one that needs arithmetic: it is enforced per uid, not per process, so a flat
64 on a machine where the owner already has three hundred processes would deny the grader its first
fork. The ceiling is therefore the owner's measured baseline plus the budget, and the number that
was set is recorded.
"""

from __future__ import annotations

import math
import os
import resource
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import landlock as _landlock
from . import seccomp as _seccomp
from .bwrap import bwrap_argv
from .errors import SandboxTierUnavailable
from .limits import (
    DEFAULT_LIMITS,
    Limits,
    address_space_ceiling,
    require_egress_mechanism,
    scrub_env,
)
from .scratch import stage_source
from .supervisor import Result, supervise

__all__ = ["LinuxSandbox", "LINUX_TIERS", "user_process_baseline"]

LINUX_TIERS = ("T0", "L0", "L1", "L2", "L3")

#: Read-only for every tier that has a ruleset at all. `/proc` is here because a grader that cannot
#: read `/proc/self` is a grader that cannot introspect its own limits, and reading it escapes
#: nothing; the containment claim is about writes and about the network.
BASE_READ_ROOTS = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc")

#: Character devices a grader may open. `/dev/null` is read-write; the entropy sources are not.
READ_WRITE_DEVICES = ("/dev/null", "/dev/zero")
READ_ONLY_DEVICES = ("/dev/urandom", "/dev/random")

_NOFILE_CEILING = 1024


def user_process_baseline() -> int:
    """How many processes this uid already owns, which is what `RLIMIT_NPROC` counts against."""
    uid = os.getuid()
    count = 0
    try:
        entries = sorted(os.listdir("/proc"))
    except OSError:  # pragma: no cover
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid == uid:
                count += 1
        except OSError:
            continue
    return count


class LinuxSandbox:
    """One sandbox per tier. The tier is what goes in the record, and never a guess."""

    def __init__(self, *, tier: str = "L0", abi: int | None = None) -> None:
        if tier not in LINUX_TIERS:
            raise SandboxTierUnavailable(
                tier=tier, probe="Linux ladder", reason=f"unknown Linux tier {tier}"
            )
        if sys.platform != "linux":  # pragma: no cover - the Linux ladder on Linux only
            raise SandboxTierUnavailable(
                tier=tier,
                probe="uname",
                reason=f"the Linux ladder is pending on {sys.platform}; it is not claimed here",
            )
        self.tier = tier
        self.sandbox_os = "linux"
        self._abi = _landlock.abi_version() if abi is None else abi

    # --- what the ruleset may read -------------------------------------------------------------

    def read_roots_for(self, executable: str) -> tuple[str, ...]:
        """The stated read set: the system roots plus the interpreter's own prefix.

        The write roots are not in here and do not need to be: a write root is a read root too,
        at every tier (`write_roots_for`).
        """
        return tuple(sorted(set(BASE_READ_ROOTS) | set(_landlock.interpreter_read_roots(executable))))

    # --- what the ruleset may write ------------------------------------------------------------

    def write_roots_for(self, cwd: Path, extra: Sequence[Path | str] = ()) -> tuple[str, ...]:
        """The stated write set: the working directory, plus any root the caller named.

        **A write root is read as well as written**, at every tier. At L1 and L2 the Landlock rule
        is `ACCESS_FS_WRITE`, which is `ACCESS_FS_READ` plus the write bits; at L3 bubblewrap
        binds each root read-write with `--bind`; at T0 and L0 nothing is confined at all. So a
        process can open a file staged for it in its own working directory, create a new one
        there, append to it and truncate it.

        `extra` is for the caller that stages machinery outside the directory the code under test
        runs in: a supervisor script, a request file, a capture file in a scratch directory whose
        child is the working directory. Without it such a path is in no root, and the run fails
        with EACCES at L1 and L2 and ENOENT at L3 (P-AUDIT-1's tier table).
        """
        roots = {str(Path(cwd))}
        roots.update(str(Path(one)) for one in extra)
        return tuple(sorted(roots))

    # --- the child side ------------------------------------------------------------------------

    def _child_setup(
        self,
        *,
        limits: Limits,
        cwd: Path,
        read_roots: tuple[str, ...],
        write_roots: tuple[str, ...],
    ):
        tier = self.tier
        abi = self._abi
        nofile = min(_NOFILE_CEILING, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
        nproc_ceiling = user_process_baseline() + max(1, limits.processes)
        cpu_hard = int(math.ceil(limits.cpu_s)) + 1
        as_ceiling = address_space_ceiling(limits.memory_bytes)
        work = tuple(write_roots)

        def setup() -> None:  # pragma: no cover - runs in the forked child
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (limits.artifact_bytes, limits.artifact_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
            if tier != "T0":
                # Above the budget on purpose: the supervisor's rule is what ends a run that grows,
                # and RLIMIT_AS is the backstop behind it (`address_space_ceiling`).
                resource.setrlimit(resource.RLIMIT_AS, (as_ceiling, as_ceiling))
                resource.setrlimit(resource.RLIMIT_CPU, (int(math.ceil(limits.cpu_s)), cpu_hard))
                if tier != "L3":
                    # RLIMIT_NPROC is what a new user namespace is charged against, so setting it
                    # here would make bubblewrap fail with EAGAIN before the grader starts. At L3
                    # the process ceiling is bubblewrap's PID namespace and the supervisor's count.
                    resource.setrlimit(resource.RLIMIT_NPROC, (nproc_ceiling, nproc_ceiling))
                _seccomp.set_no_new_privs()
            if tier in ("L1", "L2"):
                _landlock.apply_ruleset(
                    read_roots=read_roots,
                    write_roots=work,
                    read_write_files=READ_WRITE_DEVICES + READ_ONLY_DEVICES,
                    allow_net=limits.network,
                    abi=abi,
                )
            if tier == "L2":
                _seccomp.install_filter()

        return setup

    # --- the protocol --------------------------------------------------------------------------

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits = DEFAULT_LIMITS,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
        write_roots: Sequence[Path | str] = (),
    ) -> Result:
        cwd = Path(cwd)
        egress = require_egress_mechanism(
            self.tier, network=limits.network, ladder=LINUX_TIERS, landlock_abi=self._abi
        )
        read_roots = self.read_roots_for(argv[0] if argv else sys.executable)
        write = self.write_roots_for(cwd, write_roots)
        child_env = scrub_env(env, cwd=cwd)
        launch = list(argv)
        pass_fds: tuple[int, ...] = ()
        seccomp_file = None
        if self.tier == "L3":
            seccomp_file = tempfile.NamedTemporaryFile(prefix="rl-seccomp-", suffix=".bpf")
            prog = _seccomp.build_filter()
            seccomp_file.write(b"".join(bytes(ins) for ins in prog))
            seccomp_file.flush()
            seccomp_file.seek(0)
            pass_fds = (seccomp_file.fileno(),)
            launch = bwrap_argv(
                launch,
                cwd=cwd,
                limits=limits,
                read_roots=read_roots,
                write_roots=write,
                seccomp_fd=seccomp_file.fileno(),
                env=child_env,
            )
        setup = self._child_setup(
            limits=limits, cwd=cwd, read_roots=read_roots, write_roots=write
        )
        try:
            result = supervise(
                launch,
                limits=limits,
                cwd=cwd,
                env=child_env,
                stdin=stdin,
                tier=self.tier,
                sandbox_os="linux",
                child_setup=setup,
                pass_fds=pass_fds,
            )
        finally:
            if seccomp_file is not None:
                seccomp_file.close()
        result.detail.setdefault("rlimit_nproc", user_process_baseline() + max(1, limits.processes))
        result.detail.setdefault("read_roots", read_roots)
        result.detail.setdefault("write_roots", write)
        result.detail.setdefault("rlimit_as", address_space_ceiling(limits.memory_bytes))
        # What kept `network=False` on this run, in the mechanism's own words, or the fact that
        # egress was asked for. A record that says "no network" says what said no.
        result.detail.setdefault("egress", egress if egress is not None else "network=True")
        return result

    def run_source(
        self,
        source: str,
        *,
        args: list[str] | tuple[str, ...] = (),
        limits: Limits = DEFAULT_LIMITS,
        cwd: Path,
        env: dict[str, str] | None = None,
        write_roots: Sequence[Path | str] = (),
    ) -> Result:
        """Write the source into a fresh 0700 scratch directory under `cwd` and run it there.

        A convenience for the tests and for callers holding a particular tier, not part of the
        `Sandbox` interface: `run_python` stages the source itself and calls `run`, so nothing in
        the product needs a sandbox to have this method.

        `PYTHONHASHSEED=0` is set by the environment of a fresh interpreter, which is the only
        place it can be set: in process it is already too late (D-38).
        """
        scratch, main = stage_source(source, cwd=Path(cwd))
        argv = [sys.executable, "-B", "-s", str(main), *[str(a) for a in args]]
        return self.run(
            argv,
            limits=limits,
            cwd=scratch,
            env=dict(env or {}),
            stdin=None,
            write_roots=write_roots,
        )
