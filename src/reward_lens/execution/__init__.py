"""Execution, containment and counters.

Containment is a measured property of the run, not a promise in the docs (D-37). Every grader call
in the product goes through this module, and the tier it records is the one the probe established
on that machine.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from reward_lens.contracts import Counters

from .errors import ExecutionLimitBreached, SandboxTierBelowRequired, SandboxTierUnavailable
from .limits import DEFAULT_LIMITS, Limits, egress_mechanism, scrub_env
from .probe import SandboxProbe, TierResult, default_sandbox, probe, tier_ladder
from .scratch import stage_source
from .supervisor import Result

__all__ = [
    "Limits",
    "DEFAULT_LIMITS",
    "Counters",
    "SandboxProbe",
    "TierResult",
    "probe",
    "tier_ladder",
    "Sandbox",
    "Result",
    "default_sandbox",
    "run_python",
    "scrub_env",
    "egress_mechanism",
    "SandboxTierUnavailable",
    "SandboxTierBelowRequired",
    "ExecutionLimitBreached",
]


@runtime_checkable
class Sandbox(Protocol):
    """One protocol for every tier and every operating system, so a provider is never a dependency.

    An out-of-tree plugin (Podman, gVisor, a microVM service) implements this and nothing else.
    """

    tier: str

    def run(
        self,
        argv: list[str],
        *,
        limits: Limits,
        cwd: Path,
        env: dict[str, str],
        stdin: bytes | None = None,
        write_roots: Sequence[Path | str] = (),
    ) -> Result: ...


def run_python(
    source: str,
    *,
    args: list[str] | tuple[str, ...] = (),
    limits: Limits = DEFAULT_LIMITS,
    cwd: Path,
    env: dict[str, str] | None = None,
    write_roots: Sequence[Path | str] = (),
) -> Result:
    """Run one piece of Python source under the tier that held, in a fresh scratch directory.

    The interpreter is a new process with `PYTHONHASHSEED=0` in its environment, because setting it
    in process is too late (D-38), and with an environment no key reaches.

    Staging the source is done here rather than by the sandbox, so that this function needs
    nothing from a provider but `Sandbox.run`: the Protocol below stays at one method, and an
    out-of-tree plugin implements that one method and is usable from here.
    """
    sandbox = default_sandbox()
    scratch, main = stage_source(source, cwd=Path(cwd))
    argv = [sys.executable, "-B", "-s", str(main), *[str(a) for a in args]]
    result = sandbox.run(
        argv,
        limits=limits,
        cwd=scratch,
        env=dict(env or {}),
        stdin=None,
        write_roots=write_roots,
    )
    result.probe = probe()
    return result
