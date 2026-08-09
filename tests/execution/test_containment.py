"""What the sandbox contains, observed through the mechanism rather than through the label.

Three claims are tested here, each by making the run try the thing:

* `network=False` is either enforced by a filter the tier names, or refused (RL0402). No tier
  accepts the flag and then ignores it.
* the artifact budget counts what the run wrote, not what it was handed, so a project staged as
  the working directory is not a breach on arrival.
* a read outside the allowed paths is blocked, and the tier in the `Result` is the tier whose
  mechanism the failure came from.

Nothing here reaches the network. The one connection attempt is to a listener this process owns
on loopback, and the point of it is that the run cannot reach even that.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from reward_lens.execution import Limits, default_sandbox, egress_mechanism, probe
from reward_lens.execution.errors import SandboxTierBelowRequired
from reward_lens.execution.limits import LANDLOCK_NET_ABI, require_egress_mechanism
from reward_lens.execution.linux import LINUX_TIERS, LinuxSandbox

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the Linux ladder")

#: The egress attempt the review asked for: a connect to loopback on a port nothing in the run can
#: reach, and a DNS lookup. It prints what happened rather than raising, so the assertions can name
#: the errno the mechanism produced.
EGRESS_SOURCE = """
import errno, socket, sys

port = int(sys.argv[1])
s = socket.socket()
s.settimeout(5)
try:
    s.connect(("127.0.0.1", port))
except OSError as exc:
    print("tcp", errno.errorcode.get(exc.errno, exc.errno))
else:
    print("tcp CONNECTED")
try:
    socket.getaddrinfo("one.one.one.one", 53)
except OSError as exc:
    print("dns", type(exc).__name__)
else:
    print("dns RESOLVED")
print("interfaces", " ".join(sorted(name for _index, name in socket.if_nameindex())))
"""

READ_SOURCE = """
import errno, sys

try:
    open(sys.argv[1], "rb").read()
except OSError as exc:
    print(type(exc).__name__, errno.errorcode.get(exc.errno, exc.errno))
else:
    print("READ")
"""


def _lines(result) -> dict[str, str]:
    return {
        line.split(" ", 1)[0]: line.split(" ", 1)[1] if " " in line else ""
        for line in result.stdout.decode().splitlines()
    }


# --- network=False: enforced, or refused ---------------------------------------------------------


def test_every_tier_either_filters_egress_or_refuses_the_flag(work: Path) -> None:
    """The invariant behind `default_sandbox()`: no tier accepts `network=False` and ignores it."""
    abi = probe().landlock_abi
    for tier in LINUX_TIERS:
        mechanism = egress_mechanism(tier, landlock_abi=abi)
        if mechanism is not None:
            assert mechanism.strip()
            continue
        with pytest.raises(SandboxTierBelowRequired):
            LinuxSandbox(tier=tier).run(
                [sys.executable, "-c", "pass"],
                limits=Limits(wall_s=10.0, cpu_s=10.0, network=False),
                cwd=work,
                env={},
            )


@pytest.mark.parametrize("tier", ["T0", "L0"])
def test_an_egress_attempt_at_t0_and_l0_is_refused_before_it_runs(tier: str, work: Path) -> None:
    """T0 and L0 have no egress filter, so the run that would try is refused by name."""
    assert egress_mechanism(tier, landlock_abi=probe().landlock_abi) is None
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        with pytest.raises(SandboxTierBelowRequired) as excinfo:
            LinuxSandbox(tier=tier).run_source(
                EGRESS_SOURCE,
                args=[str(port)],
                limits=Limits(wall_s=20.0, cpu_s=20.0, network=False),
                cwd=work,
                env={},
            )
    error = excinfo.value
    assert error.code == "RL0402"
    assert error.context["held"] == tier
    assert tier in str(error)


def test_the_landlock_tiers_have_no_egress_filter_before_abi_4() -> None:
    """L1 and L2 filter TCP through a Landlock ruleset, which has no network rights before ABI 4."""
    assert egress_mechanism("L1", landlock_abi=LANDLOCK_NET_ABI - 1) is None
    assert egress_mechanism("L1", landlock_abi=LANDLOCK_NET_ABI) is not None
    with pytest.raises(SandboxTierBelowRequired) as excinfo:
        require_egress_mechanism("L2", network=False, ladder=LINUX_TIERS, landlock_abi=1)
    assert excinfo.value.context["required"] == "L3"


def test_the_held_tier_blocks_the_egress_attempt(work: Path) -> None:
    """At the tier that held, the run cannot reach a listener this process owns on loopback."""
    held = probe().tier_held
    mechanism = egress_mechanism(held, landlock_abi=probe().landlock_abi)
    if mechanism is None:
        pytest.skip(f"the tier that held here is {held}, which has no egress filter")
    sandbox = default_sandbox()
    assert sandbox.tier == held
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        result = sandbox.run_source(
            EGRESS_SOURCE,
            args=[str(port)],
            limits=Limits(wall_s=25.0, cpu_s=25.0, network=False),
            cwd=work,
            env={},
        )
    assert result.exit_code == 0, result.stderr
    out = _lines(result)
    assert out["tcp"] != "CONNECTED"
    assert out["dns"] != "RESOLVED"
    assert result.detail["egress"] == mechanism
    assert result.tier == held
    if held == "L3":
        # The mechanism itself: a network namespace whose only interface is a loopback that is not
        # this machine's, which is why the listener above is unreachable rather than merely closed.
        assert out["interfaces"] == "lo"


def test_network_true_is_a_request_for_egress_and_is_not_refused(work: Path) -> None:
    """`network=True` claims nothing, so there is nothing to enforce and nothing to refuse."""
    result = LinuxSandbox(tier="T0").run(
        [sys.executable, "-c", "print('ran')"],
        limits=Limits(wall_s=15.0, cpu_s=15.0, network=True),
        cwd=work,
        env={},
    )
    assert result.stdout == b"ran\n"
    assert result.detail["egress"] == "network=True"


# --- the artifact budget is growth, not contents -------------------------------------------------


def _stage_project(root: Path, total_bytes: int) -> None:
    """A project of `total_bytes` apparent size, staged as the run's working directory (A-005)."""
    root.mkdir(parents=True, exist_ok=True)
    per_file = total_bytes // 4
    for index in range(4):
        with open(root / f"data-{index}.bin", "wb") as handle:
            handle.truncate(per_file)


def test_a_staged_project_is_the_baseline_not_a_breach(tmp_path: Path) -> None:
    """100 MiB staged as `cwd`, a grader that writes 1 KiB, a 1 MiB budget, and no breach."""
    project = tmp_path / "project"
    _stage_project(project, 100 * 2**20)
    limits = Limits(wall_s=25.0, cpu_s=25.0, artifact_bytes=1 << 20)
    sandbox = default_sandbox()
    result = sandbox.run(
        [sys.executable, "-c", "open('out.txt', 'wb').write(b'x' * 1024)"],
        limits=limits,
        cwd=project,
        env={},
    )
    assert result.exit_code == 0, result.stderr
    assert result.breach is None
    assert result.counters.disk_bytes < limits.artifact_bytes
    assert result.detail["disk_baseline_bytes"] >= 100 * 2**20


def test_growth_past_the_budget_is_still_an_artifact_breach(tmp_path: Path) -> None:
    """The same staged project, and a grader that writes past the budget on top of it."""
    project = tmp_path / "project"
    _stage_project(project, 100 * 2**20)
    limits = Limits(wall_s=25.0, cpu_s=25.0, artifact_bytes=1 << 20)
    source = (
        "for i in range(64):\n"
        "    open('grown-%d.bin' % i, 'wb').write(b'x' * (256 * 1024))\n"
        "import time; time.sleep(20)\n"
    )
    sandbox = default_sandbox()
    result = sandbox.run(
        [sys.executable, "-c", source], limits=limits, cwd=project, env={}
    )
    assert result.breach == "artifact"
    assert result.counters.disk_bytes > limits.artifact_bytes


# --- a read outside the allowed paths ------------------------------------------------------------


def test_a_read_outside_the_allowed_paths_is_blocked_by_landlock_at_l1(tmp_path: Path) -> None:
    """L1's mechanism is the ruleset: the path exists and the open is denied, EACCES."""
    if not probe().tiers["L1"].held:
        pytest.skip(f"L1 did not hold here: {probe().tiers['L1'].reason}")
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("not for the grader", encoding="utf-8")
    sandbox = LinuxSandbox(tier="L1")
    result = sandbox.run_source(
        READ_SOURCE,
        args=[str(outside)],
        limits=Limits(wall_s=20.0, cpu_s=20.0),
        cwd=work,
        env={},
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.decode().split() == ["PermissionError", "EACCES"]
    assert result.tier == "L1"


def test_a_read_outside_the_allowed_paths_is_blocked_at_the_tier_that_held(tmp_path: Path) -> None:
    """At the held tier the mechanism is asserted, not the label, and the label follows it."""
    held = probe().tier_held
    if held in ("T0", "L0"):
        pytest.skip(f"the tier that held here is {held}, which confines no read")
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("not for the grader", encoding="utf-8")
    sandbox = default_sandbox()
    result = sandbox.run_source(
        READ_SOURCE,
        args=[str(outside)],
        limits=Limits(wall_s=25.0, cpu_s=25.0),
        cwd=work,
        env={},
    )
    assert result.exit_code == 0, result.stderr
    error, code = result.stdout.decode().split()
    if held == "L3":
        # L3 binds the stated read roots and nothing else, so the path is not in the run's mount
        # namespace at all. That is a different mechanism from L1's denial, and a different errno.
        assert (error, code) == ("FileNotFoundError", "ENOENT")
    else:
        assert (error, code) == ("PermissionError", "EACCES")
    assert result.tier == held
