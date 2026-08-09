"""The probe, the platform matrix, and what `--require-tier` refuses.

D-37's rule is that a tier label in a record is one the probe established on that machine. These
tests read the probe rather than the constants, and the two refusals are exercised against a probe
result a test constructs, so the refusal is proved on a machine where the tier does hold.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import pytest

from reward_lens.execution import Limits, SandboxProbe, default_sandbox, probe, run_python
from reward_lens.execution.bwrap import BWRAP_PROBE_ARGV, probe_bwrap
from reward_lens.execution.errors import SandboxTierBelowRequired, SandboxTierUnavailable
from reward_lens.execution.probe import TierResult, tier_ladder


def test_probe_is_cached_per_process() -> None:
    first = probe()
    assert probe() is first


def test_probe_carries_the_os_and_the_kernel() -> None:
    p = probe()
    assert p.os in {"linux", "macos", "windows"}
    assert p.os == {"linux": "linux", "darwin": "macos", "win32": "windows"}[sys.platform]
    if p.os == "linux":
        assert p.kernel and p.kernel[0].isdigit()


def test_every_tier_of_this_os_has_a_result_with_a_reason_and_a_duration() -> None:
    p = probe()
    assert set(p.tiers) == set(tier_ladder(p.os))
    for name, t in sorted(p.tiers.items()):
        assert isinstance(t.held, bool)
        assert t.reason, name
        assert t.ms >= 0.0


def test_t0_always_holds() -> None:
    assert probe().tiers["T0"].held


def test_the_tier_held_is_the_top_of_an_unbroken_ladder() -> None:
    p = probe()
    ladder = tier_ladder(p.os)
    held_through = 0
    for i, name in enumerate(ladder):
        if p.tiers[name].held:
            held_through = i
        else:
            break
    assert p.tier_held == ladder[held_through]


def test_the_platform_matrix_marks_the_other_operating_systems_pending() -> None:
    """The macOS and Windows paths exist; nothing claims them on a Linux box."""
    p = probe()
    assert set(p.platform_matrix) == {"linux", "macos", "windows"}
    for os_name, tiers in sorted(p.platform_matrix.items()):
        for tier, status in sorted(tiers.items()):
            if os_name == p.os:
                assert status in {"held", "unavailable"}, (os_name, tier)
            else:
                assert status == "pending", (os_name, tier)


def test_the_bwrap_probe_is_the_documented_command_and_runs_under_100ms() -> None:
    assert BWRAP_PROBE_ARGV == ["bwrap", "--unshare-all", "--ro-bind", "/", "/", "/bin/true"]
    start = time.monotonic()
    held, reason, ms = probe_bwrap()
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 100.0, f"the bubblewrap probe took {elapsed_ms:.1f} ms"
    assert ms <= elapsed_ms + 1.0
    assert isinstance(held, bool)
    assert reason


def test_l3_is_recorded_from_its_probe_not_assumed() -> None:
    p = probe()
    if p.os != "linux":
        pytest.skip("the L3 row is Linux")
    held, reason, _ms = probe_bwrap()
    assert p.tiers["L3"].held == (held and p.tiers["L2"].held)
    if not held:
        assert reason in p.tiers["L3"].reason


def test_default_sandbox_returns_the_tier_that_held(work: Path) -> None:
    s = default_sandbox()
    assert s.tier == probe().tier_held
    r = s.run(["/bin/echo", "ok"], limits=Limits(wall_s=10.0), cwd=work, env={})
    assert r.tier == s.tier
    assert r.stdout == b"ok\n"


def test_require_tier_at_or_below_the_held_tier_is_accepted() -> None:
    p = probe()
    assert default_sandbox("T0").tier == "T0"
    assert default_sandbox(p.tier_held).tier == p.tier_held


def _probe_with(tier_held: str, tiers: dict[str, tuple[bool, str]]) -> SandboxProbe:
    return SandboxProbe(
        os="linux",
        tier_held=tier_held,
        tiers={
            name: TierResult(held=held, reason=reason, ms=0.0)
            for name, (held, reason) in tiers.items()
        },
        kernel="7.0.0-test",
        landlock_abi=8,
        platform_matrix={"linux": {}, "macos": {}, "windows": {}},
    )


def test_sandbox_tier_unavailable_names_the_probe_that_failed() -> None:
    p = _probe_with(
        "L2",
        {
            "T0": (True, "always"),
            "L0": (True, "rlimits"),
            "L1": (True, "landlock abi 8"),
            "L2": (True, "seccomp filter installed"),
            "L3": (False, "bwrap --unshare-all --ro-bind / / /bin/true exited 1"),
        },
    )
    with pytest.raises(SandboxTierUnavailable) as exc:
        default_sandbox("L3", probe_result=p)
    assert exc.value.code == "RL0401"
    assert exc.value.exit_code == 5
    assert "bwrap --unshare-all --ro-bind / / /bin/true" in exc.value.context["probe"]
    assert exc.value.context["tier"] == "L3"


def test_sandbox_tier_below_required() -> None:
    """L3's own probe passed but L2's did not, so the ladder stops below what was asked."""
    p = _probe_with(
        "L1",
        {
            "T0": (True, "always"),
            "L0": (True, "rlimits"),
            "L1": (True, "landlock abi 8"),
            "L2": (False, "seccomp filter refused"),
            "L3": (True, "bwrap probe exited 0"),
        },
    )
    with pytest.raises(SandboxTierBelowRequired) as exc:
        default_sandbox("L3", probe_result=p)
    assert exc.value.code == "RL0402"
    assert exc.value.exit_code == 5
    assert exc.value.context["required"] == "L3"
    assert exc.value.context["held"] == "L1"


def test_an_unknown_tier_for_this_os_is_unavailable_not_silently_ignored() -> None:
    p = _probe_with("L2", {"T0": (True, "always"), "L0": (True, "rlimits"), "L2": (True, "x")})
    with pytest.raises(SandboxTierUnavailable) as exc:
        default_sandbox("W1", probe_result=p)
    assert exc.value.code == "RL0401"
    assert exc.value.context["tier"] == "W1"


def test_the_result_carries_the_os_and_the_tier_that_held(work: Path) -> None:
    """What goes into the record: never a bare tier letter (D-37)."""
    r = run_python("print('x')", limits=Limits(wall_s=10.0), cwd=work, env={})
    p = probe()
    assert r.tier == p.tier_held
    assert r.sandbox_os == p.os
    assert dataclasses.asdict(r.probe)["os"] == p.os
