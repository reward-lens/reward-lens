"""The macOS and Windows paths, exercised with mocked syscalls and claimed nowhere.

These run on Linux. They prove the code exists and is shaped right; the probe's platform matrix is
what says the tiers are pending on any machine that is not that operating system.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from reward_lens.execution import Limits
from reward_lens.execution.macos import (
    SANDBOX_EXEC_PROBE_ARGV,
    MacosSandbox,
    probe_sandbox_exec,
    seatbelt_profile,
)
from reward_lens.execution.windows import (
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
    JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_JOB_TIME,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    WindowsSandbox,
    job_limit_flags,
    probe_appcontainer,
    probe_job_object,
)


class FakeCompleted(SimpleNamespace):
    pass


def test_seatbelt_profile_denies_by_default_and_writes_only_the_work_dir(tmp_path: Path) -> None:
    profile = seatbelt_profile(tmp_path / "work")
    assert profile.startswith("(version 1)")
    assert "(deny default)" in profile
    assert '(allow file-read* (subpath "/usr"))' in profile
    assert f'(allow file-write* (subpath "{tmp_path / "work"}"))' in profile
    assert "(deny network*)" in profile
    assert profile.count("(allow file-write*") == 1


def test_the_macos_probe_is_the_documented_command(monkeypatch: pytest.MonkeyPatch) -> None:
    assert SANDBOX_EXEC_PROBE_ARGV == [
        "sandbox-exec",
        "-p",
        "(version 1)(allow default)",
        "/usr/bin/true",
    ]
    calls: list[list[str]] = []

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        calls.append(list(argv))
        return FakeCompleted(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("reward_lens.execution.macos.subprocess.run", fake_run)
    monkeypatch.setattr("reward_lens.execution.macos.platform.system", lambda: "Darwin")
    held, reason, ms = probe_sandbox_exec()
    assert held is True
    assert "deprecat" in reason.lower()  # the warning is recorded, not suppressed
    assert calls == [SANDBOX_EXEC_PROBE_ARGV]
    assert ms >= 0.0


def test_m1_wraps_the_argv_in_sandbox_exec(tmp_path: Path) -> None:
    s = MacosSandbox(tier="M1")
    work = tmp_path / "work"
    work.mkdir()
    argv, profile_path = s.wrap(["/usr/bin/python3", "main.py"], cwd=work)
    assert argv[0] == "sandbox-exec"
    assert argv[1] == "-f"
    assert argv[2] == str(profile_path)
    assert argv[3:] == ["/usr/bin/python3", "main.py"]
    assert profile_path.read_text() == seatbelt_profile(work)


def test_macos_memory_comes_from_the_supervisor_never_rlimit_as() -> None:
    """`RLIMIT_AS` is not enforced by XNU and `RLIMIT_NPROC` is per user: neither is used."""
    s = MacosSandbox(tier="M1")
    names = s.rlimit_names(Limits())
    assert "RLIMIT_AS" not in names
    assert "RLIMIT_NPROC" not in names
    assert {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE"} <= set(names)
    assert s.memory_enforcement == "supervisor_rss_sampling"


def test_windows_job_limits_are_the_four_d37_names() -> None:
    flags = job_limit_flags()
    for bit in (
        JOB_OBJECT_LIMIT_JOB_MEMORY,
        JOB_OBJECT_LIMIT_JOB_TIME,
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    ):
        assert flags & bit


def test_the_windows_probe_calls_createjobobject(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class FakeKernel32:
        def CreateJobObjectW(self, a, b):  # noqa: N802
            seen.append("CreateJobObjectW")
            return 4242

        def SetInformationJobObject(self, h, cls, info, size):  # noqa: N802
            seen.append(f"SetInformationJobObject:{cls}")
            return 1

        def CloseHandle(self, h):  # noqa: N802
            seen.append("CloseHandle")
            return 1

    monkeypatch.setattr("reward_lens.execution.windows._kernel32", lambda: FakeKernel32())
    held, reason, ms = probe_job_object()
    assert held is True
    assert "CreateJobObjectW" in seen
    assert "CloseHandle" in seen
    assert reason
    assert ms >= 0.0


def test_w0_sets_exactly_the_four_limits_on_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}

    class FakeKernel32:
        def CreateJobObjectW(self, a, b):  # noqa: N802
            return 7
        def SetInformationJobObject(self, h, cls, info, size):  # noqa: N802
            recorded["class"] = cls
            recorded["limits"] = ctypes.cast(
                info, ctypes.POINTER(WindowsSandbox.EXTENDED_LIMIT_INFORMATION)
            ).contents
            return 1
        def AssignProcessToJobObject(self, h, p):  # noqa: N802
            recorded["assigned"] = p
            return 1
        def CloseHandle(self, h):  # noqa: N802
            return 1

    monkeypatch.setattr("reward_lens.execution.windows._kernel32", lambda: FakeKernel32())
    s = WindowsSandbox(tier="W0")
    limits = Limits(wall_s=5.0, memory_bytes=128 * 2**20, processes=9)
    handle = s.create_job(limits)
    assert handle == 7
    info = recorded["limits"]
    assert info.BasicLimitInformation.LimitFlags == job_limit_flags()
    assert info.JobMemoryLimit == 128 * 2**20
    assert info.BasicLimitInformation.ActiveProcessLimit == 9


def test_w1_asks_for_an_appcontainer_with_zero_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeUserenv:
        def CreateAppContainerProfile(self, name, display, desc, caps, count, sid):  # noqa: N802
            seen["capabilities"] = caps
            seen["count"] = count
            return 0

        def DeleteAppContainerProfile(self, name):  # noqa: N802
            return 0

    monkeypatch.setattr("reward_lens.execution.windows._userenv", lambda: FakeUserenv())
    held, reason, ms = probe_appcontainer()
    assert held is True
    assert seen["count"] == 0
    assert not seen["capabilities"]
    assert reason
    assert ms >= 0.0


def test_both_foreign_sandboxes_satisfy_the_protocol() -> None:
    from reward_lens.execution import Sandbox

    assert isinstance(MacosSandbox(tier="M1"), Sandbox)
    assert isinstance(WindowsSandbox(tier="W1"), Sandbox)


def test_a_foreign_sandbox_refuses_to_run_on_this_machine(tmp_path: Path) -> None:
    """Nothing pretends: asking a macOS sandbox to run on Linux is RL0401, not a silent T0."""
    from reward_lens.execution.errors import SandboxTierUnavailable

    with pytest.raises(SandboxTierUnavailable) as exc:
        MacosSandbox(tier="M1").run(["/bin/true"], limits=Limits(), cwd=tmp_path, env={})
    assert exc.value.code == "RL0401"
