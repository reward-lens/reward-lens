"""D-27 audit persistence: interruption leaves sealed work that a later call replays."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from reward_lens import api, contracts
from reward_lens.examples.code_reward import write as write_example
from reward_lens.instruments.base import Panel
from reward_lens.product.audit import instruments
from reward_lens.product.audit import run as audit_run
from reward_lens.store import Project, RunDir

SCRIPT = Path(os.sys.executable).parent / "reward-lens"


def _project(tmp_path: Path) -> Project:
    root = tmp_path / "example"
    write_example(root)
    return Project.open(root)


def _wait_for_partial(process: subprocess.Popen[str], state_home: Path, entry_id: str) -> Path:
    """Wait for the actual CLI to seal one boundary before interrupting it."""
    deadline = time.monotonic() + 15
    filename = entry_id + ".json"
    while time.monotonic() < deadline:
        for path in (state_home / "reward-lens" / "runs").glob(f"run-*/partial/{filename}"):
            return path.parents[1]
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"audit stopped before sealing {entry_id}: exit {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.02)
    process.kill()
    stdout, stderr = process.communicate()
    pytest.fail(
        f"audit did not seal {entry_id} before its local blocking fixture timed out\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


def _resume_line(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["name"] or manifest["run_id"]
    return f"reward-lens audit {manifest['project']} --resume {name}"


def _subprocess_env(state_home: Path, *, pythonpath: Path | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(
        {
            "XDG_STATE_HOME": str(state_home),
            "NO_COLOR": "1",
            "REWARD_LENS_NO_PROGRESS": "1",
            "COLUMNS": "100",
            "TERM": "dumb",
        }
    )
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return env


@pytest.fixture(autouse=True)
def _private_run_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run store is persistent by design; each test gets an explicit durable root."""
    state_home = tmp_path / "in_process_state"
    state_home.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))


def test_interrupted_audit_replays_the_sealed_static_instrument(
    tmp_path: Path, monkeypatch
) -> None:
    project = _project(tmp_path)
    original = instruments.input_handling_entry

    def stop(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(instruments, "input_handling_entry", stop)
    with pytest.raises(KeyboardInterrupt):
        audit_run(api.AuditRequest(path=project.root), project=project)

    interrupted = RunDir.most_recent(project.root)
    assert interrupted is not None
    assert interrupted.finished is False
    assert "validity.static" in interrupted.partial()
    monkeypatch.setattr(instruments, "input_handling_entry", original)

    record = audit_run(
        api.AuditRequest(path=project.root, resume=interrupted.run_id), project=project
    )
    assert record.assay_id
    resumed = RunDir.find(interrupted.run_id)
    assert resumed.finished is True
    assert "instrument validity.static replayed" in (resumed.path / "log.txt").read_text()
    assert resumed.manifest["record_path"]


def test_resume_refuses_when_the_subject_digest_changed(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)

    def stop(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(instruments, "input_handling_entry", stop)
    with pytest.raises(KeyboardInterrupt):
        audit_run(api.AuditRequest(path=project.root), project=project)
    interrupted = RunDir.most_recent(project.root)
    assert interrupted is not None
    grader = project.root / "grader.py"
    grader.write_text(grader.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(Exception) as mismatch:
        audit_run(api.AuditRequest(path=project.root, resume=interrupted.run_id), project=project)
    assert getattr(mismatch.value, "code", None) == "RL0620"


def test_resume_refuses_when_the_declared_method_contract_changed(
    tmp_path: Path, monkeypatch
) -> None:
    project = _project(tmp_path)

    def stop(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(instruments, "input_handling_entry", stop)
    with pytest.raises(KeyboardInterrupt):
        audit_run(api.AuditRequest(path=project.root), project=project)
    interrupted = RunDir.most_recent(project.root)
    assert interrupted is not None

    # The execution binding is deliberately downstream of discovery: this is the exact value a
    # changed instrument method/parameter declaration supplies before any partial can replay.
    import importlib

    runner = importlib.import_module("reward_lens.product.audit.run")

    # `_requested` binds declared project methods before any instrument runs.  Its optional
    # project and trace-record arguments are part of that pre-execution binding, so retain both
    # in this test double.
    monkeypatch.setattr(
        runner,
        "_requested",
        lambda discovered, project=None, *, trace_record=None, include_builtin_checks=False: {
            "validity.input_handling": "changed@digest"
        },
    )
    with pytest.raises(Exception) as mismatch:
        audit_run(api.AuditRequest(path=project.root, resume=interrupted.run_id), project=project)
    assert getattr(mismatch.value, "code", None) == "RL0620"


def test_discovered_instrument_partial_replays_without_a_second_run(
    tmp_path: Path, monkeypatch
) -> None:
    project = _project(tmp_path)
    calls = {"first": 0, "second": 0}

    class First:
        id = "reach.first"
        section = "reach"
        method = contracts.Method(
            id="reach.first",
            version="1.0.0",
            params_digest=contracts.digest({}),
            procedure="fixture",
        )

        def run(self, subject, corpus, ctx):
            calls["first"] += 1
            return [
                ctx.absence(
                    "reach",
                    self.id,
                    "fixture",
                    "no fixture",
                    "supply fixture",
                    ("no fixture claim",),
                    subject_ref=contracts.digest({"source": subject.source}),
                )
            ]

    class Second(First):
        id = "reach.second"
        method = contracts.Method(
            id="reach.second",
            version="1.0.0",
            params_digest=contracts.digest({}),
            procedure="fixture",
        )

        def run(self, subject, corpus, ctx):
            calls["second"] += 1
            if calls["second"] == 1:
                raise KeyboardInterrupt()
            return [
                ctx.absence(
                    "reach",
                    self.id,
                    "fixture",
                    "no fixture",
                    "supply fixture",
                    ("no fixture claim",),
                    subject_ref=contracts.digest({"source": subject.source}),
                )
            ]

    from reward_lens.product.audit import registry

    monkeypatch.setattr(registry, "discover", lambda: (Panel.of("reach", (First(), Second())),))
    with pytest.raises(KeyboardInterrupt):
        audit_run(api.AuditRequest(path=project.root), project=project)
    interrupted = RunDir.most_recent(project.root)
    assert interrupted is not None and "reach.first" in interrupted.partial()

    audit_run(api.AuditRequest(path=project.root, resume=interrupted.run_id), project=project)
    assert calls == {"first": 1, "second": 2}
    log = (interrupted.path / "log.txt").read_text(encoding="utf-8")
    assert "instrument reach.first replayed" in log
    assert "instrument reach.first ran" not in log.split("instrument reach.first replayed", 1)[1]


def test_cli_sigint_keeps_a_sealed_partial_and_the_printed_resume_completes(
    tmp_path: Path,
) -> None:
    """A real terminal interrupt comes after a durable boundary, not a test-runner exception.

    The flag file is copied into the grader's read-only project staging area.  Removing it leaves
    the grader source unchanged for resume, while making the second invocation quick enough to
    prove that the exact command line can finish the interrupted audit.
    """
    project = _project(tmp_path)
    grader = project.root / "grader.py"
    grader.write_text(
        grader.read_text(encoding="utf-8")
        + "\nfrom pathlib import Path\nimport time\n"
        + "_pause = Path(__file__).with_name('.pause-audit')\n"
        + "_original_score = score\n"
        + "def score(task, response):\n"
        + "    if _pause.exists():\n"
        + "        time.sleep(5)\n"
        + "    return _original_score(task, response)\n",
        encoding="utf-8",
    )
    pause = project.root / ".pause-audit"
    pause.write_text("block the input-handling grader only", encoding="utf-8")
    state_home = tmp_path / "state"
    state_home.mkdir()
    env = _subprocess_env(state_home)
    process = subprocess.Popen(
        [str(SCRIPT), "audit", str(project.root), "--name", "sigint-resume", "--format", "text"],
        cwd=project.root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    run_dir = _wait_for_partial(process, state_home, "validity.static")
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 130, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert "RL0130" in stderr
    assert stderr.rstrip().splitlines()[-1] == _resume_line(run_dir)
    assert (run_dir / "partial" / "validity.static.json").is_file()

    pause.unlink()
    resume = _resume_line(run_dir).split()
    # Watchdog for the full bwrap audit battery, not a product latency SLA.
    completed = subprocess.run(
        [str(SCRIPT), *resume[1:], "--format", "text"],
        cwd=project.root,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert completed.returncode in (0, 1, 2), completed.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "finished"
    assert Path(manifest["record_path"]).is_file()
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "instrument validity.static replayed" in log
    assert (
        "instrument validity.static ran"
        not in log.split("instrument validity.static replayed", 1)[1]
    )


def test_cli_running_budget_cap_keeps_partials_and_resume_skips_sealed_work(tmp_path: Path) -> None:
    """The cap exit is emitted by the CLI after an audit has already sealed real work.

    The fixture is loaded in the child process through ``sitecustomize`` so this test executes
    the installed CLI, Click's error path, the run store and a later fresh-process resume.  It
    uses the actual ``BudgetExhausted`` class and a flag file as its bounded local resource cap;
    no provider, network request or in-process test monkeypatch is involved.
    """
    project = _project(tmp_path)
    state_home = tmp_path / "state"
    state_home.mkdir()
    hook = tmp_path / "child_fixture"
    hook.mkdir()
    cap_flag = tmp_path / "cap-open"
    cap_flag.write_text("the local fixture has reached its cap", encoding="utf-8")
    (hook / "sitecustomize.py").write_text(
        """\
from pathlib import Path
import os
import time

from reward_lens import contracts
from reward_lens.instruments.base import Panel
from reward_lens.instruments.seeker.budget import BudgetExhausted
from reward_lens.product.audit import registry


class SealedFixture:
    id = "exploits.local_sealed"
    section = "exploits"
    method = contracts.Method(
        id=id, version="1.0.0", params_digest=contracts.digest({"fixture": "sealed"}),
        procedure="a deterministic local cap fixture",
    )

    def run(self, subject, corpus, ctx):
        return [ctx.absence(
            "exploits", self.id, "the pre-cap local fixture", "the fixture is an absence",
            "remove the fixture", ("no exploit claim",),
            subject_ref=contracts.digest({"source": subject.source}),
        )]


class CapFixture(SealedFixture):
    id = "exploits.local_cap"
    method = contracts.Method(
        id=id, version="1.0.0", params_digest=contracts.digest({"fixture": "cap"}),
        procedure="a deterministic local cap fixture",
    )

    def run(self, subject, corpus, ctx):
        if Path(os.environ["RL_EXEC_CAP_FLAG"]).exists():
            time.sleep(0.05)
            raise BudgetExhausted(cap="0.01", spent="0.01", asked="0.01")
        return super().run(subject, corpus, ctx)


registry.discover = lambda: (Panel.of("exploits", (SealedFixture(), CapFixture())),)
""",
        encoding="utf-8",
    )
    env = _subprocess_env(state_home, pythonpath=hook)
    env["RL_EXEC_CAP_FLAG"] = str(cap_flag)
    started = subprocess.run(
        [str(SCRIPT), "audit", str(project.root), "--name", "budget-resume", "--format", "text"],
        cwd=project.root,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert started.returncode == 6, f"stdout:\n{started.stdout}\nstderr:\n{started.stderr}"
    run_dir = next((state_home / "reward-lens" / "runs").glob("run-*"))
    assert (run_dir / "partial" / "exploits.local_sealed.json").is_file()
    assert not (run_dir / "partial" / "exploits.local_cap.json").exists()
    assert "RL0501" in started.stderr
    assert started.stderr.rstrip().splitlines()[-1] == _resume_line(run_dir)

    cap_flag.unlink()
    resume = _resume_line(run_dir).split()
    completed = subprocess.run(
        [str(SCRIPT), *resume[1:], "--format", "text"],
        cwd=project.root,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert completed.returncode in (0, 1, 2), completed.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "finished"
    log = (run_dir / "log.txt").read_text(encoding="utf-8")
    assert "instrument exploits.local_sealed replayed" in log
    assert (
        "instrument exploits.local_sealed ran"
        not in log.split("instrument exploits.local_sealed replayed", 1)[1]
    )


def test_l3_does_not_bind_a_callers_script_directory_into_candidate_reads(tmp_path: Path) -> None:
    """A candidate can read its staged canary but never a protected sibling of the caller script.

    The outer process is intentionally a file under ``tmp_path`` rather than ``python -c``.  That
    puts its directory on ``sys.path`` and reproduces the public Improve route: treating arbitrary
    ``sys.path`` entries as interpreter roots made Bubblewrap bind that whole directory, including
    the protected candidate file.  The inner process must retain its real L3 canary read while the
    unbound sibling path is absent.
    """
    from reward_lens.execution import probe

    if not probe().tiers["L3"].held:
        pytest.skip("L3 unavailable on this host")
    root = tmp_path / "caller-root"
    root.mkdir()
    runner = root / "runner.py"
    runner.write_text(
        '''\
import json
from pathlib import Path
from reward_lens.execution import run_python

root = Path(__file__).resolve().parent
staged = root / "staged"
staged.mkdir()
canary = staged / "canary.txt"
canary.write_text("canary", encoding="utf-8")
protected = root / "protected.txt"
protected.write_text("protected", encoding="utf-8")
work = root / "work"
work.mkdir()
program = """import json, pathlib, sys
result = {}
for name, value in zip((\"canary\", \"protected\"), sys.argv[1:]):
    try:
        result[name] = pathlib.Path(value).read_text()
    except OSError as error:
        result[name] = type(error).__name__
print(json.dumps(result, sort_keys=True))
"""
result = run_python(program, args=[str(canary), str(protected)], cwd=work, write_roots=[staged])
print(json.dumps({"exit_code": result.exit_code, "tier": result.tier, "stdout": result.stdout.decode()}, sort_keys=True))
''',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [os.sys.executable, str(runner)],
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["exit_code"] == 0
    assert observed["tier"] == "L3"
    assert json.loads(observed["stdout"]) == {
        "canary": "canary",
        "protected": "FileNotFoundError",
    }


def test_finished_audit_resume_leaves_the_original_record_bytes_unchanged(tmp_path: Path) -> None:
    """A finished audit forks its execution state without rewriting the sealed old record."""
    project = _project(tmp_path)
    record = audit_run(api.AuditRequest(path=project.root, name="finished"), project=project)
    original = RunDir.most_recent(project.root)
    assert original is not None and original.finished
    record_path = Path(original.manifest["record_path"])
    before = record_path.read_bytes()

    replayed = audit_run(
        api.AuditRequest(path=project.root, resume=original.run_id), project=project
    )
    fork = RunDir.most_recent(project.root)
    assert fork is not None and fork.run_id != original.run_id and fork.finished
    assert fork.manifest["forked_from"] == original.run_id
    assert replayed.assay_id == record.assay_id
    assert record_path.read_bytes() == before
