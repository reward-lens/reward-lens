"""The local deterministic adapter and the `plain` shape (D-34, D-35, D-36, D-65, section 7.1).

`PythonGrader.from_path(path)` adapts a user's `score(task, response) -> float`. It never imports
that function into this process. A supervisor script is staged into a per-run scratch directory
with a copy of the source, and the sandbox of interfaces section 3 runs it; the supervisor points
file descriptor 1 at a capture file before importing the graded module, so nothing the graded code
prints can reach the channel the result comes back on. Provenance, counters, timing and tier are
stamped from what the sandbox reported and from nothing else (D-34).

The three returns that frameworks swallow are loud here (D-65). `None` is `unscored` and RL0210,
which says that TRL turns it into NaN and drops the function for that row. A raised exception is
`grader_error` and RL0211, which says `verifiers` would have scored it 0.0. A `bool` scores, and
RL0212 says so: it passes `float()` and reads as a legitimate score. Each finding carries a
witness.

A capability is an observation. `from_path` claims only what reading the source shows; `probe`
runs the grader twice and claims what those runs showed. Nothing else may be claimed (RL0702).

Owned by P-ADAPT-PY.
"""

from __future__ import annotations

import ast
import json
import math
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from reward_lens.contracts import Counters, RewardLensError, digest, quantise
from reward_lens.execution.scratch import stage_source
from reward_lens.graders.base import (
    CAPABILITY_KEYS,
    AdapterCapabilityUnproven,
    CapabilityVector,
    EvidenceEnvelope,
    InvalidInput,
    Limits,
    Manifest,
    Result,
    Sandbox,
    ValidityFinding,
)

__all__ = ["DEFAULT_SEED", "PythonGrader"]

#: `AuditRequest.seed`, so that an adapter run on its own reproduces an adapter run under an audit.
DEFAULT_SEED = 20260911

#: The line the supervisor writes after it has taken the graded code's stdout away from it.
_MARKER = "@@RL@@"

#: The supervisor's file name inside the machinery directory. It is never at the root of the tree
#: the grader is given, so a project that happens to hold a file of this name is not shadowed.
_SUPERVISOR_NAME = "_rl_supervisor.py"

#: The environment the graded process gets. No key of the host's reaches it.
_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
}

_INPUT_SCHEMA = {
    "type": "object",
    "required": ["id", "prompt"],
    "properties": {"id": {"type": "string"}, "prompt": {"type": "string"}},
}
_OUTPUT_SCHEMA = {"type": ["number", "boolean", "null"]}

_PROBE_TASK = {"id": "rl.probe", "prompt": "reward-lens capability probe", "reference": ""}
_PROBE_RESPONSE = "reward-lens capability probe"

_SUPERVISOR = '''\
"""Staged by reward_lens.graders.python_grader. Runs inside the sandbox, never in the parent."""

import io
import json
import os
import random
import sys
import traceback

_grader_path, _entrypoint, _request_path, _capture_path, _seed = sys.argv[1:6]
random.seed(int(_seed))

with open(_request_path, "r", encoding="utf-8") as _fh:
    _request = json.load(_fh)

# Take stdout away from the graded code before importing it. Fd 1 is pointed at a capture file, so
# neither print(), sys.stdout.write() nor os.write(1, ...) can reach the parent's result channel.
_real = os.dup(1)
_capture = open(_capture_path, "w", encoding="utf-8", errors="replace")
os.dup2(_capture.fileno(), 1)
sys.stdout = _capture

try:
    import importlib.util

    _spec = importlib.util.spec_from_file_location("rl_graded_module", _grader_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["rl_graded_module"] = _module
    _spec.loader.exec_module(_module)
    _fn = getattr(_module, _entrypoint)
    _value = _fn(_request["task"], _request["response"])
except BaseException as _exc:
    _payload = {
        "outcome": "raised",
        "type": type(_exc).__name__,
        "message": str(_exc),
        "traceback": traceback.format_exc()[-4000:],
    }
else:
    if _value is None:
        _payload = {"outcome": "returned", "kind": "none", "value": None, "type": "NoneType"}
    elif isinstance(_value, bool):
        _payload = {"outcome": "returned", "kind": "bool", "value": _value, "type": "bool"}
    elif isinstance(_value, (int, float)):
        _payload = {
            "outcome": "returned",
            "kind": "number",
            "value": repr(float(_value)),
            "type": type(_value).__name__,
        }
    else:
        _payload = {
            "outcome": "returned",
            "kind": "other",
            "value": repr(_value)[:200],
            "type": type(_value).__name__,
        }

sys.stdout = io.StringIO()
os.dup2(_real, 1)
try:
    _capture.close()
except Exception:
    pass
try:
    with open(_capture_path, "r", encoding="utf-8", errors="replace") as _fh:
        _payload["stdout"] = _fh.read()[:8192]
except OSError:
    _payload["stdout"] = ""
os.write(1, ("@@RL@@" + json.dumps(_payload) + "\\n").encode("utf-8"))
'''


#: The operators an arithmetic mutation can flip. Counted as a mutation surface; nothing else is.
_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)

#: The counters that are a cost meter rather than a resource observation. The adapter fills in none
#: of them: it passes `Result.counters` through untouched and constructs an empty `Counters()` when
#: the sandbox never started, so a value here came from the sandbox and from nowhere else.
_METERED_COUNTERS = ("usd", "input_tokens", "output_tokens", "api_calls")

#: The capabilities reading the source establishes. Every other key needs a run (D-34).
_STATIC_CAPABILITIES = ("source_visible", "has_mutation_surface")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _has_mutation_surface(tree: ast.AST) -> bool:
    """Whether an `ast` walk finds a construct the mutation instrument can actually operate on.

    Four constructs count and nothing else does: a comparison, an arithmetic operator, a boolean
    operator, and a constant inside a branch condition. A grader whose whole body is `return 1.0`
    gives the instrument nothing to change, and this reports it false rather than claiming a
    surface that is not there for it to find.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, _ARITHMETIC):
            return True
    for branch in ast.walk(tree):
        if not isinstance(branch, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            continue
        if any(isinstance(node, ast.Constant) for node in ast.walk(branch.test)):
            return True
    return False


def _is_metered(counters: Counters) -> bool:
    """Whether the sandbox reported a metered quantity. An empty counter is no meter at all."""
    return any(getattr(counters, name, None) is not None for name in _METERED_COUNTERS)


def _make_read_only(root: Path, *, keep: Path | None = None) -> None:
    """Take write off every staged file and directory: the copy is a reference, not a workspace.

    `keep` is the machinery directory staged inside `root`, which keeps its own 0o700 so the
    supervisor can write the stdout capture there. `root` itself still loses write, so the grader
    cannot create anything beside the project's own files; traversing into `keep` needs only the
    execute bit `root` keeps.
    """
    for path in sorted(root.rglob("*"), reverse=True):
        if keep is not None and (path == keep or keep in path.parents):
            continue
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def _remove_tree(root: Path) -> None:
    """Delete a scratch directory, read-only staging and all. Modes go back before the removal."""
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(0o700)
        except OSError:  # pragma: no cover - a path that vanished under us is already gone
            pass
    shutil.rmtree(root, ignore_errors=True)


def _finding(code: str, rule: str, trainer: str, behaviour: str, message: str, witness: dict) -> ValidityFinding:
    return ValidityFinding(
        code=code,
        rule=rule,
        level="warning",
        trainer=trainer,
        trainer_behaviour=behaviour,
        message=message,
        witness=witness,
    )


class PythonGrader:
    """A local deterministic grader of the `plain` shape, run in the sandbox and nowhere else."""

    def __init__(
        self,
        *,
        path: Path,
        entrypoint: str,
        source: str,
        manifest: Manifest,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.path = path
        self.entrypoint = entrypoint
        self._source = source
        self._manifest = manifest
        self.seed = seed

    # --- construction and the static probe --------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str, entrypoint: str = "score") -> PythonGrader:
        """Read the grader, confirm the declared shape is there, and claim only what reading shows."""
        path = Path(path)
        if not path.is_file():
            raise AdapterCapabilityUnproven(
                capability="plain_shape_entrypoint",
                message=f"there is no grader at {path}, so the plain shape cannot be established",
                remediation="point reward.entry at a file that exists",
            )
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AdapterCapabilityUnproven(
                capability="source_visible",
                message=f"{path} could not be read, so nothing can see its source",
                remediation="give the grader file a mode this process can read",
            ) from exc
        # This read, and only this read, is what establishes `source_visible` below.
        source_visible = True
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise AdapterCapabilityUnproven(
                capability="plain_shape_entrypoint",
                message=f"{path} does not parse, so no entrypoint named {entrypoint} is provable",
                remediation=f"fix the syntax error at line {exc.lineno}",
            ) from exc
        found = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint
        ]
        if not found:
            raise AdapterCapabilityUnproven(
                capability="plain_shape_entrypoint",
                message=(
                    f"{path} defines no module-level {entrypoint}(task, response), so the plain "
                    "shape is not established by anything"
                ),
                remediation=f"define {entrypoint}(task, response) or set reward.entry",
            )
        if isinstance(found[0], ast.AsyncFunctionDef):
            raise AdapterCapabilityUnproven(
                capability="plain_shape_entrypoint",
                message=(
                    f"{path}'s {entrypoint} is async; the plain shape of D-65 is a synchronous "
                    "score(task, response), and the async shapes arrive with P-CONNECT"
                ),
                remediation="make the entrypoint synchronous",
            )

        source_digest = digest({"source": source, "entrypoint": entrypoint})
        capabilities = CapabilityVector(
            probed={
                **{key: False for key in CAPABILITY_KEYS},
                "source_visible": source_visible,
                "has_mutation_surface": _has_mutation_surface(tree),
            }
        )
        manifest = Manifest(
            family="local_deterministic",
            implementation_revision="reward-lens/graders/python_grader/1",
            source_digest=source_digest,
            input_schema_digest=digest(_INPUT_SCHEMA),
            output_schema_digest=digest(_OUTPUT_SCHEMA),
            shape="plain",
            declared_inputs=("task", "response"),
            declared_outputs=("score",),
            resource_needs={
                "wall_s": 30,
                "cpu_s": 30,
                "memory_bytes": 2 * 2**30,
                "processes": 64,
                "network": False,
            },
            capabilities=capabilities,
        )
        return cls(path=path, entrypoint=entrypoint, source=source, manifest=manifest)

    # --- the protocol ------------------------------------------------------------------------

    def describe(self) -> Manifest:
        return self._manifest

    def source(self) -> str:
        return self._source

    def reset(self) -> None:
        """Stateless by manifest: there is nothing to reset, and saying so is the point."""
        return None

    def validate_input(self, task: Any, response: Any) -> None | InvalidInput:
        if not isinstance(task, Mapping):
            return InvalidInput(field="task", reason="a task is an object, not a " + type(task).__name__)
        for key in ("id", "prompt"):
            if key not in task:
                return InvalidInput(field=key, reason=f"a task carries {key}, and this one does not")
            if not isinstance(task[key], str):
                return InvalidInput(
                    field=key, reason=f"task.{key} is a string, not a {type(task[key]).__name__}"
                )
        if not isinstance(response, (str, Mapping)):
            return InvalidInput(
                field="response",
                reason=f"a response is a string or an object, not a {type(response).__name__}",
            )
        return None

    def score(
        self,
        task: Any,
        response: Any,
        *,
        sandbox: Sandbox,
        limits: Limits,
        seed: int | None = None,
        project_dir: Path | None = None,
    ) -> EvidenceEnvelope:
        """Score one request. `project_dir` is the grader's own project, staged as its working
        tree (A-005); with none, the grader runs alone in the scratch directory."""
        seed = self.seed if seed is None else seed
        started_at = _utc_now()
        clock = time.monotonic()
        request = {"task": task, "response": response, "entrypoint": self.entrypoint, "seed": seed}
        request_digest = digest({"task": task, "response": response, "seed": seed, "entrypoint": self.entrypoint})

        bad = self.validate_input(task, response)
        if bad is not None:
            return self._envelope(
                started_at=started_at,
                clock=clock,
                seed=seed,
                request=request,
                request_digest=request_digest,
                verdict="invalid_input",
                score=None,
                counters=Counters(),
                tier="T0",
                errors=(f"{bad.field}: {bad.reason}",),
                limitations=("the grader did not run: the request was refused before the sandbox",),
                observations={"grader": {"stdout": "", "returned_type": None}, "input": bad.to_dict()},
            )

        scratch = Path(tempfile.mkdtemp(prefix="rl-grader-"))
        try:
            outcome = self._run(
                scratch, request, seed, sandbox=sandbox, limits=limits, project_dir=project_dir
            )
        finally:
            _remove_tree(scratch)

        return self._envelope(
            started_at=started_at,
            clock=clock,
            seed=seed,
            request=request,
            request_digest=request_digest,
            **outcome,
        )

    def replay(
        self,
        envelope: EvidenceEnvelope,
        *,
        sandbox: Sandbox,
        limits: Limits,
        project_dir: Path | None = None,
    ) -> EvidenceEnvelope:
        """Re-run the recorded request under the recorded seed. Replay is local by manifest."""
        request = (envelope.observations or {}).get("request")
        if not request:
            raise AdapterCapabilityUnproven(
                capability="locally_replayable",
                message="the envelope carries no request, so there is nothing to replay",
                remediation="replay an envelope this adapter produced",
            )
        return self.score(
            request["task"],
            request["response"],
            sandbox=sandbox,
            limits=limits,
            seed=envelope.seed,
            project_dir=project_dir,
        )

    # --- the dynamic probe -------------------------------------------------------------------

    def probe(
        self, *, sandbox: Sandbox, limits: Limits, project_dir: Path | None = None
    ) -> PythonGrader:
        """Run the grader twice on a fixed probe and return a grader whose manifest says what ran.

        D-34's rule in one method: the capabilities this returns are the ones two real runs
        established, and nothing here consults a method name.

        A run the adapter did not observe establishes nothing. A probe run that raised, timed out,
        was cut off or came back with any verdict other than `scored` is not an observation of the
        grader running, and every capability that needs a run stays false, as does the determinism
        class and the replay mode. The alternative is the vacuous instrument: a grader that fails on
        every call, carrying a manifest that says the adapter controls its execution.
        """
        first = self.score(
            _PROBE_TASK, _PROBE_RESPONSE, sandbox=sandbox, limits=limits, seed=self.seed,
            project_dir=project_dir,
        )
        second = self.score(
            _PROBE_TASK, _PROBE_RESPONSE, sandbox=sandbox, limits=limits, seed=self.seed,
            project_dir=project_dir,
        )

        ran = first.verdict == "scored" and second.verdict == "scored"
        replayable = ran and first.normalised() == second.normalised()
        metered = ran and _is_metered(first.counters) and _is_metered(second.counters)
        components = ran and bool(first.components) and bool(second.components)

        # Every key that needs a run, and what this probe observed of it. The three at the bottom
        # are the ones this probe does not look at: no run raises them off the floor.
        run_dependent = {
            "controls_execution": ran,
            "locally_replayable": replayable,
            "cost_metered": metered,
            "returns_components": components,
            "outcome_independent": False,
            "state_observable": False,
            "protected_partition": False,
        }
        probed = {
            **{key: False for key in CAPABILITY_KEYS},
            **{
                key: value
                for key, value in self._manifest.capabilities.probed.items()
                if key in _STATIC_CAPABILITIES
            },
            **{key: bool(ran and value) for key, value in run_dependent.items()},
        }
        manifest = replace(
            self._manifest,
            determinism_class=(
                "deterministic_observed"
                if replayable
                else "nondeterministic_observed"
                if ran
                else self._manifest.determinism_class
            ),
            replay_mode=(
                "local_replay"
                if replayable
                else "no_local_replay"
                if ran
                else self._manifest.replay_mode
            ),
            capabilities=CapabilityVector(
                **{
                    name: getattr(self._manifest.capabilities, name)
                    for name in ("component_dag", "token_quantities", "checkpoint_capture", "controlled_updates", "cost_metering")
                },
                probed=probed,
            ),
        )
        return PythonGrader(
            path=self.path,
            entrypoint=self.entrypoint,
            source=self._source,
            manifest=manifest,
            seed=self.seed,
        )

    # --- the supervisor ----------------------------------------------------------------------

    def _stage(self, scratch: Path, project_dir: Path | None) -> tuple[Path, Path, Path]:
        """Put the grader where it runs and return its path, the working directory and the
        machinery directory (A-005).

        With no project, the grader runs alone in the scratch directory, which is what it did
        before A-005 and is all a self-contained grader needs. With a project, the project tree is
        copied into the scratch directory, every copied file and directory loses write, and the
        grader runs from its own place inside that copy, with the copy as the working directory.

        The copy is what makes the tree read-only, rather than pointing `cwd` at the real project.
        Two reasons. The sandbox tier is not the adapter's to choose: at T0 there is no isolation
        at all, so nothing but a mode on a throwaway copy holds at every tier a `Sandbox` may be.
        And a grader is graded code: pointing it at the caller's own directory would make the first
        thing an audit does a write into the tree it is auditing. So a grader that reads
        `outcome/test_solution.py` beside itself, or relative to where it was started, finds it,
        and a grader that writes there does not get to.

        The machinery that runs the grader is staged too, and inside the working directory rather
        than beside it. `Sandbox.run` takes a working directory and no read roots, and from L3 down
        the only path of the caller's that a run may read is that directory: a supervisor left in
        the scratch parent is not on the far side of the bind, and the interpreter opens nothing.
        It goes in a directory of its own, made by the execution package's own `stage_source`, so
        the name is a fresh `rl-*` from `mkdtemp` that no project path can collide with and the
        supervisor is not a file at the root of the tree the grader is handed.
        """
        if project_dir is None:
            grader_path = scratch / "grader.py"
            grader_path.write_text(self._source, encoding="utf-8")
            machinery, _ = stage_source(_SUPERVISOR, cwd=scratch, name=_SUPERVISOR_NAME)
            return grader_path, scratch, machinery

        project = Path(project_dir).resolve()
        staged = scratch / "project"
        shutil.copytree(project, staged)
        try:
            relative = self.path.resolve().relative_to(project)
        except ValueError:
            relative = Path(self.path.name)
        grader_path = staged / relative
        grader_path.parent.mkdir(parents=True, exist_ok=True)
        # The text that runs is the text this adapter read and digested, never whatever the copy
        # happened to hold.
        grader_path.write_text(self._source, encoding="utf-8")
        machinery, _ = stage_source(_SUPERVISOR, cwd=staged, name=_SUPERVISOR_NAME)
        _make_read_only(staged, keep=machinery)
        return grader_path, staged, machinery

    def _run(
        self,
        scratch: Path,
        request: dict,
        seed: int,
        *,
        sandbox: Sandbox,
        limits: Limits,
        project_dir: Path | None = None,
    ) -> dict[str, Any]:
        # The supervisor, the request and the capture stay out of the project the grader is shown:
        # they are in a directory of the staging's own, which `_stage` makes and names, and which
        # is inside the working directory because that is the one path of the caller's a sandboxed
        # run can read from L3 down.
        grader_path, cwd, machinery = self._stage(scratch, project_dir)
        supervisor = machinery / _SUPERVISOR_NAME
        request_path = machinery / "request.json"
        request_path.write_text(
            json.dumps({"task": request["task"], "response": request["response"]}),
            encoding="utf-8",
        )
        capture = machinery / "grader_stdout.txt"
        argv = [
            sys.executable,
            str(supervisor),
            str(grader_path),
            self.entrypoint,
            str(request_path),
            str(capture),
            str(seed),
        ]
        env = dict(_ENV, HOME=str(scratch), TMPDIR=str(scratch))

        try:
            result: Result = sandbox.run(argv, limits=limits, cwd=cwd, env=env, stdin=None)
        except RewardLensError as exc:
            if exc.code in ("RL0401", "RL0402"):
                return {
                    "verdict": "provider_unavailable",
                    "score": None,
                    "counters": Counters(),
                    "tier": "T0",
                    "errors": (f"{exc.code}: {exc.message}",),
                    "limitations": (
                        "the sandbox did not start, so the tier recorded is T0: none held",
                    ),
                    "observations": {"grader": {"stdout": "", "returned_type": None}},
                }
            if exc.code == "RL0410":
                return {
                    "verdict": "resource_exhausted",
                    "score": None,
                    "counters": Counters(),
                    "tier": "T0",
                    "errors": (f"{exc.code}: {exc.message}",),
                    "limitations": (),
                    "observations": {"grader": {"stdout": "", "returned_type": None}},
                }
            raise

        base: dict[str, Any] = {
            "counters": result.counters,
            "tier": result.tier,
            "observations": {
                "grader": {"stdout": "", "returned_type": None},
                "sandbox": {
                    "tier": result.tier,
                    "exit_code": result.exit_code,
                    "breach": result.breach,
                },
            },
        }

        if result.breach == "wall":
            return {
                **base,
                "verdict": "timeout",
                "score": None,
                "errors": (f"the wall limit of {limits.wall_s}s was reached",),
                "limitations": ("a timeout is not a score, and never a reward of zero (D-36)",),
            }
        if result.breach is not None:
            return {
                **base,
                "verdict": "resource_exhausted",
                "score": None,
                "errors": (f"the {result.breach} limit was reached",),
                "limitations": ("a breach is not a score, and never a reward of zero (D-36)",),
            }
        if result.exit_code in (-2, -15, 130, 143):
            return {
                **base,
                "verdict": "cancelled",
                "score": None,
                "errors": (f"the graded process was signalled (exit {result.exit_code})",),
                "limitations": (),
            }

        payload = self._parse(result.stdout)
        if payload is None:
            stderr = result.stderr.decode("utf-8", "replace")[-2000:]
            return {
                **base,
                "verdict": "parse_error",
                "score": None,
                "errors": (
                    "the graded process produced no result line"
                    + (f"; stderr: {stderr}" if stderr else ""),
                ),
                "limitations": (),
            }

        base["observations"]["grader"] = {
            "stdout": payload.get("stdout", ""),
            "returned_type": payload.get("type"),
        }
        return {**base, **self._interpret(payload, request)}

    @staticmethod
    def _parse(stdout: bytes) -> dict[str, Any] | None:
        text = stdout.decode("utf-8", "replace")
        for line in reversed(text.splitlines()):
            if line.startswith(_MARKER):
                try:
                    parsed = json.loads(line[len(_MARKER) :])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
        return None

    def _interpret(self, payload: dict[str, Any], request: dict) -> dict[str, Any]:
        witness_inputs = {"task": request["task"], "response": request["response"]}

        if payload.get("outcome") == "raised":
            kind = payload.get("type", "Exception")
            message = payload.get("message", "")
            return {
                "verdict": "grader_error",
                "score": None,
                "errors": (f"{kind}: {message}", payload.get("traceback", "")),
                "limitations": (),
                "findings": (
                    _finding(
                        code="RL0211",
                        rule="validity.grader_raised_on_input",
                        trainer="verifiers",
                        behaviour=(
                            "verifiers swallows an exception in a reward function and scores it "
                            "0.0, so the optimiser sees a bad answer where there was no answer"
                        ),
                        message=f"the grader raised {kind} on a well-formed request",
                        witness={
                            "inputs": witness_inputs,
                            "procedure": (
                                f"ran {self.entrypoint}(task, response) in the sandbox on the "
                                "request above"
                            ),
                            "observed": f"the grader raised {kind}: {message}",
                        },
                    ),
                ),
            }

        kind = payload.get("kind")
        if kind == "none":
            return {
                "verdict": "unscored",
                "score": None,
                "errors": (),
                "limitations": (),
                "findings": (
                    _finding(
                        code="RL0210",
                        rule="validity.none_on_input",
                        trainer="trl",
                        behaviour=(
                            "TRL turns a returned None into NaN and drops that function for that "
                            "row, so the row is optimised against the remaining functions alone"
                        ),
                        message="the grader returned None on a well-formed request",
                        witness={
                            "inputs": witness_inputs,
                            "procedure": (
                                f"ran {self.entrypoint}(task, response) in the sandbox on the "
                                "request above"
                            ),
                            "observed": "the grader returned None rather than a score",
                        },
                    ),
                ),
            }

        if kind == "bool":
            value = bool(payload.get("value"))
            return {
                "verdict": "scored",
                "score": float(value),
                "errors": (),
                "limitations": (
                    "the grader returned a bool, so the score is 1.0 or 0.0 by coercion",
                ),
                "findings": (
                    _finding(
                        code="RL0212",
                        rule="validity.bool_as_score",
                        trainer="trl",
                        behaviour=(
                            "a bool passes float() and reads as a legitimate score, so TRL, "
                            "verifiers and inspect all optimise against 1.0 or 0.0 without "
                            "anything reporting that the grader never produced a number"
                        ),
                        message="the grader returned a bool where a float was declared",
                        witness={
                            "inputs": witness_inputs,
                            "procedure": (
                                f"ran {self.entrypoint}(task, response) in the sandbox on the "
                                "request above"
                            ),
                            "observed": f"the grader returned the bool {value}",
                        },
                    ),
                ),
            }

        if kind == "number":
            try:
                value = float(payload.get("value"))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                return {
                    "verdict": "grader_error",
                    "score": None,
                    "errors": (
                        f"the grader returned {payload.get('value')}, which the record cannot "
                        "carry: NaN and Infinity are refused at the model layer",
                    ),
                    "limitations": (),
                }
            return {"verdict": "scored", "score": value, "errors": (), "limitations": ()}

        return {
            "verdict": "grader_error",
            "score": None,
            "errors": (
                f"the grader returned a {payload.get('type')}, and the plain shape of D-65 is "
                "score(task, response) -> float",
            ),
            "limitations": (),
        }

    # --- the envelope ------------------------------------------------------------------------

    def _envelope(
        self,
        *,
        started_at: str,
        clock: float,
        seed: int,
        request: dict,
        request_digest: str,
        verdict: str,
        score: float | None,
        counters: Counters,
        tier: str,
        errors: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
        observations: dict[str, Any] | None = None,
        findings: tuple[ValidityFinding, ...] = (),
    ) -> EvidenceEnvelope:
        duration = quantise(max(time.monotonic() - clock, 0.0), 3)
        provenance = {
            "started": started_at,
            "duration_s": duration,
            "sandbox_tier": tier,
            "offline": True,
            "counters": counters.model_dump(exclude_none=True),
        }
        observations = dict(observations or {})
        observations["request"] = request
        return EvidenceEnvelope(
            run_id="run-" + secrets.token_hex(6),
            subject_digest=self._manifest.source_digest,
            manifest_digest=self._manifest.digest(),
            request_digest=request_digest,
            start=started_at,
            end=_utc_now(),
            seed=seed,
            verdict=verdict,
            score=score,
            components={},
            observations=observations,
            artifacts=(),
            counters=counters,
            provenance=provenance,
            errors=tuple(e for e in errors if e),
            limitations=tuple(limitations),
            findings=findings,
        )


