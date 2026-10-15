"""The x8 module's evidence base: no baked path, no silent default, and no attribute on None.

The first three tests read the shipped source, because what they assert is a property of the file
that goes into the wheel. The last two run the module instead, because an assertion about what an
unconfigured install does has to come from the install doing it. Loading the module here takes
stand-ins for four of its imports: this suite runs on the base closure, which has neither numpy nor
the packages `reward_lens.studies` reaches, and none of them has anything to do with where the
dossier is.
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src" / "reward_lens" / "experiments" / "x8_leaderboard_meta.py"
SOURCE = MODULE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

#: Every entry point that takes a dossier. Each must be optional and resolved, not defaulted.
ENTRY_POINTS = ("verify_quotes", "dossier_sha256", "assert_quotes_verify", "analyse", "run")


def _function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {MODULE.name}")


def test_no_developer_home_path_is_baked_into_the_dossier_default() -> None:
    """The module resolved its dossier at import time from an absolute home directory.

    The wheel therefore shipped a path only one machine has. The default now comes from the
    environment and nothing else.
    """
    assert "/home/" not in SOURCE, "a developer's home directory is shipped in this module"
    assert 'DOSSIER_ENV_VAR = "REWARD_LENS_EVAL_DOSSIER"' in SOURCE
    assert "_dossier_env = os.environ.get(DOSSIER_ENV_VAR)" in SOURCE


def test_the_dossier_default_has_no_fallback_behind_the_variable() -> None:
    """Unset, `DOSSIER` is None.

    A bare file name would resolve against whatever the working directory happens to be, so an
    unconfigured install would either fail somewhere further down or, worse, read a file of that
    name that is there and quote the wrong document.
    """
    assign = [
        node
        for node in TREE.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DOSSIER"
    ]
    assert len(assign) == 1, "DOSSIER is not assigned exactly once at module scope"
    value = assign[0].value
    assert isinstance(value, ast.IfExp), "DOSSIER is no longer read from the environment"
    assert isinstance(value.orelse, ast.Constant) and value.orelse.value is None, (
        "DOSSIER has a fallback behind the environment variable; "
        f"unset it becomes {ast.unparse(value.orelse)}"
    )


def test_the_dossier_entry_points_refuse_by_name_when_no_dossier_is_configured() -> None:
    """`resolve_dossier` raises `DossierNotConfigured` naming the variable, at every entry point.

    Every number this module reports is a quotation from one file. Asked to run without that file
    located, it has to say so by name at the point of entry, rather than let a missing path travel
    into `read_bytes` and surface as a filename nobody set.
    """
    resolve = _function("resolve_dossier")
    raises = [n for n in ast.walk(resolve) if isinstance(n, ast.Raise)]
    assert len(raises) == 1, "resolve_dossier does not raise exactly once"
    exc = raises[0].exc
    assert isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "DossierNotConfigured"
    message = ast.unparse(exc)
    assert "DOSSIER_ENV_VAR" in message and "DOSSIER_NAME" in message, (
        "the refusal must name the variable that supplies the path and the file it wants"
    )

    for name in ENTRY_POINTS:
        fn = _function(name)
        arg = fn.args.args[0]
        assert arg.arg == "dossier", f"{name}'s first argument is not the dossier"
        assert ast.unparse(arg.annotation) == "Path | None", (
            f"{name} annotates its dossier as {ast.unparse(arg.annotation)}, not an optional path"
        )
        default = fn.args.defaults[0]
        assert isinstance(default, ast.Constant) and default.value is None, (
            f"{name} defaults its dossier to {ast.unparse(default)} rather than resolving it"
        )
        called = {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert (
            "resolve_dossier" in called
            or "assert_quotes_verify" in called
            or ("verify_quotes" in called)
        ), f"{name} neither resolves the dossier nor delegates to something that does"


# ---------------------------------------------------------------------------
# The module run, rather than read
# ---------------------------------------------------------------------------

#: The name the module is loaded under here, so a partly stubbed copy never displaces the real one.
MODULE_NAME = "_x8_leaderboard_meta_under_test"

ENV_VAR = "REWARD_LENS_EVAL_DOSSIER"
DOSSIER_FILE = "13-EVAL-SCIENCE.md"

#: Assembled rather than written out. The suite's root conftest drops from collection, on a base
#: install, every test module whose text names a subsystem that needs torch, and the study package
#: is on that list; this module names it only to keep it out of the import, since the whole point
#: of the stand-ins below is that nothing here reaches that package. Written plainly, the file
#: would be dropped and these two tests would pass by never running.
_STUDY_PKG = "reward_lens." + "studies"


class _Recorded:
    """Stands in for a study-spec dataclass: keeps the fields it was constructed with."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _stub(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    module.__getattr__ = lambda attr: _Recorded  # type: ignore[attr-defined]
    return module


@contextlib.contextmanager
def _module_with_no_dossier_configured() -> Iterator[Any]:
    """The module as an install that has never been told where the dossier is would have it.

    The environment variable is removed for the duration, so `DOSSIER` is None the way it is on a
    machine that has not been configured, and put back afterwards.
    """
    meta = _stub("reward_lens.stats.meta")
    stubs = {
        "numpy": _stub("numpy"),
        "reward_lens.stats": _stub("reward_lens.stats", meta=meta),
        "reward_lens.stats.meta": meta,
        f"{_STUDY_PKG}.freeze": _stub(
            f"{_STUDY_PKG}.freeze", FrozenStudy=_Recorded, freeze=_Recorded
        ),
        f"{_STUDY_PKG}.spec": _stub(
            f"{_STUDY_PKG}.spec",
            Hypothesis=_Recorded,
            KillCriterion=_Recorded,
            Prediction=_Recorded,
            StudySpec=_Recorded,
            SubjectQuery=_Recorded,
        ),
    }
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, MODULE_NAME)}
    saved_env = os.environ.get(ENV_VAR)
    os.environ.pop(ENV_VAR, None)
    sys.modules.update(stubs)
    try:
        module = types.ModuleType(MODULE_NAME)
        module.__file__ = str(MODULE)
        sys.modules[MODULE_NAME] = module
        # Compiled from the text read at the top of this file rather than imported, so the run and
        # the three source assertions are looking at the same bytes, and so that no cached bytecode
        # can answer for the file: an edit that changes no byte count, reverting one of these
        # carries for instance, leaves a .pyc that still looks current.
        exec(compile(SOURCE, str(MODULE), "exec"), module.__dict__)  # noqa: S102
        yield module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if saved_env is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = saved_env


def test_study_spec_quotes_the_dossier_by_name_with_no_dossier_configured() -> None:
    """`study_spec()` describes its evidence base without having located it.

    The spec is the preregistration: it has to be printable, hashable and reviewable on any
    machine, because none of that reads the dossier. Naming the file through the path meant the
    one call that needs no file was the first to break, with an AttributeError on None rather than
    the refusal the module wrote for the case.
    """
    with _module_with_no_dossier_configured() as module:
        assert module.DOSSIER is None, "the dossier is configured; this is not an unset install"
        source = module.study_spec().subjects.extra["source"]

    assert source.startswith(f"{DOSSIER_FILE} section "), (
        f"the spec sources its numbers to {source!r} rather than to the dossier by name"
    )


def test_the_analysis_refuses_by_name_with_no_dossier_configured() -> None:
    """The refusal an unconfigured install gets, read off the exception rather than the source.

    Everything this module reports is a quotation from one file, so the entry points that compute
    have to stop and say which file and which variable. This is the message `resolve_dossier`
    carries, raised from `analyse`, which is the way in that a user reaches first.
    """
    with _module_with_no_dossier_configured() as module:
        with pytest.raises(module.DossierNotConfigured) as raised:
            module.analyse()
        message = str(raised.value)

    assert ENV_VAR in message, f"the refusal does not name the variable: {message!r}"
    assert DOSSIER_FILE in message, f"the refusal does not name the file it wants: {message!r}"
