"""The declared release environments must reach the imports of their test suites.

`tests/packaging/test_closure.py` asserts the shipping half of the closure: nothing under `src/`
imports a distribution the base dependencies do not carry. This file asserts the development half.
`release/validation-matrix.json` is the source for CI's Python profiles and this static closure
check. The dev profile is the shipped CPU/product surface run after a fresh CPU-only resolution
of the declared dev group; the all-extras profile runs those suites plus explicitly inventoried
root and acceptance regressions with all declared extras. TRL 1.10 and 1.13 conformance tests
run in dedicated exact-TRL CPU environments. Research tests importing the separate, unshipped top-level
`studies` or `experiments` packages, and two tests requiring an external v1 cache, remain in the
repository but are explicitly classified outside the production release gate. The inventory
assertion below refuses to let a newly added test disappear from both lists.

The check is static and cheap on purpose. It never runs `uv`, never imports a suite, and never needs
a populated environment: it parses `pyproject.toml` for what is declared, `uv.lock` for what those
declarations reach transitively, and the suites themselves with `ast`.

Three boundaries are deliberate, and each one is a claim about what this file can and cannot know.

*A guarded import is not a requirement.* Three suites import a distribution that lives in an extra
rather than in the dev group, and each guards it. `tests/conftest.py:24` reaches matplotlib behind
`importlib.util.find_spec` because matplotlib is in the `viz` extra.
`tests/test_verifier_fuzz.py:94` reaches crosshair behind `pytest.importorskip` because
`crosshair-tool` is in the `verifier` extra. `tests/product/test_audit.py:376` imports numpy inside
a `try`, and the `except ImportError` branch is where it asserts the COULD_NOT_CHECK hole, so that
suite is written to want numpy *absent*: installing numpy for it would turn its own central
assertion off. A test that demanded every import be declared would be wrong about all three. So an
import is required here only when the file guards it nowhere.

That rule is applied per file rather than per scope: a name guarded anywhere in a file is treated as
guarded throughout it. The sloppiness runs in the safe direction, since the cost is a missed
requirement rather than a demand for a distribution the suite was written to live without.

*Reachable, not declared.* An import satisfied transitively (pytest's `pluggy`, say) passes. The
requirement is that `uv sync --group dev` yields an environment the suite runs in, and a transitive
dependency does yield one.

*An import the suite makes through `reward_lens` is out of range.* `tests/render` tier B writes
Parquet by calling the library's writer, which imports pyarrow; no file under `tests/render` names
pyarrow at all. A static scan of the suites' own imports cannot see that, which is why the dev group
carries pyarrow with a comment naming the suite rather than because a test found it. What catches
that class is running the suite in a fresh environment, not this file.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

MATRIX = REPO / "release" / "validation-matrix.json"


def _matrix() -> dict:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "reward-lens.release-validation/1"
    assert data["version"] == "3.1.0"
    assert set(data["profiles"]) == {"dev", "all_extras", "trl_110", "trl_113"}
    return data


def _profile(name: str) -> dict:
    return _matrix()["profiles"][name]


# Import name to the distribution that provides it, where the two differ. A name absent from this
# table is assumed to normalise to its own distribution, which covers the common case. A wrong guess
# fails loudly with the suite and the import in the message, and the fix is a row here.
IMPORT_TO_DIST: dict[str, str] = {
    "yaml": "pyyaml",
    "jsonschema_rs": "jsonschema-rs",
    "hypothesis_jsonschema": "hypothesis-jsonschema",
    "dateutil": "python-dateutil",
    "crosshair": "crosshair-tool",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "pkg_resources": "setuptools",
    "attr": "attrs",
    "OpenSSL": "pyopenssl",
    "yaml_include": "pyyaml-include",
}

# Identifiers that mean "this module may be absent" when they appear in an `if` test.
_PROBES = frozenset({"find_spec", "importorskip", "has_module", "module_available"})
# What an `except` clause has to name for the `try` around an import to count as a guard.
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")


def _norm(name: str) -> str:
    """PEP 503 normalisation, so `SALib`, `sal_ib` and `sal-ib` are one key."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement(spec: str) -> tuple[str, frozenset[str]]:
    """`"pytest>=7.4.0"` to `("pytest", frozenset())`; `"reward-lens[trace]"` keeps the extra."""
    match = _REQUIREMENT.match(spec)
    if match is None:  # pragma: no cover - a malformed requirement is a pyproject bug
        raise AssertionError(f"unparseable requirement in pyproject.toml: {spec!r}")
    extras = match.group(2) or ""
    return _norm(match.group(1)), frozenset(e for e in (x.strip() for x in extras.split(",")) if e)


def _declared_roots(extras: tuple[str, ...] = ()) -> set[tuple[str, frozenset[str]]]:
    """Every distribution named by the base dependencies, the dev group, or an extra it includes.

    A dev-group entry naming this project with an extra (`reward-lens[trace]`) is expanded through
    `[project.optional-dependencies]`, because that is what "an extra the dev group includes" means.
    """
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    table = project["project"]
    optional = {
        _norm(name): list(specs) for name, specs in table.get("optional-dependencies", {}).items()
    }
    own = _norm(table["name"])

    pending = list(table.get("dependencies", []))
    pending += list(project.get("dependency-groups", {}).get("dev", []))
    pending += [f"{own}[{extra}]" for extra in extras]
    roots: set[tuple[str, frozenset[str]]] = set()
    expanded: set[str] = set()
    while pending:
        name, extras = _requirement(pending.pop())
        if name == own:
            for extra in sorted(_norm(e) for e in extras):
                if extra not in expanded:
                    expanded.add(extra)
                    pending.extend(optional.get(extra, []))
            continue
        roots.add((name, frozenset(_norm(e) for e in extras)))
    return roots


def _lock_graph() -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """`uv.lock` as two maps: distribution to its dependencies, and to its optional ones.

    Markers are ignored, which over-approximates: a dependency that applies only on Windows counts
    as reachable here. That is the safe direction for this question. What is asked is whether a
    declaration exists at all; platform-conditional reach is argued in `test_closure.py`.
    """
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    deps: dict[str, set[str]] = {}
    optional: dict[str, dict[str, set[str]]] = {}
    for package in lock.get("package", []):
        name = _norm(package["name"])
        edge = deps.setdefault(name, set())
        for dependency in package.get("dependencies", []):
            edge.add(_norm(dependency["name"]))
        for extra, entries in package.get("optional-dependencies", {}).items():
            optional.setdefault(name, {})[_norm(extra)] = {
                _norm(entry["name"]) for entry in entries
            }
    return deps, optional


def _reachable(extras: tuple[str, ...] = ()) -> set[str]:
    """The transitive closure of the declared roots through the lock."""
    deps, optional = _lock_graph()
    frontier: list[str] = []
    for name, root_extras in _declared_roots(extras):
        frontier.append(name)
        for extra in root_extras:
            frontier.extend(optional.get(name, {}).get(extra, ()))
    closure: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        closure.add(name)
        frontier.extend(deps.get(name, ()))
    return closure


def _first_party() -> set[str]:
    """Import names this repository provides, so a sibling module is not a missing distribution."""
    names = {"tests", "fleet", "conftest"}
    for base in (REPO, REPO / "src"):
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            if entry.suffix == ".py":
                names.add(entry.stem)
            elif entry.is_dir() and (entry / "__init__.py").exists():
                names.add(entry.name)
    return names


def _suite_paths(profile: str = "dev") -> list[str]:
    """The exact test and helper paths listed for this release profile."""
    return list(_profile(profile)["suites"])


def _imported_names(node: ast.AST) -> set[str]:
    """Top-level package names of every import anywhere under `node`."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            found.update(alias.name.split(".")[0] for alias in child.names)
        elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
            found.add(child.module.split(".")[0])
    return found


def _catches_import(handler: ast.ExceptHandler) -> bool:
    """Whether this `except` clause is one an absent module would land in."""
    if handler.type is None:
        return True
    candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in candidates:
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        if name in _IMPORT_ERRORS:
            return True
    return False


def _guarded_names(tree: ast.Module) -> set[str]:
    """Module names the file itself declares optional, by any of the three guards it uses."""
    guarded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "importorskip" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    guarded.add(first.value.split(".")[0])
        elif isinstance(node, ast.Try):
            if any(_catches_import(handler) for handler in node.handlers):
                for statement in node.body:
                    guarded |= _imported_names(statement)
        elif isinstance(node, ast.If):
            probes = {
                child.attr if isinstance(child, ast.Attribute) else getattr(child, "id", "")
                for child in ast.walk(node.test)
                if isinstance(child, (ast.Name, ast.Attribute))
            }
            if probes & _PROBES or any(p.endswith("_available") for p in probes if p):
                for statement in [*node.body, *node.orelse]:
                    guarded |= _imported_names(statement)
    return guarded


def _required_imports(path: Path) -> set[str]:
    """Every top-level package name the file imports without guarding it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imported_names(tree) - _guarded_names(tree)


def _python_files(path: Path, *, collectable_only: bool = False) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*.py")
        if "__pycache__" not in p.parts
        and (
            not collectable_only
            or p.name.startswith("test_")
            or p.name.endswith("_test.py")
            or p.name == "conftest.py"
        )
    )


def _provided_by_suite(name: str, directory: Path) -> bool:
    """Does `directory` itself provide the import `name`, as a module or as a package?

    A suite imports its own helpers by bare name. pytest prepends a test file's directory to
    `sys.path` when that directory carries no `__init__.py`, and `pythonpath = ["."]` puts the
    repository root there too. Both mechanisms import a package directory exactly as readily as
    a single file, so a check that stops at a module file reports a suite's own helper package
    as a distribution nobody declared. `tests/cli/fixtures/` is such a package, and it is what
    taught this function the second half of its condition.
    """
    return (directory / f"{name}.py").exists() or (directory / name / "__init__.py").exists()


def _third_party_imports_by_suite(profile: str = "dev") -> dict[str, dict[str, list[str]]]:
    """Suite to import name to the files requiring it, third-party unguarded imports only."""
    stdlib = set(sys.stdlib_module_names)
    local = _first_party()
    by_suite: dict[str, dict[str, list[str]]] = {}
    collectable_only = _profile(profile)["scan"] == "pytest_collectable"
    for suite in _suite_paths(profile):
        root = REPO / suite
        assert root.exists(), f"release matrix names a missing suite: {suite}"
        found: dict[str, list[str]] = {}
        for file in _python_files(root, collectable_only=collectable_only):
            for name in _required_imports(file):
                if name in stdlib or name in local or name.startswith("_"):
                    continue
                if _provided_by_suite(name, file.parent) or _provided_by_suite(name, root):
                    continue  # a module or package of the suite itself, on `sys.path` already
                found.setdefault(name, []).append(str(file.relative_to(REPO)))
        by_suite[suite] = found
    return by_suite


def test_the_release_suites_exist_and_cover_all_tests() -> None:
    """Every collected test is in the gate or explicitly classified as research."""
    for name in _matrix()["profiles"]:
        suites = _suite_paths(name)
        assert suites and len(suites) == len(set(suites)), name
        assert all((REPO / suite).exists() for suite in suites), name
    assert set(_suite_paths("dev")) <= set(_suite_paths("all_extras"))
    observed = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "tests").rglob("test_*.py")
        if "__pycache__" not in path.parts
    }
    included: set[str] = set()
    for profile in _matrix()["profiles"]:
        for suite in _suite_paths(profile):
            target = REPO / suite
            if target.is_file():
                if target.name.startswith("test_"):
                    included.add(suite)
            else:
                included.update(
                    path.relative_to(REPO).as_posix()
                    for path in target.rglob("test_*.py")
                    if "__pycache__" not in path.parts
                )
    historical = _matrix()["non_release_research_tests"]
    excluded = {entry["path"] for entry in historical}
    assert len(excluded) == len(historical), "duplicate research classification"
    assert all(entry["reason"] for entry in historical), "unexplained research classification"
    assert not included & excluded, "a test cannot be both gated and classified as research"
    assert observed == included | excluded, (
        f"release test inventory drift: ungated={sorted(observed - included - excluded)}, "
        f"stale={sorted((included | excluded) - observed)}"
    )
    assert "tests/fixtures" not in _suite_paths("dev"), (
        "fixtures/verifiers/rubric.py is pinned source data read by a test, not an imported suite"
    )
    assert set(_profile("dev")["python"]) == {"3.11", "3.12", "3.13"}
    assert set(_profile("all_extras")["python"]) == {"3.11", "3.12", "3.13"}
    assert _profile("trl_110")["python"] == ["3.11"]
    assert _profile("trl_113")["python"] == ["3.11"]
    assert _suite_paths("trl_110") == ["tests/learning", "tests/product/compare"]
    assert _suite_paths("trl_113") == ["tests/tap"]
    assert "tests/product/compare" not in _suite_paths("dev")
    assert "tests/product/compare" not in _suite_paths("all_extras")


@pytest.mark.parametrize("profile", ["dev", "all_extras", "trl_110", "trl_113"])
def test_declared_group_reaches_every_unguarded_import(profile: str) -> None:
    """The CI install for each profile must collect its exact suite list."""
    closure = _reachable(tuple(_profile(profile)["extras"]))
    undeclared: list[str] = []
    for suite, imports in sorted(_third_party_imports_by_suite(profile).items()):
        for name, files in sorted(imports.items()):
            dist = _norm(IMPORT_TO_DIST.get(name, name))
            if dist in closure:
                continue
            undeclared.append(
                f"{suite}: `import {name}` needs `{dist}`, which neither the base dependencies "
                f"nor the dev group reach (first at {files[0]})"
            )
    assert not undeclared, (
        f"a release suite in {profile} imports a distribution its declared environment does not install, "
        "and does not guard the import, so a clean checkout is red. Add it to [dependency-groups] "
        "dev or the relevant extra in pyproject.toml with a comment naming the suite, then run `uv lock`. If the import "
        "is meant to be optional, guard it where it is written. If the import name and the "
        "distribution name simply differ, the fix is a row in IMPORT_TO_DIST:\n  "
        + "\n  ".join(undeclared)
    )


@pytest.mark.parametrize("name", ["pytest", "pyarrow", "hypothesis-jsonschema", "pyyaml"])
def test_the_closure_is_computed_and_not_vacuous(name: str) -> None:
    """A closure that came out empty would let the assertion above pass for the wrong reason."""
    closure = _reachable()
    assert len(closure) > 10, f"the declared closure is {sorted(closure)}, which cannot be right"
    assert _norm(name) in closure, f"{name} is declared but is not in the computed closure"


def test_the_scan_visits_the_suites_and_sees_their_imports() -> None:
    """An empty walk would make the assertion above vacuous in the other direction."""
    by_suite = _third_party_imports_by_suite()
    assert by_suite, "no release suite was scanned"
    assert "pytest" in by_suite["tests/packaging"], (
        "tests/packaging imports pytest unguarded; a scan that misses it is broken"
    )
    assert "yaml" in by_suite["tests/contracts"], (
        "tests/contracts imports yaml unguarded; a scan that misses it is broken"
    )


def test_an_undeclared_import_would_be_caught(tmp_path: Path) -> None:
    """The negative control: the scanner has to report a plain unguarded import of a missing name."""
    module = tmp_path / "test_fake_suite.py"
    module.write_text("import nowhere_at_all\n", encoding="utf-8")
    assert _required_imports(module) == {"nowhere_at_all"}
    assert _norm("nowhere_at_all") not in _reachable()


def test_the_three_guards_are_recognised(tmp_path: Path) -> None:
    """Each guard the exit suites actually use, pinned so an edit cannot widen the rule silently."""
    module = tmp_path / "test_guards.py"
    module.write_text(
        "import os\n"
        "import importlib.util\n"
        "if importlib.util.find_spec('spec_guarded') is not None:\n"
        "    import spec_guarded\n"
        "try:\n"
        "    import try_guarded\n"
        "except ImportError:\n"
        "    try_guarded = None\n"
        "def test_one():\n"
        "    import pytest\n"
        "    pytest.importorskip('skip_guarded')\n"
        "    from skip_guarded.inner import thing\n"
        "    return thing\n"
        "def test_two():\n"
        "    import plainly_required\n"
        "    return plainly_required\n",
        encoding="utf-8",
    )
    required = _required_imports(module)
    assert {"spec_guarded", "try_guarded", "skip_guarded"} & required == set()
    assert "plainly_required" in required


def test_numpy_is_declared_for_the_forecast_exit_suites() -> None:
    """Forecast tests now import numpy directly, while the audit fixture keeps its absence branch.

    The development group is the environment that runs every wave-exit suite, so its numpy
    declaration is required by the Forecast controls. ``test_audit.py`` still guards its synthetic
    grader import and therefore continues to exercise its COULD_NOT_CHECK branch on base installs.
    """
    audit = REPO / "tests" / "product" / "test_audit.py"
    assert audit.exists(), "the audit control moved or disappeared; update the matrix and this test"
    assert "numpy" not in _required_imports(audit), (
        "tests/product/test_audit.py now imports numpy unguarded; revisit the audit fixture's "
        "base-install COULD_NOT_CHECK control"
    )
    dev = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {_requirement(spec)[0] for spec in dev.get("dependency-groups", {}).get("dev", [])}
    assert "numpy" in declared, "Forecast exit suites import numpy unguarded; dev must declare it"
