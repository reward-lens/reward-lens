"""D-77: no function in the slice's nine directories is only a refusal.

`fleet/status.py` clause 5 holds the package-wide line. This is the slice-local ratchet underneath
it, so a stub reintroduced into `studies`, `policy`, `data`, `access`, `dynamics`, `runtime`,
`loops`, `geometry` or `stats` fails at this packet's own boundary instead of waiting for the
package-wide scan, and the six names D-77 removed cannot come back as names that raise.

A function is flagged when its body, after any docstring, is exactly one statement and that
statement is ``raise NotImplementedError`` (with a message or without), ``pass``, or ``...``.

Three exemptions, all structural, and no per-site list. A function carrying ``@abstractmethod`` or
``@abc.abstractmethod`` raises as its contract, not as a gap. A member of a ``typing.Protocol``
class is the same thing said structurally instead of nominally: its ``...`` is the declaration of
an interface, the body is unreachable by construction, and there is nothing there to implement.
A site the controller has already classed ``abstract`` in `fleet/ALLOWED_INCOMPLETE.md` is allowed
too; that file records each site as ``path:line``, but line numbers move under every edit, so the
match here is on the path plus the function name the entry's own reason spells out, never on the
recorded line. A ``deferred`` entry exempts nothing: deferred is the class this test exists to
catch, and clearing one means deleting the code or writing it, not adding a line to the allow list.

The Protocol exemption is why this test is a ratchet and not a tripwire. Clause 5's own `MARK_RE`
in `fleet/status.py` is ``TODO | FIXME | raise NotImplementedError | pass  # stub``, so a bare
``...`` was never a package-wide marker, and the wave-0 baseline in the allow list accordingly
carries no Protocol entry at all. Widening the rule to ``pass`` and ``...`` is what makes this
stricter than the package-wide line; exempting Protocol members is what keeps the widening true.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import reward_lens.data

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SLICE = "studies-runtime"
_SLICES_JSON = _ROOT / "fleet" / "port" / "SLICES.json"
_ALLOWED_MD = _ROOT / "fleet" / "ALLOWED_INCOMPLETE.md"

#: The six public names D-77 removed rather than implement. They are listed here so the deletion is
#: locked by a test and not only by a MIGRATION.md row.
D77_REMOVED = (
    "style_controls",
    "quadruples",
    "tournament_from_judges",
    "train_rm_pythia",
    "estimate_llc",
    "essential_dynamics",
)

#: `- \`path:line\` | <packet or controller> | <class> | <reason>`, the one line shape the allow
#: list defines for itself. The file's own format note starts with a backtick, so it never matches.
_ENTRY = re.compile(r"^-\s+`([^`:]+):([^`]*)`\s*\|([^|]*)\|([^|]*)\|(.*)$")

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _slice_dirs() -> list[pathlib.Path]:
    """The slice's nine directories, read from the ledger that defines the slice."""
    paths = json.loads(_SLICES_JSON.read_text(encoding="utf-8"))[_SLICE]
    dirs = [_ROOT / p for p in paths]
    assert len(dirs) == 9, f"{_SLICE} should name nine directories, got {len(dirs)}"
    for d in dirs:
        assert d.is_dir(), f"{d} is not a directory"
    return dirs


def _sources() -> list[tuple[str, pathlib.Path]]:
    """Every `.py` file under the slice's directories, as (path relative to the repo root, path)."""
    out: list[tuple[str, pathlib.Path]] = []
    for d in _slice_dirs():
        for py in sorted(d.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            out.append((py.relative_to(_ROOT).as_posix(), py))
    assert out, "the slice's directories hold no Python source"
    return out


def _allowed_abstract() -> tuple[dict[str, set[str]], set[str]]:
    """The `abstract` entries of the allow list, as path -> names, plus the whole-file paths.

    The names come out of each entry's reason, which spells the site out (``BaseBaseline.score
    abstract; overridden in ...``). Matching on those and on the path, rather than on the recorded
    line, is what keeps the exemption stable when the file above the site grows or shrinks.
    """
    by_path: dict[str, set[str]] = {}
    whole_file: set[str] = set()
    for raw in _ALLOWED_MD.read_text(encoding="utf-8").splitlines():
        m = _ENTRY.match(raw.strip())
        if not m:
            continue
        path, line, _owner, cls, reason = (g.strip() for g in m.groups())
        if cls != "abstract":
            continue
        if line == "*":
            whole_file.add(path)
            continue
        by_path.setdefault(path, set()).update(_IDENT.findall(reason))
    return by_path, whole_file


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the function is declared abstract, as `@abstractmethod` or `@abc.abstractmethod`."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod":
            return True
    return False


def _is_protocol(cls: ast.ClassDef) -> bool:
    """Whether the class is a `typing.Protocol`, however the base is spelled or parameterised."""
    for base in cls.bases:
        if isinstance(base, ast.Subscript):  # Protocol[T]
            base = base.value
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def _protocol_members(tree: ast.Module) -> set[int]:
    """Every function node declared inside a Protocol class, by node identity."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_protocol(node):
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(id(child))
    return out


def _stub_kind(stmt: ast.stmt) -> str | None:
    """What kind of empty body this single statement is, or None if it is real code."""
    if isinstance(stmt, ast.Pass):
        return "pass"
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    ):
        return "..."
    if isinstance(stmt, ast.Raise):
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        name = None
        if isinstance(exc, ast.Attribute):
            name = exc.attr
        elif isinstance(exc, ast.Name):
            name = exc.id
        if name == "NotImplementedError":
            return "raise NotImplementedError"
    return None


def _stub_sites() -> list[tuple[str, str, int, str]]:
    """Every flagged (path, function name, line, kind) in the slice, exemptions already applied."""
    by_path, whole_file = _allowed_abstract()
    found: list[tuple[str, str, int, str]] = []
    for rel, py in _sources():
        if rel in whole_file:
            continue
        allowed = by_path.get(rel, set())
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=rel)
        protocol = _protocol_members(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_abstract(node) or id(node) in protocol or node.name in allowed:
                continue
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if len(body) != 1:
                continue
            kind = _stub_kind(body[0])
            if kind is not None:
                found.append((rel, node.name, node.lineno, kind))
    return found


def _exported_names() -> dict[str, set[str]]:
    """Each source file's `__all__`, as far as it is a literal list or tuple of strings."""
    out: dict[str, set[str]] = {}
    for rel, py in _sources():
        names: set[str] = set()
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), filename=rel)):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                names.update(
                    e.value
                    for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
        if names:
            out[rel] = names
    return out


def test_no_function_in_the_slice_is_only_a_refusal():
    """No unexempted function in the nine directories has a body that is only a raise, pass or ...

    The failure names every site, because the remedy is per-site: write the function, delete it, or
    have the controller class it `abstract` in the allow list with a reason. Adding a `deferred`
    line does not silence this.
    """
    sites = _stub_sites()
    assert not sites, (
        "functions with an empty body in the "
        + _SLICE
        + " slice:\n"
        + "\n".join(f"  {rel}:{line} {name}() -> {kind}" for rel, name, line, kind in sorted(sites))
    )


def test_the_six_d77_names_are_defined_nowhere_in_the_slice():
    """The six names D-77 removed are not back as functions anywhere in the nine directories."""
    back: list[str] = []
    for rel, py in _sources():
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), filename=rel)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in D77_REMOVED
            ):
                back.append(f"{rel}:{node.lineno} {node.name}")
    assert not back, "D-77 names redefined: " + ", ".join(back)


def test_the_six_d77_names_are_exported_nowhere_in_the_slice():
    """No `__all__` in the nine directories re-exports one of the six removed names."""
    leaks = [
        f"{rel}: {name}"
        for rel, names in _exported_names().items()
        for name in sorted(names & set(D77_REMOVED))
    ]
    assert not leaks, "D-77 names still exported: " + ", ".join(leaks)


def test_the_data_package_no_longer_offers_the_three_removed_builders():
    """`reward_lens.data` is torch-free and importable, so the removal is checked at runtime too."""
    for name in ("style_controls", "quadruples", "tournament_from_judges"):
        assert not hasattr(reward_lens.data, name), f"reward_lens.data still exposes {name}"
        assert name not in reward_lens.data.__all__, f"{name} is still in reward_lens.data.__all__"
