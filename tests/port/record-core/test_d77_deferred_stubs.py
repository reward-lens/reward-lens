"""D-77: the record-core slice carries no deferred stub.

A deferred stub is a function that was written as a placeholder: its whole body, once the
docstring is set aside, is a single ``raise NotImplementedError`` with any message or none, or a
single ``pass``, or a single ``...``. Three of them came in with this slice. They are removed
rather than implemented, because what they need is a GPU sibling-base population and a Hugging
Face Hub upload path, neither of which 4.0 has.

The test is an invariant over the source rather than an absence check against imported names,
for two reasons. A name that is gone today comes back under a different spelling tomorrow, and
this package imports ``torch``, so on a base install nothing here is importable at all.

Attempt 1 keyed the invariant on marker words in the docstring and the raise message, so a stub
spelled ``raise NotImplementedError("later")`` walked past it, and it required an ``ast.Raise``,
so ``pass`` and ``...`` bodies walked past it too. Both holes are closed here: the shape of the
body is the whole test, and no wording is required of it.

Three exemptions, because in each of them an empty body is the declaration rather than a
deferral:

* a function decorated ``@abstractmethod`` or ``@abc.abstractmethod``;
* an ``...`` or ``pass`` member of a class that declares an interface, meaning it derives from
  ``Protocol`` or ``ABC`` or sets ``metaclass=ABCMeta``. Such a body is never executed. A member
  of one that raises ``NotImplementedError`` is not exempt: that body does run, and the ratchet
  below is where it belongs;
* a site carried with class ``abstract`` in ``fleet/ALLOWED_INCOMPLETE.md``. Those entries are
  keyed ``path:line``, and line numbers drift, so the match here is on the path plus the function
  name. Every entry in that file opens its reason with the name of the site it covers, bare
  ("``corrupt_step`` docstring TODO(M8+)") or dotted ("``BaseObservable.measure`` abstract"), so
  the name read here is the leading dotted identifier of the reason, last segment. An entry whose
  line part is ``*`` covers the whole file. An ``abstract`` entry that opens with anything else
  exempts nothing, which is the intended pressure: lead the reason with the function name, or the
  site is flagged.

The walk covers this slice's four directories only. Clause 5 of the terminal predicate, scanned
by ``fleet/status.py``, holds the package-wide line.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Iterator

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src" / "reward_lens"
_SLICE = ("record", "core", "organisms", "signals")

_ALLOWED = _ROOT / "fleet" / "ALLOWED_INCOMPLETE.md"

#: ``- `path:line` | <packet or controller> | <class> | <reason>``, one allowed marker per line.
_ENTRY = re.compile(r"^-\s+`([^`:]+):(\d+|\*)`\s*\|([^|]*)\|([^|]*)\|(.*)$")
#: The site the reason covers, named first: ``corrupt_step ...`` or ``BaseObservable.measure ...``.
_LEADING_NAME = re.compile(r"^\s*`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def _modules() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for part in _SLICE:
        out.extend(sorted((_SRC / part).rglob("*.py")))
    return out


def _allowed_abstract() -> tuple[set[str], set[tuple[str, str]]]:
    """The ``abstract`` ratchet entries, as whole-file paths and as (path, function) pairs."""
    whole: set[str] = set()
    sites: set[tuple[str, str]] = set()
    if not _ALLOWED.is_file():
        return whole, sites
    for line in _ALLOWED.read_text(encoding="utf-8").splitlines():
        m = _ENTRY.match(line.strip())
        if m is None or m.group(3).strip() == "" or m.group(4).strip().lower() != "abstract":
            continue
        path = m.group(1).strip()
        if m.group(2) == "*":
            whole.add(path)
            continue
        named = _LEADING_NAME.match(m.group(5))
        if named is not None:
            sites.add((path, named.group(1).split(".")[-1]))
    return whole, sites


def _functions(
    tree: ast.AST, cls: ast.ClassDef | None = None
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.ClassDef | None]]:
    """Every function in ``tree``, paired with the class that encloses it, if any."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, cls
            yield from _functions(node, None)
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node, node)
        else:
            yield from _functions(node, cls)


def _stub_shape(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """What kind of empty body this function has, if its whole body is one, else ``None``.

    The docstring, and only a leading docstring, is set aside first. No wording is required of
    the raise: a bare ``raise NotImplementedError`` and one carrying any message both count.
    """
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1:
        return None
    stmt = body[0]
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
        name = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(name, ast.Name) and name.id == "NotImplementedError":
            return "raise NotImplementedError"
        if isinstance(name, ast.Attribute) and name.attr == "NotImplementedError":
            return "raise NotImplementedError"
    return None


def _is_abstract_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id in {"abstractmethod", "abstractproperty"}:
            return True
        if isinstance(target, ast.Attribute) and target.attr in {
            "abstractmethod",
            "abstractproperty",
        }:
            return True
    return False


def _declares_interface(cls: ast.ClassDef | None) -> bool:
    """Whether ``cls`` is a ``Protocol`` or an ABC, so an empty member body is a declaration."""
    if cls is None:
        return False
    for base in cls.bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Name) and target.id in {"Protocol", "ABC"}:
            return True
        if isinstance(target, ast.Attribute) and target.attr in {"Protocol", "ABC"}:
            return True
    for kw in cls.keywords:
        if kw.arg != "metaclass":
            continue
        value = kw.value
        if isinstance(value, ast.Name) and value.id == "ABCMeta":
            return True
        if isinstance(value, ast.Attribute) and value.attr == "ABCMeta":
            return True
    return False


def test_no_deferred_stub_in_the_slice() -> None:
    """No function in the slice has a placeholder body instead of an implementation."""
    whole_files, sites = _allowed_abstract()
    found: list[str] = []
    for path in _modules():
        rel = path.relative_to(_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, cls in _functions(tree):
            shape = _stub_shape(node)
            if shape is None:
                continue
            if _is_abstract_method(node):
                continue
            if shape in {"...", "pass"} and _declares_interface(cls):
                continue
            if rel in whole_files or (rel, node.name) in sites:
                continue
            found.append(f"{rel}:{node.lineno} {node.name} -> {shape}")
    assert found == [], "deferred stubs still present: " + "; ".join(found)


def test_the_removed_stub_names_are_gone_from_the_slice() -> None:
    """The three names D-77 removes are not reachable, exported, or registered anywhere."""
    gone = ("kinship_organism", "export_rewardbench_gt", "load_rewardbench_gt")
    hits = [
        f"{p.relative_to(_SRC)}: {n}"
        for p in _modules()
        for n in gone
        if n in p.read_text(encoding="utf-8")
    ]
    assert hits == [], "removed names still referenced: " + "; ".join(hits)


@pytest.mark.parametrize("part", _SLICE)
def test_every_module_in_the_slice_still_parses(part: str) -> None:
    """The removal left every module in the slice syntactically whole."""
    for path in sorted((_SRC / part).rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
