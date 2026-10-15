"""The TRL completions converter is importable without the campaign store's dependencies.

`reward_lens.record.convert.__init__` used to import all five converters eagerly, so reading a TRL
completions parquet dragged in the campaign evidence store, `reward_lens.core.config` and
`pydantic_settings`, which is declared in no dependency group. The two import checks below run in
a fresh interpreter: an import that has already happened in this process cannot be unhappened, so
an in-process assertion would pass for the wrong reason forever after.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PUBLIC_NAMES = (
    "COMPLETIONS_GLOB",
    "CompletionsTable",
    "RESERVED_COLUMNS",
    "TRL_TAG",
    "TRL_VERIFIED_COMMAND",
    "TRL_VERSION",
    "WRITER_LOCATION",
    "WRITER_LOCATION_1_9_2",
    "completions_files",
    "read_completions",
    "read_directory",
)


def _fresh(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _top_level_bindings(path: Path) -> set[str]:
    """Every name one module binds at its top level, read without importing it."""
    bound: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
    return bound


def test_importing_the_converter_does_not_reach_pydantic_settings() -> None:
    done = _fresh(
        "import sys\n"
        "import reward_lens.record.convert.trl_parquet as m\n"
        "assert m.TRL_VERSION\n"
        "leaked = [n for n in sys.modules if n.split('.')[0] == 'pydantic_settings']\n"
        "print('LEAKED', leaked)\n"
    )
    assert done.returncode == 0, done.stderr
    assert "LEAKED []" in done.stdout, done.stdout


def test_the_converters_public_surface_is_reachable() -> None:
    done = _fresh(
        "from reward_lens.record.convert import trl_parquet\n"
        f"missing = [n for n in {PUBLIC_NAMES!r} if not hasattr(trl_parquet, n)]\n"
        "print('MISSING', missing)\n"
    )
    assert done.returncode == 0, done.stderr
    assert "MISSING []" in done.stdout, done.stdout


def test_every_name_the_package_declares_is_mapped_to_a_module_that_defines_it() -> None:
    """The lazy `__getattr__` must not quietly drop a name from `__all__`.

    The runtime form of this question, `hasattr(c, n)` for every `n` in `c.__all__`, cannot be
    asked in this build: nine of the declared names come from `campaign` and `store`, whose import
    chain reaches `reward_lens.core.config` and so `pydantic_settings`, a package this pyproject
    declares in no extra and no group. That is a packaging finding, recorded for P-PKG, not
    something a test may install around. So the same question is put to the source. Every declared
    name must be mapped to a converter module, and that module must bind the name at its top
    level, which is exactly what `__getattr__` will go looking for. A name `_SOURCES` still claims
    and its module no longer defines fails here, which is the regression the runtime form catches.
    """
    import reward_lens.record.convert as convert

    package = Path(convert.__file__).parent
    unmapped = sorted(
        name
        for name in convert.__all__
        if name not in convert._MODULE_OF and name not in convert._SOURCES
    )
    assert unmapped == [], unmapped

    for module, names in convert._SOURCES.items():
        bound = _top_level_bindings(package / f"{module}.py")
        undefined = sorted(set(names) - bound)
        assert undefined == [], (module, undefined)
