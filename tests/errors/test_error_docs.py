"""The committed pages under docs/errors/ are what the generator produces from the catalogue."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

from reward_lens.errors import CATALOGUE

DOCS = Path(__file__).resolve().parents[2] / "docs" / "errors"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rl_error_docs", DOCS / "generate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_is_a_page_for_every_code_and_no_page_without_one() -> None:
    pages = {p.stem for p in DOCS.glob("RL*.md")}
    assert pages == set(CATALOGUE)


def test_regenerating_reproduces_the_committed_pages(tmp_path: Path) -> None:
    _generator().write_pages(tmp_path)
    produced = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    committed = sorted(
        p.name for p in DOCS.iterdir() if p.is_file() and p.name != "generate.py"
    )
    assert produced == committed
    for name in produced:
        assert (tmp_path / name).read_text(encoding="utf-8") == (
            DOCS / name
        ).read_text(encoding="utf-8"), name


def test_the_index_lists_every_code() -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for code, spec in CATALOGUE.items():
        assert code in index, code
        assert spec.title in index, code


def test_no_page_names_an_internal_module_or_a_document_section() -> None:
    internal_module = re.compile(r"reward_lens[./][A-Za-z_]")
    document_section = re.compile(r"\b[A-Z]-[0-9]{1,3}\b|\bsection [0-9]|§")
    for page in sorted(DOCS.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        assert not internal_module.search(text), page.name
        assert not document_section.search(text), page.name
