"""The instrument base wave 2 plugs into, frozen here (wave-2 interfaces section 2)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import get_type_hints

import pytest

from reward_lens import contracts
from reward_lens.instruments.base import Corpus, Instrument, Panel, RunContext, Subject
from reward_lens.product.audit import registry


def context() -> RunContext:
    return RunContext(sandbox=None, project=None, budget="0.00", offline=True, clock=time.monotonic)


def test_run_context_provenance_is_what_every_entry_carries() -> None:
    provenance = context().provenance()
    assert isinstance(provenance, contracts.EntryProvenance)
    assert provenance.offline is True
    assert provenance.sandbox_tier
    assert provenance.duration_s >= 0.0


def test_run_context_absence_stamps_that_provenance(tmp_path: Path) -> None:
    ctx = context()
    entry = ctx.absence(
        "signal",
        "signal.contrast_fraction",
        "the fraction of groups carrying a within-group contrast",
        "no sampled responses grouped by prompt",
        "supply --responses with at least two responses per prompt",
        subject_ref="sha256:" + "0" * 64,
    )
    assert entry.kind == "absence"
    assert entry.state == "absent"
    assert entry.provenance.offline is True
    assert entry.absence.state == "NOT_MEASURED"


def test_the_budget_is_named_budget_and_budget_usd_reads_it() -> None:
    """The interfaces name the field `budget`. `budget_usd` survives one release, read-only."""
    ctx = RunContext(budget="1.50")
    assert ctx.budget == "1.50"
    assert ctx.budget_usd == "1.50"
    with pytest.raises(AttributeError):
        ctx.budget_usd = "2.00"  # type: ignore[misc]
    with pytest.raises(TypeError):
        RunContext(budget_usd="1.50")  # type: ignore[call-arg]


def test_the_clock_is_a_field_the_runner_sets() -> None:
    ticks = iter([1.0, 2.5])
    ctx = RunContext(clock=lambda: next(ticks))
    assert ctx.clock() == 1.0
    assert ctx.clock() == 2.5


def test_panel_holds_its_section_and_its_instruments() -> None:
    panel = Panel(section="validity", instruments=())
    assert panel.section == "validity"
    assert panel.instruments == ()


def test_the_protocol_is_structural() -> None:
    class Stub:
        id = "validity.stub"
        section = "validity"

        def run(self, subject, corpus, ctx):  # noqa: ANN001, ANN201
            return []

    assert isinstance(Stub(), Instrument)


def test_the_frozen_types_are_the_contracts_types() -> None:
    """`section` is the record's section type, and the operands are named, not erased to `Any`."""
    assert get_type_hints(Instrument)["section"] == contracts.Section
    hints = get_type_hints(Instrument.run)
    assert hints["subject"] is Subject
    assert hints["corpus"] is Corpus
    assert hints["ctx"] is RunContext
    assert hints["return"] == list[contracts.Entry]
    assert get_type_hints(Panel)["section"] == contracts.Section


def test_subject_and_corpus_are_structural() -> None:
    """An instrument packet types against these without importing `product/audit/`."""

    class StubSubject:
        grader_path = Path("grader.py")
        entrypoint = "score"
        project_dir = None
        tasks_path = None
        responses_path = None
        trainer = None
        name = "grader"
        source = "def score(task, response): return 0.0\n"

    class StubCorpus:
        rollouts = ()
        origin = "nothing was supplied"
        tasks = ()
        n_tasks = 0

        def as_list(self):  # noqa: ANN201
            return []

        def __len__(self) -> int:
            return 0

    assert isinstance(StubSubject(), Subject)
    assert isinstance(StubCorpus(), Corpus)


def test_the_audit_corpus_satisfies_the_corpus_protocol() -> None:
    from reward_lens.product.audit.instruments import AuditCorpus

    assert isinstance(AuditCorpus(rollouts=(), origin="nothing was supplied"), Corpus)


def test_discover_walks_the_instruments_package() -> None:
    panels = registry.discover()
    assert isinstance(panels, tuple)
    for panel in panels:
        assert isinstance(panel, Panel)
        assert panel.section in contracts.models.Measurement.model_fields


def test_discover_finds_a_panel_a_package_registered(tmp_path: Path) -> None:
    """The real registration path: a package, a module, a module-level `PANEL`, no edit here."""
    package = tmp_path / "fake_instruments"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "soundness.py").write_text(
        "from reward_lens.instruments.base import Panel\n"
        "\n"
        "class Battery:\n"
        "    id = 'soundness.battery'\n"
        "    section = 'soundness'\n"
        "\n"
        "    def run(self, subject, corpus, ctx):\n"
        "        return []\n"
        "\n"
        "PANEL = Panel.of('soundness', [Battery()])\n",
        encoding="utf-8",
    )
    (package / "_private.py").write_text("PANEL = 'not a panel'\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        found = registry.discover("fake_instruments")
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.startswith("fake_instruments")]:
            del sys.modules[name]
    assert [panel.section for panel in found] == ["soundness"]
    assert [instrument.id for instrument in found[0].instruments] == ["soundness.battery"]
    assert registry.import_failures == {}
