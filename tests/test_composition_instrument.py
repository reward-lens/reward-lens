"""Public panel regression for the composition instrument."""

from pathlib import Path
from types import SimpleNamespace

from reward_lens.instruments import composition
from reward_lens.product.audit.registry import discover
from reward_lens.product.trace import open_trace


def test_discovered_panel_keeps_a_real_trace_at_its_supported_evidence_level() -> None:
    """A reader-produced trace without trainer telemetry yields precise absences, never a proxy."""
    panel = next(panel for panel in discover() if panel is composition.PANEL)
    trace = open_trace(Path("tests/fixtures/grpo_run/short"))
    corpus = SimpleNamespace(trace=trace)
    subject = SimpleNamespace(source="def score(): return 1")
    entries = panel.instruments[0].run(subject, corpus, composition.RunContext(offline=True))

    by_id = {entry.entry_id: entry for entry in entries}
    assert by_id["trace.composition.advantages"].kind == "absence"
    assert (
        by_id["trace.composition.scalar_loss"].absence.missing_access
        == "no numerical scalar-loss telemetry was recorded"
    )
    assert by_id["trace.composition.gradient"].kind == "absence"
    assert by_id["trace.composition.optimizer_transform"].kind == "absence"
    assert by_id["trace.composition.parameter_update"].kind == "absence"
