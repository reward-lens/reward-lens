"""The report projection of the one D-47 product join."""

from __future__ import annotations

from html import escape
from typing import Any

from reward_lens.product import join as product_join

__all__ = ["render_join"]


def _content(value, evidence: str | None = None) -> str:
    if isinstance(value, product_join.Absence):
        body = (
            '<span data-join-state="absent">'
            + escape(value.reason)
            + "; remedy: "
            + escape(value.remedy)
            + "</span>"
        )
        return body
    if isinstance(value, product_join.Estimate):
        label = f"{value.value} {value.unit}"
    else:
        label = "yes" if value else "no"
    if evidence:
        target = f"assay-entry-{escape(evidence, quote=True)}"
        body = f'<a href="#{target}" data-evidence-entry="{escape(evidence, quote=True)}">{escape(label)}</a>'
    else:
        body = f"<span>{escape(label)}</span>"
    return body


def _cell(value, evidence: str | None = None) -> str:
    return f"<td>{_content(value, evidence)}</td>"


def render_join(record: Any) -> str:
    """Render the saved record's join without constructing a second join model."""
    value = product_join.join(record)
    headings = "".join(
        f'<th scope="col">{escape(heading)}</th>' for heading in product_join.HEADINGS
    )
    rows = "".join(
        "<tr>"
        + f'<th scope="row">{escape(row.family)}</th>'
        + _cell(row.selected_in_your_run, row.evidence.get("selected_in_your_run"))
        + _cell(row.applied_update_weight, row.evidence.get("applied_update_weight"))
        + _cell(row.fix_changes_pressure, row.evidence.get("fix_changes_pressure"))
        + _cell(row.cost_of_the_fix, row.evidence.get("cost_of_the_fix"))
        + "</tr>"
        for row in value.rows
    )
    signal = _content(value.signal, value.signal_evidence)
    return (
        '<section class="panel report-join" data-section="join"><h2>The join</h2>'
        '<div class="scroller" tabindex="0" role="region" aria-label="Join measurements">'
        '<table class="kv"><thead><tr><th scope="col">family</th>'
        + headings
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div>"
        + f'<p data-join-signal="true">signal: {signal}</p></section>'
    )


def render_evidence_targets(record: Any) -> str:
    """Anchors for the actual saved measurement entries the join cells cite."""
    from reward_lens.product.join import _entry_index

    entries = _entry_index(record if isinstance(record, dict) else record.to_dict())
    return "".join(
        f'<span id="assay-entry-{escape(entry_id, quote=True)}" data-assay-entry="{escape(entry_id, quote=True)}"></span>'
        for entry_id in sorted(entries)
    )
