"""The manuscript claims checker (section 2.15.5, R-anti-self-deception).

This is the structural fix for the PAPER_DISCREPANCIES failure class: v1's paper numbers disagreed
with the CSVs (stale appendix tables, transposed rows, invented SNR values) and nobody could tell
which was authoritative. Here the evidence store is the single source of truth, and a document may
not claim a number the store does not contain. A claim is a value tagged with the Evidence id it
came from; the checker loads that Evidence, extracts the comparable value, and verifies the claim
within a tolerance. A tag pointing at an id the store does not have, or a value that disagrees with
the stored one, is a failure. It runs in CI over the repo's own docs.

Claim syntax, chosen to be readable in prose and unambiguous to parse:

    [[claim value=-0.171 ev=ev:ab12... field=per_model_mean_rho.Skywork tol=0.01]]

``value`` is the number as written in the prose; ``ev`` is the Evidence id; ``field`` (optional) is
a dotted path into the Evidence value when it is a dict or dataclass (omit it when the value is a
scalar); ``tol`` (optional) overrides the default absolute tolerance. The checker also verifies bare
``ev:...`` references resolve, so a citation to a nonexistent measurement is caught even without a
value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reward_lens.core.store import EvidenceStore, default_store

_CLAIM_RE = re.compile(r"\[\[claim\s+(?P<body>[^\]]+)\]\]")
_KV_RE = re.compile(r"(\w+)=(\S+)")
_BARE_EV_RE = re.compile(r"(?<![\w:])(ev:[0-9a-f]{8,})")


@dataclass
class ClaimResult:
    """The verification outcome for one claim."""

    claimed: float | None
    evidence_id: str
    field: str | None
    actual: Any
    ok: bool
    message: str


@dataclass
class ClaimReport:
    """The full result of checking a document."""

    results: list[ClaimResult] = field(default_factory=list)
    unresolved_refs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results) and not self.unresolved_refs

    @property
    def n_failures(self) -> int:
        return sum(1 for r in self.results if not r.ok) + len(self.unresolved_refs)

    def render(self) -> str:
        lines = [f"Claims checked: {len(self.results)}. Failures: {self.n_failures}."]
        for r in self.results:
            mark = "ok" if r.ok else "FAIL"
            lines.append(f"  [{mark}] {r.evidence_id} {r.field or ''}: {r.message}")
        for ref in self.unresolved_refs:
            lines.append(f"  [FAIL] {ref}: referenced but not in the store")
        return "\n".join(lines)


def _extract_field(value: Any, dotted: str | None) -> Any:
    if dotted is None:
        return value
    cur = value
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def check_text(
    text: str, store: EvidenceStore | None = None, default_tol: float = 1e-6
) -> ClaimReport:
    """Check every tagged claim in ``text`` against the store, returning a ClaimReport.

    A claim fails if its Evidence id is not in the store, if the named field cannot be extracted, or
    if the claimed value differs from the stored value by more than the tolerance. Bare ``ev:...``
    references that do not resolve are collected separately so a dangling citation is also caught.
    """
    store = store if store is not None else default_store()
    report = ClaimReport()
    tagged_ids: set[str] = set()

    for m in _CLAIM_RE.finditer(text):
        kv = dict(_KV_RE.findall(m.group("body")))
        ev_id = kv.get("ev", "")
        tagged_ids.add(ev_id)
        claimed = float(kv["value"]) if "value" in kv else None
        fld = kv.get("field")
        tol = float(kv.get("tol", default_tol))
        if ev_id not in store:
            report.results.append(
                ClaimResult(claimed, ev_id, fld, None, False, "evidence id not in the store")
            )
            continue
        ev = store.get(ev_id)
        try:
            actual = _extract_field(ev.value, fld)
        except (KeyError, AttributeError) as exc:
            report.results.append(
                ClaimResult(claimed, ev_id, fld, None, False, f"field '{fld}' not found ({exc})")
            )
            continue
        if claimed is None:
            report.results.append(ClaimResult(None, ev_id, fld, actual, True, "reference resolves"))
            continue
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            report.results.append(
                ClaimResult(claimed, ev_id, fld, actual, False, "stored value is not numeric")
            )
            continue
        diff = abs(actual_f - claimed)
        ok = diff <= tol
        msg = (
            f"claimed {claimed:g}, stored {actual_f:g}, |diff|={diff:g} <= tol {tol:g}"
            if ok
            else f"claimed {claimed:g} but stored {actual_f:g} (|diff|={diff:g} > tol {tol:g})"
        )
        report.results.append(ClaimResult(claimed, ev_id, fld, actual_f, ok, msg))

    # Bare ev: references not already covered by a claim tag: verify they resolve.
    for m in _BARE_EV_RE.finditer(text):
        ref = m.group(1)
        if ref in tagged_ids:
            continue
        if ref not in store:
            report.unresolved_refs.append(ref)

    return report


def check_files(paths: list[str | Path], store: EvidenceStore | None = None) -> ClaimReport:
    """Check a set of documents, aggregating into one report (the CI entry point)."""
    combined = ClaimReport()
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        rep = check_text(text, store)
        combined.results.extend(rep.results)
        combined.unresolved_refs.extend(rep.unresolved_refs)
    return combined


__all__ = ["ClaimResult", "ClaimReport", "check_text", "check_files"]
