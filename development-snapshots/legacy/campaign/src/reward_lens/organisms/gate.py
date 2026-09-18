"""Wiring the calibration gate: scorecards to the measurement runner (section 1.3, gate 1).

The scorecard (`organisms/scorecard.py`) grades an instrument against planted ground truth and
produces a `CalibrationRef`. The measurement runner (`measure/base.py`) asks a pluggable provider,
at the moment it builds Evidence, whether the observable has a scorecard covering this subject and
regime. This module is the connection between the two: a registry of scorecard entries and the
provider that measure consults, plus ``install`` which registers that provider.

Keeping the connection here, rather than importing organisms into measure, preserves the dependency
direction (measure knows nothing of organisms; organisms installs itself into measure). ``install``
is a no-op until a scorecard is registered: with an empty registry the provider returns None and
every ad hoc number is correctly EXPLORATORY, so wiring the gate never silently upgrades anything.
Trust rises only once an instrument has actually earned a scorecard entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from reward_lens.core.gates import CalibrationRef
from reward_lens.core.types import SubjectRef
from reward_lens.measure.base import set_calibration_provider
from reward_lens.organisms.scorecard import ScorecardEntry, ScorecardSummary

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reward_lens.core.evidence import Evidence
    from reward_lens.core.store import EvidenceStore

# How `MethodScorecard.evaluate` stamps its Evidence: the store-facing observable name wraps the
# graded instrument's name, and the payload is a `ScorecardSummary`.
_SCORECARD_STAMP = "MethodScorecard["

# Scorecard entries indexed for the two lookups the provider needs: by observable name (the coarse
# match) and by (observable, organism_family) (the regime-aware match).
_BY_OBSERVABLE: dict[str, ScorecardEntry] = {}
_BY_FAMILY: dict[tuple[str, str], ScorecardEntry] = {}


def register_scorecard(observable: str, entry: ScorecardEntry) -> None:
    """Register a scorecard entry so downstream measurements of ``observable`` can cite it (gate 1).

    An instrument earns calibration by being graded against an organism family; registering the
    resulting entry is what makes that calibration visible to the runner. The entry is indexed both
    by observable name and by (observable, family), so a measurement that declares its regime gets
    the family-specific scorecard and one that does not still finds the observable's calibration.
    """
    _BY_OBSERVABLE[observable] = entry
    _BY_FAMILY[(observable, entry.calibration_ref.organism_family)] = entry


def scorecard_calibration_provider(
    observable: str, subject: SubjectRef, regime: dict
) -> CalibrationRef | None:
    """The gate-1 provider: return the CalibrationRef covering this observable and regime, or None.

    When the measurement's ``regime`` names an ``organism_family`` and a scorecard exists for that
    exact (observable, family) pair, the family-specific reference is returned. Otherwise, if the
    observable has any registered scorecard, its reference is returned so the number is CALIBRATED
    with the regime caveat carried in the reference's ``regime_match`` note. A missing entry returns
    None, which keeps the number EXPLORATORY.
    """
    family = regime.get("organism_family") if regime else None
    if family is not None and (observable, family) in _BY_FAMILY:
        return _BY_FAMILY[(observable, family)].calibration_ref
    entry = _BY_OBSERVABLE.get(observable)
    return entry.calibration_ref if entry is not None else None


def install() -> None:
    """Install the scorecard provider into the measurement runner (makes gate 1 live)."""
    set_calibration_provider(scorecard_calibration_provider)


def clear() -> None:
    """Forget all registered scorecards (used by tests to isolate the registry)."""
    _BY_OBSERVABLE.clear()
    _BY_FAMILY.clear()


def _entry_from_evidence(evidence: "Evidence") -> ScorecardEntry:
    """Rebuild a `ScorecardEntry` from stored scorecard Evidence.

    The store holds the summary payload and the Evidence around it, not the entry bundle, so
    the `CalibrationRef` is re-derived exactly as `MethodScorecard.evaluate` derived it: the
    reference cites the Evidence id, names the graded family, and carries the operating point
    at the detection dose when the instrument reached one.
    """
    summary = evidence.value
    if not isinstance(summary, ScorecardSummary):
        raise ValueError(
            f"evidence {evidence.id} does not carry a ScorecardSummary payload and cannot be "
            f"registered as a scorecard."
        )
    operating_point = None
    if summary.detects_rho_at is not None:
        for row in summary.operating_points:
            if row.get("rho") == summary.detects_rho_at:
                operating_point = row
                break
    ref = CalibrationRef(
        scorecard_entry=evidence.id,
        organism_family=summary.organism_family,
        regime_match="exact",
        operating_point=operating_point,
    )
    return ScorecardEntry(summary=summary, evidence=evidence, calibration_ref=ref)


def install_default_scorecards(
    store: "EvidenceStore", *, entries: "Mapping[str, str] | None" = None
) -> int:
    """Register every scorecard the store holds and make gate 1 live, in one call.

    The registry is in-memory and per-process, so calibration earned in one run is invisible to
    the next until something re-registers it. This is that something: it scans ``store`` for the
    Evidence `MethodScorecard.evaluate` stamps (observable ``MethodScorecard[<name>]`` with a
    `ScorecardSummary` payload), rebuilds an entry per (instrument, organism family) pair, keeping
    the most recent where a pair was graded more than once, registers each through
    `register_scorecard`, and calls `install`. After it returns, a measurement of any graded
    observable renders CALIBRATED and cites the stored scorecard, instead of every number in a
    fresh process falling back to EXPLORATORY.

    ``entries`` pins specific evidence: a ``{observable_name: evidence_id}`` mapping replaces
    whatever the scan found for those observables with the named Evidence. The named Evidence
    must be a scorecard for that same observable; registering one instrument's grades under
    another's name would let a measurement cite calibration it never earned, so a mismatch
    raises rather than aliasing silently.

    Returns the number of registrations made.
    """
    candidates: dict[tuple[str, str], "Evidence"] = {}
    for evidence in store:
        if not evidence.observable.startswith(_SCORECARD_STAMP):
            continue
        if not isinstance(evidence.value, ScorecardSummary):
            continue
        key = (evidence.value.observable, evidence.value.organism_family)
        current = candidates.get(key)
        if current is None or evidence.created_at > current.created_at:
            candidates[key] = evidence

    if entries:
        for observable, evidence_id in entries.items():
            pinned = store.get(evidence_id)
            summary = pinned.value
            if not isinstance(summary, ScorecardSummary):
                raise ValueError(
                    f"entries names {evidence_id} for {observable!r}, but that evidence does "
                    f"not carry a ScorecardSummary payload."
                )
            if summary.observable != observable:
                raise ValueError(
                    f"entries names {evidence_id} for {observable!r}, but that scorecard grades "
                    f"{summary.observable!r}; a scorecard only calibrates the instrument it "
                    f"graded."
                )
            for key in [k for k in candidates if k[0] == observable]:
                del candidates[key]
            candidates[(observable, summary.organism_family)] = pinned

    # Register in ascending created_at order: each (observable, family) pair keeps its own slot,
    # and the most recently earned scorecard wins the observable-level fallback slot.
    count = 0
    for evidence in sorted(candidates.values(), key=lambda e: e.created_at):
        entry = _entry_from_evidence(evidence)
        register_scorecard(entry.summary.observable, entry)
        count += 1
    install()
    return count


__all__ = [
    "register_scorecard",
    "scorecard_calibration_provider",
    "install",
    "install_default_scorecards",
    "clear",
]
