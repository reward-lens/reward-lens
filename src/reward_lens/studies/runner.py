"""The study runner: execute a frozen study into REGISTERED Evidence and adjudicate it (section 2.14).

The runner gives a science's analysis function a ``StudyRun`` handle: it resolves the subjects,
exposes ``measure`` (which runs an Observable under the study's id so the Evidence is REGISTERED by
gate 3, and appends it to the store), and collects the evidence ids the study produced. The
analysis computes the metrics its frozen predictions named and returns a StudyResult; the runner,
not the analysis, then checks each prediction and kill criterion against those metrics. That split
matters: the engine adjudicates against the frozen predictions, so a science cannot quietly move a
threshold after seeing the data (R12, I4).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from reward_lens.core.evidence import Evidence
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.core.types import FrameID
from reward_lens.measure.base import Context, Observable
from reward_lens.measure.base import run as run_observable
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import StudyResult, StudySpec


@dataclass
class StudyRun:
    """The handle an analysis function uses to measure and record under a frozen study.

    ``subjects`` holds the resolved objects (signals, organisms, datasets) keyed by id;
    ``measure`` runs an Observable with the study id threaded through the Context, so gate 3 stamps
    the Evidence REGISTERED and it is appended to the store; ``record`` appends an Evidence the
    analysis built itself. Every id produced is collected in ``evidence_ids`` so the report and the
    scoreboard can cite the study's adjudicating evidence.
    """

    study: FrozenStudy
    store: EvidenceStore
    subjects: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)

    def measure(
        self,
        observable: Observable,
        signal: Any,
        *,
        view: Any = None,
        readout: str = "reward",
        others: tuple = (),
        frame: FrameID | None = None,
        is_comparison: bool = False,
        regime: dict | None = None,
    ) -> Evidence:
        ctx = Context(
            signal=signal,
            view=view,
            readout=readout,
            others=others,
            frame=frame,
            study=self.study.study_id,
            is_comparison=is_comparison,
            regime=regime or {},
        )
        ev = run_observable(observable, ctx)
        self.store.append(ev)
        self.evidence_ids.append(ev.id)
        return ev

    def record(self, evidence: Evidence) -> str:
        self.store.append(evidence)
        self.evidence_ids.append(evidence.id)
        return evidence.id

    def signal(self, key: str) -> Any:
        return self.subjects[key]


def _resolve_analysis(path: str) -> Callable[[StudyRun], StudyResult]:
    module_path, _, func = path.rpartition(".")
    if not module_path:
        raise ValueError(f"analysis path '{path}' must be a dotted module.function path")
    module = importlib.import_module(module_path)
    fn = getattr(module, func, None)
    if fn is None or not callable(fn):
        raise ValueError(f"analysis '{path}' does not resolve to a callable")
    return fn  # type: ignore[no-any-return]


def run_study(
    spec_or_frozen: StudySpec | FrozenStudy,
    *,
    subjects: dict[str, Any] | None = None,
    store: EvidenceStore | None = None,
    analysis_fn: Callable[[StudyRun], StudyResult] | None = None,
) -> tuple[FrozenStudy, StudyResult]:
    """Run a study end to end: freeze if needed, execute the analysis, adjudicate against the spec.

    The study must be frozen before it runs (a spec is frozen here if a raw spec was passed, so the
    freeze provably predates the evidence). The analysis function (resolved from the spec's dotted
    path, or supplied directly for tests) computes the metrics its predictions named. The runner
    then checks each hypothesis's prediction and each kill criterion against those metrics, sets the
    outcomes, and returns the StudyResult with its adjudicating evidence ids attached. Refutations
    and fired kill criteria are recorded as prominently as confirmations (I4).
    """
    frozen = spec_or_frozen if isinstance(spec_or_frozen, FrozenStudy) else freeze(spec_or_frozen)
    spec = frozen.spec
    # An empty EvidenceStore is falsy (it defines __len__ but not __bool__), so `store or ...`
    # would silently swap a freshly-created empty store for the global default. Check identity.
    store = store if store is not None else default_store()
    run = StudyRun(study=frozen, store=store, subjects=subjects or {})

    fn = analysis_fn or _resolve_analysis(spec.analysis)
    result = fn(run)

    # Adjudicate against the FROZEN predictions, not whatever the analysis claimed.
    outcomes: dict[str, str] = {}
    for h in spec.hypotheses:
        value = result.metrics.get(h.prediction.metric)
        if value is None:
            outcomes[h.id] = "inconclusive"
        else:
            outcomes[h.id] = "confirmed" if h.prediction.check(float(value)) else "refuted"
    result.outcomes = outcomes

    killed_by = []
    for k in spec.kill_criteria:
        value = result.metrics.get(k.metric)
        if value is not None and k.fired(float(value)):
            killed_by.append(k.id)
    result.killed = bool(killed_by)
    result.killed_by = killed_by

    if not result.evidence:
        result.evidence = list(run.evidence_ids)
    return frozen, result


__all__ = ["StudyRun", "run_study"]
