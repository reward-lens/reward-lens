"""The Observable protocol, the measurement Context, and the gate-enforcing runner (section 2.8.1).

An Observable is a functional of a reward signal's internals on structured data (I1). Every one
declares what capability it requires (R3), how its value transforms under the gauge group
(``gauge_status``), and which formal theory object it instantiates (``faithful_to``) with an
explicit list of any departures (``deviations``). Those last two fields are the structural fix
for operationalization drift (liability 2): an Observable that computes a coverage statistic while
claiming Wang-Huang's distortion index must either match Appendix A or list the deviation, and the
deviation then surfaces on every card that consumes it.

The runner is where gates 1 and 2 are enforced before Evidence is returned, so no downstream code
can bypass them. This is a frozen interface (section 4.6): the whole battery and the index library
compile against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from reward_lens.core.errors import CapabilityError
from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef, require_frame_for_comparison
from reward_lens.core.provenance import Cost, Provenance, capture_provenance
from reward_lens.core.types import Capability, EvidenceID, FrameID, GaugeStatus, SubjectRef

if TYPE_CHECKING:
    from reward_lens.signals.base import RewardSignal

# ---------------------------------------------------------------------------
# The calibration provider (gate 1's seam)
# ---------------------------------------------------------------------------

# A calibration provider answers "is there a scorecard entry for this observable on this subject's
# signal family and data regime?" It is populated by `organisms.scorecard` once M4 lands; until
# then the default returns None and every ad hoc number is correctly EXPLORATORY. Making this a
# seam (rather than importing organisms into measure) keeps the dependency direction clean.
CalibrationProvider = Callable[[str, SubjectRef, dict], "CalibrationRef | None"]


def _no_calibration(observable: str, subject: SubjectRef, regime: dict) -> "CalibrationRef | None":
    return None


_PROVIDER: CalibrationProvider = _no_calibration


def set_calibration_provider(provider: CalibrationProvider) -> None:
    """Install the calibration provider (organisms.scorecard does this at import)."""
    global _PROVIDER
    _PROVIDER = provider


def lookup_calibration(
    observable: str, subject: SubjectRef, regime: dict | None = None
) -> "CalibrationRef | None":
    return _PROVIDER(observable, subject, regime or {})


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Everything an Observable needs to run, plus the machinery to emit gated Evidence.

    ``signal`` is the primary subject; ``others`` names additional signals for cross-signal
    comparisons (an effective-angle Observable puts the second model here). ``view`` is the
    DataView; ``readout`` selects which readout to read; ``frame`` supplies the gauge frame for
    covariant/invariant observables; ``study`` is the frozen StudyID when the run is registered
    (gate 3). ``regime`` describes the data regime for the calibration lookup (gate 1).

    Observables call ``emit`` to build their Evidence; ``emit`` applies gates 1 and 3 centrally so
    an Observable cannot forget them. The runner (`run`) applies the capability check and gate 2.
    """

    signal: "RewardSignal"
    view: Any = None
    readout: str = "reward"
    others: tuple["RewardSignal", ...] = ()
    frame: FrameID | None = None
    study: str | None = None
    is_comparison: bool = False
    regime: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    _observable: "Observable | None" = None

    def subject(self, extra: dict | None = None) -> SubjectRef:
        """Build the SubjectRef naming the signals, dataset, readout, frame, and interventions."""
        sigs = [self.signal] + list(self.others)
        fingerprints = tuple(s.meta.fingerprint for s in sigs)
        dataset = None
        if self.view is not None:
            dataset = getattr(self.view, "dataset_id", None)
            if dataset is None and hasattr(self.view, "checksum"):
                dataset = self.view.checksum()
        interventions = tuple(getattr(self.signal, "intervention_fingerprints", ()) or ())
        return SubjectRef(
            signals=fingerprints,
            dataset=dataset,
            readout=self.readout,
            frame=self.frame,
            interventions=interventions,
            extra=extra or {},
        )

    def emit(
        self,
        value: Any,
        *,
        uncertainty: Uncertainty | None = None,
        gauge: GaugeStatus | None = None,
        parents: tuple[EvidenceID, ...] = (),
        cost: Cost | None = None,
        subject_extra: dict | None = None,
    ) -> Evidence:
        """Build a gated Evidence for the current Observable's result.

        Applies gate 1 (looks up a calibration reference for this observable and subject; absent
        means EXPLORATORY) and gate 3 (a frozen study makes it REGISTERED). Gate 2's gauge status
        is taken from the Observable's declaration unless overridden. The trust level falls out of
        `make_evidence`, never set here directly.
        """
        obs = self._observable
        name = obs.name if obs else "anonymous"
        version = obs.version if obs else "0"
        gauge_status = gauge or (obs.gauge_status if obs else GaugeStatus.INVARIANT)
        subject = self.subject(subject_extra)
        calibration = lookup_calibration(name, subject, self.regime)
        prov = capture_provenance(parents=parents, study=self.study, cost=cost)
        # capture_provenance stamps git sha; merge in the explicit cost if provided.
        if cost is not None:
            prov = Provenance(
                git_sha=prov.git_sha,
                config_hash=prov.config_hash,
                seeds=prov.seeds,
                cost=cost,
                oracle_calls=prov.oracle_calls,
                parents=tuple(parents),
                study=self.study,
                extra=prov.extra,
            )
        return make_evidence(
            observable=name,
            observable_version=version,
            subject=subject,
            value=value,
            uncertainty=uncertainty,
            gauge=gauge_status,
            calibration=calibration,
            provenance=prov,
            registered=self.study is not None,
        )


# ---------------------------------------------------------------------------
# Observable protocol + runner
# ---------------------------------------------------------------------------


@runtime_checkable
class Observable(Protocol):
    """A measurement (section 2.8.1).

    ``requires`` is the capability the signal must declare; ``gauge_status`` is how the value
    transforms under the gauge group; ``faithful_to`` names the Appendix A theory object it
    instantiates (or None), and ``deviations`` lists explicit departures from it. ``measure``
    computes the Evidence, calling ``ctx.emit`` to build it so gates 1 and 3 are applied centrally.
    """

    name: str
    version: str
    requires: Capability
    gauge_status: GaugeStatus
    faithful_to: str | None
    deviations: tuple[str, ...]

    def measure(self, ctx: Context) -> Evidence: ...


def run(observable: Observable, ctx: Context) -> Evidence:
    """Run an Observable under the gates (section 2.8.1).

    Enforces R3 (the signal must declare the required capability) and gate 2 (a covariant
    cross-signal comparison requires a frame) before delegating to ``measure``. Gates 1 and 3 are
    applied inside ``ctx.emit``. The result is a fully gated Evidence; there is no path that
    returns an ungated number.
    """
    missing = observable.requires.missing_from(ctx.signal.caps)
    if missing and missing != Capability.NONE:
        raise CapabilityError(
            f"observable '{observable.name}' requires {observable.requires!r} but signal "
            f"{ctx.signal.meta.fingerprint} declares {ctx.signal.caps!r}; missing {missing!r}"
        )
    if ctx.is_comparison:
        require_frame_for_comparison(observable.gauge_status, ctx.frame)
    ctx._observable = observable
    try:
        return observable.measure(ctx)
    finally:
        ctx._observable = None


class BaseObservable:
    """A convenience base for Observables that sets the declaration fields as class attributes.

    Subclasses override the class attributes and implement ``measure``; this saves every
    Observable from restating the protocol fields. It is deliberately a plain class, not a
    dataclass: a dataclass ``__init__`` would overwrite a subclass's class-attribute overrides with
    the base defaults, so overriding ``requires``/``gauge_status`` in the subclass body would
    silently not take effect. Any object satisfying the protocol works; the battery and the index
    library use this base for uniformity.
    """

    name: str = "observable"
    version: str = "1.0"
    requires: Capability = Capability.SCORES
    gauge_status: GaugeStatus = GaugeStatus.INVARIANT
    faithful_to: str | None = None
    deviations: tuple[str, ...] = ()

    def measure(self, ctx: Context) -> Evidence:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = [
    "CalibrationProvider",
    "set_calibration_provider",
    "lookup_calibration",
    "Context",
    "Observable",
    "BaseObservable",
    "run",
]
