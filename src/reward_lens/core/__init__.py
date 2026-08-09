"""``reward_lens.core`` — types, evidence, provenance, the gates, and the store.

This package is the frozen contract of the kernel: the identity types, the Evidence atom, the three
gates, and the append-only store. Changing anything here is an ADR-level decision (section 4.6).

Every public name below resolves lazily, through PEP 562's module ``__getattr__``. That is not
tidiness, it is correction C-004 under decision D-58. This module used to import fourteen
submodules eagerly, two of which reach for numpy and pydantic-settings at their own module scope,
so ``import reward_lens.core.reading`` — which needs nothing but the standard library — pulled the
numeric stack in through its parent package. `uvx` downloads the base closure on every cold
invocation, which made the install-in-seconds claim cost a numpy download, and it made the three
instruments the wave-1 audit is built on unimportable on a base install.

What a caller sees is unchanged. ``from reward_lens.core import Evidence`` works, ``dir()`` lists
the whole surface so tab completion still works, an unknown name is still an ``AttributeError``
naming the module, and a type checker sees the eager form through the ``TYPE_CHECKING`` block
below. What changed is when a submodule is read: on first use of one of its names rather than on
import of the package.

One thing is deliberately *not* lazy, at the bottom of this file: the quantity catalogue still
loads at import. Section 4.2's lint rule ("an Instrument whose quantity is not registered fails at
import") is only meaningful if the registry is there to check against at import, and a lazy load
would make import order decide a contract. The catalogue is a packaged YAML read once and
``core.quantity`` pulls nothing outside the base closure, so it costs a file read rather than a
dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the eager form, for type checkers and for anyone reading the surface
    from reward_lens.core.budget import (
        DIVISORS,
        LIMITS,
        BudgetLintError,
        BudgetTerm,
        CalibrationCurve,
        IncrementalValidity,
        LimitOfDetection,
        LODCache,
        SubstrateKey,
        UncertaintyBudget,
        UncertaintyType,
        Verdict,
        budget_of,
        refuse_below_lod,
    )
    from reward_lens.core.closure import (
        ArcSpec,
        ClosureError,
        ClosureReport,
        CostBudget,
        Demand,
        Gap,
        GapKind,
        MetricBinding,
        Output,
        analyse,
        on,
        subject_key,
    )
    from reward_lens.core.config import Settings, get_settings, set_settings
    from reward_lens.core.envelope import (
        UNCONDITIONAL,
        ConditionReading,
        EnvelopeLintError,
        EnvelopeSpec,
        OnViolation,
        RegimeCondition,
        RegimeReading,
    )
    from reward_lens.core.errors import (
        CalibrationWarning,
        CapabilityError,
        ConformanceError,
        DataError,
        GaugeError,
        NumericsError,
        ProvenanceError,
        RegistryError,
        RewardLensError,
    )
    from reward_lens.core.evidence import (
        Evidence,
        Uncertainty,
        ValueCodec,
        evidence_from_envelope,
        make_evidence,
        register_payload,
    )
    from reward_lens.core.gates import (
        CalibrationRef,
        compute_trust,
        require_frame_for_comparison,
    )
    from reward_lens.core.provenance import Cost, Provenance, capture_provenance, git_sha
    from reward_lens.core.quantity import (
        DIMENSIONLESS,
        ESTIMATORS,
        FREE,
        QUANTITIES,
        TRIVIAL_GROUP,
        BaselineID,
        BiasDirection,
        BiasStatement,
        CostModel,
        EstimatorEntry,
        EstimatorID,
        InvarianceGroupID,
        LoadReport,
        Quantity,
        QuantityID,
        ReferenceID,
        Unit,
        best_estimator,
        catalogue_path,
        ladder,
        load_quantities,
        open_quantities,
        register_estimator,
        register_quantity,
        what_would_it_take,
    )
    from reward_lens.core.reading import (
        REASON_MEANING,
        Reading,
        ReadingResult,
        Refusal,
        RefusalReason,
        bounded_refusal,
        is_refusal,
        refuse_access,
        refuse_incomplete,
        refuse_undefined,
        value_or_none,
    )
    from reward_lens.core.reference import (
        CalibrationChain,
        ChainLevel,
        MatrixDescription,
        ReferenceKind,
        ReferenceMaterial,
        Transfer,
        ladder_disagreement,
        uncertified_refusal,
    )
    from reward_lens.core.registry import (
        CARD_SECTIONS,
        DATASETS,
        INTERVENTIONS,
        OBSERVABLES,
        ORACLES,
        ORGANISMS,
        SIGNALS,
        Registry,
    )
    from reward_lens.core.store import EvidenceStore, default_store, set_default_store
    from reward_lens.core.types import (
        Capability,
        DatasetID,
        DirectionID,
        EvidenceID,
        FrameID,
        GaugeStatus,
        ModelFP,
        OrganismID,
        Site,
        Span,
        StudyID,
        SubjectRef,
        TrustLevel,
        content_hash,
        hash_bytes,
    )

#: Public name -> the submodule of ``reward_lens.core`` that defines it. Generated from the eager
#: block above, and the two have to agree: a name here the block does not import is a name a type
#: checker cannot see, and a name in the block missing here does not resolve at runtime.
_LAZY: dict[str, str] = {
    "ArcSpec": "closure",
    "BaselineID": "quantity",
    "BiasDirection": "quantity",
    "BiasStatement": "quantity",
    "BudgetLintError": "budget",
    "BudgetTerm": "budget",
    "CARD_SECTIONS": "registry",
    "CalibrationChain": "reference",
    "CalibrationCurve": "budget",
    "CalibrationRef": "gates",
    "CalibrationWarning": "errors",
    "Capability": "types",
    "CapabilityError": "errors",
    "ChainLevel": "reference",
    "ClosureError": "closure",
    "ClosureReport": "closure",
    "ConditionReading": "envelope",
    "ConformanceError": "errors",
    "Cost": "provenance",
    "CostBudget": "closure",
    "CostModel": "quantity",
    "DATASETS": "registry",
    "DIMENSIONLESS": "quantity",
    "DIVISORS": "budget",
    "DataError": "errors",
    "DatasetID": "types",
    "Demand": "closure",
    "DirectionID": "types",
    "ESTIMATORS": "quantity",
    "EnvelopeLintError": "envelope",
    "EnvelopeSpec": "envelope",
    "EstimatorEntry": "quantity",
    "EstimatorID": "quantity",
    "Evidence": "evidence",
    "EvidenceID": "types",
    "EvidenceStore": "store",
    "FREE": "quantity",
    "FrameID": "types",
    "Gap": "closure",
    "GapKind": "closure",
    "GaugeError": "errors",
    "GaugeStatus": "types",
    "INTERVENTIONS": "registry",
    "IncrementalValidity": "budget",
    "InvarianceGroupID": "quantity",
    "LIMITS": "budget",
    "LODCache": "budget",
    "LimitOfDetection": "budget",
    "LoadReport": "quantity",
    "MatrixDescription": "reference",
    "MetricBinding": "closure",
    "ModelFP": "types",
    "NumericsError": "errors",
    "OBSERVABLES": "registry",
    "ORACLES": "registry",
    "ORGANISMS": "registry",
    "OnViolation": "envelope",
    "OrganismID": "types",
    "Output": "closure",
    "Provenance": "provenance",
    "ProvenanceError": "errors",
    "QUANTITIES": "quantity",
    "Quantity": "quantity",
    "QuantityID": "quantity",
    "REASON_MEANING": "reading",
    "Reading": "reading",
    "ReadingResult": "reading",
    "ReferenceID": "quantity",
    "ReferenceKind": "reference",
    "ReferenceMaterial": "reference",
    "Refusal": "reading",
    "RefusalReason": "reading",
    "RegimeCondition": "envelope",
    "RegimeReading": "envelope",
    "Registry": "registry",
    "RegistryError": "errors",
    "RewardLensError": "errors",
    "SIGNALS": "registry",
    "Settings": "config",
    "Site": "types",
    "Span": "types",
    "StudyID": "types",
    "SubjectRef": "types",
    "SubstrateKey": "budget",
    "TRIVIAL_GROUP": "quantity",
    "Transfer": "reference",
    "TrustLevel": "types",
    "UNCONDITIONAL": "envelope",
    "Uncertainty": "evidence",
    "UncertaintyBudget": "budget",
    "UncertaintyType": "budget",
    "Unit": "quantity",
    "ValueCodec": "evidence",
    "Verdict": "budget",
    "analyse": "closure",
    "best_estimator": "quantity",
    "bounded_refusal": "reading",
    "budget_of": "budget",
    "capture_provenance": "provenance",
    "catalogue_path": "quantity",
    "compute_trust": "gates",
    "content_hash": "types",
    "default_store": "store",
    "evidence_from_envelope": "evidence",
    "get_settings": "config",
    "git_sha": "provenance",
    "hash_bytes": "types",
    "is_refusal": "reading",
    "ladder": "quantity",
    "ladder_disagreement": "reference",
    "load_quantities": "quantity",
    "make_evidence": "evidence",
    "on": "closure",
    "open_quantities": "quantity",
    "refuse_access": "reading",
    "refuse_below_lod": "budget",
    "refuse_incomplete": "reading",
    "refuse_undefined": "reading",
    "register_estimator": "quantity",
    "register_payload": "evidence",
    "register_quantity": "quantity",
    "require_frame_for_comparison": "gates",
    "set_default_store": "store",
    "set_settings": "config",
    "subject_key": "closure",
    "uncertified_refusal": "reference",
    "value_or_none": "reading",
    "what_would_it_take": "quantity",
}

#: The submodules that are themselves public paths, so ``reward_lens.core.quantity`` is a
#: documented path rather than a reach into a private one.
_SUBMODULES: frozenset[str] = frozenset(
    {
        "budget",
        "closure",
        "config",
        "envelope",
        "errors",
        "evidence",
        "extras",
        "features",
        "gates",
        "invariance",
        "migrations",
        "provenance",
        "quantity",
        "reading",
        "reference",
        "registry",
        "store",
        "types",
    }
)

__all__ = [
    # plan closure (W2.5)
    "ArcSpec",
    "ClosureError",
    "ClosureReport",
    "CostBudget",
    "Demand",
    "Gap",
    "GapKind",
    "MetricBinding",
    "Output",
    "analyse",
    "on",
    "subject_key",
    # types
    "Capability",
    "TrustLevel",
    "GaugeStatus",
    "Site",
    "Span",
    "SubjectRef",
    "ModelFP",
    "DatasetID",
    "DirectionID",
    "FrameID",
    "EvidenceID",
    "StudyID",
    "OrganismID",
    "content_hash",
    "hash_bytes",
    # evidence
    "Evidence",
    "Uncertainty",
    "make_evidence",
    "evidence_from_envelope",
    "register_payload",
    "ValueCodec",
    # gates
    "CalibrationRef",
    "compute_trust",
    "require_frame_for_comparison",
    # provenance
    "Provenance",
    "Cost",
    "capture_provenance",
    "git_sha",
    # store
    "EvidenceStore",
    "default_store",
    "set_default_store",
    # registry
    "Registry",
    "SIGNALS",
    "OBSERVABLES",
    "INTERVENTIONS",
    "ORGANISMS",
    "DATASETS",
    "ORACLES",
    "CARD_SECTIONS",
    # config
    "Settings",
    "get_settings",
    "set_settings",
    # errors
    "RewardLensError",
    "CapabilityError",
    "GaugeError",
    "CalibrationWarning",
    "ConformanceError",
    "ProvenanceError",
    "RegistryError",
    "DataError",
    "NumericsError",
    # typed refusals and the reading wrapper (reading)
    "REASON_MEANING",
    "Reading",
    "ReadingResult",
    "Refusal",
    "RefusalReason",
    "bounded_refusal",
    "is_refusal",
    "refuse_access",
    "refuse_incomplete",
    "refuse_undefined",
    "value_or_none",
    # uncertainty budgets and the limit of detection (budget)
    "DIVISORS",
    "LIMITS",
    "BudgetLintError",
    "BudgetTerm",
    "CalibrationCurve",
    "IncrementalValidity",
    "LODCache",
    "LimitOfDetection",
    "SubstrateKey",
    "UncertaintyBudget",
    "UncertaintyType",
    "Verdict",
    "budget_of",
    "refuse_below_lod",
    # reference materials and the calibration chain (reference)
    "CalibrationChain",
    "ChainLevel",
    "MatrixDescription",
    "ReferenceKind",
    "ReferenceMaterial",
    "Transfer",
    "ladder_disagreement",
    "uncertified_refusal",
    # the validity envelope (envelope)
    "UNCONDITIONAL",
    "ConditionReading",
    "EnvelopeLintError",
    "EnvelopeSpec",
    "OnViolation",
    "RegimeCondition",
    "RegimeReading",
    # the quantity and estimator registries (quantity)
    "DIMENSIONLESS",
    "ESTIMATORS",
    "FREE",
    "QUANTITIES",
    "TRIVIAL_GROUP",
    "BaselineID",
    "BiasDirection",
    "BiasStatement",
    "CostModel",
    "EstimatorEntry",
    "EstimatorID",
    "InvarianceGroupID",
    "LoadReport",
    "Quantity",
    "QuantityID",
    "ReferenceID",
    "Unit",
    "best_estimator",
    "catalogue_path",
    "ladder",
    "load_quantities",
    "open_quantities",
    "register_estimator",
    "register_quantity",
    "what_would_it_take",
    # The modules themselves, so that a full path such as
    # ``reward_lens.core.quantity`` is a public path and not a reach into a
    # private one. `core.quantity.Registry` is the one name in the five
    # families above that is not re-exported flat: it is a different class from
    # `core.registry.Registry`, which already holds this package's `Registry`,
    # and re-exporting it would shadow. Reach it as
    # ``from reward_lens.core.quantity import Registry``.
    "budget",
    "closure",
    "config",
    "envelope",
    "errors",
    "evidence",
    "extras",
    "features",
    "gates",
    "invariance",
    "migrations",
    "provenance",
    "quantity",
    "reading",
    "reference",
    "registry",
    "store",
    "types",
]


def __getattr__(name: str) -> Any:
    """PEP 562. Resolve a public name by importing the submodule that defines it."""
    import importlib

    submodule = _LAZY.get(name)
    if submodule is not None:
        value = getattr(importlib.import_module(f"reward_lens.core.{submodule}"), name)
        globals()[name] = value  # cache, so the second lookup is an ordinary global
        return value
    if name in _SUBMODULES:
        module = importlib.import_module(f"reward_lens.core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep ``dir()`` and tab completion honest; a lazy module otherwise looks empty."""
    return sorted(set(__all__) | set(globals()))


# See the last paragraph of the module docstring: this one stays eager on purpose.
from reward_lens.core.quantity import load_quantities as _load_quantities  # noqa: E402

_LOAD_REPORT = _load_quantities()
