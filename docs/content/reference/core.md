# Core and evidence

**What does a measurement carry besides its value?** In `reward_lens.core` the answer is: an uncertainty with an honest sample size, a gauge status, a calibration reference, a provenance record, and a computed trust level. This subsystem is the epistemics layer, and it pulls only numpy, so `from reward_lens.core import ...` never loads torch.

The design and the plain-words tour of these objects live in [the anatomy of evidence](../discipline/anatomy-of-evidence.md) and [a measurement you can trust](../concepts/measurement-you-can-trust.md). This page is the exact surface.

## The evidence object

`Evidence` is a frozen, generic dataclass. You never build one field by field; `make_evidence` computes the trust level for you from the calibration and registration facts, so the value and its credentials cannot drift apart.

::: reward_lens.core.evidence.Evidence
    options:
      heading_level: 3

::: reward_lens.core.evidence.make_evidence
    options:
      heading_level: 3

The uncertainty is where the honesty lives. `n_effective` can fall far below `n` when rows are clones of each other, and `method` records how the interval was built, down to the `bootstrap-CLONE-INFLATED` stamp when someone opts into resampling correlated rows.

::: reward_lens.core.evidence.Uncertainty
    options:
      heading_level: 3

## Trust and gauge

Trust is an ordered ladder, computed by the gates and never set by the caller. `REGISTERED` outranks `CALIBRATED` outranks `EXPLORATORY`; `ADJUDICATED` is the top rung.

::: reward_lens.core.types.TrustLevel
    options:
      heading_level: 3

Gauge status says whether a number is a fact or a coordinate. `INVARIANT` survives a change of basis. `COVARIANT` is the "needs a frame" rung: it means nothing across models until a shared [frame](../discipline/gauge-and-frames.md) is supplied, and the comparison gate raises without one. `RAW_ONLY` is a bare coordinate.

::: reward_lens.core.types.GaugeStatus
    options:
      heading_level: 3

## Provenance and identity

A calibration reference is the token that lifts a measurement off the exploratory floor: it names the scorecard and the regime the instrument was validated on.

::: reward_lens.core.gates.CalibrationRef
    options:
      heading_level: 3

`SubjectRef` and `ModelFP` say what was measured; `Provenance` and `Cost` say how, recording the git sha, the seeds, the parent evidence, and the metered compute.

::: reward_lens.core.types.SubjectRef
    options:
      heading_level: 3

::: reward_lens.core.types.ModelFP
    options:
      heading_level: 3

::: reward_lens.core.provenance.Provenance
    options:
      heading_level: 3

::: reward_lens.core.provenance.Cost
    options:
      heading_level: 3

## The gates

Two of the three gates live here as plain functions. `compute_trust` is gate 1: it reads the calibration and registration facts and returns the trust level, downgrading to `EXPLORATORY` when there is no scorecard rather than raising.

::: reward_lens.core.gates.compute_trust
    options:
      heading_level: 3

`require_frame_for_comparison` is gate 2: it raises `GaugeError` when a `COVARIANT` quantity is asked to compare across subjects with no frame. This is the check that stops a raw cross-model cosine from being reported as if it meant something. The ladder as a whole is documented on [the trust ladder](../discipline/trust-ladder.md).

::: reward_lens.core.gates.require_frame_for_comparison
    options:
      heading_level: 3

## The store

Every measurement lands in an append-only store: a JSONL file you can append to, read, and search, but never edit or delete. It refuses derived evidence whose parents it cannot resolve, so the provenance graph stays whole. The design is on [the evidence store](../discipline/evidence-store.md).

::: reward_lens.core.store.EvidenceStore
    options:
      heading_level: 3

::: reward_lens.core.store.default_store
    options:
      heading_level: 3
