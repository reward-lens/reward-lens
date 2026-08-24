"""The policy node, the peer of `signals/` (§2.1, §4.1, W5.1).

Section 2.1: the grader and the policy are the same kind of object, so an instrument that reads
internals should not care which side of the loop it points at, and the "pivot" between the two prior
designs is an argument value rather than a pivot. This package is that claim made structural. A
`PolicySubject` carries the same `Runtime` a `RewardSignal` carries and exposes the same readout,
score and capture surface, so `LensCrystallization`, `DirectLinearAttribution` and `PathEffect` run
against a policy with the subject as the only argument that changed, and no second implementation of
any of them exists.

Four modules:

``base``    the `PolicySubject` protocol, the sampling and gradient payloads, and `SiteWeights`,
            the contract that removed the last reach-through in the battery.
``arch``    where a decoder stack keeps its blocks, resolved structurally. The replacement for the
            eleven-method `ModelAdapter` ABC.
``hf``      `HFPolicyRuntime` and `HFPolicy`: the eager backend where gradients work.
``vllm``    and ``sglang``: the Plane A boundary, where they structurally cannot.

``recoverability`` is the package's own instrument and it is here because of §6.4 rather than
because a policy needs a probe: it produces the first reading in this library that carries an
`IncrementalValidity` record, which is what lint rule four has never had anything to check.

`credit.py` and `selection.py` appear in the §4.1 layout and are not in this package. They are W5.4
and W5.5, in a later wave, and an empty module with the right name is a worse placeholder than an
absent one: it imports, it exports nothing, and it makes a package look finished. The two work
packages own those files.

Importing this package does not import torch. Every torch reference is under `TYPE_CHECKING` or
inside the function that needs it, which is what `runtime/` and `signals/base` already do and what
lets `measure/battery/path.py` reach the `SiteWeights` contract without installing a dependency it
did not already have. A test asserts it, because the property is one edit away from being lost.
"""

from __future__ import annotations

from reward_lens.policy.arch import ArchitectureError, ArchitectureView, describe
from reward_lens.policy.base import (
    PolicyMeta,
    PolicySubject,
    Rollouts,
    SampleSpec,
    SiteWeights,
    TokenGradients,
    WeightsUnavailable,
    runtime_provenance,
    site_weights,
)
from reward_lens.policy.hf import (
    POLICY_CAPS,
    HFPolicy,
    HFPolicyRuntime,
    contrast_readout,
    from_pretrained,
    logit_readout,
    logprob_readout,
    wrap_hf_policy,
)
from reward_lens.policy.recoverability import PolicyReadoutProbe, Recoverability
from reward_lens.policy.sglang import SGLangPolicy, sglang_policy
from reward_lens.policy.vllm import ENGINE_LIMITS, SERVING_CAPS, EngineBoundary, ServingPolicy

__all__ = [
    "ENGINE_LIMITS",
    "POLICY_CAPS",
    "SERVING_CAPS",
    "ArchitectureError",
    "ArchitectureView",
    "EngineBoundary",
    "HFPolicy",
    "HFPolicyRuntime",
    "PolicyMeta",
    "PolicyReadoutProbe",
    "PolicySubject",
    "Recoverability",
    "Rollouts",
    "SGLangPolicy",
    "SampleSpec",
    "ServingPolicy",
    "SiteWeights",
    "TokenGradients",
    "WeightsUnavailable",
    "contrast_readout",
    "describe",
    "from_pretrained",
    "logit_readout",
    "logprob_readout",
    "runtime_provenance",
    "sglang_policy",
    "site_weights",
    "wrap_hf_policy",
]
