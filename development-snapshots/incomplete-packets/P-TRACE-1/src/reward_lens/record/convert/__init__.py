"""``reward_lens.record.convert`` — turning what a framework already recorded into a `Run`.

One converter so far, for the evidence store the 2.0 campaign produced. It is the first real test
of whether the canonical record of section 2.2 can hold something that was not designed for it, and
the answer is a qualified yes: the rollouts, their scores, the group structure and the capture
manifests all land, and six things do not fit without a departure. Those are constants in
`campaign.SCHEMA_FINDINGS` and they travel on every `ConversionReport`, because a finding that
lives only in a build report is a finding nobody reads twice.

Four modules, in dependency order:

`store` opens the campaign's evidence store read-only and decodes its payloads permissively,
because the fifteen dataclasses that wrote them are in a package this library does not depend on.

`campaign` builds the `Run`, lazily: 992 score banks and 8 ProcessBench banks over 616,023 items,
decoded one bank at a time.

`readjudicate` re-runs all twenty-seven preregistered cards through the void-aware runner, from
their own frozen specs and their own recorded metrics, which is the second half of W0.6.

`instruments` points the shipped battery at the converted record and classifies what comes back.
"""

from __future__ import annotations

import importlib
from typing import Any

# The converter modules are imported on first use, not when this package is imported. The eager
# form made every consumer of one converter pay for all five: reading a TRL completions parquet
# pulled in the campaign evidence store, which reaches `core.config` and so needs
# `pydantic-settings`, an optional dependency no converter on that path touches. The public surface
# is unchanged. Each name below resolves through `__getattr__` from the one module that defines it,
# and the module objects stay reachable under the same names.
_SOURCES: dict[str, tuple[str, ...]] = {
    "campaign": (
        "SCHEMA_FINDINGS",
        "Bank",
        "CampaignStepStream",
        "ConversionReport",
        "campaign_arms",
        "convert_campaign",
        "count_run",
    ),
    "inspect_eval": (
        "InspectEvalLog",
        "InspectHeader",
        "InspectSample",
        "iter_scores",
        "read_eval",
        "read_header",
        "verify_reductions",
    ),
    "instruments": (
        "InstrumentOutcome",
        "RecordSignal",
        "SweepReport",
        "access_declaration_findings",
        "capabilities_in_record",
        "context_for",
        "is_record_only",
        "reader_access",
        "regime_over",
        "run_instrument",
        "shipped_instruments",
        "sweep",
    ),
    "readjudicate": (
        "CardReadjudication",
        "ReadjudicationReport",
        "frozen_study",
        "load_frozen_specs",
        "metric_arcs_from_reason",
        "readjudicate",
        "verify_spec_hash",
    ),
    "store": ("CampaignRow", "CampaignStore"),
    "trl_parquet": (),
}

_MODULE_OF: dict[str, str] = {
    name: module for module, names in _SOURCES.items() for name in names
}


def __getattr__(name: str) -> Any:
    """Resolve one public name, or one converter module, on first use."""
    module = _MODULE_OF.get(name)
    if module is None:
        if name in _SOURCES:
            # A module reached as an attribute: `convert.store`, `convert.trl_parquet`. The
            # `readjudicate` module is the exception noted under `__all__`: the function of that
            # name shadows it, so it is reached by `importlib.import_module`.
            value: Any = importlib.import_module(f"reward_lens.record.convert.{name}")
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        value = getattr(importlib.import_module(f"reward_lens.record.convert.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_MODULE_OF, *_SOURCES})

__all__ = [
    "SCHEMA_FINDINGS",
    "Bank",
    "CampaignRow",
    "CampaignStepStream",
    "CampaignStore",
    "CardReadjudication",
    "ConversionReport",
    "InstrumentOutcome",
    "ReadjudicationReport",
    "RecordSignal",
    "SweepReport",
    "access_declaration_findings",
    "campaign_arms",
    "capabilities_in_record",
    "context_for",
    "convert_campaign",
    "count_run",
    "frozen_study",
    "is_record_only",
    "load_frozen_specs",
    "metric_arcs_from_reason",
    "readjudicate",
    "reader_access",
    "regime_over",
    "run_instrument",
    "shipped_instruments",
    "sweep",
    "verify_spec_hash",
    # reading a published Inspect AI `.eval` archive without importing the vendor package
    # (inspect_eval)
    "InspectEvalLog",  # header plus samples, as the archive carries them
    "InspectHeader",  # the eval header: task, model, config, and the recorded reductions
    "InspectSample",  # one sample with its per scorer values, unflattened
    "read_header",  # read only the header, without decoding every sample
    "read_eval",  # read the whole archive
    "verify_reductions",  # recompute the archive's own reductions and report where they disagree
    "iter_scores",  # stream one scorer's values across the archive
    # The modules themselves, so that a full path such as
    # ``reward_lens.record.convert.inspect_eval`` is a public path rather than a reach into a
    # private one. ``readjudicate`` is the one module missing from this list: the package already
    # exports a function of that name, so the module name cannot be added without shadowing it.
    # Reach it as ``from reward_lens.record.convert import readjudicate as readjudicate_module``
    # or by ``importlib.import_module``.
    "campaign",
    "inspect_eval",
    "instruments",
    "store",
]
