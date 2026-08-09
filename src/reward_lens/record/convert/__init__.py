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

from reward_lens.record.convert.campaign import (
    SCHEMA_FINDINGS,
    Bank,
    CampaignStepStream,
    ConversionReport,
    campaign_arms,
    convert_campaign,
    count_run,
)
from reward_lens.record.convert.inspect_eval import (
    InspectEvalLog,
    InspectHeader,
    InspectSample,
    iter_scores,
    read_eval,
    read_header,
    verify_reductions,
)
from reward_lens.record.convert.instruments import (
    InstrumentOutcome,
    RecordSignal,
    SweepReport,
    access_declaration_findings,
    capabilities_in_record,
    context_for,
    is_record_only,
    reader_access,
    regime_over,
    run_instrument,
    shipped_instruments,
    sweep,
)
from reward_lens.record.convert.readjudicate import (
    CardReadjudication,
    ReadjudicationReport,
    frozen_study,
    load_frozen_specs,
    metric_arcs_from_reason,
    readjudicate,
    verify_spec_hash,
)
from reward_lens.record.convert.store import CampaignRow, CampaignStore

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
