"""W4.3: the Price ledger (F1) and the selection-explained fraction (F2).

Four instruments over one identity. For any measurable feature `f` of a rollout, with
`z(θ) = E_{y~π_θ}[f(y)]`, a Fisher-preconditioned step gives, to first order,

    Δz_i = η · Cov_group(A, f_i) + ρ_i

`SelectionTerm` and `SelectionResidual` report the two halves per feature per step, which is F1.
`SelectionExplainedFraction` and `EffectiveStepSize` report the fit of the left side on the right
across steps, which is F2's `Λ` and `η_eff`.

**This is a measurement instrument on behavioural traits in a live run, not a derivation of an
update rule in parameter space.** Frank's Price partition of optimiser updates (arXiv 2507.18549) is
the nearest prior art and is a different use of the same equation; the full statement of the
distinction is at the top of `measure.ledger.price` and it belongs on page one rather than in a
footnote.

`Λ` is the validity certificate for every other Level 1 claim in the library. `RegimeCondition.
LINEAR_RESPONSE` names `selection.explained_fraction` as the quantity that measures it, so F1's own
envelope cannot be satisfied until F2 has run, and neither can the envelope of anything else that
expands to first order about the current step.

Both instruments need a record and a featuriser and nothing else, so they run at `RECORD` access on
a training run somebody else did. `measure.ledger.features` is the featuriser contract and a
surface bank that works on any record carrying turn text; `measure.ledger.labelled` adapts a
published per-rollout table, which is what puts a labelled series inside reach.

**Reading somebody else's run is four determinations, and they are types.** `measure.ledger.
imported` holds them: a reward as named components at stated resolutions, the optimiser's group
identity kept separate from the semantic problem, an explicit advantage convention carrying the
provenance of the determination, and typed step axes joined only through an alignment object. They
were four keyword arguments with plausible defaults until the first published artifact showed all
four defaults to be wrong at once, on a series this project had already published a number from.
`measure.ledger.reconstruct` is how a composition is checked against the trainer's own logs and
`measure.ledger.trainer_log` is how those logs are read without assuming what the keys are. All
three are exported here, because a module that only the tests and one experiment can reach is not
part of the library whatever its docstring says.
"""

from reward_lens.measure.ledger.explained import (
    EXPLAINED_ENVELOPE,
    EffectiveStepSize,
    LambdaFit,
    SelectionExplainedFraction,
    feature_scales,
    fit_lambda,
    lambda_by_step,
)
from reward_lens.measure.ledger.features import (
    RecordedFeatures,
    SurfaceFeatures,
    TrajectoryFeaturiser,
    assistant_text,
    matrix_of,
    surface_features,
)
from reward_lens.measure.ledger.imported import (
    UNDETERMINED_CONVENTION,
    AdvantageConvention,
    AlignmentStrength,
    AxisAlignment,
    ComponentResolution,
    ComposedReward,
    ConventionSource,
    DeclaredConvention,
    DeclaredFloor,
    FloorReachability,
    ImportedRun,
    JoinedSeries,
    PromptGroups,
    ProxyReward,
    RewardComponent,
    RolloutColumns,
    StepAxis,
    check_floors,
    identity_alignment,
    index_gaps,
    join_on_axis,
    recover_prompt_groups,
    semantic_collisions,
)
from reward_lens.measure.ledger.labelled import (
    AISI_RUN,
    LabelRate,
    StepAxisReport,
    check_step_axis,
    label_rate,
    parse_hack_config,
    rate_series,
    read_parquet,
    steps_from_table,
)
from reward_lens.measure.ledger.nulls import (
    NullResult,
    permuted_advantage_null,
    permuted_step_null,
    random_feature_null,
    summarise,
)
from reward_lens.measure.ledger.prediction import transition_window
from reward_lens.measure.ledger.price import (
    LEDGER_ACCESS,
    LEDGER_ENVELOPE,
    Differential,
    LedgerRow,
    SelectionResidual,
    SelectionTerm,
    StepLedger,
    StepSample,
    advantages_from_rewards,
    learning_rates,
    ledger_between,
    ledger_series,
    selection_differential,
    steps_from_run,
)
from reward_lens.measure.ledger.reconstruct import (
    DEFAULT_SHIFTS,
    MAX_SIMPLE_NUMERATOR,
    REGISTERED_ANALYSIS_LAGS,
    SIMPLE_DENOMINATORS,
    ClosureResidual,
    CompositionFit,
    InformativeGroups,
    ShiftCurve,
    StepGroups,
    SubsetResult,
    closure_residuals,
    compose,
    fit_composition,
    group_statistics,
    informative_group_counts,
    predicted_reward_std,
    search_subsets,
    shift_residual_curve,
    snap_to_simple_ratio,
    steps_and_means,
)
from reward_lens.measure.ledger.trainer_log import (
    STEP_KEY,
    KeyCensus,
    Series,
    TrainerLog,
    quantisation_step,
    read_trainer_state,
    rounding_rms,
)

__all__ = [
    "AISI_RUN",
    "AdvantageConvention",
    "AlignmentStrength",
    "AxisAlignment",
    "ClosureResidual",
    "ComponentResolution",
    "ComposedReward",
    "CompositionFit",
    "ConventionSource",
    "DEFAULT_SHIFTS",
    "DeclaredConvention",
    "DeclaredFloor",
    "Differential",
    "EXPLAINED_ENVELOPE",
    "EffectiveStepSize",
    "FloorReachability",
    "ImportedRun",
    "InformativeGroups",
    "JoinedSeries",
    "KeyCensus",
    "LEDGER_ACCESS",
    "LEDGER_ENVELOPE",
    "LabelRate",
    "LambdaFit",
    "LedgerRow",
    "MAX_SIMPLE_NUMERATOR",
    "NullResult",
    "PromptGroups",
    "ProxyReward",
    "REGISTERED_ANALYSIS_LAGS",
    "RecordedFeatures",
    "RewardComponent",
    "RolloutColumns",
    "SIMPLE_DENOMINATORS",
    "STEP_KEY",
    "SelectionExplainedFraction",
    "SelectionResidual",
    "SelectionTerm",
    "Series",
    "ShiftCurve",
    "StepAxis",
    "StepAxisReport",
    "StepGroups",
    "StepLedger",
    "StepSample",
    "SubsetResult",
    "SurfaceFeatures",
    "TrainerLog",
    "TrajectoryFeaturiser",
    "UNDETERMINED_CONVENTION",
    "advantages_from_rewards",
    "assistant_text",
    "check_floors",
    "check_step_axis",
    "closure_residuals",
    "compose",
    "feature_scales",
    "fit_composition",
    "fit_lambda",
    "group_statistics",
    "identity_alignment",
    "index_gaps",
    "informative_group_counts",
    "join_on_axis",
    "label_rate",
    "lambda_by_step",
    "ledger_between",
    "ledger_series",
    "learning_rates",
    "matrix_of",
    "parse_hack_config",
    "permuted_advantage_null",
    "permuted_step_null",
    "predicted_reward_std",
    "quantisation_step",
    "random_feature_null",
    "rate_series",
    "read_parquet",
    "read_trainer_state",
    "recover_prompt_groups",
    "rounding_rms",
    "search_subsets",
    "selection_differential",
    "semantic_collisions",
    "shift_residual_curve",
    "snap_to_simple_ratio",
    "steps_and_means",
    "steps_from_run",
    "steps_from_table",
    "summarise",
    "surface_features",
    "transition_window",
]
