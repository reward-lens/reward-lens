"""The discovered signal panel: clustered reward contrast and component diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_lens import contracts
from reward_lens.instruments.base import Panel, RunContext
from reward_lens.product.audit.response_scores import ResponseScore
from reward_lens.product.trace import PlannedContrastDesign, StepMetric, TraceRecord

from .bootstrap import uncertainty_for_tasks, uncertainty_for_weighted_tasks
from .clustering import effective_sample_size, icc_interval
from .contrast import Metric, component_disagreement, contrast_fraction, style_share
from .null_result import null_result

__all__ = [
    "PANEL",
    "SignalInstrument",
    "SignalObservation",
    "component_disagreement",
    "contrast_fraction",
    "effective_sample_size",
    "null_result",
    "style_share",
    "uncertainty_for_tasks",
]

VERSION = "1.0.0"
FINDING_CODES = {
    "RL0241": "contrast present",
    "RL0242": "low effective sample size",
    "RL0243": "presentation carries reward",
    "RL0244": "components disagree",
    "RL0245": "TRL zero-std reconciliation differs",
}


@dataclass(frozen=True)
class SignalObservation:
    """One recorded rollout. ``None`` remains missing and is never converted to zero."""

    task_id: str
    group_id: str
    reward_applied: float | None
    components: Mapping[str, float | None] = field(default_factory=dict)
    trl_frac_reward_zero_std: float | None = None
    step: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "SignalObservation":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            task = value.get("task_id")
            group = value.get("group_id")
            if task is None or group is None:
                missing = ", ".join(
                    name
                    for name, field_value in (("task_id", task), ("declared group_id", group))
                    if field_value is None
                )
                raise ValueError(f"recorded signal row is missing {missing}")
            return cls(
                str(task),
                str(group),
                value.get("reward_applied"),
                dict(value.get("components") or {}),
                value.get("frac_reward_zero_std"),
                value.get("step"),
            )
        task, group = getattr(value, "task_id", None), getattr(value, "group_id", None)
        if task is None or group is None:
            missing = ", ".join(
                name
                for name, field_value in (("task_id", task), ("declared group_id", group))
                if field_value is None
            )
            raise ValueError(f"recorded signal row is missing {missing}")
        return cls(
            str(task),
            str(group),
            getattr(value, "reward_applied", None),
            dict(getattr(value, "components", {}) or {}),
            getattr(value, "frac_reward_zero_std", None),
            getattr(value, "step", None),
        )


@dataclass(frozen=True)
class SignalInstrument:
    id: str = "signal.clustered_reward"
    section: contracts.Section = "signal"

    @property
    def method(self) -> contracts.Method:
        return contracts.Method(
            id=self.id,
            version=VERSION,
            params_digest=contracts.digest(
                {
                    "covariance_model": "exchangeable_within_task",
                    "icc": "one_way_anova_unequal_m0",
                    "design_effect": "1+(m_A-1)*rho",
                    "resampling_unit": "task",
                    "resamples": 999,
                }
            ),
            procedure="compute per-task signal statistics; estimate the ICC by unequal-group one-way ANOVA; report n_eff under exchangeable within-task covariance; bootstrap complete tasks, use Rademacher wild task-cluster bootstrap with t(G-1) below 40 clusters, and report no interval below 15",
        )

    @property
    def response_bank_method(self) -> contracts.Method:
        """The separately predeclared method for executed response-bank scores."""
        return contracts.Method(
            id="signal.response_bank_score",
            version=VERSION,
            params_digest=contracts.digest(
                {
                    "score_source": "actual_python_grader_response_bank_execution",
                    "grouping": "declared_response_task_id",
                    "contrast": "within_task_any_finite_score_difference",
                    "covariance_model": "exchangeable_within_declared_task",
                    "resamples": 999,
                }
            ),
            procedure=(
                "score each joined response-bank row through the supplied local grader; group "
                "only by its declared task_id; compute the fraction of eligible prompt groups "
                "with any within-group finite grader-score contrast and its response-bank "
                "precision diagnostic"
            ),
        )

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        reference = contracts.digest({"source": str(getattr(subject, "source", ""))})
        trace = getattr(corpus, "trace", None)
        if trace is None:
            response_scores = getattr(corpus, "response_scores", None)
            if response_scores is not None:
                return self.response_bank_entries(
                    tuple(response_scores), ctx, subject_ref=reference
                )
            if hasattr(corpus, "response_scores"):
                return [
                    self._response_absence(
                        "signal.instrument_absent",
                        "the within-group contrast, the effective sample size, the style share and the disagreement between components (D-43)",
                        "needs sampled responses grouped by prompt",
                        "pass sampled responses with `--responses <file>`, grouped by prompt",
                        ("no claim that the score carries usable signal",),
                        ctx,
                        subject_ref=reference,
                    )
                ]
            return [
                ctx.absence(
                    "signal",
                    self.id,
                    "recorded reward signal over declared task groups",
                    "the audit corpus has no immutable TraceRecord; ordinary rollouts do not declare task groups or applied reward",
                    "supply one canonical trace record through AuditCorpus.trace",
                    (
                        "no claim about reward contrast, effective sample size, style share, or component disagreement",
                    ),
                    subject_ref=reference,
                    depends_on=("digest:training_semantics",),
                    method=self.method,
                )
            ]
        if not isinstance(trace, TraceRecord):
            return [
                ctx.absence(
                    "signal",
                    self.id,
                    "recorded reward signal over declared task groups",
                    "AuditCorpus.trace is not a canonical TraceRecord",
                    "supply the immutable record returned by reward_lens.product.trace",
                    (
                        "no claim about reward contrast, effective sample size, style share, or component disagreement",
                    ),
                    subject_ref=reference,
                    depends_on=("digest:training_semantics",),
                    method=self.method,
                )
            ]
        missing = sorted(
            {
                field_name
                for row in trace.rows
                for field_name, value in (
                    ("task_id", row.task_id),
                    ("declared group_id", row.group_id),
                )
                if value is None
            }
        )
        if missing:
            return [
                ctx.absence(
                    "signal",
                    self.id,
                    "recorded reward signal over declared task groups",
                    f"canonical TraceRecord has rows missing {', '.join(missing)}; identities are not inferred",
                    "record task_id and declared group_id for every row used by the signal panel",
                    (
                        "no claim about reward contrast, effective sample size, style share, or component disagreement",
                    ),
                    subject_ref=trace.subject_ref or reference,
                    depends_on=trace.depends_on,
                    method=self.method,
                )
            ]
        declared = trace.metadata.declared_reward
        component_universe = tuple(declared.partition) if declared is not None else None
        rows = tuple(
            SignalObservation(
                str(row.task_id),
                str(row.group_id),
                row.reward_applied,
                {name: row.components.get(name) for name in component_universe}
                if component_universe is not None
                else {},
                step=row.step,
            )
            for row in trace.rows
        )
        if not rows:
            return [
                ctx.absence(
                    "signal",
                    self.id,
                    "recorded reward signal over declared task groups",
                    "the canonical TraceRecord contains no rows",
                    "record task_id, declared group_id, reward_applied, and the declared component partition",
                    (
                        "no claim about reward contrast, effective sample size, style share, or component disagreement",
                    ),
                    subject_ref=trace.subject_ref or reference,
                    depends_on=trace.depends_on,
                    method=self.method,
                )
            ]
        return self.entries_for(
            rows,
            ctx,
            subject_ref=trace.subject_ref or reference,
            trl_metrics=trace.metadata.metrics,
            planned_contrast=trace.metadata.planned_contrast,
            declared_component_universe=component_universe,
            require_declared_component_universe=True,
        )

    def _response_absence(
        self,
        entry_id: str,
        measurand: str,
        missing_access: str,
        remedy: str,
        affected_claims: tuple[str, ...],
        ctx: RunContext,
        *,
        subject_ref: str,
    ) -> contracts.Entry:
        entry = ctx.absence(
            "signal",
            entry_id,
            measurand,
            missing_access,
            remedy,
            affected_claims,
            subject_ref=subject_ref,
            depends_on=("digest:source", "digest:samples"),
        )
        entry.method = self.response_bank_method
        # The canonical audit transcript groups this actionable missing input with its remedy.
        # Import locally: a direct instrument use does not need the audit package until it emits
        # this public-route absence.
        from reward_lens.product.audit.absences import SAMPLED_RESPONSES, needs_note

        needs_note(entry, SAMPLED_RESPONSES)
        return entry

    def response_bank_entries(
        self,
        scores: Sequence[ResponseScore],
        ctx: RunContext,
        *,
        subject_ref: str,
    ) -> list[contracts.Entry]:
        """Measure only response-bank grader-score variation, never training pressure."""
        by_status = {
            status: sum(row.status == status for row in scores)
            for status in ("measured", "invalid", "error")
        }
        rows = tuple(
            SignalObservation(
                task_id=row.task_id,
                group_id=row.task_id,
                reward_applied=row.score,
            )
            for row in scores
            if row.status == "measured" and row.score is not None
        )
        metric = contrast_fraction(rows)
        measured_note = (
            f"{by_status['measured']} finite scores were measured; "
            f"{by_status['invalid']} inputs were invalid and {by_status['error']} executions errored"
        )
        if metric.eligible == 0:
            return [
                self._response_absence(
                    "signal.response_score_contrast_fraction",
                    "fraction of declared prompt groups with within-group audited grader-score contrast",
                    measured_note + "; no declared prompt group has two finite scored responses",
                    "supply at least two response-bank rows per declared task_id that the grader scores finitely",
                    (
                        "no response-bank grader-score contrast claim",
                        "no claim about applied training reward, optimiser grouping, components, or training pressure",
                    ),
                    ctx,
                    subject_ref=subject_ref,
                )
            ]
        uncertainty = uncertainty_for_weighted_tasks(
            metric.task_values, metric.task_weights, seed=0
        )
        ess = effective_sample_size(rows)
        interval = icc_interval(rows)
        uncertainty = _record_uncertainty(uncertainty, ess, interval)
        common_result = {
            "value_definition": "executed response-bank grader-score contrast over declared prompt groups",
            "input_contract": {
                "tasks": "JSONL task records keyed by task_id",
                "responses": "JSONL responses carrying task_id and response text",
                "grouping": "response.task_id, never row adjacency",
                "score_source": "actual local PythonGrader invocation",
            },
            "scored_response_count": by_status["measured"],
            "invalid_response_count": by_status["invalid"],
            "execution_error_count": by_status["error"],
            "eligible_prompt_groups": metric.eligible,
            "raw_response_n": ess.raw_n,
            "prompt_clusters": ess.clusters,
            "m_0": ess.m_0,
            "m_A": ess.m_a,
            "icc_raw": ess.icc_raw,
            "icc_truncated_at_zero": ess.icc_truncated_at_zero,
            "icc_identified": ess.identifiable,
            "icc_reason": ess.reason,
            "design_effect": ess.design_effect,
            "n_eff": ess.n_eff,
        }
        limitations = (
            "these are scores freshly evaluated over a supplied response bank, not a recorded applied training reward",
            "the response bank supplies no optimiser steps, declared reward components, or evidence of training pressure",
        )
        contrast = contracts.estimate(
            "signal",
            "signal.response_score_contrast_fraction",
            "fraction of declared prompt groups with within-group audited grader-score contrast",
            round(metric.value, 6),
            "fraction",
            metric.eligible,
            "declared prompt group",
            uncertainty,
            subject_ref=subject_ref,
            provenance=ctx.provenance(),
            method=self.response_bank_method,
            result=common_result,
            depends_on=("digest:source", "digest:samples"),
            limitations=limitations,
            assumptions=(
                "task_id is the response bank's declared prompt identity",
                "within-task response-score covariance is exchangeable for this precision diagnostic",
            ),
            rate_definition="contrast_fraction",
        )
        if ess.n_eff is None:
            n_eff = self._response_absence(
                "signal.response_score_n_eff",
                "effective response-bank score count under exchangeable within-prompt covariance",
                ess.reason,
                "supply finite scored responses across at least two prompt groups with within-group variation",
                ("no response-bank effective-sample-size claim",),
                ctx,
                subject_ref=subject_ref,
            )
        else:
            n_eff_metric = Metric(
                ess.n_eff, tuple(float(ess.n_eff) for _ in range(ess.clusters)), ess.raw_n
            )
            n_eff_uncertainty = uncertainty_for_tasks(
                n_eff_metric.task_values, lambda values: sum(values) / len(values), seed=0
            )
            n_eff_uncertainty = _record_uncertainty(n_eff_uncertainty, ess, interval)
            n_eff = contracts.estimate(
                "signal",
                "signal.response_score_n_eff",
                "effective response-bank score count under exchangeable within-prompt covariance",
                round(ess.n_eff, 6),
                "effective response count",
                ess.raw_n,
                "response clustered by declared prompt",
                n_eff_uncertainty,
                subject_ref=subject_ref,
                provenance=ctx.provenance(),
                method=self.response_bank_method,
                result={
                    **common_result,
                    "warning": "response-bank n_eff is a score precision diagnostic, not a training or outcome sample count",
                },
                depends_on=("digest:source", "digest:samples"),
                limitations=limitations,
                assumptions=(
                    "task_id is the response bank's declared prompt identity",
                    "within-task response-score covariance is exchangeable for this precision diagnostic",
                ),
            )
        return [contrast, n_eff]

    def _entry(
        self,
        entry_id: str,
        measurand: str,
        metric: Metric,
        rows: Sequence[SignalObservation],
        ctx: RunContext,
        *,
        subject_ref: str,
        unit: str,
        extra: dict[str, Any],
        power: contracts.Power | None = None,
    ) -> contracts.Entry:
        uncertainty = (
            uncertainty_for_weighted_tasks(metric.task_values, metric.task_weights, seed=0)
            if metric.task_weights
            else uncertainty_for_tasks(
                metric.task_values, lambda values: sum(values) / len(values), seed=0
            )
        )
        ess = effective_sample_size(rows)
        interval = icc_interval(rows)
        uncertainty = _record_uncertainty(uncertainty, ess, interval)
        result = {
            "value_definition": measurand,
            "eligible": metric.eligible,
            "missing_components": metric.missing_components,
            "zero_total_magnitude_vectors": metric.zero_total_magnitude,
            "covariance_model": "exchangeable within declared task",
            "raw_rollout_n": ess.raw_n,
            "task_clusters": ess.clusters,
            "m_0": ess.m_0,
            "m_A": ess.m_a,
            "mean_group_size": ess.mean_group_size,
            "icc_raw": ess.icc_raw,
            "icc_truncated_at_zero": ess.icc_truncated_at_zero,
            "icc_identified": ess.identifiable,
            "icc_reason": ess.reason,
            "icc_interval_method": "task_cluster_pairs_bootstrap"
            if interval is not None
            else "not_available",
            "design_effect": ess.design_effect,
            "n_eff": ess.n_eff,
            **extra,
        }
        return contracts.estimate(
            "signal",
            entry_id,
            measurand,
            round(metric.value, 6),
            unit,
            metric.eligible,
            "task",
            uncertainty,
            subject_ref=subject_ref,
            provenance=ctx.provenance(),
            method=self.method,
            result=result,
            depends_on=("digest:training_semantics", "digest:samples"),
            rate_definition="contrast_fraction"
            if entry_id.endswith("contrast_fraction")
            else "other",
            power=power,
            assumptions=(
                "task is the independent sampling unit",
                "within-task covariance is exchangeable",
            ),
        )

    def contrast_entry(
        self,
        rows: Sequence[SignalObservation],
        ctx: RunContext,
        *,
        subject_ref: str | None = None,
        trl_frac_reward_zero_std: float | None = None,
        trl_metrics: Sequence[StepMetric] = (),
        planned_contrast: PlannedContrastDesign | None = None,
        design_standard_error: float | None = None,
    ) -> contracts.Entry:
        metric = contrast_fraction(rows)
        extra: dict[str, Any] = {"trl_reconciliation": "not_available_in_recorded_input"}
        if trl_frac_reward_zero_std is not None:
            zero_std_fraction = 1.0 - metric.value
            extra = {
                "trl_frac_reward_zero_std": float(trl_frac_reward_zero_std),
                "computed_frac_reward_zero_std": zero_std_fraction,
                "trl_reconciliation_difference": zero_std_fraction
                - float(trl_frac_reward_zero_std),
            }
        if trl_metrics:
            extra = _reconcile_step_metrics(rows, trl_metrics)
        extra["planned_contrast_design"] = _planned_design_result(planned_contrast)
        power = None
        if metric.value == 0.0 and planned_contrast is not None:
            power = null_result(
                clustered_standard_error=planned_contrast.standard_error,
                n=planned_contrast.n,
                design_effect=planned_contrast.design_effect,
                alpha=planned_contrast.alpha,
                power=planned_contrast.power,
            )
        elif metric.value == 0.0 and design_standard_error is not None:
            power = null_result(
                clustered_standard_error=design_standard_error,
                n=len(metric.task_values),
                design_effect=effective_sample_size(rows).design_effect or 1.0,
                alpha=0.05,
                power=0.80,
            )
            extra["null_design_standard_error"] = design_standard_error
        return self._entry(
            "signal.contrast_fraction",
            "fraction of declared reward groups with any within-group applied-reward contrast",
            metric,
            rows,
            ctx,
            subject_ref=subject_ref or contracts.digest({"source": "signal observations"}),
            unit="fraction",
            extra=extra,
            power=power,
        )

    def entries_for(
        self,
        rows: Sequence[SignalObservation],
        ctx: RunContext,
        *,
        subject_ref: str | None = None,
        trl_frac_reward_zero_std: float | None = None,
        trl_metrics: Sequence[StepMetric] = (),
        planned_contrast: PlannedContrastDesign | None = None,
        declared_component_universe: Sequence[str] | None = None,
        require_declared_component_universe: bool = False,
    ) -> list[contracts.Entry]:
        reference = subject_ref or contracts.digest({"source": "signal observations"})
        ess = effective_sample_size(rows)
        contrast = self.contrast_entry(
            rows,
            ctx,
            subject_ref=reference,
            trl_frac_reward_zero_std=trl_frac_reward_zero_std,
            trl_metrics=trl_metrics,
            planned_contrast=planned_contrast,
        )
        if ess.n_eff is None:
            return [
                contrast,
                ctx.absence(
                    "signal",
                    "signal.n_eff",
                    "effective rollout count under exchangeable within-task covariance",
                    ess.reason,
                    "record at least two applied rewards within enough declared tasks",
                    ("no effective-sample-size claim",),
                    subject_ref=reference,
                    depends_on=("digest:training_semantics",),
                    method=self.method,
                ),
            ]
        n_eff_metric = Metric(
            ess.n_eff, tuple(float(ess.n_eff) for _ in range(ess.clusters)), ess.raw_n
        )
        n_eff = self._entry(
            "signal.n_eff",
            "effective rollout count under declared exchangeable within-task covariance",
            n_eff_metric,
            rows,
            ctx,
            subject_ref=reference,
            unit="effective rollout count",
            extra={
                "outcome_sample_size": ess.clusters,
                "warning": "n_eff is a precision diagnostic and is not an outcome sample count",
            },
        )
        n_eff.n = ess.raw_n
        n_eff.sampling_unit = "rollout clustered by task"
        if require_declared_component_universe and declared_component_universe is None:
            component_absence = ctx.absence(
                "signal",
                "signal.style_share",
                "share of recorded absolute component magnitude attributable to the style component",
                "the canonical TraceRecord has no declared reward component partition",
                "record the source or caller-declared reward component partition",
                ("no style-share or component-disagreement claim",),
                subject_ref=reference,
                depends_on=("digest:training_semantics",),
                method=self.method,
            )
            disagreement_absence = ctx.absence(
                "signal",
                "signal.component_disagreement",
                "fraction of complete recorded component vectors containing opposed signed components",
                "the canonical TraceRecord has no declared reward component partition",
                "record the source or caller-declared reward component partition",
                ("no style-share or component-disagreement claim",),
                subject_ref=reference,
                depends_on=("digest:training_semantics",),
                method=self.method,
            )
            return [contrast, n_eff, component_absence, disagreement_absence]
        style = style_share(rows)
        component_extra = {
            "missing_components_preserved": True,
            "declared_component_universe": list(declared_component_universe or ()),
        }
        style_entry = (
            ctx.absence(
                "signal",
                "signal.style_share",
                "share of recorded absolute component magnitude attributable to the style component",
                "all complete component vectors have zero total magnitude",
                "record a nonzero component magnitude or state another predeclared denominator",
                ("no style-share claim",),
                subject_ref=reference,
                depends_on=("digest:training_semantics",),
                method=self.method,
            )
            if style.eligible == 0 and style.zero_total_magnitude
            else self._entry(
                "signal.style_share",
                "share of recorded absolute component magnitude attributable to the style component",
                style,
                rows,
                ctx,
                subject_ref=reference,
                unit="fraction",
                extra={"style_component": "style", **component_extra},
            )
        )
        return [
            contrast,
            n_eff,
            style_entry,
            self._entry(
                "signal.component_disagreement",
                "fraction of complete recorded component vectors containing opposed signed components",
                component_disagreement(rows),
                rows,
                ctx,
                subject_ref=reference,
                unit="fraction",
                extra=component_extra,
            ),
        ]


def _record_uncertainty(
    uncertainty: contracts.Uncertainty,
    ess: Any,
    icc_interval_value: list[float] | None,
) -> contracts.Uncertainty:
    """Rebuild the persisted uncertainty through the declared D-10 contract boundary.

    Signal computes ESS after bootstrap construction.  Assignment would bypass Pydantic's
    validators, so the raw calculation stays in the result payload while this declared record
    block is validated anew, applying every field's schema scale consistently.
    """
    payload = uncertainty.model_dump(mode="python", exclude_none=True)
    for field_name, value in (
        ("design_effect", ess.design_effect),
        ("icc", ess.icc),
        ("n_eff", ess.n_eff),
    ):
        if value is not None:
            payload[field_name] = value
    if icc_interval_value is not None:
        payload["icc_interval"] = icc_interval_value
    return contracts.Uncertainty.model_validate(payload)


def _reconcile_step_metrics(
    rows: Sequence[SignalObservation], metrics: Sequence[StepMetric]
) -> dict[str, Any]:
    """Reconcile only metric scopes that match a complete TraceRow step exactly."""
    logged = [
        metric
        for metric in metrics
        if metric.name == "frac_reward_zero_std"
        and metric.scope == "optimizer_step"
        and metric.value is not None
    ]
    matched: dict[str, dict[str, float]] = {}
    unavailable: list[int] = []
    for metric in logged:
        scoped_rows = tuple(row for row in rows if row.step == metric.step)
        computed = contrast_fraction(scoped_rows)
        if not scoped_rows or computed.eligible == 0:
            unavailable.append(metric.step)
            continue
        zero_std = 1.0 - computed.value
        matched[str(metric.step)] = {
            "trl_frac_reward_zero_std": float(metric.value),
            "computed_frac_reward_zero_std": zero_std,
            "trl_reconciliation_difference": zero_std - float(metric.value),
        }
    if not matched:
        return {
            "trl_reconciliation": "not_available_for_matching_optimizer_step",
            "trl_unmatched_optimizer_steps": unavailable,
        }
    if len(matched) == 1:
        step, values = next(iter(matched.items()))
        return {
            "trl_reconciliation": "matched_optimizer_step",
            "trl_metric_step": int(step),
            "trl_metric_scope": "optimizer_step",
            **values,
        }
    return {
        "trl_reconciliation": "matched_optimizer_steps",
        "trl_metric_scope": "optimizer_step",
        "trl_reconciliation_by_step": matched,
        "trl_unmatched_optimizer_steps": unavailable,
    }


def _planned_design_result(design: PlannedContrastDesign | None) -> dict[str, Any]:
    """Expose the immutable declaration without confusing it with realised trace support."""
    if design is None:
        return {
            "status": "not_declared",
            "reason": "the canonical TraceRecord carries no declared planned contrast design",
        }
    return {
        "status": "declared",
        "standard_error": design.standard_error,
        "alpha": design.alpha,
        "power": design.power,
        "n": design.n,
        "clusters": design.clusters,
        "covariance": None if design.covariance is None else dict(design.covariance),
        "design_effect": design.design_effect,
        "provenance": dict(design.provenance),
    }


PANEL = Panel(section="signal", instruments=(SignalInstrument(),))
