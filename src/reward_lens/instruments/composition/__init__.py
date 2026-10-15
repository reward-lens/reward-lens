"""The composition panel: scalar-loss evidence is never presented as update attribution."""

from __future__ import annotations

from typing import Any

from reward_lens import contracts
from reward_lens.instruments.base import Panel, RunContext
from reward_lens.product.trace import (
    CompositionTelemetry,
    ScalarLossStep,
    ScalarLossToken,
    TraceRecord,
)

from . import advantages as advantages
from .advantages import (
    Agreement,
    CompositionRecord,
    LossReconstruction,
    ReconstructedStep,
    price_differential,
    recompute_advantages,
    reconstruct_scalar_loss,
)
from .identity import (
    AppliedWeightIdentity,
    GradientControl,
    OptimizerControl,
    check_applied_weight_identity,
    equal_gradient_different_optimizer,
    equal_loss_different_gradient,
    share,
)

__all__ = [
    "Agreement",
    "AppliedWeightIdentity",
    "CompositionInstrument",
    "CompositionRecord",
    "CompositionTelemetry",
    "FINDING_CODE",
    "FINDING_ID",
    "GradientControl",
    "LossReconstruction",
    "OptimizerControl",
    "ReconstructedStep",
    "ScalarLossStep",
    "ScalarLossToken",
    "PANEL",
    "check_applied_weight_identity",
    "equal_gradient_different_optimizer",
    "equal_loss_different_gradient",
    "findings_for",
    "price_differential",
    "recompute_advantages",
    "reconstruct_scalar_loss",
    "share",
]

VERSION = "1.0.0"
FINDING_CODE = "RL0261"
FINDING_ID = "RGX-local-0261"


def _method() -> contracts.Method:
    return contracts.Method(
        id="trace.composition",
        version=VERSION,
        params_digest=contracts.digest(
            {"replay": "record.scores.replay_advantages", "identity": "sum(applied_weight)==-loss"}
        ),
        procedure=(
            "the declared trace groups are replayed under their recorded estimator; numerical "
            "token masks, log probabilities, ratios, clipping, denominator and KL or auxiliary "
            "terms reconstruct the supported scalar loss; gradients and updates remain separate"
        ),
        credited_to="reward_lens.instruments.composition (section 7.4)",
    )


def _subject_ref(subject: Any) -> str:
    return contracts.digest({"source": getattr(subject, "source", "") or ""})


class CompositionInstrument:
    id = "trace.composition"
    section: contracts.Section = "trace"
    method = _method()

    def run(self, subject: Any, corpus: Any, ctx: RunContext) -> list[contracts.Entry]:
        joined = getattr(corpus, "composition_record", None)
        if joined is None:
            trace = getattr(corpus, "trace", None)
            if isinstance(trace, TraceRecord):
                joined = CompositionRecord(trace=trace)
        subject_ref = _subject_ref(subject)
        if not isinstance(joined, CompositionRecord):
            return [
                self._absence(
                    ctx,
                    "trace",
                    "trace.composition.trace_join",
                    "whether recorded rewards and advantages reproduce the trainer's scalar-loss inputs",
                    "audit has no TraceRecord",
                    "run trace on a training record and route its canonical TraceRecord to this panel",
                    (
                        "no scalar-loss reconstruction, gradient, optimizer-transform, or update-attribution claim",
                    ),
                    subject_ref=subject_ref,
                    depends_on=("digest:training_semantics",),
                )
            ]
        agreement = recompute_advantages(joined)
        entries = [self._advantage_entry(agreement, joined, ctx, subject_ref)]
        entries.extend(self._scalar_entries(joined, ctx, subject_ref))
        entries.extend(self._unobserved_entries(ctx, subject_ref))
        return entries

    def _advantage_entry(
        self, agreement: Agreement, record: CompositionRecord, ctx: RunContext, subject_ref: str
    ) -> contracts.Entry:
        if agreement.agreement is None:
            return self._absence(
                ctx,
                "trace",
                "trace.composition.advantages",
                "whether recorded advantages reproduce from the declared groups and normalisation",
                "; ".join(agreement.missing) or "the trace contains no recorded advantages",
                "record declared group ids, estimator semantics, applied rewards and advantages",
                ("no advantage-replay agreement claim",),
                subject_ref=subject_ref,
                depends_on=("digest:training_semantics",),
            )
        return contracts.Entry(
            entry_id="trace.composition.advantages",
            section="trace",
            kind="check",
            measurand="whether recorded advantages reproduce from declared groups and normalisation",
            method=self.method,
            scope="evaluator_comparison",
            subject_ref=subject_ref,
            depends_on=["digest:training_semantics"],
            state="complete",
            provenance=ctx.provenance(),
            limitations=[
                "agreement is an advantage replay only; it is not gradient or update evidence"
            ],
            n=len([value for value in agreement.recorded if value is not None]),
            result={
                "agreement": agreement.agreement,
                "reconstructed": list(agreement.reconstructed),
                "recorded": list(agreement.recorded),
                "missing": list(agreement.missing),
            },
            check=contracts.Check(
                predicate="recorded advantages equal replayed advantages without adjustment",
                passed=agreement.agreement == 1.0,
                scope_tested=f"{len(agreement.recorded)} trace rows",
            ),
        )

    def _scalar_entries(
        self, record: CompositionRecord, ctx: RunContext, subject_ref: str
    ) -> list[contracts.Entry]:
        reconstruction = reconstruct_scalar_loss(record)
        identities = check_applied_weight_identity(record)
        if not reconstruction.available:
            missing = (
                "; ".join(reconstruction.missing) or "no scalar-loss reconstruction is available"
            )
            scalar = self._absence(
                ctx,
                "trace",
                "trace.composition.scalar_loss",
                "the trainer's scalar loss decomposition",
                missing,
                "record numerical token masks, log probabilities, ratios, clipping, denominator, KL or auxiliary terms, and the supported trainer profile",
                (
                    "no scalar-loss reconstruction beyond the separately checked applied-weight identity",
                ),
                subject_ref=subject_ref,
                depends_on=("digest:training_semantics",),
            )
            return [scalar, self._identity_entry(identities, ctx, subject_ref)]
        return [
            self._scalar_entry(reconstruction, ctx, subject_ref),
            self._identity_entry(identities, ctx, subject_ref),
        ]

    def _scalar_entry(
        self, reconstruction: LossReconstruction, ctx: RunContext, subject_ref: str
    ) -> contracts.Entry:
        steps = [
            {
                "step": item.step,
                "reconstructed_loss": item.reconstructed_loss,
                "observed_loss": item.observed_loss,
                "agreement": item.agreement,
                "aggregation_denominator": item.aggregation_denominator,
                "share_denominator": item.share_denominator,
                "row_loss_contributions": dict(item.row_loss_contributions),
                "row_signed_negative_loss_contributions": dict(
                    item.row_signed_negative_loss_contributions
                ),
                "row_shares": dict(item.row_shares),
                "share_sign_convention": "row_signed_negative_loss_contribution = -row_loss_contribution",
                "reason": item.reason,
            }
            for item in reconstruction.steps
        ]
        comparable = [item for item in reconstruction.steps if item.agreement is not None]
        passed = bool(comparable) and all(item.agreement is True for item in comparable)
        return contracts.Entry(
            entry_id="trace.composition.scalar_loss",
            section="trace",
            kind="check",
            measurand="whether numerical token telemetry reconstructs the recorded scalar loss",
            method=self.method,
            scope="reconstructed_training_pressure",
            subject_ref=subject_ref,
            depends_on=["digest:training_semantics"],
            state="complete",
            provenance=ctx.provenance(),
            limitations=[
                "signed loss contributions and shares are scalar-loss quantities, not gradients or updates",
                "shares are absent when their reconstructed-loss denominator is zero or unstable",
            ],
            n=len(reconstruction.steps),
            result={"steps": steps},
            check=contracts.Check(
                predicate="reconstructed token-mean clipped loss equals the recorded scalar loss",
                passed=passed,
                scope_tested=f"{len(reconstruction.steps)} recorded optimizer steps",
            ),
        )

    def _identity_entry(
        self, identities: tuple[AppliedWeightIdentity, ...], ctx: RunContext, subject_ref: str
    ) -> contracts.Entry:
        available = [row for row in identities if row.available]
        if not available:
            return self._absence(
                ctx,
                "trace",
                "trace.composition.applied_weight_identity",
                "whether signed applied weights sum to negative scalar loss",
                "record one signed applied weight per row and scalar loss per step",
                "record the signed per-row applied-weight contributions and scalar loss for each optimizer step",
                ("no applied-weight identity claim",),
                subject_ref=subject_ref,
                depends_on=("digest:training_semantics",),
            )
        passed = all(row.passed is True for row in available) and len(available) == len(identities)
        first = available[0]
        rows = [row.__dict__ for row in identities]
        findings = (
            []
            if passed
            else [
                {
                    "code": FINDING_CODE,
                    "kind": "fail",
                    "level": "error",
                    "message": "the signed applied-weight sum differs from negative recorded loss",
                }
            ]
        )
        return contracts.Entry(
            entry_id="trace.composition.applied_weight_identity",
            section="trace",
            kind="check",
            measurand="whether signed applied weights sum to negative scalar loss for every recorded step",
            method=self.method,
            scope="evaluator_comparison",
            subject_ref=subject_ref,
            depends_on=["digest:training_semantics"],
            state="complete",
            provenance=ctx.provenance(),
            limitations=[
                "this scalar identity does not establish a gradient, optimizer transform, or parameter update"
            ],
            n=len(identities),
            result={
                "identities": rows,
                "sum_applied_weight": first.sum_applied_weight,
                "negative_loss": first.negative_loss,
                "findings": findings,
            },
            check=contracts.Check(
                predicate="sum(applied_weight) == -loss for each observed step",
                passed=passed,
                scope_tested=f"{len(identities)} recorded optimizer steps",
            ),
        )

    def _unobserved_entries(self, ctx: RunContext, subject_ref: str) -> list[contracts.Entry]:
        return [
            self._absence(
                ctx,
                "trace",
                "trace.composition.gradient",
                "gradient contribution",
                "no gradients were recorded",
                "record gradients and use a controlled replay",
                ("no gradient attribution claim",),
                subject_ref=subject_ref,
            ),
            self._absence(
                ctx,
                "trace",
                "trace.composition.optimizer_transform",
                "optimizer transform",
                "no optimizer runtime state was recorded",
                "record optimizer state and transform telemetry",
                ("no optimizer-transform claim",),
                subject_ref=subject_ref,
            ),
            self._absence(
                ctx,
                "trace",
                "trace.composition.parameter_update",
                "measured parameter update",
                "no controlled replay or parameter delta was recorded",
                "record a controlled replay or measured parameter delta",
                ("no parameter-update attribution claim",),
                subject_ref=subject_ref,
            ),
        ]

    def _absence(
        self,
        ctx: RunContext,
        section: str,
        entry_id: str,
        measurand: str,
        missing_access: str,
        remedy: str,
        affected_claims: tuple[str, ...],
        *,
        subject_ref: str,
        depends_on: tuple[str, ...] = (),
    ) -> contracts.Entry:
        """Keep a declared method on an absence so audit reuse sees this instrument's contract."""
        return contracts.absence(
            section,
            entry_id,
            measurand,
            missing_access,
            remedy,
            affected_claims,
            subject_ref=subject_ref,
            provenance=ctx.provenance(),
            method=self.method,
            depends_on=depends_on,
        )


def findings_for(entry: contracts.Entry) -> tuple[contracts.Finding, ...]:
    rows = list((entry.result or {}).get("findings") or ())
    return tuple(
        contracts.Finding(
            id=FINDING_ID,
            rule=entry.entry_id,
            level=str(row["level"]),
            kind=str(row["kind"]),
            severity_rationale="the declared scalar-loss decomposition does not match its signed applied-weight contributions",
            scope=entry.scope,
            entries=[entry.entry_id],
            partial_fingerprint=contracts.digest(
                {"entry": entry.entry_id, "code": row["code"], "index": index}
            ).removeprefix("sha256:")[:16],
            message=str(row["message"]),
            code=str(row["code"]),
            arm="white_box",
        )
        for index, row in enumerate(rows)
        if row.get("code") == FINDING_CODE
    )


PANEL = Panel(section="trace", instruments=(CompositionInstrument(),))
