"""Rule-table planning for `reward-lens improve`.

The table states what each rung can distinguish.  The planner deliberately does
not calculate an expected-information-gain score because no likelihood model is
available: it ranks declared costs and preserves the evidence that made a choice
possible before execution starts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import yaml

from reward_lens.commons.corpus import Corpus
from reward_lens.commons.corpus.protocol import (
    PlannerCase,
    PlanningAnnotation,
    consult_corpus,
    load_planner_cases,
)
from reward_lens.commons.corpus.store import CorpusStore
from reward_lens.contracts import canonical_bytes, digest
from reward_lens.graders.base import EvidenceEnvelope

__all__ = [
    "EXPLANATION_KINDS",
    "MANDATORY_ORDERINGS",
    "DiscriminationCell",
    "EvaluatorContract",
    "ExecutionEvidence",
    "Explanation",
    "HmacEvidenceAuthenticator",
    "PlanAlternative",
    "PlanChoice",
    "PlannerRefusal",
    "PlannerRefusalInfo",
    "PlannerLedger",
    "PlannerResult",
    "RungCost",
    "TrustedEvaluator",
    "load_discrimination_table",
    "plan",
    "record_result",
]


EXPLANATION_KINDS: tuple[str, ...] = (
    "score_treatment",
    "selection_response",
    "advantage_pressure",
    "update_mechanism",
    "horizon_behavior",
    "transfer_generalisation",
)

# The spelling is public so callers cannot replace either diagnostic with prose.
MANDATORY_ORDERINGS: tuple[str, ...] = (
    "replay_nondeterminism_environment_identity",
    "neural_intervention",
    "masks_and_loss_aggregation",
    "reward_model_feature",
)

_ORDERING_RULES: tuple[tuple[str, str], ...] = (
    ("replay_nondeterminism_environment_identity", "neural_intervention"),
    ("masks_and_loss_aggregation", "reward_model_feature"),
)


def _pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("a discrimination pair needs two distinct explanation kinds")
    return tuple(sorted((left, right)))


@dataclass(frozen=True)
class Explanation:
    """One live explanation to separate by an experiment, never a probability."""

    kind: str
    statement: str

    def __post_init__(self) -> None:
        if self.kind not in EXPLANATION_KINDS:
            raise ValueError(
                f"unknown explanation kind {self.kind!r}; the shipped table has no row"
            )
        if not isinstance(self.statement, str) or not self.statement:
            raise TypeError("explanation statement must be a non-empty string")


@dataclass(frozen=True)
class RungCost:
    """One declared cost row.  `rank` resolves cost tradeoffs visibly."""

    rung: int
    rank: int
    money_usd: Decimal
    samples: int
    minutes: int
    access: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rung not in range(1, 7) or self.rank <= 0:
            raise ValueError("rung and rank must be positive declared table values")
        if type(self.money_usd) is not Decimal or self.money_usd < 0:
            raise TypeError("declared money cost must be a non-negative Decimal")
        if type(self.samples) is not int or self.samples < 0:
            raise TypeError("declared samples must be a non-negative integer")
        if type(self.minutes) is not int or self.minutes < 0:
            raise TypeError("declared minutes must be a non-negative integer")
        if not self.access or any(not isinstance(item, str) or not item for item in self.access):
            raise ValueError("each rung cost names its required access")

    @property
    def ordering_key(self) -> tuple[int, Decimal, int, int, int]:
        return (self.rank, self.money_usd, self.samples, self.minutes, self.rung)


@dataclass(frozen=True)
class DiscriminationCell:
    """One pair/rung table cell, including whether the rung can separate it."""

    pair: tuple[str, str]
    rung: int
    separates: bool
    measurement: str
    required_access: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pair != _pair(*self.pair) or any(
            kind not in EXPLANATION_KINDS for kind in self.pair
        ):
            raise ValueError("discrimination cell pair must be a sorted pair of known kinds")
        if self.rung not in range(1, 7) or type(self.separates) is not bool:
            raise TypeError("discrimination cell needs a valid rung and boolean separation")
        if not isinstance(self.measurement, str) or not self.measurement:
            raise TypeError("discrimination cell measurement must be non-empty")


@dataclass(frozen=True)
class PlannerRefusalInfo:
    code: str
    detail: str


class PlannerRefusal(ValueError):
    """A required planner precondition was not registered before execution."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class PlanAlternative:
    rung: int
    cost: RungCost
    separates_pairs: tuple[tuple[str, str], ...]
    unresolved_pairs: tuple[tuple[str, str], ...]
    feasible: bool
    missing_access: tuple[str, ...]
    over_budget: bool
    prior_support_case_ids: tuple[str, ...] = ()

    @property
    def separates(self) -> bool:
        return bool(self.separates_pairs)


@dataclass(frozen=True)
class PlanChoice:
    """An execution-ready pre-registration, with no observed result attached."""

    kind: Literal["complete", "partial", "unresolved"]
    rung: int | None
    cost: RungCost | None
    separated_pairs: tuple[tuple[str, str], ...]
    unresolved_pairs: tuple[tuple[str, str], ...]
    alternatives: tuple[PlanAlternative, ...]
    expected_outcomes: tuple[str, ...]
    selection_rule: str = "declared_cost_coverage_then_prior_case_support_v1"
    compared_prior_support: tuple[tuple[int, tuple[str, ...]], ...] = ()
    missing_access: tuple[str, ...] = ()
    missing_measurement: str | None = None
    prior_case: PlannerCase | None = None
    evaluator_contract: "EvaluatorContract | None" = None
    refusal: PlannerRefusalInfo | None = None
    registration_id: str | None = None
    choice_digest: str | None = None
    corpus_provenance: dict[str, object] | None = None
    cost_table_provenance: dict[str, object] | None = None

    @property
    def signature(self) -> str | None:
        if self.rung is None:
            return None
        names = "|".join("/".join(pair) for pair in self.separated_pairs)
        return f"r{self.rung}:{names}"


@dataclass(frozen=True)
class HmacEvidenceAuthenticator:
    """Signature primitive held by a trusted evaluator configured before planning."""

    key: bytes
    issuer: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, bytes) or len(self.key) < 16:
            raise ValueError("evaluator authentication key must be at least sixteen bytes")
        if not isinstance(self.issuer, str) or not self.issuer:
            raise TypeError("evaluator authentication issuer must be a non-empty string")

    def sign(self, payload: dict[str, Any]) -> str:
        return hmac.new(self.key, canonical_bytes(payload), "sha256").hexdigest()

    def verifies(self, evidence: ExecutionEvidence) -> bool:
        return evidence.evaluator_id == self.issuer and hmac.compare_digest(
            evidence.signature, self.sign(evidence.unsigned_payload())
        )


@dataclass(frozen=True)
class EvaluatorContract:
    """Trusted evaluator identity and implementation pinned before a run begins."""

    evaluator_id: str
    implementation_digest: str
    rule_id: str
    key_id: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.evaluator_id, self.rule_id, self.key_id)
        ):
            raise TypeError("evaluator contract identity, rule, and key id are required")
        if (
            not self.implementation_digest.startswith("sha256:")
            or len(self.implementation_digest) != 71
        ):
            raise ValueError("evaluator contract implementation needs a sha256 digest")

    def payload(self) -> dict[str, str]:
        return {
            "evaluator_id": self.evaluator_id,
            "implementation_digest": self.implementation_digest,
            "rule_id": self.rule_id,
            "key_id": self.key_id,
        }

    @property
    def contract_digest(self) -> str:
        return digest(self.payload())


@dataclass(frozen=True)
class TrustedEvaluator:
    """A principal-configured evaluator, available before plans are registered."""

    contract: EvaluatorContract
    authenticator: HmacEvidenceAuthenticator

    def __post_init__(self) -> None:
        if self.authenticator.issuer != self.contract.evaluator_id:
            raise ValueError(
                "trusted evaluator authenticator issuer must equal its contract evaluator id"
            )


@dataclass(frozen=True)
class ExecutionEvidence:
    """Signed pointer to immutable paired-grader-envelope bytes, never an asserted outcome."""

    choice_digest: str
    evaluator_id: str
    evaluator_contract_digest: str
    artifact_path: str
    artifact_digest: str
    signature: str

    def __post_init__(self) -> None:
        if any(
            not value.startswith("sha256:") or len(value) != 71
            for value in (self.choice_digest, self.evaluator_contract_digest, self.artifact_digest)
        ):
            raise ValueError(
                "evidence must bind sha256 plan, evaluator-contract, and artifact digests"
            )
        if not all(
            isinstance(value, str) and value
            for value in (self.evaluator_id, self.artifact_path, self.signature)
        ):
            raise TypeError(
                "evidence evaluator identity, artifact path, and signature are required"
            )
        if not Path(self.artifact_path).is_absolute():
            raise ValueError("evidence artifact path must be absolute")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "choice_digest": self.choice_digest,
            "evaluator_id": self.evaluator_id,
            "evaluator_contract_digest": self.evaluator_contract_digest,
            "artifact_path": self.artifact_path,
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def issue(
        cls,
        *,
        choice_digest: str | None,
        artifact_path: Path | str,
        evaluator: TrustedEvaluator,
    ) -> "ExecutionEvidence":
        if choice_digest is None:
            raise PlannerRefusal(
                "PLANNER_CHOICE_UNREGISTERED", "evaluator evidence needs a registered choice digest"
            )
        if not isinstance(evaluator, TrustedEvaluator):
            raise TypeError("only a configured TrustedEvaluator may issue evaluator evidence")
        source = Path(artifact_path).resolve(strict=True)
        artifact_digest = _artifact_digest(source.read_bytes())
        unsigned = {
            "choice_digest": choice_digest,
            "evaluator_id": evaluator.contract.evaluator_id,
            "evaluator_contract_digest": evaluator.contract.contract_digest,
            "artifact_path": str(source),
            "artifact_digest": artifact_digest,
        }
        return cls(
            choice_digest=choice_digest,
            evaluator_id=evaluator.contract.evaluator_id,
            evaluator_contract_digest=evaluator.contract.contract_digest,
            artifact_path=str(source),
            artifact_digest=artifact_digest,
            signature=evaluator.authenticator.sign(unsigned),
        )


@dataclass(frozen=True)
class PlannerResult:
    registration_id: str
    choice_digest: str
    evidence: ExecutionEvidence
    ruled_out: tuple[str, ...]
    informative: bool
    analogy_survived: bool | None


@dataclass(frozen=True)
class _LedgerEntry:
    kind: Literal["choice", "result"]
    entry_digest: str
    previous_entry_digest: str | None
    payload: dict[str, Any]


class PlannerLedger:
    """Append-only persisted choice/result history with a validated digest chain."""

    def __init__(self, path: Path | str, *, evaluators: Iterable[TrustedEvaluator]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        configured = tuple(evaluators)
        if not configured or any(not isinstance(item, TrustedEvaluator) for item in configured):
            raise TypeError("planner ledger needs one or more trusted evaluators before planning")
        self._evaluators = {item.contract.evaluator_id: item for item in configured}
        if len(self._evaluators) != len(configured):
            raise ValueError("planner ledger trusted evaluator ids must be unique")
        self._entries = self._load()

    @classmethod
    def open(cls, path: Path | str, *, evaluators: Iterable[TrustedEvaluator]) -> "PlannerLedger":
        return cls(path, evaluators=evaluators)

    @property
    def entries(self) -> tuple[_LedgerEntry, ...]:
        return tuple(self._entries)

    def _load(self) -> list[_LedgerEntry]:
        if not self.path.exists():
            return []
        rows: list[_LedgerEntry] = []
        previous: str | None = None
        choices: dict[str, dict[str, Any]] = {}
        results: set[str] = set()
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = yaml.safe_load(line)
            except yaml.YAMLError as error:
                raise PlannerRefusal(
                    "PLANNER_LEDGER_INVALID", f"line {number} is not a ledger record"
                ) from error
            if not isinstance(row, dict):
                raise PlannerRefusal("PLANNER_LEDGER_INVALID", f"line {number} is not a mapping")
            entry_digest = row.pop("entry_digest", None)
            if not isinstance(entry_digest, str) or digest(row) != entry_digest:
                raise PlannerRefusal(
                    "PLANNER_LEDGER_TAMPERED", f"line {number} digest does not match"
                )
            if row.get("previous_entry_digest") != previous:
                raise PlannerRefusal(
                    "PLANNER_LEDGER_TAMPERED", f"line {number} breaks the append chain"
                )
            kind = row.get("kind")
            payload = row.get("payload")
            if kind not in {"choice", "result"} or not isinstance(payload, dict):
                raise PlannerRefusal(
                    "PLANNER_LEDGER_INVALID", f"line {number} has no valid kind/payload"
                )
            if kind == "choice":
                registration_id = payload.get("registration_id")
                choice = payload.get("choice")
                if not isinstance(registration_id, str) or not isinstance(choice, dict):
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_INVALID", f"line {number} has no registered choice"
                    )
                _validate_cost_table_provenance(choice)
                if payload.get("choice_digest") != digest(choice) or registration_id in choices:
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_TAMPERED", f"line {number} changes a registered choice"
                    )
                choices[registration_id] = payload
            else:
                registration_id = payload.get("registration_id")
                if registration_id not in choices or registration_id in results:
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_TAMPERED", f"line {number} has an invalid result binding"
                    )
                if payload.get("choice_digest") != choices[registration_id]["choice_digest"]:
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_TAMPERED", f"line {number} changes a result choice digest"
                    )
                contract = _contract_from_payload(
                    choices[registration_id]["choice"].get("evaluator_contract")
                )
                evaluator = self._trusted_evaluator(contract)
                evidence_payload = payload.get("evidence")
                if not isinstance(evidence_payload, dict):
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_INVALID", f"line {number} has no typed executor evidence"
                    )
                try:
                    evidence = ExecutionEvidence(
                        choice_digest=evidence_payload["choice_digest"],
                        evaluator_id=evidence_payload["evaluator_id"],
                        evaluator_contract_digest=evidence_payload["evaluator_contract_digest"],
                        artifact_path=evidence_payload["artifact_path"],
                        artifact_digest=evidence_payload["artifact_digest"],
                        signature=evidence_payload["signature"],
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_INVALID", f"line {number} has malformed executor evidence"
                    ) from error
                if not self._evidence_matches_contract(evidence, contract, evaluator):
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_TAMPERED",
                        f"line {number} executor evidence signature does not verify",
                    )
                if tuple(payload.get("ruled_out", ())) != _derive_eliminations_from_evidence(
                    choices[registration_id]["choice"], evidence
                ) or payload.get("informative") is not bool(payload.get("ruled_out", ())):
                    raise PlannerRefusal(
                        "PLANNER_LEDGER_TAMPERED", f"line {number} changes supported eliminations"
                    )
                results.add(registration_id)
            rows.append(
                _LedgerEntry(
                    kind=kind,
                    entry_digest=entry_digest,
                    previous_entry_digest=previous,
                    payload=payload,
                )
            )
            previous = entry_digest
        return rows

    def _trusted_evaluator(self, contract: "EvaluatorContract") -> TrustedEvaluator:
        evaluator = self._evaluators.get(contract.evaluator_id)
        if evaluator is None or evaluator.contract != contract:
            raise PlannerRefusal(
                "PLANNER_EVALUATOR_UNTRUSTED",
                "the pre-execution choice does not name a configured trusted evaluator contract",
            )
        return evaluator

    def evaluator_contract(self, evaluator_id: str | None) -> "EvaluatorContract":
        if evaluator_id is None:
            if len(self._evaluators) != 1:
                raise PlannerRefusal(
                    "PLANNER_EVALUATOR_REQUIRED",
                    "choose one preconfigured evaluator before planning",
                )
            return next(iter(self._evaluators.values())).contract
        evaluator = self._evaluators.get(evaluator_id)
        if evaluator is None:
            raise PlannerRefusal(
                "PLANNER_EVALUATOR_UNTRUSTED", f"{evaluator_id!r} was not trusted before planning"
            )
        return evaluator.contract

    def _evidence_matches_contract(
        self,
        evidence: ExecutionEvidence,
        contract: "EvaluatorContract",
        evaluator: TrustedEvaluator,
    ) -> bool:
        return (
            evidence.evaluator_id == contract.evaluator_id
            and evidence.evaluator_contract_digest == contract.contract_digest
            and evaluator.authenticator.verifies(evidence)
        )

    def _append(self, kind: Literal["choice", "result"], payload: dict[str, Any]) -> _LedgerEntry:
        body = {
            "kind": kind,
            "previous_entry_digest": self._entries[-1].entry_digest if self._entries else None,
            "payload": payload,
        }
        entry = {**body, "entry_digest": digest(body)}
        encoded = (
            json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        stored = _LedgerEntry(
            kind=kind,
            entry_digest=entry["entry_digest"],
            previous_entry_digest=body["previous_entry_digest"],
            payload=payload,
        )
        self._entries.append(stored)
        return stored

    def register_choice(self, choice: PlanChoice, *, retry_limit: int) -> tuple[str, str]:
        if choice.registration_id is not None or choice.choice_digest is not None:
            raise PlannerRefusal(
                "PLANNER_CHOICE_ALREADY_REGISTERED",
                "only a fresh pre-execution choice may be registered",
            )
        if type(retry_limit) is not int or retry_limit < 0:
            raise ValueError("planner retry limit must be a non-negative integer")
        serialised = _choice_payload(choice)
        choice_digest = digest(serialised)
        registration_id = "planner-choice:" + digest(
            {
                "previous_entry_digest": self._entries[-1].entry_digest if self._entries else None,
                "choice_digest": choice_digest,
            }
        )
        self._append(
            "choice",
            {
                "registration_id": registration_id,
                "choice_digest": choice_digest,
                "retry_limit": retry_limit,
                "choice": serialised,
            },
        )
        return registration_id, choice_digest

    def registered_choice(self, choice: PlanChoice) -> dict[str, Any]:
        if not isinstance(choice.registration_id, str) or not isinstance(choice.choice_digest, str):
            raise PlannerRefusal(
                "PLANNER_CHOICE_UNREGISTERED",
                "result recording needs a persisted pre-execution choice",
            )
        for entry in self._entries:
            if (
                entry.kind == "choice"
                and entry.payload["registration_id"] == choice.registration_id
            ):
                stored = entry.payload
                if stored["choice_digest"] != choice.choice_digest or stored[
                    "choice"
                ] != _choice_payload(choice):
                    raise PlannerRefusal(
                        "PLANNER_CHOICE_DIGEST_MISMATCH",
                        "choice differs from its registered pre-execution digest",
                    )
                return stored
        raise PlannerRefusal(
            "PLANNER_CHOICE_UNREGISTERED", "choice registration is absent from this planner ledger"
        )

    def result_exists(self, registration_id: str) -> bool:
        return any(
            entry.kind == "result" and entry.payload["registration_id"] == registration_id
            for entry in self._entries
        )

    def inconclusive_attempts(self, signature: str | None) -> int:
        if signature is None:
            return 0
        choices = {
            entry.payload["registration_id"]: entry.payload["choice"]
            for entry in self._entries
            if entry.kind == "choice"
        }
        return sum(
            1
            for entry in self._entries
            if entry.kind == "result"
            and choices[entry.payload["registration_id"]]["signature"] == signature
            and not entry.payload["informative"]
        )

    def require_consistent_retry_limit(self, signature: str | None, retry_limit: int) -> None:
        if signature is None:
            return
        prior_limits = {
            entry.payload["retry_limit"]
            for entry in self._entries
            if entry.kind == "choice" and entry.payload["choice"]["signature"] == signature
        }
        if prior_limits and prior_limits != {retry_limit}:
            raise PlannerRefusal(
                "PLANNER_RETRY_HISTORY_INCONSISTENT",
                f"{signature} was registered with retry limit {sorted(prior_limits)}; requested {retry_limit}",
            )

    def append_result(self, result: PlannerResult) -> PlannerResult:
        if self.result_exists(result.registration_id):
            raise PlannerRefusal(
                "PLANNER_RESULT_ALREADY_RECORDED", "a registered choice accepts one result record"
            )
        self._append(
            "result",
            {
                "registration_id": result.registration_id,
                "choice_digest": result.choice_digest,
                "evidence": {
                    **result.evidence.unsigned_payload(),
                    "signature": result.evidence.signature,
                },
                "ruled_out": list(result.ruled_out),
                "informative": result.informative,
                "analogy_survived": result.analogy_survived,
            },
        )
        return result


def _table_path() -> Path:
    return Path(__file__).with_name("discrimination.yaml")


def _cost_payload(cost: RungCost | None) -> dict[str, Any] | None:
    if cost is None:
        return None
    return {
        "rung": cost.rung,
        "rank": cost.rank,
        "money_usd": str(cost.money_usd),
        "samples": cost.samples,
        "minutes": cost.minutes,
        "access": list(cost.access),
    }


def _contract_from_payload(payload: Any) -> "EvaluatorContract":
    if not isinstance(payload, dict):
        raise PlannerRefusal(
            "PLANNER_LEDGER_INVALID", "registered executable choice has no evaluator contract"
        )
    try:
        return EvaluatorContract(
            evaluator_id=payload["evaluator_id"],
            implementation_digest=payload["implementation_digest"],
            rule_id=payload["rule_id"],
            key_id=payload["key_id"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PlannerRefusal(
            "PLANNER_LEDGER_INVALID", "registered evaluator contract is malformed"
        ) from error


def _choice_payload(choice: PlanChoice) -> dict[str, Any]:
    """The complete immutable pre-execution record, excluding registration metadata."""

    return {
        **(
            {"corpus_provenance": choice.corpus_provenance}
            if choice.corpus_provenance is not None
            else {}
        ),
        **(
            {"cost_table_provenance": choice.cost_table_provenance}
            if choice.cost_table_provenance is not None
            else {}
        ),
        "kind": choice.kind,
        "rung": choice.rung,
        "cost": _cost_payload(choice.cost),
        "separated_pairs": [list(pair) for pair in choice.separated_pairs],
        "unresolved_pairs": [list(pair) for pair in choice.unresolved_pairs],
        "alternatives": [
            {
                "rung": alternative.rung,
                "cost": _cost_payload(alternative.cost),
                "separates_pairs": [list(pair) for pair in alternative.separates_pairs],
                "unresolved_pairs": [list(pair) for pair in alternative.unresolved_pairs],
                "feasible": alternative.feasible,
                "missing_access": list(alternative.missing_access),
                "over_budget": alternative.over_budget,
                "prior_support_case_ids": list(alternative.prior_support_case_ids),
            }
            for alternative in choice.alternatives
        ],
        "expected_outcomes": list(choice.expected_outcomes),
        "selection_rule": choice.selection_rule,
        "compared_prior_support": [
            {"rung": rung, "case_ids": list(case_ids)}
            for rung, case_ids in choice.compared_prior_support
        ],
        "missing_access": list(choice.missing_access),
        "missing_measurement": choice.missing_measurement,
        "prior_case": (
            None
            if choice.prior_case is None
            else {
                "id": choice.prior_case.id,
                "explanation_kinds": list(choice.prior_case.explanation_kinds),
                "rung": choice.prior_case.rung,
                "analogy": choice.prior_case.analogy,
                "case_digest": choice.prior_case.case_digest,
                "source_digest": choice.prior_case.source_digest,
                "annotation_digest": choice.prior_case.annotation_digest,
            }
        ),
        "evaluator_contract": (
            None if choice.evaluator_contract is None else choice.evaluator_contract.payload()
        ),
        "refusal": (
            None
            if choice.refusal is None
            else {"code": choice.refusal.code, "detail": choice.refusal.detail}
        ),
        "signature": choice.signature,
    }


def _derive_eliminations_from_evidence(
    choice_payload: dict[str, Any], evidence: ExecutionEvidence
) -> tuple[str, ...]:
    """Read and apply the trusted rule to the raw artifact, never caller assertions."""

    contract = _contract_from_payload(choice_payload.get("evaluator_contract"))
    if contract.rule_id != "paired-grader-envelope-v1":
        raise PlannerRefusal(
            "PLANNER_EVALUATOR_RULE_UNSUPPORTED", f"unsupported evaluator rule {contract.rule_id!r}"
        )
    try:
        raw = Path(evidence.artifact_path).read_bytes()
    except OSError as error:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_MISSING", "the signed measurement artifact is absent"
        ) from error
    if _artifact_digest(raw) != evidence.artifact_digest:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_CHANGED", "measurement artifact bytes differ from signed digest"
        )
    try:
        artifact = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", "measurement artifact is not UTF-8 JSON"
        ) from error
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != "reward-lens.paired-grader-envelope.v1"
    ):
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", "measurement artifact has no supported schema"
        )
    if (
        artifact.get("choice_digest") != evidence.choice_digest
        or artifact.get("evaluator_contract_digest") != contract.contract_digest
    ):
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_BINDING_MISMATCH",
            "measurement artifact is not bound to this registered execution",
        )
    try:
        pair = _pair(*tuple(artifact["pair"]))
    except (KeyError, TypeError, ValueError) as error:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", "measurement artifact has no valid explanation pair"
        ) from error
    if pair not in {tuple(item) for item in choice_payload["separated_pairs"]}:
        raise PlannerRefusal(
            "PLANNER_ELIMINATION_UNSUPPORTED",
            "artifact measured a pair the registered experiment did not separate",
        )
    left = _grader_envelope(artifact.get("left"), side="left")
    right = _grader_envelope(artifact.get("right"), side="right")
    if left[1] == right[1]:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", "paired scores need distinct executed result envelopes"
        )
    if left[0] > right[0]:
        return (pair[1],)
    if right[0] > left[0]:
        return (pair[0],)
    return ()


def _artifact_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _grader_envelope(value: Any, *, side: str) -> tuple[float, str]:
    """Validate an actual adapter result; handwritten score/witness fields are not evidence."""

    if not isinstance(value, dict) or set(value) != {"envelope", "envelope_digest"}:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", f"{side} needs exactly one grader envelope and digest"
        )
    payload, claimed_digest = value["envelope"], value["envelope_digest"]
    if not isinstance(payload, dict) or not isinstance(claimed_digest, str):
        raise PlannerRefusal("PLANNER_ARTIFACT_INVALID", f"{side} grader envelope is malformed")
    try:
        envelope = EvidenceEnvelope.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", f"{side} is not an EvidenceEnvelope producer result"
        ) from error
    if envelope.digest() != claimed_digest:
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_CHANGED", f"{side} serialized envelope does not match its digest"
        )
    if (
        envelope.verdict != "scored"
        or type(envelope.score) not in (int, float)
        or not math.isfinite(float(envelope.score))
    ):
        raise PlannerRefusal(
            "PLANNER_ARTIFACT_INVALID", f"{side} producer did not return a finite scored envelope"
        )
    return float(envelope.score), envelope.digest()


def load_discrimination_table(
    path: Path | str | None = None,
) -> tuple[tuple[RungCost, ...], tuple[DiscriminationCell, ...]]:
    """Read and validate the shipped rule table before planning from it."""

    source = Path(path) if path is not None else _table_path()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("discrimination table must be a mapping")
    declared_kinds = tuple(loaded.get("explanation_kinds", {}))
    if declared_kinds != EXPLANATION_KINDS:
        raise ValueError("discrimination table explanation kinds must match the public kind order")
    costs = tuple(
        RungCost(
            rung=row["rung"],
            rank=row["rank"],
            money_usd=Decimal(str(row["money_usd"])),
            samples=row["samples"],
            minutes=row["minutes"],
            access=tuple(row["access"]),
        )
        for row in loaded.get("costs", ())
    )
    if tuple(cost.rung for cost in costs) != (1, 2, 3, 4, 5, 6):
        raise ValueError("discrimination table needs one declared cost for each rung")
    if len({cost.rank for cost in costs}) != len(costs):
        raise ValueError("discrimination table ranks must be unique")
    cost_for = {cost.rung: cost for cost in costs}
    measurements = {
        1: "score frozen samples under both versions",
        2: "apply declared selection stress and independent outcome check",
        3: "recompute advantages under recorded grouping and normalisation",
        4: "apply each candidate once to one restored trainer batch",
        5: "run a clamped short fork with useful-task controls",
        6: "compare held-out families, exploit, and checkpoint",
    }
    cells: list[DiscriminationCell] = []
    for row in loaded.get("pairs", ()):
        pair = _pair(*tuple(row["pair"]))
        flags = tuple(row["separates"])
        if len(flags) != 6 or any(type(flag) is not bool for flag in flags):
            raise ValueError(f"pair {pair!r} needs six boolean rung rows")
        for rung, separates in enumerate(flags, start=1):
            cells.append(
                DiscriminationCell(
                    pair=pair,
                    rung=rung,
                    separates=separates,
                    measurement=measurements[rung],
                    required_access=cost_for[rung].access,
                )
            )
    expected_pairs = {
        _pair(left, right)
        for index, left in enumerate(EXPLANATION_KINDS)
        for right in EXPLANATION_KINDS[index + 1 :]
    }
    expected_cells = {(rung, pair) for rung in range(1, 7) for pair in expected_pairs}
    actual_cells = {(cell.rung, cell.pair) for cell in cells}
    if actual_cells != expected_cells or len(cells) != len(expected_cells):
        raise ValueError("discrimination table must contain every explanation pair at every rung")
    return costs, tuple(cells)


_DECLARED_COST_KEYS = {"rung", "rank", "money_usd", "samples", "minutes"}
_DECLARED_MONEY = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_COST_TABLE_MAX_BYTES = 65536


def _cost_table_error(detail: str) -> PlannerRefusal:
    return PlannerRefusal("PLANNER_COST_TABLE_INVALID", detail)


def _declared_cost_rows(raw: bytes) -> tuple[dict[str, Any], ...]:
    """Parse declared local planning costs without accepting access or separation overrides."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _cost_table_error("cost declaration contains a duplicate JSON key")
            result[key] = value
        return result

    if len(raw) > _COST_TABLE_MAX_BYTES:
        raise _cost_table_error("cost declaration exceeds 65536 bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except RecursionError as error:
        raise _cost_table_error("cost declaration nesting exceeds the JSON parser limit") from error
    except (UnicodeDecodeError, ValueError) as error:
        raise _cost_table_error("cost declaration must be a complete UTF-8 JSON object") from error
    if not isinstance(value, dict) or set(value) != {"version", "costs"}:
        raise _cost_table_error("cost declaration requires exactly version and costs")
    if type(value["version"]) is not int or value["version"] != 1:
        raise _cost_table_error("cost declaration version must be integer 1")
    rows = value["costs"]
    if not isinstance(rows, list) or len(rows) != 6:
        raise _cost_table_error("cost declaration requires exactly six rung rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != _DECLARED_COST_KEYS:
            raise _cost_table_error(
                "each cost row requires exactly rung, rank, money_usd, samples and minutes"
            )
        if type(row["rung"]) is not int or row["rung"] not in range(1, 7):
            raise _cost_table_error("cost rung must be an integer from 1 through 6")
        if type(row["rank"]) is not int or row["rank"] <= 0:
            raise _cost_table_error("cost rank must be a positive integer; equal tiers are allowed")
        for key in ("samples", "minutes"):
            if type(row[key]) is not int or row[key] < 0:
                raise _cost_table_error(f"cost {key} must be a non-negative integer")
        money = row["money_usd"]
        if not isinstance(money, str) or not _DECLARED_MONEY.fullmatch(money):
            raise _cost_table_error("cost money_usd must be a finite non-negative decimal string")
    if {row["rung"] for row in rows} != set(range(1, 7)):
        raise _cost_table_error("cost declaration must contain each rung exactly once")
    return tuple(sorted(rows, key=lambda row: row["rung"]))


def _read_cost_bytes(source: Path) -> bytes:
    with source.open("rb") as stream:
        return stream.read(_COST_TABLE_MAX_BYTES + 1)


def _load_declared_costs(path: Path | str, shipped: tuple[RungCost, ...]):
    if not isinstance(path, (Path, str)):
        raise _cost_table_error("cost_table must name a local JSON file")
    source = Path(path)
    try:
        raw = _read_cost_bytes(source)
        rows = _declared_cost_rows(raw)
        if _read_cost_bytes(source) != raw:
            raise PlannerRefusal(
                "PLANNER_COST_TABLE_CHANGED", "cost declaration changed while being read"
            )
    except OSError as error:
        raise _cost_table_error("cost declaration could not be read") from error
    fixed_access = {cost.rung: cost.access for cost in shipped}
    costs = tuple(
        RungCost(
            **{**row, "money_usd": Decimal(row["money_usd"]), "access": fixed_access[row["rung"]]}
        )
        for row in rows
    )
    provenance = {
        "schema": "reward-lens.declared-planning-costs/1",
        "meaning": "declared_local_planning_costs_not_measured_performance",
        "source_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "source_bytes_utf8": raw.decode("utf-8"),
        "shipped_discrimination_digest": "sha256:"
        + hashlib.sha256(_table_path().read_bytes()).hexdigest(),
    }
    return costs, provenance, source


def _require_unchanged_cost_table(
    source: Path | None, provenance: dict[str, object] | None
) -> None:
    if source is None:
        return
    try:
        current = "sha256:" + hashlib.sha256(_read_cost_bytes(source)).hexdigest()
    except OSError as error:
        raise PlannerRefusal(
            "PLANNER_COST_TABLE_CHANGED", "cost declaration disappeared before registration"
        ) from error
    if current != provenance["source_digest"]:
        raise PlannerRefusal(
            "PLANNER_COST_TABLE_CHANGED", "cost declaration changed before registration"
        )


def _validate_cost_table_provenance(choice: dict[str, Any]) -> None:
    """A reopened ledger consumes its sealed bytes, never a changed external file."""
    if "cost_table_provenance" not in choice:
        return
    provenance = choice["cost_table_provenance"]
    try:
        if set(provenance) != {
            "schema",
            "meaning",
            "source_digest",
            "source_bytes_utf8",
            "shipped_discrimination_digest",
        }:
            raise ValueError("unexpected cost provenance fields")
        if (
            provenance["schema"] != "reward-lens.declared-planning-costs/1"
            or provenance["meaning"] != "declared_local_planning_costs_not_measured_performance"
        ):
            raise ValueError("invalid cost provenance schema")
        raw = provenance["source_bytes_utf8"].encode("utf-8")
        if "sha256:" + hashlib.sha256(raw).hexdigest() != provenance["source_digest"]:
            raise ValueError("sealed cost bytes do not match their digest")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["shipped_discrimination_digest"]):
            raise ValueError("invalid shipped discrimination identity")
        declared = {row["rung"]: row for row in _declared_cost_rows(raw)}
        alternatives = choice["alternatives"]
        if len(alternatives) != 6 or {row["rung"] for row in alternatives} != set(declared):
            raise ValueError("cost declaration does not cover registered alternatives")
        for item in alternatives:
            actual = item["cost"]
            expected = declared[item["rung"]]
            if any(
                actual[key] != expected[key] for key in ("rung", "rank", "samples", "minutes")
            ) or Decimal(actual["money_usd"]) != Decimal(expected["money_usd"]):
                raise ValueError("cost declaration disagrees with registered alternatives")
    except (AttributeError, KeyError, TypeError, ValueError, InvalidOperation) as error:
        raise PlannerRefusal(
            "PLANNER_LEDGER_TAMPERED", "registered cost declaration is invalid"
        ) from error


def _require_orderings(orderings: Sequence[str]) -> None:
    positions = {step: position for position, step in enumerate(orderings)}
    for before, after in _ORDERING_RULES:
        if (
            before not in positions
            or after not in positions
            or positions[before] >= positions[after]
        ):
            raise PlannerRefusal(
                "PLANNER_ORDERING_REQUIRED",
                f"{before} must be registered before {after}",
            )


def _live_pairs(live: Iterable[Explanation]) -> tuple[tuple[str, str], ...]:
    explanations = tuple(live)
    if len(explanations) < 2:
        raise ValueError("planner needs at least two live explanations")
    kinds = tuple(item.kind for item in explanations)
    if len(kinds) != len(set(kinds)):
        raise ValueError("live explanations must have distinct kinds")
    return tuple(
        _pair(left, right) for index, left in enumerate(kinds) for right in kinds[index + 1 :]
    )


def _prior_case(live: tuple[Explanation, ...], cases: Iterable[PlannerCase]) -> PlannerCase | None:
    live_kinds = {item.kind for item in live}
    applicable = [case for case in cases if set(case.explanation_kinds).issubset(live_kinds)]
    return min(
        applicable,
        key=lambda case: (-len(case.explanation_kinds), case.rung, case.id),
        default=None,
    )


def plan(
    live: Iterable[Explanation],
    availability: Iterable[str],
    budget: Decimal,
    *,
    orderings: Sequence[str],
    ledger: PlannerLedger,
    evaluator_id: str | None = None,
    retry_limit: int = 2,
    corpus: Iterable[PlannerCase] | CorpusStore | Corpus | None = None,
    planning_annotations: Iterable[PlanningAnnotation] | None = None,
    cost_table: Path | str | None = None,
) -> PlanChoice:
    """Pre-register the cheapest feasible complete or partial discrimination.

    `budget` is an explicit declared ceiling for the choice.  It does not claim
    that the amount was spent; actual provider spend remains a later record.
    ``cost_table`` optionally declares all six local cost rows in version-1 JSON.
    Equal declared costs permit prior support to break ties. Access and separation
    remain shipped facts; exact declaration bytes are sealed before execution.
    """

    if type(budget) is not Decimal or budget < 0:
        raise TypeError("planner budget must be a non-negative Decimal")
    if not isinstance(ledger, PlannerLedger):
        raise TypeError("planner needs a persisted PlannerLedger before selecting an experiment")
    if type(retry_limit) is not int or retry_limit < 0:
        raise ValueError("planner retry limit must be a non-negative integer")
    _require_orderings(orderings)
    evaluator_contract = ledger.evaluator_contract(evaluator_id)
    live_rows = tuple(live)
    pairs = _live_pairs(live_rows)
    costs, cells = load_discrimination_table()
    cost_table_provenance = None
    cost_table_source = None
    if cost_table is not None:
        costs, cost_table_provenance, cost_table_source = _load_declared_costs(cost_table, costs)
    corpus_provenance = None
    if corpus is None or isinstance(corpus, Corpus):
        if planning_annotations is not None:
            raise ValueError("supply planning annotations when constructing the Corpus facade")
        query = " ".join(item.kind.replace("_", " ") for item in live_rows)
        cases, corpus_provenance = consult_corpus(Corpus() if corpus is None else corpus, query)
    elif isinstance(corpus, CorpusStore):
        cases = load_planner_cases(corpus=corpus, annotations=planning_annotations)
    else:
        if planning_annotations is not None:
            raise ValueError("planning annotations require a CorpusStore")
        cases = tuple(load_planner_cases() if corpus is None else corpus)
    available = frozenset(availability)
    by_cell = {(cell.rung, cell.pair): cell for cell in cells}
    alternatives: list[PlanAlternative] = []
    for cost in sorted(costs, key=lambda value: value.ordering_key):
        rung_cells = tuple(by_cell[(cost.rung, pair)] for pair in pairs)
        separated = tuple(cell.pair for cell in rung_cells if cell.separates)
        missing = tuple(sorted(set(cost.access) - available))
        support_ids = tuple(
            sorted(
                case.id
                for case in cases
                if case.rung == cost.rung
                and any(set(pair).issubset(case.explanation_kinds) for pair in separated)
            )
        )
        alternatives.append(
            PlanAlternative(
                rung=cost.rung,
                cost=cost,
                separates_pairs=separated,
                unresolved_pairs=tuple(pair for pair in pairs if pair not in separated),
                feasible=not missing and cost.money_usd <= budget,
                missing_access=missing,
                over_budget=cost.money_usd > budget,
                prior_support_case_ids=support_ids,
            )
        )
    complete = tuple(
        alternative
        for alternative in alternatives
        if alternative.feasible and not alternative.unresolved_pairs
    )
    candidates = complete or tuple(
        alternative
        for alternative in alternatives
        if alternative.feasible and alternative.separates
    )
    selected: PlanAlternative | None = None
    compared_support: tuple[tuple[int, tuple[str, ...]], ...] = ()
    if candidates:
        # `rung` is the old stable-id suffix of ordering_key, not a declared cost.
        # Corpus support may break only a tie before that suffix would have done so.
        cheapest = min(item.cost.ordering_key[:-1] for item in candidates)
        equal_cost = tuple(item for item in candidates if item.cost.ordering_key[:-1] == cheapest)
        coverage = max(len(item.separates_pairs) for item in equal_cost)
        ties = tuple(item for item in equal_cost if len(item.separates_pairs) == coverage)
        compared_support = tuple(
            (item.rung, item.prior_support_case_ids)
            for item in sorted(ties, key=lambda item: item.rung)
        )
        selected = min(ties, key=lambda item: (-len(item.prior_support_case_ids), item.rung))
        # The pre-execution alternatives expose the same deterministic tie order.
        alternative_order = {
            item.rung: index
            for index, item in enumerate(
                sorted(ties, key=lambda item: (-len(item.prior_support_case_ids), item.rung))
            )
        }
        alternatives.sort(
            key=lambda item: (
                item.cost.ordering_key[:-1],
                -len(item.separates_pairs) if item in equal_cost else 0,
                alternative_order.get(item.rung, item.rung),
            )
        )
    prior = _prior_case(live_rows, cases)
    if selected is None:
        missing = tuple(
            sorted(
                {
                    item
                    for alternative in alternatives
                    if alternative.separates
                    for item in alternative.missing_access
                }
            )
        )
        measurement = (
            "no shipped rung discriminates the remaining live explanation pairs"
            if not any(alternative.separates for alternative in alternatives)
            else "supply the named access for a discriminating rung within the declared budget"
        )
        choice = PlanChoice(
            kind="unresolved",
            rung=None,
            cost=None,
            separated_pairs=(),
            unresolved_pairs=pairs,
            alternatives=tuple(alternatives),
            expected_outcomes=(),
            compared_prior_support=compared_support,
            missing_access=missing,
            missing_measurement=measurement,
            prior_case=prior,
            corpus_provenance=corpus_provenance,
            cost_table_provenance=cost_table_provenance,
            evaluator_contract=evaluator_contract,
            refusal=PlannerRefusalInfo("PLANNER_NO_FEASIBLE_DISCRIMINATOR", measurement),
        )
        _require_unchanged_cost_table(cost_table_source, cost_table_provenance)
        registration_id, choice_digest = ledger.register_choice(choice, retry_limit=retry_limit)
        return replace(choice, registration_id=registration_id, choice_digest=choice_digest)
    kind: Literal["complete", "partial"] = (
        "complete" if not selected.unresolved_pairs else "partial"
    )
    signature = f"r{selected.rung}:" + "|".join("/".join(pair) for pair in selected.separates_pairs)
    attempts = ledger.inconclusive_attempts(signature)
    ledger.require_consistent_retry_limit(signature, retry_limit)
    if attempts >= retry_limit:
        raise PlannerRefusal(
            "PLANNER_RETRY_EXHAUSTED",
            f"{signature} already has {attempts} inconclusive attempts; limit is {retry_limit}",
        )
    expectations = tuple(
        f"rung {selected.rung} can distinguish {' versus '.join(pair)} by {by_cell[(selected.rung, pair)].measurement}"
        for pair in selected.separates_pairs
    )
    if prior is not None:
        expectations += (f"prior-case analogy {prior.id}: {prior.analogy}",)
    else:
        expectations += ("no applicable prior case was found in the consulted planner corpus",)
    choice = PlanChoice(
        kind=kind,
        rung=selected.rung,
        cost=selected.cost,
        separated_pairs=selected.separates_pairs,
        unresolved_pairs=selected.unresolved_pairs,
        alternatives=tuple(alternatives),
        expected_outcomes=expectations,
        compared_prior_support=compared_support,
        prior_case=prior,
        corpus_provenance=corpus_provenance,
        cost_table_provenance=cost_table_provenance,
        evaluator_contract=evaluator_contract,
    )
    _require_unchanged_cost_table(cost_table_source, cost_table_provenance)
    registration_id, choice_digest = ledger.register_choice(choice, retry_limit=retry_limit)
    return replace(choice, registration_id=registration_id, choice_digest=choice_digest)


def record_result(
    choice: PlanChoice,
    ledger: PlannerLedger,
    *,
    evidence: ExecutionEvidence,
    analogy_survived: bool | None,
) -> PlannerResult:
    """Append evaluator-derived result evidence for a registered choice."""

    if not isinstance(ledger, PlannerLedger):
        raise TypeError("record_result needs the persisted PlannerLedger")
    if not isinstance(evidence, ExecutionEvidence):
        raise TypeError("record_result needs typed evaluator artifact evidence")
    if choice.kind == "unresolved" or choice.rung is None:
        raise ValueError("an unresolved plan has no experiment to record")
    registered = ledger.registered_choice(choice)
    if evidence.choice_digest != registered["choice_digest"]:
        raise PlannerRefusal(
            "PLANNER_EVIDENCE_CHOICE_MISMATCH", "executor evidence names another choice digest"
        )
    if choice.evaluator_contract is None:
        raise PlannerRefusal(
            "PLANNER_EVALUATOR_UNTRUSTED", "registered executable choice has no evaluator contract"
        )
    evaluator = ledger._trusted_evaluator(choice.evaluator_contract)
    if not ledger._evidence_matches_contract(evidence, choice.evaluator_contract, evaluator):
        raise PlannerRefusal(
            "PLANNER_EVIDENCE_UNAUTHENTICATED",
            "evidence is not signed by the evaluator contract registered before execution",
        )
    if ledger.result_exists(choice.registration_id):
        raise PlannerRefusal(
            "PLANNER_RESULT_ALREADY_RECORDED", "a registered choice accepts one result record"
        )
    ruled = _derive_eliminations_from_evidence(registered["choice"], evidence)
    if choice.prior_case is not None and type(analogy_survived) is not bool:
        raise TypeError("a consulted applicable prior case needs a recorded survival result")
    if choice.prior_case is None and analogy_survived is not None:
        raise ValueError("no prior-case analogy was consulted for this choice")
    return ledger.append_result(
        PlannerResult(
            registration_id=choice.registration_id,
            choice_digest=choice.choice_digest,
            evidence=evidence,
            ruled_out=ruled,
            informative=bool(ruled),
            analogy_survived=analogy_survived,
        )
    )
