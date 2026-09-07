"""The sealed round: one per candidate set, spent by a sealed request, persisted to disk.

D-40's mechanism, exactly: the worker accepts only a sealed request carrying the candidate digest,
the protocol digest, the partition id and a one-time nonce; a replayed nonce or a forged candidate
digest is rejected and logged; and the counter lives in a file, so a restart cannot reset the round
budget by restarting the process that held it.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, replace
from pathlib import Path

from reward_lens.contracts.canonical import digest as canonical_digest
from reward_lens.partitions.access import utc_now

from .errors import AcceptanceRoundExhausted

__all__ = ["Round", "RoundLedger", "SealedRequest"]


@dataclass(frozen=True)
class SealedRequest:
    """What the outcome worker will accept, and the only thing it will accept."""

    candidate_digest: str
    protocol_digest: str
    partition_id: str
    nonce: str


@dataclass(frozen=True)
class Round:
    """One sealed comparison round. `state` is `sealed` until it is spent, then `spent`."""

    round_id: str
    candidate_set: str
    partition_id: str
    protocol_digest: str
    nonce: str
    sealed_at: str
    state: str = "sealed"
    spent_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "round_id": self.round_id,
            "candidate_set": self.candidate_set,
            "partition_id": self.partition_id,
            "protocol_digest": self.protocol_digest,
            "nonce": self.nonce,
            "sealed_at": self.sealed_at,
            "state": self.state,
            "spent_at": self.spent_at,
        }


class RoundLedger:
    """The persisted round budget: the rounds that were sealed, and every request refused.

    The file is rewritten whole on each change and replaced atomically, so a ledger is either the
    state before a seal or the state after it, never half of each.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._rounds: list[Round] = []
        self._rejections: list[dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        state = json.loads(self.path.read_text(encoding="utf-8"))
        self._rounds = [Round(**row) for row in state.get("rounds", [])]
        self._rejections = list(state.get("rejections", []))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rounds": [row.to_dict() for row in self._rounds],
            "rejections": self._rejections,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    # --- reading ----------------------------------------------------------------------------

    def rounds(self) -> tuple[Round, ...]:
        return tuple(self._rounds)

    def rejections(self) -> tuple[dict[str, object], ...]:
        return tuple(self._rejections)

    # --- writing ----------------------------------------------------------------------------

    def seal(self, *, candidate_set: str, partition_id: str, protocol_digest: str) -> Round:
        """Seal the one round this candidate set gets on this partition."""
        for row in self._rounds:
            if row.candidate_set == candidate_set and row.partition_id == partition_id:
                self._reject(
                    candidate_set,
                    partition_id,
                    "a round was already sealed for this candidate set",
                    nonce="",
                )
        nonce = secrets.token_hex(16)
        sealed = Round(
            round_id=canonical_digest(
                {
                    "candidate_set": candidate_set,
                    "partition_id": partition_id,
                    "protocol_digest": protocol_digest,
                    "nonce": nonce,
                }
            )[7:23],
            candidate_set=candidate_set,
            partition_id=partition_id,
            protocol_digest=protocol_digest,
            nonce=nonce,
            sealed_at=utc_now(),
        )
        self._rounds.append(sealed)
        self._save()
        return sealed

    def spend(self, request: SealedRequest) -> Round:
        """Spend the sealed round a request names, or refuse and log why."""
        for index, row in enumerate(self._rounds):
            if row.nonce != request.nonce or row.partition_id != request.partition_id:
                continue
            if row.candidate_set != request.candidate_digest:
                self._reject(
                    request.candidate_digest,
                    request.partition_id,
                    "no sealed round carries this candidate digest",
                    nonce=request.nonce,
                )
            if row.protocol_digest != request.protocol_digest:
                self._reject(
                    request.candidate_digest,
                    request.partition_id,
                    "no sealed round carries this protocol digest",
                    nonce=request.nonce,
                )
            if row.state == "spent":
                self._reject(
                    request.candidate_digest,
                    request.partition_id,
                    "the nonce was already spent",
                    nonce=request.nonce,
                )
            spent = replace(row, state="spent", spent_at=utc_now())
            self._rounds[index] = spent
            self._save()
            return spent
        self._reject(
            request.candidate_digest,
            request.partition_id,
            "no sealed round carries this nonce",
            nonce=request.nonce,
        )
        raise AssertionError("unreachable: _reject always raises")  # pragma: no cover

    def _reject(self, candidate_set: str, partition_id: str, reason: str, *, nonce: str) -> None:
        self._rejections.append(
            {
                "at": utc_now(),
                "candidate_digest": candidate_set,
                "partition_id": partition_id,
                "nonce": nonce,
                "reason": reason,
            }
        )
        self._save()
        raise AcceptanceRoundExhausted(
            candidate_set=candidate_set, partition_id=partition_id, reason=reason
        )
