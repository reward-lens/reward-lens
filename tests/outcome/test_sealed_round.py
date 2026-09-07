"""One sealed round per candidate set, a one-time nonce, and a counter that a restart cannot reset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_lens.contracts.errors import UsageError
from reward_lens.outcome import AcceptanceRoundExhausted, Round, RoundLedger, SealedRequest

CANDIDATE = "sha256:" + "1" * 64
OTHER_CANDIDATE = "sha256:" + "2" * 64
PROTOCOL = "sha256:" + "3" * 64


@pytest.fixture
def ledger(tmp_path: Path) -> RoundLedger:
    return RoundLedger(tmp_path / "rounds.json")


def seal(ledger: RoundLedger, candidate: str = CANDIDATE) -> Round:
    return ledger.seal(
        candidate_set=candidate, partition_id="acceptance-suite-1", protocol_digest=PROTOCOL
    )


def test_one_candidate_set_gets_one_sealed_round(ledger: RoundLedger) -> None:
    first = seal(ledger)
    assert isinstance(first, Round)
    assert first.candidate_set == CANDIDATE
    assert first.state == "sealed"
    with pytest.raises(AcceptanceRoundExhausted) as caught:
        seal(ledger)
    assert caught.value.code == "RL0302"
    assert caught.value.exit_code == 4
    assert CANDIDATE in caught.value.message
    assert caught.value.context["candidate_set"] == CANDIDATE


def test_a_different_candidate_set_gets_its_own_round(ledger: RoundLedger) -> None:
    first = seal(ledger)
    second = seal(ledger, OTHER_CANDIDATE)
    assert first.round_id != second.round_id
    assert {row.candidate_set for row in ledger.rounds()} == {CANDIDATE, OTHER_CANDIDATE}


def test_the_round_is_spent_by_a_sealed_request_carrying_the_four_fields(ledger: RoundLedger) -> None:
    sealed = seal(ledger)
    request = SealedRequest(
        candidate_digest=CANDIDATE,
        protocol_digest=PROTOCOL,
        partition_id="acceptance-suite-1",
        nonce=sealed.nonce,
    )
    spent = ledger.spend(request)
    assert spent.state == "spent"
    assert spent.round_id == sealed.round_id


def test_a_replayed_nonce_is_rejected_and_logged(ledger: RoundLedger) -> None:
    sealed = seal(ledger)
    request = SealedRequest(
        candidate_digest=CANDIDATE,
        protocol_digest=PROTOCOL,
        partition_id="acceptance-suite-1",
        nonce=sealed.nonce,
    )
    ledger.spend(request)
    with pytest.raises(AcceptanceRoundExhausted) as caught:
        ledger.spend(request)
    assert caught.value.code == "RL0302"
    assert caught.value.context["reason"] == "the nonce was already spent"
    assert [row["reason"] for row in ledger.rejections()] == ["the nonce was already spent"]


def test_a_forged_candidate_digest_is_rejected_and_logged(ledger: RoundLedger) -> None:
    sealed = seal(ledger)
    forged = SealedRequest(
        candidate_digest=OTHER_CANDIDATE,
        protocol_digest=PROTOCOL,
        partition_id="acceptance-suite-1",
        nonce=sealed.nonce,
    )
    with pytest.raises(AcceptanceRoundExhausted) as caught:
        ledger.spend(forged)
    assert caught.value.context["reason"] == "no sealed round carries this candidate digest"
    assert ledger.rejections()[-1]["candidate_digest"] == OTHER_CANDIDATE
    assert ledger.rounds()[0].state == "sealed"


def test_a_forged_protocol_digest_is_rejected(ledger: RoundLedger) -> None:
    sealed = seal(ledger)
    forged = SealedRequest(
        candidate_digest=CANDIDATE,
        protocol_digest="sha256:" + "9" * 64,
        partition_id="acceptance-suite-1",
        nonce=sealed.nonce,
    )
    with pytest.raises(AcceptanceRoundExhausted) as caught:
        ledger.spend(forged)
    assert "protocol" in caught.value.context["reason"]


def test_the_counter_is_on_disk_so_a_restart_cannot_reset_it(tmp_path: Path) -> None:
    path = tmp_path / "rounds.json"
    first = RoundLedger(path)
    seal(first)
    assert json.loads(path.read_text())["rounds"][0]["candidate_set"] == CANDIDATE

    restarted = RoundLedger(path)
    assert len(restarted.rounds()) == 1
    with pytest.raises(AcceptanceRoundExhausted):
        seal(restarted)


def test_a_spent_round_stays_spent_across_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "rounds.json"
    first = RoundLedger(path)
    sealed = seal(first)
    first.spend(
        SealedRequest(
            candidate_digest=CANDIDATE,
            protocol_digest=PROTOCOL,
            partition_id="acceptance-suite-1",
            nonce=sealed.nonce,
        )
    )
    restarted = RoundLedger(path)
    assert restarted.rounds()[0].state == "spent"
    with pytest.raises(AcceptanceRoundExhausted):
        restarted.spend(
            SealedRequest(
                candidate_digest=CANDIDATE,
                protocol_digest=PROTOCOL,
                partition_id="acceptance-suite-1",
                nonce=sealed.nonce,
            )
        )


def test_the_exhaustion_refusal_is_a_usage_error(ledger: RoundLedger) -> None:
    assert issubclass(AcceptanceRoundExhausted, UsageError)
    seal(ledger)
    with pytest.raises(UsageError):
        seal(ledger)


def test_a_second_round_needs_a_fresh_partition_or_a_new_protocol(ledger: RoundLedger) -> None:
    seal(ledger)
    second = ledger.seal(
        candidate_set=CANDIDATE, partition_id="acceptance-suite-2", protocol_digest=PROTOCOL
    )
    assert second.partition_id == "acceptance-suite-2"
    assert len(ledger.rounds()) == 2
