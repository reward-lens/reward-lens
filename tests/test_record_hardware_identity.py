"""BLK-018: the hardware a series was read on is a recorded field, and a mismatch raises.

`CHAIN_GAPS.md:56` gives the reason in one sentence: "Two card generations gave 53.0% and 57.1% on
one checkpoint, which is inside the range of transitions this design locates." A behavioural series
read across two machines can show a step change that has nothing to do with the policy, at a
magnitude the design is trying to measure, and nothing in the record could say so because the three
fields did not exist and no read asserted them.

The three fields are the hardware identifier, the engine version and the decoding parameters, and
they are required to be fixed across every checkpoint of every run. What is proven here:

- the identifier is the five parts the row names, and two different cards are different identities;
- a read whose context differs from the run's own **raises**, and the exception names which of the
  three parts differed rather than saying only that something did;
- a part that was never captured raises too, rather than comparing equal to everything. An
  uncaptured field is not a matching one, and Law 4's "unavailable is not pass" is exactly the
  assumption a nan-like default would smuggle in;
- capture degrades honestly on a machine with no usable GPU, which is the machine this runs on: it
  returns an identity that says it was not captured rather than one that looks captured and is
  empty.
"""

from __future__ import annotations

import pytest

from reward_lens.record.hardware import (
    DecodingParameters,
    EngineIdentity,
    HardwareIdentity,
    ReadContext,
    ReadContextMismatch,
    capture_hardware,
)

A100 = HardwareIdentity(
    gpu_name="NVIDIA A100-SXM4-40GB",
    pci_device_id="10DE:20B0",
    driver_version="535.104.05",
    cuda_version="12.2",
    smi_digest="smi:1111111111111111",
)
H100 = HardwareIdentity(
    gpu_name="NVIDIA H100 80GB HBM3",
    pci_device_id="10DE:2330",
    driver_version="535.104.05",
    cuda_version="12.2",
    smi_digest="smi:2222222222222222",
)
VLLM = EngineIdentity(name="vllm", version="0.6.3")
GREEDY = DecodingParameters(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0)


def _context(hardware=A100, engine=VLLM, decoding=GREEDY) -> ReadContext:
    return ReadContext(hardware=hardware, engine=engine, decoding=decoding)


# ---------------------------------------------------------------------------
# The identity itself
# ---------------------------------------------------------------------------


def test_the_identifier_is_the_five_parts_the_contract_names():
    """GPU name, PCI device id, driver, CUDA, and a hash of `nvidia-smi -q`."""
    canonical = A100.__canonical__()
    assert set(canonical) == {
        "gpu_name",
        "pci_device_id",
        "driver_version",
        "cuda_version",
        "smi_digest",
    }
    assert HardwareIdentity.from_canonical(canonical) == A100


def test_two_card_generations_are_two_identities():
    """The case the gap record is about: 53.0% and 57.1% on one checkpoint, two cards."""
    assert A100 != H100
    assert A100.fingerprint != H100.fingerprint


def test_the_same_card_reported_twice_is_one_identity():
    """Otherwise every read would mismatch and the assertion would be noise rather than a check."""
    again = HardwareIdentity(**A100.__canonical__())
    assert again == A100
    assert again.fingerprint == A100.fingerprint


# ---------------------------------------------------------------------------
# The refusal at read time, which is the closure proof
# ---------------------------------------------------------------------------


def test_a_read_on_a_different_card_raises():
    """The closure proof: a bank read on a different hardware identifier raises.

    The assertion is the raise. Nothing is read out of the exception and turned into a number.
    """
    run = _context()

    with pytest.raises(ReadContextMismatch) as excinfo:
        run.require_same(_context(hardware=H100), what="behavioural series at step 88")

    message = str(excinfo.value)
    assert "hardware" in message
    assert "A100" in message and "H100" in message
    assert "behavioural series at step 88" in message


def test_a_read_on_the_same_context_returns_rather_than_raising():
    """The input perturbation's other side. Without this the check could be an unconditional raise."""
    run = _context()
    assert run.require_same(_context(), what="behavioural series at step 88") is None


def test_a_read_under_a_different_engine_version_raises_and_says_so():
    """Same card, different engine. The message names the part that moved, not just that one did."""
    run = _context()

    with pytest.raises(ReadContextMismatch) as excinfo:
        run.require_same(_context(engine=EngineIdentity(name="vllm", version="0.7.0")), what="read")

    assert "engine" in str(excinfo.value)
    assert "hardware" not in str(excinfo.value)


def test_a_read_at_a_different_temperature_raises_and_says_so():
    """Decoding parameters are the third fixed field, and temperature is the one that bites."""
    run = _context()
    hotter = DecodingParameters(
        temperature=1.2, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0
    )

    with pytest.raises(ReadContextMismatch) as excinfo:
        run.require_same(_context(decoding=hotter), what="read")

    assert "decoding" in str(excinfo.value)
    assert "temperature" in str(excinfo.value)


def test_every_part_that_moved_is_named_not_only_the_first():
    """A refusal that stops at the first difference sends somebody back for a second round."""
    run = _context()
    other = ReadContext(
        hardware=H100,
        engine=EngineIdentity(name="vllm", version="0.7.0"),
        decoding=GREEDY,
    )

    with pytest.raises(ReadContextMismatch) as excinfo:
        run.require_same(other, what="read")

    message = str(excinfo.value)
    assert "hardware" in message and "engine" in message


# ---------------------------------------------------------------------------
# Unavailable is not pass
# ---------------------------------------------------------------------------


def test_an_uncaptured_identity_does_not_compare_equal_to_a_captured_one():
    """A field nobody captured is not a field that matched."""
    run = _context()
    blind = ReadContext(hardware=HardwareIdentity.uncaptured(), engine=VLLM, decoding=GREEDY)

    with pytest.raises(ReadContextMismatch) as excinfo:
        run.require_same(blind, what="read")

    assert "not captured" in str(excinfo.value)


def test_two_uncaptured_identities_do_not_match_each_other_either():
    """Two machines nobody identified are not thereby the same machine.

    This is the assertion the whole field turns on. If "unknown equals unknown" passed, a run read
    on two cards with no capture on either would sail through the check that exists to catch it.
    """
    blind = ReadContext(hardware=HardwareIdentity.uncaptured(), engine=VLLM, decoding=GREEDY)

    with pytest.raises(ReadContextMismatch):
        blind.require_same(
            ReadContext(hardware=HardwareIdentity.uncaptured(), engine=VLLM, decoding=GREEDY),
            what="read",
        )


def test_an_uncaptured_identity_is_not_silently_falsy():
    """`is_captured` says it out loud, so a caller cannot mistake it for a captured blank."""
    assert A100.is_captured is True
    assert HardwareIdentity.uncaptured().is_captured is False


# ---------------------------------------------------------------------------
# Capture on this machine, which has no usable GPU
# ---------------------------------------------------------------------------


def test_capture_never_invents_an_identity():
    """On a machine with no usable card, capture returns an uncaptured identity and says so.

    It is a real call rather than a mock: this box has a CUDA driver too old to run on, which is the
    condition the honest path exists for, so the honest path is what runs here.
    """
    captured = capture_hardware()

    assert isinstance(captured, HardwareIdentity)
    if not captured.is_captured:
        assert captured.gpu_name is None
        assert captured.smi_digest is None
    else:
        # On a machine that does have a card, everything the format names must be there.
        assert captured.gpu_name and captured.pci_device_id
        assert captured.driver_version and captured.cuda_version
        assert captured.smi_digest and captured.smi_digest.startswith("smi:")
