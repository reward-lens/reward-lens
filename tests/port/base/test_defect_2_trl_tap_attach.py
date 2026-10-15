"""Defect 2 of C-005: the TRL tap attaching on a machine without transformers.

The base-install CI job has no transformers. In the assay tree `_build_callback` imports
`TrainerCallback` at the top of its body, so on that machine `attach` raises ImportError, the
guard in `attach` files it as an adapter note, and the tap returns the trainer having registered
no callback at all. Nothing says transformers was what it wanted; the run simply records nothing.

The release carries `_callback_base()`, which falls back to `object` when transformers is absent,
so the callback class is still built and still registered and the step boundary still works over a
trainer-shaped object. That is the half of the adapter meant to work on a base install.

These tests run on the base closure. If transformers is ever installed in the test environment the
fallback is not the path taken, and the assertions still hold, because the real `TrainerCallback`
is a plain class and `add_callback` here is a stub.
"""

from __future__ import annotations

from typing import Any

from reward_lens.tap.adapters.trl import TRLTap


class _TrainerShaped:
    """The one method `attach` calls on the trainer, and nothing else."""

    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def add_callback(self, cb: Any) -> None:
        self.callbacks.append(cb)


def test_the_tap_registers_its_callback_without_transformers() -> None:
    tap = TRLTap()
    trainer = _TrainerShaped()

    assert tap.attach(trainer) is trainer
    assert trainer.callbacks, (
        "attach registered no callback: "
        f"{tap.adapter_exceptions} adapter exception(s) {tap.adapter_exception_keys}"
    )


def test_attaching_without_transformers_files_no_adapter_exception() -> None:
    """The silent half of the defect: the ImportError was absorbed as an adapter note."""
    tap = TRLTap()
    tap.attach(_TrainerShaped())

    assert tap.adapter_exception_keys == []
    assert tap.adapter_exceptions == 0


def test_the_registered_callback_carries_the_step_boundary() -> None:
    """A stand-in base class is only useful if the methods the trainer calls are still there."""
    tap = TRLTap()
    trainer = _TrainerShaped()
    tap.attach(trainer)

    cb = trainer.callbacks[0]
    assert callable(getattr(cb, "on_step_end", None))
    assert callable(getattr(cb, "on_pre_optimizer_step", None))
