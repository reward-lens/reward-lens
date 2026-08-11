"""Writing an adapter checkpoint at a chosen precision. BLK-025.

**The defect this exists for.** PEFT's `get_peft_model` upcasts LoRA adapter
parameters to fp32 even under a bf16 base, and TRL 1.10.0 names the problem in a
comment at `grpo_trainer.py:429-431` and then gates its fix, verbatim, on

    args.deepspeed_plugin is not None
    and args.deepspeed_plugin.zero_stage == 3
    and not _is_quantized_model
    and Version(peft.__version__) >= Version("0.12.0")

before setting `autocast_adapter_dtype = False` at `:446`. This project runs on
one GPU with no DeepSpeed, so the branch never fires and every adapter is
written in fp32. On the corrected inventory that is 49.45 GB against 90.94 GB,
on a host that bills stopped volumes by the GB-month.

**One correction to how the gap ledger words it.** `G25` calls this "bf16
adapter conversion **on save**". It is a **load-time** `get_peft_model` gate,
not a save-time conversion. Part 4.3 has it right: the conversion is "an
explicit post-save step". This module is that step.

**What must NOT be converted, and why it is a correctness matter rather than a
storage one.** Two checkpoints stay fp32:

    the resume checkpoint   a run resumed from a bf16 round-trip continues from
                            different low bits than the run that wrote it, so
                            the resumed trajectory is not the parent's
    step 130                every fork in the design is taken from it, and a
                            fork's whole claim is that it starts where the
                            parent was

`ARCHIVE` is therefore the default and `EXACT` is asked for by name, because a
writer whose safe mode is the non-default is a writer that will be called wrong
once.

**TRL and PEFT are read-only here.** Part 3.7's rule and the project's standing
rule that a pinned third-party trainer is not patched. Nothing in this module
imports either.
"""

from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Precision", "write_adapter", "read_adapter", "adapter_dtypes", "AdapterPrecisionError"]


class Precision(enum.Enum):
    """What a checkpoint is written for."""

    #: bf16. For the 314 adapters nothing resumes or forks from.
    ARCHIVE = "archive"
    #: fp32. For the resume checkpoint and for the step every fork is taken from.
    EXACT = "exact"


class AdapterPrecisionError(RuntimeError):
    """A checkpoint was written at a precision its role does not permit."""


#: steps whose checkpoint may never be written at ARCHIVE precision, because
#: something resumes or forks from them. The design's fork step is 130.
def _forbidden(step: int | None, fork_steps: tuple[int, ...]) -> bool:
    return step is not None and step in fork_steps


def _to(tensor, dtype):
    return tensor.detach().to(dtype).contiguous()


def adapter_dtypes(state: Mapping[str, Any]) -> dict[str, str]:
    """`{name: dtype}` as strings, so a caller can assert on dtype rather than
    on file size. Asserting on file size is how a conversion that never happened
    passes: a compressed fp32 tensor of a low-rank update is not twice a bf16
    one."""
    return {k: str(v.dtype) for k, v in state.items()}


def write_adapter(
    state: Mapping[str, Any],
    path: str | Path,
    *,
    precision: Precision = Precision.ARCHIVE,
    step: int | None = None,
    is_resume_checkpoint: bool = False,
    fork_steps: tuple[int, ...] = (130,),
) -> Path:
    """Write one adapter state dict, converting only where it is permitted.

    Raises `AdapterPrecisionError` rather than silently writing fp32 when a
    caller asks for ARCHIVE on a checkpoint something forks or resumes from. A
    silent upgrade would put the storage back and a silent downgrade would move
    the low bits of a forked trajectory; refusing is the only branch that is
    neither.
    """
    import torch
    from safetensors.torch import save_file

    if precision is Precision.ARCHIVE and (is_resume_checkpoint or _forbidden(step, fork_steps)):
        raise AdapterPrecisionError(
            f"step={step} is_resume_checkpoint={is_resume_checkpoint}: this "
            f"checkpoint is resumed or forked from, so it is written at "
            f"Precision.EXACT. A bf16 round trip here means the resumed or "
            f"forked trajectory continues from different low bits than the "
            f"parent, which is a correctness question and not a storage one. "
            f"Pass precision=Precision.EXACT."
        )

    dtype = torch.bfloat16 if precision is Precision.ARCHIVE else torch.float32
    out = {k: _to(v, dtype) for k, v in state.items()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        out,
        str(p),
        metadata={
            "reward_lens.precision": precision.value,
            "reward_lens.step": "" if step is None else str(step),
            "reward_lens.is_resume_checkpoint": str(bool(is_resume_checkpoint)),
            "reward_lens.dtypes": json.dumps(adapter_dtypes(out)),
        },
    )
    return p


def read_adapter(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    """`(state, metadata)`. The metadata carries the precision it was written at,
    so a reader can assert on the recorded intent as well as on the dtype it got."""
    from safetensors import safe_open

    state, meta = {}, {}
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state, meta
