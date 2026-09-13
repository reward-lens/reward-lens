"""BLK-025: the fp32 adapter upcast stands on one GPU, doubling storage.

The closure proof the row states: "a saved adapter reloads to bf16 and the
resume checkpoint reloads to fp32, both asserted on dtype rather than on file
size." Both halves are here, and the assertions are on dtype for the reason the
row gives: a compressed fp32 tensor of a low-rank update is not twice a bf16
one, so a file-size assertion passes on a conversion that never happened.

**PEFT is not installed in this environment**, so these fixtures are plain
tensors in the shape a LoRA state dict has rather than a real `PeftModel`. That
is a real limit and it is stated: what is proved here is that the post-save step
converts, refuses and round-trips. What is not proved here is the integration
with `get_peft_model`, which needs `pip install peft` and a GPU-free but
PEFT-bearing environment.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reward_lens.dynamics.adapter_io import (  # noqa: E402
    AdapterPrecisionError,
    Precision,
    adapter_dtypes,
    read_adapter,
    write_adapter,
)


def lora_state(rank: int = 8, dim: int = 64):
    """A LoRA state dict's shape, in fp32, which is what PEFT hands back."""
    g = torch.Generator().manual_seed(0)
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
            rank, dim, generator=g, dtype=torch.float32
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
            dim, rank, generator=g, dtype=torch.float32
        ),
    }


def test_the_fixture_starts_in_fp32_which_is_the_defect():
    assert set(adapter_dtypes(lora_state()).values()) == {"torch.float32"}


def test_an_archived_adapter_reloads_to_bf16(tmp_path):
    p = write_adapter(
        lora_state(), tmp_path / "a.safetensors", precision=Precision.ARCHIVE, step=40
    )
    state, meta = read_adapter(p)
    assert set(adapter_dtypes(state).values()) == {"torch.bfloat16"}
    assert meta["reward_lens.precision"] == "archive"


def test_the_resume_checkpoint_reloads_to_fp32(tmp_path):
    p = write_adapter(
        lora_state(),
        tmp_path / "r.safetensors",
        precision=Precision.EXACT,
        is_resume_checkpoint=True,
    )
    state, meta = read_adapter(p)
    assert set(adapter_dtypes(state).values()) == {"torch.float32"}
    assert meta["reward_lens.precision"] == "exact"
    assert meta["reward_lens.is_resume_checkpoint"] == "True"


def test_step_130_reloads_to_fp32(tmp_path):
    p = write_adapter(lora_state(), tmp_path / "f.safetensors", precision=Precision.EXACT, step=130)
    state, _ = read_adapter(p)
    assert set(adapter_dtypes(state).values()) == {"torch.float32"}


def test_archiving_the_fork_step_refuses_rather_than_writing_it(tmp_path):
    """The branch that would silently move a forked trajectory's low bits."""
    with pytest.raises(AdapterPrecisionError) as e:
        write_adapter(
            lora_state(), tmp_path / "x.safetensors", precision=Precision.ARCHIVE, step=130
        )
    assert "forked" in str(e.value)


def test_archiving_the_resume_checkpoint_refuses(tmp_path):
    with pytest.raises(AdapterPrecisionError):
        write_adapter(
            lora_state(),
            tmp_path / "y.safetensors",
            precision=Precision.ARCHIVE,
            is_resume_checkpoint=True,
        )


def test_the_exact_checkpoint_round_trips_bit_for_bit(tmp_path):
    """fp32 in, fp32 out, `torch.equal` and not `allclose`, because a resume
    that differs in the low bits is not the parent's trajectory."""
    s = lora_state()
    p = write_adapter(
        s, tmp_path / "e.safetensors", precision=Precision.EXACT, is_resume_checkpoint=True
    )
    back, _ = read_adapter(p)
    for k, v in s.items():
        assert torch.equal(back[k], v), k


def test_the_archive_round_trip_is_lossy_and_the_test_says_so(tmp_path):
    """Not a defect. It is what bf16 costs, and a test that pretended otherwise
    would be asserting the conversion did not happen."""
    s = lora_state()
    p = write_adapter(s, tmp_path / "l.safetensors", precision=Precision.ARCHIVE, step=40)
    back, _ = read_adapter(p)
    k = next(iter(s))
    assert not torch.equal(back[k].float(), s[k])
    assert torch.allclose(back[k].float(), s[k], atol=1e-2)


def test_dtype_and_not_file_size_is_what_discriminates(tmp_path):
    """The row's own instruction, made explicit: file size does not separate
    these two the way dtype does."""
    s = lora_state()
    a = write_adapter(s, tmp_path / "aa.safetensors", precision=Precision.ARCHIVE, step=40)
    e = write_adapter(s, tmp_path / "ee.safetensors", precision=Precision.EXACT, step=40)
    da = set(adapter_dtypes(read_adapter(a)[0]).values())
    de = set(adapter_dtypes(read_adapter(e)[0]).values())
    assert da == {"torch.bfloat16"} and de == {"torch.float32"}
