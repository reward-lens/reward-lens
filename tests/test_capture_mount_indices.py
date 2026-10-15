"""BLK-005 guard: a capture never substitutes an index or mutates the caller's indices."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from reward_lens.core.types import Site
from reward_lens.runtime.backend import CaptureSpec
from reward_lens.runtime.hf import resolve_capture_positions
from reward_lens.runtime.hooks import CaptureMount
from reward_lens.signals.base import PositionSpec


def _mount(positions: torch.Tensor) -> CaptureMount:
    # Exercise the actual gather hooks without loading a model or changing their inputs.
    mount = CaptureMount.__new__(CaptureMount)
    mount.positions = positions
    mount.full_sequence = False
    mount.dtype = None
    mount.tensors = {}
    mount.site_map = SimpleNamespace(n_heads=2)
    return mount


@pytest.mark.parametrize("index", [-1, 3])
def test_residual_capture_refuses_invalid_index_without_mutating_caller(index: int) -> None:
    positions = torch.tensor([index], dtype=torch.long)
    before = positions.clone()
    mount = _mount(positions)
    hidden = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    with pytest.raises(ValueError, match="outside.*sequence"):
        mount._store(Site(0, "resid_post"), hidden)

    assert torch.equal(positions, before)
    assert mount.tensors == {}


def test_head_capture_refuses_invalid_index_without_mutating_caller() -> None:
    positions = torch.tensor([7], dtype=torch.long)
    before = positions.clone()
    mount = _mount(positions)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    with pytest.raises(ValueError, match="outside.*sequence"):
        mount._make_head_hook(0, [0])(None, (hidden,))

    assert torch.equal(positions, before)
    assert mount.tensors == {}


def test_valid_residual_and_head_capture_still_gather_requested_row() -> None:
    positions = torch.tensor([0, 2], dtype=torch.long)
    mount = _mount(positions)
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    residual = Site(0, "resid_post")
    head = Site(0, "head_out", 1)

    mount._store(residual, hidden)
    mount._make_head_hook(0, [1])(None, (hidden,))

    assert torch.equal(mount.tensors[residual], hidden[torch.arange(2), positions])
    expected_head = hidden.reshape(2, 3, 2, 2)[torch.arange(2), positions, 1, :]
    assert torch.equal(mount.tensors[head], expected_head)
    assert positions.tolist() == [0, 2]


def test_existing_collated_per_row_and_negative_index_contract_is_preserved() -> None:
    final = torch.tensor([7, 5, 9], dtype=torch.long)
    sites = (Site(0, "resid_post"),)
    per_row = CaptureSpec(sites=sites, position=PositionSpec("explicit", [2, 3, 4]))
    negative = CaptureSpec(sites=sites, position=PositionSpec("explicit", -1))

    assert resolve_capture_positions(per_row, final).tolist() == [2, 3, 4]
    assert resolve_capture_positions(negative, final).tolist() == [7, 5, 9]
    with pytest.raises(ValueError, match="outside that row's valid span"):
        resolve_capture_positions(
            CaptureSpec(sites=sites, position=PositionSpec("explicit", [1, 6, 1])), final
        )
