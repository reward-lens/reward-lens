"""Shared fixtures for the execution tests.

Every test here runs real processes. Nothing reaches the network: the one socket test connects to
a listener this process owns, and the point of it is that the connection is refused by the kernel
before it leaves the machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reward_lens.execution import Limits


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A fresh mode-0700 working directory, which is what T0 promises a grader."""
    d = tmp_path / "work"
    d.mkdir(mode=0o700)
    return d


@pytest.fixture
def small() -> Limits:
    """Limits small enough that a test can exceed each of them in a second or two."""
    return Limits(
        wall_s=6.0,
        cpu_s=6.0,
        memory_bytes=512 * 2**20,
        processes=8,
        network=False,
        stdout_bytes=1 << 16,
        artifact_bytes=1 << 18,
    )


def secret_parent_env() -> dict[str, str]:
    """The caller's environment with four keys in it that must never reach a grader."""
    env = dict(os.environ)
    env.update(
        {
            "OPENAI_API_KEY": "sk-parent-openai",
            "ANTHROPIC_API_KEY": "sk-parent-anthropic",
            "HF_TOKEN": "hf-parent",
            "AWS_SECRET_ACCESS_KEY": "aws-parent",
        }
    )
    return env
