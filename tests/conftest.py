"""Pytest configuration for reward-lens tests.

Two pieces of global hygiene. Matplotlib runs headless. And the evidence store and activation
cache are redirected to a per-session temporary directory before any reward_lens module reads the
setting, so tests never write to the developer's real ``~/.reward_lens`` and never see each other's
state through the default store. Tests that want to assert on a store should still construct an
explicit ``EvidenceStore(tmp_path)``; this only guarantees the default is harmless.
"""

import os
import sys
import tempfile

import matplotlib
import pytest

matplotlib.use("Agg")  # Non-interactive backend for tests

# Redirect the store/cache root before reward_lens.core.config resolves it. Set at import time so
# the very first `get_settings()` in any test already points at the throwaway home.
_TEST_HOME = tempfile.mkdtemp(prefix="reward_lens_test_home_")
os.environ.setdefault("REWARD_LENS_HOME", _TEST_HOME)


@pytest.fixture(autouse=True)
def _restore_torch_grad_state():
    """Restore global autograd state after every test.

    Some tests deliberately disable gradients for scoring or E-parity work with the global
    ``torch.set_grad_enabled(False)`` rather than a scoped ``with torch.no_grad()``. Under a full
    pytest run that leaks into later tests, and the HVP, Hessian, and organism-training tests then
    fail with "does not require grad". This fixture returns autograd to its default enabled state
    after each test, so test ordering cannot make a grad-requiring test fail. It touches torch only
    if a test already imported it, so pure-numpy tests stay torch-free.
    """
    yield
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_grad_enabled(True)
