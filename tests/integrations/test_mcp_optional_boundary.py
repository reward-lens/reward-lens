"""The MCP 1 OpenEnv lane cannot impersonate the optional MCP 2 server lane."""

from __future__ import annotations

import os
import subprocess
import sys


def test_openenv_only_mcp_import_names_missing_extra(tmp_path):
    """A fresh process with MCP 1 visible and mcp_types absent refuses before vendor loading."""
    script = """
import importlib.abc
import importlib.util
import sys

assert importlib.util.find_spec('mcp') is not None, 'dev must include OpenEnv MCP 1'

class HideMCPTypes(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp_types' or fullname.startswith('mcp_types.'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, HideMCPTypes())
from reward_lens.core.extras import ExtraRequiredError
try:
    import reward_lens.integrations.mcp
except ExtraRequiredError as error:
    assert "'mcp' extra" in str(error), str(error)
    assert "pip install 'reward-lens[mcp]'" in str(error), str(error)
    assert 'mcp_types' not in sys.modules
    print('typed MCP extra refusal')
else:
    raise AssertionError('OpenEnv MCP 1 was accepted as the MCP 2 server extra')
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "typed MCP extra refusal"
