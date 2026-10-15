"""The public MCP subprocess exposes the complete SDK surface over stdio."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
TOOLS = {
    "audit",
    "trace",
    "compare",
    "forecast",
    "improve",
    "open",
    "explain",
    "doctor",
    "describe",
    "seeker.propose",
}


def test_full_inventory_and_typed_results_over_stdio(tmp_path: Path) -> None:
    script = r"""
import asyncio
import json
import sys

from reward_lens.integrations import mcp as _reward_lens_mcp
from mcp import Client
from mcp.client.stdio import StdioServerParameters

async def main():
    async with Client(
        StdioServerParameters(
            command=sys.executable,
            args=['-u', '-m', 'reward_lens.integrations.mcp'],
            cwd='.',
            env={'PYTHONDONTWRITEBYTECODE': '1',
                 'XDG_STATE_HOME': sys.argv[1]},
        ), mode='auto', read_timeout_seconds=20,
    ) as client:
        listed = await client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        assert set(tools) == set(json.loads(sys.argv[2])), sorted(tools)
        assert 'request' in tools['trace'].input_schema['properties']
        assert 'request' in tools['compare'].input_schema['properties']
        assert 'request' in tools['improve'].input_schema['properties']

        doctor = await client.call_tool('doctor', {})
        assert doctor.structured_content['resultType'] == 'capabilities'
        assert 'capabilities' in doctor.structured_content['result']

        described = await client.call_tool('describe', {})
        assert described.structured_content['resultType'] == 'command_tree'
        assert described.structured_content['result']['program'] == 'reward-lens'

        explained = await client.call_tool('explain', {'code': 'RL0001'})
        assert explained.structured_content['resultType'] == 'explanation'
        assert explained.structured_content['result']['code'] == 'RL0001'

        ledger = await client.call_tool('forecast', {'action': 'ledger', 'request': {}})
        assert ledger.structured_content['resultType'] == 'assay'
        assert ledger.structured_content['result']['assay_id']

        progress = []
        async def on_progress(done, total, message):
            progress.append((done, total, message))
        traced = await client.call_tool(
            'trace', {'request': {'run': sys.argv[3]}}, progress_callback=on_progress
        )
        assert traced.structured_content['resultType'] == 'assay'
        assert traced.structured_content['result']['assay_id']
        assert traced.structured_content['path'].endswith('.assay.json')
        assert __import__('pathlib').Path(traced.structured_content['path']).is_file()
        assert any(message == 'trace started' for _, _, message in progress), progress
        assert any(message == 'trace finished' for _, _, message in progress), progress

        bad_compare = await client.call_tool('compare', {'request': {}})
        assert bad_compare.is_error
        assert bad_compare.structured_content['resultType'] == 'refused'
        assert bad_compare.structured_content['result']['code'] == 'RL0002'

        missing = await client.call_tool('open', {'path': sys.argv[1] + '/missing.assay.json'})
        assert missing.is_error
        assert missing.structured_content['resultType'] == 'refused'
        assert missing.structured_content['result']['code'] == 'RL0621'

        invalid = await client.call_tool('explain', {'code': 'RL9999'})
        assert invalid.is_error
        assert invalid.structured_content['result']['code'] == 'RL0001'

        print(json.dumps({'tools': sorted(tools), 'typed': ['doctor', 'describe', 'explain',
              'forecast', 'trace', 'compare-error', 'open-error', 'explain-error']}))

asyncio.run(main())
"""
    trace_fixture = tmp_path / "trace-run"
    shutil.copytree(ROOT / "tests/fixtures/grpo_run/short", trace_fixture)
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            script,
            str(tmp_path / "state"),
            json.dumps(sorted(TOOLS)),
            str(trace_fixture),
        ],
        cwd=ROOT,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert completed.returncode == 0, completed.stderr
