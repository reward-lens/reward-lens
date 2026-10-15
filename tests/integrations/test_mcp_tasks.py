"""A declared MCP Tasks client can poll real SDK work and cancel an audit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_task_lifecycle_over_real_stdio(tmp_path: Path) -> None:
    trace_run = tmp_path / "trace-run"
    shutil.copytree(ROOT / "tests/fixtures/grpo_run/short", trace_run)
    script = r"""
import asyncio
import copy
import json
import pathlib
import sys

from reward_lens.integrations import mcp as _reward_lens_mcp
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

EXTENSION = 'io.modelcontextprotocol/tasks'

async def main():
    async with Client(
        StdioServerParameters(
            command=sys.executable, args=['-u', '-m', 'reward_lens.integrations.mcp'],
            cwd='.', env={'PYTHONDONTWRITEBYTECODE': '1',
                          'XDG_STATE_HOME': sys.argv[1]},
        ), mode='auto', read_timeout_seconds=20,
    ) as client:
        discovered = client.session.discover_result
        assert EXTENSION in discovered.capabilities.extensions

        async def raw(method, params, *, declare=True):
            request = {'method': method, 'params': params}
            options = {}
            client.session._stamp(request, options)
            request['params']['_meta'] = copy.deepcopy(request['params'].get('_meta', {}))
            meta = request['params']['_meta']
            caps = meta.setdefault('io.modelcontextprotocol/clientCapabilities', {})
            caps = copy.deepcopy(caps)
            meta['io.modelcontextprotocol/clientCapabilities'] = caps
            if declare:
                caps.setdefault('extensions', {})[EXTENSION] = {}
            else:
                caps.get('extensions', {}).pop(EXTENSION, None)
            return await client.session._dispatcher.send_raw_request(
                request['method'], request['params'], options
            )

        started = await raw('tools/call', {
            'name': 'trace', 'arguments': {'request': {'run': sys.argv[2]}}
        })
        assert started['resultType'] == 'task', started
        task_id = started['taskId']
        assert task_id.startswith('rl-') and started['status'] == 'working'

        try:
            await raw('tasks/get', {'taskId': task_id}, declare=False)
        except MCPError as error:
            assert error.code == -32021
        else:
            raise AssertionError('task polling without capability was accepted')

        ack = await raw('tasks/update', {'taskId': task_id, 'inputResponses': {}})
        assert ack['resultType'] == 'complete'
        for _ in range(100):
            state = await raw('tasks/get', {'taskId': task_id})
            if state['status'] != 'working':
                break
            await asyncio.sleep(0.05)
        assert state['resultType'] == 'complete'
        assert state['status'] == 'completed', state
        result = state['result']
        assert result['structuredContent']['resultType'] == 'assay'
        assert pathlib.Path(result['structuredContent']['path']).is_file()

        try:
            await raw('tasks/get', {'taskId': 'not-a-task'})
        except MCPError as error:
            assert error.code == -32602
        else:
            raise AssertionError('unknown task was accepted')

        slow = pathlib.Path(sys.argv[1]) / 'slow_grader.py'
        slow.parent.mkdir(parents=True, exist_ok=True)
        slow.write_text('import time\ndef score(task, response):\n    time.sleep(60)\n    return 1.0\n')
        tasks = slow.parent / 'tasks.jsonl'
        responses = slow.parent / 'responses.jsonl'
        tasks.write_text(''.join(json.dumps({'task_id': f't-{i}', 'prompt': 'p'}) + '\n' for i in range(60)))
        responses.write_text(''.join(json.dumps({'response_id': f'r-{i}', 'task_id': f't-{i//2}',
                                                  'response': 'x'}) + '\n' for i in range(120)))
        pending = await raw('tools/call', {'name': 'audit', 'arguments': {
            'path': str(slow), 'tasks': str(tasks), 'responses': str(responses)
        }})
        assert pending['resultType'] == 'task' and pending['status'] == 'working'
        cancel_id = pending['taskId']
        cancelled = await raw('tasks/cancel', {'taskId': cancel_id})
        assert cancelled['resultType'] == 'complete'
        for _ in range(120):
            state = await raw('tasks/get', {'taskId': cancel_id})
            if state['status'] != 'working':
                break
            await asyncio.sleep(0.05)
        assert state['status'] == 'cancelled', state
        print(json.dumps({'trace': task_id, 'audit': cancel_id,
                          'trace_status': 'completed', 'audit_status': 'cancelled'}))

asyncio.run(main())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            script,
            str(tmp_path / "state"),
            str(trace_run),
        ],
        cwd=ROOT,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert completed.returncode == 0, completed.stderr
