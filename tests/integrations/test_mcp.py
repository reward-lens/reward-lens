"""The stateless MCP boundary: SDK transport, typed calls, and refusals."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from reward_lens.integrations.mcp import build_server, dispatch


def test_server_is_the_v2_sdk_server_and_exposes_discovery() -> None:
    server = build_server()
    assert type(server).__name__ == "MCPServer"
    assert dispatch(server, "server/discover", {}).supported_versions == ["2026-07-28"]


def test_sdk_stdio_subprocess_lists_tools_after_discovery(tmp_path: Path) -> None:
    """Exercise discovery, typed calls, task polling, and cancellation over real stdio."""
    fixture = tmp_path / "plain_grader.py"
    profile = tmp_path / "stdio-profile.txt"
    fixture.write_text(
        'def score(task, response):\n    return float(response == task.get("expected"))\n',
        encoding="utf-8",
    )
    script = f"""\
import asyncio
import json
import os
import sys
import time

from reward_lens.integrations import mcp as _reward_lens_mcp
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError


def profile(event):
    with open({str(profile)!r}, 'a', encoding='utf-8') as handle:
        handle.write(f'{{event}} {{time.monotonic():.9f}}\\n')


profile('client_process_start')


async def main():
    async with Client(
        StdioServerParameters(
            command=sys.executable,
            args=['-u', '-m', 'reward_lens.integrations.mcp'],
            cwd='.',
            env={{'PYTHONDONTWRITEBYTECODE': '1', 'XDG_STATE_HOME': {str(tmp_path / "state")!r}, 'REWARD_LENS_MCP_PROFILE_PATH': {str(profile)!r}}},
        ),
        mode='auto',
        read_timeout_seconds=10,
    ) as client:
        profile('client_ready')
        discovered = client.session.discover_result
        assert discovered is not None
        assert discovered.supported_versions == ['2026-07-28']
        assert discovered.ttl_ms == 0
        assert discovered.cache_scope == 'private'
        listed = await client.list_tools()
        assert {{tool.name for tool in listed.tools}} >= {{'audit', 'doctor', 'seeker.propose'}}
        assert listed.ttl_ms == 0
        assert listed.cache_scope == 'private'

        for method in ('initialize', 'ping'):
            request = {{'method': method, 'params': {{}}}}
            options = {{}}
            client.session._stamp(request, options)
            try:
                await client.session._dispatcher.send_raw_request(
                    request['method'], request['params'], options
                )
            except MCPError as error:
                assert error.code in {{-32601, -32022}}
            else:
                raise AssertionError(f'{{method}} was accepted on the real stdio wire')

        missing = await client.session.call_tool('audit', {{}}, allow_input_required=True)
        assert missing.result_type == 'input_required'
        audited = await client.call_tool('audit', {{'path': {str(fixture)!r}}})
        assert audited.structured_content['resultType'] == 'assay'

        grader = {str(tmp_path / "response_grader.py")!r}
        tasks = {str(tmp_path / "tasks.jsonl")!r}
        responses = {str(tmp_path / "responses.jsonl")!r}
        open(grader, 'w', encoding='utf-8').write(
            'def score(task, response):\\n    return float(response == task["expected"])\\n'
        )
        with open(tasks, 'w', encoding='utf-8') as handle:
            for index in range(60):
                handle.write(json.dumps({{'task_id': f'task-{{index}}', 'prompt': 'p', 'expected': 'yes'}}) + '\\n')
        with open(responses, 'w', encoding='utf-8') as handle:
            for index in range(120):
                handle.write(json.dumps({{'response_id': f'row-{{index}}', 'task_id': f'task-{{index // 2}}', 'response': 'yes'}}) + '\\n')
        profile('bank_request_start')
        try:
            response_bank = await client.call_tool(
                'audit', {{'path': grader, 'tasks': tasks, 'responses': responses}}
            )
        except Exception:
            profile('bank_request_failed')
            raise
        profile('bank_request_end')
        signal = next(
            entry for entry in response_bank.structured_content['result']['measurement']['signal']
            if entry['entry_id'] == 'signal.response_score_contrast_fraction'
        )
        assert signal['result']['scored_response_count'] == 120

        def descendants(parent):
            pending = [parent]
            found = set()
            while pending:
                current = pending.pop()
                try:
                    task_ids = os.listdir(f'/proc/{{current}}/task')
                except OSError:
                    continue
                for task_id in task_ids:
                    try:
                        children = open(
                            f'/proc/{{current}}/task/{{task_id}}/children', 'r', encoding='utf-8'
                        ).read().split()
                    except OSError:
                        continue
                    for child in children:
                        if child not in found:
                            found.add(child)
                            pending.append(child)
            return found

        def is_server(pid):
            try:
                argv = open(f'/proc/{{pid}}/cmdline', 'rb').read().split(b'\\0')
            except OSError:
                return False
            return b'-m' in argv and b'reward_lens.integrations.mcp' in argv

        def parent_pid(pid):
            try:
                return open(f'/proc/{{pid}}/stat', 'r', encoding='utf-8').read().split()[3]
            except OSError:
                return None

        server = next(
            pid for pid in os.listdir('/proc')
            if pid.isdigit() and parent_pid(pid) == str(os.getpid()) and is_server(pid)
        )
        server_group = os.getpgid(int(server))
        prior_descendants = descendants(server)

        def contained_groups():
            groups = set()
            for pid in descendants(server) - prior_descendants:
                try:
                    group = os.getpgid(int(pid))
                except ProcessLookupError:
                    continue
                if group != server_group:
                    groups.add(group)
            return groups

        def group_exists(group):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        slow = {str(tmp_path / "slow_response_grader.py")!r}
        open(slow, 'w', encoding='utf-8').write(
            'import time\\ndef score(task, response):\\n    time.sleep(60)\\n    return 1.0\\n'
        )
        prior_requests = set(client.session._dispatcher._pending)
        progress = []
        async def on_progress(done, total, message):
            progress.append((done, total, message))
        pending = asyncio.create_task(
            client.call_tool('audit', {{'path': slow, 'tasks': tasks, 'responses': responses}},
                             progress_callback=on_progress)
        )
        await asyncio.sleep(0)
        request_ids = list(set(client.session._dispatcher._pending) - prior_requests)
        assert len(request_ids) == 1
        # This observes the next audit's contained work; it does not change the
        # client's existing 10-second response deadline for any request.
        deadline = time.monotonic() + 8
        active_groups = set()
        while time.monotonic() < deadline and not active_groups:
            await asyncio.sleep(0.05)
            active_groups = contained_groups()
        current_descendants = descendants(server)
        assert 0 < len(active_groups) <= 24, (
            f"server={{server}} group={{server_group}} prior={{sorted(prior_descendants)}} "
            f"current={{sorted((pid, os.getpgid(int(pid))) for pid in current_descendants)}}"
        )  # twelve workers, each with at most its L3 wrapper and child
        progress_deadline = time.monotonic() + 3
        while time.monotonic() < progress_deadline and not any(
            message == 'audit running' for _, _, message in progress
        ):
            await asyncio.sleep(0.05)
        assert any(message == 'audit started' for _, _, message in progress)
        assert any(message == 'audit running' for _, _, message in progress)
        await client.session._dispatcher.notify(
            'notifications/cancelled',
            {{'requestId': request_ids[0], 'reason': 'transport cancellation regression'}},
        )
        reap_deadline = time.monotonic() + 4
        while time.monotonic() < reap_deadline and any(
            group in active_groups for group in contained_groups()
        ):
            await asyncio.sleep(0.05)
        assert not any(group in active_groups for group in contained_groups())
        assert not any(group_exists(group) for group in active_groups)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('the real response-bank request did not cancel')
        assert (await client.call_tool('doctor')).structured_content['resultType'] == 'capabilities'

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    events = [line.split()[0] for line in profile.read_text(encoding="utf-8").splitlines()]
    assert events[0] == "client_process_start"
    assert "client_ready" in events
    assert "stdio_outbound_frame" in events
    assert "stdio_outbound_drained" in events
    assert "audit_tool_response_ready" in events
    assert "bank_request_start" in events
    assert "bank_request_end" in events
    assert "response_score_bank_start" in events
    assert "response_score_bank_end" in events
    assert "stdio_inbound_cancel" in events
    assert events.index("audit_tool_response_ready") < events.index("stdio_inbound_cancel")


def test_streamable_http_loopback_enforces_native_guards() -> None:
    """Exercise the SDK's actual Streamable HTTP ASGI route over loopback TCP."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    root = Path(__file__).parents[2]
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "from reward_lens.integrations.mcp import run; "
            f"run(transport='streamable-http', host='127.0.0.1', port={port})",
        ],
        cwd=root,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise AssertionError("the Streamable HTTP server did not start")
                time.sleep(0.05)

        meta = {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientInfo": {"name": "integration-test", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }

        def request(
            method: str,
            *,
            params: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
        ) -> httpx.Response:
            return httpx.post(
                f"http://127.0.0.1:{port}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": {"_meta": meta, **(params or {})},
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Protocol-Version": "2026-07-28",
                    "Mcp-Method": method,
                    **(headers or {}),
                },
                timeout=5,
            )

        discovered = request("server/discover")
        assert discovered.status_code == 200
        assert discovered.json()["result"]["ttlMs"] == 0
        assert discovered.json()["result"]["cacheScope"] == "private"

        listed = request("tools/list")
        assert listed.status_code == 200
        assert listed.json()["result"]["ttlMs"] == 0
        assert listed.json()["result"]["cacheScope"] == "private"

        accepted = request(
            "tools/call",
            params={"name": "audit", "arguments": {}},
            headers={"Mcp-Name": "audit"},
        )
        assert accepted.status_code == 200
        assert accepted.json()["result"]["resultType"] == "input_required"

        missing_name = request("tools/call", params={"name": "audit", "arguments": {}})
        assert missing_name.status_code == 400
        assert missing_name.json()["error"]["code"] == -32020

        wrong_method = request("tools/list", headers={"Mcp-Method": "ping"})
        assert wrong_method.status_code == 400
        assert wrong_method.json()["error"]["code"] == -32020

        for method in ("initialize", "ping"):
            refused = request(method)
            assert refused.status_code == 404
            assert "not supported by stateless MCP" in refused.json()["error"]["message"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_streamable_http_response_bank_cancellation_reaps_current_server_children(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cancel an actual HTTP audit POST and retain only this server's L3 census."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    root = Path(__file__).parents[2]
    profile_path = tmp_path / "mcp-profile.log"
    server_stderr_path = tmp_path / "mcp-server.stderr.log"
    server_stderr = server_stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "from reward_lens.integrations.mcp import run; "
            f"run(transport='streamable-http', host='127.0.0.1', port={port})",
        ],
        cwd=root,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "REWARD_LENS_MCP_PROFILE_PATH": str(profile_path),
        },
        stdout=subprocess.DEVNULL,
        stderr=server_stderr,
    )
    completed: subprocess.CompletedProcess[str] | None = None
    ready_started = time.monotonic()
    ready_seconds: float | None = None
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise AssertionError("the Streamable HTTP server did not start")
                time.sleep(0.05)
        ready_seconds = time.monotonic() - ready_started

        script = f"""\
import json
import os
import sys
import threading
import time

import httpx

port = {port}
root = {str(root)!r}
tmp = {str(tmp_path)!r}
grader = os.path.join(tmp, 'response_grader.py')
slow = os.path.join(tmp, 'slow_response_grader.py')
tasks = os.path.join(tmp, 'tasks.jsonl')
responses = os.path.join(tmp, 'responses.jsonl')
open(grader, 'w', encoding='utf-8').write(
    'def score(task, response):\\n    return float(response == task["expected"])\\n'
)
open(slow, 'w', encoding='utf-8').write(
    'import time\\ndef score(task, response):\\n    time.sleep(60)\\n    return 1.0\\n'
)
with open(tasks, 'w', encoding='utf-8') as handle:
    for index in range(60):
        handle.write(json.dumps({{'task_id': f'task-{{index}}', 'prompt': 'p', 'expected': 'yes'}}) + '\\n')
with open(responses, 'w', encoding='utf-8') as handle:
    for index in range(120):
        handle.write(json.dumps({{'response_id': f'row-{{index}}', 'task_id': f'task-{{index // 2}}', 'response': 'yes'}}) + '\\n')

meta = {{
    'io.modelcontextprotocol/protocolVersion': '2026-07-28',
    'io.modelcontextprotocol/clientInfo': {{'name': 'integration-test', 'version': '1'}},
    'io.modelcontextprotocol/clientCapabilities': {{}},
}}
url = f'http://127.0.0.1:{{port}}/mcp'

try:
    affinity = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    affinity = 'unavailable'
try:
    with open('/sys/fs/cgroup/cpu.max', encoding='utf-8') as quota_file:
        quota = quota_file.read(80).strip()
except OSError:
    quota = 'unavailable'
print('RL_MCP_DIAG affinity=%s cpu_count=%s cpu_max=%s' %
      (affinity, os.cpu_count(), quota), file=sys.stderr, flush=True)

def call(method, params, request_id, *, headers=None, timeout=10):
    return httpx.post(
        url,
        json={{'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': {{'_meta': meta, **params}}}},
        headers={{
            'Accept': 'application/json, text/event-stream',
            'Mcp-Protocol-Version': '2026-07-28',
            'Mcp-Method': method,
            **(headers or {{}}),
        }},
        timeout=timeout,
    )

def descendants(parent):
    pending, found = [parent], set()
    while pending:
        current = pending.pop()
        try:
            task_ids = os.listdir(f'/proc/{{current}}/task')
        except OSError:
            continue
        for task_id in task_ids:
            try:
                children = open(f'/proc/{{current}}/task/{{task_id}}/children', encoding='utf-8').read().split()
            except OSError:
                continue
            for child in children:
                if child not in found:
                    found.add(child)
                    pending.append(child)
    return found

server = {process.pid}
server_group = os.getpgid(server)
prior_descendants = descendants(server)

def contained_groups():
    groups = set()
    for pid in descendants(server) - prior_descendants:
        try:
            group = os.getpgid(int(pid))
        except ProcessLookupError:
            continue
        if group != server_group:
            groups.add(group)
    return groups

def group_exists(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

complete_started = time.monotonic()
try:
    complete = call(
        'tools/call',
        {{'name': 'audit', 'arguments': {{'path': grader, 'tasks': tasks, 'responses': responses}}}},
        'complete-bank',
        headers={{'Mcp-Name': 'audit'}},
    )
finally:
    print('RL_MCP_DIAG complete_bank_elapsed=%.3f' %
          (time.monotonic() - complete_started), file=sys.stderr, flush=True)
assert complete.status_code == 200, complete.text
assert 'result' in complete.json(), complete.text
assert 'structuredContent' in complete.json()['result'], complete.text
signal = next(
    entry for entry in complete.json()['result']['structuredContent']['result']['measurement']['signal']
    if entry['entry_id'] == 'signal.response_score_contrast_fraction'
)
assert signal['result']['scored_response_count'] == 120

answer = {{}}
def start_slow_audit():
    answer['response'] = call(
        'tools/call',
        {{'name': 'audit', 'arguments': {{'path': slow, 'tasks': tasks, 'responses': responses}}}},
        'active-audit',
        headers={{'Mcp-Name': 'audit'}},
    )

pending = threading.Thread(target=start_slow_audit)
pending.start()
active_started = time.monotonic()
deadline = time.monotonic() + 8
active_groups = set()
while time.monotonic() < deadline and not active_groups:
    time.sleep(0.05)
    active_groups = contained_groups()
print('RL_MCP_DIAG active_groups_elapsed=%.3f groups=%d' %
      (time.monotonic() - active_started, len(active_groups)), file=sys.stderr, flush=True)
assert 0 < len(active_groups) <= 24, (server, sorted(prior_descendants), sorted(descendants(server)))

cancel_started = time.monotonic()
cancelled = httpx.post(
    url,
    json={{
        'jsonrpc': '2.0',
        'method': 'notifications/cancelled',
        'params': {{'requestId': 'active-audit', 'reason': 'transport cancellation regression'}},
    }},
    headers={{
        'Accept': 'application/json, text/event-stream',
        'Mcp-Protocol-Version': '2026-07-28',
        'Mcp-Method': 'notifications/cancelled',
    }},
    timeout=10,
)
print('RL_MCP_DIAG cancel_elapsed=%.3f status=%d' %
      (time.monotonic() - cancel_started, cancelled.status_code), file=sys.stderr, flush=True)
assert cancelled.status_code in {{200, 202, 204}}, cancelled.text
reap_started = time.monotonic()
reap_deadline = time.monotonic() + 4
while time.monotonic() < reap_deadline and any(group in active_groups for group in contained_groups()):
    time.sleep(0.05)
print('RL_MCP_DIAG reap_elapsed=%.3f' %
      (time.monotonic() - reap_started), file=sys.stderr, flush=True)
assert not any(group in active_groups for group in contained_groups()), (active_groups, contained_groups())
assert not any(group_exists(group) for group in active_groups), active_groups
pending.join(timeout=5)
assert not pending.is_alive()
assert 'response' in answer
doctor = call(
    'tools/call', {{'name': 'doctor', 'arguments': {{}}}}, 'after-cancel', headers={{'Mcp-Name': 'doctor'}}
)
print('RL_MCP_DIAG doctor_status=%d' % doctor.status_code, file=sys.stderr, flush=True)
assert doctor.status_code == 200, doctor.text
"""
        completed = subprocess.run(
            [sys.executable, "-u", "-c", script],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=35,
            env={
                **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=5)
        try:
            server_stderr.close()
        except OSError:
            pass
        try:
            try:
                profile_lines = profile_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                profile_lines = []
            marks: dict[str, list[float]] = {}
            for line in profile_lines[:20]:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        marks.setdefault(parts[0], []).append(float(parts[1]))
                    except ValueError:
                        pass
            bank_starts = marks.get("response_score_bank_start", [])
            bank_ends = marks.get("response_score_bank_end", [])
            bank_elapsed = (
                f"{bank_ends[0] - bank_starts[0]:.3f}s"
                if bank_starts and bank_ends
                else "started-but-unfinished"
                if bank_starts
                else "not-started"
            )
            ready_text = f"{ready_seconds:.3f}s" if ready_seconds is not None else "unavailable"
            diagnostic_lines = [
                f"RL_MCP_DIAG server_ready={ready_text} server_pid={process.pid} "
                f"child_exit={completed.returncode if completed is not None else 'unavailable'} "
                f"first_bank={bank_elapsed}"
            ]
            if completed is not None:
                diagnostic_lines.extend(
                    line[:240]
                    for line in completed.stderr.splitlines()
                    if line.startswith("RL_MCP_DIAG ")
                )
                if completed.returncode != 0:
                    diagnostic_lines.extend(
                        "RL_MCP_DIAG child_stderr_tail: " + line[:240]
                        for line in completed.stderr.splitlines()[-8:]
                    )
            if completed is None or completed.returncode != 0:
                try:
                    with server_stderr_path.open("rb") as server_log:
                        server_log.seek(0, 2)
                        server_log.seek(max(0, server_log.tell() - 4096))
                        server_tail = (
                            server_log.read().decode("utf-8", errors="replace").splitlines()
                        )
                    diagnostic_lines.extend(
                        "RL_MCP_DIAG server_stderr_tail: " + line[:240] for line in server_tail[-8:]
                    )
                except OSError:
                    diagnostic_lines.append("RL_MCP_DIAG server_stderr_tail=unavailable")
        except Exception as diagnostic_error:
            diagnostic_lines = [
                f"RL_MCP_DIAG summary_unavailable={type(diagnostic_error).__name__}"
            ]
        try:
            with capsys.disabled():
                print("\n".join(diagnostic_lines), flush=True)
        except Exception:
            pass


def test_missing_tool_input_round_trips_as_input_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = dispatch(None, "tools/call", {"name": "audit", "arguments": {}})
    assert first.result_type == "input_required"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    fixture = tmp_path / "project"
    shutil.copytree(Path(__file__).parents[2] / "src/reward_lens/examples/code_reward", fixture)
    second = dispatch(
        None,
        "tools/call",
        {"name": "audit", "arguments": {"path": str(fixture), "inputResponses": {}}},
    )
    assert second.result_type == "complete"


def test_doctor_returns_capabilities_without_a_fake_task() -> None:
    result = dispatch(None, "tools/call", {"name": "doctor", "arguments": {}})
    assert result.structured_content["resultType"] == "capabilities"


def test_paid_seeker_needs_explicit_spend_argument() -> None:
    result = dispatch(
        None, "tools/call", {"name": "seeker.propose", "arguments": {"session": "missing"}}
    )
    assert result.structured_content["resultType"] == "refused"
