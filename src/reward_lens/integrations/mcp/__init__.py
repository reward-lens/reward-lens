"""The optional, stateless 2026-07-28 MCP surface.

The SDK owns JSON-RPC, stdio, Streamable HTTP and `server/discover`.  This
module only translates tool arguments to the public SDK.  It deliberately has
no CLI fallback and no process-wide protocol session.

The pinned MCP 2.2.0 SDK has no built-in Tasks runtime.  A bounded extension
uses its generic extension hook for detached measurement calls; ordinary calls
use its native request-scoped progress channel.

Install this isolated runtime lane with ``uv sync --extra mcp``.  The command
starts the private MCP 2 child, while OpenEnv retains its public MCP 1 lane;
the ``all`` extra intentionally includes both capabilities.
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from reward_lens.core.extras import require_extra

require_extra("mcp", subsystem="reward-lens MCP server")

import anyio

from reward_lens._vendor.mcp2.runtime import materialize_private_runtime, require_private_mcp

_PRIVATE_MCP2 = materialize_private_runtime()
sys.path.insert(0, str(_PRIVATE_MCP2.site_root))

import mcp as _mcp
from mcp.server import MCPServer
from mcp.server.caching import CacheHint
from mcp.server.context import ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver.context import Context
from mcp.shared.exceptions import MCPError
from mcp_types import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    CallToolResult,
    DiscoverResult,
    InputRequiredResult,
    ListToolsResult,
    RequestParams,
    ServerCapabilities,
    TextContent,
    Tool,
    ToolsCapability,
)
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from reward_lens import api, contracts
from reward_lens.api.requests import _refusal as request_refusal
from reward_lens.instruments.seeker.actors.agent import propose_tool
from reward_lens.product.audit.response_scores import (
    ResponseScoreCancelled,
    response_score_cancellation,
)

require_private_mcp(_mcp.__file__, site_root=_PRIVATE_MCP2.site_root)

PROTOCOL_VERSION = "2026-07-28"
CACHE_SCOPE = "private"
_AUDIT_CANCELLATIONS: dict[str, threading.Event] = {}
_HTTP_CANCELLATION_BODY_LIMIT = 1_048_576
_AUDIT_THREAD_POLL_SECONDS = 0.05
_TASK_EXTENSION = "io.modelcontextprotocol/tasks"
_TASK_TTL_MS = 3_600_000
_TASK_POLL_MS = 250


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _profile(event: str) -> None:
    path = os.environ.get("REWARD_LENS_MCP_PROFILE_PATH")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"{event} {time.monotonic():.9f}\n")
        except OSError:
            pass


def cancel_audit_request(request_id: object) -> bool:
    """Signal an active audit before a transport cancels its handler scope."""
    event = _AUDIT_CANCELLATIONS.get(str(request_id))
    if event is None:
        return False
    event.set()
    return True


def _http_cancellation_bridge(app: Any) -> Any:
    """Observe bounded cancellation POSTs before the public SDK ASGI app."""

    async def bridge(scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/mcp"
        ):
            await app(scope, receive, send)
            return
        buffered: list[dict[str, Any]] = []
        body: list[bytes] = []
        size = 0
        complete = False
        while size <= _HTTP_CANCELLATION_BODY_LIMIT:
            message = await receive()
            buffered.append(message)
            if message.get("type") != "http.request":
                break
            chunk = message.get("body", b"")
            size += len(chunk)
            body.append(chunk)
            if not message.get("more_body", False):
                complete = True
                break
        if complete and size <= _HTTP_CANCELLATION_BODY_LIMIT:
            try:
                decoded = json.loads(b"".join(body))
            except (TypeError, ValueError, UnicodeDecodeError):
                decoded = None
            if isinstance(decoded, dict) and decoded.get("method") == "notifications/cancelled":
                params = decoded.get("params")
                if isinstance(params, dict):
                    cancel_audit_request(params.get("requestId"))

        async def replay() -> Any:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await app(scope, replay, send)

    return bridge


async def _refuse_legacy_methods(context: ServerRequestContext[Any, Any], call_next: Any) -> Any:
    """Reject legacy lifecycle methods at the real MCPServer boundary."""
    if context.method == "notifications/cancelled":
        request_id = (context.params or {}).get("requestId")
        cancel_audit_request(request_id)
    if context.method in {"initialize", "ping"}:
        raise MCPError(
            code=METHOD_NOT_FOUND,
            message=f"{context.method} is not supported by stateless MCP {PROTOCOL_VERSION}",
        )
    return await call_next(context)


def _declares_tasks(context: ServerRequestContext[Any, Any]) -> bool:
    meta = context.meta
    capabilities = (
        meta.get("io.modelcontextprotocol/clientCapabilities")
        if isinstance(meta, Mapping)
        else None
    )
    extensions = capabilities.get("extensions") if isinstance(capabilities, Mapping) else None
    return (
        context.protocol_version == PROTOCOL_VERSION
        and isinstance(extensions, Mapping)
        and _TASK_EXTENSION in extensions
    )


def _task_candidate(name: str, arguments: dict[str, Any]) -> bool:
    """Only requests with enough input to begin work may detach."""
    request = arguments.get("request")
    if name == "audit":
        return bool(arguments.get("path"))
    if name == "trace":
        return isinstance(request, dict) and bool(request.get("run"))
    if name == "compare":
        return (
            isinstance(request, dict)
            and bool(request.get("baseline"))
            and bool(request.get("candidate"))
        )
    if name == "improve":
        return isinstance(request, dict) and bool(request.get("path"))
    if name == "forecast":
        return isinstance(request, dict) and arguments.get("action") in {"issue", "resolve"}
    if name == "seeker.propose":
        return (
            bool(arguments.get("projectPath"))
            and bool(arguments.get("taskId"))
            and bool(arguments.get("response"))
        )
    return False


@dataclass
class _TaskRecord:
    task_id: str
    created_monotonic: float = field(default_factory=time.monotonic)
    status: str = "working"
    status_message: str = "reward-lens operation is running"
    created_at: str = field(default_factory=_now_utc)
    last_updated_at: str = field(default_factory=_now_utc)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancellation: threading.Event = field(default_factory=threading.Event)
    worker: asyncio.Task[Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        answer: dict[str, Any] = {
            "taskId": self.task_id,
            "status": self.status,
            "statusMessage": self.status_message,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
            "ttlMs": _TASK_TTL_MS,
            "pollIntervalMs": _TASK_POLL_MS,
        }
        if self.status == "completed":
            answer["result"] = self.result
        elif self.status == "failed":
            answer["error"] = self.error
        return answer

    def set_status(
        self,
        status: str,
        message: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.status_message = message
        self.last_updated_at = _now_utc()
        self.result = result
        self.error = error


class _TaskIdParams(RequestParams):
    task_id: str = Field(alias="taskId")


class _TaskUpdateParams(_TaskIdParams):
    input_responses: dict[str, Any] = Field(alias="inputResponses")


class _TasksExtension(Extension):
    """A small SEP-2663 adapter around existing SDK tool calls."""

    identifier = _TASK_EXTENSION

    def __init__(self) -> None:
        self.values: dict[str, _TaskRecord] = {}

    def _prune_expired(self) -> None:
        cutoff = time.monotonic() - _TASK_TTL_MS / 1000
        for task_id, record in list(self.values.items()):
            if record.status != "working" and record.created_monotonic < cutoff:
                del self.values[task_id]

    def _record(self, task_id: str) -> _TaskRecord:
        self._prune_expired()
        record = self.values.get(task_id)
        if record is None:
            raise MCPError(code=INVALID_PARAMS, message="unknown taskId")
        return record

    def get(self, task_id: str) -> dict[str, Any]:
        return self._record(task_id).snapshot()

    @staticmethod
    def _require_capability(context: ServerRequestContext[Any, Any]) -> None:
        if not _declares_tasks(context):
            raise MCPError(
                code=-32021,
                message="Missing required client capability",
                data={"requiredCapabilities": {"extensions": {_TASK_EXTENSION: {}}}},
            )

    async def _get(
        self, context: ServerRequestContext[Any, Any], params: _TaskIdParams
    ) -> dict[str, Any]:
        self._require_capability(context)
        return {"resultType": "complete", **self.get(params.task_id)}

    async def _update(
        self, context: ServerRequestContext[Any, Any], params: _TaskUpdateParams
    ) -> dict[str, Any]:
        self._require_capability(context)
        self._record(params.task_id)
        # This bounded adapter resolves missing input before task creation, so
        # no task has outstanding inputRequests. Unknown responses are ignored
        # as SEP-2663 prescribes.
        return {"resultType": "complete"}

    async def _cancel(
        self, context: ServerRequestContext[Any, Any], params: _TaskIdParams
    ) -> dict[str, Any]:
        self._require_capability(context)
        record = self._record(params.task_id)
        if record.status == "working":
            record.cancellation.set()
            record.status_message = "cancellation requested; waiting for the SDK operation"
            record.last_updated_at = _now_utc()
        return {"resultType": "complete"}

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding(
                method=method,
                params_type=model,
                handler=handler,
                protocol_versions=frozenset({PROTOCOL_VERSION}),
            )
            for method, model, handler in (
                ("tasks/get", _TaskIdParams, self._get),
                ("tasks/update", _TaskUpdateParams, self._update),
                ("tasks/cancel", _TaskIdParams, self._cancel),
            )
        ]

    async def intercept_tool_call(
        self, params: Any, context: ServerRequestContext[Any, Any], call_next: Any
    ) -> Any:
        arguments = dict(params.arguments or {})
        if not _declares_tasks(context) or not _task_candidate(params.name, arguments):
            return await call_next(context)
        self._prune_expired()
        if len(self.values) >= 128:
            raise MCPError(code=-32030, message="reward-lens task capacity reached")
        record = _TaskRecord(task_id="rl-" + secrets.token_urlsafe(24))
        self.values[record.task_id] = record

        async def work() -> None:
            try:

                def call() -> CallToolResult | InputRequiredResult:
                    if params.name == "audit":
                        with response_score_cancellation(record.cancellation):
                            return dispatch(
                                None, "tools/call", {"name": params.name, "arguments": arguments}
                            )
                    return dispatch(
                        None, "tools/call", {"name": params.name, "arguments": arguments}
                    )

                result = await anyio.to_thread.run_sync(call)
                if isinstance(result, InputRequiredResult):
                    record.set_status(
                        "failed",
                        "input was required before task creation",
                        error={"code": -32603, "message": "unexpected input_required task result"},
                    )
                else:
                    record.set_status(
                        "completed",
                        "reward-lens operation completed",
                        result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
                    )
            except ResponseScoreCancelled:
                record.set_status("cancelled", "reward-lens audit cancelled")
            except MCPError as error:
                record.set_status(
                    "failed",
                    "reward-lens protocol error",
                    error={"code": error.code, "message": error.message},
                )
            except Exception:
                record.set_status(
                    "failed",
                    "reward-lens operation failed",
                    error={"code": -32603, "message": "internal execution error"},
                )

        record.worker = asyncio.create_task(work())
        return {"resultType": "task", **record.snapshot()}


_TASKS = _TasksExtension()


def task_extension() -> _TasksExtension:
    """The bounded task registry used by the official MCP Extension hook."""
    return _TASKS


def _summary(result: Any) -> str:
    if isinstance(result, dict) and "commands" in result:
        return f"{len(result['commands'])} commands"
    if hasattr(result, "code") and hasattr(result, "title"):
        return f"{result.code}: {result.title}"
    if hasattr(result, "summary"):
        return str(result.summary)
    if hasattr(result, "assay_id"):
        return f"assay {result.assay_id}"
    if hasattr(result, "capabilities"):
        return f"{len(result.capabilities)} capabilities"
    return "seeker proposal scored"


def _envelope(result: Any, *, kind: str, path: str | None = None) -> CallToolResult:
    payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    structured = {"resultType": kind, "path": path, "summary": _summary(result), "result": payload}
    return CallToolResult(
        content=[TextContent(text=structured["summary"])],
        structuredContent=structured,
        resultType="complete",
    )


def _input_needed(field: str, *, request_state: str) -> InputRequiredResult:
    # `requestState` is opaque and lets a modern client retry the original call
    # with `inputResponses`; no protocol session is created.
    return InputRequiredResult(requestState=request_state)


def _tool_list() -> list[Tool]:
    # The in-process seam advertises the same request contracts as the SDK
    # registrations below.  The established audit/proposal tools retain their
    # flat arguments for existing clients.
    request_tools = {
        "trace": api.TraceRequest,
        "compare": api.CompareRequest,
        "improve": api.ImproveRequest,
    }
    tools = [
        Tool(
            name=name,
            description=f"Run reward_lens.api.{name}.",
            inputSchema={
                "type": "object",
                "properties": {"request": model.model_json_schema()},
                "required": ["request"],
                "additionalProperties": False,
            },
        )
        for name, model in request_tools.items()
    ]
    tools.append(
        Tool(
            name="forecast",
            description="Issue, resolve, or read the forecast ledger.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"enum": ["issue", "resolve", "ledger"]},
                    "request": {"type": "object"},
                },
                "required": ["action", "request"],
                "additionalProperties": False,
            },
        )
    )
    for name in ("audit", "open", "explain", "doctor", "describe", "seeker.propose"):
        tools.append(
            Tool(
                name=name,
                description=f"Run reward-lens {name}.",
                inputSchema={"type": "object", "additionalProperties": True},
            )
        )
    return tools


def _sdk_request(model: type[Any], arguments: dict[str, Any]) -> Any:
    """Validate JSON tool arguments through the public SDK's request contract."""
    try:
        return model.model_validate_json(json.dumps(arguments))
    except PydanticValidationError as invalid:
        raise request_refusal(model, invalid) from invalid


def _saved_assay_path(record: Any, *roots: Any) -> str | None:
    """Find the exact record the SDK just returned through the existing artifact locator."""
    from reward_lens.cli.verbs import _shared

    for root in roots:
        if root is None:
            continue
        artifacts, project = _shared.artifacts_for_record(record, root)
        if artifacts.get("assay"):
            return str(Path(project) / artifacts["assay"])
    return None


def _sdk_result(name: str, arguments: dict[str, Any]) -> CallToolResult:
    if name in {"trace", "compare", "improve"}:
        model, operation = {
            "trace": (api.TraceRequest, api.trace),
            "compare": (api.CompareRequest, api.compare),
            "improve": (api.ImproveRequest, api.improve),
        }[name]
        request = _sdk_request(model, arguments)
        result = operation(request)
        roots = (
            getattr(request, "project", None),
            getattr(request, "path", None),
            getattr(request, "run_dir", None),
            getattr(request, "run", None),
        )
        saved = _saved_assay_path(result, *roots)
        if name == "trace" and saved is None and request.run:
            from reward_lens.cli.verbs import _shared

            artifacts, project = _shared.write_trace_artifacts(result, Path(request.run))
            if artifacts.get("assay"):
                saved = str(Path(project) / artifacts["assay"])
        return _envelope(result, kind="assay", path=saved)
    if name == "forecast":
        action = arguments.pop("action", None)
        routes = {
            "issue": (api.ForecastIssueRequest, api.forecast_issue),
            "resolve": (api.ForecastResolveRequest, api.forecast_resolve),
            "ledger": (api.ForecastLedgerRequest, api.forecast_ledger),
        }
        if action not in routes:
            raise contracts.UsageError(
                code="RL0003",
                message="forecast action must be issue, resolve, or ledger",
                remediation="supply a forecast action",
                context={"action": action},
            )
        model, operation = routes[action]
        request = _sdk_request(model, arguments)
        result = operation(request)
        saved = _saved_assay_path(result, request.project or Path.cwd())
        return _envelope(result, kind="assay", path=saved)
    if name == "open":
        path = arguments.get("path")
        if not path:
            return _input_needed("path", request_state="reward-lens:open:path")
        return _envelope(api.open_record(Path(str(path))), kind="assay", path=str(path))
    if name == "explain":
        code = arguments.get("code")
        if not code:
            return _input_needed("code", request_state="reward-lens:explain:code")
        return _envelope(api.explain(str(code)), kind="explanation")
    if name == "describe":
        return _envelope(api.describe(), kind="command_tree")
    raise ValueError(f"unknown SDK tool {name}")


async def _run_with_progress(
    context: Context, server: MCPServer, name: str, arguments: dict[str, Any]
) -> CallToolResult | InputRequiredResult:
    """Keep synchronous SDK work off the transport loop and report native MCP progress."""
    await context.report_progress(0, 1, f"{name} started")
    task = asyncio.create_task(
        anyio.to_thread.run_sync(
            lambda: dispatch(server, "tools/call", {"name": name, "arguments": arguments})
        )
    )
    last_report = time.monotonic()
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), _AUDIT_THREAD_POLL_SECONDS)
        except TimeoutError:
            if time.monotonic() - last_report >= 1:
                await context.report_progress(0, 1, f"{name} running")
                last_report = time.monotonic()
    result = task.result()
    await context.report_progress(1, 1, f"{name} finished")
    return result


def dispatch(
    server: MCPServer | None,
    method: str,
    params: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> DiscoverResult | ListToolsResult | CallToolResult | InputRequiredResult:
    """A typed in-process seam used by the SDK wrappers and transport tests.

    Passing headers means this is the HTTP path, so the required modern
    `Mcp-Method` and `Mcp-Name` pair is checked before dispatch.
    """
    if headers is not None and (not headers.get("Mcp-Method") or not headers.get("Mcp-Name")):
        raise ValueError("Mcp-Method and Mcp-Name are required on HTTP POST")
    if method in {"initialize", "ping"}:
        raise ValueError(f"{method} is not supported by stateless MCP {PROTOCOL_VERSION}")
    if method == "server/discover":
        return DiscoverResult(
            supportedVersions=[PROTOCOL_VERSION],
            capabilities=ServerCapabilities(
                tools=ToolsCapability(), extensions={_TASK_EXTENSION: {}}
            ),
            ttlMs=0,
            cacheScope=CACHE_SCOPE,
            instructions="Use the typed reward-lens tools; each result names its artifact path and summary.",
        )
    if method == "tools/list":
        return ListToolsResult(tools=_tool_list(), ttlMs=0, cacheScope=CACHE_SCOPE)
    if method != "tools/call":
        raise ValueError(f"{method} is not supported")
    name = str(params.get("name", ""))
    arguments = dict(params.get("arguments") or {})
    if name in {"trace", "compare", "forecast", "improve", "open", "explain", "describe"}:
        if name in {"trace", "compare", "forecast", "improve"}:
            nested = arguments.get("request")
            if not isinstance(nested, dict):
                return _input_needed("request", request_state=f"reward-lens:{name}:request")
            arguments = dict(nested)
            if name == "forecast":
                arguments["action"] = params.get("arguments", {}).get("action")
        try:
            return _sdk_result(name, arguments)
        except contracts.RewardLensError as error:
            return CallToolResult(
                content=[TextContent(text=error.message)],
                structuredContent={
                    "resultType": "refused",
                    "path": None,
                    "summary": error.message,
                    "result": {"code": error.code, "context": dict(error.context)},
                },
                isError=True,
            )
    if name == "audit":
        path = arguments.get("path")
        if not path:
            return _input_needed("path", request_state="reward-lens:audit:path")
        run_dir = arguments.get("runDir")
        tasks = arguments.get("tasks")
        responses = arguments.get("responses")
        result = api.audit(
            api.AuditRequest(
                path=Path(str(path)),
                run_dir=Path(str(run_dir)) if run_dir else None,
                tasks=Path(str(tasks)) if tasks else None,
                responses=Path(str(responses)) if responses else None,
            )
        )
        return _envelope(result, kind="assay", path=_saved_assay_path(result, path, run_dir))
    if name == "doctor":
        result = api.doctor()
        return _envelope(result, kind="capabilities")
    if name == "seeker.propose":
        # The caller must name spending deliberately.  This server never makes
        # a paid call; any value other than `none` is refused before the actor.
        if arguments.get("spend", "none") != "none":
            return CallToolResult(
                content=[
                    TextContent(
                        text="paid seeker execution requires an explicit authorized surface"
                    )
                ],
                structuredContent={
                    "resultType": "refused",
                    "path": None,
                    "summary": "paid spend refused",
                },
                isError=True,
            )
        session = arguments.get("session")
        project_path = arguments.get("projectPath")
        if bool(session) == bool(project_path):
            return CallToolResult(
                content=[TextContent(text="provide exactly one of session or projectPath")],
                structuredContent={
                    "resultType": "refused",
                    "path": None,
                    "summary": "ambiguous seeker proposal route",
                },
                isError=True,
            )
        if project_path:
            try:
                result = api.seeker_propose(
                    api.SeekerProposalRequest(
                        project_path=Path(str(project_path)),
                        task_id=str(arguments.get("taskId", arguments.get("task_id", ""))),
                        response=str(arguments.get("response", "")),
                        rationale=str(arguments.get("rationale", "")),
                        spend=str(arguments.get("spend", "none")),
                        run_dir=(
                            Path(str(arguments["runDir"]))
                            if arguments.get("runDir") is not None
                            else None
                        ),
                    )
                )
            except contracts.RewardLensError as error:
                return CallToolResult(
                    content=[TextContent(text=error.message)],
                    structuredContent={
                        "resultType": "refused",
                        "path": None,
                        "summary": "seeker proposal refused",
                        "result": {"code": error.code, "context": dict(error.context)},
                    },
                    isError=True,
                )
            return _envelope(result, kind="seeker_proposal", path=result.record_path)
        outcome = propose_tool(arguments)
        if "error" in outcome:
            return CallToolResult(
                content=[TextContent(text=str(outcome["error"]["message"]))],
                structuredContent={
                    "resultType": "refused",
                    "path": None,
                    "summary": "seeker proposal refused",
                    "result": outcome,
                },
                isError=True,
            )
        return _envelope(outcome, kind="seeker_outcome")
    raise ValueError(f"unknown tool {name}")


def build_server() -> MCPServer:
    """Build the official v2 server for stdio or Streamable HTTP."""
    server = MCPServer(
        "reward-lens",
        instructions="Stateless reward-lens typed SDK server.",
        extensions=[_TASKS],
        cache_hints={
            "server/discover": CacheHint(ttl_ms=0, scope=CACHE_SCOPE),
            "tools/list": CacheHint(ttl_ms=0, scope=CACHE_SCOPE),
        },
        middleware=[_refuse_legacy_methods],
    )

    @server.tool(name="audit")
    async def audit_tool(
        context: Context,
        path: str | None = None,
        run_dir: str | None = None,
        tasks: str | None = None,
        responses: str | None = None,
    ) -> CallToolResult | InputRequiredResult:
        cancellation = threading.Event()
        request_id = context.request_id
        _AUDIT_CANCELLATIONS[request_id] = cancellation
        await context.report_progress(0, 1, "audit started")

        def call() -> CallToolResult | InputRequiredResult:
            try:
                _profile("audit_dispatch_start")
                with response_score_cancellation(cancellation):
                    result = dispatch(
                        server,
                        "tools/call",
                        {
                            "name": "audit",
                            "arguments": {
                                "path": path,
                                "runDir": run_dir,
                                "tasks": tasks,
                                "responses": responses,
                            },
                        },
                    )
                _profile("audit_dispatch_result")
                return result
            except BaseException:
                _profile("audit_dispatch_error")
                raise
            finally:
                _profile("audit_dispatch_end")
                _AUDIT_CANCELLATIONS.pop(request_id, None)

        task = asyncio.create_task(anyio.to_thread.run_sync(call))
        last_report = time.monotonic()
        try:
            # The custom stdio pipe can remain parked in its read readiness
            # wait after this worker finishes.  A bounded scheduler poll keeps
            # the handler responsive without changing the client's deadline or
            # making the synchronous audit route run on the event loop.
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), _AUDIT_THREAD_POLL_SECONDS)
                except TimeoutError:
                    if time.monotonic() - last_report >= 1:
                        await context.report_progress(0, 1, "audit running")
                        last_report = time.monotonic()
                    continue
            result = task.result()
            await context.report_progress(1, 1, "audit finished")
            _profile("audit_tool_response_ready")
            return result
        except asyncio.CancelledError:
            cancellation.set()
            try:
                await asyncio.shield(task)
            except (asyncio.CancelledError, ResponseScoreCancelled):
                pass
            raise

    @server.tool(name="doctor")
    async def doctor_tool() -> CallToolResult | InputRequiredResult:
        return dispatch(server, "tools/call", {"name": "doctor", "arguments": {}})

    @server.tool(name="trace")
    async def trace_tool(
        context: Context, request: dict[str, Any]
    ) -> CallToolResult | InputRequiredResult:
        return await _run_with_progress(context, server, "trace", {"request": request})

    @server.tool(name="compare")
    async def compare_tool(
        context: Context, request: dict[str, Any]
    ) -> CallToolResult | InputRequiredResult:
        return await _run_with_progress(context, server, "compare", {"request": request})

    @server.tool(name="forecast")
    async def forecast_tool(
        context: Context, action: str, request: dict[str, Any]
    ) -> CallToolResult | InputRequiredResult:
        return await _run_with_progress(
            context, server, "forecast", {"action": action, "request": request}
        )

    @server.tool(name="improve")
    async def improve_tool(
        context: Context, request: dict[str, Any]
    ) -> CallToolResult | InputRequiredResult:
        return await _run_with_progress(context, server, "improve", {"request": request})

    @server.tool(name="open")
    async def open_tool(path: str) -> CallToolResult | InputRequiredResult:
        return dispatch(server, "tools/call", {"name": "open", "arguments": {"path": path}})

    @server.tool(name="explain")
    async def explain_tool(code: str) -> CallToolResult | InputRequiredResult:
        return dispatch(server, "tools/call", {"name": "explain", "arguments": {"code": code}})

    @server.tool(name="describe")
    async def describe_tool() -> CallToolResult | InputRequiredResult:
        return dispatch(server, "tools/call", {"name": "describe", "arguments": {}})

    @server.tool(name="seeker.propose")
    async def seeker_propose(
        session: str | None = None,
        projectPath: str | None = None,
        taskId: str | None = None,
        task_id: str | None = None,
        response: str = "",
        rationale: str = "",
        spend: str = "none",
        runDir: str | None = None,
    ) -> CallToolResult | InputRequiredResult:
        return dispatch(
            server,
            "tools/call",
            {
                "name": "seeker.propose",
                "arguments": {
                    "session": session,
                    "projectPath": projectPath,
                    "taskId": taskId,
                    "task_id": task_id,
                    "response": response,
                    "rationale": rationale,
                    "spend": spend,
                    "runDir": runDir,
                },
            },
        )

    # Advertise the exact SDK request schemas while retaining a dict boundary:
    # the SDK's eager model construction would turn a malformed tool argument
    # into an untyped protocol error before our RewardLensError refusal path.
    for name, model in (
        ("trace", api.TraceRequest),
        ("compare", api.CompareRequest),
        ("improve", api.ImproveRequest),
    ):
        registration = server._tool_manager.get_tool(name)
        assert registration is not None
        registration.parameters["properties"]["request"] = model.model_json_schema()

    return server


def run(*, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run only stdio or Streamable HTTP; SSE and legacy transports are refused."""
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("transport must be stdio or streamable-http")
    server = build_server()
    if transport == "stdio":
        import anyio

        from .stdio import run_stdio_server

        anyio.run(run_stdio_server, server)
    else:
        import uvicorn

        application = server.streamable_http_app(
            host=host,
            streamable_http_path="/mcp",
            stateless_http=True,
        )
        config = uvicorn.Config(_http_cancellation_bridge(application), host=host, port=port)
        uvicorn.Server(config).run()


__all__ = ["PROTOCOL_VERSION", "build_server", "dispatch", "run", "task_extension"]
