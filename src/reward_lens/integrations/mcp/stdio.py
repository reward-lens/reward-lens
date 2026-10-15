"""Asyncio-backed stdio bridge for the official MCP server.

``mcp.server.stdio.stdio_server`` uses AnyIO's wrapped-file reader.  In the
isolated 2.2.0 runtime lane that reader never observes bytes supplied through
the SDK client's subprocess pipe.  This bridge keeps the SDK's JSON-RPC
models, ``SessionMessage`` framing, and low-level server loop, but uses
``asyncio``'s pipe transports for the process-owned file descriptors.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import anyio
import mcp_types as types
from mcp.shared.message import SessionMessage

from . import _profile, cancel_audit_request

if TYPE_CHECKING:
    from mcp.server import MCPServer


async def run_stdio_server(server: MCPServer) -> None:
    """Serve one real stdin/stdout connection until its input closes."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)
    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)
    inbound_send, inbound_receive = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    outbound_send, outbound_receive = anyio.create_memory_object_stream[SessionMessage](0)

    async def read_stdin() -> None:
        async with inbound_send:
            while line := await reader.readline():
                try:
                    message = types.jsonrpc_message_adapter.validate_json(line)
                except Exception as error:
                    await inbound_send.send(error)
                else:
                    if getattr(message, "method", None) == "notifications/cancelled":
                        params = getattr(message, "params", None)
                        if isinstance(params, dict):
                            cancel_audit_request(params.get("requestId"))
                            _profile("stdio_inbound_cancel")
                    await inbound_send.send(SessionMessage(message))

    async def write_stdout() -> None:
        async with outbound_receive:
            async for message in outbound_receive:
                wire = message.message.model_dump_json(by_alias=True, exclude_unset=True)
                _profile("stdio_outbound_frame")
                writer.write((wire + "\n").encode("utf-8"))
                await writer.drain()
                _profile("stdio_outbound_drained")

    async with anyio.create_task_group() as group:
        group.start_soon(read_stdin)
        group.start_soon(write_stdout)
        try:
            await server._lowlevel_server.run(
                inbound_receive,
                outbound_send,
                server._lowlevel_server.create_initialization_options(),
            )
        finally:
            group.cancel_scope.cancel()
