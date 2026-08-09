"""Block every socket entry point, then import the errors package and explain a code.

Run as a child process by `test_error_explain.py`. If anything on the import path or inside
`explain` reaches for the network, the process dies with a non-zero status and the test says so.
"""

from __future__ import annotations

import socket
import sys


def _refuse(*args: object, **kwargs: object) -> object:
    raise AssertionError("the network was touched")


socket.socket = _refuse  # type: ignore[assignment]
socket.create_connection = _refuse  # type: ignore[assignment]
socket.getaddrinfo = _refuse  # type: ignore[assignment]
socket.gethostbyname = _refuse  # type: ignore[assignment]

from reward_lens.errors import explain  # noqa: E402

sys.stdout.write(explain("RL0210"))
