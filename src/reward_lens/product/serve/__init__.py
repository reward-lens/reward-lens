"""The local, authenticated reader used by ``reward-lens open --workbench``.

This package only reads bytes already produced by the CLI.  In particular it
does not import a connector, an audit engine, or a grader.
"""

from .server import A009_STATES, DisplayState, WorkbenchServer, state_for

__all__ = ["A009_STATES", "DisplayState", "WorkbenchServer", "state_for"]
