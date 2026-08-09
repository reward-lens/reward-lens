"""The error catalogue (D-24): every raised error has a stable RL-prefixed four-digit code, a one-line cause and a runnable remedy, and reward-lens explain prints the long form offline. Owned by P-ERRORS; the base exception type lives in contracts/.
"""

from __future__ import annotations

from .catalogue import CATALOGUE, ErrorSpec, check_spec
from .explain import explain, make, render_two_line

__all__ = [
    "CATALOGUE",
    "ErrorSpec",
    "check_spec",
    "explain",
    "make",
    "render_two_line",
]
