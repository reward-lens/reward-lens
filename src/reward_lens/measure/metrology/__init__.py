"""Series A, signal metrology: the grader as a measurement device (section 5.A).

Why someone who does not care about interpretability at all would install this library. A grader is
an instrument, instruments have a repeatability and a reproducibility, and nothing in this field
reports either.

This ``__init__`` is deliberately thin and is owned by the integrator rather than by either builder,
because W3.2a and W3.2b land in the same package in the same wave and a shared export list is the one
file two concurrent agents would both have to write. Each module is importable directly; the exports
are merged at integration.
"""

from __future__ import annotations

__all__: list[str] = []
