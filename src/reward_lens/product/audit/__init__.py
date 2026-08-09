"""`audit`: ten panels, and every one of them either runs or writes its absence (section 7.3).

`plan(req)` first, then `run(req, *, project, sandbox)`. Both are what `reward_lens.api` dispatches
to through its lazy seam, so importing this package is what turns the SDK's honest all-absence
record into a real measurement.

`run` measures a subject version once. Asked again for a version the store already holds, it hands
back the stored record rather than making a second measurement of the same thing, and `last_reuse()`
is where the run says so.
"""

from __future__ import annotations

from .plan import plan
from .run import Reused, last_reuse, run

__all__ = ["Reused", "last_reuse", "plan", "run"]
