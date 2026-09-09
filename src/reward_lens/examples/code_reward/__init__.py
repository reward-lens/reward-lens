"""The code-reward example (D-63): a real reward system with a real planted defect.

`write(dest)` puts a whole project on disk: a grader over forty small Python tasks, a hundred and
twenty sampled responses, and a protected test suite as the independent outcome check. It runs on
CPU, offline, with no key and no model.

The defect is not described anywhere in what is written. The grader reads the test file back off disk
from inside the directory the graded response just ran in, so a response that rewrites that file is
paid in full for solving nothing. Which responses do that is recoverable only by running them.
"""

from .writer import MANIFEST, build, check_tasks, write

__all__ = ["MANIFEST", "build", "check_tasks", "write"]
