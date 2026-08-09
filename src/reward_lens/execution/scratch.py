"""Where a piece of source is written before it is run.

This lives on its own so that `run_python` needs nothing from a sandbox but `Sandbox.run`. The
staging is the same for every tier and every operating system: a fresh 0700 directory under the
caller's working directory, one file in it, and that directory as the run's own cwd, so the
grader's writes land where the write budget is measured.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

__all__ = ["stage_source"]


def stage_source(source: str, *, cwd: Path, name: str = "main.py") -> tuple[Path, Path]:
    """Write `source` into a fresh 0700 scratch directory under `cwd`.

    Returns the scratch directory and the file, in that order: the directory is what a run uses as
    its working directory, and the file is what goes on the interpreter's argv.
    """
    scratch = Path(tempfile.mkdtemp(prefix="rl-", dir=str(Path(cwd))))
    os.chmod(scratch, stat.S_IRWXU)
    main = scratch / name
    main.write_text(source, encoding="utf-8")
    return scratch, main
