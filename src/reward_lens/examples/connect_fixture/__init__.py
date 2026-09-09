"""The connector fixture (D-65): one callable per shape, and the reward system section 5.9 scans.

`write(destination)` lays down what an adopter already has when they first meet reward-lens: a
`rewards.py` with three reward functions and a `data/train.jsonl` of tasks, and no
`rewardlens.yaml`. `reward-lens init <dest> --detect` is run against exactly that.

Owned by P-CONNECT.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["MANIFEST", "TASK_ROWS", "write"]

_HERE = Path(__file__).resolve().parent

#: Rows in `data/train.jsonl`. The transcript's count is read off the file, not off this number;
#: this is what the generator writes and what a test compares the file against.
TASK_ROWS = 312

MANIFEST: tuple[tuple[str, str], ...] = (
    ("rewards.py", "three reward functions, two TRL-shaped and one plain"),
    ("data/train.jsonl", f"{TASK_ROWS} tasks with prompt, answer and tests"),
)


def write(destination: str | Path) -> list[Path]:
    """Copy the fixture reward system into `destination`. Returns what was written, in order."""
    root = Path(destination)
    (root / "data").mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("rewards.py",):
        target = root / name
        shutil.copyfile(_HERE / name, target)
        written.append(target)
    target = root / "data" / "train.jsonl"
    shutil.copyfile(_HERE / "data" / "train.jsonl", target)
    written.append(target)
    return written
