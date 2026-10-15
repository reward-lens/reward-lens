"""The public ``audit --policy`` route persists its immutable decision."""

from __future__ import annotations

import json
from pathlib import Path

from reward_lens.acceptance import load_policy

POLICIES = Path(__file__).parents[2] / "src/reward_lens/acceptance/policies"


def test_audit_cli_saves_and_reopens_the_requested_policy(invoke, tmp_path: Path) -> None:
    project = tmp_path / "policy-project"
    project.mkdir()
    (project / "grader.py").write_text(
        "def score(task, response): return 1.0 if response == 'ok' else 0.0\n",
        encoding="utf-8",
    )
    (project / "tasks.jsonl").write_text('{"id":"t1","prompt":"answer"}\n', encoding="utf-8")
    (project / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome: null\n",
        encoding="utf-8",
    )
    policy_path = POLICIES / "default-selection-v1.json"

    result = invoke(
        "audit", str(project), "--name", "cli-policy", "--policy", str(policy_path), "--exit-zero"
    )

    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    expected = load_policy(policy_path)
    assert document["decision"]["policy"] == {"id": expected.id, "digest": expected.digest}
    saved = json.loads((project / "assays" / "cli-policy.assay.json").read_text(encoding="utf-8"))
    assert saved["decision"] == {
        **document["decision"],
        "tradeoff": None,
        "signature": None,
        "regression_cases": [],
    }
