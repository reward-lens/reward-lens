"""Public Audit policy finalization over its canonical saved Assay."""

from __future__ import annotations

from pathlib import Path

from reward_lens import api
from reward_lens.acceptance import load_policy

POLICIES = Path(__file__).parents[2] / "src/reward_lens/acceptance/policies"


def _project(root: Path) -> Path:
    root.mkdir()
    (root / "grader.py").write_text(
        "def score(task, response):\n    return 1.0 if response == 'correct' else 0.0\n",
        encoding="utf-8",
    )
    (root / "tasks.jsonl").write_text('{"id":"t1","prompt":"answer"}\n', encoding="utf-8")
    (root / "rewardlens.yaml").write_text(
        "reward:\n  entry: grader.py\n  kind: plain\n"
        "tasks:\n  path: tasks.jsonl\n  prompt: prompt\n"
        "outcome: null\n",
        encoding="utf-8",
    )
    return root


def test_sdk_policy_finalizes_the_actual_audit_assay_and_reopens_it(tmp_path: Path) -> None:
    project = _project(tmp_path / "sdk-project")
    policy_path = POLICIES / "default-selection-v1.json"

    record = api.audit(api.AuditRequest(path=project, name="sdk-policy", policy=policy_path))

    expected = load_policy(policy_path)
    saved = project / "assays" / "sdk-policy.assay.json"
    assert saved.is_file()
    reopened = api.open_record(saved)
    assert record.assay_id == reopened.assay_id
    assert reopened.decision.policy.id == expected.id
    assert reopened.decision.policy.digest == expected.digest
    assert reopened.decision.state == "unresolved"


def test_an_audit_policy_does_not_rewrite_a_previously_saved_decision(tmp_path: Path) -> None:
    project = _project(tmp_path / "immutable-project")
    policy_path = POLICIES / "default-selection-v1.json"

    api.audit(api.AuditRequest(path=project, name="first-policy", policy=policy_path))
    first = project / "assays" / "first-policy.assay.json"
    first_bytes = first.read_bytes()
    api.audit(api.AuditRequest(path=project, name="second-policy", policy=policy_path))

    assert first.read_bytes() == first_bytes
