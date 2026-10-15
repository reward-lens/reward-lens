"""The doctor matrix is assembled from local adapter probes, with no live-provider claim."""

from __future__ import annotations

from reward_lens.product.access import matrix


def test_six_families_have_separate_fixture_and_live_status() -> None:
    rows = matrix.build()
    assert set(rows) == {
        "local_deterministic",
        "stateful_tool_environment",
        "llm_judge",
        "neural_reward_model",
        "composite",
        "remote_black_box",
    }
    for family, row in rows.items():
        assert row["live"] == "unverified", family
        if row["conformance"] == "ran":
            assert row["fixture_identity"]
            assert row["supported_records"] == "manifest, evidence_envelope"
            assert row["install_route"].startswith("pip install")
        else:
            assert row["conformance"] == "missing extra"
            assert row["install_route"].startswith("pip install")


def test_missing_neural_extra_is_distinct_from_a_failed_probe(monkeypatch) -> None:
    original = matrix.find_spec
    monkeypatch.setattr(
        matrix, "find_spec", lambda name: None if name == "torch" else original(name)
    )
    row = matrix.build()["neural_reward_model"]
    assert row["conformance"] == "missing extra"
    assert row["install_route"] == "pip install 'reward-lens[train]'"


def test_only_local_deterministic_conformance_unblocks_local_audit(monkeypatch) -> None:
    monkeypatch.setitem(
        matrix.FAMILIES,
        "local_deterministic",
        lambda: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    assert matrix.runs_local_graders() is False
