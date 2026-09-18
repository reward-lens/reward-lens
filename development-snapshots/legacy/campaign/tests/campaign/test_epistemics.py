"""Tests for the Tier 0 epistemics machinery: nulls, power, the freeze gates, and smoke.

Everything runs on CPU in seconds: the null simulators are exercised at small draw counts
(determinism cares about seeding, not precision), the power gate reuses one small null bank,
and the freeze refusal paths run against throwaway git repositories under tmp_path, never
against the working tree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from campaign import freeze, nulls, power, smoke
from campaign.payloads import TablePayload
from campaign.registry import TIER2_CARDS

# ---------------------------------------------------------------------------
# nulls
# ---------------------------------------------------------------------------

_CHEAP_STATS = ("contested_raterspread_spearman", "forecast_mae", "hump_kl_abs_error")


@pytest.mark.parametrize("statistic", _CHEAP_STATS)
def test_null_simulation_is_deterministic_per_seed(statistic):
    a = nulls.simulate(statistic, n_sims=150)
    b = nulls.simulate(statistic, n_sims=150)
    assert a.seed == b.seed
    assert a.quantiles() == b.quantiles()
    assert np.array_equal(a.samples, b.samples)


def test_null_entries_carry_the_required_schema():
    res = nulls.simulate("verdict_prefix_match_rate", n_sims=150)
    entry = res.entry()
    for key in ("p50", "p90", "p95", "p99"):
        assert key in entry["quantiles"]
    for key in ("n", "n_sims", "seed"):
        assert isinstance(entry[key], int)
    assert entry["direction"] in ("greater", "less")


def test_realized_n_prefers_the_lock_over_planning():
    lock = {"slices": {"rb2-full": {"n": 1865, "n_effective": 1700.4}}}
    assert nulls.realized_n(lock, "rb2-full") == 1700
    assert nulls.realized_n(None, "rb2-full") == 1865
    assert nulls.realized_n({"slices": {}}, "judge-pairs-1000") == 1000


def test_write_nulls_is_byte_stable_and_covers_every_simulator(tmp_path):
    results = nulls.simulate_all(n_sims=120)
    path = tmp_path / "nulls.json"
    payload, table = nulls.write_nulls(path, results=results)
    first = path.read_bytes()
    nulls.write_nulls(path, results=results)
    assert path.read_bytes() == first
    assert set(payload) == set(nulls.SIMULATORS)
    parsed = json.loads(first)
    assert set(parsed) == set(nulls.SIMULATORS)
    assert table, "the clearance table must not be empty"
    for row in table:
        assert row["direction"] in ("greater", "less")
        assert row["clears"] in (True, False, None)


def test_clearance_direction_semantics():
    results = nulls.simulate_all(n_sims=200)
    rows = {(r["card"], r["metric"]): r for r in nulls.clearance_table(results)}
    # A rate far above chance clears; a threshold below a same-bank null median cannot.
    assert rows[("JUDGE-VBC", "verdict_prefix_match_rate")]["clears"] is True
    assert rows[("CHI-DRIFT", "chi_bon_spearman")]["clears"] is False
    # Equivalence-style hypotheses are marked, not scored.
    assert rows[("GAUGE-E19", "raw_cos_v01_v02")]["clears"] is None


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------


def test_spearman_power_rises_with_n():
    p = [power.spearman_power(0.3, n, crit_rho=0.2) for n in (20, 80, 320, 1280)]
    assert all(b > a for a, b in zip(p, p[1:]))
    assert p[-1] > 0.99


def test_rate_power_rises_with_n_and_effect():
    by_n = [power.rate_power(0.6, 0.5, n) for n in (25, 100, 400)]
    assert all(b > a for a, b in zip(by_n, by_n[1:]))
    by_effect = [power.rate_power(p1, 0.5, 100) for p1 in (0.55, 0.65, 0.75)]
    assert all(b > a for a, b in zip(by_effect, by_effect[1:]))


def test_shift_power_matches_its_mde_definition():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(4000)
    mde = power.shift_mde(samples)
    assert power.shift_power(samples, mde) == pytest.approx(0.8, abs=0.02)
    assert power.shift_power(samples, 0.0) == pytest.approx(0.05, abs=0.02)


@pytest.fixture(scope="module")
def small_bank():
    return nulls.simulate_all(n_sims=120)


def test_power_gate_covers_every_tier2_card(small_bank):
    verdicts = power.power_gate(TIER2_CARDS, None, nulls_results=small_bank)
    assert [v.card for v in verdicts] == [c.card for c in TIER2_CARDS]
    for v in verdicts:
        assert v.note, f"{v.card} verdict must explain its basis"
        if v.card in power.FLEET_INTENT:
            assert v.passes and np.isnan(v.power)
            assert "design intent" in v.note
        else:
            assert np.isfinite(v.power) or v.passes


def test_power_gate_fails_unknown_cards(small_bank):
    from campaign.registry import CardDef

    stranger = CardDef(card="NO-SUCH", spec_id="x", title="", science="", analysis="x.y",
                       tier="tier2", hypotheses=())
    verdicts = power.power_gate([stranger], None, nulls_results=small_bank)
    assert not verdicts[0].passes


# ---------------------------------------------------------------------------
# freeze refusal paths
# ---------------------------------------------------------------------------


def _make_repo(path: Path) -> Path:
    repo = path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "file.txt").write_text("content\n")
    subprocess.run(["git", "-C", str(repo), "add", "file.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "init", "--no-gpg-sign"],
        check=True,
    )
    return repo


def test_freeze_refuses_a_dirty_tree(tmp_path):
    repo = _make_repo(tmp_path)
    sha = freeze.assert_clean_tree(repo)
    assert len(sha) == 40
    (repo / "untracked.txt").write_text("dirty\n")
    with pytest.raises(freeze.FreezeRefused, match="dirty tree"):
        freeze.assert_clean_tree(repo)


def test_freeze_refuses_outside_a_repo(tmp_path):
    with pytest.raises(freeze.FreezeRefused, match="git repository"):
        freeze.assert_clean_tree(tmp_path / "nowhere")


def test_freeze_refuses_a_missing_lock(tmp_path):
    with pytest.raises(freeze.FreezeRefused, match="slices lock"):
        freeze.require_lock(tmp_path / "absent.lock.json")
    lock_path = tmp_path / "slices.lock.json"
    lock_path.write_text(json.dumps({"slices": {}}))
    assert freeze.require_lock(lock_path) == {"slices": {}}


def test_freeze_refuses_failed_power_gates():
    good = power.PowerVerdict("A", 0.95, 0.1, 100, True, "fine")
    bad = power.PowerVerdict("B", 0.35, 0.5, 10, False, "underpowered")
    freeze.require_power([good])
    with pytest.raises(freeze.FreezeRefused, match="B"):
        freeze.require_power([good, bad])


def test_freeze_refuses_an_existing_tag(tmp_path):
    repo = _make_repo(tmp_path)
    freeze.assert_tag_absent(repo, "campaign-freeze-v1")
    subprocess.run(["git", "-C", str(repo), "tag", "campaign-freeze-v1"], check=True)
    with pytest.raises(freeze.FreezeRefused, match="already exists"):
        freeze.assert_tag_absent(repo, "campaign-freeze-v1")


def test_ladder_table_extraction_reads_rows_then_meta():
    payload = TablePayload(
        name="rung", columns=["metric", "value"],
        rows=[["crystallization", 0.4], ["snr", 3.0]],
        meta={"teacher_variance": 1.5, "notes": "text is ignored"},
    )
    values = freeze._table_values(payload)
    assert values == {"crystallization": 0.4, "snr": 3.0, "teacher_variance": 1.5}


# ---------------------------------------------------------------------------
# smoke building blocks
# ---------------------------------------------------------------------------


def test_smoke_codec_round_trip_passes(tmp_path):
    checks = smoke.check_registry_and_codec(tmp_path)
    assert checks, "the codec section must produce checks"
    failures = [c for c in checks if c.failed]
    assert not failures, [f"{c.name}: {c.detail}" for c in failures]
    assert any(c.name == "sidecar-files" and c.status == "PASS" for c in checks)


def test_smoke_freeze_dry_round_trip(tmp_path):
    checks = smoke.check_t0_freeze_dry(tmp_path)
    by_name = {c.name: c for c in checks}
    assert by_name["study-id-stability"].status == "PASS", by_name[
        "study-id-stability"].detail
    assert by_name["claims-fabricated-id"].status == "PASS"
    assert by_name["claims-real-id"].status == "PASS"
    assert by_name["claims-cli-exit"].status == "PASS", by_name["claims-cli-exit"].detail


def test_smoke_clone_section_states_a_missing_lock(capsys):
    checks = smoke.check_t0_clone(None, None, out=lambda *_: None)
    lock_rows = [c for c in checks if c.name == "slices.lock.json"]
    assert lock_rows and lock_rows[0].status == "FAIL"
    assert "ingest" in lock_rows[0].detail
