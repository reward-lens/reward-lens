"""X7 acceptance: the campaign's 0.214, re-adjudicated against the floor its encoding imposes.

The clause, in full:

    *The campaign's published `intransitive_mass` is reproduced from its own store; the corpus is
    shown to contain no cyclic structure by two checks that rule out different things; the
    closed-form encoding floor `(n-2)/(3n)` is derived in the module and verified against B1's own
    decomposition rather than quoted; the exact floor of the campaign's own comparison design is
    computed and the observed value is shown to sit on it; the registered threshold is shown to lie
    below that floor, so the prediction could not have failed; and the published findings section
    contains no number the experiment's own evidence store cannot verify.*

Most of this file reads the released artifacts under `experiments/x7_intransitivity/` rather than
rebuilding them, because the release is the deliverable and a test that only checked a fresh rebuild
would pass while the published files said something else. The parts that are cheap enough to rebuild
live are rebuilt live, so the pipeline is proved to run rather than proved to have run once.

Producing the artifacts:

    python -m experiments.x7_intransitivity

It reads one evidence store read-only, writes nothing back to it, and needs no network, no GPU and
no hosted model. It takes about a hundred seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.x7_intransitivity import (
    CAMPAIGN_STORE,
    REPO,
    Ledger,
    check_the_closed_form,
    curl_mass_from_out_degrees,
    exact_bounds_on_complete_graphs,
    load_campaign,
)
from reward_lens.artifacts.claims import check_files, find_unbound_numbers
from reward_lens.core.store import EvidenceStore
from reward_lens.measure.composition.hodge import PairCount, edge_flow, split_flow
from reward_lens.measure.composition.nulls import transitive_curl_mass_complete

RELEASE = REPO / "experiments" / "x7_intransitivity"

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with `python -m experiments.x7_intransitivity`; "
        f"this file checks the published artifacts, and skipping it leaves them unchecked rather "
        f"than checked."
    ),
)

needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=f"the campaign evidence store is not at {CAMPAIGN_STORE}",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence", readonly=True)


@pytest.fixture(scope="module")
def rows(store: EvidenceStore) -> dict[str, Any]:
    """Every row of the release, keyed by the observable name that wrote it."""
    return {store.get(i).observable: store.get(i).value for i in list(store._index)}


# ---------------------------------------------------------------------------
# Clause 1: the published number reproduces from the campaign's own store
# ---------------------------------------------------------------------------


def test_the_published_intransitive_mass_reproduces(rows: dict[str, Any]) -> None:
    """An independent solve against the stored adjudication row, to fifteen significant figures.

    Without this the rest of the experiment is a measurement of a different thing that happens to
    be nearby.
    """
    r = rows["x7.reproduction"]
    assert r["curl_mass"] == pytest.approx(r["stored_curl_mass"], abs=1e-14)
    assert r["gradient_mass"] == pytest.approx(r["stored_gradient_mass"], abs=1e-14)
    assert r["curl_abs_error"] < 1e-14
    assert r["harmonic_mass"] < 1e-20, "float zero, so the mass is entirely curl"
    assert r["n_tournaments"] == 10_000
    assert r["n_edges"] == 130_129
    assert r["n_triangles"] == 186_378
    assert r["betti1"] == 338
    # The structural self-checks the decomposition will not return a result without.
    assert r["orthogonality_residual"] < 1e-10
    assert r["reconstruction_residual"] < 1e-10


def test_every_pair_was_compared_exactly_once(rows: dict[str, Any]) -> None:
    """The fact that decides which nulls can run at all, recorded rather than assumed."""
    r = rows["x7.reproduction"]
    assert r["min_replications"] == 1.0
    assert r["unanimous_fraction"] == 1.0


# ---------------------------------------------------------------------------
# Clause 2: no cyclic structure, by two checks that rule out different things
# ---------------------------------------------------------------------------


def test_not_one_filled_triangle_circulates(rows: dict[str, Any]) -> None:
    """A cyclic triple on ±1 data circulates 3 and a transitive one circulates 1."""
    r = rows["x7.cyclic_census"]
    assert r["max_triangle_circulation"] == pytest.approx(1.0)
    assert r["max_triangle_circulation"] < r["cyclic_triple_circulation"]
    assert r["n_triangles"] == 186_378
    for name, delta in r["per_slice_max_circulation"].items():
        assert delta == pytest.approx(1.0), name


def test_every_tournament_is_acyclic_which_the_triangle_scan_cannot_establish(
    rows: dict[str, Any],
) -> None:
    """The check that carries the incomplete designs.

    UltraFeedback compares between three and six of the six pairs of four responses, so a
    four-cycle across uncompared diagonals fills no triangle and the triple scan cannot see it. A
    minimum feedback arc set of zero is the statement that no cycle of any length survives.
    """
    r = rows["x7.cyclic_census"]
    assert r["n_acyclic"] == r["n_tournaments"] == 10_000
    assert r["n_cyclic"] == 0
    assert r["max_feedback_arc_cost"] == 0.0
    assert r["n_tied_pairs"] == 0
    assert r["n_comparisons_reproduced_by_a_scalar"] == 130_129


# ---------------------------------------------------------------------------
# Clause 3: the closed form is derived and checked, not quoted
# ---------------------------------------------------------------------------


def test_the_closed_form_agrees_with_b1s_own_decomposition(rows: dict[str, Any]) -> None:
    r = rows["x7.closed_form"]
    assert r["max_abs_error"] < 1e-12
    assert r["transitive_floor_max_abs_error"] < 1e-12
    assert r["n_tournaments_checked"] >= 1_000
    # Exhaustive at three, four and five items rather than sampled.
    for n, expected in (("K3", 8), ("K4", 64), ("K5", 1024)):
        assert r["per_n"][n]["n_tournaments"] == expected


def test_the_closed_form_is_rederived_live_rather_than_only_read_off_the_release() -> None:
    """The derivation in the module, run again here on every tournament of `K4`.

    `curl_mass_from_out_degrees` is the one piece of arithmetic in the experiment that is not a
    shipped instrument, so it is the one piece worth re-running rather than reading.
    """
    ledger = _throwaway_ledger()
    result = check_the_closed_form(ledger, n_random=8)
    assert result["max_abs_error"] < 1e-12

    for n in range(3, 8):
        assert transitive_curl_mass_complete(n) == pytest.approx((n - 2) / (3 * n), abs=1e-15)
        assert curl_mass_from_out_degrees(range(n)) == pytest.approx((n - 2) / (3 * n), abs=1e-14)

    # And against the decomposition itself on a transitive seven-item tournament, which is Nectar's
    # design: 5/21 to fifteen significant figures.
    flow = edge_flow(
        [PairCount(a, b, 1.0, 0.0) for a in range(7) for b in range(a + 1, 7)],
        7,
    )
    assert split_flow(flow, with_betti=False).curl_mass == pytest.approx(5.0 / 21.0, abs=1e-15)


def test_the_transitive_tournament_is_the_global_minimiser(rows: dict[str, Any]) -> None:
    """Exact, not sampled: the minimum over every tournament on `K_n` is `(n-2)/(3n)`.

    The curl mass depends on a tournament only through its score sequence, so the minimum over
    `2^C(n,2)` tournaments is a minimum over the sequences Landau's condition admits. At seven items
    that is 59 sequences rather than 2,097,152 tournaments.
    """
    r = rows["x7.exact_bounds"]
    assert r["every_minimum_is_the_closed_form"]
    assert r["every_minimiser_is_transitive"]
    for key, row in r["by_n"].items():
        assert row["minimum_abs_error_against_closed_form"] < 1e-12, key
        assert row["minimum"] <= row["maximum"]
    # Where both routes are available they agree, and the number of minimisers is the number of
    # total orders, which is what "attained by the transitive tournament and nothing else" means.
    for key in ("K3", "K4", "K5"):
        row = r["by_n"][key]
        assert row["exhaustive_minimum"] == pytest.approx(row["minimum"], abs=1e-12)
        assert row["exhaustive_maximum"] == pytest.approx(row["maximum"], abs=1e-12)
        assert row["n_at_the_floor"] == row["n_total_orders"]


def test_the_exact_bounds_recompute_live_at_seven_items() -> None:
    """Nectar's own design, rebuilt: 59 score sequences, minimum exactly 5/21."""
    ledger = _throwaway_ledger()
    result = exact_bounds_on_complete_graphs(ledger, up_to=7)
    seven = result["by_n"]["K7"]
    assert seven["n_score_sequences"] == 59
    assert seven["n_tournaments"] == 2**21
    assert seven["minimum"] == pytest.approx(5.0 / 21.0, abs=1e-15)
    assert seven["minimiser_is_transitive"]


# ---------------------------------------------------------------------------
# Clause 4: the design's exact floor, and the corpus sitting on it
# ---------------------------------------------------------------------------


def test_the_corpus_sits_on_the_exact_floor_of_its_own_design(rows: dict[str, Any]) -> None:
    """The strongest form of the finding: not near the floor, on it.

    The pooled flow's components are the individual tournaments and the Hodge energies add, so the
    minimum over every possible grader is the sum of the per-design minima. Fifteen designs, all
    enumerated exactly.
    """
    r = rows["x7.design_floor"]
    assert r["n_distinct_designs"] == 15
    assert r["total_edges"] == 130_129
    assert abs(r["observed_minus_floor"]) < 1e-12, "the observed value is the floor"
    assert r["floor"] < r["ceiling"], (
        "the design does admit higher values, so the floor is not trivial"
    )
    assert r["every_minimiser_is_acyclic"], "nothing cyclic attains a per-design minimum"
    # The quantity minimised is the registered one, curl plus harmonic, and on the 338 chordless
    # four-cycle designs that is what makes the previous assertion true: their curl is identically
    # zero for every orientation, so a curl-only floor would let a cyclic grader tie for it.
    holed = [d for d in r["per_design"] if d["betti1"] > 0]
    assert sum(d["n_tournaments"] for d in holed) == 338
    assert all(d["n_filled_triangles"] == 0 for d in holed)
    assert all(d["maximum"] > d["minimum"] for d in holed), "the harmonic half does move"
    assert r["n_tournaments_with_a_hole"] == 338, "and it agrees with the pooled betti1"
    # And each half separately, because the two designs are not alike and only one has a closed form.
    for name, entry in r["by_slice"].items():
        assert abs(entry["observed_minus_floor"]) < 1e-12, name
    nectar = r["by_slice"]["nectar-tournaments"]
    assert nectar["floor"] == pytest.approx(5.0 / 21.0, abs=1e-15)


def test_the_registered_threshold_sits_below_the_floor(rows: dict[str, Any]) -> None:
    """The clause the whole experiment exists to discharge.

    `intransitive_mass > 0.03` was true for every grader that could have been run on this design, so
    the prediction could not have failed and the kill condition on the same number could not have
    fired.
    """
    r = rows["x7.design_floor"]
    assert r["registered_comparator"] == ">"
    assert r["registered_outcome"] == "confirmed"
    assert r["registered_threshold"] == 0.03
    assert r["kill_comparator"] == "<"
    assert r["kill_threshold"] == 0.03
    assert r["floor"] > r["registered_threshold"]
    assert r["floor_minus_threshold"] > 0.0
    assert r["threshold_as_fraction_of_floor"] < 0.2


# ---------------------------------------------------------------------------
# Clause 5: the registered instrument, and what it said
# ---------------------------------------------------------------------------


def test_the_registered_null_refuses_and_the_refusal_carries_its_remedy(
    rows: dict[str, Any],
) -> None:
    """R1 named a split-half null. It cannot run here, and saying so is the correct return value.

    This is why `PREDICTIONS.md` records R1 as resolved with a caveat rather than as resolved: the
    finding was reached by a different instrument than the one the prediction registered.
    """
    r = rows["x7.registered_null"]
    assert r["refused"] is True
    assert r["reason"] == "ACCESS_INSUFFICIENT"
    assert r["observed_min_replications"] == 1.0
    assert r["required_replications"] == 11.0
    assert r["remedy"].strip(), "a refusal without a remedy is a failure wearing a result's clothes"


def test_the_declared_baseline_puts_the_observation_on_top_of_the_null(
    rows: dict[str, Any],
) -> None:
    """B1's mandatory baseline, which SPEC-ERRATA E32 promoted from optional.

    The transitive null's mean is the observed value and the excess is zero, on the design where a
    closed form exists to check it against.
    """
    r = rows["x7.baselines"]
    nectar = r["nectar_transitive"]
    assert nectar["curl_null_mean"] == pytest.approx(nectar["curl_observed"], abs=1e-12)
    assert abs(nectar["curl_excess"]) < 1e-12
    assert nectar["curl_p_value"] > 0.5, "the observation sits on the null, not above it"
    # At one replication per pair the classical incoherence null is the same distribution, so it
    # cannot distinguish anything on this corpus. That is a property of the design.
    assert r["min_replications"] == 1.0
    random_profile = r["nectar_random_profile"]
    assert random_profile["curl_null_mean"] == pytest.approx(nectar["curl_null_mean"], abs=1e-12)


# ---------------------------------------------------------------------------
# Clause 6: the findings section carries no number the store cannot verify
# ---------------------------------------------------------------------------


def test_every_number_in_the_findings_section_is_bound_to_a_row(store: EvidenceStore) -> None:
    """`FINDINGS.md` runs the claims checker with no baseline, so this has to hold with none here."""
    path = RELEASE / "FINDINGS-x7.md"
    report = check_files([path], store)
    assert report.ok, report.render()
    assert len(report.results) >= 30, "a findings section with no claims in it proves nothing"
    assert find_unbound_numbers(path.read_text(encoding="utf-8")) == []


def test_the_figures_name_the_evidence_rows_they_drew_from() -> None:
    """A figure is prose with axes. Every one ships the ids of the rows its coordinates came from."""
    figures = sorted((RELEASE / "figures").glob("*.png"))
    assert figures, "no figures were produced"
    for figure in figures:
        caption = figure.with_suffix(".txt")
        assert caption.exists(), f"{figure.name} has no caption naming its evidence"
        text = caption.read_text(encoding="utf-8")
        assert "ev:" in text, figure.name


def test_the_release_names_the_store_it_read_and_the_sha_it_ran_at(
    manifest: dict[str, Any],
) -> None:
    assert manifest["n_tournaments"] == 10_000
    assert manifest["n_distinct_designs"] == 15
    assert set(manifest["slices"]) == {"nectar-tournaments", "ultrafeedback-tournaments"}
    assert manifest["git_sha"]
    assert manifest["evidence_rows"], "the manifest carries the row ids the findings cite"


# ---------------------------------------------------------------------------
# The campaign store itself, read live
# ---------------------------------------------------------------------------


@needs_campaign
def test_the_store_this_reads_is_the_one_that_holds_the_adjudication() -> None:
    """SPEC-ERRATA E3: the other campaign store carries no adjudication row and reproduces nothing."""
    corpus = load_campaign(CAMPAIGN_STORE)
    assert corpus.adjudication["card"] == "TOPO-HODGE"
    assert corpus.adjudication["outcomes"]["H-intransitive"] == "confirmed"
    assert corpus.adjudication["thresholds"]["H-intransitive"] == 0.03
    assert sum(len(g) for g in corpus.flows.values()) == 10_000
    assert len(corpus.designs) == 15


def _throwaway_ledger() -> Ledger:
    """A ledger writing to a temporary store, for the parts this file recomputes live."""
    import tempfile

    return Ledger(Path(tempfile.mkdtemp(prefix="x7-acceptance-")), "test")


def test_the_score_sequence_route_and_the_exhaustive_route_are_the_same_answer() -> None:
    """The check that licenses using the cheap route at seven items.

    At five items both routes are available: 1,024 tournaments enumerated directly through the
    shipped decomposition, and 9 score sequences through the closed form. They agree, which is what
    makes the seven-item number exact rather than an extrapolation.
    """
    n = 5
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    direct = []
    for bits in range(2 ** len(pairs)):
        counts = [
            PairCount(a, b, 1.0, 0.0) if (bits >> k) & 1 else PairCount(a, b, 0.0, 1.0)
            for k, (a, b) in enumerate(pairs)
        ]
        direct.append(split_flow(edge_flow(counts, n), with_betti=False).curl_mass)
    ledger = _throwaway_ledger()
    bounds = exact_bounds_on_complete_graphs(ledger, up_to=5)["by_n"]["K5"]
    assert float(np.min(direct)) == pytest.approx(bounds["minimum"], abs=1e-12)
    assert float(np.max(direct)) == pytest.approx(bounds["maximum"], abs=1e-12)
