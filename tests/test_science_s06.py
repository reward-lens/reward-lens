"""S6 preference topology: the Hodge decomposition, calibrated, run as a frozen study on T12.

This exercises the cheap-and-deep headline of the topology science. The pure-numpy Hodge
decomposition splits a preference flow into an orthogonal gradient (transitive), curl (locally
cyclic), and harmonic (globally cyclic) triple whose masses sum to one. The tests pin the three
channels on constructions where the answer is known (a graded total order is pure gradient, a
rock-paper-scissors three-cycle is pure curl, a chordless ring is pure harmonic), then run the study
end to end and confirm it emits REGISTERED Evidence, confirms the registered calibration, and folds
T12 on the scoreboard.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s06_topology.analysis import (
    _planted_harmonic_corpus,
    _synthetic_judge_corpus,
    _tournament,
    build_spec,
)
from studies.s06_topology.hodge import (
    decompose_corpus,
    decompose_tournament,
    enumerate_triangles,
    hodge_decomposition,
    incidence_b1,
    triangle_b2,
)


def _complete_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def test_masses_sum_to_one_and_components_are_orthogonal():
    """On an arbitrary flow the three masses sum to one and the components are mutually orthogonal."""
    n = 6
    edges = _complete_edges(n)
    flow = np.random.default_rng(0).standard_normal(len(edges))
    d = hodge_decomposition(n, edges, flow)

    assert abs(d.gradient_mass + d.curl_mass + d.harmonic_mass - 1.0) < 1e-9
    assert d.intransitive_mass == d.curl_mass + d.harmonic_mass
    # Orthogonality and exact reconstruction are the structural guarantees, at float64 zero.
    assert d.orthogonality_residual < 1e-8
    assert d.reconstruction_residual < 1e-9
    # The identity B1 @ B2 = 0 is what makes the curl subspace live inside the cycle space, which is
    # the reason the decomposition is orthogonal at all.
    triangles = enumerate_triangles(edges)
    b1 = incidence_b1(n, edges)
    b2 = triangle_b2(edges, triangles)
    assert np.allclose(b1 @ b2, 0.0, atol=1e-12)


def test_pure_gradient_flow_is_all_transitive():
    """A flow that is literally the gradient of a potential decomposes to gradient mass one."""
    n = 5
    edges = _complete_edges(n)
    rng = np.random.default_rng(1)
    potential = rng.standard_normal(n)
    b1 = incidence_b1(n, edges)
    flow = b1.T @ potential  # a discrete gradient by construction
    d = hodge_decomposition(n, edges, flow)
    assert d.gradient_mass > 1.0 - 1e-9
    assert d.intransitive_mass < 1e-9


def test_purely_transitive_tournament_has_no_cyclic_mass():
    """A total order with additively graded margins is a pure gradient: no curl, no harmonic.

    Equal-margin shutouts are not additive and would leak a little curl, so the margins here are
    graded to match integer scores, which is what a genuinely transitive preference looks like as a
    flow.
    """
    scores = [0, 1, 2, 3]
    wins_total = 6
    specs: list[tuple[int, int, int, int]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            diff = scores[j] - scores[i]
            wins_j = wins_total // 2 + diff
            wins_i = wins_total // 2 - diff
            specs.append((i, j, wins_i, wins_j))
    d = decompose_tournament(_tournament("transitive", 4, specs, "transitive"))
    assert d.gradient_mass > 0.999
    assert d.intransitive_mass < 1e-6


def test_rock_paper_scissors_is_pure_curl():
    """A three-cycle A > B > C > A with its triangle filled is pure curl."""
    specs = [(0, 1, 5, 0), (1, 2, 5, 0), (2, 0, 5, 0)]
    d = decompose_tournament(_tournament("rps", 3, specs, "rps"))
    assert d.n_triangles == 1
    assert d.curl_mass > 0.99
    assert d.harmonic_mass < 1e-9
    assert d.gradient_mass < 1e-9


def test_chordless_cycle_is_pure_harmonic():
    """A ring of comparisons with no interior pair filled is pure harmonic (a hole, not a triangle)."""
    corpus = _planted_harmonic_corpus()
    d = decompose_corpus(corpus)
    assert d.n_triangles == 0
    assert d.harmonic_mass > 0.99
    assert d.curl_mass < 1e-9
    assert d.gradient_mass < 1e-9


def test_synthetic_corpus_carries_measurable_intransitivity():
    """The synthetic judge corpus is decisively intransitive, and sparsity opens harmonic mass too."""
    d = decompose_corpus(_synthetic_judge_corpus())
    assert d.intransitive_mass > 0.03
    assert d.curl_mass > 0.0
    assert d.harmonic_mass > 0.0  # holes from unjudged pairs carry global cyclicity


def test_s06_runs_and_registers(tmp_path):
    """The study runs end to end, confirms the registered calibration, and emits REGISTERED Evidence."""
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The curl channel is calibrated (pure three-cycles recover intransitive mass one) and the
    # synthetic corpus is decisively intransitive, so neither hypothesis is refuted and nothing dies.
    assert result.outcomes["H1-calibration-recovers-planted"] == "confirmed", result.metrics
    assert result.outcomes["H2-synthetic-corpus-nonzero"] == "confirmed", result.metrics
    assert not result.killed
    assert result.metrics["calib_intransitive_mass"] > 0.99
    assert result.metrics["calib_curl_mass"] > 0.99
    assert result.metrics["planted_harmonic_recovered"] > 0.99
    assert result.metrics["synthetic_intransitive_mass"] > 0.03

    # The headline intransitive-mass Evidence is REGISTERED and traces to the three corpus measurements.
    mass = store.find(observable="S06.IntransitiveMass")
    assert mass and mass[0].trust is TrustLevel.REGISTERED
    parents = store.parents(mass[0])
    assert len(parents) == 3
    assert all(p.observable == "S06.HodgeMass" for p in parents)


def test_s06_updates_scoreboard(tmp_path):
    """The confirmed outcomes fold T12 (the Hodge-obstruction candidate law) to confirmed."""
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T12"].status == "confirmed"
    assert board.rows["T12"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
