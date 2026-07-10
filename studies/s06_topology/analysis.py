"""Preference topology (S6): a computable fraction of reward error is topologically obligatory.

The claim this study registers is sharp. Take any collection of pairwise preferences over a set of
responses, form the edge flow, and run the combinatorial Hodge decomposition (``hodge.py``). The
gradient part is exactly what a scalar (Bradley-Terry) reward can represent; the curl and harmonic
parts are what it provably cannot. So the ``intransitive_mass`` of a preference corpus is a
coordinate-free lower bound on the error of every scalar reward model on that corpus, computable in
pure numpy with no model and no training. When it is large, no amount of scalar reward-model
capacity closes the gap; the obstruction is in the data's topology, not the fit.

The study calibrates the instrument before it reports, in the corpus's usual discipline of measuring
first where the answer is known by construction:

    Calibration A (curl channel). The planted-intransitivity organism emits tournaments that are
    each a pure three-cycle A > B > C > A. A three-cycle with its triangle filled is pure curl, so
    the decomposition must return an intransitive mass of one, all of it curl. This is the registered
    calibration row T12: the method recovers the planted intransitive mass within tolerance.

    Calibration B (harmonic channel). The foundry's ``curl_harmonic_organism`` is a marked stub, so
    the harmonic channel is calibrated here against a planted ground truth of its own: a chordless
    directed cycle. A ring of comparisons with no interior pair filled has a hole the flow wraps
    around, which is pure harmonic (locally consistent yet globally cyclic), so the decomposition
    must return a harmonic mass of one. This proves the harmonic channel is real and separable from
    curl, not an artifact.

    Measurement. A synthetic judge-tournament corpus stands in for a real preference corpus. Each
    judge scores a pair by a dominant transitive quality plus a context-dependent skew criterion, the
    mechanism by which a genuine multi-attribute judge produces cycles, and the comparison graph is
    left sparse (not every pair is judged) as real judge data is. The measured intransitive mass is
    the headline: measurably nonzero, so scalar reward is provably lossy on this corpus.

The kill criterion is a real scientific fork. If the cyclic mass were uniformly tiny (a few percent
or less), that would be a publishable defense of scalar reward modeling: Bradley-Terry transitivity
would be empirically benign and the scalar bottleneck a non-issue in practice. The registered
prediction is the opposite, and the synthetic corpus is where it is first checked.

The real Nectar, UltraFeedback, HelpSteer, and PRISM slices are the same analysis on human and
judge tournaments loaded through the ``datasets`` extra, which is not installed in this environment,
so that corpus is the marked follow-on rather than run here.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.data.lineage import make_lineage
from reward_lens.data.schema import EdgeObs, Response, Tournament, response_content
from reward_lens.organisms import intransitivity_organism
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies.s06_topology.hodge import HodgeDecomposition, decompose_corpus

_VERSION = "1.0"

# The registered calibration and measurement thresholds. A pure three-cycle recovers an intransitive
# mass of exactly one, so 0.99 is a tolerance, not a margin. The synthetic corpus is engineered to
# carry a decisive cyclic mass, well clear of the "few percent" band the kill criterion reserves for
# an empirically benign scalar bottleneck.
_CALIBRATION_TOL = 0.99
_NONZERO_THRESHOLD = 0.03


def build_spec() -> StudySpec:
    """The frozen S6 spec: recover the planted intransitive mass, then measure it in a corpus (T12)."""
    return StudySpec(
        id="s06-topology",
        title="Preference topology: a computable fraction of reward error is topologically "
        "obligatory (Hodge decomposition of pairwise preference)",
        science="S06-topology",
        hypotheses=(
            Hypothesis(
                id="H1-calibration-recovers-planted",
                statement="on the planted-intransitivity organism, whose tournaments are pure "
                "three-cycles, the Hodge decomposition recovers an intransitive mass of one within "
                "tolerance (the curl channel is calibrated)",
                prediction=Prediction(
                    metric="calib_intransitive_mass", comparator=">", threshold=_CALIBRATION_TOL
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H2-synthetic-corpus-nonzero",
                statement="a synthetic judge-tournament corpus carries a measurably nonzero "
                "intransitive mass, so scalar reward is provably lossy on it",
                prediction=Prediction(
                    metric="synthetic_intransitive_mass",
                    comparator=">",
                    threshold=_NONZERO_THRESHOLD,
                ),
                scoreboard_row="T12",
            ),
        ),
        analysis="studies.s06_topology.analysis.analyze",
        subjects=SubjectQuery(
            organisms=("intransitivity",),
            extra={
                "note": "controlled organisms plus a synthetic judge corpus; the real Nectar, "
                "UltraFeedback, HelpSteer, and PRISM tournaments are the datasets-extra follow-on"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-cyclic-mass-benign",
                metric="synthetic_intransitive_mass",
                comparator="<",
                threshold=_NONZERO_THRESHOLD,
                description="cyclic mass is uniformly tiny, so Bradley-Terry transitivity is "
                "empirically benign and the scalar bottleneck a non-issue in practice, which is a "
                "publishable defense of scalar reward modeling",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tournament builders (controlled ground truth and the synthetic corpus)
# ---------------------------------------------------------------------------


def _tournament(
    prompt: str, n_items: int, edge_specs: list[tuple[int, int, int, int]], seed_id: str
) -> Tournament:
    """Assemble a `Tournament` from a list of ``(i, j, wins_i, wins_j)`` edges with a stamped lineage.

    The lineage content mirrors ``schema.content_of`` for a tournament exactly, so the item's content
    hash agrees with the dataset checksum a `DataView` would compute over it.
    """
    responses = tuple(Response(text=f"{prompt}::response-{t}") for t in range(n_items))
    edges = tuple(
        EdgeObs(i=i, j=j, wins_i=wins_i, wins_j=wins_j) for (i, j, wins_i, wins_j) in edge_specs
    )
    content = [
        "Tournament",
        prompt,
        [response_content(r) for r in responses],
        [e.__canonical__() for e in edges],
    ]
    lineage = make_lineage(seed_id, "s06.topology", ("synthetic",), content)
    return Tournament(prompt=prompt, responses=responses, edges=edges, lineage=lineage)


def _planted_harmonic_corpus(
    lengths: tuple[int, ...] = (4, 5, 6, 7), wins: int = 5
) -> list[Tournament]:
    """Chordless directed cycles: the planted harmonic ground truth for calibration B.

    A ring ``0 > 1 > ... > (L-1) > 0`` whose only edges are the ring itself has no filled triangle,
    so its single cycle is a hole the flow wraps around. That flow is divergence-free and curl-free
    yet not a gradient, which is the definition of harmonic, and the decomposition should assign it a
    harmonic mass of one. Building several lengths keeps the calibration from depending on one graph.
    """
    tournaments: list[Tournament] = []
    for length in lengths:
        specs: list[tuple[int, int, int, int]] = []
        for step in range(length):
            a, b = step, (step + 1) % length
            # Orient the winner->loser ring edge into canonical (min, max) index order.
            if a < b:
                specs.append((a, b, wins, 0))
            else:
                specs.append((b, a, 0, wins))
        tournaments.append(
            _tournament(f"harmonic-ring-{length}", length, specs, f"harmonic:{length}")
        )
    return tournaments


def _synthetic_judge_corpus(
    *,
    n_prompts: int = 60,
    n_items: int = 6,
    n_dims: int = 6,
    skew: float = 1.5,
    beta: float = 1.5,
    wins_total: int = 10,
    drop: float = 0.25,
    seed: int = 0,
) -> list[Tournament]:
    """A synthetic multi-attribute judge corpus that produces genuine, measurable intransitivity.

    Each response ``t`` carries a scalar quality ``q[t]`` and a feature vector ``phi[t]``. For a pair
    the judge's preference score is the transitive quality gap ``q[b] - q[a]`` plus a skew term
    ``skew * phi[a]^T A phi[b]`` with ``A`` skew-symmetric, so the skew term is antisymmetric in the
    pair and rotates preference through the feature plane the way a context-dependent criterion does.
    Win counts follow a logistic of that score over ``wins_total`` comparisons. A fraction ``drop`` of
    the pairs is left unjudged, so the comparison graph is sparse and some cycles enclose holes: this
    is what lets the corpus carry harmonic mass alongside curl, exactly as real judge data does.
    """
    rng = np.random.default_rng(seed)
    tournaments: list[Tournament] = []
    for prompt_idx in range(n_prompts):
        quality = rng.standard_normal(n_items)
        features = rng.standard_normal((n_items, n_dims))
        raw = rng.standard_normal((n_dims, n_dims))
        skew_op = raw - raw.T
        norm = float(np.linalg.norm(skew_op))
        if norm > 0.0:
            skew_op = skew_op / norm
        specs: list[tuple[int, int, int, int]] = []
        for a in range(n_items):
            for b in range(a + 1, n_items):
                if rng.random() < drop:
                    continue  # this pair was not judged, leaving a hole in the complex
                score = (quality[b] - quality[a]) + skew * float(
                    features[a] @ skew_op @ features[b]
                )
                p_b = 1.0 / (1.0 + np.exp(-beta * score))
                wins_b = int(round(wins_total * p_b))
                wins_a = wins_total - wins_b
                specs.append((a, b, wins_a, wins_b))
        if not specs:
            continue
        tournaments.append(
            _tournament(f"judge-prompt-{prompt_idx}", n_items, specs, f"judge:{seed}:{prompt_idx}")
        )
    return tournaments


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _mass_evidence(
    observable: str, corpus_label: str, decomposition: HodgeDecomposition, study_id: str
) -> "object":
    """A base (unregistered) Evidence carrying one corpus's Hodge masses, to be cited as a parent."""
    return make_evidence(
        observable=observable,
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id, "corpus": corpus_label}),
        value=decomposition.to_dict(),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
    )


def analyze(run) -> StudyResult:
    """Calibrate both cyclic channels, measure the synthetic corpus, and register the headline mass."""
    study_id = run.study.study_id

    # Calibration A: the planted-intransitivity organism (pure three-cycles) fixes the curl channel.
    organism_view, _key = intransitivity_organism(n_triads=24, seed=0)
    calib = decompose_corpus(organism_view)
    ev_calib = _mass_evidence("S06.HodgeMass", "organism-intransitivity", calib, study_id)
    run.record(ev_calib)

    # Calibration B: planted chordless cycles fix the harmonic channel (the foundry organism is a stub).
    harmonic_corpus = _planted_harmonic_corpus()
    harmonic = decompose_corpus(harmonic_corpus)
    ev_harmonic = _mass_evidence("S06.HodgeMass", "planted-harmonic", harmonic, study_id)
    run.record(ev_harmonic)

    # Measurement: the synthetic judge corpus stands in for a real preference corpus.
    synthetic_corpus = _synthetic_judge_corpus()
    synthetic = decompose_corpus(synthetic_corpus)
    ev_synthetic = _mass_evidence("S06.HodgeMass", "synthetic-judge", synthetic, study_id)
    run.record(ev_synthetic)

    calib_intransitive_mass = float(calib.intransitive_mass)
    planted_harmonic_recovered = float(harmonic.harmonic_mass)
    synthetic_intransitive_mass = float(synthetic.intransitive_mass)

    # The registered headline: the intransitive-mass measurement, tracing to the three corpora it
    # summarizes. This is the number a card or paper cites.
    ev_mass = make_evidence(
        observable="S06.IntransitiveMass",
        observable_version=_VERSION,
        subject=SubjectRef(extra={"study": study_id}),
        value={
            "calib_intransitive_mass": calib_intransitive_mass,
            "calib_curl_mass": float(calib.curl_mass),
            "calib_harmonic_mass": float(calib.harmonic_mass),
            "planted_harmonic_recovered": planted_harmonic_recovered,
            "synthetic_intransitive_mass": synthetic_intransitive_mass,
            "synthetic_curl_mass": float(synthetic.curl_mass),
            "synthetic_harmonic_mass": float(synthetic.harmonic_mass),
            "synthetic_gradient_mass": float(synthetic.gradient_mass),
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(
            study=study_id, parents=(ev_calib.id, ev_harmonic.id, ev_synthetic.id)
        ),
        registered=True,
    )
    run.record(ev_mass)

    return StudyResult(
        outcomes={},
        metrics={
            "calib_intransitive_mass": calib_intransitive_mass,
            "calib_curl_mass": float(calib.curl_mass),
            "planted_harmonic_recovered": planted_harmonic_recovered,
            "synthetic_intransitive_mass": synthetic_intransitive_mass,
            "synthetic_curl_mass": float(synthetic.curl_mass),
            "synthetic_harmonic_mass": float(synthetic.harmonic_mass),
            "synthetic_gradient_mass": float(synthetic.gradient_mass),
        },
        summary=(
            f"The Hodge decomposition recovered {calib_intransitive_mass:.3f} of the planted "
            f"three-cycle mass as intransitive (all curl), and {planted_harmonic_recovered:.3f} of "
            f"the planted chordless-cycle mass as harmonic, calibrating both cyclic channels. On the "
            f"synthetic judge corpus the intransitive mass was {synthetic_intransitive_mass:.3f} "
            f"(curl {float(synthetic.curl_mass):.3f}, harmonic {float(synthetic.harmonic_mass):.3f}), "
            f"a computable lower bound on the error of any scalar reward model on that corpus."
        ),
    )


__all__ = ["build_spec", "analyze"]
