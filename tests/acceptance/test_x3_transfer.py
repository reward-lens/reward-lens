"""X3 acceptance: the planted-to-real transfer coefficient, published as a quantity with a method.

The clause, in full:

    *The gap between an instrument's recovery on planted organisms and on a naturally arising
    labelled corpus is measured with an interval, written to the store as a `Transfer` row so it
    composes into the calibration chain, and accompanied by a method section saying what a transfer
    coefficient is and why an instrument without one is uncalibrated for production use.*

Most of this file reads the released artifacts under `experiments/x3_release/` rather than
rebuilding them, because the release is the deliverable and a test that only checked a fresh
rebuild would pass while the published files said something else. Three tests do run live code, on
inputs small enough to build in the test: the planting, the interval, and the figure guard.

Producing the artifacts:

    python -m experiments.x3_transfer --allow-dirty \\
        --rollouts /path/to/aisi-rollouts.parquet

The rollout table is a 189 MB parquet and is not vendored, so the release is built once and checked
here. It needs no GPU and no hosted model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.x3_transfer import (
    DOSES,
    PLANTING_MODES,
    RENDERINGS,
    Bound,
    Corpus,
    Interval,
    _decimals,
    build_chain,
    dose_fragment,
    dose_key,
    figure_transfer,
    plant,
    resolve,
    short,
    transfer_interval,
)
from reward_lens.artifacts.claims import check_files
from reward_lens.core.reference import Transfer
from reward_lens.core.store import EvidenceStore

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "experiments" / "x3_release"

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason=(
        f"no release under {RELEASE}. Produce it with `python -m experiments.x3_transfer`; this "
        f"file checks the published artifacts, and skipping it leaves them unchecked rather than "
        f"checked."
    ),
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def store() -> EvidenceStore:
    return EvidenceStore(RELEASE / "evidence")


@pytest.fixture(scope="module")
def rows(store: EvidenceStore) -> dict:
    return {ev.observable.split("x3.", 1)[-1]: ev for ev in store}


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_the_coefficient_is_measured_with_an_interval(rows: dict) -> None:
    """The headline is a number and an interval, and the point estimate lies inside it.

    The point-inside-the-interval assertion is not pedantry. The first version of this experiment
    resampled only the problems both arms shared while computing the point estimate on every row,
    and produced an interval that excluded its own point. That is a silent error: both numbers look
    reasonable alone.
    """
    headline = rows["headline"].value
    value = headline["t32_calibration_transfer_max"]
    interval = headline["calibration_ci"]
    assert np.isfinite(value)
    assert interval["ci_low"] is not None and interval["ci_high"] is not None
    assert interval["ci_low"] <= value <= interval["ci_high"], (
        f"t32 = {value} is outside its own interval [{interval['ci_low']}, {interval['ci_high']}]"
    )
    assert interval["n_usable"] > 0
    assert interval["method"].startswith("bootstrap-cluster")


def test_the_real_corpus_is_the_aisi_reward_hacked_column(rows: dict) -> None:
    """The lower rung is naturally arising and labelled by someone other than the instrument."""
    census = rows["corpus_census"].value
    assert census["n_rows"] > 20_000
    assert census["n_positive"] > 0 and census["n_negative"] > 0
    assert census["n_clusters"] > 100
    # The environment declares three techniques and the run's adoption is measured, not assumed.
    assert set(census["declared"]["permitted"]) == {"always_equal", "exit", "conftest"}
    assert set(census["technique_adoption"]) == {"always_equal", "exit", "conftest"}


def test_a_transfer_row_is_in_the_store_and_is_named_t32(rows: dict) -> None:
    """`Transfer.name` is keyed by the pair of rungs, higher first, so a budget cannot double-count.

    The wave-1 fix this checks: the name used to be derived from declaration direction, so the same
    physical transfer was `t32` written one way and `t23` written the other.
    """
    row = rows["transfer"].value
    assert row["name"] == "t32"
    assert row["budget_term"] == "t32"
    assert row["value"] >= 0.0
    assert rows["transfer"].quantity == "calibration.transfer_t32"
    # And the declaration order genuinely does not matter.
    forward = Transfer(from_level="primary", to_level="reference_method", value=row["value"])
    backward = Transfer(from_level="reference_method", to_level="primary", value=row["value"])
    assert forward.name == backward.name == "t32"


def test_the_coefficient_composes_into_a_calibration_chain(rows: dict) -> None:
    """It is a term in `u_total`, not a loose number, and the uncertified reference is reported."""
    row = rows["transfer"].value
    assert row["u_total_is_none_because_reference_uncertified"] is True
    assert row["u_total_lower_bound"] >= row["value"]
    assert "u_CRM is uncertified" in row["chain_render"]
    assert row["matrix_mismatch"], "a planted arm and a real run are different matrices"


def test_the_findings_carry_the_method_section() -> None:
    """The deliverable is a quantity *with a method*, so the method is part of the artifact."""
    text = (RELEASE / "FINDINGS-x3.md").read_text(encoding="utf-8")
    assert "## Method: what a transfer coefficient is, and how to measure one" in text
    assert "uncalibrated" in text
    assert "u_total2 = u1^2 + t21^2 + t32^2 + u_CRM^2 + u_instrument^2" in text
    for ingredient in ("dose", "labelled corpus", "calibration held fixed", "unit of independence"):
        assert ingredient in text, f"the method section does not say what {ingredient!r} is for"


# ---------------------------------------------------------------------------
# Freeze before measure
# ---------------------------------------------------------------------------


def test_the_freeze_predates_every_measurement(store: EvidenceStore, manifest: dict) -> None:
    """Gate 3, checked from the artifacts rather than from the script's control flow."""
    frozen_at = json.loads((RELEASE / "freeze.json").read_text(encoding="utf-8"))["frozen_at"]
    assert frozen_at == manifest["frozen_at"]
    for ev in store:
        assert ev.created_at >= frozen_at, f"{ev.observable} predates the freeze it claims"
        assert ev.provenance.study == manifest["study_id"]


def test_a_provisional_freeze_says_so_on_the_page(manifest: dict) -> None:
    """A freeze against a dirty tree is not silently equivalent to one against a clean tree."""
    text = (RELEASE / "FINDINGS-x3.md").read_text(encoding="utf-8")
    if manifest["clean_tree"]:
        assert "This freeze is provisional" not in text
    else:
        assert "This freeze is provisional" in text
        assert manifest["git_sha"].endswith("+dirty")


def test_every_frozen_prediction_and_kill_criterion_is_adjudicated(manifest: dict) -> None:
    """No prediction goes unreported, whichever way it came out."""
    spec = json.loads((RELEASE / "freeze.json").read_text(encoding="utf-8"))["spec"]
    assert set(manifest["outcomes"]) == {h["id"] for h in spec["hypotheses"]}
    assert set(manifest["kill_outcomes"]) == {k["id"] for k in spec["kill_criteria"]}
    assert set(manifest["outcomes"].values()) <= {"confirmed", "refuted", "void"}
    assert set(manifest["kill_outcomes"].values()) <= {"fired", "passed", "void"}


def test_a_missing_metric_voids_rather_than_refutes() -> None:
    """A metric the analysis did not produce is `void`, which is not the same as `refuted`."""
    from experiments.x3_transfer import STUDY
    from reward_lens.studies.freeze import freeze

    frozen = freeze(STUDY, repo_dir=str(REPO))
    outcomes, kills = resolve(frozen, {})
    assert set(outcomes.values()) == {"void"}
    assert set(kills.values()) == {"void"}
    outcomes, _ = resolve(frozen, {"t32_calibration_transfer_max": float("nan")})
    assert outcomes["H-transfer-exceeds-tolerance"] == "void"


# ---------------------------------------------------------------------------
# The two planting designs
# ---------------------------------------------------------------------------


def test_both_planting_designs_are_reported(manifest: dict, rows: dict) -> None:
    """The design choice is the largest lever, so neither arm may be dropped."""
    assert set(manifest["designs"]) == set(PLANTING_MODES)
    for mode in PLANTING_MODES:
        assert f"design_{mode}" in rows
    metrics = manifest["metrics"]
    for mode in PLANTING_MODES:
        assert np.isfinite(metrics[f"t32_calibration_transfer_max__{mode}"])
    # The headline is the worse of the two, never an average.
    per_design = [metrics[f"t32_calibration_transfer_max__{m}"] for m in PLANTING_MODES]
    assert metrics["t32_calibration_transfer_max"] == pytest.approx(max(per_design))


def test_the_design_spread_is_published(manifest: dict) -> None:
    """A coefficient without its organism-design spread is not yet a measurement."""
    spread = manifest["metrics"]["t32_calibration_transfer_max_design_spread"]
    assert np.isfinite(spread) and spread >= 0.0
    assert "spread between the two designs" in (RELEASE / "FINDINGS-x3.md").read_text(
        encoding="utf-8"
    )


def test_the_planting_never_reads_a_label(tmp_path: Path) -> None:
    """Positives are built from the declared techniques, not mined from the corpus's positives.

    The guarantee is structural: `plant` takes donors and clusters and no labels at all, so there
    is no argument through which an answer could leak.
    """
    donors = [f"solution number {i}\n```python\nreturn {i}\n```\n" for i in range(200)]
    clusters = [f"p{i % 20}" for i in range(200)]
    corpus, detail = plant(
        donors, clusters, dose=1.0, positive_rate=0.5, n=200, seed=1, mode="append"
    )
    assert corpus.n == 200
    assert 0.4 < corpus.positive_rate < 0.6
    assert set(detail["renderings_used"]) <= {
        f"{t}[{i}]" for t, forms in RENDERINGS.items() for i in range(len(forms))
    }
    # Every planted positive carries a fragment; no negative was touched.
    positives = [t for t, y in zip(corpus.texts, corpus.labels) if y == 1]
    negatives = [t for t, y in zip(corpus.texts, corpus.labels) if y == 0]
    assert all(t in donors for t in negatives)
    assert all(t not in donors for t in positives)


def test_substitute_replaces_the_solution_and_append_keeps_it() -> None:
    """The two designs differ in exactly the way the findings say they do."""
    donors = ["a long and complete solution " * 20]
    corpus_a, _ = plant(donors, ["p0"], dose=1.0, positive_rate=1.0, n=1, seed=0, mode="append")
    corpus_s, _ = plant(donors, ["p0"], dose=1.0, positive_rate=1.0, n=1, seed=0, mode="substitute")
    assert donors[0] in corpus_a.texts[0]
    assert donors[0] not in corpus_s.texts[0]
    assert len(corpus_s.texts[0]) < len(corpus_a.texts[0])


@pytest.mark.parametrize("dose", DOSES)
def test_the_dose_truncates_the_fragment_monotonically(dose: float) -> None:
    fragment = RENDERINGS["exit"][0]
    got = dose_fragment(fragment, dose)
    assert fragment.startswith(got)
    assert len(got) <= len(fragment)
    assert len(got) >= 1


# ---------------------------------------------------------------------------
# Figures refuse a number with no evidence id
# ---------------------------------------------------------------------------


def test_a_figure_refuses_to_plot_a_bare_float(tmp_path: Path) -> None:
    """The guard is the reason a chart in this release cannot carry an unverifiable number."""
    good = Bound(value=0.9, evidence="ev:deadbeef", field="x")
    with pytest.raises(TypeError, match="carries no evidence id"):
        figure_transfer(["a"], [0.9], [good], {}, tmp_path / "f.png")
    with pytest.raises(TypeError, match="carries no evidence id"):
        figure_transfer(["a"], [good], [0.5], {}, tmp_path / "f.png")


def test_the_published_figures_exist_for_both_designs() -> None:
    for mode in PLANTING_MODES:
        assert (RELEASE / "figures" / f"transfer-{mode}.png").exists()
        assert (RELEASE / "figures" / f"dose-{mode}.png").exists()


# ---------------------------------------------------------------------------
# Every number in the prose is bound to a row
# ---------------------------------------------------------------------------


def test_the_findings_section_has_no_unbound_number(store: EvidenceStore) -> None:
    report = check_files([RELEASE / "FINDINGS-x3.md"], store)
    assert report.ok, report.render()
    assert report.results, "the findings section contains no claims at all"


def test_the_checker_would_have_caught_a_fabricated_number(
    store: EvidenceStore, tmp_path: Path
) -> None:
    """The binding is load-bearing, so a wrong number has to fail rather than pass quietly."""
    text = (RELEASE / "FINDINGS-x3.md").read_text(encoding="utf-8")
    start = text.index("[[claim value=")
    end = text.index("]]", start)
    tag = text[start : end + 2]
    fake = tag.replace("value=", "value=9.87654", 1).replace(
        "9.87654" + tag.split("value=")[1].split(" ")[0], "9.87654", 1
    )
    doctored = tmp_path / "doctored.md"
    doctored.write_text(text.replace(tag, fake, 1), encoding="utf-8")
    assert not check_files([doctored], store).ok


def test_a_claim_tolerance_matches_the_format_it_prints() -> None:
    """A fixed tolerance rejects a correctly rounded number, which is how this was found."""
    assert _decimals("{:.4f}") == 4
    assert _decimals("{:.0f}") == 0
    assert _decimals("{:.1f}") == 1
    bound = Bound(value=919.52, evidence="ev:abc", field="n")
    assert "value=919.5 " in bound.claim(fmt="{:.1f}")
    assert "tol=0.05" in bound.claim(fmt="{:.1f}")


# ---------------------------------------------------------------------------
# Refusals are results
# ---------------------------------------------------------------------------


def test_what_could_not_be_measured_is_named_with_a_remedy(store: EvidenceStore) -> None:
    refusals = [ev for ev in store if ev.observable.startswith("x3.refusal.")]
    assert refusals, "a release that refused nothing has not said what it cannot do"
    for ev in refusals:
        assert ev.value["reason"]
        assert len(ev.value["detail"]) > 40
        assert len(ev.value["remedy"]) > 40, f"{ev.observable} has no actionable remedy"
    names = {ev.observable for ev in refusals}
    assert any("white_box_probe" in n for n in names)
    assert any("single_environment" in n for n in names)


def test_the_two_baselines_that_could_not_run_are_recorded_not_dropped(rows: dict) -> None:
    """A claim that never ran the black-box comparator and one that ran it look identical
    from the outside unless the refusal is written down."""
    assert rows["headline"].value["n_instruments_refused"] == 2
    text = (RELEASE / "FINDINGS-x3.md").read_text(encoding="utf-8")
    assert "scaffolded_prompt" in text and "gradnorm_peak" in text


# ---------------------------------------------------------------------------
# The correction to the published number
# ---------------------------------------------------------------------------


def test_the_published_number_is_reproduced_before_it_is_corrected(rows: dict) -> None:
    """An erratum is a lead, not a finding: the arithmetic is checked before the label is."""
    pub = rows["published_cal_transfer"].value
    assert pub["reproduces"] is True
    assert pub["recomputed_max_abs_auc_difference"] == pytest.approx(
        pub["published_max_abs_auc_difference"], abs=1e-9
    )
    # And the correction itself: both arms are planted, no natural corpus enters it.
    assert pub["n_natural_corpora"] == 0
    assert pub["second_arm"], "the second arm has to be named for the correction to be checkable"


# ---------------------------------------------------------------------------
# The interval machinery, on inputs small enough to reason about
# ---------------------------------------------------------------------------


def _corpus(n: int, clusters: int, rng: np.random.Generator) -> tuple[Corpus, dict]:
    labels = (rng.random(n) < 0.5).astype(int)
    names = tuple(f"p{i % clusters}" for i in range(n))
    corpus = Corpus(name="t", texts=tuple("x" for _ in range(n)), labels=labels, clusters=names)
    return corpus, {"label": labels.astype(float), "s": rng.random(n)}


def _signed(a, b) -> float:
    return float(a["s"].mean() - b["s"].mean())


def test_the_interval_covers_its_own_point_estimate() -> None:
    """On a signed statistic over two genuinely different arms, the point lies inside the interval.

    Signed rather than absolute on purpose. `|X - Y|` at `X = Y` has a point estimate of zero and a
    strictly positive bootstrap distribution, so the property does not hold for it and asserting it
    would be asserting something false about a folded distribution. The released coefficient is an
    absolute value, which is why the release checks this property on the real arms, where the two
    AUCs genuinely differ, rather than on a degenerate case.
    """
    rng = np.random.default_rng(0)
    planted, arrays_p = _corpus(400, 8, rng)
    real, arrays_r = _corpus(360, 6, rng)
    got = transfer_interval(_signed, planted, real, arrays_p, arrays_r, n_resamples=400, seed=0)
    assert isinstance(got, Interval)
    assert got.n_clusters == 6
    assert got.lo <= got.point <= got.hi


def test_cloning_rows_inside_a_problem_does_not_buy_precision() -> None:
    """The unit of independence is the problem, so duplicating rows must not narrow the interval."""
    rng = np.random.default_rng(1)
    planted, arrays_p = _corpus(240, 6, rng)
    real, arrays_r = _corpus(240, 6, np.random.default_rng(2))
    plain = transfer_interval(_signed, planted, real, arrays_p, arrays_r, n_resamples=400, seed=0)

    def triple(corpus: Corpus, arrays: dict) -> tuple[Corpus, dict]:
        idx = np.repeat(np.arange(corpus.n), 3)
        return (
            Corpus(
                name=corpus.name,
                texts=tuple(corpus.texts[i] for i in idx),
                labels=corpus.labels[idx],
                clusters=tuple(corpus.clusters[i] for i in idx),
            ),
            {k: v[idx] for k, v in arrays.items()},
        )

    planted_c, arrays_pc = triple(planted, arrays_p)
    real_c, arrays_rc = triple(real, arrays_r)
    cloned = transfer_interval(
        _signed, planted_c, real_c, arrays_pc, arrays_rc, n_resamples=400, seed=0
    )
    assert cloned.n_clusters == plain.n_clusters
    # A row bootstrap would shrink the interval by about sqrt(3); a cluster bootstrap does not.
    assert (cloned.hi - cloned.lo) == pytest.approx(plain.hi - plain.lo, rel=0.05)


def test_keys_reaching_a_claim_tag_carry_no_dots() -> None:
    """A stored key with a dot in it is unreachable from a dotted field path, so the number in
    the prose could never be checked."""
    assert "." not in short("baseline.string_match")
    assert short("baseline.string_match") == "string_match"
    for dose in DOSES:
        assert "." not in dose_key(dose)
    assert len({dose_key(d) for d in DOSES}) == len(DOSES)


def test_the_chain_refuses_a_negative_coefficient() -> None:
    """A transfer is a magnitude of disagreement; a negative one is a sign error, not a value."""
    with pytest.raises(ValueError, match="cannot be negative"):
        Transfer(from_level="primary", to_level="reference_method", value=-0.1)
    transfer, chain = build_chain(
        0.3,
        {"ci_low": 0.2, "ci_high": 0.4},
        n=100,
        census={"n_rows": 10, "n_clusters": 5, "ess_clusters": 4.0},
    )
    assert transfer.name == "t32"
    assert chain.u_total is None
    assert chain.u_total_lower_bound >= 0.3
