"""BLK-073-a: a fitted direction's ``method`` names the estimator that produced it.

The link-2 template registers **difference in means over the two labelled populations** as the
primary estimator and logistic regression as the secondary. Two properties have to hold for that
registration to mean anything, and they are independent.

- The primary has to be runnable. Asking `fit_probe` for it has to produce a mean-difference
  direction, in the full fit and in every cross-validation fold, not only in the headline vector.
- The ``method`` field has to be earned. It is hashed into the `DirectionID`, so it is the only
  record of which estimator produced a vector, and a caller-supplied string that nothing checks
  lets a logistic vector travel as the registered primary through four downstream links with an id
  that agrees with the lie.

Everything here is graded against an estimator recomputed from the same inputs inside the test, so
a correct independent reimplementation of "difference in means over the two labelled populations"
passes and the shipped logistic fit does not. The planted data makes the two estimators disagree by
construction, and `test_the_two_estimators_are_not_the_same_answer_under_two_names` is the guard
that keeps the rest of the file from passing vacuously: if the estimators ever agreed on this
fixture, the label would be cosmetic and these proofs would be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.concepts.probes import (
    SiteCaptures,
    fit_probe,
    group_kfold_indices,
)
from reward_lens.core.types import Site
from reward_lens.stats.roc import roc_pr

SITE = Site(0, "resid_post")


# ---------------------------------------------------------------------------
# The fixture, and the estimator the design registers, recomputed independently
# ---------------------------------------------------------------------------


def _planted(
    *, seed: int = 0, n: int = 400, d: int = 24, nuisance_sd: float = 4.0, rho: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Labelled activations on which the two registered estimators must disagree.

    The class signal is a shift of 1.0 along axis 0. Axis 1 is a high-variance nuisance direction
    shared by both classes, and axis 0 carries a ``rho`` share of it, so the class axis and the
    nuisance axis are correlated. A difference of means reads the class shift and is blind to the
    nuisance; a logistic fit whitens by the covariance and mixes the nuisance axis in to cancel the
    shared noise. The two directions therefore separate, which is the whole reason the ``method``
    label is load-bearing rather than documentation.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, d))
    y = np.repeat([0, 1], n // 2).astype(np.int64)
    x = z.copy()
    x[:, 1] = nuisance_sd * z[:, 1]
    x[:, 0] = z[:, 0] + rho * x[:, 1]
    x[y == 1, 0] += 1.0
    return x.astype(np.float32), y


def _captures(seed: int = 0) -> SiteCaptures:
    x, y = _planted(seed=seed)
    return SiteCaptures(
        features={SITE: x},
        labels=y,
        groups=np.arange(y.shape[0]),
        answer_key=None,
        name="planted-estimator-identity",
    )


def _difference_in_means(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """The registered primary, written out from its definition and nothing else.

    ``mean(positives) - mean(negatives)``, with the decision boundary at the midpoint of the two
    class means. Returned unnormalized; every comparison below is by cosine, so scale is irrelevant
    and sign is not.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).ravel()
    mean_pos = x[y == 1].mean(axis=0)
    mean_neg = x[y == 0].mean(axis=0)
    coef = mean_pos - mean_neg
    return coef, float(-0.5 * (mean_pos + mean_neg) @ coef)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Signed cosine. Signed, because a direction that points from positives to negatives is wrong."""
    u = np.asarray(a, dtype=np.float64).ravel()
    v = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _oof_auc_under(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, cv: int, seed: int
) -> float:
    """Pooled out-of-fold AUC when every fold is fitted by difference in means.

    The fold structure is the library's own `group_kfold_indices`, because the split is not what is
    under test here; the fitter inside each fold is.
    """
    x = np.asarray(x, dtype=np.float64)
    oof = np.full(x.shape[0], np.nan, dtype=np.float64)
    for train_idx, test_idx in group_kfold_indices(groups, cv, seed=seed):
        if np.unique(y[train_idx]).size < 2:
            continue
        coef, intercept = _difference_in_means(x[train_idx], y[train_idx])
        oof[test_idx] = x[test_idx] @ coef + intercept
    finite = np.isfinite(oof)
    return float(roc_pr(oof[finite], y[finite]).auc)


# ---------------------------------------------------------------------------
# The guard: without this, everything below could pass for the wrong reason
# ---------------------------------------------------------------------------


def test_the_two_estimators_are_not_the_same_answer_under_two_names():
    """On this fixture the primary and the secondary return materially different directions.

    If they agreed, a wrong ``method`` label would be a documentation defect and the proofs below
    would hold no matter which estimator ran. They do not agree: the directions are far enough
    apart that they disagree about individual rows, and the out-of-fold AUCs differ by more than
    any tolerance used below.
    """
    caps = _captures()
    x = caps.features[SITE]
    dim_coef, _ = _difference_in_means(x, caps.labels)

    logistic = fit_probe(caps, cv=5, solver="numpy", seed=0)

    assert _cosine(logistic.direction.vector, dim_coef) < 0.95, (
        "the fixture no longer separates the estimators; the proofs below would be vacuous"
    )
    dim_auc = _oof_auc_under(x, caps.labels, caps.groups, cv=5, seed=0)
    assert abs(logistic.held_out_auc - dim_auc) > 0.05, (
        f"out-of-fold AUCs too close to tell the estimators apart: "
        f"logistic {logistic.held_out_auc:.4f}, difference in means {dim_auc:.4f}"
    )


# ---------------------------------------------------------------------------
# Half one: the registered primary is runnable, and it is what runs
# ---------------------------------------------------------------------------


def test_the_registered_primary_estimator_can_be_run():
    """`fit_probe` asked for the registered primary returns the mean-difference direction.

    Graded against `_difference_in_means` recomputed on the same activations and labels, by signed
    cosine, so an implementation that returned the negated direction fails too.
    """
    caps = _captures()
    expected, _ = _difference_in_means(caps.features[SITE], caps.labels)

    fit = fit_probe(caps, cv=5, method="diff_in_means", solver="numpy", seed=0)

    assert _cosine(fit.direction.vector, expected) > 0.999, (
        "fit_probe asked for the registered primary did not return a difference of means"
    )


def test_the_cross_validated_folds_use_the_estimator_that_was_asked_for():
    """The fold path honours the estimator too, not only the headline vector.

    The out-of-fold AUC is produced by a second fitting site inside the module, one call per fold.
    An estimator honoured in the full fit and ignored in the folds reports a held-out number for a
    probe nobody is going to use, which is the harder half of the defect to see.
    """
    caps = _captures()
    expected_auc = _oof_auc_under(caps.features[SITE], caps.labels, caps.groups, cv=5, seed=0)

    fit = fit_probe(caps, cv=5, method="diff_in_means", solver="numpy", seed=0)

    assert fit.held_out_auc == pytest.approx(expected_auc, abs=1e-6), (
        f"the folds did not run the estimator that was asked for: reported "
        f"{fit.held_out_auc:.6f}, difference in means gives {expected_auc:.6f}"
    )


# ---------------------------------------------------------------------------
# Half two: the label is earned by the estimator that ran
# ---------------------------------------------------------------------------


def test_the_method_field_names_the_estimator_that_produced_the_vector():
    """``method`` and the vector agree, both ways round.

    The direction returned under the primary is a mean difference and says so; the direction
    returned under the default is a logistic weight and says that instead. The failure this rules
    out is a vector fitted by one estimator carrying the other's name into the id hash.
    """
    caps = _captures()
    dim_coef, _ = _difference_in_means(caps.features[SITE], caps.labels)

    primary = fit_probe(caps, cv=5, method="diff_in_means", solver="numpy", seed=0)
    secondary = fit_probe(caps, cv=5, method="probe_lr", solver="numpy", seed=0)

    assert primary.direction.method == "diff_in_means"
    assert _cosine(primary.direction.vector, dim_coef) > 0.999

    assert secondary.direction.method == "probe_lr"
    assert _cosine(secondary.direction.vector, dim_coef) < 0.95

    assert primary.direction.id != secondary.direction.id


def test_an_estimator_the_library_cannot_run_is_refused():
    """A ``method`` naming no registered estimator raises instead of being hashed into the id.

    This is the half that makes a wrong result invisible. A free string reaches `make_direction`
    and then `content_hash`, so the artifact's identity agrees with whatever the caller typed.
    """
    caps = _captures()

    with pytest.raises(ValueError, match="diff_of_means"):
        fit_probe(caps, cv=5, method="diff_of_means", solver="numpy", seed=0)

    with pytest.raises(ValueError):
        fit_probe(caps, cv=5, method="", solver="numpy", seed=0)


def test_the_default_estimator_and_its_label_are_unchanged():
    """The default path still fits logistic and still calls itself ``probe_lr``.

    A repair that renamed the default would silently move every existing `DirectionID`, since the
    method string is part of the hashed material.
    """
    caps = _captures()

    default = fit_probe(caps, cv=5, solver="numpy", seed=0)
    named = fit_probe(caps, cv=5, method="probe_lr", solver="numpy", seed=0)

    assert default.direction.method == "probe_lr"
    assert default.direction.id == named.direction.id
