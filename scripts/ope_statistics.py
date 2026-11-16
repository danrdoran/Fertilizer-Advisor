"""Numerical OPE helpers, independent of the expensive model-training imports."""

from __future__ import annotations

import importlib
import warnings

import numpy as np


class InferenceUnavailable(ValueError):
    """The declared sample cannot support this estimator or bootstrap interval."""


def require_psis():
    """Never silently label an unsmoothed run as PSIS."""
    try:
        az = importlib.import_module("arviz")
    except ImportError as exc:
        raise RuntimeError(
            "PSIS requires ArviZ. Install requirements-research.txt, or explicitly "
            "choose capped weights for a separately labelled analysis."
        ) from exc
    if not callable(getattr(az, "psislw", None)):
        raise RuntimeError("This ArviZ version lacks psislw; use requirements-research.txt.")
    return az


def _valid_weights(raw_weights):
    w = np.asarray(raw_weights, dtype=float)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Importance ratios must be a finite, nonnegative vector.")
    return w


def psis_khat_from_weights(raw_weights):
    w = _valid_weights(raw_weights)
    positive = w > 0
    if positive.sum() < 5:
        return float("nan")
    try:
        az = require_psis()
    except RuntimeError:
        return float("nan")  # Diagnostic only; the PSIS estimator fails closed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, k = az.psislw(np.log(w[positive]))
    return float(np.asarray(k))


def psis_smooth_weights(raw_weights):
    """Smooth positive ratios, retaining exact zeros and the original total mass.

    ArviZ returns sum-one weights. Restoring the input sum is essential for the
    ordinary DR residual mean; SNDR is invariant to a common positive scale.
    This mass-preserving convention is recorded in revision run manifests.
    """
    az = require_psis()
    w = _valid_weights(raw_weights)
    positive = w > 0
    result = np.zeros_like(w)
    if not positive.any():
        return result, float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lw, k = az.psislw(np.log(w[positive]))
    smoothed = np.exp(np.asarray(lw))
    total = smoothed.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSIS returned invalid weights.")
    result[positive] = smoothed * (w.sum() / total)
    return result, float(np.asarray(k))


def median_impute_from_training(X_train, X_test):
    """Fit medians on training rows only; retain all columns, including empty ones."""
    train = np.asarray(X_train, float).copy()
    test = np.asarray(X_test, float).copy()
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Training and test features must have matching columns.")
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Infinite context values are invalid.")
    medians = np.array([
        np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        for col in train.T
    ])
    return (np.where(np.isnan(train), medians, train),
            np.where(np.isnan(test), medians, test), medians)


def make_cluster_boot_indices_from_keys(site_keys, B, rng=None):
    if B < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    rng = rng if rng is not None else np.random.default_rng(42)
    uniq, inv = np.unique(site_keys, return_inverse=True)
    if len(uniq) < 2:
        raise InferenceUnavailable("At least two geographic clusters are required for inference.")
    per = [np.where(inv == i)[0] for i in range(len(uniq))]
    return [np.concatenate([per[i] for i in rng.choice(len(uniq), len(uniq), replace=True)])
            for _ in range(B)]


def _sndr_value(mu_pi, w, Y, mu_logged, idx):
    ww = w[idx]
    if ww.sum() <= 0:
        raise InferenceUnavailable("SNDR is undefined: this sample has no positive importance weights.")
    return float(np.mean(mu_pi[idx]) + np.sum(ww * (Y[idx] - mu_logged[idx])) / ww.sum())


def _dr_value(mu_pi, w, Y, mu_logged, idx):
    return float(np.mean(mu_pi[idx] + w[idx] * (Y[idx] - mu_logged[idx])))


def cluster_summary(site_keys):
    _, sizes = np.unique(site_keys, return_counts=True)
    return {"n_clusters": len(sizes), "n_rows": int(sizes.sum()),
            "min_size": int(sizes.min()) if len(sizes) else 0,
            "median_size": float(np.median(sizes)) if len(sizes) else 0.0,
            "max_size": int(sizes.max()) if len(sizes) else 0}
