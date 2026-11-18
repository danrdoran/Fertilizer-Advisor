"""Fixed physical grid, prediction caching and paired exploratory comparisons.

The original evaluator is loaded into a private namespace to substitute only
its binner. Its source and the live original module remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from ope_statistics import InferenceUnavailable, make_cluster_boot_indices_from_keys
import revision_audit as audit
import run_revision_evaluation as revision
from temporal_diagnostics import NUTRIENTS, reference_edges


POLICIES = ("recorded_rule", "guarded_greedy", "supported_argmax")
COHORTS = ("native_overlap", "within_grid_common", "cross_grid_common", "cross_grid_common_reference_range")


class ReferenceBinner:
    def __init__(self, reference):
        self.bin_edges = reference_edges(reference)
        self.bin_centers = {k: np.asarray(reference["dimensions"][k]["centers"], float) for k in NUTRIENTS}

    def fit(self, actions, outcomes, action_names):
        # The reference was fitted once using 2012; later outcomes cannot move it.
        if tuple(action_names) != NUTRIENTS:
            raise ValueError("Unexpected fertilizer dimensions.")
        return self

    def transform(self, actions, action_names):
        if tuple(action_names) != NUTRIENTS or not np.isfinite(actions).all() or (actions < 0).any():
            raise ValueError("Expected nonnegative finite N/P/K rates.")
        return np.column_stack([np.searchsorted(self.bin_edges[k][1:-1], actions[:, j], side="right") for j, k in enumerate(NUTRIENTS)])


def policy_configs(selection):
    recorded = revision.make_config(selection["spec"], selection["selected_candidate"])
    if asdict(recorded) != selection["selected_config"]:
        raise ValueError("Frozen policy configuration was modified.")
    return {"recorded_rule": recorded,
            "guarded_greedy": replace(recorded, epsilon_explore=0.0, lambda_mix=0.0),
            "supported_argmax": replace(recorded, epsilon_explore=0.0, lambda_mix=0.0, spibb_min_deviation_count=0)}


def validate_fit(fitted, year, data, cfg):
    expected_years = sorted(int(y) for y in np.unique(data["year"][data["year"] < year]))
    idx = np.where(data["year"] == year)[0]
    if (fitted["year"] != year or fitted["cf_use_eval"] or not fitted["fit_audit"] or
            any(f["fit_years"] != expected_years or f["n_fit"] != int((data["year"] < year).sum()) for f in fitted["fit_audit"])):
        raise ValueError("Fit violates the prior-years-only training rule.")
    if not np.array_equal(fitted["test_indices"], idx):
        raise ValueError("Cached prediction rows do not match the data split.")
    np.testing.assert_allclose(fitted["A_te"], data["actions"][idx], rtol=0, atol=0)
    np.testing.assert_allclose(fitted["Yy_te"], data["reward"][idx], rtol=0, atol=0)
    profit = cfg.maize_price * data["reward"][idx] - data["actions"][idx] @ np.array([cfg.priceN, cfg.priceP, cfg.priceK])
    np.testing.assert_allclose(fitted["Yp_te"], profit, rtol=1e-12, atol=1e-8)


def fit_fixed_year(year, data, cfg, reference):
    name = "_fertilizer_temporal_fixed_engine"
    module_spec = importlib.util.spec_from_file_location(name, revision.ROOT / "scripts/learn_and_evaluate_policy.py")
    private = importlib.util.module_from_spec(module_spec)
    previous = sys.modules.get(name)
    sys.modules[name] = private  # dataclass resolves annotations through sys.modules.
    try:
        module_spec.loader.exec_module(private)
        original = revision.engine()
        private.AdaptiveBinner = lambda **kwargs: ReferenceBinner(reference)
        private.StackedRewardModel = original.StackedRewardModel
        private.JointPropensityModel = original.JointPropensityModel
        profit = cfg.maize_price * data["reward"] - data["actions"] @ np.array([cfg.priceN, cfg.priceP, cfg.priceK])
        fitted = private.fit_year_nuisances(year, data["obs"], data["actions"], profit, data["reward"], data["year"], data["lat"], data["lon"], cfg)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    if fitted is None:
        raise ValueError("Fixed-grid fit needs both earlier training rows and evaluation rows.")
    validate_fit(fitted, year, data, cfg)
    return fitted


def cached_fixed_fit(year, data, cfg, reference, identity, cache_dir, *, cache_only=False):
    path = Path(cache_dir) / f"fixed_{year}.npz"
    expected = {"schema_version": 1, "year": int(year), "identity": identity,
                "fit_config": asdict(cfg), "reference": reference}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            if metadata["cache_identity"] != expected:
                raise ValueError(f"Incompatible fixed-grid cache {path}; use a new cache directory.")
            fitted = {k: saved[k] for k in metadata["array_keys"]}
            fitted.update(metadata["other_values"])
            fitted["bin_edges"] = {k: saved[f"__edges_{k}"].tolist() for k in NUTRIENTS}
        print(f"Reusing fixed-grid predictions for {year}.", flush=True)
    else:
        if cache_only:
            raise FileNotFoundError(f"No fixed-grid cache for {year}; omit --cache-only to fit it.")
        print(f"Fitting fixed grid for {year} using earlier years only; this may take substantial CPU time...", flush=True)
        fitted = fit_fixed_year(year, data, cfg, reference)
        arrays = {k: v for k, v in fitted.items() if isinstance(v, np.ndarray)}
        other = {k: v for k, v in fitted.items() if k not in arrays and k != "bin_edges"}
        metadata = {"cache_identity": expected, "array_keys": list(arrays), "other_values": other}
        arrays.update({f"__edges_{k}": np.asarray(fitted["bin_edges"][k], float) for k in NUTRIENTS})
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".fixed_{year}_", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays, metadata_json=np.array(json.dumps(revision.clean_json(metadata), allow_nan=False)))
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    validate_fit(fitted, year, data, cfg)
    binner = ReferenceBinner(reference)
    expected_grid = np.stack(np.meshgrid(*(binner.bin_centers[k] for k in NUTRIENTS), indexing="ij"), axis=-1).reshape(-1, 3)
    if not np.array_equal(fitted["grid"], expected_grid):
        raise ValueError("Cached physical actions differ from the fixed reference.")
    for k in NUTRIENTS:
        if not np.array_equal(fitted["bin_edges"][k], binner.bin_edges[k]):
            raise ValueError("Cached boundaries differ from the fixed reference.")
    return fitted, revision.sha256_file(path)


def attach_cohorts(frames):
    """Align all variants and intersect their native overlap masks by row ID."""
    variants = list(frames)
    if len(variants) != 2 * len(POLICIES):
        raise ValueError("Expected all policies on both grids.")
    baseline = frames[variants[0]][["year", "field_id", "site_key"]].reset_index(drop=True)
    if baseline[["year", "field_id"]].duplicated().any():
        raise ValueError("Duplicate evaluation row IDs.")
    for frame in frames.values():
        if not frame[["year", "field_id", "site_key"]].reset_index(drop=True).equals(baseline):
            raise ValueError("Policy comparison rows are not aligned.")
    cross = np.logical_and.reduce([f["policy_overlap"].to_numpy(bool) for f in frames.values()])
    for grid in ("adaptive", "fixed"):
        common = np.logical_and.reduce([frames[(grid, p)]["policy_overlap"].to_numpy(bool) for p in POLICIES])
        for policy in POLICIES:
            frame = frames[(grid, policy)]
            frame["native_overlap"] = frame["policy_overlap"].to_numpy(bool)
            frame["within_grid_common"] = common
            frame["cross_grid_common"] = cross
            frame["cross_grid_common_reference_range"] = cross & frame["within_reference_range"].to_numpy(bool)
    return frames


def paired_policy_contrast(left, right, method, maize_price, n_bootstrap, alpha, modes=audit.MODES):
    """Compare two policies on identical rows with identical geographic draws.

    Only within-grid contrasts are valid here: each pair must share the same
    behavior baseline and logged-action outcome model. Weights remain fixed.
    """
    kept_l, a = audit.retained_arrays(left, maize_price)
    kept_r, b = audit.retained_arrays(right, maize_price)
    if not kept_l[["year", "field_id", "site_key"]].equals(kept_r[["year", "field_id", "site_key"]]):
        raise ValueError("Paired comparisons require identical retained rows.")
    for key in ("profit", "yield", "mu_profit_logged", "mu_yield_logged", "mu_profit_pi0", "mu_yield_pi0", "cost_logged", "cost_pi0"):
        if not np.allclose(a[key], b[key], rtol=0, atol=1e-8):
            raise ValueError("Paired comparisons require the same grid and behavior baseline.")
    wa = audit.prepare_weights(a, method)
    wb = audit.prepare_weights(b, method)
    try:
        boot = make_cluster_boot_indices_from_keys(kept_l["site_key"].to_numpy(), n_bootstrap)
        reason = None
    except InferenceUnavailable as exc:
        boot, reason = [], str(exc)
    metrics = ("profit_delta", "yield_delta", "consistent_profit_delta")
    records = []
    for mode in modes:
        index = 0 if mode == audit.MODES[0] else 1
        for estimator in ("sndr", "dr"):
            info = {"mode": mode, "estimator": estimator, "n_rows": len(kept_l), "n_clusters": kept_l["site_key"].nunique(),
                    "bootstrap_replicates": n_bootstrap, "intervals_adjusted_for_multiplicity": False}
            def value(idx=None):
                x = audit.point_estimate(a, wa[index], estimator, maize_price, by_year=mode == audit.MODES[2], idx=idx)
                y = audit.point_estimate(b, wb[index], estimator, maize_price, by_year=mode == audit.MODES[2], idx=idx)
                return np.array([x[k] - y[k] for k in metrics])
            try:
                point = value()
            except InferenceUnavailable as exc:
                records.append({**info, "point_status": "unavailable", "interval_status": "unavailable", "reason": str(exc)})
                continue
            draws, failures = [], 0
            for idx in boot:
                try:
                    draws.append(value(idx))
                except InferenceUnavailable:
                    failures += 1
            available = bool(boot) and failures == 0
            record = {**info, "point_status": "estimable", "interval_status": "available" if available else "unavailable",
                      "invalid_bootstrap_replicates": failures}
            limits = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0) if available else np.full((2, len(metrics)), np.nan)
            for j, k in enumerate(metrics):
                record[k + "_contrast"] = float(point[j])
                record[k + "_contrast_ci_lo"], record[k + "_contrast_ci_hi"] = limits[:, j]
            if not available:
                record["reason"] = reason or "At least one paired bootstrap draw is undefined; no draws were discarded."
            records.append(record)
    return records
