"""Conditional numerical audits of an already frozen policy; never select a policy.

Keep this module separate from the original evaluation sources so their recorded
hashes and estimators remain reproducible. All alternative aggregations are
exploratory diagnostics after inspection of the original final-year results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ope_statistics import (
    InferenceUnavailable, cluster_summary, make_cluster_boot_indices_from_keys,
    psis_smooth_weights,
)


MODES = (
    "pooled_weights_pooled_normalization",
    "yearwise_weights_pooled_normalization",
    "yearwise_weights_yearwise_normalization",
)
ARRAY_COLUMNS = (
    "year", "profit", "yield", "mu_profit_logged", "mu_profit_pi1",
    "mu_profit_pi0", "mu_yield_logged", "mu_yield_pi1", "mu_yield_pi0",
    "cost_logged", "cost_pi1", "cost_pi0", "w_raw", "w_cap",
)
INTERVAL_METRICS = (
    "profit_delta", "yield_delta", "matched_cost_delta",
    "consistent_profit_delta", "surrogate_gap", "baseline_cost_gap",
    "naive_profit_delta",
)


def _weights(raw, capped, method):
    raw, capped = np.asarray(raw, float), np.asarray(capped, float)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Audit ratios must be finite and nonnegative.")
    if method == "psis":
        return psis_smooth_weights(raw)
    if method not in {"cap", "raw"}:
        raise ValueError("Unknown audit weight method.")
    w = capped.copy() if method == "cap" else raw.copy()
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Audit weights must be finite and nonnegative.")
    return w, float("nan")


def prepare_weights(arrays, method):
    pooled, k = _weights(arrays["w_raw"], arrays["w_cap"], method)
    yearwise = np.zeros_like(pooled)
    for year in np.unique(arrays["year"]):
        mask = arrays["year"] == year
        yearwise[mask], _ = _weights(arrays["w_raw"][mask], arrays["w_cap"][mask], method)
    return pooled, yearwise, k


def _residual_mean(weights, values, estimator):
    if estimator == "dr":
        return float(np.mean(weights * values))
    if estimator != "sndr":
        raise ValueError("Estimator must be dr or sndr.")
    if weights.sum() <= 0:
        raise InferenceUnavailable("SNDR has no positive weights in this sample/year.")
    return float(np.dot(weights, values) / weights.sum())


def _one_sample(a, weights, estimator, maize_price, idx):
    w = weights[idx]
    result = {}
    for outcome in ("profit", "yield"):
        residual = a[outcome][idx] - a[f"mu_{outcome}_logged"][idx]
        result[f"{outcome}_V1"] = float(np.mean(a[f"mu_{outcome}_pi1"][idx])) + _residual_mean(w, residual, estimator)
        result[f"{outcome}_V0"] = float(np.mean(a[f"mu_{outcome}_pi0"][idx]) + np.mean(residual))
        result[f"{outcome}_delta"] = result[f"{outcome}_V1"] - result[f"{outcome}_V0"]
    result["cost_V1"] = float(np.mean(a["cost_pi1"][idx]))
    result["cost_V0"] = float(np.mean(a["cost_pi0"][idx]))
    result["matched_cost_delta"] = result["cost_V1"] - result["cost_V0"]
    result["consistent_profit_delta"] = maize_price * result["yield_delta"] - result["matched_cost_delta"]
    result["surrogate_gap"] = result["profit_delta"] - result["consistent_profit_delta"]
    result["baseline_cost_gap"] = float(np.mean(a["cost_pi0"][idx] - a["cost_logged"][idx]))
    result["naive_profit_delta"] = maize_price * result["yield_delta"] - float(np.mean(a["cost_pi1"][idx] - a["cost_logged"][idx]))
    gaps = {
        suffix: a[f"mu_profit_{suffix}"][idx] - (
            maize_price * a[f"mu_yield_{suffix}"][idx] - a[f"cost_{suffix}"][idx]
        ) for suffix in ("logged", "pi1", "pi0")
    }
    result["gap_direct"] = float(np.mean(gaps["pi1"] - gaps["pi0"]))
    result["gap_residual"] = float(np.mean(gaps["logged"])) - _residual_mean(w, gaps["logged"], estimator)
    result["gap_identity_error"] = result["surrogate_gap"] - result["gap_direct"] - result["gap_residual"]
    result["naive_identity_error"] = (
        result["profit_delta"] - result["naive_profit_delta"]
        - result["surrogate_gap"] - result["baseline_cost_gap"]
    )
    return result


def point_estimate(arrays, weights, estimator, maize_price, *, by_year=False, idx=None):
    """Yearwise mode averages year-specific values by their sampled row counts.

    In bootstrap samples the year shares vary with the resampled geographic
    clusters. Years with no sampled rows contribute zero mass. Weights are held
    fixed, matching the original conditional bootstrap convention.
    """
    idx = np.arange(len(weights)) if idx is None else np.asarray(idx, int)
    if len(idx) == 0:
        raise InferenceUnavailable("No retained rows in this sample.")
    if not by_year:
        return _one_sample(arrays, weights, estimator, maize_price, idx)
    result = None
    for year in np.unique(arrays["year"][idx]):
        group = idx[arrays["year"][idx] == year]
        values = _one_sample(arrays, weights, estimator, maize_price, group)
        if result is None:
            result = {k: 0.0 for k in values}
        for key, value in values.items():
            result[key] += len(group) / len(idx) * value
    return result


def retained_arrays(rows, maize_price):
    kept = rows.loc[rows["included_cohort"]].reset_index(drop=True)
    if kept.empty:
        raise InferenceUnavailable("No retained rows in the audit cohort.")
    arrays = {key: kept[key].to_numpy(dtype=float) for key in ARRAY_COLUMNS}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite prediction, outcome, cost or ratio in audit rows.")
    if not np.allclose(arrays["profit"], maize_price * arrays["yield"] - arrays["cost_logged"], rtol=1e-10, atol=1e-6):
        raise ValueError("Observed profit does not equal maize revenue minus logged fertilizer cost.")
    return kept, arrays


def audit_estimates(rows, method, maize_price, n_bootstrap, alpha, modes=MODES):
    """Return matched profit/yield/cost estimates and paired cluster intervals.

    A failed bootstrap draw is not silently discarded: any non-estimable draw
    makes that interval unavailable, while preserving an estimable point value.
    """
    if n_bootstrap < 2 or not 0 < alpha < 1:
        raise ValueError("Invalid audit bootstrap settings.")
    kept, a = retained_arrays(rows, maize_price)
    pooled, yearwise, k = prepare_weights(a, method)
    clusters = cluster_summary(kept["site_key"].to_numpy())
    try:
        boot = make_cluster_boot_indices_from_keys(kept["site_key"].to_numpy(), n_bootstrap)
        bootstrap_reason = None
    except InferenceUnavailable as exc:
        boot, bootstrap_reason = [], str(exc)
    records = []
    for mode in modes:
        if mode not in MODES:
            raise ValueError("Unknown aggregation mode.")
        w = pooled if mode == MODES[0] else yearwise
        for estimator in ("sndr", "dr"):
            record = {
                "mode": mode, "estimator": estimator, "n_rows": len(kept),
                "n_clusters": clusters["n_clusters"], "weight_method": method,
                "psis_k_pooled": k if mode == MODES[0] else float("nan"),
                "bootstrap_replicates": n_bootstrap,
            }
            try:
                record.update(point_estimate(a, w, estimator, maize_price, by_year=mode == MODES[2]))
            except InferenceUnavailable as exc:
                records.append({**record, "point_status": "unavailable", "interval_status": "unavailable", "reason": str(exc)})
                continue
            record["point_status"] = "estimable"
            draws, failed = [], 0
            for idx in boot:
                try:
                    value = point_estimate(a, w, estimator, maize_price, by_year=mode == MODES[2], idx=idx)
                    draws.append([value[key] for key in INTERVAL_METRICS])
                except InferenceUnavailable:
                    failed += 1
            record["invalid_bootstrap_replicates"] = failed
            record["interval_status"] = "available" if boot and failed == 0 else "unavailable"
            if record["interval_status"] == "available":
                lo, hi = np.quantile(np.asarray(draws), [alpha / 2, 1 - alpha / 2], axis=0)
                for i, key in enumerate(INTERVAL_METRICS):
                    record[key + "_ci_lo"], record[key + "_ci_hi"] = float(lo[i]), float(hi[i])
            else:
                record["reason"] = bootstrap_reason or "At least one bootstrap sample has no positive weights in a required year."
                for key in INTERVAL_METRICS:
                    record[key + "_ci_lo"] = record[key + "_ci_hi"] = float("nan")
            if abs(record["gap_identity_error"]) > 1e-6 or abs(record["naive_identity_error"]) > 1e-6:
                raise ValueError("Audit accounting decomposition failed its algebraic identity.")
            records.append(record)
    return records


def aggregation_decomposition(rows, method, maize_price):
    """Attribute pooled SNDR differences to PSIS smoothing and normalization.

    Contributions, unlike the annual point values themselves, sum exactly to
    the corresponding aggregate SNDR profit improvement.
    """
    kept, a = retained_arrays(rows, maize_price)
    pooled, yearwise, _ = prepare_weights(a, method)
    records = []
    for year in np.unique(a["year"]):
        mask = a["year"] == year
        n, share = int(mask.sum()), float(mask.mean())
        residual = a["profit"][mask] - a["mu_profit_logged"][mask]
        direct = float(np.mean(a["mu_profit_pi1"][mask] - a["mu_profit_pi0"][mask]))
        baseline_residual = float(np.mean(residual))
        item = {
            "year": int(year), "n_rows": n, "row_share": share,
            "direct_advantage": direct, "baseline_residual_mean": baseline_residual,
            "raw_weight_mean": float(a["w_raw"][mask].mean()),
            "n_clusters": int(kept.loc[mask, "site_key"].nunique()),
        }
        _, item["psis_k_within_year"] = _weights(a["w_raw"][mask], a["w_cap"][mask], method)
        for label, w in (("pooled", pooled), ("yearwise", yearwise)):
            local, total = float(w[mask].sum()), float(w.sum())
            item[label + "_weight_sum"] = local
            item[label + "_weight_share"] = local / total if total else float("nan")
            item[label + "_weight_ess"] = local ** 2 / float(np.dot(w[mask], w[mask])) if local else 0.0
            weighted = float(np.dot(w[mask], residual))
            item[label + "_residual_mean"] = weighted / local if local else float("nan")
            item[label + "_global_contribution"] = share * (direct - baseline_residual) + (weighted / total if total else float("nan"))
        item["yearwise_local_contribution"] = share * (direct - baseline_residual + item["yearwise_residual_mean"])
        item["smoothing_shift_contribution"] = item["yearwise_global_contribution"] - item["pooled_global_contribution"]
        item["normalization_shift_contribution"] = item["yearwise_local_contribution"] - item["yearwise_global_contribution"]
        records.append(item)
    return records


def rows_from_fit(fitted, block, cfg, data, candidate, stage):
    """Probe known cost through the original policy construction, without cloning it.

    The policy depends on profit predictions only. Replacing the auxiliary yield
    predictions with exact fertilizer costs returns E_pi1[cost] and E_pi0[cost]
    under precisely the same policy, grid and support rules. No model is refitted.
    """
    import learn_and_evaluate_policy as engine

    prices = np.array([cfg.priceN, cfg.priceP, cfg.priceK], float)
    probe_fit = dict(fitted)
    probe_fit["mu_grid_yield"] = np.broadcast_to(fitted["grid"] @ prices, fitted["mu_grid_yield"].shape)
    probe_fit["mu_logged_yield"] = fitted["A_te"] @ prices
    probe = engine.apply_policy_to_year(probe_fit, cfg)
    for key in ("w_raw", "mu_pi1", "mu_pi0", "pi1_logged", "keep_mask"):
        np.testing.assert_allclose(probe[key], block[key], rtol=0, atol=0)
    idx = block["test_indices"]
    result = pd.DataFrame({
        "candidate": candidate, "stage": stage, "year": int(block["year"]),
        "field_id": data["field_id"][idx], "site_key": block["site_keys"],
        "policy_overlap": block["keep_mask"], "included_cohort": block["keep_mask"],
        "profit": block["Y"], "yield": block["Y_yield"],
        "mu_profit_logged": block["mu_logged"], "mu_profit_pi1": block["mu_pi1"],
        "mu_profit_pi0": block["mu_pi0"], "mu_yield_logged": block["mu_logged_yield"],
        "mu_yield_pi1": block["mu_pi1_yield"], "mu_yield_pi0": block["mu_pi0_yield"],
        "cost_logged": fitted["A_te"] @ prices,
        "cost_pi1": probe["mu_pi1_yield"], "cost_pi0": probe["mu_pi0_yield"],
        "w_raw": block["w_raw"], "w_cap": block["w_cap"],
        "pi0_logged": block["pi0_logged"], "pi1_logged": block["pi1_logged"],
        "count_logged": block["count_logged"], "count_target": block["count_target"],
    })
    return result
