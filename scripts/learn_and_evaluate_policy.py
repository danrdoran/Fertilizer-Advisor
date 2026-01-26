"""
What this script does:

Runs by-year offline policy evaluation for a contextual bandit fertilizer policy.
For each evaluation year, it trains reward and behavior-policy (π0) models on prior years, discretizes
N–P₂O₅–K₂O actions into supported bins, constructs a safe target policy π1 (greedy on predicted
yield/profit with optional ε-exploration and π0-mixing, constrained by support and SPIBB-style rules),
and estimates π1 vs π0 using self-normalized doubly robust (SNDR) and doubly robust (DR) OPE with
overlap trimming and PSIS-smoothed importance weights. Outputs per-year and pooled summaries
(JSON/CSV) and optional per-row policy/weight diagnostics.
"""

from __future__ import annotations

import os, json, argparse, warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

try:
    import arviz as az  # PSIS
    _HAS_ARVIZ = True
except Exception:
    _HAS_ARVIZ = False

from bandit_models import (
    SEED,
    set_random_seeds,
    make_recency_weights,
    AdaptiveBinner,
    StackedRewardModel,
    JointPropensityModel,
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# --------------------------- Config ---------------------------

@dataclass
class OPEConfig:
    # Policy construction (default: no exploration, no pi0 mixing)
    epsilon_explore: float = 0.0
    lambda_mix: float = 0.0
    min_support_count: int = 30

    # Safe deviation (SPIBB-style)
    spibb_min_deviation_count: int = 60

    # Ratio cap for non-PSIS (unused if weights='psis')
    max_density_ratio: float = 50.0

    # Overlap trimming (OVERLAP cohort definition)
    trim_enable: bool = True
    trim_pi0_tau: float = 0.01
    trim_ratio_max: float = 10.0
    trim_min_count_logged: int = 30
    trim_min_count_target: int = 30

    # Cross-fitting and recency
    recency_lambda: float = 0.10
    cf_k: int = 5
    cf_use_eval: bool = True

    # Bootstrap
    n_bootstrap: int = 1000
    alpha: float = 0.05

    # Decision
    noninferiority_epsilon: float = 0.0

    # Objective (controls how reward Y is formed; Y_raw is always yield)
    objective: str = "yield"
    maize_price: Optional[float] = None
    priceN: Optional[float] = None
    priceP: Optional[float] = None
    priceK: Optional[float] = None

# ------------------------- Utilities --------------------------

def geo_site_keys(lat: np.ndarray, lon: np.ndarray, decimals: int = 2) -> np.ndarray:
    latr = np.round(lat.astype(float), decimals)
    lonr = np.round(lon.astype(float), decimals)
    return np.array([f"{la:.{decimals}f},{lo:.{decimals}f}" for la, lo in zip(latr, lonr)])

def ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    s1 = float(np.sum(w)); s2 = float(np.sum(w ** 2))
    return (s1 * s1) / (s2 + 1e-12)

# --- PSIS helpers ---

def psis_khat_from_weights(raw_weights: np.ndarray) -> float:
    w = np.asarray(raw_weights, float)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size < 5 or not _HAS_ARVIZ:
        return float("nan")
    try:
        _, k = az.psislw(np.log(w))
        return float(np.asarray(k).squeeze())
    except Exception:
        return float("nan")

def psis_smooth_weights(raw_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    w = np.asarray(raw_weights, float)
    w = np.clip(w, 1e-300, np.inf)
    if not _HAS_ARVIZ:
        return w, float("nan")
    lw_smooth, k = az.psislw(np.log(w))
    return np.exp(np.asarray(lw_smooth)), float(np.asarray(k))

# Bootstrap cluster indices by site -------------------------------------------

def make_cluster_boot_indices_from_keys(site_keys: np.ndarray, B: int, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    rng = rng or np.random.default_rng(SEED)
    uniq, inv = np.unique(site_keys, return_inverse=True)
    per = {i: np.where(inv == i)[0] for i in range(len(uniq))}
    idxs = []
    for _ in range(B):
        draw = rng.choice(len(uniq), size=len(uniq), replace=True)
        idxs.append(np.concatenate([per[i] for i in draw]))
    return idxs

# Estimators ------------------------------------------------------------------

def _sndr_value(mu_pi: np.ndarray, w: np.ndarray, Y: np.ndarray, mu_logged: np.ndarray, idx: np.ndarray) -> float:
    ww = w[idx]
    return float(np.mean(mu_pi[idx]) + (np.sum(ww * (Y[idx] - mu_logged[idx])) / (np.sum(ww) + 1e-12)))

def _dr_value(mu_pi: np.ndarray, w: np.ndarray, Y: np.ndarray, mu_logged: np.ndarray, idx: np.ndarray) -> float:
    ww = w[idx]
    return float(np.mean(mu_pi[idx] + ww * (Y[idx] - mu_logged[idx])))

# ----------------------- Core evaluation ----------------------

def evaluate_year(
    year: int,
    X: np.ndarray,
    A: np.ndarray,
    Y_profit: np.ndarray,      # reward used to learn π1 and for profit OPE
    Y_yield: np.ndarray,       # raw yield, for yield OPE under same π1
    years: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    cfg: OPEConfig,
) -> Optional[dict]:
    """Train nuisances on ≤(year-1); evaluate policy and diagnostics on ==year.
       Returns per-row arrays for export."""
    test_mask  = (years == year)
    train_mask = (years <  year)
    if not np.any(test_mask) or not np.any(train_mask):
        return None

    test_indices = np.where(test_mask)[0]

    X_tr, A_tr, Yp_tr, Yy_tr = X[train_mask], A[train_mask], Y_profit[train_mask], Y_yield[train_mask]
    X_te, A_te, Yp_te, Yy_te = X[test_mask],  A[test_mask],  Y_profit[test_mask],  Y_yield[test_mask]
    lat_tr, lon_tr   = lat[train_mask], lon[train_mask]
    lat_te, lon_te   = lat[test_mask],  lon[test_mask]
    years_tr         = years[train_mask]

    # Train-time binner & support (use PROFIT target for choosing bins/support)
    binner = AdaptiveBinner(min_samples_per_bin=max(cfg.min_support_count, 5),
                            max_bins_per_dim={'N': 6, 'P': 3, 'K': 2})
    binner.fit(A_tr, Yp_tr, action_names=['N', 'P', 'K'])
    centers = binner.bin_centers

    def _grid_from_centers(centers: Dict[str, np.ndarray]):
        cN, cP, cK = centers['N'], centers['P'], centers['K']
        grid = np.stack(np.meshgrid(cN, cP, cK, indexing='ij'), axis=-1).reshape(-1, 3)
        return grid, (len(cN), len(cP), len(cK)), cN, cP, cK

    grid, n_bins, cN, cP, cK = _grid_from_centers(centers)
    A_count = int(np.prod(n_bins))

    def _joint_index(idxN: np.ndarray, idxP: np.ndarray, idxK: np.ndarray, n_bins: Tuple[int,int,int]) -> np.ndarray:
        return idxN * (n_bins[1] * n_bins[2]) + idxP * n_bins[2] + idxK

    AB_tr = binner.transform(A_tr, action_names=['N', 'P', 'K'])
    AB_te = binner.transform(A_te, action_names=['N', 'P', 'K'])
    joint_tr = _joint_index(AB_tr[:,0], AB_tr[:,1], AB_tr[:,2], n_bins)
    joint_te = _joint_index(AB_te[:,0], AB_te[:,1], AB_te[:,2], n_bins)
    counts = np.bincount(joint_tr, minlength=A_count)

    support_mask = (counts >= cfg.min_support_count)

    # Nuisances with geo-based CF
    n_te = X_te.shape[0]
    pi0_grid_all   = np.zeros((n_te, A_count), float)

    # PROFIT reward model (for policy + profit OPE)
    mu_grid_profit   = np.zeros((n_te, A_count), float)
    mu_logged_profit = np.zeros(n_te, float)

    # YIELD reward model (for yield OPE under same π1)
    mu_grid_yield    = np.zeros((n_te, A_count), float)
    mu_logged_yield  = np.zeros(n_te, float)

    site_keys_te = geo_site_keys(lat_te, lon_te, decimals=2)
    uniq_keys = np.unique(site_keys_te)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq_keys)
    buckets = np.array_split(uniq_keys, max(1, cfg.cf_k))
    folds = [np.where(np.isin(site_keys_te, b))[0] for b in buckets]

    for test_idx in folds:
        if cfg.cf_use_eval and len(folds) > 1:
            oof = np.setdiff1d(np.arange(n_te), test_idx, assume_unique=True)
            X_fit = np.concatenate([X_tr, X_te[oof]], axis=0)
            A_fit = np.concatenate([A_tr, A_te[oof]], axis=0)
            Yp_fit = np.concatenate([Yp_tr, Yp_te[oof]], axis=0)
            Yy_fit = np.concatenate([Yy_tr, Yy_te[oof]], axis=0)
            years_fit = np.concatenate([years_tr, np.full(len(oof), year, int)])
            sw = make_recency_weights(years_fit, cfg.recency_lambda)
        else:
            X_fit, A_fit, Yp_fit, Yy_fit = X_tr, A_tr, Yp_tr, Yy_tr
            sw = make_recency_weights(years_tr, cfg.recency_lambda)

        # Fit nuisances
        rew_profit = StackedRewardModel(); rew_profit.fit(X_fit, A_fit, Yp_fit, sample_weight=sw)
        rew_yield  = StackedRewardModel(); rew_yield.fit( X_fit, A_fit, Yy_fit, sample_weight=sw)
        pi0 = JointPropensityModel();     pi0.fit(       X_fit, A_fit, bin_edges=binner.bin_edges)

        te_rows = test_idx

        def _predict_mu_for_rows(model: StackedRewardModel, X_rows: np.ndarray) -> np.ndarray:
            mu = np.zeros((len(X_rows), A_count), float)
            for j, a in enumerate(grid):
                mu[:, j] = model.predict(X_rows, np.tile(a, (len(X_rows), 1)))
            return mu

        # PROFIT nuisances
        mu_grid_profit[te_rows, :]   = _predict_mu_for_rows(rew_profit, X_te[te_rows])
        mu_logged_profit[te_rows]    = rew_profit.predict(X_te[te_rows], A_te[te_rows])

        # YIELD nuisances
        mu_grid_yield[te_rows, :]    = _predict_mu_for_rows(rew_yield, X_te[te_rows])
        mu_logged_yield[te_rows]     = rew_yield.predict(X_te[te_rows], A_te[te_rows])

        # Shared π0 grid
        pi0_grid_all[te_rows, :]     = pi0.predict_proba_grid(X_te[te_rows])

    # Build π1 with support + SPIBB guard (using PROFIT expectations)
    def _make_pi1(mu_row: np.ndarray, pi0_row: np.ndarray) -> Tuple[np.ndarray, int]:
        ok = support_mask & (pi0_row > 0)
        if not np.any(ok):
            return pi0_row.copy(), int(np.argmax(pi0_row))
        best = int(np.argmax(np.where(ok, mu_row, -np.inf)))
        if cfg.spibb_min_deviation_count > 0 and counts[best] < cfg.spibb_min_deviation_count:
            base = pi0_row.copy()
            base[~ok] = 0.0
            s = base.sum(); base = base / s if s > 0 else np.where(ok, 1.0/ok.sum(), 0.0)
        else:
            base = np.zeros_like(mu_row); base[best] = 1.0
        support_u = np.where(ok, 1.0 / max(ok.sum(), 1), 0.0)
        pi1 = (1.0 - cfg.epsilon_explore) * base + cfg.epsilon_explore * support_u
        pi0_masked = np.where(ok, pi0_row, 0.0)
        s = pi0_masked.sum(); pi0_masked = pi0_masked / s if s > 0 else support_u
        pi1 = (1.0 - cfg.lambda_mix) * pi1 + cfg.lambda_mix * pi0_masked
        z = pi1.sum();  pi1 = pi1 / z if z > 0 else support_u
        return pi1, best

    n_te = X_te.shape[0]
    pi1_grid_all = np.zeros_like(pi0_grid_all)
    target_best_idx = np.zeros(n_te, dtype=int)
    for i in range(n_te):
        pi1_grid_all[i], target_best_idx[i] = _make_pi1(mu_grid_profit[i], pi0_grid_all[i])

    # Expectations (profit and yield) under π1/π0
    mu_pi1_profit = (mu_grid_profit * pi1_grid_all).sum(axis=1)
    mu_pi0_profit = (mu_grid_profit * pi0_grid_all).sum(axis=1)

    mu_pi1_yield = (mu_grid_yield *  pi1_grid_all).sum(axis=1)
    mu_pi0_yield = (mu_grid_yield *  pi0_grid_all).sum(axis=1)

    # Ratios on logged action (weights are tied to the policy, not the outcome)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_full_raw = np.where(pi0_grid_all > 0, pi1_grid_all / pi0_grid_all, 0.0)

    AB_te = binner.transform(A_te, action_names=['N','P','K'])
    def _joint_index_from_AB(AB: np.ndarray, n_bins: Tuple[int,int,int]) -> np.ndarray:
        return AB[:,0] * (n_bins[1] * n_bins[2]) + AB[:,1] * n_bins[2] + AB[:,2]
    joint_te_idx = _joint_index_from_AB(AB_te, n_bins)

    ratio_logged_raw = ratio_full_raw[np.arange(n_te), joint_te_idx]
    ratio_logged_cap = np.clip(ratio_logged_raw, 0.0, cfg.max_density_ratio)

    # Overlap trimming mask
    pi0_logged = pi0_grid_all[np.arange(n_te), joint_te_idx]
    pi1_logged = pi1_grid_all[np.arange(n_te), joint_te_idx]
    count_logged = counts[joint_te_idx]
    count_target = counts[target_best_idx]

    keep_mask = (
        (pi0_logged >= cfg.trim_pi0_tau) &
        (ratio_logged_raw <= cfg.trim_ratio_max) &
        (count_logged >= cfg.trim_min_count_logged) &
        (count_target >= cfg.trim_min_count_target)
    ) if cfg.trim_enable else np.ones(n_te, dtype=bool)

    # ---------- Fertilizer diagnostics ----------
    # Pre-trim (descriptive, all test rows)
    pi1_exp_actions = pi1_grid_all @ grid  # [n_te, 3] columns: N,P,K
    fert_allrows = dict(
        avg_logged={'N': float(A_te[:,0].mean()), 'P': float(A_te[:,1].mean()), 'K': float(A_te[:,2].mean())},
        avg_pi1   ={'N': float(pi1_exp_actions[:,0].mean()),
                    'P': float(pi1_exp_actions[:,1].mean()),
                    'K': float(pi1_exp_actions[:,2].mean())},
        delta_pi1_minus_logged={'N': float(pi1_exp_actions[:,0].mean() - A_te[:,0].mean()),
                                'P': float(pi1_exp_actions[:,1].mean() - A_te[:,1].mean()),
                                'K': float(pi1_exp_actions[:,2].mean() - A_te[:,2].mean())},
    )
    # Apples-to-apples on the OPE cohort (overlap if trim enabled; else full)
    mask = keep_mask if cfg.trim_enable else np.ones_like(keep_mask, bool)
    if mask.any():
        avg_log = {'N': float(A_te[mask,0].mean()), 'P': float(A_te[mask,1].mean()), 'K': float(A_te[mask,2].mean())}
        avg_pi1 = {'N': float(pi1_exp_actions[mask,0].mean()),
                   'P': float(pi1_exp_actions[mask,1].mean()),
                   'K': float(pi1_exp_actions[mask,2].mean())}
        fert_on_cohort = dict(
            avg_logged=avg_log,
            avg_pi1=avg_pi1,
            delta_pi1_minus_logged={
                'N': float(avg_pi1['N'] - avg_log['N']),
                'P': float(avg_pi1['P'] - avg_log['P']),
                'K': float(avg_pi1['K'] - avg_log['K']),
            },
        )
    else:
        fert_on_cohort = dict(
            avg_logged={'N': np.nan, 'P': np.nan, 'K': np.nan},
            avg_pi1={'N': np.nan, 'P': np.nan, 'K': np.nan},
            delta_pi1_minus_logged={'N': np.nan, 'P': np.nan, 'K': np.nan},
        )

    year_details = dict(
        year=int(year),
        test_indices=test_indices,
        joint_logged=joint_te_idx.astype(int),
        joint_target=target_best_idx.astype(int),
        pi0_logged=pi0_logged.astype(float),
        pi1_logged=pi1_logged.astype(float),
        ratio_logged_raw=ratio_logged_raw.astype(float),
        count_logged=count_logged.astype(int),
        count_target=count_target.astype(int),
        keep_mask=keep_mask.astype(bool),

        site_keys=geo_site_keys(lat_te, lon_te, decimals=2),

        # PROFIT outcome (legacy behavior)
        Y=Yp_te.astype(float),
        mu_logged=mu_logged_profit.astype(float),
        mu_pi1=mu_pi1_profit.astype(float),
        mu_pi0=mu_pi0_profit.astype(float),

        # YIELD outcome (NEW)
        Y_yield=Yy_te.astype(float),
        mu_logged_yield=mu_logged_yield.astype(float),
        mu_pi1_yield=mu_pi1_yield.astype(float),
        mu_pi0_yield=mu_pi0_yield.astype(float),

        # weights
        w_raw=ratio_logged_raw.astype(float),
        w_cap=ratio_logged_cap.astype(float),

        diagnostics=dict(
            mean_ratio=float(np.mean(ratio_logged_cap)),
            ess_ratio=float(ess(ratio_logged_cap) / max(n_te, 1)),
            cap_share=float(np.mean(ratio_logged_cap >= (cfg.max_density_ratio - 1e-12))),
            supported_frac=float(np.mean((count_logged >= cfg.min_support_count))),
            mass_in_support=float(np.mean((pi1_grid_all[:, (counts >= cfg.min_support_count)]).sum(axis=1))),
            n_eval=int(n_te),
            n_in_cohort=int(mask.sum()),
            retained_frac=float(np.mean(keep_mask)),
            psis_k_raw=psis_khat_from_weights(ratio_logged_raw),

            # Fertilizer diagnostics
            fertilizer_allrows=fert_allrows,      # descriptive (pre-trim)
            fertilizer_on_cohort=fert_on_cohort,  # apples-to-apples (OPE cohort)
        ),
    )
    return year_details

# ------------------- Pooled stats and bootstrap ----------------

def _sndr_or_dr(Y, mu_logged, mu_pi1, mu_pi0, w_raw, w_cap, site_keys, subset_mask, est, use_psis, n_bootstrap, alpha):
    idx_all = np.where(subset_mask)[0]
    if idx_all.size == 0:
        return dict(estimator=est, V1=np.nan, V0=np.nan, delta=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, lcb=np.nan,
                    uplift_pct=np.nan, uplift_ci_lo=np.nan, uplift_ci_hi=np.nan,
                    uplift_lcb=np.nan, psis_k_raw=np.nan, psis_k_subset=np.nan)
    if use_psis:
        w_use, k_subset = psis_smooth_weights(w_raw[idx_all])
        w = np.zeros_like(w_raw); w[idx_all] = w_use
    else:
        w = w_cap
        k_subset = psis_khat_from_weights(w_raw[idx_all])
    v_fn = _sndr_value if est == 'sndr' else _dr_value
    V1_hat = v_fn(mu_pi1, w, Y, mu_logged, idx_all)
    V0_hat = v_fn(mu_pi0, np.ones_like(w), Y, mu_logged, idx_all)
    Delta_hat = V1_hat - V0_hat
    Uplift_pct = Delta_hat / (V0_hat + 1e-12)
    # Cluster bootstrap by site
    boot_indices = make_cluster_boot_indices_from_keys(site_keys[subset_mask], n_bootstrap)
    deltas = np.empty(n_bootstrap, float); uplifts = np.empty(n_bootstrap, float)
    for b, local_idx in enumerate(boot_indices):
        idx = idx_all[local_idx]
        V1_b = v_fn(mu_pi1, w, Y, mu_logged, idx)
        V0_b = v_fn(mu_pi0, np.ones_like(w), Y, mu_logged, idx)
        d_b = V1_b - V0_b
        deltas[b] = d_b
        uplifts[b] = d_b / (V0_b + 1e-12)
    lo_d, hi_d = np.quantile(deltas, [alpha/2, 1 - alpha/2])
    lo_r, hi_r = np.quantile(uplifts, [alpha/2, 1 - alpha/2])
    return dict(estimator=est, V1=V1_hat, V0=V0_hat, delta=Delta_hat,
                ci_lo=lo_d, ci_hi=hi_d, lcb=lo_d,
                uplift_pct=Uplift_pct, uplift_ci_lo=lo_r, uplift_ci_hi=hi_r,
                uplift_lcb=lo_r, psis_k_subset=k_subset, psis_k_raw=psis_khat_from_weights(w_raw[idx_all]))

def pooled_profit(details_list, subset_mask, est, use_psis, n_bootstrap, alpha):
    site_keys = np.concatenate([d['site_keys'] for d in details_list])
    Y = np.concatenate([d['Y'] for d in details_list])
    mu_logged = np.concatenate([d['mu_logged'] for d in details_list])
    mu_pi1 = np.concatenate([d['mu_pi1'] for d in details_list])
    mu_pi0 = np.concatenate([d['mu_pi0'] for d in details_list])
    w_raw = np.concatenate([d['w_raw'] for d in details_list])
    w_cap = np.concatenate([d['w_cap'] for d in details_list])
    return _sndr_or_dr(Y, mu_logged, mu_pi1, mu_pi0, w_raw, w_cap, site_keys,
                       subset_mask, est, use_psis, n_bootstrap, alpha)

def pooled_yield(details_list, subset_mask, est, use_psis, n_bootstrap, alpha):
    site_keys = np.concatenate([d['site_keys'] for d in details_list])
    Y = np.concatenate([d['Y_yield'] for d in details_list])
    mu_logged = np.concatenate([d['mu_logged_yield'] for d in details_list])
    mu_pi1 = np.concatenate([d['mu_pi1_yield'] for d in details_list])
    mu_pi0 = np.concatenate([d['mu_pi0_yield'] for d in details_list])
    w_raw = np.concatenate([d['w_raw'] for d in details_list])
    w_cap = np.concatenate([d['w_cap'] for d in details_list])
    return _sndr_or_dr(Y, mu_logged, mu_pi1, mu_pi0, w_raw, w_cap, site_keys,
                       subset_mask, est, use_psis, n_bootstrap, alpha)

# ----------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/processed/processed_data.npz')
    p.add_argument('--objective', type=str, default='yield', choices=['yield','profit'])
    p.add_argument('--maize_price', type=float, default=None)
    p.add_argument('--priceN', type=float, default=None)
    p.add_argument('--priceP', type=float, default=None)
    p.add_argument('--priceK', type=float, default=None)

    p.add_argument('--out_json', type=str, default='results/ope_by_year_result.json')
    p.add_argument('--out_csv', type=str, default='results/ope_by_year_summary.csv')
    p.add_argument('--by_year_csv', type=str, default='results/ope_by_year.csv')
    p.add_argument('--out_policy_csv', type=str, default='',
                   help='Optional: path to write per-row policy assignments and overlap mask')

    # Base knobs (match defaults in prior runs)
    p.add_argument('--recency_lambda', type=float, default=0.10)
    p.add_argument('--cf_k', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_bootstrap', type=int, default=1000)
    p.add_argument('--noninferiority_epsilon', type=float, default=0.0)

    # Policy knobs (no exploration, no lambda mixing by default)
    p.add_argument('--epsilon_explore', type=float, default=0.0)
    p.add_argument('--lambda_mix', type=float, default=0.0)

    # Overlap/SPIBB
    p.add_argument('--trim_enable', action='store_true')
    p.add_argument('--trim_pi0_tau', type=float, default=0.01)
    p.add_argument('--trim_ratio_max', type=float, default=10.0)
    p.add_argument('--trim_min_count_logged', type=int, default=30)
    p.add_argument('--trim_min_count_target', type=int, default=30)
    p.add_argument('--spibb_min_deviation_count', type=int, default=60)

    # Choice of cohort + weight type
    p.add_argument('--cohort', type=str, choices=['overlap','full'], default='overlap',
                   help='Which cohort to evaluate (default overlap = trimmed).')
    p.add_argument('--weights', type=str, choices=['psis','cap'], default='psis',
                   help='Use PSIS smoothing (default) or capped weights (legacy).')
    
    p.add_argument(
        '--out_weights_csv',
        type=str,
        default='',
        help='Optional: write per-row importance weights (raw, capped) and overlap mask'
    )

    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    set_random_seeds(SEED)

    # Load processed data
    data = np.load(args.data, allow_pickle=True)
    X = data['obs']
    A = data['actions']    # columns: [N, P, K]
    Y_raw = data['reward'] # <-- raw yield
    years = data['year'].astype(int)
    lat = data['lat'] if 'lat' in data else data['Lat']
    lon = data['lon'] if 'lon' in data else data['Long']
    field_id = data['Field_ID'] if 'Field_ID' in data else None

    # Build rewards for policy-learning and OPE
    if args.objective == 'profit':
        if None in (args.maize_price, args.priceN, args.priceP, args.priceK):
            raise ValueError('For objective=profit, provide --maize_price --priceN --priceP --priceK')
        fert_cost = (args.priceN * A[:,0]) + (args.priceP * A[:,1]) + (args.priceK * A[:,2])
        Y_profit = args.maize_price * Y_raw - fert_cost   # used to learn π1 and profit OPE
        Y_yield  = Y_raw.astype(float)                     # used for yield OPE under same π1
    else:
        # If objective==yield, π1 is yield-optimized; keep parity anyway
        Y_profit = (args.maize_price * Y_raw - ((args.priceN or 0.0)*A[:,0]
                                               + (args.priceP or 0.0)*A[:,1]
                                               + (args.priceK or 0.0)*A[:,2])) if args.maize_price is not None else Y_raw*0.0
        Y_yield  = Y_raw.astype(float)

    # Years: forward validation (train ≤ y-1, eval y)
    unique_years = np.sort(np.unique(years))
    eval_years = [y for y in unique_years if np.any(years < y) and np.any(years == y)]

    cfg = OPEConfig(
        epsilon_explore=args.epsilon_explore,
        lambda_mix=args.lambda_mix,
        recency_lambda=args.recency_lambda,
        cf_k=args.cf_k,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        noninferiority_epsilon=args.noninferiority_epsilon,
        objective=args.objective,
        maize_price=args.maize_price,
        priceN=args.priceN, priceP=args.priceP, priceK=args.priceK,
        trim_enable=bool(args.trim_enable),
        trim_pi0_tau=args.trim_pi0_tau,
        trim_ratio_max=args.trim_ratio_max,
        trim_min_count_logged=args.trim_min_count_logged,
        trim_min_count_target=args.trim_min_count_target,
        spibb_min_deviation_count=args.spibb_min_deviation_count,
    )

    # Evaluate each year
    year_blocks: List[dict] = []
    per_row_frames: List[pd.DataFrame] = []

    for y in eval_years:
        det = evaluate_year(y, X, A, Y_profit, Y_yield, years, lat, lon, cfg)
        if det is None:
            continue
        year_blocks.append(det)

        if args.out_policy_csv:
            idx = det['test_indices']
            eps = cfg.epsilon_explore; lam = cfg.lambda_mix
            cap = cfg.max_density_ratio
            df_year = pd.DataFrame({
                'Field_ID': (field_id[idx] if field_id is not None else np.full(len(idx), np.nan)),
                'Year': np.full(len(idx), int(y), dtype=int),
                'joint_bin_logged': det['joint_logged'].astype(int),
                'joint_bin_target': det['joint_target'].astype(int),
                'count_logged': det['count_logged'].astype(int),
                'count_target': det['count_target'].astype(int),
                'pi0_logged': det['pi0_logged'].astype(float),
                'pi1_logged': det['pi1_logged'].astype(float),
                'ratio': det['ratio_logged_raw'].astype(float),
                'include_overlap': det['keep_mask'].astype(bool),
                'epsilon_explore': np.full(len(idx), eps, dtype=float),
                'lambda_mix': np.full(len(idx), lam, dtype=float),
                'max_density_ratio': np.full(len(idx), cap, dtype=float),
            })
            per_row_frames.append(df_year)

    if not year_blocks:
        print("No evaluation rows found; nothing saved.")
        return

    # Build cohort masks (full vs. overlap)
    keep_masks = [b['keep_mask'] for b in year_blocks]
    n_eval_year = [b['diagnostics']['n_eval'] for b in year_blocks]
    n_in_cohort_year = [b['diagnostics']['n_in_cohort'] for b in year_blocks]

    full_mask = np.concatenate([np.ones_like(k, dtype=bool) for k in keep_masks])
    overlap_mask = np.concatenate(keep_masks)

    # --- collect per-row weights across all evaluated years ---
    w_all_raw = np.concatenate([b['w_raw'] for b in year_blocks])
    w_all_cap = np.concatenate([b['w_cap'] for b in year_blocks])
    in_overlap = np.concatenate([b['keep_mask'] for b in year_blocks])
    years_rep = np.concatenate([[b['year']] * len(b['w_raw']) for b in year_blocks])

    if args.out_weights_csv:
        df_w = pd.DataFrame({
            'year': years_rep.astype(int),
            'w_raw': w_all_raw.astype(float),
            'w_capped': w_all_cap.astype(float),
            'in_overlap': in_overlap.astype(bool),
        })
        df_w.to_csv(args.out_weights_csv, index=False)

        def _summ(name, w):
            if len(w) == 0:
                print(f"{name}: empty")
                return
            q = np.quantile(w, [0, 0.5, 0.9, 0.99])
            print(f"{name}: n={len(w)} min={q[0]:.4g} p50={q[1]:.4g} p90={q[2]:.4g} p99={q[3]:.4g} max={w.max():.4g}")

        _summ("weights: BEFORE overlap", w_all_raw)
        _summ("weights: AFTER overlap", w_all_raw[in_overlap])


    cohort_name = args.cohort
    use_psis = (args.weights == 'psis')

    chosen_mask = overlap_mask if cohort_name == 'overlap' else full_mask

    # Aggregate diagnostics (overall + per-year passthrough)
    diag = {
        'full': {
            'retained_frac': 1.0,
            'psis_k_raw': psis_khat_from_weights(np.concatenate([b['w_raw'] for b in year_blocks]))
        },
        'overlap': {
            'retained_frac': float(np.mean(overlap_mask)),
            'psis_k_raw': psis_khat_from_weights(np.concatenate([b['w_raw'][b['keep_mask']] for b in year_blocks]))
        },
        'per_year': {int(b['year']): b['diagnostics'] for b in year_blocks},
    }

    # --------- POOL fertilizer: pre-trim (all rows) ----------
    def _vec_allrows(b, key):
        f = b['diagnostics']['fertilizer_allrows']
        return np.array([f[key]['N'], f[key]['P'], f[key]['K']], float)

    wsum_all = float(sum(n_eval_year))
    avg_logged_all = (np.sum([w * _vec_allrows(b,'avg_logged') for w,b in zip(n_eval_year, year_blocks)], axis=0) / max(wsum_all,1.0))
    avg_pi1_all    = (np.sum([w * _vec_allrows(b,'avg_pi1')    for w,b in zip(n_eval_year, year_blocks)], axis=0) / max(wsum_all,1.0))
    delta_all      = avg_pi1_all - avg_logged_all
    diag['fertilizer_overall_allrows'] = {
        'avg_logged': {'N': float(avg_logged_all[0]), 'P': float(avg_logged_all[1]), 'K': float(avg_logged_all[2])},
        'avg_pi1':    {'N': float(avg_pi1_all[0]),    'P': float(avg_pi1_all[1]),    'K': float(avg_pi1_all[2])},
        'delta_pi1_minus_logged': {'N': float(delta_all[0]), 'P': float(delta_all[1]), 'K': float(delta_all[2])},
    }

    # --------- POOL fertilizer: on cohort (apples-to-apples) ----------
    def _vec_on_cohort(b, key):
        f = b['diagnostics']['fertilizer_on_cohort']
        return np.array([f[key]['N'], f[key]['P'], f[key]['K']], float)

    # weight by number of rows in the cohort each year
    wsum_cohort = float(sum(n_in_cohort_year))
    if wsum_cohort > 0:
        avg_logged_coh = (np.sum([w * _vec_on_cohort(b,'avg_logged') for w,b in zip(n_in_cohort_year, year_blocks)], axis=0) / wsum_cohort)
        avg_pi1_coh    = (np.sum([w * _vec_on_cohort(b,'avg_pi1')    for w,b in zip(n_in_cohort_year, year_blocks)], axis=0) / wsum_cohort)
        delta_coh      = avg_pi1_coh - avg_logged_coh
    else:
        avg_logged_coh = avg_pi1_coh = delta_coh = np.array([np.nan, np.nan, np.nan])

    diag['fertilizer_overall_on_cohort'] = {
        'avg_logged': {'N': float(avg_logged_coh[0]), 'P': float(avg_logged_coh[1]), 'K': float(avg_logged_coh[2])},
        'avg_pi1':    {'N': float(avg_pi1_coh[0]),    'P': float(avg_pi1_coh[1]),    'K': float(avg_pi1_coh[2])},
        'delta_pi1_minus_logged': {'N': float(delta_coh[0]), 'P': float(delta_coh[1]), 'K': float(delta_coh[2])},
    }

    # Helper to run pooled result
    def pooled(details, mask, est, use_psis):
        return pooled_profit(details, mask, est, use_psis,
                             n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha)

    def pooled_y(details, mask, est, use_psis):
        return pooled_yield(details, mask, est, use_psis,
                            n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha)

    # Overall (profit) — preserved
    overall = {
        'sndr': pooled(year_blocks, chosen_mask, 'sndr', use_psis),
        'dr':   pooled(year_blocks, chosen_mask, 'dr',   use_psis),
    }

    # Overall (yield under same π1) — NEW
    overall_yield = {
        'sndr': pooled_y(year_blocks, chosen_mask, 'sndr', use_psis),
        'dr':   pooled_y(year_blocks, chosen_mask, 'dr',   use_psis),
    }

    # By-year results
    by_year_rows = []
    by_year = {}
    by_year_yield = {}

    for b in year_blocks:
        ymask = b['keep_mask'] if cohort_name == 'overlap' else np.ones_like(b['keep_mask'], dtype=bool)

        sndr_y = pooled([b], ymask, 'sndr', use_psis)
        dr_y   = pooled([b], ymask, 'dr',   use_psis)
        by_year[int(b['year'])] = {'diagnostics': b['diagnostics'], 'sndr': sndr_y, 'dr': dr_y}

        sndr_yield = pooled_y([b], ymask, 'sndr', use_psis)
        dr_yield   = pooled_y([b], ymask, 'dr',   use_psis)
        by_year_yield[int(b['year'])] = {'sndr': sndr_yield, 'dr': dr_yield}

        # CSV rows (profit only; schema unchanged)
        for est_name, res in [('sndr', sndr_y), ('dr', dr_y)]:
            by_year_rows.append({
                'year': int(b['year']),
                'cohort': cohort_name,
                'weights': 'psis' if use_psis else 'cap',
                'estimator': est_name,
                'V1': res['V1'], 'V0': res['V0'], 'delta': res['delta'],
                'ci_lo': res['ci_lo'], 'ci_hi': res['ci_hi'], 'lcb': res['lcb'],
                'uplift_pct': res.get('uplift_pct', np.nan), 'uplift_lcb': res.get('uplift_lcb', np.nan),
                'psis_k_subset': res.get('psis_k_subset', np.nan),
                'psis_k_raw_in_year': b['diagnostics']['psis_k_raw'],
                'retained_frac': b['diagnostics']['retained_frac'], 'n_eval': b['diagnostics']['n_eval'],
            })

    # Decision (same logic as before) — on PROFIT SNDR
    primary_key = 'sndr'
    lcb = overall[primary_key]['lcb']
    decision = bool(lcb >= -args.noninferiority_epsilon)

    summary = {
        'arguments': vars(args),
        'cohort': cohort_name,
        'weights': 'psis' if use_psis else 'cap',
        'diagnostics': diag,

        # PROFIT (unchanged)
        'overall': overall,
        'by_year': by_year,

        # YIELD under SAME π1 (NEW)
        'overall_yield': overall_yield,
        'by_year_yield': by_year_yield,

        'accept_by_lcb': decision,
        'primary': primary_key,
    }

    # Write outputs
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # Small overall CSV (profit only; schema unchanged)
    overall_rows = []
    for est_name, res in overall.items():
        overall_rows.append({
            'cohort': cohort_name,
            'weights': 'psis' if use_psis else 'cap',
            'estimator': est_name,
            'V1': res['V1'], 'V0': res['V0'], 'delta': res['delta'],
            'ci_lo': res['ci_lo'], 'ci_hi': res['ci_hi'], 'lcb': res['lcb'],
            'uplift_pct': res.get('uplift_pct', np.nan), 'uplift_lcb': res.get('uplift_lcb', np.nan),
        })
    pd.DataFrame(overall_rows).to_csv(args.out_csv, index=False)

    # Full by-year CSV (profit only; schema unchanged)
    pd.DataFrame(by_year_rows).to_csv(args.by_year_csv, index=False)

    # Optional per-row dump
    if per_row_frames and args.out_policy_csv:
        os.makedirs(os.path.dirname(args.out_policy_csv), exist_ok=True)
        pd.concat(per_row_frames, ignore_index=True).to_csv(args.out_policy_csv, index=False)

    # Console one-liner (profit deltas)
    print(f"POOLED[{cohort_name}|{'psis' if use_psis else 'cap'}] "
          f"SNDR Δ_profit={overall['sndr']['delta']:.3f} (LCB={overall['sndr']['lcb']:.3f}); "
          f"DR Δ_profit={overall['dr']['delta']:.3f} (LCB={overall['dr']['lcb']:.3f}) | accept={decision}")

if __name__ == '__main__':
    main()