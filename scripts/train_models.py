"""
What this script does:

- Loads processed data (obs, actions, profit, year) and feature names
- Learns adaptive bins on training years
- Trains a stacked reward model with forward-by-year CV
- Reports proper fold R^2 using sklearn
- Prints a simple overlap/coverage diagnostic per fold
- Trains calibrated, factorized propensities (one head per N/P/K)
- Evaluates reward model on the holdout year
- Computes and saves π0 reliability curves (TRAIN & TEST), with ECE
- Saves artifacts:
    - ensemble_model.pkl
    - binner.pkl
    - propensity_model.pkl
    - cv_results.csv
    - test_metrics.json
    - training_settings.json
    - pi0_reliability_train.csv / .png
    - pi0_reliability_test.csv / .png
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, log_loss

# Import the core building blocks from your core module
from bandit_models import (
    SEED,
    set_random_seeds,
    rmse_fn,
    forward_year_folds,
    make_recency_weights,
    AdaptiveBinner,
    StackedRewardModel,
    JointPropensityModel,
)

# ---------------------------------------------------------------------
# Geo grouping utility (as in your original)
# ---------------------------------------------------------------------
def geo_clusters(lat: np.ndarray, lon: np.ndarray, decimals: int = 2) -> np.ndarray:
    coords = np.round(np.column_stack([lat, lon]), decimals)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    return inv

# ---------------------------------------------------------------------
# Overlap diagnostic (unchanged)
# ---------------------------------------------------------------------
def fold_overlap_diagnostic(A_tr_binned: np.ndarray,
                            A_va_binned: np.ndarray,
                            min_count: int = 5) -> float:
    from collections import Counter
    keys_tr = [tuple(row) for row in A_tr_binned]
    ctr = Counter(keys_tr)
    covered = np.array([ctr.get(tuple(row), 0) >= min_count for row in A_va_binned], dtype=bool)
    return float(covered.mean())

# ---------------------------------------------------------------------
# NEW: Multiclass reliability curve + plotting for joint π0
#   - One-vs-rest stacking across all joint classes
#   - Returns dataframe with per-bin stats and an ECE scalar
# ---------------------------------------------------------------------
def multiclass_reliability_curve(probs: np.ndarray,
                                 y_true_joint: np.ndarray,
                                 n_bins: int = 20,
                                 sample_weight: np.ndarray | None = None):
    """
    Inputs
      probs: (n_samples, C) predicted class probabilities (sum=1 per row)
      y_true_joint: (n_samples,) true joint class index in [0..C-1]
      n_bins: number of probability bins on [0,1]
      sample_weight: (n_samples,) optional row weights
    Returns
      df: DataFrame with columns [bin_mid, mean_pred, mean_true, count, weight_sum]
      ece: Expected Calibration Error (|mean_true - mean_pred| weighted by bin prob mass)
    """
    n, C = probs.shape
    if sample_weight is None:
        w_rows = np.ones(n, dtype=float)
    else:
        w_rows = np.asarray(sample_weight, float)
        if w_rows.shape[0] != n:
            raise ValueError("sample_weight must have length n_samples")

    # One-hot for true labels
    y_onehot = np.zeros_like(probs, dtype=float)
    y_onehot[np.arange(n), y_true_joint] = 1.0

    # Flatten to one-vs-rest pairs across all classes
    p = probs.ravel()                      # (n*C,)
    t = y_onehot.ravel()                   # (n*C,)
    w = np.repeat(w_rows, C).astype(float) # (n*C,)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    out = []
    ece_num = 0.0
    total_w = float(w.sum())
    for b in range(n_bins):
        sel = (bin_idx == b)
        if not np.any(sel):
            continue
        wb = w[sel]; pb = p[sel]; tb = t[sel]
        wsum = float(wb.sum())
        mean_pred = float(np.average(pb, weights=wb))
        mean_true = float(np.average(tb, weights=wb))
        count = int(sel.sum())
        mid = 0.5 * (edges[b] + edges[b+1])
        out.append((mid, mean_pred, mean_true, count, wsum))
        ece_num += (abs(mean_true - mean_pred) * (wsum / max(total_w, 1e-12)))

    df = pd.DataFrame(out, columns=["bin_mid", "mean_pred", "mean_true", "count", "weight_sum"])
    ece = float(ece_num)
    return df, ece

def plot_reliability_curve(df: pd.DataFrame,
                           title: str = "π₀ reliability (multiclass, one-vs-rest)",
                           path: str | None = None):
    plt.figure(figsize=(6, 6))
    sizes = 20 + 80 * (df["weight_sum"] / max(df["weight_sum"].max(), 1e-12))
    plt.scatter(df["mean_pred"], df["mean_true"], s=sizes)
    lims = [0, 1]
    plt.plot(lims, lims, linestyle="--")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

# ---------------------------------------------------------------------
# Cross-validated reward model training (unchanged)
# ---------------------------------------------------------------------
def cross_validate_with_overlap(
    obs: np.ndarray,
    actions: np.ndarray,
    profits: np.ndarray,
    years: np.ndarray,
    feature_names: list,
    recency_lambda: float,
    start_val_year: int,
    end_val_year: int,
    n_trials: int = 2,
    lat: np.ndarray = None,         
    lon: np.ndarray = None
):
    folds = forward_year_folds(years, start_val_year, end_val_year)
    if not folds:
        raise ValueError("No forward-year folds constructed")

    ensemble = StackedRewardModel()
    fold_results = []

    final_binner = None

    for fold_num, (tr_idx, va_idx, val_year) in enumerate(folds, start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold_num}: Training <= {val_year-1}, Validation = {val_year}")
        print("=" * 60)

        X_tr, X_va = obs[tr_idx], obs[va_idx]
        A_tr, A_va = actions[tr_idx], actions[va_idx]
        Y_tr, Y_va = profits[tr_idx], profits[va_idx]
        years_tr = years[tr_idx]

        # build geo groups for the training portion of this fold
        groups_tr = None
        if (lat is not None) and (lon is not None):
            groups_tr = geo_clusters(lat[tr_idx], lon[tr_idx], decimals=2)

        # FIT BINNER PER FOLD (train-only)
        print(f"Learning adaptive bins for fold {fold_num} from training data only...")
        fold_binner = AdaptiveBinner(min_samples_per_bin=80, max_bins_per_dim={"N": 6, "P": 3, "K": 2})
        fold_binner.fit(A_tr, Y_tr)

        # Report the fold-specific bin edges
        for name, edges in fold_binner.bin_edges.items():
            print(f"  Fold {fold_num} - {name} bins ({len(edges)-1}): {edges}")

        # Transform actions using fold-specific bins
        A_tr_b = fold_binner.transform(A_tr)
        A_va_b = fold_binner.transform(A_va)

        # Recency weights for the reward model
        sw = make_recency_weights(years_tr, recency_lambda)

        # Fit reward model on this fold
        ensemble.fit(
            X_tr, A_tr, Y_tr,
            X_val=X_va, actions_val=A_va, y_val=Y_va,
            sample_weight=sw,
            n_trials=n_trials // 2,
            feature_names=feature_names,
            groups=groups_tr
        )

        # Predict on the validation part
        va_pred = ensemble.predict(X_va, A_va)

        # Proper fold metrics
        r2 = r2_score(Y_va, va_pred)
        rmse = rmse_fn(Y_va, va_pred)
        mae = mean_absolute_error(Y_va, va_pred)

        # Overlap coverage diagnostic
        cover_share = fold_overlap_diagnostic(A_tr_b, A_va_b, min_count=5)

        fold_results.append(
            {
                "year": int(val_year),
                "r2": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "overlap_covered": float(cover_share),
            }
        )

        cov_txt = "good" if cover_share >= 0.75 else "low"
        print(f"Overlap covered (≥5/train-bin): {cover_share:.3f} ({cov_txt})")
        print(f"Fold {fold_num}: R²={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f}")

    # Train final model on all training data up through end_val_year
    print("\n" + "=" * 60)
    print("Training FINAL reward model on all ≤ end_val_year data...")
    print("=" * 60)

    final_mask = years <= end_val_year
    final_sw = make_recency_weights(years[final_mask], recency_lambda)

    print("Learning final adaptive bins from all training data...")
    final_binner = AdaptiveBinner(min_samples_per_bin=80, max_bins_per_dim={"N": 6, "P": 3, "K": 2})
    final_binner.fit(actions[final_mask], profits[final_mask])

    for name, edges in final_binner.bin_edges.items():
        print(f"Final {name} bins ({len(edges)-1}): {edges}")

    final_ensemble = StackedRewardModel()

    final_groups = None
    if (lat is not None) and (lon is not None):
        final_groups = geo_clusters(lat[final_mask], lon[final_mask], decimals=2)

    final_ensemble.fit(
        obs[final_mask],
        actions[final_mask],
        profits[final_mask],
        sample_weight=final_sw,
        n_trials=n_trials,
        feature_names=feature_names,
        groups=final_groups
    )

    return final_ensemble, final_binner, fold_results

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--data", default="data/processed/processed_data.npz")
    parser.add_argument("--features", default="data/processed/feature_cols.csv")
    parser.add_argument("--output_dir", default="ensemble_model")

    # Training settings
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--recency_lambda", type=float, default=0.2)
    parser.add_argument("--start_val_year", type=int, default=2014)
    parser.add_argument("--end_val_year", type=int, default=2017)
    parser.add_argument("--holdout_year", type=int, default=2018)

    # Placeholder to keep parity with prior CLIs (ignored here)
    parser.add_argument("--use_gps", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seeds(SEED)

    # ===========================
    # Load data
    # ===========================
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = np.load(args.data)
    obs = data["obs"]
    actions = data["actions"]
    profits = data["profit"]
    years = data["year"]
    lat = data["lat"]
    lon = data["lon"]
    feature_names = pd.read_csv(args.features, header=None)[0].tolist()

    train_mask = years <= args.end_val_year
    test_mask = years == args.holdout_year
    if test_mask.sum() == 0:
        raise ValueError(f"No samples for holdout year {args.holdout_year}")

    X_train, A_train, Y_train, years_train = (
        obs[train_mask],
        actions[train_mask],
        profits[train_mask],
        years[train_mask],
    )
    X_test, A_test, Y_test = obs[test_mask], actions[test_mask], profits[test_mask]

    print(f"Training samples (≤{args.end_val_year}): {len(X_train)}")
    print(f"Test samples ({args.holdout_year}): {len(X_test)}")

    # ===========================
    # Train reward model (CV + final)
    # ===========================
    print("\n" + "=" * 70)
    print("TRAINING REWARD MODEL")
    print("=" * 70)

    ensemble, binner, cv_results = cross_validate_with_overlap(
        obs,
        actions,
        profits,
        years,
        feature_names,
        recency_lambda=args.recency_lambda,
        start_val_year=args.start_val_year,
        end_val_year=args.end_val_year,
        n_trials=args.n_trials,
        lat=lat,
        lon=lon
    )

    # Save CV results and summarize
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)
    mean_r2 = float(cv_df["r2"].mean())
    std_r2 = float(cv_df["r2"].std())
    print(f"\nCV Summary - Mean R²: {mean_r2:.3f} (±{std_r2:.3f})")

    # ===========================
    # Train Joint Propensity Model π₀
    # ===========================
    print("\n" + "=" * 70)
    print("TRAINING JOINT PROPENSITY MODEL π₀")
    print("=" * 70)

    def ess(weights):
        """Effective sample size"""
        w = np.asarray(weights)
        return (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)

    # Configure joint propensity parameters
    joint_prop_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    }

    # Initialize joint propensity model
    joint_prop_model = JointPropensityModel(
        base_params=joint_prop_params,
        epsilon=1e-4,
        calibration_cv=3,             # enable calibration if class counts allow
        calibration_method="isotonic" # sigmoid (Platt) is ok, isotonic often better here
    )


    # Use recency weights for propensity training
    prop_weights = make_recency_weights(years_train, args.recency_lambda)

    # Train on all data ≤ end_val_year
    print(f"Training joint π₀ on {len(X_train)} samples (years ≤ {args.end_val_year})")
    joint_prop_model.fit(
        X_train,
        A_train,
        binner.bin_edges,
        sample_weight=prop_weights
    )

    # ---------------------------
    # Diagnostics (unchanged)
    # ---------------------------
    print("\nJoint propensity model diagnostics:")

    # Transform actions to joint indices for evaluation
    A_train_binned = binner.transform(A_train)
    n_bins = tuple(len(binner.bin_edges[k]) - 1 for k in ['N', 'P', 'K'])
    train_joint_idx = np.array([
        nb * n_bins[1] * n_bins[2] + pb * n_bins[2] + kb
        for nb, pb, kb in A_train_binned
    ], dtype=int)

    # Get predicted probabilities
    train_probs_grid = joint_prop_model.predict_proba_grid(X_train)
    n_actions = int(np.prod(n_bins))

    # Log loss
    train_ll = log_loss(train_joint_idx, train_probs_grid, labels=np.arange(n_actions))
    print(f"  Training log loss: {train_ll:.4f}")

    # π0 at logged actions
    pi0_at_logged = train_probs_grid[np.arange(len(X_train)), train_joint_idx]
    print(f"  Mean π₀(logged action): {pi0_at_logged.mean():.4f}")
    print(f"  Min  π₀(logged action): {pi0_at_logged.min():.6f}")
    print(f"  Max  π₀(logged action): {pi0_at_logged.max():.4f}")

    ess_ratio = ess(1.0 / np.clip(pi0_at_logged, 1e-4, 1.0)) / len(X_train)
    print(f"  ESS ratio (1/π₀): {ess_ratio:.3f}")

    # Coverage analysis
    action_counts = np.bincount(train_joint_idx, minlength=n_actions)
    n_observed = (action_counts > 0).sum()
    n_common = (action_counts >= 10).sum()
    print(f"  Action cells: {n_observed}/{n_actions} observed, {n_common} with 10+ samples")

    # ===========================
    # NEW: π0 RELIABILITY (TRAIN & TEST) — save CSV + PNG and print ECE
    # ===========================
    print("\nComputing π₀ reliability curves...")

    # TRAIN reliability
    rel_train_df, rel_train_ece = multiclass_reliability_curve(
        train_probs_grid, train_joint_idx, n_bins=20
    )
    rel_train_csv = os.path.join(args.output_dir, "pi0_reliability_train.csv")
    rel_train_png = os.path.join(args.output_dir, "pi0_reliability_train.png")
    rel_train_df.to_csv(rel_train_csv, index=False)
    plot_reliability_curve(rel_train_df,
                           title=f"π₀ reliability (TRAIN) | ECE={rel_train_ece:.3f}",
                           path=rel_train_png)
    print(f"  TRAIN ECE: {rel_train_ece:.3f}  | saved {os.path.basename(rel_train_csv)}, {os.path.basename(rel_train_png)}")

    # TEST reliability (if any test rows)
    if len(X_test) > 0:
        A_test_binned = binner.transform(A_test)
        test_joint_idx = np.array([
            nb * n_bins[1] * n_bins[2] + pb * n_bins[2] + kb
            for nb, pb, kb in A_test_binned
        ], dtype=int)

        test_probs_grid = joint_prop_model.predict_proba_grid(X_test)
        test_ll = log_loss(test_joint_idx, test_probs_grid, labels=np.arange(n_actions))
        print(f"  Test log loss: {test_ll:.4f}")

        rel_test_df, rel_test_ece = multiclass_reliability_curve(
            test_probs_grid, test_joint_idx, n_bins=20
        )
        rel_test_csv = os.path.join(args.output_dir, "pi0_reliability_test.csv")
        rel_test_png = os.path.join(args.output_dir, "pi0_reliability_test.png")
        rel_test_df.to_csv(rel_test_csv, index=False)
        plot_reliability_curve(rel_test_df,
                               title=f"π₀ reliability (TEST) | ECE={rel_test_ece:.3f}",
                               path=rel_test_png)
        print(f"  TEST  ECE: {rel_test_ece:.3f}  | saved {os.path.basename(rel_test_csv)}, {os.path.basename(rel_test_png)}")

    # ===========================
    # Test-set predictive metrics (reward model)
    # ===========================
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    mu_test = ensemble.predict(X_test, A_test)
    test_r2 = r2_score(Y_test, mu_test)          # Proper R² (can be negative)
    test_mae = mean_absolute_error(Y_test, mu_test)
    test_rmse = rmse_fn(Y_test, mu_test)
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.1f}")
    print(f"Test RMSE: {test_rmse:.1f}")

    test_metrics = {
        "r2": float(test_r2),
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "n_test": int(len(Y_test)),
    }
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ===========================
    # Save artifacts + settings
    # ===========================
    joblib.dump(ensemble, os.path.join(args.output_dir, "ensemble_model.pkl"))
    joblib.dump(binner, os.path.join(args.output_dir, "binner.pkl"))
    joblib.dump(joint_prop_model, os.path.join(args.output_dir, "propensity_model.pkl"))

    with open(os.path.join(args.output_dir, "training_settings.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nArtifacts saved to:", args.output_dir)
    print("=" * 70)

if __name__ == "__main__":
    main()