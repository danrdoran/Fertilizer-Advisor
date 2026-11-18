"""Forward-year diagnostics; none of these functions fits or selects a model."""

from __future__ import annotations

import numpy as np
import pandas as pd


NUTRIENTS = ("N", "P", "K")


def feature_names(data_path, width):
    from pathlib import Path

    path = Path(data_path).parent / "feature_cols.csv"
    names = pd.read_csv(path, header=None)[0].astype(str).tolist() if path.exists() else [f"feature_{j}" for j in range(width)]
    if len(names) != width or len(set(names)) != width:
        raise ValueError("Companion feature names do not match the context matrix.")
    return names


def make_reference(data, split_audits, reference_year):
    """Recover the first grid, already learned using the earliest year only.

    Interior boundaries and finite midpoint actions are frozen. Assignment has
    open upper tails; physical extrapolation is flagged separately, never hidden.
    """
    earliest = int(np.min(data["year"]))
    if reference_year != earliest:
        raise ValueError("Reference year must be the earliest available training year.")
    first = min(split_audits, key=lambda a: a["year"])
    if first["year"] != reference_year + 1:
        raise ValueError("First evaluated year must immediately follow the reference year.")
    fits = first["model_fits"]
    train = data["actions"][data["year"] == reference_year]
    if not fits or any(f["fit_years"] != [reference_year] or f["n_fit"] != len(train) for f in fits):
        raise ValueError("Reference grid was not fitted using the earliest training year only.")
    if not np.isfinite(train).all() or (train < 0).any():
        raise ValueError("Fertilizer rates must be finite and nonnegative.")
    result = {"schema_version": 1, "reference_year": reference_year,
              "n_reference_rows": len(train), "first_evaluation_year": first["year"],
              "assignment": "Nonnegative rates; fixed interior cuts; open upper tails.",
              "representatives": "Finite midpoints of the original reference-year training edges.",
              "dimensions": {}}
    for j, nutrient in enumerate(NUTRIENTS):
        edges = np.asarray(first["bin_edges"][nutrient], float)
        lo, hi = float(train[:, j].min()), float(train[:, j].max())
        if (len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all()
                or not np.isclose(edges[0], lo) or edges[-1] < hi
                or not np.isclose(edges[-1], hi, atol=1e-4, rtol=1e-7)):
            raise ValueError("Reference boundaries do not match the earliest training range.")
        result["dimensions"][nutrient] = {
            "training_edges": edges.tolist(), "interior_cuts": edges[1:-1].tolist(),
            "centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "observed_min": lo, "observed_max": hi,
        }
    return result


def reference_edges(reference):
    return {k: np.r_[0.0, reference["dimensions"][k]["interior_cuts"], np.inf] for k in NUTRIENTS}


def action_labels(actions, edges):
    actions = np.asarray(actions, float)
    if actions.ndim != 2 or actions.shape[1] != 3 or not np.isfinite(actions).all() or (actions < 0).any():
        raise ValueError("Expected finite, nonnegative N/P/K action rows.")
    dims = tuple(len(edges[k]) - 1 for k in NUTRIENTS)
    labels = [np.clip(np.searchsorted(np.asarray(edges[k])[1:-1], actions[:, j], side="right"), 0, dims[j] - 1) for j, k in enumerate(NUTRIENTS)]
    return np.ravel_multi_index(tuple(labels), dims)


def reference_range_flags(actions, reference):
    flags = {f"outside_reference_{k}": (actions[:, j] < reference["dimensions"][k]["observed_min"]) |
             (actions[:, j] > reference["dimensions"][k]["observed_max"]) for j, k in enumerate(NUTRIENTS)}
    flags["within_reference_range"] = ~np.logical_or.reduce(list(flags.values()))
    return flags


def _metrics(y, prediction):
    error = prediction - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {"n_rows": len(y), "observed_mean": float(y.mean()), "predicted_mean": float(prediction.mean()),
            "prediction_minus_observed_bias": float(error.mean()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": 1 - float(error @ error) / denominator if denominator > 0 else float("nan")}


def prediction_table(rows, data, prices):
    """Logged continuous-action predictions, not training OOF scores."""
    result = []
    costs = data["actions"] @ np.array([prices["price" + k] for k in NUTRIENTS])
    outcomes = {"yield": data["reward"], "profit": prices["maize_price"] * data["reward"] - costs}
    group_keys = ["grid_mode", "stage", "year"] if "grid_mode" in rows else ["stage", "year"]
    for key, group in rows.groupby(group_keys, sort=True):
        info = dict(zip(group_keys, key))
        year = int(info["year"])
        train = data["year"] < year
        last = data["year"] == np.max(data["year"][train])
        for cohort, mask in (("all", np.ones(len(group), bool)), ("audit_cohort", group["included_cohort"].to_numpy(bool))):
            if not mask.any():
                continue
            for outcome, original in outcomes.items():
                y = group.loc[mask, outcome].to_numpy(float)
                predictions = {
                    "outcome_model": group.loc[mask, f"mu_{outcome}_logged"].to_numpy(float),
                    "historical_training_mean": np.full(len(y), original[train].mean()),
                    "last_training_year_mean": np.full(len(y), original[last].mean()),
                }
                for model, pred in predictions.items():
                    result.append({**info, "cohort": cohort, "outcome": outcome, "predictor": model,
                                   "n_clusters": group.loc[mask, "site_key"].nunique(), **_metrics(y, pred)})
    return pd.DataFrame(result)


def shift_table(data, years, names, prices):
    """Descriptive distribution shifts. Observed outcomes never enter fitting."""
    variables = [("context", name, data["obs"][:, j]) for j, name in enumerate(names)]
    variables += [("action", k, data["actions"][:, j]) for j, k in enumerate(NUTRIENTS)]
    profit = prices["maize_price"] * data["reward"] - data["actions"] @ np.array([prices["price" + k] for k in NUTRIENTS])
    variables += [("outcome_diagnostic_only", "yield", data["reward"]), ("outcome_diagnostic_only", "profit", profit)]
    result = []
    for year in years:
        train, test = data["year"] < year, data["year"] == year
        for role, name, values in variables:
            a, b = values[train], values[test]
            x, y = a[np.isfinite(a)], b[np.isfinite(b)]
            record = {"year": year, "variable_role": role, "variable": name,
                      "training_missing_fraction": 1 - len(x) / len(a), "evaluation_missing_fraction": 1 - len(y) / len(b)}
            if len(x) and len(y):
                sd_x = float(x.std(ddof=1)) if len(x) > 1 and np.ptp(x) > 0 else 0.0
                sd_y = float(y.std(ddof=1)) if len(y) > 1 and np.ptp(y) > 0 else 0.0
                scale = np.sqrt((sd_x ** 2 + sd_y ** 2) / 2)
                points = np.sort(np.r_[x, y])
                ks = np.max(np.abs(np.searchsorted(np.sort(x), points, side="right") / len(x) - np.searchsorted(np.sort(y), points, side="right") / len(y)))
                record.update(training_mean=x.mean(), evaluation_mean=y.mean(), training_sd=sd_x, evaluation_sd=sd_y,
                              training_min=x.min(), training_max=x.max(), evaluation_min=y.min(), evaluation_max=y.max(),
                              standardized_mean_difference=(y.mean() - x.mean()) / scale if scale > 0 else (0.0 if x.mean() == y.mean() else float("nan")),
                              ks_statistic=float(ks), evaluation_outside_training_range_fraction=float(((y < x.min()) | (y > x.max())).mean()),
                              training_constant_changes=bool(sd_x == 0 and (y != x[0]).any()))
            result.append(record)
    return pd.DataFrame(result)


def bin_stability_table(data, split_audits, reference):
    result = []
    for split in split_audits:
        actions = data["actions"][data["year"] == split["year"]]
        for j, k in enumerate(NUTRIENTS):
            edges = np.asarray(split["bin_edges"][k], float)
            current = edges[1:-1]
            cuts = np.asarray(reference["dimensions"][k]["interior_cuts"])
            distances = np.abs(current[:, None] - cuts[None, :]) if len(current) and len(cuts) else np.empty((0, 0))
            left = np.clip(np.searchsorted(current, actions[:, j], side="right"), 0, len(edges) - 2)
            right = np.searchsorted(cuts, actions[:, j], side="right")
            centers = (edges[:-1] + edges[1:]) / 2
            ref_centers = np.asarray(reference["dimensions"][k]["centers"])
            displacement = np.abs(centers[left] - ref_centers[right])
            result.append({"year": split["year"], "nutrient": k, "adaptive_n_bins": len(edges) - 1,
                           "fixed_n_bins": len(ref_centers),
                           "interior_boundary_hausdorff_kg_ha": float(max(distances.min(axis=0).max(), distances.min(axis=1).max())) if distances.size else (0.0 if not len(current) and not len(cuts) else float("nan")),
                           "mean_assigned_representative_displacement_kg_ha": displacement.mean(),
                           "max_assigned_representative_displacement_kg_ha": displacement.max(),
                           "index_disagreement_fraction_not_treatment_equivalence": float((left != right).mean())})
    return pd.DataFrame(result)


def residual_cell_table(rows, data, edges_by_year):
    result = []
    keys = ["grid_mode", "stage", "year"] if "grid_mode" in rows else ["stage", "year"]
    for key, group in rows.groupby(keys, sort=True):
        info = dict(zip(keys, key))
        year = int(info["year"])
        edges = edges_by_year[(info.get("grid_mode", "adaptive"), year)]
        test = data["year"] == year
        if not np.array_equal(group["field_id"].to_numpy(), data["field_id"][test]):
            raise ValueError("Diagnostic rows are misaligned with the original data.")
        train_labels = action_labels(data["actions"][data["year"] < year], edges)
        counts = np.bincount(train_labels, minlength=np.prod([len(edges[k]) - 1 for k in NUTRIENTS]))
        labels = action_labels(data["actions"][test], edges)
        if "count_logged" in group and not np.array_equal(group["count_logged"].to_numpy(), counts[labels]):
            raise ValueError("Saved logged-action counts do not match the training grid.")
        for cell in np.unique(labels):
            local = group.loc[labels == cell]
            for outcome in ("profit", "yield"):
                result.append({**info, "joint_cell": int(cell), "training_count": int(counts[cell]), "outcome": outcome,
                               "n_retained": int(local["included_cohort"].sum()),
                               **_metrics(local[outcome].to_numpy(float), local[f"mu_{outcome}_logged"].to_numpy(float))})
    return pd.DataFrame(result)


def propensity_diagnostics(probabilities, labels, counts, n_bins=10, floor=1e-12):
    """Multiclass scores and top-label reliability (not flattened zero cells).

    Multiclass Brier is the unscaled sum over classes, range [0, 2]. Log loss
    uses an explicit floor and separately records zero logged probabilities.
    """
    p, labels, counts = np.asarray(probabilities, float), np.asarray(labels, int), np.asarray(counts)
    if (p.ndim != 2 or len(labels) != len(p) or counts.shape != (p.shape[1],) or
            not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6) or
            (labels < 0).any() or (labels >= p.shape[1]).any() or not 0 < floor < 1 or n_bins < 2):
        raise ValueError("Invalid multiclass propensity predictions or labels.")
    logged = p[np.arange(len(p)), labels]
    top = p.argmax(axis=1)
    confidence = p[np.arange(len(p)), top]
    correct = top == labels
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    reliability = []
    for b in range(n_bins):
        mask = bins == b
        reliability.append({"bin": b, "lower_inclusive": b / n_bins, "upper": (b + 1) / n_bins,
                            "n_rows": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()) if mask.any() else float("nan"),
                            "accuracy": float(correct[mask].mean()) if mask.any() else float("nan")})
    summary = {"n_rows": len(p), "n_action_cells": p.shape[1],
               "multiclass_brier_sum": float(np.mean(np.sum(p ** 2, axis=1) - 2 * logged + 1)),
               "log_loss_clipped": float(-np.log(np.maximum(logged, floor)).mean()),
               "log_loss_unclipped": float(-np.log(logged).mean()) if (logged > 0).all() else float("inf"),
               "log_loss_floor": floor, "zero_logged_probability_fraction": float((logged == 0).mean()),
               "unseen_training_cell_fraction": float((counts[labels] == 0).mean()),
               "top_label_accuracy": float(correct.mean()),
               "top_label_ece": float(sum(r["n_rows"] / len(p) * abs(r["accuracy"] - r["mean_confidence"]) for r in reliability if r["n_rows"]))}
    empirical = np.bincount(labels, minlength=p.shape[1]) / len(p)
    cells = [{"joint_cell": j, "training_count": int(counts[j]), "predicted_frequency": float(p[:, j].mean()),
              "observed_frequency": float(empirical[j]), "observed_minus_predicted": float(empirical[j] - p[:, j].mean())} for j in range(p.shape[1])]
    return summary, reliability, cells
