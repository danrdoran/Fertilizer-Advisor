"""Audit a saved chronological run without changing its selection or estimators.

summary: inspect saved outputs and raw cohort characteristics, without fitting.
refit: recover/cache missing predictions, reproduce the original point estimates,
       and decompose aggregation and economic-accounting differences.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import run_revision_evaluation as revision
from ope_statistics import InferenceUnavailable, cluster_summary
from revision_audit import MODES, aggregation_decomposition, audit_estimates, rows_from_fit


NOTICE = (
    "Exploratory audit of a previously inspected retrospective run. The original "
    "selection, data, estimators and output files are not changed. Alternative "
    "aggregations and accounting-consistent estimates do not constitute a new "
    "independent confirmation or a replacement policy-selection rule. Intervals "
    "condition on fitted predictions, weights, policy and overlap inclusion. "
    "Action-dependent trimming and unmeasured confounding remain limitations."
)
AUDIT_SOURCES = ("scripts/audit_revision_evaluation.py", "scripts/revision_audit.py")


def _json(path):
    return json.loads(Path(path).read_text())


def _audit_sources():
    return {name: revision.sha256_file(revision.ROOT / name) for name in AUDIT_SOURCES}


def load_original(selection_path, data_path, final_dir):
    selection = _json(selection_path)
    if selection.get("schema_version") != 1:
        raise ValueError("Expected a schema-version-1 frozen selection.")
    revision.validate_spec(selection["spec"])
    if revision.sha256_file(data_path) != selection["data_sha256"]:
        raise ValueError("Data changed since the frozen selection.")
    if revision.fingerprint_sources() != selection["source_sha256"]:
        raise ValueError("Original evaluation source changed. Restore the source recorded in selection.json.")
    if selection["selected_candidate"] not in selection["spec"]["candidates"]:
        raise ValueError("Frozen candidate is not in the recorded grid.")
    years = selection["selection_years"]
    if not years or max(years) >= selection["spec"]["holdout_year"]:
        raise ValueError("Invalid chronological split in the frozen selection.")
    final = _json(Path(final_dir) / "final_results.json")
    if final["selection_sha256"] != revision.sha256_file(selection_path) or final["selection"] != selection:
        raise ValueError("Final results do not belong to this frozen selection.")
    data = revision.load_data(data_path, through_year=selection["spec"]["holdout_year"])
    return selection, final, data


def _boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy()
    values = series.astype(str).map({"True": True, "False": False})
    if values.isna().any():
        raise ValueError("Expected explicit True/False overlap flags.")
    return values.to_numpy(dtype=bool)


def cohort_tables(data, holdout, holdout_year, prices, data_path):
    idx = np.where(data["year"] == holdout_year)[0]
    if len(holdout) != len(idx) or not np.array_equal(holdout["field_id"], data["field_id"][idx]):
        raise ValueError("Holdout field IDs or row order do not match the recorded data.")
    if not (holdout["year"] == holdout_year).all():
        raise ValueError("Holdout rows contain a different evaluation year.")
    retained = _boolean(holdout["included_overlap"])
    years = data["year"][idx]
    A, Y, X = data["actions"][idx], data["reward"][idx], data["obs"][idx]
    values = {"observed_yield_kg_ha": Y, "logged_N_kg_ha": A[:, 0],
              "logged_P_kg_ha": A[:, 1], "logged_K_kg_ha": A[:, 2]}
    values["observed_revenue_minus_fertilizer_MXN_ha"] = prices["maize_price"] * Y - A @ np.array([prices["priceN"], prices["priceP"], prices["priceK"]])
    names_path = Path(data_path).parent / "feature_cols.csv"
    names = pd.read_csv(names_path, header=None)[0].astype(str).tolist() if names_path.exists() else [f"feature_{i}" for i in range(X.shape[1])]
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ValueError("Companion feature names do not match the context matrix.")
    values.update({"context:" + name: X[:, i] for i, name in enumerate(names)})
    records = []
    for name, a in values.items():
        groups = [a[retained], a[~retained]]
        groups = [g[np.isfinite(g)] for g in groups]
        means = [float(g.mean()) if len(g) else float("nan") for g in groups]
        variances = [float(g.var(ddof=1)) if len(g) > 1 else float("nan") for g in groups]
        scale = np.sqrt(np.mean(variances))
        smd = (means[0] - means[1]) / scale if scale > 0 else (0.0 if means[0] == means[1] else float("nan"))
        records.append({"variable": name, "retained_n_nonmissing": len(groups[0]),
                        "excluded_n_nonmissing": len(groups[1]), "retained_mean": means[0],
                        "excluded_mean": means[1], "retained_minus_excluded": means[0] - means[1],
                        "standardized_mean_difference": smd})
    clusters = []
    for label, mask in (("all", np.ones(len(idx), bool)), ("retained", retained), ("excluded", ~retained)):
        clusters.append({"year": int(years[0]), "cohort": label, **cluster_summary(holdout.loc[mask, "site_key"].to_numpy())})
    return pd.DataFrame(records), pd.DataFrame(clusters)


def saved_aggregation(selection, selection_dir):
    candidates = pd.read_csv(Path(selection_dir) / "validation_candidates.csv")
    annual = pd.read_csv(Path(selection_dir) / "validation_by_year.csv")
    names = [c["name"] for c in selection["spec"]["candidates"]]
    if candidates["name"].duplicated().any() or set(candidates["name"]) != set(names):
        raise ValueError("Saved candidate table does not match the frozen grid.")
    selected = selection["selected_candidate"]["name"]
    selected_row = candidates.loc[candidates["name"] == selected].iloc[0]
    for key in ("delta", "lcb", "n_common_rows"):
        if not np.isclose(selected_row[key], selection["selected_validation_result"][key]):
            raise ValueError("Saved candidate table differs from the frozen selected result.")
    records = []
    for _, row in candidates.iterrows():
        group = annual.loc[annual["candidate"] == row["name"]]
        if group["year"].duplicated().any() or set(group["year"]) != set(selection["selection_years"]):
            raise ValueError("Saved annual table has missing or duplicate validation years.")
        if int(group["n_common_rows"].sum()) != selection["common_cohort"]["n_rows"]:
            raise ValueError("Saved annual cohort counts do not sum to the frozen cohort.")
        usable = np.isfinite(group["delta"]).all() and group["n_common_rows"].sum() > 0
        average = float(np.average(group["delta"], weights=group["n_common_rows"])) if usable else float("nan")
        records.append({"candidate": row["name"], "frozen_selected": row["name"] == selected,
                        "reported_pooled_delta": row["delta"], "reported_pooled_ci_lo": row["ci_lo"],
                        "reported_pooled_ci_hi": row["ci_hi"], "annual_row_weighted_delta": average,
                        "pooled_minus_annual": row["delta"] - average,
                        "annual_average_has_new_confidence_interval": False,
                        "validation_lcb_positive": bool(np.isfinite(row["lcb"]) and row["lcb"] > 0)})
    return pd.DataFrame(records)


def saved_cost_check(selection, final):
    prices = selection["spec"]["prices"]
    rates = final["diagnostics"]["fertilizer_on_cohort"]["delta_pi1_minus_logged"]
    cost = sum(prices["price" + k] * rates[k] for k in ("N", "P", "K"))
    records = []
    for est in ("sndr", "dr"):
        revenue = prices["maize_price"] * final["yield"][est]["delta"]
        naive = revenue - cost
        records.append({"estimator": est, "reported_profit_delta": final["profit"][est]["delta"],
                        "yield_implied_revenue_delta": revenue, "descriptive_cost_delta_vs_logged": cost,
                        "naive_revenue_minus_cost_delta": naive,
                        "unreconciled_gap": final["profit"][est]["delta"] - naive,
                        "matched_baseline_check": False,
                        "interpretation": "Different baselines and separate outcome models; not an alternative OPE profit estimate. Refit audit needed."})
    return pd.DataFrame(records)


def bin_table(audits):
    records, previous = [], {}
    for audit in sorted(audits, key=lambda x: x["year"]):
        for nutrient, boundaries in audit["bin_edges"].items():
            for i, value in enumerate(boundaries):
                key = nutrient, i
                records.append({"year": audit["year"], "nutrient": nutrient,
                                "boundary_index": i, "boundary_kg_ha": value,
                                "interior": 0 < i < len(boundaries) - 1,
                                "change_from_previous_year": value - previous[key] if key in previous else float("nan"),
                                "n_bins": len(boundaries) - 1})
                previous[key] = value
    return pd.DataFrame(records)


def summarize_run(selection_path, data_path, final_dir, output):
    selection, final, data = load_original(selection_path, data_path, final_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Audit output directory must be new; preserve previous/partial outputs.")
    holdout_path = Path(final_dir) / "holdout_rows.csv"
    holdout = pd.read_csv(holdout_path)
    cohort, clusters = cohort_tables(data, holdout, selection["spec"]["holdout_year"], selection["spec"]["prices"], data_path)
    aggregate = saved_aggregation(selection, Path(selection_path).parent)
    audits = _json(Path(selection_path).parent / "split_audit.json")
    bins = bin_table([*audits, final["fit_audit"]])
    output.mkdir(parents=True)
    cohort.to_csv(output / "retained_excluded.csv", index=False)
    clusters.to_csv(output / "final_cluster_counts.csv", index=False)
    aggregate.to_csv(output / "saved_aggregation_comparison.csv", index=False)
    saved_cost_check(selection, final).to_csv(output / "saved_descriptive_cost_check.csv", index=False)
    bins.to_csv(output / "bin_boundaries.csv", index=False)
    report = {
        "notice": NOTICE, "complete_summary": True, "refit_performed": False,
        "selected_candidate": selection["selected_candidate"],
        "validation_lcb_positive": bool(selection["selected_validation_result"]["lcb"] > 0),
        "original_primary_lcb_positive": bool(final["primary_lcb_positive"]),
        "selection_is_not_acceptance": "The original runner ranks estimable candidates; a negative validation LCB is not evidence of safe improvement.",
        "model_level_accounting_available": False,
        "data_sha256": selection["data_sha256"],
        "selection_sha256": revision.sha256_file(selection_path),
        "original_source_sha256": selection["source_sha256"],
        "audit_source_sha256": _audit_sources(),
        "input_sha256": {str(p): revision.sha256_file(p) for p in (
            Path(final_dir) / "final_results.json", holdout_path,
            Path(selection_path).parent / "validation_candidates.csv",
            Path(selection_path).parent / "validation_by_year.csv",
            Path(selection_path).parent / "split_audit.json")},
    }
    revision.write_json(output / "summary_report.json", report)
    print("Saved-file audit complete. This stage fits no models.", flush=True)
    print(f"Validation LCB positive: {report['validation_lcb_positive']}. Model-level accounting requires refit.", flush=True)
    return report


def cached_fit(year, data, cfg, identity, cache_dir, *, cache_only=False):
    path = Path(cache_dir) / f"fitted_{year}.npz"
    expected = {"schema_version": 1, "year": int(year), "identity": identity, "fit_config": asdict(cfg)}
    if path.exists():
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            if metadata["cache_identity"] != expected:
                raise ValueError(f"Incompatible cache {path}; use a new cache directory.")
            fitted = {name: source[name] for name in metadata["array_keys"]}
            fitted.update(metadata["other_values"])
        print(f"Reusing fitted predictions for {year}.", flush=True)
    else:
        if cache_only:
            raise FileNotFoundError(f"No cached fit for {year}; first run refit without --cache-only.")
        print(f"Fitting {year} using earlier years only; this may take substantial CPU time...", flush=True)
        fitted = revision.fit_year(year, data, cfg)
        arrays = {k: v for k, v in fitted.items() if isinstance(v, np.ndarray)}
        other = {k: v for k, v in fitted.items() if k not in arrays}
        metadata = {"cache_identity": expected, "array_keys": list(arrays), "other_values": other}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".fitted_{year}_", suffix=".npz", dir=cache_dir)
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays, metadata_json=np.array(json.dumps(revision.clean_json(metadata))))
            os.replace(temporary, path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    if fitted["year"] != year or fitted["cf_use_eval"] or any(max(x["fit_years"]) >= year for x in fitted["fit_audit"]):
        raise ValueError("Cached predictions violate the chronological fitting rule.")
    if not np.array_equal(fitted["test_indices"], np.where(data["year"] == year)[0]):
        raise ValueError("Cached prediction rows do not match the current data split.")
    return fitted, revision.sha256_file(path)


def prediction_metrics(rows):
    records = []
    # Logged-action predictions are identical for all candidates. Use only the
    # frozen candidate when called, avoiding duplicate model-quality summaries.
    for (stage, year), group in rows.groupby(["stage", "year"], sort=False):
        for cohort, mask in (("all", np.ones(len(group), bool)), ("audit_cohort", group["included_cohort"].to_numpy())):
            if not mask.any():
                continue
            for outcome in ("profit", "yield"):
                y, mu = group.loc[mask, outcome].to_numpy(), group.loc[mask, f"mu_{outcome}_logged"].to_numpy()
                error = y - mu
                denominator = float(np.sum((y - y.mean()) ** 2))
                records.append({"stage": stage, "year": year, "cohort": cohort, "outcome": outcome,
                                "n_rows": len(y), "n_clusters": group.loc[mask, "site_key"].nunique(),
                                "r2": 1 - float(np.dot(error, error)) / denominator if denominator > 0 else float("nan"),
                                "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))})
    return pd.DataFrame(records)


def reproduction_table(accounting, selection_path, final):
    records = []
    prior = pd.read_csv(Path(selection_path).parent / "validation_candidates.csv")
    for _, row in accounting.loc[accounting["mode"] == MODES[0]].iterrows():
        if row["stage"] == "validation":
            if row["estimator"] != "sndr":
                continue
            reference = prior.loc[prior["name"] == row["candidate"]].iloc[0]
            expected = {"profit": reference["delta"]}
        else:
            expected = {k: final[k][row["estimator"]]["delta"] for k in ("profit", "yield")}
        for outcome, value in expected.items():
            actual = row.get(outcome + "_delta", float("nan"))
            available = bool(np.isfinite(value))
            records.append({"stage": row["stage"], "candidate": row["candidate"], "estimator": row["estimator"],
                            "outcome": outcome, "original_delta": value, "audit_delta": actual,
                            "difference": actual - value,
                            "reference_available": available,
                            "matches_original": bool(np.isclose(actual, value, atol=1e-5, rtol=1e-8)) if available else None})
    return pd.DataFrame(records)


def refit_run(selection_path, data_path, final_dir, output, cache_dir, *, n_bootstrap=None,
              allow_environment_difference=False, cache_only=False):
    selection, final, _ = load_original(selection_path, data_path, final_dir)
    actual_env = revision.environment()
    same_env = all(actual_env[k] == selection["environment"][k] for k in ("python", "packages"))
    if not same_env and not allow_environment_difference:
        raise ValueError("Research environment differs from the original run. Reuse it, or explicitly allow an audit-only refit with --allow-environment-difference.")
    spec = selection["spec"]
    selected_cfg = revision.make_config(spec, selection["selected_candidate"])
    if asdict(selected_cfg) != selection["selected_config"]:
        raise ValueError("Frozen policy configuration was modified.")
    B = spec["n_bootstrap"] if n_bootstrap is None else n_bootstrap
    if B < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    report = summarize_run(selection_path, data_path, final_dir, output)
    output = Path(output)
    cache_dir = Path(cache_dir)
    identity = {"data_sha256": selection["data_sha256"], "sources": selection["source_sha256"], "environment": actual_env}
    configs = {c["name"]: revision.make_config(spec, c) for c in spec["candidates"]}
    rows, fit_hashes, fits = [], {}, []
    validation_data = revision.load_data(data_path, before_year=spec["holdout_year"])
    for year in selection["selection_years"]:
        fitted, fit_hashes[str(year)] = cached_fit(year, validation_data, next(iter(configs.values())), identity, cache_dir, cache_only=cache_only)
        fits.append(revision.audit_fitted(fitted))
        blocks = {name: revision.engine().apply_policy_to_year(fitted, cfg) for name, cfg in configs.items()}
        common = np.logical_and.reduce([b["keep_mask"] for b in blocks.values()])
        for name, cfg in configs.items():
            frame = rows_from_fit(fitted, blocks[name], cfg, validation_data, name, "validation")
            frame["included_cohort"] = common
            rows.append(frame)
    final_data = revision.load_data(data_path, through_year=spec["holdout_year"])
    year = spec["holdout_year"]
    fitted, fit_hashes[str(year)] = cached_fit(year, final_data, selected_cfg, identity, cache_dir, cache_only=cache_only)
    fits.append(revision.audit_fitted(fitted))
    block = revision.engine().apply_policy_to_year(fitted, selected_cfg)
    rows.append(rows_from_fit(fitted, block, selected_cfg, final_data, selection["selected_candidate"]["name"], "final"))
    rows = pd.concat(rows, ignore_index=True)
    rows.to_csv(output / "audit_rows.csv", index=False)
    revision.write_json(output / "refit_split_audit.json", fits)
    estimates, contributions, annual, cohort_sizes = [], [], [], []
    for (stage, name), group in rows.groupby(["stage", "candidate"], sort=False):
        print(f"Auditing {stage}: {name}...", flush=True)
        info = {"stage": stage, "candidate": name}
        try:
            estimates.extend({**info, **r} for r in audit_estimates(group, spec["weights"], spec["prices"]["maize_price"], B, spec["alpha"]))
            contributions.extend({**info, **r} for r in aggregation_decomposition(group, spec["weights"], spec["prices"]["maize_price"]))
        except InferenceUnavailable as exc:
            estimates.append({**info, "point_status": "unavailable", "reason": str(exc)})
        for y, local in group.groupby("year", sort=False):
            detail = {**info, "year": y}
            cohort_sizes.append({**detail, "n_evaluation_rows": len(local), **cluster_summary(local.loc[local["included_cohort"], "site_key"].to_numpy())})
            try:
                annual.extend({**detail, **r} for r in audit_estimates(local, spec["weights"], spec["prices"]["maize_price"], B, spec["alpha"], modes=(MODES[0],)))
            except InferenceUnavailable as exc:
                annual.append({**detail, "point_status": "unavailable", "reason": str(exc)})
    accounting = pd.DataFrame(estimates)
    accounting.to_csv(output / "accounting_audit.csv", index=False)
    columns = [k for k in ("stage", "candidate", "mode", "estimator", "point_status", "interval_status", "n_rows", "n_clusters", "profit_V1", "profit_V0", "profit_delta", "profit_delta_ci_lo", "profit_delta_ci_hi", "psis_k_pooled", "reason") if k in accounting]
    accounting[columns].to_csv(output / "aggregation_summary.csv", index=False)
    pd.DataFrame(contributions).to_csv(output / "aggregation_contributions.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "annual_accounting.csv", index=False)
    pd.DataFrame(cohort_sizes).to_csv(output / "retained_clusters_by_year.csv", index=False)
    prediction_metrics(rows.loc[rows["candidate"] == selection["selected_candidate"]["name"]]).to_csv(output / "heldout_prediction_metrics.csv", index=False)
    reproduction = reproduction_table(accounting, selection_path, final)
    reproduction.to_csv(output / "reproduction_check.csv", index=False)
    manifest = {
        **report, "complete_refit_audit": True, "refit_performed": True,
        "model_level_accounting_available": True, "environment": actual_env,
        "environment_matches_original": same_env, "bootstrap_replicates": B,
        "alpha": spec["alpha"], "fit_cache_sha256": fit_hashes,
        "all_original_point_estimates_reproduced": bool(len(reproduction) == len(configs) + 4 and reproduction.loc[reproduction["reference_available"], "matches_original"].all()),
        "unavailable_original_comparisons": int((~reproduction["reference_available"]).sum()),
        "original_policy_reselected": False,
        "output_sha256": {p.name: revision.sha256_file(p) for p in sorted(output.glob("*.csv"))},
    }
    revision.write_json(output / "audit_manifest.json", manifest)  # Written last.
    print(f"Complete audit: {output}", flush=True)
    print(f"All original point estimates reproduced: {manifest['all_original_point_estimates_reproduced']}", flush=True)
    print("Check reproduction_check.csv before interpreting alternative estimates. No policy was reselected.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "refit"):
        p = sub.add_parser(name)
        p.add_argument("--selection", default="results/revision_selection/selection.json")
        p.add_argument("--data", default="results/revision_data/data.npz")
        p.add_argument("--final", default="results/revision_final", dest="final_dir")
        p.add_argument("--output", required=True)
        if name == "refit":
            p.add_argument("--cache", default="results/revision_audit_cache", dest="cache_dir")
            p.add_argument("--bootstrap", type=int, dest="n_bootstrap")
            p.add_argument("--cache-only", action="store_true")
            p.add_argument("--allow-environment-difference", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["selection_path"] = args.pop("selection")
    args["data_path"] = args.pop("data")
    try:
        (summarize_run if command == "summary" else refit_run)(**args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
