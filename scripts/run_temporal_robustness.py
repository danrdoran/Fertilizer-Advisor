"""Package 1: diagnose temporal reliability and compare grids/simple policies.

diagnose fits nothing. compare reuses audited adaptive predictions, reproduces
the original point estimates, then fits a fixed physical grid using prior years.
All new comparisons are exploratory after inspection of the 2018 results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_revision_evaluation as original_audit
from ope_statistics import InferenceUnavailable, cluster_summary
import revision_audit as audit
import run_revision_evaluation as revision
import temporal_diagnostics as diagnostics
import temporal_policy_comparison as comparison


NOTICE = (
    "Exploratory analyses specified after inspection of validation and 2018 results. "
    "No policy is reselected and no new independent holdout is claimed. The fixed "
    "grid retains cuts and finite midpoint rates learned using 2012 only; its "
    "recorded_rule is a new application of the frozen construction rule, not the "
    "original policy. Common rows do not make behavior baselines identical across "
    "grids. Cluster intervals condition on fitted models, weights and inclusion, "
    "are not adjusted for multiple comparisons, and do not resolve confounding, "
    "within-bin treatment versions or action-dependent trimming."
)
SOURCES = ("scripts/run_temporal_robustness.py", "scripts/temporal_diagnostics.py", "scripts/temporal_policy_comparison.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path):
    config = read_json(path)
    if (set(config) != {"schema_version", "reference_year", "policies", "reliability_bins", "log_loss_probability_floor", "aggregation_modes"}
            or config["schema_version"] != 1 or config["policies"] != list(comparison.POLICIES)
            or config["reference_year"] != 2012 or config["reliability_bins"] != 10
            or config["log_loss_probability_floor"] != 1e-12
            or config["aggregation_modes"] != [audit.MODES[0], audit.MODES[2]]):
        raise ValueError("Unexpected package-1 configuration; do not tune it using the final year.")
    return config


def load_inputs(selection_path, data_path, final_dir, audit_dir, config_path):
    selection, final, data = original_audit.load_original(selection_path, data_path, final_dir)
    config = load_config(config_path)
    manifest = read_json(Path(audit_dir) / "audit_manifest.json")
    if (not manifest.get("complete_refit_audit") or not manifest.get("all_original_point_estimates_reproduced")
            or manifest.get("unavailable_original_comparisons", 0) != 0
            or manifest.get("original_policy_reselected") is not False):
        raise ValueError("A complete, successfully reproduced previous refit audit is required.")
    if (manifest["data_sha256"] != selection["data_sha256"] or
            manifest["selection_sha256"] != revision.sha256_file(selection_path) or
            manifest["original_source_sha256"] != selection["source_sha256"] or
            manifest["audit_source_sha256"] != original_audit._audit_sources()):
        raise ValueError("Previous audit provenance does not match this frozen run/source.")
    inputs = [Path(selection_path), Path(data_path), Path(final_dir) / "final_results.json"]
    by_name = {Path(name).name: digest for name, digest in manifest["input_sha256"].items()}
    for folder, names in ((Path(final_dir), ("final_results.json", "holdout_rows.csv")),
                          (Path(selection_path).parent, ("validation_candidates.csv", "validation_by_year.csv", "split_audit.json"))):
        for name in names:
            path = folder / name
            if revision.sha256_file(path) != by_name[name]:
                raise ValueError(f"Previous original input changed: {path}.")
            inputs.append(path)
    required = {"audit_rows.csv", "reproduction_check.csv", "aggregation_contributions.csv"}
    if not required.issubset(manifest["output_sha256"]):
        raise ValueError("Previous audit is missing required output fingerprints.")
    for name, digest in manifest["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Audit output filenames must be local basenames.")
        path = Path(audit_dir) / name
        if revision.sha256_file(path) != digest:
            raise ValueError(f"Previous audit output changed: {path}.")
        inputs.append(path)
    inputs += [Path(audit_dir) / "audit_manifest.json", Path(config_path)]
    splits = read_json(Path(selection_path).parent / "split_audit.json")
    if [s["year"] for s in splits] != selection["selection_years"]:
        raise ValueError("Split audit years do not match the frozen selection.")
    reference = diagnostics.make_reference(data, splits, config["reference_year"])
    reference["source_split_audit_sha256"] = revision.sha256_file(Path(selection_path).parent / "split_audit.json")
    reference["source_selection_sha256"] = revision.sha256_file(selection_path)
    selected = pd.read_csv(Path(audit_dir) / "audit_rows.csv")
    selected = selected.loc[selected["candidate"] == selection["selected_candidate"]["name"]].copy()
    for key in ("included_cohort", "policy_overlap"):
        selected[key] = original_audit._boolean(selected[key])
    expected_years = [*selection["selection_years"], selection["spec"]["holdout_year"]]
    if sorted(selected["year"].unique().tolist()) != expected_years:
        raise ValueError("Previous audit rows do not cover the frozen evaluated years.")
    for year, frame in selected.groupby("year", sort=True):
        idx = np.where(data["year"] == year)[0]
        if not np.array_equal(frame["field_id"], data["field_id"][idx]):
            raise ValueError("Previous audit row IDs/order differ from the frozen data.")
        np.testing.assert_allclose(frame["yield"], data["reward"][idx], rtol=1e-12, atol=1e-8)
        prices = selection["spec"]["prices"]
        profit = prices["maize_price"] * data["reward"][idx] - data["actions"][idx] @ np.array([prices["price" + k] for k in diagnostics.NUTRIENTS])
        np.testing.assert_allclose(frame["profit"], profit, rtol=1e-12, atol=1e-8)
    provenance = {"notice": NOTICE, "schema_version": 1, "original_policy_reselected": False,
                  "original_source_sha256": selection["source_sha256"],
                  "package_source_sha256": {name: revision.sha256_file(revision.ROOT / name) for name in SOURCES},
                  "input_sha256": {str(p): revision.sha256_file(p) for p in inputs},
                  "config": config, "reference": reference, "environment": revision.environment()}
    return selection, final, data, manifest, config, splits, selected, provenance


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def finish(output, provenance, **extra):
    manifest = {**provenance, **extra, "complete": True,
                "output_sha256": {p.name: revision.sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    revision.write_json(output / "temporal_manifest.json", manifest)  # Completion marker written last.
    return manifest


def diagnose(selection_path, data_path, final_dir, audit_dir, config_path, output):
    selection, final, data, prior, config, splits, rows, provenance = load_inputs(selection_path, data_path, final_dir, audit_dir, config_path)
    output = new_output(output)
    prices = selection["spec"]["prices"]
    years = [*selection["selection_years"], selection["spec"]["holdout_year"]]
    all_splits = [*splits, final["fit_audit"]]
    reference = provenance["reference"]
    revision.write_json(output / "fixed_grid_reference.json", reference)
    metrics = diagnostics.prediction_table(rows, data, prices)
    metrics.to_csv(output / "forward_prediction_metrics.csv", index=False)
    diagnostics.shift_table(data, years, diagnostics.feature_names(data_path, data["obs"].shape[1]), prices).to_csv(output / "distribution_shift.csv", index=False)
    diagnostics.bin_stability_table(data, all_splits, reference).to_csv(output / "adaptive_bin_stability.csv", index=False)
    diagnostics.residual_cell_table(rows, data, {("adaptive", s["year"]): s["bin_edges"] for s in all_splits}).to_csv(output / "residuals_by_action_cell.csv", index=False)
    range_records = []
    for year in years:
        flags = diagnostics.reference_range_flags(data["actions"][data["year"] == year], reference)
        range_records.append({"year": year, "n_evaluation_rows": len(flags["within_reference_range"]),
                              **{k + "_fraction": float(v.mean()) for k, v in flags.items()}})
    pd.DataFrame(range_records).to_csv(output / "reference_range_coverage.csv", index=False)
    contribution = pd.read_csv(Path(audit_dir) / "aggregation_contributions.csv")
    contribution = contribution.loc[(contribution["candidate"] == selection["selected_candidate"]["name"]) & (contribution["stage"] == "validation")].copy()
    contribution.to_csv(output / "recorded_validation_contributions.csv", index=False)
    model = metrics.loc[(metrics.predictor == "outcome_model") & (metrics.outcome == "profit") & (metrics.cohort == "all")]
    lines = ["# Forward-year diagnostic findings", "", NOTICE, "", "No models were fitted. Prediction bias below is predicted minus observed profit (MXN/ha).", ""]
    lines += [f"- {int(r.year)}: R² {r.r2:.3f}; MAE {r.mae:.1f}; bias {r.prediction_minus_observed_bias:+.1f}." for r in model.itertuples()]
    lines += ["", "Compare the outcome model with the two historical-mean baselines in forward_prediction_metrics.csv. Negative R² means it predicts worse than the evaluated year's descriptive mean; that mean is not a deployable training-only predictor.",
              "", "Distribution shifts, residuals and changing bin boundaries can identify where reliability deteriorates. They do not establish the cause of that deterioration or show that a replacement grid improves policy value.",
              "", "The original negative pooled validation estimate and the positive 2018 estimate remain part of the record. Use the compare stage to test grid and policy robustness on matched cohorts.", ""]
    (output / "findings.md").write_text("\n".join(lines))
    result = finish(output, provenance, command="diagnose", model_fits_performed=0, prior_audit_reproduced=True)
    print(f"Diagnostics complete: {output}. No models were fitted.", flush=True)
    return result


def point_records(rows, method, maize_price):
    kept, arrays = audit.retained_arrays(rows, maize_price)
    weights, _, _ = audit.prepare_weights(arrays, method)
    return [{"mode": audit.MODES[0], "estimator": est, **audit.point_estimate(arrays, weights, est, maize_price)} for est in ("sndr", "dr")]


def reproduce_adaptive(fits, data, selection, final, selection_path):
    configs = {c["name"]: revision.make_config(selection["spec"], c) for c in selection["spec"]["candidates"]}
    groups = {name: [] for name in configs}
    prices = selection["spec"]["prices"]
    for year in selection["selection_years"]:
        fitted = fits[year]
        blocks = {name: revision.engine().apply_policy_to_year(fitted, cfg) for name, cfg in configs.items()}
        common = np.logical_and.reduce([b["keep_mask"] for b in blocks.values()])
        for name, cfg in configs.items():
            frame = audit.rows_from_fit(fitted, blocks[name], cfg, data, name, "validation")
            frame["included_cohort"] = common
            groups[name].append(frame)
    estimates = []
    for name, frames in groups.items():
        estimates.extend({"stage": "validation", "candidate": name, **r} for r in point_records(pd.concat(frames, ignore_index=True), selection["spec"]["weights"], prices["maize_price"]))
    name = selection["selected_candidate"]["name"]
    fitted = fits[selection["spec"]["holdout_year"]]
    block = revision.engine().apply_policy_to_year(fitted, configs[name])
    frame = audit.rows_from_fit(fitted, block, configs[name], data, name, "final")
    estimates.extend({"stage": "final", "candidate": name, **r} for r in point_records(frame, selection["spec"]["weights"], prices["maize_price"]))
    return original_audit.reproduction_table(pd.DataFrame(estimates), selection_path, final)


def safe_estimates(rows, method, maize_price, B, alpha, modes):
    try:
        return audit.audit_estimates(rows, method, maize_price, B, alpha, modes=modes)
    except InferenceUnavailable as exc:
        return [{"mode": mode, "estimator": est, "point_status": "unavailable", "interval_status": "unavailable", "reason": str(exc), "n_rows": 0} for mode in modes for est in ("sndr", "dr")]


def comparison_tables(rows, spec, config, B, output):
    estimates, coverage, contrasts = [], [], []
    for (stage, grid, policy), group in rows.groupby(["stage", "grid_mode", "policy"], sort=True):
        print(f"Summarizing {stage}: {grid}/{policy}...", flush=True)
        scopes = [("pooled_validation" if stage == "validation" else "final_year", group)]
        if stage == "validation":
            scopes += [(str(year), local) for year, local in group.groupby("year", sort=True)]
        for scope, local in scopes:
            info = {"stage": stage, "scope": scope, "grid_mode": grid, "policy": policy}
            for cohort in comparison.COHORTS:
                frame = local.copy()
                frame["included_cohort"] = frame[cohort].to_numpy(bool)
                mask = frame["included_cohort"]
                coverage.append({**info, "cohort": cohort, "n_evaluation_rows": len(frame), "retained_fraction": float(mask.mean()),
                                 "native_overlap_fraction": float(frame["native_overlap"].mean()),
                                 "reference_range_fraction": float(frame["within_reference_range"].mean()),
                                 "failed_logged_probability_n": int((frame.pi0_logged < spec["policy"]["trim_pi0_tau"]).sum()),
                                 "failed_density_ratio_n": int((frame.w_raw > spec["policy"]["trim_ratio_max"]).sum()),
                                 "failed_logged_count_n": int((frame.count_logged < spec["policy"]["trim_min_count_logged"]).sum()),
                                 "failed_target_count_n": int((frame.count_target < spec["policy"]["trim_min_count_target"]).sum()),
                                 **cluster_summary(frame.loc[mask, "site_key"].to_numpy())})
                modes = config["aggregation_modes"] if scope == "pooled_validation" else [audit.MODES[0]]
                estimates.extend({**info, "cohort": cohort, **r} for r in safe_estimates(frame, spec["weights"], spec["prices"]["maize_price"], B, spec["alpha"], modes))
    for (stage, grid), group in rows.groupby(["stage", "grid_mode"], sort=True):
        scopes = [("pooled_validation" if stage == "validation" else "final_year", group)]
        if stage == "validation":
            scopes += [(str(year), local) for year, local in group.groupby("year", sort=True)]
        for scope, local in scopes:
            for cohort in comparison.COHORTS[1:]:
                frames = {p: g.reset_index(drop=True).copy() for p, g in local.groupby("policy", sort=False)}
                for frame in frames.values():
                    frame["included_cohort"] = frame[cohort].to_numpy(bool)
                for other in comparison.POLICIES[1:]:
                    info = {"stage": stage, "scope": scope, "grid_mode": grid, "cohort": cohort,
                            "left_policy": "recorded_rule", "right_policy": other,
                            "contrast_direction": "recorded_rule_minus_comparator"}
                    modes = config["aggregation_modes"] if scope == "pooled_validation" else [audit.MODES[0]]
                    try:
                        records = comparison.paired_policy_contrast(frames["recorded_rule"], frames[other], spec["weights"], spec["prices"]["maize_price"], B, spec["alpha"], modes=modes)
                    except InferenceUnavailable as exc:
                        records = [{"mode": mode, "estimator": est, "point_status": "unavailable", "interval_status": "unavailable", "reason": str(exc)} for mode in modes for est in ("sndr", "dr")]
                    contrasts.extend({**info, **r} for r in records)
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(output / "policy_estimates.csv", index=False)
    summary = estimates.loc[estimates["scope"].isin(["pooled_validation", "final_year"]) &
                            estimates["cohort"].isin(["cross_grid_common", "cross_grid_common_reference_range"])]
    columns = [k for k in ("stage", "scope", "grid_mode", "policy", "cohort", "mode", "estimator", "point_status", "interval_status",
                           "n_rows", "n_clusters", "profit_V0", "profit_V1", "profit_delta", "profit_delta_ci_lo", "profit_delta_ci_hi",
                           "consistent_profit_delta", "consistent_profit_delta_ci_lo", "consistent_profit_delta_ci_hi",
                           "yield_delta", "matched_cost_delta", "surrogate_gap", "psis_k_pooled", "reason") if k in estimates]
    summary[columns].to_csv(output / "reviewer_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "cohort_coverage.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_policy_contrasts.csv", index=False)


def compare(selection_path, data_path, final_dir, audit_dir, config_path, output, adaptive_cache, fixed_cache,
            *, n_bootstrap=None, cache_only=False, allow_adaptive_refit=False, allow_environment_difference=False):
    selection, final, data, prior, config, splits, saved_rows, provenance = load_inputs(selection_path, data_path, final_dir, audit_dir, config_path)
    actual_env = revision.environment()
    same_env = all(actual_env[k] == prior["environment"][k] for k in ("python", "packages"))
    if not same_env and not allow_environment_difference:
        raise ValueError("Research environment differs from the audited run. Reuse it or explicitly pass --allow-environment-difference for exploratory fits.")
    if cache_only and allow_adaptive_refit:
        raise ValueError("--cache-only cannot be combined with --allow-adaptive-refit.")
    B = selection["spec"]["n_bootstrap"] if n_bootstrap is None else n_bootstrap
    if B < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    configs = comparison.policy_configs(selection)
    spec, reference = selection["spec"], provenance["reference"]
    output = new_output(output)
    revision.write_json(output / "fixed_grid_reference.json", reference)
    # These two identities are intentionally separate: old predictions keep the
    # environment recorded in the old audit; any new fits record today's one.
    original_identity = {"data_sha256": selection["data_sha256"], "sources": selection["source_sha256"], "environment": prior["environment"]}
    fits, cache_hashes, new_fit_count = {}, {"adaptive": {}, "fixed": {}}, 0
    years = [*selection["selection_years"], spec["holdout_year"]]
    validation_data = revision.load_data(data_path, before_year=spec["holdout_year"])
    for year in years:
        cfg = revision.make_config(spec, spec["candidates"][0]) if year < spec["holdout_year"] else configs["recorded_rule"]
        path = Path(adaptive_cache) / f"fitted_{year}.npz"
        existing = path.exists()
        if existing and revision.sha256_file(path) != prior["fit_cache_sha256"][str(year)]:
            raise ValueError(f"Adaptive cache {path} differs from the previous audit; restore that cache.")
        identity = original_identity if existing else {**original_identity, "environment": actual_env}
        fits[year], digest = original_audit.cached_fit(year, validation_data if year < spec["holdout_year"] else data, cfg, identity, adaptive_cache,
                                                     cache_only=cache_only or not allow_adaptive_refit)
        comparison.validate_fit(fits[year], year, data, cfg)
        cache_hashes["adaptive"][str(year)] = digest
        new_fit_count += int(not existing)
    reproduction = reproduce_adaptive(fits, data, selection, final, selection_path)
    reproduction.to_csv(output / "original_reproduction_check.csv", index=False)
    if (len(reproduction) != len(spec["candidates"]) + 4 or not reproduction["reference_available"].all() or not reproduction["matches_original"].all()):
        raise ValueError(f"Original point estimates did not reproduce; inspect {output / 'original_reproduction_check.csv'}. Fixed-grid fitting was not started.")
    first = fits[years[0]]
    for k in diagnostics.NUTRIENTS:
        if not np.array_equal(first["bin_edges"][k], reference["dimensions"][k]["training_edges"]):
            raise ValueError("Reference grid differs from the first audited adaptive fit.")
    print("All original point estimates reproduced. Starting exploratory fixed-grid comparisons.", flush=True)
    fixed_identity = {"data_sha256": selection["data_sha256"], "original_sources": selection["source_sha256"],
                      "model_sources": {name: provenance["package_source_sha256"][name] for name in SOURCES[1:]}, "environment": actual_env}
    frames, fit_records, prop_summary, reliability, calibration_cells, control = [], [], [], [], [], []
    edges_by_year = {}
    for year in years:
        path = Path(fixed_cache) / f"fixed_{year}.npz"
        existing = path.exists()
        fixed, digest = comparison.cached_fixed_fit(year, validation_data if year < spec["holdout_year"] else data, configs["recorded_rule"], reference, fixed_identity, fixed_cache, cache_only=cache_only)
        cache_hashes["fixed"][str(year)] = digest
        new_fit_count += int(not existing)
        stage = "validation" if year < spec["holdout_year"] else "final"
        variants = {}
        for grid, fitted in (("adaptive", fits[year]), ("fixed", fixed)):
            edges_by_year[(grid, year)] = fitted["bin_edges"]
            # Numeric edge arrays in caches preserve open tails; JSON uses null
            # plus an explicit declaration rather than non-standard Infinity.
            fit_records.append({"grid_mode": grid, "open_upper_tail": grid == "fixed", **revision.audit_fitted(fitted),
                                "physical_action_grid": fitted["grid"].tolist()})
            flags = diagnostics.reference_range_flags(fitted["A_te"], reference)
            for policy, cfg in configs.items():
                block = revision.engine().apply_policy_to_year(fitted, cfg)
                frame = audit.rows_from_fit(fitted, block, cfg, data, policy, stage)
                frame["grid_mode"], frame["policy"] = grid, policy
                frame["joint_logged"], frame["joint_target"] = block["joint_logged"], block["joint_target"]
                for k, values in flags.items():
                    frame[k] = values
                variants[(grid, policy)] = frame
        comparison.attach_cohorts(variants)
        frames.extend(variants.values())
        if year == years[0]:
            for key in ("grid", "counts", "joint_te", "pi0_grid_all", "mu_grid_profit", "mu_grid_yield", "mu_logged_profit", "mu_logged_yield"):
                equal = bool(np.allclose(fits[year][key], fixed[key], atol=1e-7, rtol=1e-9))
                required = same_env or key in ("grid", "counts", "joint_te")
                control.append({"year": year, "component": key, "matches": equal, "match_required": required,
                                "max_absolute_difference": float(np.max(np.abs(np.asarray(fits[year][key]) - np.asarray(fixed[key]))))})
            # Identical first-year training/grid should reproduce. This is a
            # numerical control, not permission to tune a later-year grid.
            pd.DataFrame(control).to_csv(output / "first_year_grid_control.csv", index=False)
            if not all(r["matches"] for r in control if r["match_required"]):
                raise ValueError("First-year fixed/adaptive control failed; inspect first_year_grid_control.csv before proceeding.")
        for grid, fitted in (("adaptive", fits[year]), ("fixed", fixed)):
            for cohort in ("all", "within_grid_common", "cross_grid_common", "cross_grid_common_reference_range"):
                mask = np.ones(len(fitted["joint_te"]), bool) if cohort == "all" else variants[(grid, "recorded_rule")][cohort].to_numpy(bool)
                if not mask.any():
                    prop_summary.append({"stage": stage, "year": year, "grid_mode": grid, "cohort": cohort, "status": "unavailable", "n_rows": 0})
                    continue
                historical = fitted["counts"] / fitted["counts"].sum()
                for model, probabilities in (("behavior_model", fitted["pi0_grid_all"][mask]), ("historical_cell_frequency", np.broadcast_to(historical, (int(mask.sum()), len(historical))))):
                    info = {"stage": stage, "year": year, "grid_mode": grid, "cohort": cohort, "predictor": model}
                    summary, bins, cells = diagnostics.propensity_diagnostics(probabilities, fitted["joint_te"][mask], fitted["counts"], config["reliability_bins"], config["log_loss_probability_floor"])
                    prop_summary.append({**info, "status": "available", **summary})
                    reliability.extend({**info, **r} for r in bins)
                    calibration_cells.extend({**info, **r, **{f"representative_{k}_kg_ha": fitted["grid"][r["joint_cell"], j] for j, k in enumerate(diagnostics.NUTRIENTS)}} for r in cells)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(output / "comparison_rows.csv", index=False)
    revision.write_json(output / "split_audit.json", fit_records)
    pd.DataFrame(prop_summary).to_csv(output / "propensity_scores.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "top_label_reliability.csv", index=False)
    pd.DataFrame(calibration_cells).to_csv(output / "classwise_calibration.csv", index=False)
    comparison_tables(rows, spec, config, B, output)
    model_rows = rows.loc[rows["policy"] == "recorded_rule"].copy()
    model_rows["included_cohort"] = model_rows["within_grid_common"]
    diagnostics.prediction_table(model_rows, data, spec["prices"]).to_csv(output / "forward_prediction_metrics.csv", index=False)
    diagnostics.residual_cell_table(model_rows, data, edges_by_year).to_csv(output / "residuals_by_action_cell.csv", index=False)
    result = finish(output, provenance, command="compare", all_original_point_estimates_reproduced=True,
                    first_year_grid_control_passed=True, first_year_predictions_reproduced=all(r["matches"] for r in control), model_fits_performed=new_fit_count,
                    adaptive_refit_permitted=allow_adaptive_refit, environment_matches_audit=same_env,
                    fit_cache_sha256=cache_hashes, bootstrap_replicates=B, alpha=spec["alpha"],
                    prediction_diagnostic_cohort="within_grid_common", new_policy_selected=False)
    print(f"Exploratory comparisons complete: {output}. No policy was reselected.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("diagnose", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--selection", default="results/revision_selection/selection.json", dest="selection_path")
        p.add_argument("--data", default="results/revision_data/data.npz", dest="data_path")
        p.add_argument("--final", default="results/revision_final", dest="final_dir")
        p.add_argument("--audit", default="results/revision_audit_refit", dest="audit_dir")
        p.add_argument("--config", default=str(revision.ROOT / "configs/temporal_robustness.json"), dest="config_path")
        p.add_argument("--output", required=True)
        if command == "compare":
            p.add_argument("--adaptive-cache", default="results/revision_audit_cache")
            p.add_argument("--fixed-cache", default="results/revision_temporal_cache")
            p.add_argument("--bootstrap", type=int, dest="n_bootstrap")
            p.add_argument("--cache-only", action="store_true")
            p.add_argument("--allow-adaptive-refit", action="store_true")
            p.add_argument("--allow-environment-difference", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        (diagnose if command == "diagnose" else compare)(**args)
    except (ValueError, RuntimeError, OSError, KeyError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
