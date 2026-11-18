"""Temporal validity, fixed actions, matched inference and full cache replay."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_revision_evaluation as old_audit
import learn_and_evaluate_policy as engine
import revision_audit as audit
import run_revision_evaluation as revision
import run_temporal_robustness as runner
import temporal_diagnostics as diagnostics
import temporal_policy_comparison as comparison
from test_revision_audit import accounting_rows
from test_revision_evaluation import CountPropensity, LinearReward, synthetic_data


class DiagnosticTests(unittest.TestCase):
    def test_multiclass_scores_and_top_label_calibration_have_known_values(self):
        summary, bins, cells = diagnostics.propensity_diagnostics(np.array([[0.8, 0.2], [0.3, 0.7]]), [0, 1], [8, 2])
        self.assertAlmostEqual(summary["multiclass_brier_sum"], 0.13)
        self.assertAlmostEqual(summary["log_loss_clipped"], -np.log([0.8, 0.7]).mean())
        self.assertAlmostEqual(summary["top_label_ece"], 0.25)
        self.assertEqual(summary["top_label_accuracy"], 1)
        self.assertEqual(sum(b["n_rows"] for b in bins), 2)
        self.assertAlmostEqual(cells[0]["predicted_frequency"], 0.55)
        self.assertEqual(cells[0]["observed_frequency"], 0.5)
        # No division by number of action classes or flattened zero-cell ECE.
        extra = np.pad(np.array([[0.8, 0.2], [0.3, 0.7]]), ((0, 0), (0, 20)))
        more = diagnostics.propensity_diagnostics(extra, [0, 1], [8, 2] + [0] * 20)[0]
        self.assertAlmostEqual(more["multiclass_brier_sum"], summary["multiclass_brier_sum"])
        self.assertAlmostEqual(more["top_label_ece"], summary["top_label_ece"])

    def test_unseen_cells_are_not_hidden_by_log_loss_clipping(self):
        summary, _, _ = diagnostics.propensity_diagnostics([[1.0, 0.0], [1.0, 0.0]], [0, 1], [5, 0])
        self.assertEqual(summary["zero_logged_probability_fraction"], 0.5)
        self.assertEqual(summary["unseen_training_cell_fraction"], 0.5)
        self.assertTrue(np.isinf(summary["log_loss_unclipped"]))
        self.assertAlmostEqual(summary["log_loss_clipped"], -np.log(1e-12) / 2)
        with self.assertRaisesRegex(ValueError, "Invalid multiclass"):
            diagnostics.propensity_diagnostics([[0.4, 0.4]], [0], [1, 1])

    def test_shifts_report_missingness_and_changes_from_constant_training_values(self):
        data = synthetic_data()
        data["obs"] = np.column_stack([np.where(data["year"] == 2012, 0.1, 0.2), np.where(data["year"] == 2012, np.nan, 3.0)])
        table = diagnostics.shift_table(data, [2013], ["constant", "missing"], {"maize_price": 3.5, "priceN": 16, "priceP": 12, "priceK": 8})
        constant = table.loc[table.variable == "constant"].iloc[0]
        self.assertTrue(constant.training_constant_changes)
        self.assertEqual(constant.evaluation_outside_training_range_fraction, 1)
        self.assertEqual(constant.ks_statistic, 1)
        missing = table.loc[table.variable == "missing"].iloc[0]
        self.assertEqual(missing.training_missing_fraction, 1)
        self.assertEqual(missing.evaluation_missing_fraction, 0)
        self.assertEqual(set(table.loc[table.variable.isin(["yield", "profit"]), "variable_role"]), {"outcome_diagnostic_only"})

    def test_common_cohorts_intersect_variants_and_reject_misaligned_ids(self):
        frames = {}
        for grid in ("adaptive", "fixed"):
            for p in comparison.POLICIES:
                frames[(grid, p)] = pd.DataFrame({"year": [2013] * 4, "field_id": [1, 2, 3, 4], "site_key": ["a", "b", "c", "d"],
                                                "policy_overlap": [True] * 4, "within_reference_range": [True, True, True, False]})
        frames[("adaptive", "guarded_greedy")].loc[0, "policy_overlap"] = False
        frames[("fixed", "supported_argmax")].loc[1, "policy_overlap"] = False
        comparison.attach_cohorts(frames)
        frame = frames[("adaptive", "recorded_rule")]
        np.testing.assert_array_equal(frame.within_grid_common, [False, True, True, True])
        np.testing.assert_array_equal(frame.cross_grid_common, [False, False, True, True])
        np.testing.assert_array_equal(frame.cross_grid_common_reference_range, [False, False, True, False])
        frames[("fixed", "recorded_rule")].loc[0, "field_id"] = 999
        with self.assertRaisesRegex(ValueError, "not aligned"):
            comparison.attach_cohorts(frames)

    def test_paired_contrasts_share_draws_and_baseline_and_report_invalid_inference(self):
        rows = accounting_rows()
        rows["field_id"] = np.arange(len(rows))
        rows["mu_profit_pi1"] += 10
        other = rows.copy()
        other["mu_profit_pi1"] -= 10
        for r in comparison.paired_policy_contrast(rows, other, "raw", 3.5, 20, 0.05):
            self.assertAlmostEqual(r["profit_delta_contrast"], 10)
            self.assertAlmostEqual(r["profit_delta_contrast_ci_lo"], 10)
            self.assertAlmostEqual(r["profit_delta_contrast_ci_hi"], 10)
        for r in comparison.paired_policy_contrast(rows, rows.copy(), "psis", 3.5, 20, 0.05):
            self.assertEqual(r["profit_delta_contrast"], 0)
            self.assertEqual(r["profit_delta_contrast_ci_lo"], 0)
        other["mu_profit_pi0"] += 1
        with self.assertRaisesRegex(ValueError, "same grid and behavior"):
            comparison.paired_policy_contrast(rows, other, "raw", 3.5, 20, 0.05)
        rows["w_raw"] = rows["w_cap"] = (rows.site_key == "a").astype(float)
        result = comparison.paired_policy_contrast(rows, rows.copy(), "raw", 3.5, 100, 0.05)[0]
        self.assertEqual(result["interval_status"], "unavailable")
        self.assertGreater(result["invalid_bootstrap_replicates"], 0)
        rows["site_key"] = "one"
        self.assertIn("two geographic clusters", comparison.paired_policy_contrast(rows, rows.copy(), "raw", 3.5, 20, 0.05)[0]["reason"])


class TemporalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "data.npz"
        self.raw = synthetic_data()
        np.savez_compressed(self.path, **self.raw)
        self.spec = json.loads((ROOT / "configs/revision_evaluation.json").read_text())
        self.spec.update(holdout_year=2015, n_bootstrap=8)
        self.spec["policy"].update(min_support_count=2, spibb_min_deviation_count=2, trim_min_count_logged=2, trim_min_count_target=2)
        self.config = self.root / "original_config.json"
        self.config.write_text(json.dumps(self.spec))
        self.package_config = ROOT / "configs/temporal_robustness.json"
        for name, replacement in (("StackedRewardModel", LinearReward), ("JointPropensityModel", CountPropensity)):
            active = patch.object(engine, name, replacement)
            active.start()
            self.addCleanup(active.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            revision.select_policy(self.config, self.path, self.root / "selection")
            self.selection = self.root / "selection/selection.json"
            revision.evaluate_selected(self.selection, self.path, self.root / "final")
            old_audit.refit_run(self.selection, self.path, self.root / "final", self.root / "audit", self.root / "adaptive")
        self.originals = {str(p): p.read_bytes() for folder in ("selection", "final", "audit", "adaptive") for p in (self.root / folder).rglob("*") if p.is_file()}
        self.data = revision.load_data(self.path)
        self.frozen = json.loads(self.selection.read_text())
        self.cfg = comparison.policy_configs(self.frozen)["recorded_rule"]
        self.reference = diagnostics.make_reference(self.data, json.loads((self.root / "selection/split_audit.json").read_text()), 2012)

    def run_compare(self, folder="compare", **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.compare(self.selection, self.path, self.root / "final", self.root / "audit", self.package_config,
                                  self.root / folder, self.root / "adaptive", self.root / "fixed", **kwargs)

    def test_full_workflow_reproduces_originals_fits_only_fixed_and_replays_without_fits(self):
        with patch.object(revision, "fit_year", side_effect=AssertionError("Adaptive refit forbidden")), \
             patch.object(comparison, "fit_fixed_year", wraps=comparison.fit_fixed_year) as fitting:
            result = self.run_compare()
            self.assertEqual(fitting.call_count, 3)
        self.assertTrue(result["all_original_point_estimates_reproduced"])
        self.assertTrue(result["first_year_grid_control_passed"])
        self.assertEqual(result["model_fits_performed"], 3)
        self.assertFalse(result["new_policy_selected"])
        for path, content in self.originals.items():
            self.assertEqual(Path(path).read_bytes(), content)
        with patch.object(revision, "fit_year", side_effect=AssertionError("No fits")), \
             patch.object(comparison, "fit_fixed_year", side_effect=AssertionError("No fits")):
            replay = self.run_compare("replay", cache_only=True)
        self.assertEqual(replay["model_fits_performed"], 0)
        for name in ("policy_estimates.csv", "paired_policy_contrasts.csv", "comparison_rows.csv", "propensity_scores.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(self.root / "compare" / name), pd.read_csv(self.root / "replay" / name))
        rows = pd.read_csv(self.root / "compare/comparison_rows.csv")
        for _, group in rows.groupby(["year", "grid_mode"]):
            for column in comparison.COHORTS[1:]:
                masks = [g[column].to_numpy() for _, g in group.groupby("policy")]
                for other in masks[1:]:
                    np.testing.assert_array_equal(masks[0], other)
        for split in json.loads((self.root / "compare/split_audit.json").read_text()):
            self.assertLess(max(split["model_fits"][0]["fit_years"]), split["year"])
            if split["grid_mode"] == "fixed":
                self.assertTrue(split["open_upper_tail"])
                self.assertIsNone(split["bin_edges"]["N"][-1])
                self.assertTrue(np.isfinite(split["physical_action_grid"]).all())

    def test_diagnose_uses_saved_files_without_fitting_and_historical_baselines_exclude_future(self):
        with patch.object(revision, "fit_year", side_effect=AssertionError("No fits")), \
             patch.object(comparison, "fit_fixed_year", side_effect=AssertionError("No fits")), contextlib.redirect_stdout(io.StringIO()):
            result = runner.diagnose(self.selection, self.path, self.root / "final", self.root / "audit", self.package_config, self.root / "diagnose")
        self.assertEqual(result["model_fits_performed"], 0)
        table = pd.read_csv(self.root / "diagnose/forward_prediction_metrics.csv")
        row = table.loc[(table.year == 2013) & (table.outcome == "yield") & (table.cohort == "all") & (table.predictor == "historical_training_mean")].iloc[0]
        self.assertAlmostEqual(row.predicted_mean, self.data["reward"][self.data["year"] == 2012].mean())
        self.assertTrue((self.root / "diagnose/findings.md").exists())
        with self.assertRaises(FileExistsError):
            runner.diagnose(self.selection, self.path, self.root / "final", self.root / "audit", self.package_config, self.root / "diagnose")

    def test_fixed_grid_is_finite_frozen_uses_open_tails_and_flags_reference_extrapolation(self):
        binner = comparison.ReferenceBinner(self.reference)
        original_centers = {k: v.copy() for k, v in binner.bin_centers.items()}
        large = np.array([[0, 0, 0], [10000, 10000, 10000]], float)
        binner.fit(large, np.array([1e9, -1e9]), ["N", "P", "K"])
        indices = binner.transform(large, ["N", "P", "K"])
        np.testing.assert_array_equal(indices[0], [0, 0, 0])
        np.testing.assert_array_equal(indices[1], [len(original_centers[k]) - 1 for k in diagnostics.NUTRIENTS])
        for k in diagnostics.NUTRIENTS:
            np.testing.assert_array_equal(binner.bin_centers[k], original_centers[k])
            self.assertTrue(np.isfinite(binner.bin_centers[k]).all())
            self.assertTrue(np.isinf(binner.bin_edges[k][-1]))
        self.assertFalse(diagnostics.reference_range_flags(large, self.reference)["within_reference_range"].any())
        changed = {k: v.copy() for k, v in self.data.items()}
        changed["actions"][changed["year"] > 2012] += 10000
        splits = json.loads((self.root / "selection/split_audit.json").read_text())
        self.assertEqual(self.reference, diagnostics.make_reference(changed, splits, 2012))
        splits[0]["model_fits"][0]["fit_years"] = [2012, 2013]
        with self.assertRaisesRegex(ValueError, "earliest training year only"):
            diagnostics.make_reference(self.data, splits, 2012)

    def test_fixed_fit_ignores_future_outcomes_and_never_mutates_original_engine(self):
        original_binner = engine.AdaptiveBinner
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = comparison.fit_fixed_year(2013, self.data, self.cfg, self.reference)
            changed = {k: v.copy() for k, v in self.data.items()}
            changed["reward"][changed["year"] >= 2013] += 100000
            other = comparison.fit_fixed_year(2013, changed, self.cfg, self.reference)
        for key in ("grid", "counts", "pi0_grid_all", "mu_grid_profit", "mu_grid_yield", "mu_logged_profit"):
            np.testing.assert_array_equal(fitted[key], other[key])
        self.assertIs(engine.AdaptiveBinner, original_binner)
        self.assertNotIn("_fertilizer_temporal_fixed_engine", sys.modules)
        with patch.object(engine, "StackedRewardModel", side_effect=RuntimeError("intentional fit error")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                comparison.fit_fixed_year(2013, self.data, self.cfg, self.reference)
        self.assertIs(engine.AdaptiveBinner, original_binner)
        self.assertNotIn("_fertilizer_temporal_fixed_engine", sys.modules)

    def test_missing_or_changed_audit_and_caches_fail_before_new_fits(self):
        with self.assertRaisesRegex(FileNotFoundError, "No fixed-grid cache"):
            self.run_compare(cache_only=True)
        path = self.root / "adaptive/fitted_2013.npz"
        path.write_bytes(path.read_bytes() + b"changed")
        with patch.object(comparison, "fit_fixed_year", side_effect=AssertionError("Must not fit")):
            with self.assertRaisesRegex(ValueError, "differs from the previous audit"):
                self.run_compare("changed_cache")
        path.write_bytes(self.originals[str(path)])
        path = self.root / "audit/audit_rows.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "audit output changed"):
            self.run_compare("changed_audit")

    def test_failed_original_reproduction_prevents_any_fixed_grid_fit(self):
        failed = pd.DataFrame({"reference_available": [True] * 8, "matches_original": [False] * 8})
        with patch.object(runner, "reproduce_adaptive", return_value=failed), \
             patch.object(comparison, "fit_fixed_year", side_effect=AssertionError("Must not fit")):
            with self.assertRaisesRegex(ValueError, "did not reproduce"):
                self.run_compare()
        self.assertTrue((self.root / "compare/original_reproduction_check.csv").exists())
        self.assertFalse((self.root / "compare/temporal_manifest.json").exists())
        self.assertFalse((self.root / "fixed").exists())

    def test_first_year_prediction_mismatch_stops_matching_environment_run(self):
        real_fit = comparison.fit_fixed_year
        def changed_fit(*args, **kwargs):
            fitted = real_fit(*args, **kwargs)
            fitted["mu_grid_profit"] += 10
            return fitted
        with patch.object(comparison, "fit_fixed_year", side_effect=changed_fit) as fitting:
            with self.assertRaisesRegex(ValueError, "First-year fixed/adaptive control failed"):
                self.run_compare()
        self.assertEqual(fitting.call_count, 1)
        control = pd.read_csv(self.root / "compare/first_year_grid_control.csv")
        self.assertFalse(control.loc[control.component == "mu_grid_profit", "matches"].iloc[0])
        self.assertFalse((self.root / "compare/temporal_manifest.json").exists())

    def test_explicit_different_environment_preserves_old_predictions_and_flags_control(self):
        actual = revision.environment()
        real_fit = comparison.fit_fixed_year
        def changed_fit(*args, **kwargs):
            fitted = real_fit(*args, **kwargs)
            fitted["mu_grid_profit"] += 10
            return fitted
        with patch.object(revision, "environment", return_value={**actual, "python": "different"}), \
             patch.object(comparison, "fit_fixed_year", side_effect=changed_fit):
            result = self.run_compare(allow_environment_difference=True)
        self.assertTrue(result["all_original_point_estimates_reproduced"])
        self.assertTrue(result["first_year_grid_control_passed"])
        self.assertFalse(result["first_year_predictions_reproduced"])
        self.assertFalse(result["environment_matches_audit"])
        for path, content in self.originals.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_changed_environment_config_or_fixed_cache_identity_is_rejected(self):
        actual = revision.environment()
        with patch.object(revision, "environment", return_value={**actual, "python": "different"}):
            with self.assertRaisesRegex(ValueError, "environment differs"):
                self.run_compare()
        self.run_compare()
        path = self.root / "fixed/fixed_2013.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in source.files}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["cache_identity"]["reference"]["reference_year"] = 2013
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "Incompatible fixed-grid cache"):
            self.run_compare("bad_fixed", cache_only=True)
        changed = json.loads(self.package_config.read_text())
        changed["reference_year"] = 2013
        config = self.root / "changed_config.json"
        config.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "do not tune"):
            runner.load_config(config)

    def test_policy_definitions_keep_support_and_ablated_guard_explicit(self):
        cfgs = comparison.policy_configs(self.frozen)
        self.assertEqual(cfgs["guarded_greedy"].spibb_min_deviation_count, self.cfg.spibb_min_deviation_count)
        self.assertEqual(cfgs["guarded_greedy"].epsilon_explore, 0)
        self.assertEqual(cfgs["guarded_greedy"].lambda_mix, 0)
        self.assertEqual(cfgs["supported_argmax"].spibb_min_deviation_count, 0)
        self.assertEqual(cfgs["supported_argmax"].min_support_count, self.cfg.min_support_count)
        self.assertEqual(cfgs["supported_argmax"].trim_min_count_target, self.cfg.trim_min_count_target)

    def test_first_year_control_with_real_boosting_models(self):
        import bandit_models as models
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        real_propensity = models.JointPropensityModel
        with patch.object(engine, "StackedRewardModel", models.StackedRewardModel), \
             patch.object(models.StackedRewardModel, "_create_xgb", lambda self: xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)), \
             patch.object(models.StackedRewardModel, "_create_lgb", lambda self: lgb.LGBMRegressor(n_estimators=3, max_depth=2, n_jobs=1, verbose=-1, random_state=42)), \
             patch.object(models.StackedRewardModel, "_create_cat", lambda self: cb.CatBoostRegressor(iterations=3, depth=2, thread_count=1, verbose=False, allow_writing_files=False)), \
             patch.object(engine, "JointPropensityModel", lambda: real_propensity(base_params={"n_estimators": 3, "n_jobs": 1})), contextlib.redirect_stdout(io.StringIO()):
            adaptive = revision.fit_year(2013, self.data, self.cfg)
            fixed = comparison.fit_fixed_year(2013, self.data, self.cfg, self.reference)
        for key in ("grid", "counts", "joint_te", "pi0_grid_all", "mu_grid_profit", "mu_grid_yield", "mu_logged_profit"):
            np.testing.assert_allclose(adaptive[key], fixed[key], rtol=1e-9, atol=1e-7)
        block = engine.apply_policy_to_year(fixed, self.cfg)
        rows = audit.rows_from_fit(fixed, block, self.cfg, self.data, "recorded_rule", "validation")
        self.assertTrue(np.isfinite(rows.cost_pi1).all())
        score, _, _ = diagnostics.propensity_diagnostics(fixed["pi0_grid_all"], fixed["joint_te"], fixed["counts"])
        self.assertTrue(np.isfinite(score["multiclass_brier_sum"]))


if __name__ == "__main__":
    unittest.main()
