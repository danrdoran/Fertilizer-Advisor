"""Numerical identities, misleading aggregates, replay safety and real run I/O."""

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
import audit_revision_evaluation as runner
import learn_and_evaluate_policy as engine
import revision_audit as audit
import run_revision_evaluation as revision
from test_revision_evaluation import CountPropensity, LinearReward, synthetic_data


def accounting_rows():
    n = 16
    y = np.linspace(1000, 2000, n)
    costs = np.linspace(100, 150, n)
    logged = y - np.tile([5.0, -3.0], n // 2)
    row = pd.DataFrame({
        "year": np.repeat([2013, 2014], n // 2),
        "site_key": np.tile(np.repeat(["a", "b", "c", "d"], 2), 2),
        "included_cohort": True, "yield": y, "profit": 3.5 * y - costs,
        "mu_yield_logged": logged, "mu_yield_pi0": logged + 2,
        "mu_yield_pi1": logged + 22, "cost_logged": costs,
        "cost_pi0": costs + 7, "cost_pi1": costs + 17,
        "w_raw": np.linspace(0.3, 2, n), "w_cap": np.linspace(0.3, 2, n),
    })
    for key in ("logged", "pi0", "pi1"):
        row["mu_profit_" + key] = 3.5 * row["mu_yield_" + key] - row["cost_" + key]
    return row


class NumericalAuditTests(unittest.TestCase):
    def test_accounting_identity_for_both_estimators_and_every_aggregation(self):
        for method in ("raw", "cap", "psis"):
            results = audit.audit_estimates(accounting_rows(), method, 3.5, 20, 0.05)
            for result in results:
                self.assertEqual(result["point_status"], "estimable")
                self.assertAlmostEqual(result["surrogate_gap"], 0, places=9)
                self.assertAlmostEqual(result["gap_identity_error"], 0, places=9)
                self.assertAlmostEqual(result["naive_identity_error"], 0, places=9)
                self.assertAlmostEqual(result["profit_delta"] - result["naive_profit_delta"], 7, places=9)
                self.assertEqual(result["interval_status"], "available")

    def test_separate_surrogate_bias_is_measured_rather_than_hidden(self):
        rows = accounting_rows()
        rows["mu_profit_pi1"] += 75
        results = audit.audit_estimates(rows, "psis", 3.5, 20, 0.05)
        for result in results:
            self.assertAlmostEqual(result["surrogate_gap"], 75, places=8)
            self.assertAlmostEqual(result["gap_direct"], 75, places=8)
            self.assertAlmostEqual(result["gap_residual"], 0, places=8)
            self.assertAlmostEqual(result["surrogate_gap_ci_lo"], 75, places=8)
            self.assertAlmostEqual(result["surrogate_gap_ci_hi"], 75, places=8)

    def test_positive_annual_gains_can_have_negative_global_sndr(self):
        rows = accounting_rows()
        rows["yield"] = np.repeat([110.0, 90.0], 8)
        rows["profit"] = rows["yield"] - 100
        rows["cost_logged"] = rows["cost_pi0"] = rows["cost_pi1"] = 100.0
        rows["mu_profit_logged"] = rows["mu_profit_pi0"] = 0.0
        rows["mu_profit_pi1"] = 1.0
        rows["mu_yield_logged"] = rows["mu_yield_pi0"] = 100.0
        rows["mu_yield_pi1"] = 101.0
        rows["w_raw"] = rows["w_cap"] = np.repeat([1.0, 100.0], 8)
        results = audit.audit_estimates(rows, "raw", 1.0, 20, 0.05)
        sndr = {r["mode"]: r["profit_delta"] for r in results if r["estimator"] == "sndr"}
        self.assertAlmostEqual(sndr[audit.MODES[0]], 1 + (8 * 10 - 8 * 100 * 10) / (8 + 800))
        self.assertLess(sndr[audit.MODES[0]], 0)
        self.assertAlmostEqual(sndr[audit.MODES[2]], 1)
        pieces = pd.DataFrame(audit.aggregation_decomposition(rows, "raw", 1.0))
        self.assertAlmostEqual(pieces["pooled_global_contribution"].sum(), sndr[audit.MODES[0]])
        self.assertAlmostEqual(pieces["yearwise_local_contribution"].sum(), sndr[audit.MODES[2]])

    def test_smoothing_and_normalization_contributions_reconstruct_estimates(self):
        rows = accounting_rows()
        rows.loc[rows.year == 2014, "w_raw"] *= 3
        results = audit.audit_estimates(rows, "psis", 3.5, 20, 0.05)
        estimates = {r["mode"]: r["profit_delta"] for r in results if r["estimator"] == "sndr"}
        pieces = pd.DataFrame(audit.aggregation_decomposition(rows, "psis", 3.5))
        for column, mode in zip(("pooled_global_contribution", "yearwise_global_contribution", "yearwise_local_contribution"), audit.MODES):
            self.assertAlmostEqual(pieces[column].sum(), estimates[mode], places=9)
        self.assertAlmostEqual(pieces["smoothing_shift_contribution"].sum(), estimates[audit.MODES[1]] - estimates[audit.MODES[0]], places=9)
        self.assertAlmostEqual(pieces["normalization_shift_contribution"].sum(), estimates[audit.MODES[2]] - estimates[audit.MODES[1]], places=9)

    def test_ordinary_dr_is_additive_across_years_with_identical_weights(self):
        results = audit.audit_estimates(accounting_rows(), "cap", 3.5, 20, 0.05)
        dr = {r["mode"]: r for r in results if r["estimator"] == "dr"}
        for key in audit.INTERVAL_METRICS:
            self.assertAlmostEqual(dr[audit.MODES[1]][key], dr[audit.MODES[2]][key], places=9)

    def test_null_policy_has_zero_gain_in_every_mode(self):
        rows = accounting_rows()
        rows["mu_yield_pi1"] = rows["mu_yield_pi0"]
        rows["mu_profit_pi1"] = rows["mu_profit_pi0"]
        rows["cost_pi1"] = rows["cost_pi0"]
        rows["w_raw"] = rows["w_cap"] = 1.0
        for result in audit.audit_estimates(rows, "psis", 3.5, 20, 0.05):
            for key in ("profit_delta", "yield_delta", "matched_cost_delta"):
                self.assertAlmostEqual(result[key], 0, places=9)

    def test_legacy_point_values_match_original_engine(self):
        rows = accounting_rows()
        rows["mu_profit_pi1"] += 30
        a = {key: rows[key].to_numpy() for key in audit.ARRAY_COLUMNS}
        w, _, _ = audit.prepare_weights(a, "psis")
        idx = np.arange(len(rows))
        from ope_statistics import _dr_value, _sndr_value
        for est, fn in (("dr", _dr_value), ("sndr", _sndr_value)):
            result = audit.point_estimate(a, w, est, 3.5)
            expected = fn(a["mu_profit_pi1"], w, a["profit"], a["mu_profit_logged"], idx) - fn(a["mu_profit_pi0"], np.ones(len(rows)), a["profit"], a["mu_profit_logged"], idx)
            self.assertAlmostEqual(result["profit_delta"], expected, places=10)

    def test_missing_weight_support_is_reported_without_discarding_draws(self):
        rows = accounting_rows()
        rows["w_raw"] = rows["w_cap"] = (rows.site_key == "a").astype(float)
        result = audit.audit_estimates(rows, "raw", 3.5, 100, 0.05)[0]
        self.assertEqual(result["point_status"], "estimable")
        self.assertEqual(result["interval_status"], "unavailable")
        self.assertGreater(result["invalid_bootstrap_replicates"], 0)
        self.assertTrue(np.isnan(result["profit_delta_ci_lo"]))

    def test_zero_weight_year_and_small_cluster_count_are_explicit(self):
        rows = accounting_rows()
        rows.loc[rows.year == 2014, ["w_raw", "w_cap"]] = 0
        results = audit.audit_estimates(rows, "raw", 3.5, 20, 0.05)
        target = next(r for r in results if r["estimator"] == "sndr" and r["mode"] == audit.MODES[2])
        self.assertEqual(target["point_status"], "unavailable")
        rows["site_key"] = "one-cluster"
        result = audit.audit_estimates(rows, "raw", 3.5, 20, 0.05)[0]
        self.assertEqual(result["interval_status"], "unavailable")
        self.assertIn("two geographic clusters", result["reason"])

    def test_observed_economic_identity_and_nonfinite_inputs_fail_closed(self):
        rows = accounting_rows()
        rows.loc[0, "profit"] += 1
        with self.assertRaisesRegex(ValueError, "Observed profit"):
            audit.audit_estimates(rows, "raw", 3.5, 20, 0.05)
        rows = accounting_rows()
        rows.loc[0, "mu_profit_logged"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            audit.audit_estimates(rows, "raw", 3.5, 20, 0.05)


class AuditWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = synthetic_data()
        self.path = self.root / "data.npz"
        np.savez_compressed(self.path, **self.data)
        self.spec = json.loads((ROOT / "configs/revision_evaluation.json").read_text())
        self.spec.update(holdout_year=2015, n_bootstrap=8)
        self.spec["policy"].update(min_support_count=2, spibb_min_deviation_count=2, trim_min_count_logged=2, trim_min_count_target=2)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(self.spec))
        for target, name, replacement in ((engine, "StackedRewardModel", LinearReward), (engine, "JointPropensityModel", CountPropensity)):
            active = patch.object(target, name, replacement)
            active.start()
            self.addCleanup(active.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            revision.select_policy(self.config, self.path, self.root / "selection")
            revision.evaluate_selected(self.root / "selection/selection.json", self.path, self.root / "final")
        self.selection = self.root / "selection/selection.json"

    def refit(self, output="audit", **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.refit_run(self.selection, self.path, self.root / "final", self.root / output, self.root / "cache", **kwargs)

    def test_saved_summary_never_fits_models_or_overwrites_results(self):
        originals = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(revision, "fit_year", side_effect=AssertionError("No fitting allowed")), contextlib.redirect_stdout(io.StringIO()):
            result = runner.summarize_run(self.selection, self.path, self.root / "final", self.root / "summary")
        self.assertFalse(result["refit_performed"])
        self.assertFalse(result["model_level_accounting_available"])
        for name, content in originals.items():
            self.assertEqual(Path(name).read_bytes(), content)
        with self.assertRaises(FileExistsError):
            runner.summarize_run(self.selection, self.path, self.root / "final", self.root / "summary")

    def test_cost_probe_preserves_policy_and_matches_known_cost_expectations(self):
        data = revision.load_data(self.path)
        cfg = revision.make_config(self.spec, self.spec["candidates"][-1])
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = revision.fit_year(2015, data, cfg)
        original_yield = fitted["mu_grid_yield"].copy()
        block = engine.apply_policy_to_year(fitted, cfg)
        rows = audit.rows_from_fit(fitted, block, cfg, data, "selected", "final")
        prices = np.array([cfg.priceN, cfg.priceP, cfg.priceK])
        np.testing.assert_allclose(rows.cost_pi0, fitted["pi0_grid_all"] @ (fitted["grid"] @ prices))
        expected = sum(block["diagnostics"]["fertilizer_allrows"]["avg_pi1"][k] * p for k, p in zip(("N", "P", "K"), prices))
        self.assertAlmostEqual(rows.cost_pi1.mean(), expected)
        np.testing.assert_array_equal(fitted["mu_grid_yield"], original_yield)

    def test_full_refit_reproduces_original_and_cache_replay_skips_all_fits(self):
        original_fit = revision.fit_year
        with patch.object(revision, "fit_year", wraps=original_fit) as fitting:
            manifest = self.refit()
            self.assertEqual(fitting.call_count, 3)
        self.assertTrue(manifest["all_original_point_estimates_reproduced"])
        with patch.object(revision, "fit_year", side_effect=AssertionError("Must use cache")):
            replay = self.refit("replay", cache_only=True)
        self.assertTrue(replay["all_original_point_estimates_reproduced"])
        for name in ("accounting_audit.csv", "aggregation_contributions.csv", "heldout_prediction_metrics.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(self.root / "audit" / name), pd.read_csv(self.root / "replay" / name))
        rows = pd.read_csv(self.root / "audit/audit_rows.csv")
        self.assertEqual(set(rows.loc[rows.stage == "validation", "year"]), {2013, 2014})
        self.assertEqual(set(rows.loc[rows.stage == "final", "candidate"]), {json.loads(self.selection.read_text())["selected_candidate"]["name"]})
        for _, group in rows.loc[rows.stage == "validation"].groupby("year"):
            masks = [g["included_cohort"].to_numpy() for _, g in group.groupby("candidate")]
            for other in masks[1:]:
                np.testing.assert_array_equal(masks[0], other)

    def test_changed_data_source_environment_and_cache_are_rejected(self):
        with patch.object(revision, "fingerprint_sources", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.refit()
        actual = revision.environment()
        with patch.object(revision, "environment", return_value={**actual, "python": "different"}):
            with self.assertRaisesRegex(ValueError, "environment differs"):
                self.refit()
        self.refit()
        with np.load(self.root / "cache/fitted_2013.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        meta = json.loads(str(arrays["metadata_json"].item()))
        meta["cache_identity"]["identity"]["data_sha256"] = "changed"
        arrays["metadata_json"] = np.array(json.dumps(meta))
        np.savez_compressed(self.root / "cache/fitted_2013.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "Incompatible cache"):
            self.refit("changed-cache")
        changed = dict(self.data)
        changed["reward"] = changed["reward"] + 1
        np.savez_compressed(self.path, **changed)
        with self.assertRaisesRegex(ValueError, "Data changed"):
            self.refit("changed-data")

    def test_missing_cache_and_changed_holdout_rows_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "No cached fit"):
            self.refit(cache_only=True)
        path = self.root / "final/holdout_rows.csv"
        holdout = pd.read_csv(path)
        holdout.iloc[::-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "field IDs"):
            self.refit("changed-holdout")

    def test_real_boosting_libraries_produce_matched_audit_components(self):
        import bandit_models as models
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        real_propensity = models.JointPropensityModel
        cfg = revision.make_config(self.spec, self.spec["candidates"][-1])
        data = revision.load_data(self.path)
        with patch.object(engine, "StackedRewardModel", models.StackedRewardModel), \
             patch.object(models.StackedRewardModel, "_create_xgb", lambda self: xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)), \
             patch.object(models.StackedRewardModel, "_create_lgb", lambda self: lgb.LGBMRegressor(n_estimators=3, max_depth=2, n_jobs=1, verbose=-1, random_state=42)), \
             patch.object(models.StackedRewardModel, "_create_cat", lambda self: cb.CatBoostRegressor(iterations=3, depth=2, thread_count=1, verbose=False, allow_writing_files=False)), \
             patch.object(engine, "JointPropensityModel", lambda: real_propensity(base_params={"n_estimators": 3, "n_jobs": 1})), \
             contextlib.redirect_stdout(io.StringIO()):
            fitted = revision.fit_year(2015, data, cfg)
            block = engine.apply_policy_to_year(fitted, cfg)
            rows = audit.rows_from_fit(fitted, block, cfg, data, "real", "final")
            estimates = audit.audit_estimates(rows, "psis", cfg.maize_price, 8, 0.05)
        for result in estimates:
            self.assertEqual(result["point_status"], "estimable")
            self.assertAlmostEqual(result["gap_identity_error"], 0, places=6)
            self.assertAlmostEqual(result["naive_identity_error"], 0, places=6)
        expected = revision.summarize([block], block["keep_mask"], cfg, "psis")["sndr"]["delta"]
        self.assertAlmostEqual(estimates[0]["profit_delta"], expected, places=7)


if __name__ == "__main__":
    unittest.main()
