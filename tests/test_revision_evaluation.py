"""Regression checks for chronological separation, numerical estimators and run identity."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import learn_and_evaluate_policy as ope
import ope_statistics as stats
import run_revision_evaluation as revision


def synthetic_data():
    years = np.repeat(np.arange(2012, 2016), 60)
    j = np.tile(np.arange(60), 4)
    X = np.column_stack([years - 2010, j / 10.0]).astype(float)
    X[::23, 1] = np.nan
    A = np.column_stack([40 + (j % 6) * 20, 10 + (j % 3) * 10, (j % 2) * 10]).astype(float)
    y = 5000 + 10 * A[:, 0] + 2 * A[:, 1] + A[:, 2] + 3 * np.nan_to_num(X[:, 1], nan=3.0)
    return dict(obs=X, actions=A, reward=y, year=years, lat=16 + (j // 6) * 0.02,
                lon=np.full(len(j), -92.0), field_id=np.arange(len(j)),
                imputation_mode=np.array('deferred'))


class LinearReward:
    """Fast deterministic learner used to exercise orchestration, not research results."""
    def fit(self, X, A, y, **kwargs):
        self.coef = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X, A]), y, rcond=None)[0]
        return self

    def predict(self, X, A):
        return np.column_stack([np.ones(len(X)), X, A]) @ self.coef


class CountPropensity:
    def fit(self, X, A, bin_edges, **kwargs):
        self.edges = bin_edges
        self.shape = tuple(len(bin_edges[k]) - 1 for k in ('N', 'P', 'K'))
        bins = [np.clip(np.digitize(A[:, i], bin_edges[k]) - 1, 0, self.shape[i] - 1)
                for i, k in enumerate(('N', 'P', 'K'))]
        ids = np.ravel_multi_index(bins, self.shape)
        self.p = np.bincount(ids, minlength=int(np.prod(self.shape))) / len(A)
        return self

    def predict_proba_grid(self, X):
        return np.tile(self.p, (len(X), 1))


class StatisticalTests(unittest.TestCase):
    def test_psis_preserves_ratio_mass_and_zeros(self):
        raw = np.r_[np.zeros(4), np.linspace(0.1, 3, 100)]
        w, _ = stats.psis_smooth_weights(raw)
        np.testing.assert_array_equal(w[:4], 0)
        self.assertAlmostEqual(w.sum(), raw.sum(), places=10)

    def test_psis_null_policy_has_zero_dr_and_sndr_gain(self):
        n = 100
        w, _ = stats.psis_smooth_weights(np.ones(n))
        y, mu, idx = np.full(n, 10.0), np.full(n, 4.0), np.arange(n)
        for fn in (stats._dr_value, stats._sndr_value):
            self.assertAlmostEqual(fn(mu, w, y, mu, idx) - fn(mu, np.ones(n), y, mu, idx), 0.0)
            self.assertAlmostEqual(fn(mu, w, y, mu, idx), 10.0)

    def test_missing_psis_fails_closed(self):
        with patch('ope_statistics.importlib.import_module', side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, 'PSIS requires ArviZ'):
                stats.psis_smooth_weights(np.ones(10))

    def test_imputation_cannot_see_evaluation_values(self):
        train = np.array([[1, np.nan], [3, np.nan]], float)
        score = np.array([[np.nan, 9999]], float)
        a, b, medians = stats.median_impute_from_training(train, score)
        np.testing.assert_array_equal(medians, [2, 0])
        np.testing.assert_array_equal(a[:, 1], [0, 0])
        self.assertEqual(b[0, 0], 2)
        self.assertTrue(np.isnan(train[0, 1]))  # Caller arrays are unchanged.

    def test_cluster_resampling_preserves_whole_clusters(self):
        keys = np.array(['a', 'a', 'b', 'b', 'b', 'c'])
        for idx in stats.make_cluster_boot_indices_from_keys(keys, 50):
            counts = np.bincount(idx, minlength=len(keys))
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[2], counts[3])
            self.assertEqual(counts[3], counts[4])
        with self.assertRaisesRegex(ValueError, 'two geographic clusters'):
            stats.make_cluster_boot_indices_from_keys(np.array(['a', 'a']), 20)

    def test_invalid_or_zero_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            stats.psis_smooth_weights(np.array([-1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, 'no positive'):
            stats._sndr_value(np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), np.arange(3))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = synthetic_data()
        self.path = self.root / 'data.npz'
        np.savez_compressed(self.path, **self.data)
        self.spec = json.loads((ROOT / 'configs/revision_evaluation.json').read_text())
        self.spec.update(holdout_year=2015, n_bootstrap=20)
        self.spec['policy'].update(min_support_count=2, spibb_min_deviation_count=2,
                                   trim_min_count_logged=2, trim_min_count_target=2)
        self.spec_path = self.root / 'config.json'
        self.spec_path.write_text(json.dumps(self.spec))
        self.reward_patch = patch.object(ope, 'StackedRewardModel', LinearReward)
        self.prop_patch = patch.object(ope, 'JointPropensityModel', CountPropensity)
        self.reward_patch.start()
        self.prop_patch.start()
        self.addCleanup(self.reward_patch.stop)
        self.addCleanup(self.prop_patch.stop)

    def select(self, out='selection', path=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return revision.select_policy(self.spec_path, path or self.path, self.root / out)

    def test_selection_and_final_evaluation_complete_with_auditable_splits(self):
        manifest = self.select()
        self.assertEqual(manifest['selection_years'], [2013, 2014])
        with contextlib.redirect_stdout(io.StringIO()):
            result = revision.evaluate_selected(self.root / 'selection/selection.json', self.path, self.root / 'final')
        self.assertEqual(result['fit_audit']['year'], 2015)
        self.assertEqual(result['fit_audit']['model_fits'][0]['fit_years'], [2012, 2013, 2014])
        self.assertTrue(np.isfinite(result['profit']['sndr']['delta']))
        self.assertEqual(len(pd.read_csv(self.root / 'final/holdout_rows.csv')), 60)
        self.assertFalse(pd.read_csv(self.root / 'final/holdout_rows.csv')['field_id'].isna().any())
        summary = pd.read_csv(self.root / 'final/final_summary.csv')
        self.assertEqual(set(summary['outcome']), {'profit', 'yield'})
        self.assertTrue(result['selection']['dataset_previously_analyzed'])

    def test_changing_final_year_outcomes_cannot_change_selection(self):
        a = self.select('a')
        changed = dict(self.data)
        changed['reward'] = self.data['reward'].copy()
        changed['reward'][changed['year'] == 2015] += 999999
        second = self.root / 'changed.npz'
        np.savez_compressed(second, **changed)
        b = self.select('b', second)
        self.assertEqual(a['selected_candidate'], b['selected_candidate'])
        self.assertEqual(a['selected_validation_result'], b['selected_validation_result'])
        pd.testing.assert_frame_equal(pd.read_csv(self.root / 'a/validation_candidates.csv'),
                                      pd.read_csv(self.root / 'b/validation_candidates.csv'))

    def test_evaluation_year_labels_do_not_enter_fitted_predictions(self):
        data = revision.load_data(self.path)
        cfg = revision.make_config(self.spec, self.spec['candidates'][0])
        a = revision.fit_year(2015, data, cfg)
        data['reward'][data['year'] == 2015] += 100000
        b = revision.fit_year(2015, data, cfg)
        np.testing.assert_array_equal(a['mu_grid_profit'], b['mu_grid_profit'])
        np.testing.assert_array_equal(a['pi0_grid_all'], b['pi0_grid_all'])
        self.assertEqual(len(a['fit_audit']), 1)  # No five identical past-only fits.

    def test_common_selection_cohort_is_shared_by_candidates(self):
        self.select()
        table = pd.read_csv(self.root / 'selection/validation_candidates.csv')
        self.assertEqual(table['n_common_rows'].nunique(), 1)
        self.assertEqual(len(table), len(self.spec['candidates']))

    def test_changed_data_or_source_blocks_final_evaluation(self):
        self.select()
        path = self.root / 'selection/selection.json'
        with patch.object(revision, 'fingerprint_sources', return_value={'changed': 'source'}):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                revision.evaluate_selected(path, self.path, self.root / 'final')
        changed = dict(self.data)
        changed['reward'] = self.data['reward'] + 1
        np.savez_compressed(self.path, **changed)
        with self.assertRaisesRegex(ValueError, 'Data changed'):
            revision.evaluate_selected(path, self.path, self.root / 'final')

    def test_frozen_policy_modifications_and_overwrites_are_rejected(self):
        self.select()
        with self.assertRaises(FileExistsError):
            self.select()
        path = self.root / 'selection/selection.json'
        manifest = json.loads(path.read_text())
        manifest['selected_config']['lambda_mix'] = 0.99
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'settings were modified'):
            revision.evaluate_selected(path, self.path, self.root / 'final')

    def test_old_globally_imputed_data_are_rejected(self):
        self.data['imputation_mode'] = np.array('global_median')
        np.savez_compressed(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, 'defer-imputation'):
            revision.load_data(self.path)

    def test_invalid_designs_are_rejected(self):
        for update in ({'holdout_year': 2013}, {'n_bootstrap': 0}, {'weights': 'unknown'}):
            with self.assertRaises(ValueError):
                revision.validate_spec({**self.spec, **update})
        duplicate = copy.deepcopy(self.spec)
        duplicate['candidates'].append(duplicate['candidates'][0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            revision.validate_spec(duplicate)

    def test_unestimable_candidates_remain_in_the_selection_audit(self):
        original = revision.selection_estimate
        def without_greedy(blocks, mask, cfg, weights):
            if cfg.epsilon_explore == 0 and cfg.lambda_mix == 0:
                return {'status': 'unestimable', 'reason': 'synthetic no-support case', 'lcb': float('nan')}
            return original(blocks, mask, cfg, weights)
        with patch.object(revision, 'selection_estimate', side_effect=without_greedy):
            manifest = self.select()
        self.assertNotEqual(manifest['selected_candidate']['name'], 'greedy')
        table = pd.read_csv(self.root / 'selection/validation_candidates.csv')
        self.assertEqual(table.iloc[0]['status'], 'unestimable')


class RealModelCompatibilityTests(unittest.TestCase):
    def test_actual_boosting_libraries_fit_the_chronological_engine(self):
        import bandit_models as models
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        data = synthetic_data()
        real_propensity = ope.JointPropensityModel
        cfg = ope.OPEConfig(min_support_count=2, spibb_min_deviation_count=2,
                            objective='profit', maize_price=3.5, priceN=16, priceP=12, priceK=8)
        with patch.object(models.StackedRewardModel, '_create_xgb',
                          lambda self: xgb.XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)), \
             patch.object(models.StackedRewardModel, '_create_lgb',
                          lambda self: lgb.LGBMRegressor(n_estimators=3, max_depth=2, n_jobs=1, verbose=-1, random_state=42)), \
             patch.object(models.StackedRewardModel, '_create_cat',
                          lambda self: cb.CatBoostRegressor(iterations=3, depth=2, thread_count=1, verbose=False, allow_writing_files=False)), \
             patch.object(ope, 'JointPropensityModel',
                          lambda: real_propensity(base_params={'n_estimators': 3, 'n_jobs': 1})), \
             contextlib.redirect_stdout(io.StringIO()):
            fitted = revision.fit_year(2013, data, cfg)
        self.assertTrue(np.isfinite(fitted['mu_grid_profit']).all())
        self.assertTrue(np.isfinite(fitted['mu_grid_yield']).all())
        np.testing.assert_allclose(fitted['pi0_grid_all'].sum(axis=1), 1, atol=1e-6)
        self.assertEqual(fitted['fit_audit'][0]['fit_years'], [2012])


class PreprocessingTests(unittest.TestCase):
    def test_deferred_preprocessing_keeps_missing_values_and_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame({
                'Field_ID': [1, 2, 3, 4], 'Year': [2012, 2012, 2013, 2013],
                'Lat': [16, 16.02, 16.04, 16.06], 'Long': [-92] * 4, 'Elev': [100] * 4,
                'Planting': [100, np.nan, 110, 120], 'YIELD': [4, 5, 6, 7],
                'Slope': [1] * 4, 'Clay': [10, np.nan, 900, 1000], 'CEC': [20] * 4,
                'SOM': [2] * 4, 'PH': [6] * 4, 'Nitrogen': [40, 60, 80, 100],
                'Phosphorus': [10] * 4, 'Potassium': [0] * 4,
                'System': ['mono'] * 4, 'Tillage': ['zero'] * 4,
            })
            for var in ('prcp', 'srad', 'tmax', 'tmin', 'vp'):
                for i in range(1, 7):
                    frame[f'{var}_V{i}'] = 1.0
            source, target = folder / 'source.csv', folder / 'separate/data.npz'
            frame.to_csv(source, index=False)
            subprocess.run([sys.executable, str(ROOT / 'scripts/preprocess_data.py'),
                            '--input', str(source), '--out', str(target), '--defer-imputation'],
                           check=True, capture_output=True, text=True)
            with np.load(target, allow_pickle=False) as result:
                self.assertEqual(result['imputation_mode'].item(), 'deferred')
                self.assertTrue(np.isnan(result['obs']).any())
                np.testing.assert_array_equal(result['field_id'], [1, 2, 3, 4])
            self.assertTrue((target.parent / 'feature_cols.csv').exists())
            self.assertTrue((target.parent / 'metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
