"""Select on earlier years, record the choice, then evaluate one later-year policy.

This is a retrospective reanalysis of previously inspected observational data.
See docs/revision_evaluation.md for the estimand and inference limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from ope_statistics import InferenceUnavailable, cluster_summary, require_psis

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_revision_evaluation.py", "scripts/learn_and_evaluate_policy.py",
    "scripts/ope_statistics.py", "scripts/bandit_models.py", "scripts/preprocess_data.py",
)
PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
    "catboost", "arviz", "joblib", "threadpoolctl", "openpyxl", "matplotlib",
)
NOTICE = (
    "Retrospective chronological reanalysis: 2018 was previously analyzed. "
    "This is not a newly untouched or prospectively preregistered test set. "
    "Intervals condition on fitted models, the chosen policy and overlap rules; "
    "they do not quantify unmeasured confounding or the entire learning procedure."
)


def engine():
    import learn_and_evaluate_policy
    return learn_and_evaluate_policy


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n")


def fingerprint_sources():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}


def validate_spec(spec):
    expected = {"holdout_year", "selection_start_year", "n_bootstrap", "alpha",
                "weights", "prices", "policy", "candidates"}
    if set(spec) != expected:
        raise ValueError(f"Configuration keys must be exactly {sorted(expected)}.")
    if not isinstance(spec['holdout_year'], int) or not isinstance(spec['selection_start_year'], int):
        raise ValueError("Year boundaries must be integers.")
    if spec['selection_start_year'] >= spec['holdout_year']:
        raise ValueError("Selection years must precede the final evaluation year.")
    if not isinstance(spec['n_bootstrap'], int) or spec['n_bootstrap'] < 2 or not 0 < spec['alpha'] < 1:
        raise ValueError("Invalid bootstrap replicate count or alpha.")
    if spec['weights'] not in {'psis', 'cap'}:
        raise ValueError("weights must be psis or cap.")
    if set(spec['prices']) != {'maize_price', 'priceN', 'priceP', 'priceK'}:
        raise ValueError("Specify all four research prices.")
    if not all(np.isfinite(v) and v > 0 for v in spec['prices'].values()):
        raise ValueError("Research prices must be finite and positive.")
    policy_keys = {'min_support_count', 'spibb_min_deviation_count', 'trim_pi0_tau',
                   'trim_ratio_max', 'trim_min_count_logged', 'trim_min_count_target',
                   'max_density_ratio', 'recency_lambda'}
    if set(spec['policy']) != policy_keys:
        raise ValueError(f"Policy configuration keys must be exactly {sorted(policy_keys)}.")
    if not all(np.isfinite(v) and v >= 0 for v in spec['policy'].values()):
        raise ValueError("Policy parameters must be finite and nonnegative.")
    for key in ('min_support_count', 'spibb_min_deviation_count',
                'trim_min_count_logged', 'trim_min_count_target'):
        if not isinstance(spec['policy'][key], int):
            raise ValueError(f"{key} must be an integer.")
    if spec['policy']['min_support_count'] < 1 or not 0 < spec['policy']['trim_pi0_tau'] <= 1:
        raise ValueError("Support count and propensity floor must be positive.")
    if spec['policy']['trim_ratio_max'] <= 0 or spec['policy']['max_density_ratio'] <= 0:
        raise ValueError("Ratio thresholds must be positive.")
    names, settings = [], []
    if not spec['candidates']:
        raise ValueError("At least one candidate policy must be declared.")
    for candidate in spec['candidates']:
        if set(candidate) != {'name', 'epsilon_explore', 'lambda_mix'}:
            raise ValueError("Each candidate needs name, epsilon_explore and lambda_mix.")
        if not isinstance(candidate['name'], str) or not candidate['name'].strip():
            raise ValueError("Candidate names must be nonempty strings.")
        values = (candidate['epsilon_explore'], candidate['lambda_mix'])
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Exploration and mixing probabilities must lie in [0, 1].")
        names.append(candidate['name'])
        settings.append(values)
    if len(set(names)) != len(names) or len(set(settings)) != len(settings):
        raise ValueError("Candidate names and parameter combinations must be unique.")


def load_data(path, *, before_year=None, through_year=None):
    """Filter the year split before exposing actions/outcomes to selection code."""
    with np.load(path, allow_pickle=False) as source:
        if 'imputation_mode' not in source or str(source['imputation_mode'].item()) != 'deferred':
            raise ValueError("Reprocess raw data with --defer-imputation; bundled imputed data are unsuitable.")
        years = source['year']
        if not np.isfinite(years).all() or not np.equal(years, np.floor(years)).all():
            raise ValueError("Dataset years must be finite integers.")
        mask = np.ones(len(years), bool)
        if before_year is not None:
            mask &= years < before_year
        if through_year is not None:
            mask &= years <= through_year
        data = {name: source[name][mask] for name in
                ('obs', 'actions', 'reward', 'year', 'lat', 'lon', 'field_id')}
    n = len(data['year'])
    if n == 0 or data['obs'].ndim != 2 or data['obs'].shape[1] == 0 or data['actions'].shape != (n, 3):
        raise ValueError("Invalid context/action array dimensions.")
    if np.isinf(data['obs']).any():
        raise ValueError("Infinite context values are invalid.")
    for name in ('actions', 'reward', 'lat', 'lon'):
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Missing or infinite values in {name}.")
    if (data['actions'] < 0).any():
        raise ValueError("Nutrient rates must be nonnegative.")
    data['year'] = data['year'].astype(int)
    return data


def make_config(spec, candidate):
    return engine().OPEConfig(
        **spec['policy'], **spec['prices'],
        epsilon_explore=candidate['epsilon_explore'], lambda_mix=candidate['lambda_mix'],
        n_bootstrap=spec['n_bootstrap'], alpha=spec['alpha'],
        objective='profit', cf_use_eval=False, trim_enable=True,
    )


def fit_year(year, data, cfg):
    prices = cfg.priceN * data['actions'][:, 0] + cfg.priceP * data['actions'][:, 1] + cfg.priceK * data['actions'][:, 2]
    profit = cfg.maize_price * data['reward'] - prices
    fitted = engine().fit_year_nuisances(
        year, data['obs'], data['actions'], profit, data['reward'], data['year'],
        data['lat'], data['lon'], cfg,
    )
    if fitted is None:
        raise ValueError(f"Year {year} needs both historical training rows and evaluation rows.")
    if fitted['cf_use_eval'] or any(max(f['fit_years']) >= year for f in fitted['fit_audit']):
        raise ValueError("Evaluation-year outcomes entered model training.")
    return fitted


def summarize(blocks, mask, cfg, weights, *, outcome='profit'):
    stats = engine().pooled_profit if outcome == 'profit' else engine().pooled_yield
    return {est: stats(blocks, mask, est, weights == 'psis', cfg.n_bootstrap, cfg.alpha)
            for est in ('sndr', 'dr')}


def selection_estimate(blocks, mask, cfg, weights):
    """Retain non-estimable candidates in the audit, without assigning a score."""
    try:
        result = summarize(blocks, mask, cfg, weights)['sndr']
        if not np.isfinite(result['lcb']):
            raise InferenceUnavailable('No finite SNDR interval on the common cohort.')
        return {'status': 'estimable', **result}
    except InferenceUnavailable as exc:
        return {'status': 'unestimable', 'reason': str(exc), 'lcb': float('nan')}


def audit_fitted(fitted):
    return {"year": fitted['year'], "n_evaluation_rows": len(fitted['test_indices']),
            "model_fits": fitted['fit_audit'], "bin_edges": fitted['bin_edges'],
            "evaluation_clusters": cluster_summary(fitted['site_keys'])}


def select_policy(spec_path, data_path, output):
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    if spec['weights'] == 'psis':
        require_psis()
    with np.load(data_path, allow_pickle=False) as source:
        if int(source['year'].max()) != spec['holdout_year']:
            raise ValueError("The final evaluation year must be the latest observed dataset year.")
    # No final-year actions, outcomes, contexts or labels reach candidate fitting/scoring.
    data = load_data(data_path, before_year=spec['holdout_year'])
    years = [int(y) for y in np.unique(data['year']) if y >= spec['selection_start_year']]
    if not years or min(data['year']) >= min(years):
        raise ValueError("Selection needs validation years with an earlier training year.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configs = {c['name']: make_config(spec, c) for c in spec['candidates']}
    first_cfg = next(iter(configs.values()))
    blocks = {name: [] for name in configs}
    common_masks, audits = [], []
    for year in years:
        print(f"Fitting selection year {year} once for {len(configs)} candidates...", flush=True)
        fitted = fit_year(year, data, first_cfg)
        for name, cfg in configs.items():
            blocks[name].append(engine().apply_policy_to_year(fitted, cfg))
        common_masks.append(np.logical_and.reduce([blocks[name][-1]['keep_mask'] for name in configs]))
        audits.append(audit_fitted(fitted))
    common = np.concatenate(common_masks)
    if not common.any():
        raise ValueError("No common overlap rows; revise the declared design before evaluation.")
    site_keys = np.concatenate([b['site_keys'] for b in next(iter(blocks.values()))])
    cluster_info = cluster_summary(site_keys[common])
    records, by_year = [], []
    for candidate in spec['candidates']:
        name, cfg = candidate['name'], configs[candidate['name']]
        result = selection_estimate(blocks[name], common, cfg, spec['weights'])
        records.append({**candidate, **result, 'n_common_rows': int(common.sum()),
                        'n_candidate_overlap_rows': int(sum(b['keep_mask'].sum() for b in blocks[name]))})
        for block, mask in zip(blocks[name], common_masks):
            r = selection_estimate([block], mask, cfg, spec['weights'])
            by_year.append({'candidate': name, 'year': block['year'], **r,
                            'n_common_rows': int(mask.sum())})
    pd.DataFrame(records).to_csv(output / 'validation_candidates.csv', index=False)
    pd.DataFrame(by_year).to_csv(output / 'validation_by_year.csv', index=False)
    write_json(output / 'split_audit.json', audits)
    eligible = [row for row in records if row['status'] == 'estimable']
    if not eligible:
        raise InferenceUnavailable('No candidate has an estimable validation interval; see validation_candidates.csv.')
    # Stable tie break: first candidate in the predeclared list, never final-year metrics.
    best = max(eligible, key=lambda row: row['lcb'])
    selected = next(c for c in spec['candidates'] if c['name'] == best['name'])
    manifest = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'notice': NOTICE, 'dataset_previously_analyzed': True,
        'data_sha256': sha256_file(data_path), 'source_sha256': fingerprint_sources(),
        'environment': environment(), 'spec': spec, 'selection_years': years,
        'selection_rule': f"maximum estimable pooled SNDR lower {(1-spec['alpha'])*100:g}% interval endpoint; first-declared tie break",
        'selection_intervals_are_exploratory': True,
        'selected_candidate': selected, 'selected_config': asdict(configs[best['name']]),
        'selected_validation_result': best, 'validation_lcb_positive': best['lcb'] > 0,
        'common_cohort': cluster_info, 'n_validation_rows': int(len(common)),
        'psis_scale': 'positive ratios smoothed; raw ratio sum preserved; exact zeros retained',
    }
    write_json(output / 'selection.json', manifest)  # Written last: only complete selections are usable.
    print(f"Frozen choice: {best['name']}; validation LCB={best['lcb']:.3f} MXN/ha.")
    print(f"Next: evaluate --selection {output / 'selection.json'} --data {data_path} --output results/revision_final")
    return manifest


def evaluate_selected(selection_path, data_path, output):
    manifest = json.loads(Path(selection_path).read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError("Unsupported selection manifest.")
    spec = manifest['spec']
    validate_spec(spec)
    if sha256_file(data_path) != manifest['data_sha256']:
        raise ValueError("Data changed after selection. Reproduce the recorded input.")
    if fingerprint_sources() != manifest['source_sha256']:
        raise ValueError("Analysis source changed after selection. Reproduce the recorded source.")
    if environment()['packages'] != manifest['environment']['packages'] or platform.python_version() != manifest['environment']['python']:
        raise ValueError("Research environment changed after selection. Reuse the recorded environment.")
    selected = manifest['selected_candidate']
    if selected not in spec['candidates']:
        raise ValueError("Selected policy is not in the declared candidate grid.")
    cfg = make_config(spec, selected)
    if asdict(cfg) != manifest['selected_config']:
        raise ValueError("Frozen policy settings were modified after selection.")
    if max(manifest['selection_years']) >= spec['holdout_year']:
        raise ValueError("Selection years overlap the final evaluation year.")
    if spec['weights'] == 'psis':
        require_psis()
    data = load_data(data_path, through_year=spec['holdout_year'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fitted = fit_year(spec['holdout_year'], data, cfg)
    block = engine().apply_policy_to_year(fitted, cfg)
    mask = block['keep_mask']
    if not mask.any():
        raise ValueError("The frozen policy has no final-year overlap observations.")
    clusters = cluster_summary(block['site_keys'][mask])
    profit = summarize([block], mask, cfg, spec['weights'])
    yields = summarize([block], mask, cfg, spec['weights'], outcome='yield')
    rows = []
    for outcome, estimates in (('profit', profit), ('yield', yields)):
        for estimator, result in estimates.items():
            rows.append({'outcome': outcome, 'candidate': selected['name'],
                         'year': spec['holdout_year'], 'weights': spec['weights'],
                         'n_eval': int(len(mask)), 'n_overlap': int(mask.sum()),
                         'retained_fraction': float(mask.mean()), **result})
    result = {
        'notice': NOTICE, 'selection_sha256': sha256_file(selection_path),
        'selection': manifest, 'fit_audit': audit_fitted(fitted),
        'overlap_clusters': clusters, 'profit': profit, 'yield': yields,
        'diagnostics': block['diagnostics'], 'primary': 'profit_sndr',
        'primary_lcb_positive': bool(profit['sndr']['lcb'] > 0),
    }
    pd.DataFrame(rows).to_csv(output / 'final_summary.csv', index=False)
    idx = block['test_indices']
    pd.DataFrame({
        'field_id': data['field_id'][idx], 'year': data['year'][idx],
        'site_key': block['site_keys'], 'included_overlap': mask,
        'pi0_logged': block['pi0_logged'], 'pi1_logged': block['pi1_logged'],
        'ratio_raw': block['w_raw'], 'count_logged': block['count_logged'],
        'count_target': block['count_target'],
    }).to_csv(output / 'holdout_rows.csv', index=False)
    write_json(output / 'final_results.json', result)
    print(f"Final-year SNDR profit gain: {profit['sndr']['delta']:.3f} MXN/ha "
          f"[{profit['sndr']['ci_lo']:.3f}, {profit['sndr']['ci_hi']:.3f}].")
    print(f"Retained {int(mask.sum())}/{len(mask)} rows across {clusters['n_clusters']} clusters.")
    print(NOTICE)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    select = sub.add_parser('select', help='Choose a policy using only pre-final-year data.')
    select.add_argument('--config', default='configs/revision_evaluation.json')
    select.add_argument('--data', required=True)
    select.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate', help='Evaluate exactly the recorded policy on the final year.')
    evaluate.add_argument('--selection', required=True)
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.stage == 'select':
            select_policy(args.config, args.data, args.output)
        else:
            evaluate_selected(args.selection, args.data, args.output)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
