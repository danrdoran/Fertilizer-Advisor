# Audit of aggregation and economic accounting

This is the next diagnostic step after the chronological selection/evaluation
workflow. It addresses Reviewer 2 major comment 6 (consistency of yield, profit
and fertilizer changes), the inference interpretation in Reviewer 2 major
comment 5, and prerequisites for answering Reviewer 1 comments 3 and 4. It does
not add price/support sensitivity experiments or resolve unmeasured confounding.

The audit is additive: the five original fingerprinted analysis sources,
selected configuration and original result directories remain unchanged. A
negative validation lower bound is reported explicitly; ranking the best
candidate is not automatically accepting a safe improvement.

## Run from an existing completed research checkout

Use the Python 3.12 research environment that generated the original run. There
are no new dependencies and no need to reprocess the original data or rerun
selection. In Bash/WSL:

```bash
git fetch origin
git switch revision/evaluation-audit-and-profit-consistency
source .venv-research-py312/bin/activate
python --version
python -m unittest discover -s tests -v
```

If creating a new environment instead, explicitly use
`python3.12 -m venv .venv-research-py312` and install
`requirements-research.txt`. A plain `python -m venv` inherits whichever Python
version that command currently resolves to; Python 3.10 cannot install these
research pins. A refit normally requires the original Python patch version and
core package versions, just as the original evaluation command does.

The default input locations are the already completed run:

- `results/revision_data/data.npz` (with deferred-imputation metadata);
- `results/revision_data/feature_cols.csv` (optional context names);
- `results/revision_selection/selection.json`, `validation_candidates.csv`,
  `validation_by_year.csv` and `split_audit.json`;
- `results/revision_final/final_results.json` and `holdout_rows.csv`.

The original source hashes, data hash, final-to-selection link and holdout row
identifiers are checked. Commands refuse an existing audit output directory.

### 1. Fast audit of saved files

```bash
python scripts/audit_revision_evaluation.py summary \
  --output results/revision_audit_summary
```

This fits no models. It produces:

| File | Interpretation |
| --- | --- |
| `saved_aggregation_comparison.csv` | Original pooled SNDR versus the sample-size-weighted annual point estimates, for every declared candidate. The annual average has no newly calculated interval in this fast audit. |
| `saved_descriptive_cost_check.csv` | The naive yield-revenue-minus-cost calculation using the old descriptive fertilizer diagnostics. It is explicitly **not** a matched-baseline OPE estimate. |
| `retained_excluded.csv` | Raw yields, revenue minus fertilizer costs, N/P/K and available context features for retained versus excluded final-year observations; includes standardized mean differences. These differences are descriptive, not treatment effects. |
| `final_cluster_counts.csv` | Cluster counts and minimum/median/maximum sizes for all, retained and excluded final-year observations. |
| `bin_boundaries.csv` | Physical thresholds across years. A boundary index is not a stable physical action label. |
| `summary_report.json` | Provenance, completion flag, original selection and validation-LCB interpretation. |

Context column names are read from the companion CSV when present. The script
never unpickles the object-typed feature-name array inside the NPZ. Without the
CSV it uses numbered feature names, with unchanged numeric calculations.

### 2. Recover missing model predictions and reconcile the estimates

The original result files do not contain the row-level nuisance predictions
needed to explain the differences. This command fits each year's original
models once, evaluates the original candidates on their original common
validation cohort, and evaluates only the frozen candidate in the final year:

```bash
python scripts/audit_revision_evaluation.py refit \
  --output results/revision_audit_refit \
  --cache results/revision_audit_cache
```

Expect substantial CPU time, comparable to repeating the original selection and
final evaluation. Completed yearly predictions are saved atomically as numeric
NPZ files. Cache identity includes data, original analysis sources, fit settings
and the actual research environment. It stores no pickled models. An interrupted
run can be resumed with the same cache and a **new output directory**:

```bash
python scripts/audit_revision_evaluation.py refit \
  --output results/revision_audit_refit_retry \
  --cache results/revision_audit_cache
```

Once all fits are cached, `--cache-only` guarantees that no training occurs. It
is useful for regenerating diagnostic tables or changing the audit's bootstrap
replicate count with `--bootstrap 1000`. Missing/incompatible cache entries cause
an explicit error. Cache-only analysis still recalculates policy probabilities,
PSIS weights and bootstrap intervals.

If the original research environment cannot be recovered,
`--allow-environment-difference` permits an explicitly labeled audit-only refit.
It records the mismatch and still reports numerical reproduction checks. Do not
claim reproduction merely because the command completed.

Alternate input locations can be set with `--selection PATH`, `--data PATH` and
`--final DIRECTORY`. The selection's companion CSV/JSON files must be alongside
it. Raw data and result files are not committed to the repository by this audit.

## Read the full audit in this order

1. **`reproduction_check.csv`:** all original candidate-level validation SNDR
   profit point estimates and the frozen final-year DR/SNDR profit/yield point
   estimates should match within the stated numerical tolerance. The check is
   about point estimates, not newly generated confidence intervals. Diagnose
   mismatches before drawing conclusions from alternative estimates. A candidate
   with no original numerical estimate is labeled `reference_available=False`
   and is not counted as a numerical reproduction success or failure.
2. **`aggregation_summary.csv` and `aggregation_contributions.csv`:** locate why
   the pooled and annual estimates differ. Contributions sum to the corresponding
   SNDR estimate and separate smoothing effects from normalization effects.
3. **`accounting_audit.csv` and `annual_accounting.csv`:** inspect the economic
   identity and its decomposition on precisely matched rows and baselines.
4. **`heldout_prediction_metrics.csv`:** actual forward-year logged-action R2,
   RMSE and MAE, on all rows and on the audit cohort. These are not training OOF
   metrics. Units are MXN/ha for profit and kg/ha for yield.
5. **`retained_clusters_by_year.csv`:** per-year retained cluster counts and size
   distributions, including the common validation cohort.

`audit_rows.csv` preserves every candidate's validation predictions, original
overlap flag, common-cohort inclusion, weights, observed outcomes and matched
cost expectations. It includes only the frozen candidate for the final year.
`refit_split_audit.json` records earlier-year-only fitting. `audit_manifest.json`
is written last and identifies a complete refit audit; it records input/source
identity, environment, cache hashes, bootstrap settings and output hashes.

## What the three aggregations mean

For SNDR the target policy value is the mean model prediction plus a
weight-normalized residual correction. When years have different average weights
and residuals, one global denominator can give a different sign from averaging
year-specific estimates. This difference alone does not prove an implementation
error. PSIS applied across all years also differs from PSIS applied within years.

| Mode | Smoothing | Residual normalization |
| --- | --- | --- |
| `pooled_weights_pooled_normalization` | All retained rows together | One denominator; reproduces the original estimator |
| `yearwise_weights_pooled_normalization` | Each year separately | One denominator; isolates the smoothing change |
| `yearwise_weights_yearwise_normalization` | Each year separately | Separate denominators; average yearly values by retained row counts |

Ordinary DR has no ratio normalization, so the last two modes must agree for DR
when they use identical weights. The tests verify this, a known sign-reversal
example, null-policy gains and the contribution identities. With capped weights,
the smoothing contrast is identically zero.

The audit does not replace the original estimator, reselect a candidate, or
choose whichever aggregation is positive. Alternative aggregations are
exploratory sensitivity analyses after the original final year was inspected.

## Matched economic accounting

The original descriptive fertilizer changes compare proposed actions with actual
logged actions. The OPE baseline is the original engine's estimated behavior
distribution over bin-center actions. These are different baselines. Also, profit
and yield use independently fitted outcome models.

The audit calculates exact expected fertilizer costs under both original policy
distributions, on the same cohort. It probes these costs through the original
policy-construction function while leaving profit predictions unchanged, so it
does not duplicate or alter the recommendation logic.

For the frozen policy it reports:

`consistent_profit_delta = maize_price * yield_delta - matched_cost_delta`

`surrogate_gap = original_profit_delta - consistent_profit_delta`

`original_profit_delta - naive_profit_delta = surrogate_gap + baseline_cost_gap`

The last identity distinguishes outcome-model disagreement from using logged
costs instead of behavior-policy expected costs. It is verified numerically for
both DR and SNDR. `gap_direct` and `gap_residual` further decompose the surrogate
gap. The accounting-consistent estimate holds the policy fixed; it does not learn
a new yield-based policy or establish that the yield surrogate is correct. A
nonzero surrogate gap is reported, not silently corrected away.

The accounting exercise also does not resolve treatment-version assumptions
created by evaluating continuous logged fertilizer rates with discretized
propensities and bin-center recommendations. That remains part of the response
to the action-discretization concern.

## Inference and manuscript use

All audit intervals use the same paired geographic-cluster draws across the
matched outcomes. Clusters spanning multiple years move together. Within-year
aggregation averages by row counts in each resampled dataset; those shares can
vary across bootstrap draws. Fitted models, overlap rules and the original PSIS
transformation are fixed, as in the original conditional bootstrap. An undefined
draw is never silently discarded: the point estimate remains when estimable,
but its interval is marked unavailable. A cohort with fewer than two clusters
cannot support these intervals.

The accounting and aggregation tables are not simultaneous confidence bands.
Report the original frozen endpoint alongside the exploratory audit. Do not
relabel 2018 as untouched, reinterpret a negative validation LCB as acceptance,
or use the audit to select a new primary result after inspecting the final year.

Use the output to answer the reviewers with (a) the precise evaluation unit and
baseline, (b) the arithmetic and statistical reason for the difference between
profit and yield results, (c) per-year predictive/cluster diagnostics, and (d)
retained/excluded characteristics. Revise any nutrient-reduction narrative to
match the actual selected policy's diagnostics. Price/support sensitivity,
unmeasured-confounding analysis, simpler-policy final comparisons, figure
improvements and a persistent archive remain subsequent revisions.

Methods references: [Vehtari et al., PSIS (2024)](https://www.jmlr.org/papers/volume25/19-556/19-556.pdf)
and [Swaminathan and Joachims, self-normalized estimation (2015)](https://proceedings.neurips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html).
