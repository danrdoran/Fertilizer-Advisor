# Package 1: temporal reliability, fixed actions and simpler policies

This package follows the [independent chronological evaluation](revision_evaluation.md) and the [aggregation/accounting audit](revision_audit.md). It addresses Reviewer 1 comments 4 (reliability across years) and 6 (simpler policies), and Reviewer 2 major comment 4 (changing action boundaries). It also reports coverage and cluster counts for the new comparisons. These are exploratory analyses specified after the earlier validation and 2018 results were inspected. They are not a new independent confirmation, and no policy is selected here.

The package leaves the original five fingerprinted evaluation sources, the two audit sources, the frozen selection and existing results unchanged. It adds a separate runner, diagnostics, cache and output directories. The original negative pooled validation result and positive 2018 result must both remain in the manuscript record. Do not replace either with whichever new estimate looks most favorable.

## Run from the repository root

Keep your existing Python 3.12 research environment, original results and `results/revision_audit_cache`. No preprocessing, selection or original audit rerun is needed.

```bash
git fetch origin
git switch revision/temporal-grid-and-baselines
source .venv-research-py312/bin/activate
python -m unittest discover -s tests -v

# Fast: reads the completed audit, fits no models.
python scripts/run_temporal_robustness.py diagnose \
  --output results/revision_temporal_diagnostics

# Expensive: reuses the adaptive cache; fits six fixed-grid yearly predictions.
python scripts/run_temporal_robustness.py compare \
  --adaptive-cache results/revision_audit_cache \
  --fixed-cache results/revision_temporal_cache \
  --output results/revision_temporal_comparison
```

Defaults are `--selection results/revision_selection/selection.json`, `--data results/revision_data/data.npz`, `--final results/revision_final`, `--audit results/revision_audit_refit`, and `--config configs/temporal_robustness.json`. Override paths if needed. `diagnose` requires the completed audit CSVs and manifest but not the prediction cache. `compare` needs the old cache as well. Use the environment in which the audit was generated; requirements remain in `requirements-research.txt`.

Each output directory must be new. If a run is interrupted, keep the caches and retry with another output path, such as `results/revision_temporal_comparison_retry`. Completed fits are reused. To replay with no fitting allowed, use the same caches, a new output directory and `--cache-only`. Cache metadata includes data, sources, environment, fitting configuration and reference-grid identity; incompatible entries fail rather than being silently overwritten.

The default uses 1,000 bootstrap replicates from the frozen specification. `--bootstrap 50` is suitable only for a separately labeled execution check, not final manuscript intervals. Comparison summaries and intervals can also take CPU time after fitting; progress messages identify each policy/grid.

An absent adaptive cache fails by default. Recover the original `fitted_2013.npz` through `fitted_2018.npz` files first. Only if they are genuinely unavailable, `--allow-adaptive-refit` explicitly permits missing adaptive fits to be regenerated. Existing adaptive files must still match the previous audit's hashes. The runner reproduces all four original validation point estimates and four final profit/yield point estimates before starting any fixed-grid fits. It stops if reproduction fails.

`--allow-environment-difference` permits explicitly exploratory fixed-grid fits under different Python/package versions. Old cached predictions retain their recorded identity; new fits record the actual environment. The first-year grid/count/assignment control remains mandatory. Prediction equivalence is reported but is mandatory only in the matching environment. Environment differences can confound adaptive-versus-fixed comparisons; use the original environment for the intended manuscript run. This flag does not bypass cache integrity or original-estimate reproduction.

## What is compared

| Policy | Construction |
| --- | --- |
| `recorded_rule` | The frozen selected ε/λ construction and all recorded support/overlap settings. With the adaptive grid, this reproduces the original selected evaluation. With the fixed grid, it is a new application of the same construction rule. |
| `guarded_greedy` | Set ε and λ to zero, retaining the original 30-count support mask, 60-count deviation guard and overlap rules. This is the original greedy candidate definition. |
| `supported_argmax` | Select maximum predicted profit with the same 30-count support and overlap requirements, ε=λ=0, and the 60-count deviation guard disabled. This explicitly tests the simpler supported argmax and the contribution of that guard. |

All three use the same fitted outcome and behavior predictions within a grid/year. This is a one-decision contextual bandit; the study should not imply that it establishes an advantage of sequential reinforcement learning. Simpler policies winning, tying or having an inconclusive contrast are valid outcomes.

The physical grid recovers the first saved adaptive boundaries, fitted on **2012 alone** before evaluating 2013. Its interior cuts and **finite midpoint representative N/P₂O₅/K₂O rates** are held constant for every evaluation year. It was not specified before the original study; it is a transparently retrospective robustness design using only earliest-year data to define actions. The reference cannot be moved to a later year via the configuration.

Assignment bins have open upper tails so later rates remain well defined; recommended representatives remain finite. Raw logged rates outside the observed 2012 range are flagged by nutrient and excluded in a separate range-restricted comparison. This restriction applies to scored observations; models and support counts still use all earlier training rows, including any rates outside that reference range. The cache stores numeric infinite upper boundaries, while `split_audit.json` writes them as `null` with `open_upper_tail: true`; finite original edges and recommended rates are in `fixed_grid_reference.json` and `physical_action_grid`. Fixed bins do not guarantee local positivity, agronomic safety or equivalence of different continuous rates within a bin.

The fixed grid requires refitting because the old cache contains predictions at the adaptive grid's representative rates, not reusable trained model objects. The original fitting code is loaded into a private namespace with only the binner replaced. Imputation, feature selection, reward models, propensity model, recency weighting and earlier-years-only training are retained. In the matching environment, the 2013 fixed/adaptive grid and predictions must agree as a control. Later-year 2018 outcomes never enter grid construction, nuisance fitting or policy choices for 2018.

## Cohorts and inference

| Cohort | Meaning |
| --- | --- |
| `native_overlap` | A variant's own overlap rows; useful for coverage, not a fair comparison with variants on other rows. |
| `within_grid_common` | Intersection of the three policy overlap masks for that grid. |
| `cross_grid_common` | Intersection of all six policy/grid masks. |
| `cross_grid_common_reference_range` | The six-way intersection restricted to logged rates within all three 2012 observed ranges. |

All intersections align year and field ID before calculating estimates. Coverage tables report row and cluster counts, size distributions and nonexclusive failure reasons. The inference target is always the retained observational population. An empty or nonestimable cohort is reported as unavailable.

Within a grid, `paired_policy_contrasts.csv` compares `recorded_rule` with each simpler policy on identical rows, with identical geographic cluster-bootstrap draws. Positive contrasts favor the recorded rule. The original weight method is retained, with PSIS computed on the relevant cohort and held fixed during resampling. Intervals condition on fitted models and cohort inclusion, use the original geographic keys, and are not adjusted for multiple comparisons. Undefined bootstrap draws make the interval unavailable; none are silently dropped.

Across grids, the learned behavior baselines can differ even on identical rows. Inspect `profit_V0`, `profit_V1`, coverage and their differences; do not interpret a difference between grid-specific Δprofit estimates as a paired effect against one identical baseline. There is intentionally no cross-grid paired policy contrast.

Pooled validation outputs retain the original globally smoothed/globally normalized aggregation and separately show within-year smoothing/normalization. These are labeled alternatives, not competing selection rules. Annual and final-year summaries use single-year aggregation. Both original profit-surrogate estimates and accounting-consistent estimates (`maize_price × yield_delta − matched_cost_delta`) are reported, with the surrogate gap retained.

## Outputs to inspect

| File | Purpose |
| --- | --- |
| Diagnostics: `findings.md` | Short forward-year prediction findings; makes no claim about the cause of a bad year. |
| `forward_prediction_metrics.csv` | All-row and retained-cohort bias, MAE, RMSE and R² for profit/yield, plus constant historical-training-mean and last-training-year-mean predictors. No evaluation outcomes enter those predictors. In comparison outputs the retained diagnostic cohort is `within_grid_common`; the fast diagnostic uses the original audit cohort. |
| Diagnostics: `distribution_shift.csv` | Context/action/outcome distributions versus prior years: missingness, means, ranges, standardized mean differences, descriptive KS statistics and out-of-training-range rates. Outcomes are explicitly diagnostic only. |
| Diagnostics: `adaptive_bin_stability.csv` | Physical boundary distances and representative-rate displacement from the 2012 grid; bin index equality alone does not mean equal fertilizer rates. |
| `residuals_by_action_cell.csv` | Prediction errors by year and logged joint cell, with historical counts; helps locate unreliable action regions. |
| Diagnostics: `reference_range_coverage.csv` | Later-year rates outside the reference range. |
| Diagnostics: `recorded_validation_contributions.csv` | The earlier audit's selected-policy aggregation decomposition, preserved for comparison with the prediction diagnostics. |
| Comparison: `original_reproduction_check.csv` | Must reproduce all eight original comparisons before fixed-grid fitting. |
| Comparison: `first_year_grid_control.csv` | First-year adaptive/fixed structural and prediction control. |
| Comparison: `reviewer_summary.csv` | Compact pooled-validation/final summaries on the six-way common and reference-range cohorts. Start here. |
| Comparison: `policy_estimates.csv` | Full annual/pooled profit, yield, cost, accounting and interval results for all policies and cohorts. |
| Comparison: `paired_policy_contrasts.csv` | Direct within-grid comparisons with simpler policies and paired intervals. |
| Comparison: `cohort_coverage.csv` | Coverage, clusters and exclusion reasons for every comparison. |
| Comparison: `propensity_scores.csv` | Multiclass Brier score, log loss, zero logged probabilities, unseen training cells, top-label accuracy and ECE; compares the fitted behavior model with training-only historical joint-cell frequencies. |
| Comparison: `top_label_reliability.csv`, `classwise_calibration.csv` | Top-label reliability in ten fixed bins and predicted versus observed class frequencies, with physical representative rates. These are diagnostics, not a fitted calibration correction. |
| Comparison: `comparison_rows.csv`, `split_audit.json` | Reusable per-row predictions/costs/masks, physical actions and exact fitting years. |
| `temporal_manifest.json` | Completion marker, source/input/output/cache hashes, environment, settings and exploratory status. Written only after success. |

The multiclass Brier score is the mean **sum** of squared class errors, range [0,2], not divided by the number of cells. Log loss explicitly floors logged probabilities at 10⁻¹² for its finite summary, and separately reports unclipped loss and true zeros. Top-label ECE uses each row's highest predicted probability and whether that class was observed; it does not flatten the numerous unobserved grid cells. Proper scores also reflect discrimination and are not pure calibration measures. Classwise average agreement does not prove conditional calibration. See the [scikit-learn calibration discussion](https://scikit-learn.org/stable/modules/calibration.html) and [log-loss definition](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.log_loss.html).

## Use in the manuscript revision

For R1.4, report actual forward-year prediction performance, prediction bias and historical-mean comparisons, then connect unusual years to the observed distribution shifts, residual-cell patterns and the earlier aggregation decomposition. Describe these as possible explanations to investigate, not demonstrated causes. For R2 major 4, report physical threshold stability and whether the conclusion persists under fixed representatives and the range restriction. For R1.6, report paired contrasts against both simpler policies, including uncertainty and coverage. Do not claim superiority from separate positive confidence intervals.

The code supplies the analyses; the comments are substantively addressed only after results are reviewed and incorporated into the manuscript and point-by-point response. Fixed-price and threshold sensitivities, quantitative confounding sensitivity, figure regeneration and final manuscript editing belong to subsequent packages. No result here justifies repeatedly tuning on 2018 or describing it as a newly untouched test set.
