# Chronological policy evaluation for the manuscript revision

This workflow addresses Reviewer 1 concern 2 and the policy-selection part of Reviewer 2 major concern 5. It separates candidate selection from final-year evaluation and fixes prerequisites for trustworthy numerical results. Run it before revising the headline effect estimate. No new scientific results are committed with this implementation.

The analysis is a **retrospective reanalysis**. The existing research has already inspected 2018, so describe the revised procedure accurately; do not call 2018 a newly untouched test set or claim prospective preregistration. Observational confounding, price robustness and other outstanding reviewer concerns remain to be addressed.

## What the code does

1. Rebuild context features from the raw workbook with missing values preserved. Each year's imputation medians are then learned only from that year's training rows. New data and companion metadata go into a separate output directory.
2. Use 2013–2017 for candidate selection. For each validation year, learn bins, support counts, outcome models and the behavior model using earlier years only. The 2012 data provide the first training set. Internal outcome-model stacking uses geographic groups from the training rows.
3. Reuse each year's fitted predictions across the four candidates declared in `configs/revision_evaluation.json`. This avoids retraining identical models for each exploration/mixing setting. The candidate named `greedy` still retains the common support and SPIBB-style rules; it is not the unrestricted baseline ablation requested elsewhere in the reviews.
4. Compare candidates on the intersection of their overlap masks, so their validation scores describe the same observations. Rank candidates by the pooled profit SNDR confidence-interval lower endpoint; resolve ties by their order in the configuration. These validation intervals are exploratory and are not adjusted simultaneous intervals. Candidates without an estimable SNDR interval remain in the audit table and are ineligible for selection. If none is estimable, no choice is frozen.
5. Write the selected policy, complete specification, input and source SHA-256 hashes, core package versions, training-year audit, bin boundaries and cluster summaries to the selection output. The final-year actions, outcomes and contexts are excluded from the data passed to candidate fitting and scoring. A full-file hash identifies the input but is not used to rank policies.
6. In a separate command, check the recorded input, source, package versions and policy settings. Refit the selected learning procedure using 2012–2017 only, then evaluate exactly that configuration on 2018. The main endpoint is final-year profit SNDR; ordinary DR and yield are secondary. The final command offers no policy or threshold overrides and refuses to overwrite an existing output directory.

The policy is a learning procedure whose fitted recommendations depend on training data. Its fixed choices are the candidate parameters, prices, estimator, support settings and split rule, not the individual action recommendations from a particular validation fold.

## Numerical changes

- ArviZ is explicitly included in the research dependencies. A PSIS estimator run raises an error when ArviZ is unavailable instead of silently returning unsmoothed ratios.
- ArviZ's sum-one PSIS output is rescaled to preserve the original positive ratios' total mass. Exact zero ratios remain zero. This fixes the extra reduction of ordinary DR's residual correction caused by passing sum-one weights into a sample mean. The convention is recorded in the run manifest; old DR numbers must be regenerated.
- The default OPE engine uses earlier years only. The legacy `--allow_eval_year_crossfit` flag explicitly opts into retrospective same-year cross-fitting; the revision runner does not permit it.
- `trim_ratio_max=10` is an observation-exclusion threshold. `max_density_ratio=50` is a different setting, used only with the explicitly selected capped-weight estimator. PSIS uses raw retained ratios before smoothing.
- SNDR samples with no positive importance weights and samples with fewer than two geographic clusters cannot support the implemented interval and are reported as unavailable. Two clusters are a minimum computational check, not a claim that such a small sample gives reliable inference.
- Field identifiers use the `field_id` key written by preprocessing, and final row-level diagnostics retain those identifiers.

## Run the analysis

Use Python 3.12 in a separate research environment. No OpenAI key, Streamlit session or GPU is needed. The full experiment fits multiple boosted ensembles and can take substantial CPU time; the fast regression suite is not a timing estimate for the research run.

From an existing repository checkout:

```bash
git fetch origin
git switch revision/independent-policy-evaluation
python3.12 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv-research` and activate it with `.\.venv-research\Scripts\Activate.ps1`. The remaining Python commands work in either shell. The research requirements pin direct dependencies; each selection manifest also records the actual core library versions used. Preserve a full environment snapshot with `python -m pip freeze` if reproducing the run on another machine.

Rebuild from the raw workbook; the bundled processed NPZ was globally imputed and is deliberately rejected by this workflow:

```bash
python scripts/preprocess_data.py --defer-imputation --out results/revision_data/data.npz
```

Review the candidate grid and analysis specification before selection. Then freeze the choice:

```bash
python scripts/run_revision_evaluation.py select --config configs/revision_evaluation.json --data results/revision_data/data.npz --output results/revision_selection
```

Evaluate the recorded choice:

```bash
python scripts/run_revision_evaluation.py evaluate --selection results/revision_selection/selection.json --data results/revision_data/data.npz --output results/revision_final
```

Both stage output directories must be new. An interrupted run can leave a partial directory; a complete selection has `selection.json`, which is written last. Retain the failed run for diagnosis and use a new output directory when retrying. Do not change the grid after inspecting final-year results and then present the reused final year as independent confirmation.

After both stages finish, run the [aggregation and economic-accounting audit](revision_audit.md). It first reads the saved files without fitting, then optionally refits and caches the missing predictions to explain aggregation and accounting differences. Keep the original selection and final outputs intact.

## Files to inspect

| File | Purpose |
| --- | --- |
| `results/revision_selection/validation_candidates.csv` | All declared candidates, eligibility or failure reason, common-cohort SNDR estimates, intervals and selection scores |
| `results/revision_selection/validation_by_year.csv` | Year-specific candidate results and common-cohort sample sizes |
| `results/revision_selection/split_audit.json` | Training years, training medians, bin boundaries and evaluation cluster counts for each validation year |
| `results/revision_selection/selection.json` | Recorded choice, specification, hashes, package versions, selection rule and interpretation notice |
| `results/revision_final/final_summary.csv` | Final-year profit and yield estimates for SNDR and DR, intervals, row counts and retention fraction |
| `results/revision_final/final_results.json` | Complete final results, linked selection manifest, diagnostics, cluster summary and training audit |
| `results/revision_final/holdout_rows.csv` | Field identifiers, overlap inclusion, propensities, raw ratios and support counts |

For a compact local readout:

```bash
python -c "import pandas as pd; print(pd.read_csv('results/revision_final/final_summary.csv').to_string(index=False))"
```

## How to use the outputs in the revision

Report the exact selected configuration, validation years, training years, final evaluation year, retained population, baseline and price assumptions. The research baseline is the estimated behavior-policy distribution, not the single modal action displayed in the app. The final-year endpoint is different from the previous pooled 2013–2018 result, so update the text and tables consistently rather than replacing a single number in the old narrative.

The bootstrap pairs target and baseline estimates while resampling geographic clusters. It holds fitted predictions, selected settings, overlap inclusion and the PSIS transformation fixed during resampling. Intervals are conditional on those fitted objects; they do not include refitting or policy-selection uncertainty. Geographic clusters can also recur across years, so chronological separation does not imply independence of farms or spatial environments.

Inspect tail diagnostics and cluster counts before making reliability claims. A positive lower bound is not proof of causal identification, guaranteed improvement on individual farms or deployability. Support trimming remains the existing action-dependent procedure; these results describe that retained sample and do not establish effects for the full population.

The original run environment and results still need reconciliation with the manuscript. This workflow deliberately changes chronological training, model stacking groups and the ordinary DR weight scaling, so it should not be expected to reproduce every original table. New price/support sensitivity analyses, unmeasured-confounding analysis, baseline ablations and figure improvements are subsequent work.
