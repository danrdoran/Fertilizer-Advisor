# Fertilizer-Advisor

This repo provides an end-to-end pipeline for:

1. Preprocessing agronomic maize field data,
2. Training a reward crop-profit model and behavior policy (π₀) model,
3. Learning and evaluating a new fertilizer policy offline, and
4. Running a decision support app that recommends N–P₂O₅–K₂O rates.

> **Note:** Running the app requires an OPENAI_API_KEY, as it utilizes LLM calls in a RAG framework to give further maize management guidance grounded in region-specific technical documentation.

See docs/paper.pdf to read more about the project.

---

## Quickstart

```bash
# 1) Preprocess raw data → processed data
python scripts/preprocess_data.py

# 2) Train ensemble yield model and behavior policy model (π0)
python scripts/train_models.py \
  --data data/transitions/transitions_yield.npz \
  --features data/transitions/feature_cols.csv \
  --output_dir ensemble_model \
  --n_trials 4 \
  --recency_lambda 0.10 \
  --holdout_year 2018

# 3) Learn & evaluate new fertilizer policy (π1) offline
python scripts/learn_and_evaluate_policy.py \
  --data data/transitions/transitions_yield.npz \
  --out_weights_csv results/weights_all.csv \
  --objective profit \
  --maize_price 3.5 --priceN 16 --priceP 12 --priceK 8 \
  --trim_enable \
  --epsilon_explore 0.1 \
  --lambda_mix 0.0 \
  --cohort overlap \
  --weights psis \
  --recency_lambda 0.10 \
  --cf_k 5 \
  --alpha 0.05 \
  --n_bootstrap 1000 \
  --out_json results/ope_by_year_result.json \
  --out_csv results/ope_by_year_summary.csv \
  --by_year_csv results/ope_by_year.csv \
  --out_policy_csv results/ope_by_year_policy_rows.csv

# 4) Build & run the app with Docker (Make sure OPENAI_API_KEY is set in a local .env file)
docker build -t fertilizer-advisor .
docker run --rm -p 8501:8501 --env-file .env fertilizer-advisor
```
