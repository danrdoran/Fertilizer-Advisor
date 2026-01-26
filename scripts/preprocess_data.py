"""
What this script does

- Loads the Chiapas 2012–2018 field dataset
- Builds a pre-planting (V1–V6) feature set
- EXCLUDES from features:
    • Outcome proxy: Mun_yield
    • Post-treatment weather: all V7..V30 windows
    • Target actions: Nitrogen, Phosphorus, Potassium (these go to `actions`, not X)
    • (Keeps System and Tillage as categorical context features)
- Keeps variables knowable at/ before planting:
    • Static/location: Lat, Long, Elev, Slope, Clay, CEC, SOM, PH
    • Year, Planting day-of-year (sin/cos encoded)
    • Pre-plant weather aggregates (V1..V6): prcp/srad/tmax/tmin/vp
    • Categorical: System, Tillage (one-hot with Unknown)
- Outputs compressed processed data for bandit training/evaluation:
    • obs          : feature matrix (N x d)
    • actions      : actual (N,3) applied N,P,K (NOT part of obs)
    • reward       : yield in kg/ha (float)
    • year         : year (int)
    • clusters     : geo-tiles (int) — 0.01° rounding (~1 km)
    • field_id     : Field_ID (int)
    • lat, lon     : coordinates (float)
    • feature_cols : list of feature names (string array)
- Also writes:
    • data/processed/feature_cols.csv
    • data/processed/metadata.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- Constants ---------------------------------

PRE_WINDOWS = list(range(1, 7))          # V1..V6 (≈ 60 days pre-plant to planting)
WEATHER_VARS = ["prcp", "srad", "tmax", "tmin", "vp"]

# Columns to DROP from features (but 'actions' are still exported separately)
DROP_FROM_FEATURES = {
    "outcome_proxy": ["Mun_yield"],                      # contemporaneous outcome aggregate
    "actions": ["Nitrogen", "Phosphorus", "Potassium"],  # target actions (kept only for 'actions' array)
    # NOTE: System and Tillage are *kept* as categorical features; Cultivar remains excluded by default
    "co_decisions_excluded": ["Cultivar"],
}

# Pre-plant, ex-ante columns we KEEP (besides pre-window weather aggregates and categoricals below)
STATIC_BASE = [
    "Year", "Planting", "Lat", "Long", "Elev",
    "Slope", "Clay", "CEC", "SOM", "PH",
    # Field_ID is metadata only (not a feature)
]

# Categorical context variables to include in X
CATEGORICAL_KEEP = ["System", "Tillage"]

REWARD_COL = "YIELD"   # t/ha in the source; converted to kg/ha

# --------------------------- Helper functions ----------------------------

def _agg_pre_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pre-planting (V1..V6) weather into compact, leakage-safe features.
    - prcp: sum, mean, std, min, max
    - srad/tmax/tmin/vp: mean, std, min, max
    """
    out = pd.DataFrame(index=df.index)

    cols = {var: [f"{var}_V{i}" for i in PRE_WINDOWS] for var in WEATHER_VARS}
    missing = [c for var in WEATHER_VARS for c in cols[var] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected pre-window weather columns: {missing}")

    pr = df[cols["prcp"]]
    out["prcp_pre_sum"] = pr.sum(axis=1)
    out["prcp_pre_mean"] = pr.mean(axis=1)
    out["prcp_pre_std"]  = pr.std(axis=1)
    out["prcp_pre_min"]  = pr.min(axis=1)
    out["prcp_pre_max"]  = pr.max(axis=1)

    for var in ["srad", "tmax", "tmin", "vp"]:
        block = df[cols[var]]
        out[f"{var}_pre_mean"] = block.mean(axis=1)
        out[f"{var}_pre_std"]  = block.std(axis=1)
        out[f"{var}_pre_min"]  = block.min(axis=1)
        out[f"{var}_pre_max"]  = block.max(axis=1)

    # simple dry-spell proxy using decad totals (not daily)
    out["dry_decads_pre"] = (pr < 1.0).sum(axis=1)
    return out


def _encode_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode planting day-of-year as sin/cos; keep Year."""
    out = pd.DataFrame(index=df.index)
    planting = df["Planting"].copy()
    if planting.isna().any():
        planting = planting.fillna(planting.median())
    theta = 2.0 * np.pi * (planting / 365.25)
    out["planting_sin"] = np.sin(theta)
    out["planting_cos"] = np.cos(theta)
    out["Year"] = df["Year"].astype(int)
    return out


def _encode_location_soils(df: pd.DataFrame) -> pd.DataFrame:
    """Keep location and soil/topo variables; impute later."""
    keep = ["Lat", "Long", "Elev", "Slope", "Clay", "CEC", "SOM", "PH"]
    return df[keep].copy()


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode System and Tillage with an 'Unknown' bucket.
    Produces columns like: System=Conventional, System=Unknown, Tillage=No-till, ...
    """
    for col in CATEGORICAL_KEEP:
        if col not in df.columns:
            raise ValueError(f"Missing required categorical column: {col}")
    cats = df[CATEGORICAL_KEEP].copy()

    # normalize text and fill unknowns
    for c in CATEGORICAL_KEEP:
        cats[c] = (
            cats[c]
            .astype(str)
            .str.strip()
            .replace({"nan": "Unknown", "NaN": "Unknown", "None": "Unknown", "": "Unknown"})
        )

    dummies = []
    for c in CATEGORICAL_KEEP:
        d = pd.get_dummies(cats[c], prefix=f"{c}", prefix_sep="=", dtype=np.float32)
        # ensure an explicit Unknown column exists
        if f"{c}=Unknown" not in d.columns:
            d[f"{c}=Unknown"] = 0.0
        dummies.append(d)

    out = pd.concat(dummies, axis=1)
    # sort columns for deterministic ordering
    out = out.reindex(sorted(out.columns), axis=1)
    return out


def _make_clusters(df: pd.DataFrame) -> np.ndarray:
    """Geo-tiles based on 0.01° rounding (~1 km)."""
    lat_tile = np.round(df["Lat"].astype(float), 2)
    lon_tile = np.round(df["Long"].astype(float), 2)
    pairs = list(zip(lat_tile, lon_tile))
    uniq = {p: i for i, p in enumerate(sorted(set(pairs)))}
    return np.array([uniq[p] for p in pairs], dtype=np.int32)


def _assert_no_leakage(feature_cols: list):
    """
    Guardrail: assert we didn't accidentally include any V7..V30 weather,
    or banned columns such as Mun_yield / action variables.
    """
    banned_tokens = [f"_V{i}" for i in range(7, 31)]  # post-plant windows
    for name in feature_cols:
        if any(tok in name for tok in banned_tokens):
            raise AssertionError(f"Leakage detected: post-plant token in feature '{name}'")
    hard_bans = set(DROP_FROM_FEATURES["outcome_proxy"] + DROP_FROM_FEATURES["actions"])
    for name in feature_cols:
        if name in hard_bans:
            raise AssertionError(f"Leakage detected: banned source column present as feature '{name}'")


def _load_dataframe(input_path: Path) -> pd.DataFrame:
    """Load CSV or Excel (sheet 'Data' if present)."""
    if input_path.suffix.lower() in [".csv"]:
        return pd.read_csv(input_path)
    elif input_path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(input_path, sheet_name="Data")
        except ValueError:
            # fallback to first sheet
            return pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .csv or .xlsx/.xls")


# ------------------------------ Main logic -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/chiapas_maize.xlsx",
                        help="Path to the raw dataset CSV or Excel.")
    parser.add_argument("--out", type=str,
                        default="data/processed/processed_data.npz",
                        help="Output .npz path for proessed data.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_npz = Path(args.out)
    aux_dir = Path("data/processed")
    aux_dir.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------ Load & basic checks ------------------------
    df = _load_dataframe(input_path)

    required_cols = set(
        ["Field_ID", "Year", "Lat", "Long", "Elev", "Planting", REWARD_COL,
         "Slope", "Clay", "CEC", "SOM", "PH",
         "Nitrogen", "Phosphorus", "Potassium"] + CATEGORICAL_KEEP
    )
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # -------------------- Build leakage-safe feature set -----------------
    pre_wx     = _agg_pre_window(df)       # (1) Pre-plant aggregates (V1..V6)
    temporal   = _encode_temporal(df)      # (2) Year + Planting sin/cos
    loc_soil   = _encode_location_soils(df)# (3) Location & soils
    cat_ctx    = _encode_categoricals(df)  # (4) System & Tillage (categorical)

    # Concatenate features
    X = pd.concat([loc_soil, temporal, pre_wx, cat_ctx], axis=1)

    # Drop any banned *source* columns if they slipped in
    drop_candidates = [c for group in DROP_FROM_FEATURES.values() for c in group]
    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")

    # Simple numeric imputation (median) for numeric NaNs (dummies are numeric already)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

    feature_cols = X.columns.tolist()
    _assert_no_leakage(feature_cols)

    # -------------------------- Targets & actions ------------------------
    reward = (df[REWARD_COL].astype(float) * 1000.0).to_numpy()  # kg/ha
    actions = df[["Nitrogen", "Phosphorus", "Potassium"]].astype(float).to_numpy()

    years    = df["Year"].astype(int).to_numpy()
    clusters = _make_clusters(df)
    field_id = df["Field_ID"].astype(int).to_numpy()
    lat      = df["Lat"].astype(float).to_numpy()
    lon      = df["Long"].astype(float).to_numpy()

    # ------------------------------ Save ---------------------------------
    np.savez_compressed(
        out_npz,
        obs=X.to_numpy(dtype=float),
        actions=actions,
        reward=reward,
        profit=reward,  # placeholder identical to reward; profit computed later if prices provided
        year=years,
        clusters=clusters,
        field_id=field_id,
        lat=lat,
        lon=lon,
        feature_cols=np.array(feature_cols, dtype=object),
    )

    # Companion artifacts for inspection
    feature_cols_path = aux_dir / "feature_cols.csv"
    pd.Series(feature_cols).to_csv(feature_cols_path, index=False, header=False)

    meta = {
        "preprocessing": "pre-planting only (V1..V6), no Mun_yield, no action features; includes System & Tillage in X",
        "input_path": str(input_path),
        "out_npz": str(out_npz),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_groups": {
            "location_soil": [c for c in ["Lat", "Long", "Elev", "Slope", "Clay", "CEC", "SOM", "PH"] if c in feature_cols],
            "temporal": [c for c in ["Year", "planting_sin", "planting_cos"] if c in feature_cols],
            "preplant_weather": [c for c in feature_cols if c.endswith("_pre_mean")
                                 or c.endswith("_pre_std")
                                 or c.endswith("_pre_min")
                                 or c.endswith("_pre_max")
                                 or c == "prcp_pre_sum"
                                 or c == "dry_decads_pre"],
            "categorical": [c for c in feature_cols if c.startswith("System=") or c.startswith("Tillage=")],
        },
        "dropped_from_features": DROP_FROM_FEATURES,
        "categorical_keep": CATEGORICAL_KEEP,
        "pre_windows": PRE_WINDOWS,
        "weather_vars": WEATHER_VARS,
        "reward_units": "kg/ha",
        "clusters_note": "0.01° geo-tiles (~1 km) based on Lat/Long (rounded)",
    }
    with open(aux_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Console summary
    print("=" * 72)
    print("PREPROCESS FOR YIELD (pre-planting, includes System & Tillage in X)")
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols):,}")
    print("\nFeature groups:")
    for k, v in meta["feature_groups"].items():
        print(f"  - {k:16s}: {len(v):2d} cols")
    print("\nReward (kg/ha): "
          f"mean {reward.mean():,.0f} | std {reward.std():,.0f} | "
          f"min {reward.min():,.0f} | max {reward.max():,.0f}")
    print("\nSaved:")
    print(f"  - {out_npz}")
    print(f"  - {feature_cols_path}")
    print(f"  - {aux_dir/'metadata.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()