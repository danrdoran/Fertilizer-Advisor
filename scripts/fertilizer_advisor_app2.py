"""
Fertilizer Advisor
Farmer-facing Streamlit app for N-P-K recommendations.
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from openai import OpenAI

from rag_faiss import (
    get_or_build_store,
    retrieve,
    answer_with_rag_chat_completions,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
)

warnings.filterwarnings("ignore")

# =============================
# Translations
# =============================

TRANSLATIONS = {
    "en": {
        # App
        "APP_TITLE": "🌾 Fertilizer Advisor",
        "APP_TAGLINE": (
            "Get personalized **N–P₂O₅–K₂O** recommendations based on your field conditions. "
            "Enter prices to see profit optimization and economic analysis."
        ),
        "TOGGLE_LABEL_ES": "Cambiar a español",
        "TOGGLE_LABEL_EN": "Switch to English",
        "LOADING_MODELS": "Loading models…",
        # Sidebar
        "ABOUT": "About",
        "ABOUT_BULLETS": (
            "- Models trained on **2012–2018** data from Chiapas, Mexico\n"
            "- Uses only **pre-plant** information\n"
            "- Recommends actions with **adequate historical support**\n"
            "- Guardrails: N≤240, P₂O₅≤90, K₂O≤60 kg/ha\n"
            "- **Baseline**: Modal action from behavior policy (π₀) among supported cells"
        ),
        # Sections
        "FIELD_INFO": "📍 Field Information",
        "LATITUDE": "Latitude",
        "LONGITUDE": "Longitude",
        "ELEVATION": "Elevation (m)",
        "SLOPE": "Slope (%)",
        "CLAY": "Clay (%)",
        "CEC": "CEC (cmolc/dm³)",
        "SOM": "Soil Organic Matter (%)",
        "PH": "Soil pH",
        "SEASON_YEAR": "Season year",
        "PLANTING_DATE": "Expected planting date",
        "MANAGEMENT": "🧑🏽‍🌾 Management",
        "SYSTEM": "System",
        "TILLAGE": "Tillage",
        "PREPLANT_WEATHER": "🌦️ Weather (V1–V6)",
        "PREPLANT_CAPTION": "Enter typical values for the ~60 days before planting.",
        "EXPANDER_PRCP_SRAD": "V{n} precipitation & radiation",
        "EXPANDER_T_VP": "V{n} temperature & vapor pressure",
        "PRCP": "Precipitation V{n} (mm/day)",
        "SRAD": "Solar radiation V{n} (MJ/m²/day)",
        "TMAX": "Max temperature V{n} (°C)",
        "TMIN": "Min temperature V{n} (°C)",
        "VP": "Vapor pressure V{n} (Pa)",
        "PRICES": "💵 Prices",
        "CURRENCY": "Currency",
        "CURRENCY_HELP": "Your local currency (e.g., MXN, USD)",
        "MAIZE_PRICE": "Maize price ({currency}/tonne)",
        "MAIZE_PRICE_HELP": "Current market price for maize grain",
        "N_PRICE": "N price ({currency}/kg)",
        "P_PRICE": "P₂O₅ price ({currency}/kg)",
        "K_PRICE": "K₂O price ({currency}/kg)",
        "GET_RECO": "🚀 Get Recommendation",
        "CALC_OPT": "Calculating optimal fertilizer rates...",
        "RECO_READY": "✅ Recommendation Generated",
        "RECO_RATES": "📊 Recommended Fertilizer Rates",
        "N_LABEL": "Nitrogen (N)",
        "P_LABEL": "Phosphorus (P₂O₅)",
        "K_LABEL": "Potassium (K₂O)",
        "EXPECTED_YIELD": "Expected Yield",
        "PRED_IMPROVEMENTS": "📈 Predicted Improvements vs Baseline",
        "PRED_YIELD_GAIN": "**Predicted Yield Gain**: {delta_yield:+,.0f} kg/ha\n\n*This is a model-based prediction comparing the recommended action ({N:.0f}-{P:.0f}-{K:.0f}) to the typical baseline action ({N_base:.0f}-{P_base:.0f}-{K_base:.0f}) for your conditions.*",
        "PRED_PROFIT_GAIN": "**Predicted Profit Gain**: {delta_profit:+,.0f} {currency}/ha\n\n*This is the predicted increase in net profit (revenue minus fertilizer cost) compared to the baseline action, based on current prices.*",
        "CONF_HIGH": "✅ **High confidence**: This recommendation is based on sufficient historical data.",
        "CONF_LOW": "⚠️ **Limited data**: Few historical observations for this combination. Consider on-farm testing.",
        "SUSTAINABILITY_CONCERN": "⚠️ **Sustainability concern**: ",
        "ECON_ANALYSIS": "💰 Economic Analysis",
        "GROSS_REVENUE": "Gross Revenue",
        "FERT_COST": "Fertilizer Cost",
        "NET_PROFIT": "Net Profit",
        "ROI": "Return on Investment",
        "WHAT_IT_MEANS": "💡 **What this means**: At current prices, the recommended rates provide **{delta_profit:+,.0f} {currency}/ha** additional profit compared to the baseline approach. Every {currency} spent on fertilizer returns {roi_per_1:.2f} {currency} in maize revenue.",
        "BASELINE_DETAILS": "📊 Baseline Comparison Details",
        "METRIC": "Metric",
        "BASELINE": "Baseline",
        "RECOMMENDED": "Recommended",
        "DIFFERENCE": "Difference",
        "BASELINE_NOTE": "**Note**: The baseline represents the modal (most common) supported action from the behavior policy (π₀) for your specific field conditions. All comparisons shown are model-based predictions, not causal guarantees.",
        "ALTERNATIVES": "Alternative Options",
        "BEST": "⭐ Best",
        "BETTER": "👌 Better",
        "GOOD": "👍 Good",
        "DELTA_YIELD_VS_BASE": "Δ Yield vs Baseline",
        "DELTA_PROFIT_VS_BASE": "Δ Profit vs Baseline",
        "CONFIDENCE": "Confidence",
        "CONF_HIGH_WORD": "High",
        "CONF_LOW_WORD": "Low",
        "TECH_DETAILS": "🔧 Technical Details",
        "BIN_INDICES": "- Bin indices: N={iN}, P={iP}, K={iK}",
        "OBJECTIVE": "- Objective: **{objective_label}**",
        "EXPLORATION": "- Exploration (ε): {epsilon}",
        "SUPPORT_THR": "- Support threshold: {min_support} observations/bin",
        "BASELINE_IDX": "- Baseline action index: {baseline_idx}",
        "POLICY": "- Policy: ε = {epsilon:.2f}, λ = {lambda_mix:.2f}",
        "PRICE_ASSUMPTIONS": "**Price assumptions:**",
        "DOCS_HEADER": "📚 Consult Technical Documentation",
        "DOCS_CAPTION": (
            "This tool searches the *National Institute of Forestry, Agricultural "
            "and Livestock Research* (INIFAP) technical agenda for follow-on maize "
            "management guidance and answers your questions with citations."
        ),
        "DOCS_LOADING_INDEX": "Loading technical documentation index…",
        "DOCS_MISSING_PDF": "Missing PDF at: {pdf_path}. Place 'agenda-tecnica-chiapas.pdf' next to the app.",
        "DOCS_CHAT_INPUT": "Ask about maize management in Chiapas (planting, fertilizer timing, weeds, pests, harvest)…",
        "DOCS_SEARCHING_ANSWER": "Searching and generating an answer…",
        "DOCS_SHOW_PASSAGES": "Show retrieved passages (with page numbers)",
        "DOCS_PASSAGE_META": "**p. {page}** • score={score:.3f}", 
    },
    "es": {
        # App
        "APP_TITLE": "🌾 Asesor de Fertilizantes",
        "APP_TAGLINE": (
            "Obtén recomendaciones personalizadas de **N–P₂O₅–K₂O** según las condiciones de tu parcela. "
            "Ingresa precios para ver la optimización de ganancias y el análisis económico."
        ),
        "TOGGLE_LABEL_ES": "Cambiar a español",
        "TOGGLE_LABEL_EN": "Switch to English",
        "LOADING_MODELS": "Cargando modelos…",
        # Sidebar
        "ABOUT": "Acerca de",
        "ABOUT_BULLETS": (
            "- Modelos entrenados con datos **2012–2018** de Chiapas, México\n"
            "- Usa solo información **antes de la siembra**\n"
            "- Recomienda acciones con **soporte histórico adecuado**\n"
            "- Límites: N≤240, P₂O₅≤90, K₂O≤60 kg/ha\n"
            "- **Línea base**: Acción modal de la política de comportamiento (π₀) entre celdas con soporte"
        ),
        # Sections
        "FIELD_INFO": "📍 Información de la parcela",
        "LATITUDE": "Latitud",
        "LONGITUDE": "Longitud",
        "ELEVATION": "Elevación (m)",
        "SLOPE": "Pendiente (%)",
        "CLAY": "Arcilla (%)",
        "CEC": "CIC (cmolc/dm³)",
        "SOM": "Materia Orgánica (%)",
        "PH": "pH del suelo",
        "SEASON_YEAR": "Año agrícola",
        "PLANTING_DATE": "Fecha estimada de siembra",
        "MANAGEMENT": "🧑🏽‍🌾 Manejo",
        "SYSTEM": "Sistema",
        "TILLAGE": "Labranza",
        "PREPLANT_WEATHER": "🌦️ Clima (V1–V6)",
        "PREPLANT_CAPTION": "Ingresa valores típicos para ~60 días antes de la siembra.",
        "EXPANDER_PRCP_SRAD": "V{n} precipitación y radiación",
        "EXPANDER_T_VP": "V{n} temperatura y presión de vapor",
        "PRCP": "Precipitación V{n} (mm/día)",
        "SRAD": "Radiación solar V{n} (MJ/m²/día)",
        "TMAX": "Temperatura máx V{n} (°C)",
        "TMIN": "Temperatura mín V{n} (°C)",
        "VP": "Presión de vapor V{n} (Pa)",
        "PRICES": "💵 Precios",
        "CURRENCY": "Moneda",
        "CURRENCY_HELP": "Tu moneda local (p. ej., MXN, USD)",
        "MAIZE_PRICE": "Precio del maíz ({currency}/tonelada)",
        "MAIZE_PRICE_HELP": "Precio de mercado del grano de maíz",
        "N_PRICE": "Precio de N ({currency}/kg)",
        "P_PRICE": "Precio de P₂O₅ ({currency}/kg)",
        "K_PRICE": "Precio de K₂O ({currency}/kg)",
        "GET_RECO": "🚀 Obtener recomendación",
        "CALC_OPT": "Calculando dosis óptimas de fertilizante...",
        "RECO_READY": "✅ Recomendación generada",
        "RECO_RATES": "📊 Dosis de fertilizante recomendadas",
        "N_LABEL": "Nitrógeno (N)",
        "P_LABEL": "Fósforo (P₂O₅)",
        "K_LABEL": "Potasio (K₂O)",
        "EXPECTED_YIELD": "Rendimiento esperado",
        "PRED_IMPROVEMENTS": "📈 Mejoras previstas vs línea base",
        "PRED_YIELD_GAIN": "**Aumento de rendimiento previsto**: {delta_yield:+,.0f} kg/ha\n\n*Es una predicción basada en el modelo comparando la acción recomendada ({N:.0f}-{P:.0f}-{K:.0f}) con la acción típica de referencia ({N_base:.0f}-{P_base:.0f}-{K_base:.0f}) para tus condiciones.*",
        "PRED_PROFIT_GAIN": "**Aumento de ganancia prevista**: {delta_profit:+,.0f} {currency}/ha\n\n*Incremento en la ganancia neta (ingreso menos costo de fertilizante) vs la línea base, con los precios actuales.*",
        "CONF_HIGH": "✅ **Alta confianza**: Esta recomendación se basa en datos históricos suficientes.",
        "CONF_LOW": "⚠️ **Datos limitados**: Pocas observaciones históricas para esta combinación. Considera ensayos en campo.",
        "SUSTAINABILITY_CONCERN": "⚠️ **Alerta de sostenibilidad**: ",
        "ECON_ANALYSIS": "💰 Análisis económico",
        "GROSS_REVENUE": "Ingreso bruto",
        "FERT_COST": "Costo de fertilizante",
        "NET_PROFIT": "Ganancia neta",
        "ROI": "Retorno sobre la inversión",
        "WHAT_IT_MEANS": "💡 **Qué significa**: Con los precios actuales, las dosis recomendadas aportan **{delta_profit:+,.0f} {currency}/ha** adicionales vs la práctica base. Cada {currency} invertido en fertilizante retorna {roi_per_1:.2f} {currency} en ingresos por maíz.",
        "BASELINE_DETAILS": "📊 Detalles de la línea base",
        "METRIC": "Métrica",
        "BASELINE": "Línea base",
        "RECOMMENDED": "Recomendado",
        "DIFFERENCE": "Diferencia",
        "BASELINE_NOTE": "**Nota**: La línea base es la acción modal (más común) de la política de comportamiento (π₀) para tus condiciones. Todas las comparaciones son predicciones del modelo, no garantías causales.",
        "ALTERNATIVES": "Opciones alternativas",
        "BEST": "⭐ Mejor",
        "BETTER": "👌 Muy buena",
        "GOOD": "👍 Buena",
        "DELTA_YIELD_VS_BASE": "Δ Rendimiento vs línea base",
        "DELTA_PROFIT_VS_BASE": "Δ Ganancia vs línea base",
        "CONFIDENCE": "Confianza",
        "CONF_HIGH_WORD": "Alta",
        "CONF_LOW_WORD": "Baja",
        "TECH_DETAILS": "🔧 Detalles técnicos",
        "BIN_INDICES": "- Índices de intervalos: N={iN}, P={iP}, K={iK}",
        "OBJECTIVE": "- Objetivo: **{objective_label}**",
        "EXPLORATION": "- Exploración (ε): {epsilon}",
        "SUPPORT_THR": "- Umbral de soporte: {min_support} observaciones/bin",
        "BASELINE_IDX": "- Índice de acción base: {baseline_idx}",
        "POLICY": "- Política: ε = {epsilon:.2f}, λ = {lambda_mix:.2f}",
        "PRICE_ASSUMPTIONS": "**Supuestos de precios:**",
        "DOCS_HEADER": "📚 Consultar documentación técnica",
        "DOCS_CAPTION": (
            "Esta herramienta busca en la agenda técnica del "
            "*Instituto Nacional de Investigaciones Forestales, Agrícolas y Pecuarias* "
            "(INIFAP) recomendaciones complementarias para el manejo del maíz y "
            "responde tus preguntas con citas."
        ),
        "DOCS_LOADING_INDEX": "Cargando índice de documentación técnica…",
        "DOCS_MISSING_PDF": "Falta el PDF en: {pdf_path}. Coloca 'agenda-tecnica-chiapas.pdf' junto a la app.",
        "DOCS_CHAT_INPUT": "Pregunta sobre manejo del maíz en Chiapas (siembra, fertilización, malezas, plagas, cosecha)…",
        "DOCS_SEARCHING_ANSWER": "Buscando y generando una respuesta…",
        "DOCS_SHOW_PASSAGES": "Mostrar pasajes recuperados (con números de página)",
        "DOCS_PASSAGE_META": "**p. {page}** • puntuación={score:.3f}",
    },
}

def t(lang: str, key: str, **kwargs) -> str:
    """Translate helper with safe fallback, supporting format placeholders."""
    base = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))
    try:
        return base.format(**kwargs)
    except Exception:
        return base

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = Path("ensemble_model")
DATA_PATH = Path("data/processed/processed_data.npz")
FEATURE_COLS_PATH = Path("data/processed/feature_cols.csv")

# Deployment defaults (policy params)
DEFAULT_YEAR = 2019
EPSILON_DEPLOY = 0.10           # acceptance_push
LAMBDA_MIX     = 0.00           # acceptance_push
MIN_SUPPORT_COUNT = 30          # acceptance_push min_support
epsilon = EPSILON_DEPLOY

# --- SPIBB: safe deviation threshold ---
SPIBB_MIN_DEVIATION_COUNT = 60  # if a joint N-P-K bin has < 60 samples, defer to baseline

# Agronomic guardrails
MAX_N = 240.0
MAX_P2O5 = 90.0
MAX_K2O = 60.0

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
@st.cache_resource
def load_models():
    """Load ensemble, binner, features, TRAIN support, and π0."""
    ensemble = joblib.load(MODEL_DIR / "ensemble_model.pkl")
    binner   = joblib.load(MODEL_DIR / "binner.pkl")
    feature_cols = pd.read_csv(FEATURE_COLS_PATH, header=None)[0].tolist()

    data = np.load(DATA_PATH)
    actions = data["actions"]
    years   = data["year"]
    # optional keys if present: X, y
    X = data["X"] if "X" in data.files else None
    y = data["y"] if "y" in data.files else None

    # TRAIN mask (<= 2018)
    train_mask = years <= 2018
    train_actions = actions[train_mask]
    support_mask, counts, n_bins = compute_support_mask(binner, train_actions, MIN_SUPPORT_COUNT)

    # π0 model for baseline
    pi0_model = None
    pi0_path = MODEL_DIR / "propensity_model.pkl"
    if pi0_path.exists():
        pi0_model = joblib.load(pi0_path)

    return (ensemble, binner, feature_cols,
            support_mask, counts, n_bins, pi0_model)

def compute_support_mask(binner, train_actions, min_count=10):
    """Mark action bins with adequate historical coverage."""
    train_binned = binner.transform(train_actions)
    n_bins = tuple(len(binner.bin_edges[k]) - 1 for k in ["N", "P", "K"])
    n_actions = int(np.prod(n_bins))

    joint_idx = (
        train_binned[:, 0] * (n_bins[1] * n_bins[2]) +
        train_binned[:, 1] * n_bins[2] +
        train_binned[:, 2]
    )
    counts = np.bincount(joint_idx, minlength=n_actions)
    support_mask = counts >= min_count
    return support_mask, counts, n_bins

def build_action_grid(binner):
    """Return action grid and helper to map flat idx -> (iN,iP,iK)."""
    centers = binner.bin_centers
    Nc, Pc, Kc = centers["N"], centers["P"], centers["K"]
    grid = np.stack(np.meshgrid(Nc, Pc, Kc, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = (len(Nc), len(Pc), len(Kc))

    def unravel(idx):
        iN = idx // (shape[1] * shape[2])
        rem = idx % (shape[1] * shape[2])
        iP = rem // shape[2]
        iK = rem % shape[2]
        return int(iN), int(iP), int(iK)

    return grid, shape, unravel

def create_feature_vector(inputs: dict, feature_cols: list):
    """Align user inputs to model feature order."""
    x = np.zeros(len(feature_cols), dtype=float)
    for i, col in enumerate(feature_cols):
        if col in inputs:
            x[i] = inputs[col]
    return x.reshape(1, -1)

def predict_yield(ensemble, X_row, action_grid):
    """Predict yields (kg/ha) across all actions for one field."""
    X_rep = np.repeat(X_row, len(action_grid), axis=0)
    mu = ensemble.predict(X_rep, action_grid)
    return mu.astype(float)

def profit_from(mu_kg_ha, action_grid, maize_price_per_tonne, pN, pP, pK):
    """Compute gross margin (revenue - fertilizer cost) per ha."""
    revenue = (mu_kg_ha / 1000.0) * maize_price_per_tonne
    costs = action_grid[:, 0] * pN + action_grid[:, 1] * pP + action_grid[:, 2] * pK
    return revenue - costs

def get_baseline_action(pi0_model, X_row, support_mask, counts):
    """Get the modal supported action from π0 distribution."""
    if pi0_model is None:
        # Fallback: most common supported action in training data
        supported_idx = np.where(support_mask)[0]
        if len(supported_idx) > 0:
            # Return the supported action with highest count
            supported_counts = counts[supported_idx]
            best_supported = supported_idx[np.argmax(supported_counts)]
            return int(best_supported)
        return 0
    
    # Get π0 probabilities
    pi0_probs = pi0_model.predict_proba_grid(X_row)  # shape (1, n_actions)
    
    # Mask to supported actions only
    pi0_masked = np.where(support_mask[None, :], pi0_probs, 0.0)
    pi0_masked = pi0_masked / np.clip(pi0_masked.sum(), 1e-12, None)
    
    # Return mode (highest probability action)
    return int(np.argmax(pi0_masked[0]))

def select_action(mu, profit, support_mask, epsilon=0.0):
    """Pick an action index using supported-only argmax, with optional ε-greedy."""
    objective = profit if profit is not None else mu
    supported = np.where(support_mask)[0]
    
    if len(supported) > 0:
        obj_masked = np.full_like(objective, -np.inf, dtype=float)
        obj_masked[supported] = objective[supported]
    else:
        obj_masked = objective

    if epsilon > 0 and np.random.rand() < epsilon and len(supported) > 0:
        idx = int(np.random.choice(supported))
    else:
        idx = int(np.nanargmax(obj_masked))
    
    is_supported = bool(support_mask[idx])
    return idx, is_supported

def calculate_roi(profit, fertilizer_cost):
    """Calculate return on investment percentage."""
    if fertilizer_cost <= 0:
        return np.inf
    return (profit / fertilizer_cost) * 100

def action_from_idx(action_grid, idx):
    N, P, K = action_grid[idx]
    return float(N), float(P), float(K)

def within_guardrails(N, P, K):
    flags = []
    if N > MAX_N: flags.append(f"N>{MAX_N:g}")
    if P > MAX_P2O5: flags.append(f"P₂O₅>{MAX_P2O5:g}")
    if K > MAX_K2O: flags.append(f"K₂O>{MAX_K2O:g}")
    return flags

def deterministic_seed_from_inputs(lat: float, lon: float, season_year: int) -> int:
    """
    Build a stable seed from available inputs (no farmer_id/field_id in dataset).
    Rounding avoids small entry jitter changing the seed.
    """
    key = (round(float(lat), 4), round(float(lon), 4), int(season_year))
    return hash(key) & 0xFFFFFFFF

def pi0_masked_grid(pi0_model, X_row, n_actions: int, support_mask: np.ndarray) -> np.ndarray:
    """
    π0 on the full action grid → mask to TRAIN support → renormalize.
    If π0 model missing, fall back to uniform over support.
    Returns shape (1, A).
    """
    if pi0_model is not None:
        p = np.asarray(pi0_model.predict_proba_grid(X_row), float)  # (1, A)
    else:
        p = np.zeros((1, n_actions), float)
        if support_mask.sum() > 0:
            p[:, support_mask] = 1.0 / float(support_mask.sum())
        else:
            p[:] = 1.0 / float(n_actions)

    pm = np.where(support_mask[None, :], p, 0.0)
    pm /= np.clip(pm.sum(axis=1, keepdims=True), 1e-12, None)
    return pm  # (1, A)

def build_pi1_distribution(objective: np.ndarray,
                           support_mask: np.ndarray,
                           pi0_masked: np.ndarray,
                           epsilon: float,
                           lambda_mix: float,
                           alpha: float = 0.0) -> np.ndarray:
    """
    π1(a|x) = λ·π0_masked(a|x) + (1−λ)·[(1−ε)·δ_greedy + ε·((1−α)·U_support)]
      - all distributions are masked to TRAIN support
    """
    A = objective.shape[0]
    # Greedy over TRAIN support
    if support_mask.sum() > 0:
        obj_masked = np.full(A, -np.inf, float)
        obj_masked[support_mask] = objective[support_mask]
        best = int(np.nanargmax(obj_masked))
        base = np.zeros(A, float); base[best] = 1.0
        uni_all = np.zeros(A, float); uni_all[support_mask] = 1.0 / float(support_mask.sum())
    else:
        best = int(np.nanargmax(objective))
        base = np.zeros(A, float); base[best] = 1.0
        uni_all = np.ones(A, float) / float(A)

    # ε-split between support
    pi_eps = (1.0 - epsilon) * base + epsilon * ((1.0 - alpha) * uni_all)

    # λ-mix with π0 (masked to TRAIN support)
    pi1 = lambda_mix * pi0_masked[0] + (1.0 - lambda_mix) * pi_eps
    pi1 /= np.clip(pi1.sum(), 1e-12, None)
    return pi1

def sample_action_from_pi1(pi1: np.ndarray, seed: int) -> int:
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(pi1), p=pi1))

# --- Percent helpers (for captions under metrics) ---
def percent_change(new, base):
    """Return percent change vs baseline, or None if undefined."""
    try:
        if base is None or not np.isfinite(base) or abs(float(base)) < 1e-9:
            return None
        return 100.0 * (float(new) - float(base)) / abs(float(base))
    except Exception:
        return None

def show_percent_caption(col, pct, invert=False):
    """
    Render a caption like '+3.2% gain' or '-3.2% loss' directly under a st.metric.
    If invert=True (e.g., for cost), lower is 'gain' and higher is 'loss'.
    """
    if pct is None:
        col.caption("—")
        return
    if invert:
        pct = -pct
    label = "gain" if pct >= 0 else "loss"
    col.caption(f"{pct:+.1f}% {label}")

def _bin_indices_from_actions(binner, actions):
    """Map continuous actions to joint bin indices."""
    b = binner.transform(actions)
    n_bins = tuple(len(binner.bin_edges[k]) - 1 for k in ["N", "P", "K"])
    idx = b[:, 0] * (n_bins[1] * n_bins[2]) + b[:, 1] * n_bins[2] + b[:, 2]
    return idx.astype(int), n_bins

def apply_spibb_projection(pi1: np.ndarray,
                           counts: np.ndarray,
                           baseline_idx: int,
                           min_deviation_count: int = SPIBB_MIN_DEVIATION_COUNT) -> np.ndarray:
    """
    SPIBB projection:
      - Only allow deviations from the baseline on "well-covered" bins (count >= min_deviation_count).
      - For all low-coverage bins, set π1 mass to 0 EXCEPT the baseline action, which always remains allowed.
      - Renormalize after dumping the removed probability mass onto the baseline.
    """
    pi1 = np.asarray(pi1, float).copy()
    assert pi1.ndim == 1, "pi1 must be a 1D action distribution"
    assert baseline_idx is not None and 0 <= baseline_idx < pi1.size

    # mask of bins where deviating away from baseline is allowed
    allowed = (counts >= min_deviation_count)

    # keep original mass to compute how much we zero out
    total_before = float(pi1.sum())

    # zero out all disallowed mass EXCEPT the baseline action
    disallowed = ~allowed
    disallowed[baseline_idx] = False
    removed_mass = float(pi1[disallowed].sum())
    pi1[disallowed] = 0.0

    # dump removed probability onto the baseline action
    pi1[baseline_idx] += removed_mass

    # guard: renormalize in case of numeric drift
    s = float(pi1.sum())
    if s <= 1e-12:
        # pathological case: give all mass to baseline
        pi1[:] = 0.0
        pi1[baseline_idx] = 1.0
    else:
        pi1 /= s

    # Optional: sanity—should still sum ~1
    # assert abs(pi1.sum() - 1.0) < 1e-6
    return pi1


def spibb_fallback_after_sampling(idx: int,
                                  counts: np.ndarray,
                                  baseline_idx: int,
                                  min_deviation_count: int = SPIBB_MIN_DEVIATION_COUNT) -> int:
    """
    Strict fallback: if the sampled action is in a low-coverage bin (< min_deviation_count),
    override with the baseline action.
    """
    if counts[int(idx)] < min_deviation_count:
        return int(baseline_idx)
    return int(idx)

# -----------------------------
# UI
# -----------------------------
def main():
    # --- Language state & toggle ---
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    lang = st.session_state["lang"]

    st.set_page_config(page_title=t(lang, "APP_TITLE"), page_icon="🌾", layout="wide")

    # top bar: title + right-aligned toggle
    c_title, c_toggle = st.columns([6, 1])
    with c_title:
        st.title(t(lang, "APP_TITLE"))
        st.markdown(t(lang, "APP_TAGLINE"))
    with c_toggle:
        # When app is Spanish, show "Switch to English"; when English, show "Cambiar a español"
        toggle_label = t(lang, "TOGGLE_LABEL_EN") if lang == "es" else t(lang, "TOGGLE_LABEL_ES")
        if st.button(toggle_label, use_container_width=True):
            st.session_state["lang"] = "en" if lang == "es" else "es"
            st.rerun()
        # refresh local after possible rerun
        lang = st.session_state["lang"]

    # Sidebar
    with st.sidebar:
        st.divider()
        st.header(t(lang, "ABOUT"))
        st.markdown(t(lang, "ABOUT_BULLETS"))

    # Load models
    with st.spinner(t(lang, "LOADING_MODELS")):
        (ensemble, binner, feature_cols,
         support_mask, counts, shape_bins, pi0_model) = load_models()
        action_grid, shape, unravel = build_action_grid(binner)

        system_options = sorted([c.split("=", 1)[1] for c in feature_cols if c.startswith("System=")]) or ["Unknown"]
        tillage_options = sorted([c.split("=", 1)[1] for c in feature_cols if c.startswith("Tillage=")]) or ["Unknown"]

        def _default_idx(opts):
            return opts.index("Unknown") if "Unknown" in opts else 0

    # Field inputs
    st.subheader(t(lang, "FIELD_INFO"))
    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input(t(lang, "LATITUDE"), value=16.5, step=0.01, format="%.4f")
        lon = st.number_input(t(lang, "LONGITUDE"), value=-93.0, step=0.01, format="%.4f")
        elev = st.number_input(t(lang, "ELEVATION"), value=700, step=10)
        slope = st.number_input(t(lang, "SLOPE"), value=3.0, step=0.5)
    with c2:
        clay = st.number_input(t(lang, "CLAY"), value=30.0, step=1.0)
        cec = st.number_input(t(lang, "CEC"), value=22.0, step=0.5)
        som = st.number_input(t(lang, "SOM"), value=1.5, step=0.1)
        ph = st.number_input(t(lang, "PH"), value=6.8, step=0.1)
    with c3:
        season_year = st.number_input(t(lang, "SEASON_YEAR"), value=DEFAULT_YEAR, step=1)
        planting_date = st.date_input(
            t(lang, "PLANTING_DATE"),
            value=datetime(DEFAULT_YEAR, 5, 15),
            min_value=datetime(DEFAULT_YEAR, 1, 1),
            max_value=datetime(DEFAULT_YEAR, 12, 31),
        )
        day_of_year = planting_date.timetuple().tm_yday
        theta = 2.0 * np.pi * (day_of_year / 365.25)
        planting_sin, planting_cos = float(np.sin(theta)), float(np.cos(theta))

    # Management
    st.markdown(f"### {t(lang, 'MANAGEMENT')}")
    m1, m2 = st.columns(2)
    with m1:
        system_choice = st.selectbox(t(lang, "SYSTEM"), options=system_options, index=_default_idx(system_options))
    with m2:
        tillage_choice = st.selectbox(t(lang, "TILLAGE"), options=tillage_options, index=_default_idx(tillage_options))

    st.subheader(t(lang, "PREPLANT_WEATHER"))
    st.caption(t(lang, "PREPLANT_CAPTION"))
    cc1, cc2 = st.columns(2)
    with cc1:
        prcp, srad = [], []
        for i in range(1, 7):
            with st.expander(t(lang, "EXPANDER_PRCP_SRAD", n=i)):
                prcp.append(st.number_input(t(lang, "PRCP", n=i), value=4.0, min_value=0.0))
                srad.append(st.number_input(t(lang, "SRAD", n=i), value=18.0, min_value=0.0))
    with cc2:
        tmax, tmin, vp = [], [], []
        for i in range(1, 7):
            with st.expander(t(lang, "EXPANDER_T_VP", n=i)):
                tmax.append(st.number_input(t(lang, "TMAX", n=i), value=30.0))
                tmin.append(st.number_input(t(lang, "TMIN", n=i), value=18.0))
                vp.append(st.number_input(t(lang, "VP", n=i), value=1900.0, min_value=0.0))

    # Aggregate weather
    def _agg(vals, prefix):
        arr = np.array(vals, dtype=float)
        return {
            f"{prefix}_pre_sum": float(np.sum(arr)),
            f"{prefix}_pre_mean": float(np.mean(arr)),
            f"{prefix}_pre_std": float(np.std(arr)),
            f"{prefix}_pre_min": float(np.min(arr)),
            f"{prefix}_pre_max": float(np.max(arr)),
        }

    weather_aggs = {}
    weather_aggs.update(_agg(prcp, "prcp"))
    weather_aggs["dry_decads_pre"] = float(np.sum(np.array(prcp, dtype=float) < 1.0))
    weather_aggs.update(_agg(srad, "srad"))
    weather_aggs.update(_agg(tmax, "tmax"))
    weather_aggs.update(_agg(tmin, "tmin"))
    weather_aggs.update(_agg(vp, "vp"))

    # Prices
    st.subheader(t(lang, "PRICES"))
    p1, p2 = st.columns(2)
    with p1:
        currency = st.text_input(t(lang, "CURRENCY"), value="MXN", help=t(lang, "CURRENCY_HELP"))
    with p2:
        maize_price = st.number_input(
            t(lang, "MAIZE_PRICE", currency=currency),
            value=3500.0,
            min_value=0.0,
            help=t(lang, "MAIZE_PRICE_HELP"),
        )

    p3, p4, p5 = st.columns(3)
    with p3:
        pN = st.number_input(t(lang, "N_PRICE", currency=currency), value=16.0, min_value=0.0)
    with p4:
        pP = st.number_input(t(lang, "P_PRICE", currency=currency), value=12.0, min_value=0.0)
    with p5:
        pK = st.number_input(t(lang, "K_PRICE", currency=currency), value=8.0, min_value=0.0)

    prices_entered = all(v > 0 for v in [maize_price, pN, pP, pK])
    objective_label = "Profit" if prices_entered else "Yield"  # Keep internal label in English

    # Get recommendation
    if st.button(t(lang, "GET_RECO"), type="primary"):
        with st.spinner(t(lang, "CALC_OPT")):
            # Build features
            inputs = {
                "Lat": lat, "Long": lon, "Elev": elev, "Slope": slope,
                "Clay": clay, "CEC": cec, "SOM": som, "PH": ph,
                "Year": season_year, "planting_sin": planting_sin, "planting_cos": planting_cos,
                **weather_aggs,
            }

            for _s in system_options:
                inputs[f"System={_s}"] = 1.0 if _s == system_choice else 0.0
            for _t in tillage_options:
                inputs[f"Tillage={_t}"] = 1.0 if _t == tillage_choice else 0.0

            X = create_feature_vector(inputs, feature_cols)

            # Predict yields
            mu = predict_yield(ensemble, X, action_grid)

            # Calculate profit if prices entered
            profit = None
            if prices_entered:
                profit = profit_from(mu, action_grid, maize_price, pN, pP, pK)

            # Build the objective vector
            objective = (profit if prices_entered else mu)  # shape (A,)
            A = action_grid.shape[0]

            # π0 masked to TRAIN support
            pi0_m = pi0_masked_grid(pi0_model, X, action_grid.shape[0], support_mask)  # (1, A)

            # π1 distribution (ε-greedy on support, λ-mix with π0)
            pi1_raw = build_pi1_distribution(
                objective=objective,
                support_mask=support_mask,
                pi0_masked=pi0_m,
                epsilon=epsilon,
                lambda_mix=LAMBDA_MIX,
            )

            # --- SPIBB: compute baseline BEFORE projection ---
            baseline_idx = get_baseline_action(pi0_model, X, support_mask, counts)

            # --- SPIBB projection at the distribution level ---
            pi1 = apply_spibb_projection(
                pi1=pi1_raw,
                counts=counts,
                baseline_idx=baseline_idx,
                min_deviation_count=SPIBB_MIN_DEVIATION_COUNT
            )

            # Deterministic seed
            seed = deterministic_seed_from_inputs(lat, lon, season_year)

            # Sample (already SPIBB-safe), plus a strict fallback just in case
            idx = sample_action_from_pi1(pi1, seed)
            idx = spibb_fallback_after_sampling(idx, counts, baseline_idx, SPIBB_MIN_DEVIATION_COUNT)

            spibb_active = (counts[idx] < SPIBB_MIN_DEVIATION_COUNT)
            if spibb_active:
                st.info(
                    f"SPIBB constraint active: bin count={int(counts[idx])} < {SPIBB_MIN_DEVIATION_COUNT}. "
                    f"Recommendation deferred to baseline."
                )

            is_supported = bool(support_mask[idx])

            N, P, K = action_from_idx(action_grid, idx)
            y_exp = float(mu[idx])
            profit_val = float(profit[idx]) if profit is not None else None

            st.session_state["last_reco"] = {
                "N": float(N),
                "P2O5": float(P),
                "K2O": float(K),
                }

            # Get baseline action (modal supported action from π0)
            baseline_idx = get_baseline_action(pi0_model, X, support_mask, counts)
            N_base, P_base, K_base = action_from_idx(action_grid, baseline_idx)
            y_base = float(mu[baseline_idx])
            profit_base = float(profit[baseline_idx]) if profit is not None else None

            # Calculate deltas
            delta_yield = y_exp - y_base
            delta_profit = (profit_val - profit_base) if profit is not None else None

            # Check guardrails
            flags = within_guardrails(N, P, K)

            # Display main recommendation
            st.success(t(lang, "RECO_READY"))

            # Main metrics with baseline comparison
            st.markdown(f"### {t(lang, 'RECO_RATES')}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t(lang, "N_LABEL"), f"{N:.1f} kg/ha", f"{N-N_base:+.1f} vs baseline")
            col2.metric(t(lang, "P_LABEL"), f"{P:.1f} kg/ha", f"{P-P_base:+.1f} vs baseline")
            col3.metric(t(lang, "K_LABEL"), f"{K:.1f} kg/ha", f"{K-K_base:+.1f} vs baseline")
            col4.metric(t(lang, "EXPECTED_YIELD"), f"{y_exp:,.0f} kg/ha", f"{delta_yield:+.0f} vs baseline")

            # % captions under the four metrics
            pct_n = percent_change(N, N_base)
            pct_p = percent_change(P, P_base)
            pct_k = percent_change(K, K_base)
            pct_y = percent_change(y_exp, y_base)

            show_percent_caption(col1, pct_n)  # N vs baseline
            show_percent_caption(col2, pct_p)  # P2O5 vs baseline
            show_percent_caption(col3, pct_k)  # K2O vs baseline
            show_percent_caption(col4, pct_y)  # Yield vs baseline

            # Predicted improvements box
            st.markdown(f"### {t(lang, 'PRED_IMPROVEMENTS')}")
            improvement_cols = st.columns(2 if prices_entered else 1)

            with improvement_cols[0]:
                st.info(
                    t(
                        lang,
                        "PRED_YIELD_GAIN",
                        delta_yield=delta_yield, N=N, P=P, K=K,
                        N_base=N_base, P_base=P_base, K_base=K_base
                    )
                )

            if prices_entered and len(improvement_cols) > 1:
                with improvement_cols[1]:
                    st.info(
                        t(
                            lang,
                            "PRED_PROFIT_GAIN",
                            delta_profit=delta_profit, currency=currency
                        )
                    )

            # Confidence indicator
            if is_supported:
                st.success(t(lang, "CONF_HIGH"))
            else:
                st.warning(t(lang, "CONF_LOW"))

            if flags:
                st.error(t(lang, "SUSTAINABILITY_CONCERN") + ", ".join(flags))

            # Economic analysis (if prices entered)
            if prices_entered:
                st.subheader(t(lang, "ECON_ANALYSIS"))

                fert_cost = N * pN + P * pP + K * pK
                fert_cost_base = N_base * pN + P_base * pP + K_base * pK
                revenue = (y_exp / 1000.0) * maize_price
                revenue_base = (y_base / 1000.0) * maize_price
                roi = calculate_roi(profit_val, fert_cost)

                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric(t(lang, "GROSS_REVENUE"), f"{revenue:,.0f} {currency}/ha",
                           f"{revenue-revenue_base:+,.0f} vs baseline")
                ec2.metric(t(lang, "FERT_COST"), f"{fert_cost:,.0f} {currency}/ha",
                           f"{fert_cost-fert_cost_base:+,.0f} vs baseline")
                ec3.metric(t(lang, "NET_PROFIT"), f"{profit_val:,.0f} {currency}/ha",
                           f"{delta_profit:+,.0f} vs baseline")
                ec4.metric(t(lang, "ROI"), f"{roi:.0f}%" if not np.isinf(roi) else "N/A")

                # % captions under economic metrics
                rev_pct    = percent_change(revenue,     revenue_base)
                cost_pct   = percent_change(fert_cost,   fert_cost_base)
                profit_pct = percent_change(profit_val,  profit_base)

                show_percent_caption(ec1, rev_pct)                 # Gross Revenue
                show_percent_caption(ec2, cost_pct, invert=True)   # Fertilizer Cost (lower = gain)
                show_percent_caption(ec3, profit_pct)              # Net Profit

                # Practical interpretation
                st.info(
                    t(
                        lang, "WHAT_IT_MEANS",
                        delta_profit=delta_profit, currency=currency,
                        roi_per_1=roi/100 if not np.isinf(roi) else 0.0
                    )
                )

            # Baseline comparison details
            with st.expander(t(lang, "BASELINE_DETAILS")):
                baseline_df = pd.DataFrame({
                    t(lang, "METRIC"): ["N (kg/ha)", "P₂O₅ (kg/ha)", "K₂O (kg/ha)",
                                        "Yield (kg/ha)", f"{t(lang,'NET_PROFIT')} ({currency}/ha)" if prices_entered else ""],
                    t(lang, "BASELINE"): [f"{N_base:.1f}", f"{P_base:.1f}", f"{K_base:.1f}",
                                          f"{y_base:,.0f}", f"{profit_base:,.0f}" if prices_entered else ""],
                    t(lang, "RECOMMENDED"): [f"{N:.1f}", f"{P:.1f}", f"{K:.1f}",
                                             f"{y_exp:,.0f}", f"{profit_val:,.0f}" if prices_entered else ""],
                    t(lang, "DIFFERENCE"): [f"{N-N_base:+.1f}", f"{P-P_base:+.1f}", f"{K-K_base:+.1f}",
                                            f"{delta_yield:+,.0f}", f"{delta_profit:+,.0f}" if prices_entered else ""]
                })
                if not prices_entered:
                    baseline_df = baseline_df[baseline_df[t(lang, "METRIC")] != ""]
                st.dataframe(
                    baseline_df,
                    hide_index=True,
                    use_container_width=False,
                    width=900,
                )

                st.caption(t(lang, "BASELINE_NOTE"))

            # Alternative recommendations
            st.subheader(t(lang, "ALTERNATIVES"))

            obj_grid = profit if prices_entered else mu
            supported_idx = np.where(support_mask)[0]

            if len(supported_idx) > 0:
                scores = obj_grid.copy()
                mask = np.full_like(scores, -np.inf, dtype=float)
                mask[supported_idx] = scores[supported_idx]
                topk = np.argsort(mask)[-3:][::-1]
            else:
                topk = np.argsort(obj_grid)[-3:][::-1]

            label_map = [t(lang, "BEST"), t(lang, "BETTER"), t(lang, "GOOD")]
            alternatives = []
            for rank, j in enumerate(topk, start=1):
                Nj, Pj, Kj = action_from_idx(action_grid, int(j))
                row = {
                    "Option": label_map[rank-1] if len(topk) == 3 else f"Option {rank}",
                    "N": f"{Nj:.0f}",
                    "P₂O₅": f"{Pj:.0f}",
                    "K₂O": f"{Kj:.0f}",
                    "Yield": f"{mu[j]:,.0f}",
                    t(lang, "DELTA_YIELD_VS_BASE"): f"{mu[j]-y_base:+,.0f}",
                    t(lang, "CONFIDENCE"): t(lang, "CONF_HIGH_WORD") if support_mask[j] else t(lang, "CONF_LOW_WORD"),
                }
                if prices_entered:
                    row[f"{t(lang,'NET_PROFIT')} ({currency})"] = f"{profit[j]:,.0f}"
                    row[t(lang, "DELTA_PROFIT_VS_BASE")] = f"{profit[j]-profit_base:+,.0f}"
                alternatives.append(row)

            alt_df = pd.DataFrame(alternatives)

            st.dataframe(
                alt_df,
                hide_index=True,              
                use_container_width=False,    
                width=1200,                  
            )

            # Technical details
            with st.expander(t(lang, "TECH_DETAILS")):
                iN, iP, iK = unravel(idx)
                # st.write(t(lang, "BIN_INDICES", iN=iN, iP=iP, iK=iK))
                st.write(t(lang, "OBJECTIVE", objective_label=objective_label))
                st.write(t(lang, "EXPLORATION", epsilon=epsilon))
                st.write(t(lang, "SUPPORT_THR", min_support=MIN_SUPPORT_COUNT))
                st.write(t(lang, "BASELINE_IDX", baseline_idx=baseline_idx))
                st.write(t(lang, "POLICY", epsilon=epsilon, lambda_mix=LAMBDA_MIX))

                if prices_entered:
                    st.write(f"\n{t(lang, 'PRICE_ASSUMPTIONS')}")
                    st.write(f"- Maize: {maize_price} {currency}/tonne")
                    st.write(f"- N: {pN} {currency}/kg")
                    st.write(f"- P₂O₅: {pP} {currency}/kg")
                    st.write(f"- K₂O: {pK} {currency}/kg")

    # =============================
    # Consult Technical Documentation (RAG)
    # =============================
    st.divider()

    def _get_openai_client():
        return OpenAI()

    from pathlib import Path
    import json
    import re

    REPO_ROOT = Path(__file__).resolve().parents[1]
    pdf_path = REPO_ROOT / "docs" / "agenda-tecnica-chiapas.pdf"
    cache_dir = Path(__file__).parent / ".rag_cache"

    top_k = 6
    force_rebuild = False

    def _strip_simple_md(s: str) -> str:
        # CSS content can't render markdown; remove emphasis markers.
        return re.sub(r"(\*\*|\*|__|_|`)", "", s or "")

    if not pdf_path.exists():
        st.error(t(lang, "DOCS_MISSING_PDF", pdf_path=pdf_path))
    else:
        client = _get_openai_client()

        with st.spinner(t(lang, "DOCS_LOADING_INDEX")):
            store = get_or_build_store(
                pdf_path=pdf_path,
                start_page_1idx=65,
                end_page_1idx=102,
                client=client,
                cache_dir=cache_dir,
                embed_model=DEFAULT_EMBED_MODEL,
                chunk_chars=1200,
                overlap=200,
                force_rebuild=force_rebuild,
            )

        if "doc_chat" not in st.session_state:
            st.session_state["doc_chat"] = []

        # Render chat history
        for m in st.session_state["doc_chat"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # ---- Pinned header + caption ABOVE the pinned chat input ----
        docs_title = _strip_simple_md(t(lang, "DOCS_HEADER"))
        docs_caption = _strip_simple_md(t(lang, "DOCS_CAPTION"))

        # keep emoji as real unicode (prevents \ud83d\udcda showing up)
        docs_title_css = json.dumps(docs_title, ensure_ascii=False)[1:-1]
        docs_caption_css = json.dumps(docs_caption, ensure_ascii=False)[1:-1]

        st.markdown(
            f"""
    <style>
    /* Give the page enough bottom padding so last messages aren't hidden */
    div[data-testid="stAppViewContainer"] .main .block-container {{
    padding-bottom: 12rem;
    }}

    /*
    Key fix:
    ::before/::after become flex items on a flex container, which is why your input got pushed to the right.
    Force the chat input container into a column layout so everything stacks vertically.
    */
    div[data-testid="stChatInput"] {{
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.35rem;
    }}

    /* Title line */
    div[data-testid="stChatInput"]::before {{
    content: "{docs_title_css}";
    order: 0;
    display: block;
    padding: 0.55rem 0.75rem 0 0.75rem;
    background: var(--background-color);
    font-size: 1.0rem;
    line-height: 1.25rem;
    font-weight: 700;
    }}

    /* Caption line (smaller, not bold) */
    div[data-testid="stChatInput"]::after {{
    content: "{docs_caption_css}";
    order: 1;
    display: block;
    padding: 0 0.75rem 0.35rem 0.75rem;
    margin-bottom: 0.1rem;
    border-bottom: 1px solid rgba(49, 51, 63, 0.20);
    background: var(--background-color);
    font-size: 0.85rem;
    line-height: 1.15rem;
    font-weight: 400;
    opacity: 0.85;
    }}

    /* Make sure the actual input form is full-width and comes after title/caption */
    div[data-testid="stChatInput"] form {{
    order: 2;
    width: 100%;
    }}

    /* Tame the placeholder so it doesn't look like a second header */
    div[data-testid="stChatInput"] textarea::placeholder {{
    font-size: 0.85rem;
    font-weight: 400;
    opacity: 0.75;
    }}

    /* Optional: reduce chat input minimum height a touch */
    div[data-testid="stChatInput"] textarea {{
    min-height: 2.6rem;
    }}
    </style>
    """,
            unsafe_allow_html=True,
        )

        # Keep Streamlit's pinned chat behavior
        user_q = st.chat_input(t(lang, "DOCS_CHAT_INPUT"), key="docs_chat_input")
        if user_q:
            st.session_state["doc_chat"].append({"role": "user", "content": user_q})

            with st.spinner(t(lang, "DOCS_SEARCHING_ANSWER")):
                hits = retrieve(store=store, client=client, query=user_q, top_k=top_k)

                reco_ctx = st.session_state.get("last_reco")  # optional
                answer = answer_with_rag_chat_completions(
                    client=client,
                    chat_model=DEFAULT_CHAT_MODEL,
                    question=user_q,
                    retrieved=hits,
                    lang=lang,
                    reco_context=reco_ctx,
                )

            st.session_state["doc_chat"].append({"role": "assistant", "content": answer})
            st.rerun()

        # Show sources used for the most recent turn
        if st.session_state["doc_chat"]:
            with st.expander(t(lang, "DOCS_SHOW_PASSAGES")):
                last_user = next(
                    (m["content"] for m in reversed(st.session_state["doc_chat"]) if m["role"] == "user"),
                    None,
                )
                if last_user:
                    hits = retrieve(store=store, client=client, query=last_user, top_k=top_k)
                    for h in hits:
                        st.markdown(t(lang, "DOCS_PASSAGE_META", page=h["page"], score=h["score"]))
                        st.write(h["text"])
                        st.markdown("---")
            
if __name__ == "__main__":
    main()