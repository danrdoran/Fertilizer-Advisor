"""
What this script does: 

Defines the core modeling components shared across this repo: 
adaptive binning of continuous N–P–K actions, feature engineering 
(robust scaling and mutual-information feature selection), a stacked ensemble
reward model, and calibrated behavior policy (π₀) propensity models, 
plus supporting utilities like forward-by-year folds, recency weighting, 
bootstrap CIs, and reliability/ECE diagnostics.
"""

import warnings, os, json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, log_loss
from sklearn.model_selection import KFold, GroupKFold 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

SEED = 42
KEEP_VAR_EPS = 1e-12

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def set_random_seeds(seed: int = SEED):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def rmse_fn(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def forward_year_folds(years_arr: np.ndarray, start_val_year: int, end_val_year: int):
    folds = []
    for y in range(start_val_year, end_val_year + 1):
        tr_mask = (years_arr <= (y - 1))
        va_mask = (years_arr == y)
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            folds.append((tr_idx, va_idx, y))
    return folds

def make_recency_weights(train_years: np.ndarray, lambda_: float) -> np.ndarray:
    y_max = np.max(train_years)
    age = (y_max - train_years).astype(float)
    return np.exp(-lambda_ * age)

def bootstrap_ci(idx_n: int, stat_fn, B: int = 1000, alpha: float = 0.05, rng: Optional[np.random.Generator]=None) -> Tuple[float,float]:
    rng = rng or np.random.default_rng(SEED)
    stats = np.empty(B, dtype=float)
    for b in range(B):
        sample_idx = rng.integers(0, idx_n, size=idx_n)
        stats[b] = float(stat_fn(sample_idx))
    lo = float(np.quantile(stats, alpha/2))
    hi = float(np.quantile(stats, 1 - alpha/2))
    return lo, hi

# ---------------------------------------------------------------------
# Adaptive Binning (reward-aware splits; per-dimension control)
# ---------------------------------------------------------------------
class AdaptiveBinner:
    def __init__(self,
                 min_samples_per_bin: int = 80,
                 max_bins_per_dim: Optional[Dict[str, int]] = None):
        self.min_samples = min_samples_per_bin
        self.max_bins_per_dim = max_bins_per_dim or {'N': 7, 'P': 3, 'K': 2}
        self.bin_edges: Dict[str, np.ndarray] = {}

    def fit(self, actions: np.ndarray, rewards: np.ndarray, action_names: List[str] = ['N','P','K']):
        for i, name in enumerate(action_names):
            max_leaf_nodes = int(self.max_bins_per_dim.get(name, 5))
            tree = DecisionTreeRegressor(
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=self.min_samples,
                random_state=SEED
            )
            X = actions[:, i].reshape(-1, 1)
            tree.fit(X, rewards)

            thresholds: List[float] = []
            def extract(node=0):
                if tree.tree_.feature[node] != -2:
                    thresholds.append(tree.tree_.threshold[node])
                    extract(tree.tree_.children_left[node])
                    extract(tree.tree_.children_right[node])
            extract()

            thresholds = sorted(set(thresholds))
            vmin, vmax = actions[:, i].min(), actions[:, i].max()
            edges = [vmin] + thresholds + [vmax + 1e-6]
            edges = sorted(set(edges))

            n_target = max_leaf_nodes
            if len(edges) - 1 != n_target:
                qs = np.linspace(0, 1, n_target + 1)
                edges = np.unique(np.quantile(actions[:, i], qs))
                if len(edges) - 1 < n_target:
                    edges = np.linspace(vmin, vmax + 1e-6, n_target + 1)

            self.bin_edges[name] = np.array(edges, dtype=float)
        return self

    def transform(self, actions: np.ndarray, action_names: List[str] = ['N','P','K']) -> np.ndarray:
        binned = np.zeros_like(actions, dtype=int)
        for i, name in enumerate(action_names):
            edges = self.bin_edges[name]
            binned[:, i] = np.digitize(actions[:, i], edges) - 1
            binned[:, i] = np.clip(binned[:, i], 0, len(edges) - 2)
        return binned
    
    @property
    def bin_centers(self) -> Dict[str, np.ndarray]:
        """
        Centers of each bin for N, P, K computed from self.bin_edges.
        Returned as {'N': array, 'P': array, 'K': array}.
        """
        if not hasattr(self, "bin_edges") or self.bin_edges is None:
            raise AttributeError("bin_edges is not set; call fit(...) before using bin_centers.")
        return {k: 0.5 * (edges[:-1] + edges[1:]) for k, edges in self.bin_edges.items()}

# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------
class FeatureEngineer:
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.selected_features = None
        self.scaler = None
        self.selected_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> 'FeatureEngineer':
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        mi_scores = mutual_info_regression(X_scaled, y, random_state=SEED)
        top_idx = np.argsort(mi_scores)[-self.max_features:]
        self.selected_features = top_idx
        self.selected_names = [feature_names[i] for i in top_idx]
        print(f"Selected {len(self.selected_features)} features based on MI scores")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_features]

# ---------------------------------------------------------------------
# Stacked Ensemble Reward Model (with sanitization + σ̂)
# ---------------------------------------------------------------------
class StackedRewardModel:
    def __init__(self, feature_engineer: Optional[FeatureEngineer] = None):
        self.feature_engineer = feature_engineer or FeatureEngineer(max_features=50)
        self.models: Dict[str, Any] = {}
        self.meta_model = None
        self.cv_models: Dict[str, list] = defaultdict(list)
        self.keep_mask = None

    def _create_xgb(self):
        return xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=5, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=10.0,
            n_jobs=-1, random_state=SEED, tree_method="hist"
        )

    def _create_lgb(self):
        return lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05,
            num_leaves=64, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=10.0, random_state=SEED,
            feature_pre_filter=False, force_col_wise=True,
            min_split_gain=0.0, verbose=-1
        )

    def _create_cat(self):
        return cb.CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=8.0,
            loss_function="RMSE", verbose=False, random_seed=SEED,
            allow_writing_files=False
        )

    def _sanitize_and_set_mask(self, X: np.ndarray, tag: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not np.isfinite(X).all():
            nan_ct = int(np.isnan(X).sum()); inf_ct = int(np.isinf(X).sum())
            print(f"[Sanitize] {tag}: replacing {nan_ct} NaNs, {inf_ct} infs with 0")
            np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)
        var = X.var(axis=0)
        keep_mask = var > KEEP_VAR_EPS
        dropped = int((~keep_mask).sum())
        if dropped:
            print(f"[Sanitize] {tag}: dropping {dropped} constant cols")
        if keep_mask.sum() == 0:
            raise ValueError(f"[Sanitize] {tag}: all columns became constant/invalid")
        self.keep_mask = keep_mask
        return X[:, keep_mask]

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        if self.keep_mask is None:
            return X
        return X[:, self.keep_mask]

    def _full_X(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if X is None:
            return None
        Xs = self.feature_engineer.transform(X)
        return np.hstack([Xs, actions])

    def fit(self, X, actions, y, X_val=None, actions_val=None, y_val=None,
            sample_weight=None, n_trials: int = 0,
            feature_names: Optional[List[str]] = None,
            groups: Optional[np.ndarray] = None  # <-- ADD
            ):
        
        if np.nanvar(y) < 1e-8:
            raise ValueError("Target variance ~ 0 after filtering; cannot train a meaningful model.")

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_engineer.fit(X, y, feature_names)

        X_full = self._full_X(X, actions)
        X_full = self._sanitize_and_set_mask(X_full, tag="train")

        X_val_full = None
        if X_val is not None:
            X_val_full = self._apply_mask(self._full_X(X_val, actions_val))

        self.models = {"xgb": self._create_xgb(), "lgb": self._create_lgb(), "cat": self._create_cat()}
        for name, model in self.models.items():
            try:
                if name == "xgb" and X_val_full is not None:
                    model.fit(X_full, y, sample_weight=sample_weight, eval_set=[(X_val_full, y_val)], verbose=False)
                else:
                    model.fit(X_full, y, sample_weight=sample_weight, verbose=False)
            except TypeError:
                model.fit(X_full, y, sample_weight=sample_weight)

        self.cv_models = defaultdict(list)
        oof_preds = np.zeros((len(X_full), len(self.models)))

        if groups is not None:
            splitter = GroupKFold(n_splits=5)
            split_iter = splitter.split(X_full, y, groups=groups)
            print("[StackedRewardModel] OOF CV = GroupKFold (5 folds) using provided geo groups")
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
            split_iter = splitter.split(X_full)
            print("[StackedRewardModel] OOF CV = KFold (5 folds, shuffled)")

        for fold, (tr, vl) in enumerate(split_iter):
            X_tr, X_vl = X_full[tr], X_full[vl]
            y_tr = y[tr]
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            for i, (name, factory) in enumerate([
                ("xgb", self._create_xgb), ("lgb", self._create_lgb), ("cat", self._create_cat)
            ]):
                m = factory()
                try:
                    m.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)
                except TypeError:
                    m.fit(X_tr, y_tr, sample_weight=sw_tr)
                oof_preds[vl, i] = m.predict(X_vl)
                self.cv_models[name].append(m)

        self.meta_model = RidgeCV(cv=5)
        self.meta_model.fit(oof_preds, y)

        oof_pred = self.meta_model.predict(oof_preds)
        print(f"Stacked ensemble OOF R²: {r2_score(y, oof_pred):.3f}")
        print(f"Stacked ensemble OOF MAE: {mean_absolute_error(y, oof_pred):.1f}")
        return self

    def predict(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        X_full = self._apply_mask(self._full_X(X, actions))
        base_preds = np.column_stack([m.predict(X_full) for m in self.models.values()])
        return self.meta_model.predict(base_preds)

    def predict_with_uncertainty(self, X: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_full = self._apply_mask(self._full_X(X, actions))
        base_final = np.column_stack([m.predict(X_full) for m in self.models.values()])
        mu = self.meta_model.predict(base_final)
        per = []
        for m in self.models.values():
            per.append(m.predict(X_full))
        for fold_models in self.cv_models.values():
            for m in fold_models:
                per.append(m.predict(X_full))
        per = np.vstack(per)
        sigma = per.std(axis=0)
        return mu, sigma

# ---------------------------------------------------------------------
# Factorized, Calibrated Propensity Model p(a|x) = Π_i p(a_i|x)
# ---------------------------------------------------------------------
class FactorizedPropensityModel:
    def __init__(self, base_params: dict = None, epsilon: float = 1e-12):
        self.base_params = base_params or {}
        self.scaler = RobustScaler()
        self.models: Dict[str, Any] = {}
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.order = ['N','P','K']
        self.classes_per_dim: Dict[str, np.ndarray] = {}
        self.epsilon = epsilon

    def _encode_dim(self, a: np.ndarray, edges: np.ndarray) -> np.ndarray:
        idx = np.digitize(a, edges) - 1
        return np.clip(idx, 0, len(edges) - 2)

    def fit(self, X: np.ndarray, actions: np.ndarray, bin_edges: Dict[str, np.ndarray], sample_weight: Optional[np.ndarray]=None):
        X_scaled = self.scaler.fit_transform(X)
        self.bin_edges = bin_edges

        for i, name in enumerate(self.order):
            y_dim = self._encode_dim(actions[:, i], bin_edges[name]).astype(int)
            n_classes = int(len(bin_edges[name]) - 1)

            class_counts = np.bincount(y_dim, minlength=n_classes)
            obs_classes = np.where(class_counts > 0)[0]
            min_count = int(class_counts[obs_classes].min()) if obs_classes.size else 0
            if min_count >= 3:
                cal_cv = 3
            elif min_count >= 2:
                cal_cv = 2
            else:
                cal_cv = 0

            params = dict(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, tree_method="hist",
                n_estimators=self.base_params.get('n_estimators', 500),
                learning_rate=self.base_params.get('learning_rate', 0.05),
                max_depth=self.base_params.get('max_depth', 4),
                subsample=self.base_params.get('subsample', 0.8),
                colsample_bytree=self.base_params.get('colsample_bytree', 0.8),
            )
            base = xgb.XGBClassifier(**params)

            if cal_cv >= 2:
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=cal_cv)
                try:
                    clf.fit(X_scaled, y_dim, sample_weight=sample_weight)
                except ValueError:
                    base.fit(X_scaled, y_dim, sample_weight=sample_weight)
                    clf = base
                probs = clf.predict_proba(X_scaled)
                ll = log_loss(y_dim, probs, labels=np.arange(n_classes))
                print(f"[Prop-{name}] calibrated (cv={cal_cv}) mlogloss={ll:.3f}")
            else:
                base.fit(X_scaled, y_dim, sample_weight=sample_weight)
                probs = base.predict_proba(X_scaled)
                ll = log_loss(y_dim, probs, labels=np.arange(n_classes))
                print(f"[Prop-{name}] uncalibrated mlogloss={ll:.3f}")
                clf = base

            self.models[name] = clf
            self.classes_per_dim[name] = np.arange(n_classes, dtype=int)

        return self

    def predict_head(self, name: str, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        clf = self.models[name]
        return clf.predict_proba(Xs)

    def predict_proba(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        probs = []
        for i, name in enumerate(self.order):
            edges = self.bin_edges[name]
            y_dim = self._encode_dim(actions[:, i], edges)
            head_probs = self.predict_head(name, X)
            y_dim = np.clip(y_dim, 0, head_probs.shape[1]-1)
            probs.append(head_probs[np.arange(len(X)), y_dim])
        p = np.ones(len(X), dtype=float)
        for v in probs:
            p *= np.clip(v, self.epsilon, 1.0)
        return p

# ---------------------------------------------------------------------
# Joint Propensity Model p(a_joint|x) for consistent evaluation
# ---------------------------------------------------------------------
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import xgboost as xgb


class JointPropensityModel:
    """
    Joint propensity model π₀(a|x) over discretized (N, P, K) action bins.

    Key features:
    - Supports missing/unobserved joint action cells by mapping observed labels
      to a compact 0..C_obs-1 index space during training.
    - Expands predicted probabilities back to the full (N×P×K) grid at inference.
    - Optional probability calibration (sigmoid/iso) via CalibratedClassifierCV.
    """

    def __init__(
        self,
        base_params: Optional[Dict] = None,
        epsilon: float = 1e-4,
        calibration_cv: int = 0,
        calibration_method: str = "sigmoid",
        random_state: int = 42,
    ):
        # Configuration
        self.epsilon = float(epsilon)
        self.calibration_cv = int(calibration_cv)  # >=2 enables calibration
        self.calibration_method = str(calibration_method)
        self.random_state = int(random_state)

        # Learner hyperparameters (good starting defaults; override via base_params)
        default_params = dict(
            objective="multi:softprob",
            eval_metric="mlogloss",
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=400,
            tree_method="hist",  # "auto" or "gpu_hist" if you like
            random_state=self.random_state,
            n_jobs=0,
        )
        if base_params:
            default_params.update(base_params)
        self.base_params = default_params

        # Fitted artifacts
        self.scaler = StandardScaler()
        self.model = None  # XGBClassifier or CalibratedClassifierCV
        self.bin_edges: Optional[Dict[str, np.ndarray]] = None
        self.n_bins: Optional[Tuple[int, int, int]] = None
        self.n_actions: Optional[int] = None

        # Class mapping for sparse label space
        self._observed_classes: Optional[np.ndarray] = None  # shape: (C_obs,)
        self._orig_to_compact: Optional[Dict[int, int]] = None

        # Diagnostics
        self.class_counts_: Optional[np.ndarray] = None
        self.min_count_: Optional[int] = None
        self.coverage_ratio_: Optional[float] = None

    # ----------------------------- Utilities -----------------------------

    def _bin_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Bin continuous (or raw) actions into (iN, iP, iK) indices using self.bin_edges.
        actions: shape (n_samples, 3) -> columns correspond to [N, P, K]
        Returns: int array shape (n_samples, 3) with values in [0..n_bins_d-1]
        """
        assert self.bin_edges is not None, "bin_edges must be set before binning actions."
        assert actions.shape[1] == 3, "actions must have 3 columns: N, P, K."
        N_edges, P_edges, K_edges = self.bin_edges["N"], self.bin_edges["P"], self.bin_edges["K"]

        # np.digitize returns bin index in 1..len(edges)-1; convert to 0-based and clip
        iN = np.clip(np.digitize(actions[:, 0], N_edges, right=False) - 1, 0, len(N_edges) - 2)
        iP = np.clip(np.digitize(actions[:, 1], P_edges, right=False) - 1, 0, len(P_edges) - 2)
        iK = np.clip(np.digitize(actions[:, 2], K_edges, right=False) - 1, 0, len(K_edges) - 2)

        return np.stack([iN, iP, iK], axis=1).astype(int)

    def _to_joint_index(self, binned_actions: np.ndarray) -> np.ndarray:
        """
        Map (iN, iP, iK) -> joint flat index in [0..(nN*nP*nK)-1].
        Uses C-order (N major, then P, then K): idx = iN*(nP*nK) + iP*(nK) + iK
        """
        nN, nP, nK = self.n_bins
        iN, iP, iK = binned_actions[:, 0], binned_actions[:, 1], binned_actions[:, 2]
        return (iN * (nP * nK) + iP * nK + iK).astype(int)

    # ------------------------------- API --------------------------------

    def fit(
            self,
            X: np.ndarray,
            actions: np.ndarray,
            bin_edges: Dict[str, np.ndarray],
            sample_weight: Optional[np.ndarray] = None,
        ):
        """
        Fit π₀(a|x) on observed (X, actions).
        - X: shape (n_samples, n_features)
        - actions: shape (n_samples, 3) for [N, P, K]
        - bin_edges: dict with keys {"N","P","K"} -> 1D arrays of bin boundaries
        """
        # -------------------- Prepare scaler and bin metadata --------------------
        X = np.asarray(X)
        actions = np.asarray(actions)

        # Persist binning metadata on the instance
        self.bin_edges = {
            "N": np.asarray(bin_edges["N"]),
            "P": np.asarray(bin_edges["P"]),
            "K": np.asarray(bin_edges["K"]),
        }
        nN = len(self.bin_edges["N"]) - 1
        nP = len(self.bin_edges["P"]) - 1
        nK = len(self.bin_edges["K"]) - 1
        self.n_bins = (nN, nP, nK)
        self.n_actions = int(nN * nP * nK)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # ------------------------- Build compact labels -------------------------
        # Bin each continuous action dimension -> (iN,iP,iK)
        binned = self._bin_actions(actions)   # shape (n_samples, 3), ints
        # Map to joint flat index over full grid
        y_joint_orig = self._to_joint_index(binned)  # shape (n_samples,)

        # Identify observed joint-action classes and build compact mapping
        obs = np.unique(y_joint_orig)
        self._observed_classes = np.sort(obs)  # shape (C_obs,)
        self._orig_to_compact = {c: i for i, c in enumerate(self._observed_classes)}
        y_compact = np.fromiter((self._orig_to_compact[c] for c in y_joint_orig), dtype=int)

        # ---------------------------- Diagnostics -------------------------------
        class_counts = np.bincount(y_compact, minlength=len(self._observed_classes))
        self.class_counts_ = class_counts
        self.min_count_ = int(class_counts[class_counts > 0].min()) if class_counts.size else 0
        self.coverage_ratio_ = float(len(self._observed_classes) / self.n_actions) if self.n_actions else 0.0
        print(f"[JointPropensity] {len(self._observed_classes)}/{self.n_actions} action cells observed")
        print(f"[JointPropensity] Min count per observed cell: {self.min_count_}")

        # Trivial edge case: only one observed class → deterministic model
        if len(self._observed_classes) <= 1:
            class _OneClassModel:
                def fit(self, X, y, sample_weight=None): return self
                def predict_proba(self, X): return np.ones((X.shape[0], 1), dtype=float)
            self.model = _OneClassModel()
            # Nothing to calibrate; return early
            return self

        # ------------------------ Configure base learner ------------------------
        params = dict(self.base_params)
        params["num_class"] = int(len(self._observed_classes))  # IMPORTANT: compact class count
        base_model = xgb.XGBClassifier(**params)

        # ---------------------- Adaptive calibration choice ---------------------
        # Decide effective CV folds based on rarest class and requested calibration_cv.
        #  - Use 3-fold if every observed class has >=3 samples
        #  - Else 2-fold if every observed class has >=2 samples
        #  - Else disable calibration
        eff_cv = 0
        if isinstance(self.calibration_cv, int) and self.calibration_cv >= 2:
            mc = int(self.min_count_ or 0)
            if mc >= 3:
                eff_cv = min(int(self.calibration_cv), 3)
            elif mc >= 2:
                eff_cv = min(int(self.calibration_cv), 2)
            else:
                eff_cv = 0  # too sparse to calibrate

        # --------------------- Fit (with safe fallbacks) ------------------------
        if eff_cv >= 2:
            try:
                # sklearn >= 1.1 signature
                calibrated = CalibratedClassifierCV(
                    estimator=base_model,
                    method=self.calibration_method,
                    cv=eff_cv,
                )
            except TypeError:
                # sklearn <= 1.0 signature
                calibrated = CalibratedClassifierCV(
                    base_estimator=base_model,
                    method=self.calibration_method,
                    cv=eff_cv,
                )

            try:
                calibrated.fit(X_scaled, y_compact, sample_weight=sample_weight)
                self.model = calibrated
            except TypeError:
                # Some sklearn versions don't plumb sample_weight to CalibratedClassifierCV
                calibrated.fit(X_scaled, y_compact)
                self.model = calibrated
            except ValueError as e:
                # e.g., "Requesting 3-fold CV but provided < 3 examples for a class."
                # Fall back to uncalibrated base model
                warnings.warn(f"Calibration disabled due to data sparsity: {e}")
                base_model.fit(X_scaled, y_compact, sample_weight=sample_weight)
                self.model = base_model
        else:
            # No calibration
            base_model.fit(X_scaled, y_compact, sample_weight=sample_weight)
            self.model = base_model

        # Quick sanity: compute training log loss in compact space
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            # labels must span [0..C_obs-1]
            _ = log_loss(y_compact, probs, labels=np.arange(len(self._observed_classes)))

        return self

    def predict_proba_grid(self, X: np.ndarray) -> np.ndarray:
        """
        Predict π₀(a|x) over the FULL (N×P×K) joint action grid.
        Returns: array shape (n_samples, n_actions) where columns align with
                 original (non-compact) joint indices in [0..n_actions-1].
        """
        assert self.model is not None, "Model is not fitted."
        assert self._observed_classes is not None, "Observed class mapping missing."
        assert self.n_actions is not None, "n_actions missing."

        X = np.asarray(X)
        X_scaled = self.scaler.transform(X)

        # Get probabilities over compact classes
        if hasattr(self.model, "predict_proba"):
            probs_compact = self.model.predict_proba(X_scaled)
        else:
            # Fallback: uniform over compact classes
            n = X_scaled.shape[0]
            probs_compact = np.ones((n, len(self._observed_classes))) / float(len(self._observed_classes))

        # Expand to full grid using the stored mapping
        n = probs_compact.shape[0]
        full = np.zeros((n, self.n_actions), dtype=float)
        full[:, self._observed_classes] = probs_compact

        # Leave unobserved cells at exactly 0 — do NOT add epsilon mass.
        # The compact probabilities already sum to 1 by construction.
        row_sums = full.sum(axis=1, keepdims=True)

        # Extremely defensive backstop: if any row sums to 0 (shouldn’t happen),
        # fall back to uniform *over observed classes only*.
        bad = (row_sums.squeeze() == 0)
        if np.any(bad):
            full[bad, :] = 0.0
            full[bad][:, self._observed_classes] = 1.0 / float(len(self._observed_classes))

        # (Row sums are 1 in normal operation; this is a no-op in practice.)
        full /= np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
        return full

    # Optional convenience wrapper
    def predict_log_proba_grid(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba_grid(X)
        return np.log(probs)
    
# ---------------------------------------------------------------------
# Target-policy kernel helpers
# ---------------------------------------------------------------------
def triangular(w: int, h: int) -> float:
    if h <= 0:
        return 1.0 if w == 0 else 0.0
    return max(0.0, 1.0 - (abs(w) / float(h)))

def kernel_mass_at_logged_bin(log_bin: np.ndarray,
                              opt_bin: np.ndarray,
                              n_bins: Tuple[int,int,int],
                              bandwidths: Tuple[int,int,int],
                              eps_uniform: float = 0.08,
                              beta_logged: float = 0.0) -> float:
    support_ranges = [
        (max(0, opt_bin[0]-bandwidths[0]), min(n_bins[0]-1, opt_bin[0]+bandwidths[0])),
        (max(0, opt_bin[1]-bandwidths[1]), min(n_bins[1]-1, opt_bin[1]+bandwidths[1])),
        (max(0, opt_bin[2]-bandwidths[2]), min(n_bins[2]-1, opt_bin[2]+bandwidths[2])),
    ]
    total_bins = n_bins[0] * n_bins[1] * n_bins[2]

    W = 0.0
    for nb in range(support_ranges[0][0], support_ranges[0][1]+1):
        for pb in range(support_ranges[1][0], support_ranges[1][1]+1):
            for kb in range(support_ranges[2][0], support_ranges[2][1]+1):
                wN = triangular(nb - opt_bin[0], bandwidths[0])
                wP = triangular(pb - opt_bin[1], bandwidths[1])
                wK = triangular(kb - opt_bin[2], bandwidths[2])
                W += (wN * wP * wK)

    if W > 0:
        wN = triangular(log_bin[0] - opt_bin[0], bandwidths[0])
        wP = triangular(log_bin[1] - opt_bin[1], bandwidths[1])
        wK = triangular(log_bin[2] - opt_bin[2], bandwidths[2])
        w_log = wN * wP * wK
        kernel_part = (1.0 - eps_uniform) * (w_log / W)
    else:
        kernel_part = 0.0

    uniform_part = eps_uniform / total_bins
    base = kernel_part + uniform_part
    return float(beta_logged + (1.0 - beta_logged) * base)

# ===================== Calibration / Reliability for π0 =====================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def multiclass_reliability_curve(probs: np.ndarray,
                                 y_true_joint: np.ndarray,
                                 n_bins: int = 20,
                                 sample_weight: np.ndarray | None = None):
    """
    Multiclass reliability for joint π0: stack one-vs-rest pairs for all classes.
    Inputs
      probs: (n_samples, C) predicted class probabilities (sum=1 per row)
      y_true_joint: (n_samples,) true joint class index in [0..C-1]
      n_bins: number of probability bins on [0,1]
      sample_weight: (n_samples,) optional row weights
    Returns
      df: DataFrame with columns [bin_mid, mean_pred, mean_true, count, weight_sum]
      ece: Expected Calibration Error (|mean_true - mean_pred| weighted by bin prob mass)
    """
    n, C = probs.shape
    if sample_weight is None:
        w_rows = np.ones(n, dtype=float)
    else:
        w_rows = np.asarray(sample_weight, float)
        if w_rows.shape[0] != n:
            raise ValueError("sample_weight must have length n_samples")

    # One-hot for true labels
    y_onehot = np.zeros_like(probs, dtype=float)
    y_onehot[np.arange(n), y_true_joint] = 1.0

    # Flatten to one-vs-rest pairs across all classes
    p = probs.ravel()                             # (n*C,)
    t = y_onehot.ravel()                          # (n*C,)
    w = np.repeat(w_rows, C).astype(float)        # (n*C,)

    # Bin edges and assignments
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    # Per-bin weighted means
    out = []
    ece_num = 0.0
    total_w = float(w.sum())
    for b in range(n_bins):
        sel = (bin_idx == b)
        if not np.any(sel):
            continue
        wb = w[sel]; pb = p[sel]; tb = t[sel]
        wsum = float(wb.sum())
        mean_pred = float(np.average(pb, weights=wb))
        mean_true = float(np.average(tb, weights=wb))
        count = int(sel.sum())
        mid = 0.5 * (edges[b] + edges[b+1])
        out.append((mid, mean_pred, mean_true, count, wsum))
        ece_num += (abs(mean_true - mean_pred) * (wsum / max(total_w, 1e-12)))

    df = pd.DataFrame(out, columns=["bin_mid", "mean_pred", "mean_true", "count", "weight_sum"])
    ece = float(ece_num)
    return df, ece

def plot_reliability_curve(df: pd.DataFrame,
                           title: str = "π₀ reliability (multiclass, one-vs-rest)",
                           path: str | None = None):
    """
    Plot mean_true vs. mean_pred with the y=x perfect calibration line.
    """
    plt.figure(figsize=(6, 6))
    # scatter with size by count (optional)
    sizes = 20 + 80 * (df["weight_sum"] / df["weight_sum"].max())
    plt.scatter(df["mean_pred"], df["mean_true"], s=sizes)
    # perfect calibration
    lims = [0, 1]
    plt.plot(lims, lims, linestyle="--")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

# Convenience method on the joint model
def _to_joint_index_from_edges(bin_edges: dict, binned_actions: np.ndarray) -> np.ndarray:
    nN = len(bin_edges["N"]) - 1
    nP = len(bin_edges["P"]) - 1
    nK = len(bin_edges["K"]) - 1
    return (binned_actions[:, 0] * (nP * nK) +
            binned_actions[:, 1] * nK +
            binned_actions[:, 2]).astype(int)

def _digitize_actions(actions: np.ndarray, bin_edges: dict) -> np.ndarray:
    iN = np.clip(np.digitize(actions[:, 0], bin_edges["N"]) - 1, 0, len(bin_edges["N"]) - 2)
    iP = np.clip(np.digitize(actions[:, 1], bin_edges["P"]) - 1, 0, len(bin_edges["P"]) - 2)
    iK = np.clip(np.digitize(actions[:, 2], bin_edges["K"]) - 1, 0, len(bin_edges["K"]) - 2)
    return np.stack([iN, iP, iK], axis=1).astype(int)

class _ReliabilityMixin:
    def reliability_on(self, X: np.ndarray, actions: np.ndarray,
                       n_bins: int = 20,
                       sample_weight: np.ndarray | None = None):
        """
        Compute reliability curve & ECE for this joint π0 on (X, actions).
        """
        if self.bin_edges is None:
            raise AttributeError("bin_edges must be set on the propensity model.")
        probs = self.predict_proba_grid(X)  # (n, C)
        binned = _digitize_actions(actions, self.bin_edges)
        y_joint = _to_joint_index_from_edges(self.bin_edges, binned)
        df, ece = multiclass_reliability_curve(probs, y_joint, n_bins=n_bins, sample_weight=sample_weight)
        return df, ece

# Monkey-patch onto your JointPropensityModel
try:
    from types import MethodType
    # Attach only if not already present
    if not hasattr(JointPropensityModel, "reliability_on"):
        JointPropensityModel.reliability_on = MethodType(_ReliabilityMixin.reliability_on, JointPropensityModel)
except Exception:
    pass