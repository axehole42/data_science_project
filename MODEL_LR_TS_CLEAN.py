# train_logreg_roa_improvement_optuna_timeseries_cv.py
#
# Purpose:
#   Train and evaluate a Logistic Regression model to predict ROA improvement (binary target),
#   using a strictly time-respecting validation approach (walk-forward / time-series CV) and
#   Optuna (TPE/Bayesian optimization) for hyperparameter tuning.
#
# Key design points (mirrors the XGBoost pipeline used elsewhere in the project):
#   1) Time-series Cross-Validation (walk-forward):
#        - Train on earlier years, validate on the next year(s)
#        - Avoids leakage from future information into the past
#   2) Split-wise preprocessing (fit ONLY on training split each time):
#        - Winsorization at 1%/99% per feature (reduce outlier impact)
#        - Mean imputation (means computed after winsorization)
#        - Optional missingness indicators (binary flags)
#        - Optional standardization (helpful for Logistic Regression)
#   3) Class imbalance handling like XGBoost's scale_pos_weight:
#        - Use sample_weight so positive samples are upweighted by neg/pos ratio
#        - Done per fold during CV and on trainval for the final saved model
#   4) Threshold selection:
#        - Choose a classification threshold based on out-of-fold predictions
#        - Criterion: maximize accuracy or F1 (configurable)
#
#
# Run:
#   pip install optuna
#   python train_logreg_roa_improvement_optuna_timeseries_cv.py

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import joblib

# ---- suppress sklearn warnings (clean logs) ----
# We ignore FutureWarning and UserWarning originating from sklearn to keep output readable.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------- PATHS ----------
# Project structure:
#   - task_data/ contains raw data, cleaned data, engineered features, and model artifacts
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

# Input datasets:
RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

# Output directory for model artifacts produced by this script:
MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv_logreg_clean"
MODEL_FILE = MODEL_DIR / "logreg_model.joblib"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
OPTUNA_TRIALS_CSV = MODEL_DIR / "optuna_trials.csv"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"

# ---------- COLUMNS ----------
# Target and time identifiers used throughout the pipeline.
TARGET_COL = "target"
YEAR_COL = "fyear"
GROUP_COL = "gvkey"

# ---------- SETTINGS ----------
SEED = 42

# Preprocessing options (kept aligned with the XGBoost script):
ADD_MISSING_INDICATORS = True
WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

# Standardization (z-scoring) is often beneficial for Logistic Regression,
# especially when penalties like L1 or elastic net are used.
STANDARDIZE = True  # used in SplitPreprocessor

# We reserve the last N years as a strict test set (never used for tuning).
TEST_LAST_N_YEARS = 2

# Walk-forward CV configuration:
#   - Need at least MIN_TRAIN_YEARS for the first training window
#   - Validate on VAL_WINDOW years ahead
#   - Optionally restrict to last N_FOLDS folds (recent periods)
MIN_TRAIN_YEARS = 4
VAL_WINDOW = 1
N_FOLDS = 3

# Optuna tuning configuration:
N_TRIALS = 50
TIMEOUT_SECONDS = None
OPTIMIZE_METRIC = "auc"  # "auc" or "ap"

# Threshold selection configuration:
# We tune a probability cutoff to convert predicted probabilities into class labels.
THRESHOLD_CRITERION = "max_accuracy"  # "max_f1" or "max_accuracy"

# Feature group mapping file:
# This JSON defines which engineered features belong to each group.
FEATURE_MAP_FILE = "task_data/feature_groups.json"

# Select which feature groups to include (subset of all engineered features).
USE_GROUPS = [
    "Liquidity_&_CashFlow",
    "Leverage_&_CapitalStructure",
    "Profitability_&_Returns",
    "Efficiency_/_Activity",
    "FirmCharacteristics_&_Dynamics",
]


def load_engineered_features(path=FEATURE_MAP_FILE, use_groups=None):
    """
    Load engineered feature names from a JSON mapping:
      {
        "GroupNameA": ["feat1", "feat2", ...],
        "GroupNameB": [...],
        ...
      }

    We optionally restrict to the groups listed in `use_groups` and return a
    de-duplicated feature list while preserving the original order.
    """
    with open(path, "r") as f:
        groups = json.load(f)

    if use_groups is None:
        use_groups = list(groups.keys())

    feats = []
    for g in use_groups:
        feats.extend(groups.get(g, []))

    # De-duplicate while keeping order
    seen = set()
    feats = [x for x in feats if not (x in seen or seen.add(x))]
    return feats


# Final engineered feature set used in this model.
ENGINEERED_FEATURES = load_engineered_features(use_groups=USE_GROUPS)


def ensure_features_exist(force: bool = False) -> None:
    """
    Ensure that required intermediate data files exist.
    If not available, call the project’s cleaning + feature engineering scripts.

    - data_cleanup.clean_data() should produce cleaned_data.parquet
    - OLD_feature_engineering.construct_features() should produce features.parquet

    `force=True` re-runs the steps even if outputs already exist.
    """
    from data_cleanup import clean_data
    from feature_engineering import construct_features

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV fehlt: {RAW_CSV}")

    if force or not CLEANED_PARQUET.exists():
        clean_data()
    if force or not FEATURES_PARQUET.exists():
        construct_features()

    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"features.parquet fehlt: {FEATURES_PARQUET}")


def to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns in X to numeric (float32), coercing non-numeric values to NaN.
    This keeps preprocessing robust even if some features are loaded as object/string.
    """
    Xn = X.copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn.astype(np.float32)


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """
    Compute common classification metrics given true labels y_true, predicted probabilities p,
    and a probability threshold used to derive binary predictions.

    Metrics returned:
      - auc (ROC AUC), ap (Average Precision / PR AUC)
      - accuracy, f1, precision, recall
      - confusion matrix counts: tn, fp, fn, tp
    """
    if y_true is None or len(y_true) == 0:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    # Convert probabilities to class predictions at the chosen threshold.
    yhat = (p >= threshold).astype(int)

    # AUC/AP are only defined if both classes appear in y_true; otherwise return NaN.
    out = {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
    }

    # Confusion matrix (fixed label order [0,1]) for TN/FP/FN/TP.
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def best_threshold(y_true: np.ndarray, p: np.ndarray, criterion: str) -> tuple[float, float]:
    """
    Choose a probability threshold that maximizes either:
      - accuracy ("max_accuracy") or
      - F1 score ("max_f1")

    We search over a grid of thresholds from 0.05 to 0.95.
    Returns (best_threshold, best_score).
    """
    if y_true is None or len(y_true) == 0:
        return 0.5, float("nan")

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        yhat = (p >= t).astype(int)

        if criterion == "max_accuracy":
            score = accuracy_score(y_true, yhat)
        elif criterion == "max_f1":
            score = f1_score(y_true, yhat, zero_division=0)
        else:
            raise ValueError("criterion must be 'max_accuracy' or 'max_f1'")

        if score > best_score:
            best_score, best_t = float(score), float(t)

    return best_t, best_score


def build_walkforward_folds(years_trainval: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build walk-forward (time-series) folds from the available non-test years.

    Example (VAL_WINDOW=1):
      years_trainval = [2010, 2011, 2012, 2013, 2014, 2015]
      MIN_TRAIN_YEARS = 4

      fold1: train=[2010,2011,2012,2013], val=[2014]
      fold2: train=[2010..2014], val=[2015]

    If N_FOLDS is set, we keep only the last N_FOLDS folds (most recent validation periods).
    """
    years_trainval = np.array(sorted(years_trainval))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    start = MIN_TRAIN_YEARS
    for i in range(start, len(years_trainval) - VAL_WINDOW + 1):
        train_years = years_trainval[:i]
        val_years = years_trainval[i : i + VAL_WINDOW]
        folds.append((train_years, val_years))

    if not folds:
        raise ValueError("Zu wenige Jahre für Walk-forward CV. MIN_TRAIN_YEARS/VAL_WINDOW anpassen.")

    if N_FOLDS is not None and len(folds) > N_FOLDS:
        folds = folds[-N_FOLDS:]

    return folds


def make_Xy_raw(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract raw feature matrix X and target vector y from the dataframe.

    We enforce that all expected engineered features exist; if not, fail early,
    because missing features would otherwise cause silent errors downstream.
    """
    missing = [c for c in ENGINEERED_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Missing engineered features in dataframe: {missing}")
    X = d[ENGINEERED_FEATURES].copy()
    y = d[TARGET_COL].astype(int).to_numpy()
    return X, y


class SplitPreprocessor:
    """
    Split-wise preprocessor (fit on train only, apply to train/val/test).
    This is designed to be IDENTICAL in spirit to the project's XGBoost preprocessing.

    Steps performed per split:
      1) Convert to numeric and compute winsorization bounds (1% and 99% quantiles)
         ignoring NaNs
      2) Clip values to those bounds (winsorization)
      3) Compute mean per feature on the winsorized training data
      4) Impute NaNs using those means
      5) Optionally standardize (z-score) using mean/std computed after imputation
      6) Optionally add missingness indicators based on original NaN pattern
    """

    def __init__(
        self,
        feature_cols: list[str],
        p_low: float = 0.01,
        p_high: float = 0.99,
        add_missing_indicators: bool = True,
    ):
        self.feature_cols = list(feature_cols)
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.add_missing_indicators = bool(add_missing_indicators)

        # Learned parameters from fit():
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.scale_mean_: pd.Series | None = None
        self.scale_std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "SplitPreprocessor":
        """
        Fit preprocessing parameters on training features X only.
        """
        Xn = to_numeric_matrix(X[self.feature_cols])

        # Quantile-based bounds for winsorization.
        lower = Xn.quantile(self.p_low, numeric_only=True).reindex(self.feature_cols)
        upper = Xn.quantile(self.p_high, numeric_only=True).reindex(self.feature_cols)

        # If a column is entirely NaN, quantiles may be NaN -> use infinities to avoid clipping issues.
        lower = lower.fillna(-np.inf)
        upper = upper.fillna(np.inf)

        # Winsorize (clip) extremes.
        Xc = Xn.clip(lower=lower, upper=upper, axis=1)

        # Mean imputation after winsorization.
        mean = Xc.mean(axis=0, skipna=True).reindex(self.feature_cols)
        mean = mean.fillna(0.0)

        Xi = Xc.fillna(mean)

        # Parameters for standardization (computed after imputation).
        scale_mean = Xi.mean(axis=0, skipna=True).reindex(self.feature_cols)
        scale_mean = scale_mean.fillna(0.0)

        scale_std = Xi.std(axis=0, skipna=True, ddof=0).reindex(self.feature_cols)
        # Avoid division by zero: if std==0 (constant feature), replace with 1.
        scale_std = scale_std.replace(0.0, 1.0).fillna(1.0)

        self.lower_ = lower
        self.upper_ = upper
        self.mean_ = mean
        self.scale_mean_ = scale_mean
        self.scale_std_ = scale_std
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing steps to a new split (train/val/test).
        """
        if (
            self.lower_ is None
            or self.upper_ is None
            or self.mean_ is None
            or self.scale_mean_ is None
            or self.scale_std_ is None
        ):
            raise RuntimeError("Preprocessor not fitted.")

        Xn = to_numeric_matrix(X[self.feature_cols])

        # Missing indicators reflect original missingness before imputation.
        if self.add_missing_indicators:
            miss = Xn.isna().astype(np.int8)
            miss.columns = [f"{c}__is_missing" for c in self.feature_cols]
        else:
            miss = None

        # Winsorize and impute with training means.
        Xc = Xn.clip(lower=self.lower_, upper=self.upper_, axis=1)
        Xi = Xc.fillna(self.mean_)

        # Optional standardization using training mean/std.
        if STANDARDIZE:
            Xi = (Xi - self.scale_mean_) / self.scale_std_

        # Append missing indicators as extra columns (if enabled).
        out = pd.concat([Xi, miss], axis=1) if miss is not None else Xi
        return out.astype(np.float32)


def safe_predict_proba(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    """
    Convenience wrapper: return the positive-class probability vector as float64.
    """
    return model.predict_proba(X)[:, 1].astype(np.float64)


def clean_logreg_params(params: dict) -> dict:
    """
    sklearn>=1.8 emits warnings if l1_ratio is passed when penalty != 'elasticnet'.
    This helper enforces that l1_ratio is ONLY present when penalty == 'elasticnet'.
    """
    p = dict(params)
    if p.get("penalty") != "elasticnet":
        p.pop("l1_ratio", None)
    return p


def main() -> None:
    import os

    # Ensure relative paths are resolved consistently (script executed from project root).
    os.chdir(PROJECT_ROOT)
    ensure_features_exist(force=False)

    # Load engineered features dataset.
    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' nicht in {FEATURES_PARQUET} gefunden.")

    # Columns not used as model inputs. (We keep this set primarily for documentation/config dumps.)
    drop_cols = {
        TARGET_COL,
        GROUP_COL,
        YEAR_COL,
        "datadate",
        "conm",
        "indfmt",
        "datafmt",
        "popsrc",
        "consol",
    }

    # Derive full year range present in the dataset.
    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Zu wenige Jahre insgesamt für MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS.")

    # Time-based split:
    #   - last TEST_LAST_N_YEARS are held out as final test set
    #   - all prior years are used for train/validation and tuning
    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    # Create walk-forward folds on trainval years.
    folds = build_walkforward_folds(trainval_years)

    # For final reporting we also define a "final train" and "final val" split
    # as the last VAL_WINDOW year(s) inside trainval.
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    # Base features are engineered features; optionally add missingness indicator columns.
    base_feature_cols = list(ENGINEERED_FEATURES)
    feature_cols = (
        base_feature_cols + [f"{c}__is_missing" for c in base_feature_cols]
        if ADD_MISSING_INDICATORS
        else base_feature_cols
    )

    # Compute an XGB-like scale_pos_weight based on the final_train split (for reporting).
    # Later, for the saved deployment artifact, we compute it again on all trainval years.
    df_final_train_for_weight = df[df[YEAR_COL].isin(final_train_years)]
    pos = int(df_final_train_for_weight[TARGET_COL].sum())
    neg = int((df_final_train_for_weight[TARGET_COL] == 0).sum())
    scale_pos_weight = (neg / pos) if (pos > 0 and neg > 0) else 1.0

    def make_sample_weight(y: np.ndarray, spw: float) -> np.ndarray:
        """
        Create per-sample weights to mimic XGBoost's scale_pos_weight:
          - weight = 1 for negatives
          - weight = spw for positives
        """
        w = np.ones_like(y, dtype=np.float64)
        w[y == 1] = float(spw)
        return w

    # Fixed Logistic Regression parameters:
    # max_iter increased to ensure convergence in penalized settings.
    # n_jobs intentionally omitted (no effect in sklearn>=1.8).
    base_params = {
        "random_state": SEED,
        "max_iter": 5000,
    }

    # Optuna setup: TPE sampler (Bayesian optimization-like).
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        """
        Train/evaluate one fold of the walk-forward CV for a given parameter set.

        Important: preprocessing is fit on fold training data ONLY, then applied to validation.
        Class imbalance is handled with per-fold scale_pos_weight.
        """
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

        X_tr_raw, y_tr = make_Xy_raw(df_tr)
        X_va_raw, y_va = make_Xy_raw(df_va)

        # Fit preprocessing only on training portion of this fold.
        pp = SplitPreprocessor(
            feature_cols=ENGINEERED_FEATURES,
            p_low=WINSOR_P_LOW,
            p_high=WINSOR_P_HIGH,
            add_missing_indicators=ADD_MISSING_INDICATORS,
        ).fit(X_tr_raw)

        # Transform train/val and align columns (safety if some indicator cols are missing in a split).
        X_tr = pp.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
        X_va = pp.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

        # Remove l1_ratio unless elasticnet is used (avoids sklearn warnings).
        params_clean = clean_logreg_params(params)
        model = LogisticRegression(**params_clean)

        # Compute fold-specific scale_pos_weight and use it via sample_weight.
        pos_tr = int(y_tr.sum())
        neg_tr = int((y_tr == 0).sum())
        spw_fold = (neg_tr / pos_tr) if (pos_tr > 0 and neg_tr > 0) else 1.0

        sw_tr = make_sample_weight(y_tr, spw_fold)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        # Predict probabilities on validation split.
        p = safe_predict_proba(model, X_va)

        # Return optimization score (AUC or AP).
        if OPTIMIZE_METRIC == "auc":
            return float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
        if OPTIMIZE_METRIC == "ap":
            return float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
        raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective: propose hyperparameters, run them through walk-forward CV,
        and return mean CV performance.
        """
        # Choose penalty type. Elastic net requires 'saga' solver in sklearn.
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

        if penalty == "elasticnet":
            solver = "saga"
        else:
            # For L1/L2 we allow liblinear or saga.
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

        # Regularization strength (inverse lambda): searched on log scale.
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)

        params = dict(base_params)
        params.update({"penalty": penalty, "solver": solver, "C": C})

        # Elastic net mixes L1 and L2 via l1_ratio.
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)

        # Evaluate on all walk-forward folds and average.
        scores = []
        for tr_years, va_years in folds:
            scores.append(score_fold(params, tr_years, va_years))
        return float(np.mean(scores))

    # Create and run Optuna study.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    # Extract best parameters and merge with base params.
    best_params = dict(base_params)
    best_params.update(study.best_params)

    # Ensure solver consistency for elasticnet (required by sklearn).
    if best_params.get("penalty") == "elasticnet":
        best_params["solver"] = "saga"

    best_params_clean = clean_logreg_params(best_params)

    # ----- OOF THRESHOLD (walk-forward folds) -----
    # We generate out-of-fold (OOF) predictions across the CV folds using the best hyperparameters.
    # Then we pick a decision threshold that maximizes the chosen criterion on these OOF predictions.
    oof_y = []
    oof_p = []
    for tr_years, va_years in folds:
        df_tr = df[df[YEAR_COL].isin(tr_years)]
        df_va = df[df[YEAR_COL].isin(va_years)]

        X_tr_raw, y_tr = make_Xy_raw(df_tr)
        X_va_raw, y_va = make_Xy_raw(df_va)

        pp = SplitPreprocessor(
            feature_cols=ENGINEERED_FEATURES,
            p_low=WINSOR_P_LOW,
            p_high=WINSOR_P_HIGH,
            add_missing_indicators=ADD_MISSING_INDICATORS,
        ).fit(X_tr_raw)

        X_tr = pp.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
        X_va = pp.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

        model_oof = LogisticRegression(**best_params_clean)

        pos_tr = int(y_tr.sum())
        neg_tr = int((y_tr == 0).sum())
        spw_fold = (neg_tr / pos_tr) if (pos_tr > 0 and neg_tr > 0) else 1.0

        sw_tr = make_sample_weight(y_tr, spw_fold)
        model_oof.fit(X_tr, y_tr, sample_weight=sw_tr)

        p = safe_predict_proba(model_oof, X_va)
        oof_y.append(y_va)
        oof_p.append(p)

    # Combine OOF arrays from all folds.
    y_oof = np.concatenate(oof_y) if len(oof_y) > 0 else np.array([], dtype=int)
    p_oof = np.concatenate(oof_p) if len(oof_p) > 0 else np.array([], dtype=float)

    # Pick threshold on OOF predictions.
    thr, thr_score = best_threshold(y_oof, p_oof, THRESHOLD_CRITERION)

    # ----- FINAL TRAIN (for reporting only) -----
    # We train on final_train_years and evaluate on final_val_years for a "final validation" report.
    df_final_train = df[df[YEAR_COL].isin(final_train_years)]
    df_final_val = df[df[YEAR_COL].isin(final_val_years)]

    X_tr_raw, y_tr = make_Xy_raw(df_final_train)
    X_va_raw, y_va = make_Xy_raw(df_final_val)

    pp_final = SplitPreprocessor(
        feature_cols=ENGINEERED_FEATURES,
        p_low=WINSOR_P_LOW,
        p_high=WINSOR_P_HIGH,
        add_missing_indicators=ADD_MISSING_INDICATORS,
    ).fit(X_tr_raw)

    X_tr = pp_final.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
    X_va = pp_final.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

    # Prepare strict test set (last TEST_LAST_N_YEARS).
    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test_raw, y_test = make_Xy_raw(df_test)
    X_test = pp_final.transform(X_test_raw).reindex(columns=feature_cols, fill_value=0.0)

    # Train model on final training split with sample weighting.
    model = LogisticRegression(**best_params_clean)
    sw_tr = make_sample_weight(y_tr, scale_pos_weight)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)

    # Predict probabilities for final val and test.
    p_val = safe_predict_proba(model, X_va)
    p_test = safe_predict_proba(model, X_test)

    # ----- FINAL REFIT (deployment artifact) on ALL non-test years -----
    # For the model artifact we save, we refit on all trainval years (everything except the held-out test years).
    df_trainval = df[df[YEAR_COL].isin(trainval_years)]
    X_trainval_raw, y_trainval = make_Xy_raw(df_trainval)

    pp_trainval = SplitPreprocessor(
        feature_cols=ENGINEERED_FEATURES,
        p_low=WINSOR_P_LOW,
        p_high=WINSOR_P_HIGH,
        add_missing_indicators=ADD_MISSING_INDICATORS,
    ).fit(X_trainval_raw)

    X_trainval = pp_trainval.transform(X_trainval_raw).reindex(columns=feature_cols, fill_value=0.0)

    # Compute scale_pos_weight on trainval (used for the final saved model).
    pos_tv = int(y_trainval.sum())
    neg_tv = int((y_trainval == 0).sum())
    scale_pos_weight_trainval = (neg_tv / pos_tv) if (pos_tv > 0 and neg_tv > 0) else 1.0

    # Fit final model artifact.
    model = LogisticRegression(**best_params_clean)
    sw_trainval = make_sample_weight(y_trainval, scale_pos_weight_trainval)
    model.fit(X_trainval, y_trainval, sample_weight=sw_trainval)

    # ----- SAVE ARTIFACTS -----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model + preprocessing + metadata needed for inference.
    joblib.dump(
        {
            "model": model,
            "preprocessor": pp_trainval,
            "feature_cols": feature_cols,
            "engineered_features": ENGINEERED_FEATURES,
            "scale_pos_weight_like": scale_pos_weight_trainval,
            "best_params": best_params_clean,
        },
        MODEL_FILE,
    )

    # Export Optuna trials for transparency/reproducibility (one row per trial).
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    # Save best hyperparameters + best mean CV score.
    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": float(study.best_value), "best_params": best_params_clean}, f, indent=2)

    # Feature importance proxy for Logistic Regression:
    # Use absolute coefficient magnitude as a simple importance measure.
    coef = model.coef_.reshape(-1)
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": np.abs(coef).astype(float), "coef": coef.astype(float)}
    ).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # Collect metrics and configuration for reporting.
    metrics = {
        "split": {
            "test_last_n_years": int(TEST_LAST_N_YEARS),
            "test_years": [int(y) for y in test_years],
            "trainval_years": [int(y) for y in trainval_years],
            "final_train_years": [int(y) for y in final_train_years],
            "final_val_years": [int(y) for y in final_val_years],
            "walkforward_folds": [{"train": [int(x) for x in tr], "val": [int(x) for x in va]} for tr, va in folds],
        },
        "tuning": {
            "method": "optuna_tpe_walkforward_cv",
            "model": "logistic_regression",
            "n_trials": int(N_TRIALS),
            "optimize_metric": OPTIMIZE_METRIC,
            "best_value_mean_cv": float(study.best_value),
            "best_params": best_params_clean,
        },
        "preprocessing": {
            "winsor_p_low": float(WINSOR_P_LOW),
            "winsor_p_high": float(WINSOR_P_HIGH),
            "mean_imputation": True,
            "missing_indicators": bool(ADD_MISSING_INDICATORS),
            "standardize": True,
            "fit_scope": "fit on each split's training data only; applied to train/val/test accordingly; saved model refit on trainval_years",
        },
        "class_imbalance_handling": {
            "xgb_like": True,
            "scale_pos_weight": float(scale_pos_weight_trainval),
            "implementation": "sample_weight: positives weighted by scale_pos_weight (estimated on trainval_years for saved model; per-fold for CV/OOF)",
        },
        "threshold_selection": {
            "optimized_on": "oof_walkforward_folds (last k folds, respecting N_FOLDS)",
            "criterion": THRESHOLD_CRITERION,
            "best_threshold": float(thr),
            "best_score_on_final_val": float(thr_score),
        },
        "final_val_metrics_at_threshold": compute_metrics(y_va, p_val, thr),
        "test_metrics_at_val_threshold": compute_metrics(y_test, p_test, thr),
    }

    # Write metrics to disk (human-readable JSON).
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write configuration for reproducibility / documentation.
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "year_col": YEAR_COL,
                "drop_cols": sorted([c for c in drop_cols if c in df.columns]),
                "test_last_n_years": int(TEST_LAST_N_YEARS),
                "min_train_years": int(MIN_TRAIN_YEARS),
                "val_window": int(VAL_WINDOW),
                "n_folds": None if N_FOLDS is None else int(N_FOLDS),
                "threshold_criterion": THRESHOLD_CRITERION,
                "winsor_p_low": float(WINSOR_P_LOW),
                "winsor_p_high": float(WINSOR_P_HIGH),
                "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
                "imputation": "mean",
                "standardize": True,
                "class_imbalance": {
                    "xgb_scale_pos_weight": float(scale_pos_weight_trainval),
                    "logreg_equivalent": "sample_weight positives=scale_pos_weight (estimated on trainval_years for saved model; per-fold for CV/OOF)",
                },
                "optuna": {
                    "sampler": "TPESampler",
                    "n_trials": int(N_TRIALS),
                    "timeout_seconds": TIMEOUT_SECONDS,
                    "optimize_metric": OPTIMIZE_METRIC,
                    "tuned_hyperparams": [
                        "penalty",
                        "solver",
                        "C",
                        "l1_ratio (only if elasticnet)",
                    ],
                },
            },
            f,
            indent=2,
        )

    # Console output: list saved artifacts.
    print("Saved:")
    print(f"  {MODEL_DIR}")
    print(f"  {MODEL_FILE}")
    print(f"  {METRICS_JSON}")
    print(f"  {CONFIG_JSON}")
    print(f"  {FEATURE_IMPORTANCE_CSV}")
    print(f"  {OPTUNA_TRIALS_CSV}")
    print(f"  {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()

