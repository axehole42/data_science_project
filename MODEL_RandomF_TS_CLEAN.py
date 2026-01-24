
# train_rf_roa_improvement_optuna_timeseries_cv_fast.py
#
# Random Forest classifier for "ROA improvement" (binary target) with:
# - Time-series cross-validation (walk-forward / expanding window)
# - Hyperparameter tuning via Optuna (TPE sampler)
# - Split-safe preprocessing:
#     * winsorization (per-feature clipping to percentiles)
#     * mean imputation
#     * optional missing-value indicator columns
#   IMPORTANT: preprocessing is fit ONLY on the training subset of each split to avoid leakage.
#
# This script mirrors the data handling choices of the corresponding XGBoost pipeline:
# - reads the same parquet inputs
# - uses the same engineered feature list from feature_groups.json
# - holds out the last N years as an untouched test set
# - uses walk-forward folds on the remaining years for tuning
#
# Speed-oriented adjustments vs. a naive RandomForest setup:
# - tighter Optuna search space and fewer trials by default
# - cap CPU threads (laptop-friendly)
# - warm_start staged fitting inside a fold to quickly abandon clearly bad trials
# - avoid sklearn warnings about class_weight by passing an explicit {0: w0, 1: w1} dict
#
# Output artifacts:
# - trained deployment model (refit on ALL non-test years)
# - preprocessing object (fitted on all non-test years)
# - metrics JSON, config JSON
# - feature importances CSV
# - Optuna trials CSV and best-params JSON

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# PATHS / FILE LOCATIONS
# ---------------------------
# Project root is the folder containing this script.
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

# Raw and intermediate files (kept consistent with the rest of the project)
RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

# Output folder for this model run
MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv_clean_rf_fast"
MODEL_FILE = MODEL_DIR / "rf_model.joblib"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
OPTUNA_TRIALS_CSV = MODEL_DIR / "optuna_trials.csv"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"

# ---------------------------
# COLUMN NAMES (PROJECT CONVENTION)
# ---------------------------
TARGET_COL = "target"  # binary label
YEAR_COL = "fyear"     # fiscal year column used for time splits
GROUP_COL = "gvkey"    # firm identifier (not used directly in this script, but kept for completeness)

# ---------------------------
# GLOBAL SETTINGS
# ---------------------------
SEED = 42  # reproducibility (Optuna sampler + RF randomness)

# Preprocessing controls
ADD_MISSING_INDICATORS = True  # add per-feature __is_missing flags
WINSOR_P_LOW = 0.01            # lower winsorization percentile
WINSOR_P_HIGH = 0.99           # upper winsorization percentile

# Test set: hold out the last N years completely untouched during tuning
TEST_LAST_N_YEARS = 2

# Walk-forward CV on the remaining years:
MIN_TRAIN_YEARS = 4  # minimum expanding window size before the first validation
VAL_WINDOW = 1       # validation window size in years
N_FOLDS = 3          # if None: use all folds; else: use last N_FOLDS (faster)

# Optuna tuning
N_TRIALS = 25
TIMEOUT_SECONDS = 1800          # cap runtime in seconds (None disables)
OPTIMIZE_METRIC = "auc"         # "auc" or "ap" (average precision)

# Threshold selection is only for reporting metrics (RF itself outputs probabilities)
THRESHOLD_CRITERION = "max_accuracy"  # "max_f1" or "max_accuracy"

# CPU usage (reduce to avoid saturating a laptop)
RF_N_JOBS = 4

# Early-abandon cutoffs (after first warm-start stage inside a fold)
AUC_BAD_CUT = 0.52
AP_BAD_CUT = 0.05

# Feature groups file and the selected groups to use
FEATURE_MAP_FILE = "task_data/feature_groups.json"

USE_GROUPS = [
    "Liquidity_&_CashFlow",
    "Leverage_&_CapitalStructure",
    "Profitability_&_Returns",
    "Efficiency_/_Activity",
    "FirmCharacteristics_&_Dynamics",
]


def load_engineered_features(path=FEATURE_MAP_FILE, use_groups=None):
    """
    Load the engineered feature list from a JSON mapping {group_name: [features...]}.

    We optionally restrict to a subset of groups (USE_GROUPS), and we keep feature order
    while removing duplicates.
    """
    with open(path, "r") as f:
        groups = json.load(f)

    if use_groups is None:
        use_groups = list(groups.keys())

    feats = []
    for g in use_groups:
        feats.extend(groups.get(g, []))

    # Preserve order while removing duplicates
    seen = set()
    feats = [x for x in feats if not (x in seen or seen.add(x))]
    return feats


# The base engineered feature set used across the script
ENGINEERED_FEATURES = load_engineered_features(use_groups=USE_GROUPS)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def ensure_features_exist(force: bool = False) -> None:
    """
    Ensure that the project has produced the required parquet files.

    - If cleaned_data.parquet is missing (or force=True), run the cleaning pipeline.
    - If features.parquet is missing (or force=True), run feature engineering.

    This keeps the script self-contained for a lecturer/reviewer:
    running this file should (re)create required inputs if needed.
    """
    from data_cleanup import clean_data
    from OLD_feature_engineering import construct_features

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV missing: {RAW_CSV}")

    if force or not CLEANED_PARQUET.exists():
        clean_data()
    if force or not FEATURES_PARQUET.exists():
        construct_features()

    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(f"features.parquet missing: {FEATURES_PARQUET}")


def to_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric where possible, coercing errors to NaN.
    Output is float32 for memory/performance.
    """
    Xn = X.copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn.astype(np.float32)


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """
    Compute standard classification metrics at a given probability threshold.

    - AUC/AP are computed on probabilities (only valid if both classes appear).
    - accuracy/f1/precision/recall are computed on hard labels (p >= threshold).
    - confusion matrix counts are included for interpretability.
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

    yhat = (p >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
    }

    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def best_threshold(y_true: np.ndarray, p: np.ndarray, criterion: str) -> tuple[float, float]:
    """
    Choose a probability threshold by scanning a grid and maximizing a simple criterion.

    This threshold is used only for reporting (not for training).
    Criterion:
      - "max_accuracy": maximize accuracy
      - "max_f1": maximize F1 score
    """
    thresholds = np.linspace(0.05, 0.95, 181)  # step size ~0.005
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
    Build walk-forward (expanding-window) folds on a set of years.

    Example with MIN_TRAIN_YEARS=4, VAL_WINDOW=1:
      train = first 4 years, val = next year
      train = first 5 years, val = next year
      ...

    If N_FOLDS is not None, we take the last N_FOLDS folds (closest to the test period),
    which is a common practical compromise for speed and relevance.
    """
    years_trainval = np.array(sorted(years_trainval))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    start = MIN_TRAIN_YEARS
    for i in range(start, len(years_trainval) - VAL_WINDOW + 1):
        train_years = years_trainval[:i]
        val_years = years_trainval[i : i + VAL_WINDOW]
        folds.append((train_years, val_years))

    if not folds:
        raise ValueError("Not enough years for walk-forward CV. Adjust MIN_TRAIN_YEARS/VAL_WINDOW.")

    if N_FOLDS is not None and len(folds) > N_FOLDS:
        folds = folds[-N_FOLDS:]

    return folds


def make_Xy_raw(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract feature matrix X (engineered features) and target vector y from a dataframe.

    We explicitly check that all expected engineered features exist in the dataframe.
    """
    missing = [c for c in ENGINEERED_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Missing engineered features in dataframe: {missing}")

    X = d[ENGINEERED_FEATURES].copy()
    y = d[TARGET_COL].astype(int).to_numpy()
    return X, y


def class_weight_dict_from_y(y: np.ndarray) -> dict[int, float]:
    """
    Compute sklearn-compatible class weights as an explicit dict {0: w0, 1: w1}.

    We prefer a fixed dict (rather than class_weight="balanced") because:
    - weights depend on each training split
    - passing an explicit dict avoids certain sklearn warnings in some configurations
    """
    if y is None or len(y) == 0 or len(np.unique(y)) < 2:
        return {0: 1.0, 1: 1.0}

    classes = np.array([0, 1], dtype=int)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {0: float(w[0]), 1: float(w[1])}


class SplitPreprocessor:
    """
    Split-safe preprocessing for tabular financial features.

    Fit ONLY on the training data of a split:
      1) Winsorize (clip) each feature to [p_low, p_high] quantiles
      2) Mean-impute missing values (means computed AFTER winsorization)
      3) Optionally add missingness indicator columns (based on original NaNs)

    Then transform any dataset (train/val/test) using the learned params.

    This design prevents data leakage:
    - quantile cutoffs and imputation means come strictly from the training subset.
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

        # Learned parameters (set during fit)
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.mean_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "SplitPreprocessor":
        """
        Fit quantile clipping bounds and imputation means on training data only.
        """
        Xn = to_numeric_matrix(X[self.feature_cols])

        # Quantiles computed per column (ignore NaNs)
        lower = Xn.quantile(self.p_low, numeric_only=True).reindex(self.feature_cols)
        upper = Xn.quantile(self.p_high, numeric_only=True).reindex(self.feature_cols)

        # If a column is entirely NaN, quantile can be NaN -> use infinite bounds (no clipping)
        lower = lower.fillna(-np.inf)
        upper = upper.fillna(np.inf)

        # Clip extreme values (winsorization)
        Xc = Xn.clip(lower=lower, upper=upper, axis=1)

        # Mean imputation values computed after clipping
        mean = Xc.mean(axis=0, skipna=True).reindex(self.feature_cols)
        mean = mean.fillna(0.0)  # fallback if mean is NaN (e.g., all values missing)

        self.lower_ = lower
        self.upper_ = upper
        self.mean_ = mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned clipping and imputation to any dataset.
        Optionally add missingness indicator columns.
        """
        if self.lower_ is None or self.upper_ is None or self.mean_ is None:
            raise RuntimeError("Preprocessor not fitted.")

        Xn = to_numeric_matrix(X[self.feature_cols])

        # Missingness indicators are computed from the ORIGINAL missingness (before imputation)
        if self.add_missing_indicators:
            miss = Xn.isna().astype(np.int8)
            miss.columns = [f"{c}__is_missing" for c in self.feature_cols]
        else:
            miss = None

        # Winsorize and then impute with training means
        Xc = Xn.clip(lower=self.lower_, upper=self.upper_, axis=1)
        Xi = Xc.fillna(self.mean_)

        # Concatenate indicators if enabled
        out = pd.concat([Xi, miss], axis=1) if miss is not None else Xi
        return out.astype(np.float32)


def main() -> None:
    """
    Main training routine:
    1) Load features.parquet
    2) Split by time: last N years = test, remaining = train/validation
    3) Create walk-forward folds on train/validation years
    4) Optuna tuning on CV folds (with split-safe preprocessing)
    5) Choose reporting threshold using out-of-fold predictions
    6) Train final model on final_train_years, evaluate on final_val_years and test_years
    7) Refit deployment model on ALL non-test years and save artifacts
    """
    import os

    # Ensure relative file lookups behave consistently
    os.chdir(PROJECT_ROOT)

    # Make sure required parquet inputs exist (build them if missing)
    ensure_features_exist(force=False)

    # Load engineered features dataset
    df = pd.read_parquet(FEATURES_PARQUET)

    # Basic schema checks
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' not found in {FEATURES_PARQUET}.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' not found in {FEATURES_PARQUET}.")

    # Determine all years available
    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Not enough total years for MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS.")

    # Time-based split: hold out last N years as test
    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    # Build walk-forward folds on train/validation years
    folds = build_walkforward_folds(trainval_years)

    # Final split (for a last sanity check similar to XGB workflow):
    # - final_val_years = last VAL_WINDOW year(s) of trainval
    # - final_train_years = all earlier trainval years
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    # Base engineered features (numeric)
    base_feature_cols = list(ENGINEERED_FEATURES)

    # If enabled, we add missing-indicator columns after preprocessing
    feature_cols = (
        base_feature_cols + [f"{c}__is_missing" for c in base_feature_cols]
        if ADD_MISSING_INDICATORS
        else base_feature_cols
    )

    # For reporting: class balance in the final training block (before tuning)
    df_final_train_for_stats = df[df[YEAR_COL].isin(final_train_years)]
    pos = int(df_final_train_for_stats[TARGET_COL].sum())
    neg = int((df_final_train_for_stats[TARGET_COL] == 0).sum())

    # Optuna setup
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        """
        Train/evaluate one fold with split-safe preprocessing and a warm-start RF.

        We do staged training (warm_start) to quickly reject configurations that look
        clearly weak after a small number of trees.
        """
        # Select rows for this fold
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

        # Extract raw X/y (before preprocessing)
        X_tr_raw, y_tr = make_Xy_raw(df_tr)
        X_va_raw, y_va = make_Xy_raw(df_va)

        # Fit preprocessing ONLY on training data of the fold
        pp = SplitPreprocessor(
            feature_cols=ENGINEERED_FEATURES,
            p_low=WINSOR_P_LOW,
            p_high=WINSOR_P_HIGH,
            add_missing_indicators=ADD_MISSING_INDICATORS,
        ).fit(X_tr_raw)

        # Transform train and validation
        X_tr = pp.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
        X_va = pp.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

        # Compute split-specific class weights
        cw = class_weight_dict_from_y(y_tr)

        # Warm-start stages: fit fewer trees first, abandon if performance is bad
        n_final = int(params["n_estimators"])
        if n_final >= 350:
            stages = [150, 300, n_final]
        elif n_final > 150:
            stages = [150, n_final]
        else:
            stages = [n_final]

        # Parameters without n_estimators (we set that per stage)
        base = dict(params)
        base.pop("n_estimators", None)

        # Initialize the RF with warm_start=True so we can increase n_estimators incrementally
        clf = RandomForestClassifier(
            random_state=SEED,
            n_jobs=RF_N_JOBS,
            class_weight=cw,
            warm_start=True,
            **base,
            n_estimators=stages[0],
        )

        # Stage 1 fit
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_va)[:, 1]

        # Choose which metric Optuna optimizes
        if OPTIMIZE_METRIC == "auc":
            best = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
            bad_cut = AUC_BAD_CUT
        elif OPTIMIZE_METRIC == "ap":
            best = float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
            bad_cut = AP_BAD_CUT
        else:
            raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

        # Early abandon: if it looks clearly poor after stage 1, skip deeper stages
        if best < bad_cut and len(stages) > 1:
            return best

        # Continue with later stages (more trees) if not abandoned
        for s in stages[1:]:
            clf.set_params(n_estimators=s)
            clf.fit(X_tr, y_tr)  # warm_start adds trees
            p = clf.predict_proba(X_va)[:, 1]

            if OPTIMIZE_METRIC == "auc":
                score = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
            else:
                score = float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0

            if score > best:
                best = score

        return best

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective: mean performance across walk-forward folds.

        Search space is intentionally limited to keep runtime manageable.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            "max_features": trial.suggest_float("max_features", 0.2, 0.6),
            "bootstrap": True,
        }

        scores = []
        for tr_years, va_years in folds:
            scores.append(score_fold(params, tr_years, va_years))

        return float(np.mean(scores))

    # Run Optuna study
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    best_params = dict(study.best_params)

    # ------------------------------------------------------------
    # OUT-OF-FOLD THRESHOLD SELECTION (for reporting)
    # ------------------------------------------------------------
    # We generate out-of-fold predictions using the best params on the same folds
    # and then pick a probability threshold that optimizes the chosen criterion.
    oof_y = []
    oof_p = []

    for tr_years, va_years in folds:
        df_tr = df[df[YEAR_COL].isin(tr_years)]
        df_va = df[df[YEAR_COL].isin(va_years)]

        X_tr_raw, y_tr = make_Xy_raw(df_tr)
        X_va_raw, y_va = make_Xy_raw(df_va)

        pp_oof = SplitPreprocessor(
            feature_cols=ENGINEERED_FEATURES,
            p_low=WINSOR_P_LOW,
            p_high=WINSOR_P_HIGH,
            add_missing_indicators=ADD_MISSING_INDICATORS,
        ).fit(X_tr_raw)

        X_tr = pp_oof.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
        X_va = pp_oof.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

        cw_oof = class_weight_dict_from_y(y_tr)

        clf_oof = RandomForestClassifier(
            random_state=SEED,
            n_jobs=RF_N_JOBS,
            class_weight=cw_oof,
            **best_params,
        )
        clf_oof.fit(X_tr, y_tr)
        p = clf_oof.predict_proba(X_va)[:, 1]

        oof_y.append(y_va)
        oof_p.append(p)

    y_oof = np.concatenate(oof_y) if len(oof_y) > 0 else np.array([], dtype=int)
    p_oof = np.concatenate(oof_p) if len(oof_p) > 0 else np.array([], dtype=float)

    thr, thr_score_oof = (
        best_threshold(y_oof, p_oof, THRESHOLD_CRITERION) if len(y_oof) > 0 else (0.5, float("nan"))
    )

    # ------------------------------------------------------------
    # FINAL TRAIN / VALIDATION EVALUATION (fixed time split)
    # ------------------------------------------------------------
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

    # Prepare test set (untouched years)
    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test_raw, y_test = make_Xy_raw(df_test)
    X_test = pp_final.transform(X_test_raw).reindex(columns=feature_cols, fill_value=0.0)

    cw_final = class_weight_dict_from_y(y_tr)

    # Train model on final training years, evaluate on final val and test (using chosen threshold)
    clf = RandomForestClassifier(
        random_state=SEED,
        n_jobs=RF_N_JOBS,
        class_weight=cw_final,
        **best_params,
    )
    clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_va)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    # Report how good the chosen threshold performs on the final validation year(s)
    yhat_va_thr = (p_val >= thr).astype(int)
    if THRESHOLD_CRITERION == "max_accuracy":
        thr_score_final_val = float(accuracy_score(y_va, yhat_va_thr)) if len(y_va) > 0 else float("nan")
    elif THRESHOLD_CRITERION == "max_f1":
        thr_score_final_val = float(f1_score(y_va, yhat_va_thr, zero_division=0)) if len(y_va) > 0 else float("nan")
    else:
        raise ValueError("criterion must be 'max_accuracy' or 'max_f1'")

    # ------------------------------------------------------------
    # FINAL REFIT FOR DEPLOYMENT
    # ------------------------------------------------------------
    # For the saved model, we refit on all non-test years (trainval_years).
    # Preprocessing is also refit on those years, and saved together with the model.
    df_trainval = df[df[YEAR_COL].isin(trainval_years)]

    X_tv_raw, y_tv = make_Xy_raw(df_trainval)

    pp_deploy = SplitPreprocessor(
        feature_cols=ENGINEERED_FEATURES,
        p_low=WINSOR_P_LOW,
        p_high=WINSOR_P_HIGH,
        add_missing_indicators=ADD_MISSING_INDICATORS,
    ).fit(X_tv_raw)

    X_tv = pp_deploy.transform(X_tv_raw).reindex(columns=feature_cols, fill_value=0.0)

    cw_deploy = class_weight_dict_from_y(y_tv)

    clf_deploy = RandomForestClassifier(
        random_state=SEED,
        n_jobs=RF_N_JOBS,
        class_weight=cw_deploy,
        **best_params,
    )
    clf_deploy.fit(X_tv, y_tv)

    # ---------------------------
    # SAVE ARTIFACTS
    # ---------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the deployment model + everything needed to reproduce predictions
    joblib.dump(
        {
            "model": clf_deploy,
            "feature_cols": feature_cols,
            "engineered_features": ENGINEERED_FEATURES,
            "preprocessor": pp_deploy,
            "class_weight": cw_deploy,
        },
        MODEL_FILE,
    )

    # Save Optuna trial results for transparency / reproducibility
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    # Save best parameters
    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": best_params}, f, indent=2)

    # Feature importance (impurity-based / Gini importance)
    importances = getattr(clf_deploy, "feature_importances_", None)
    if importances is None or len(importances) == 0:
        fi = pd.DataFrame({"feature": [], "importance": []})
    else:
        fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
            "importance", ascending=False
        )
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # ---------------------------
    # METRICS + CONFIG LOGGING
    # ---------------------------
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
            "model": "random_forest",
            "n_trials": int(N_TRIALS),
            "timeout_seconds": None if TIMEOUT_SECONDS is None else int(TIMEOUT_SECONDS),
            "optimize_metric": OPTIMIZE_METRIC,
            "best_value_mean_cv": float(study.best_value),
            "best_params": best_params,
            "rf_n_jobs": int(RF_N_JOBS),
            "early_abandon": {
                "enabled": True,
                "auc_bad_cut": float(AUC_BAD_CUT),
                "ap_bad_cut": float(AP_BAD_CUT),
            },
            "pos_neg_final_train": {"pos": pos, "neg": neg},
        },
        "preprocessing": {
            "winsor_p_low": float(WINSOR_P_LOW),
            "winsor_p_high": float(WINSOR_P_HIGH),
            "mean_imputation": True,
            "missing_indicators": bool(ADD_MISSING_INDICATORS),
            "fit_scope": "fit on each split's training data only; applied to train/val/test accordingly",
        },
        "threshold_selection": {
            "optimized_on": "oof_walkforward_folds_last_k",
            "criterion": THRESHOLD_CRITERION,
            "best_threshold": float(thr),
            "best_score_on_final_val": float(thr_score_final_val),
        },
        "final_val_metrics_at_threshold": compute_metrics(y_va, p_val, thr),
        "test_metrics_at_val_threshold": compute_metrics(y_test, p_test, thr),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "year_col": YEAR_COL,
                "test_last_n_years": int(TEST_LAST_N_YEARS),
                "min_train_years": int(MIN_TRAIN_YEARS),
                "val_window": int(VAL_WINDOW),
                "n_folds": None if N_FOLDS is None else int(N_FOLDS),
                "threshold_criterion": THRESHOLD_CRITERION,
                "winsor_p_low": float(WINSOR_P_LOW),
                "winsor_p_high": float(WINSOR_P_HIGH),
                "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
                "imputation": "mean",
                "model": "RandomForestClassifier",
                "class_weight": "computed dict (balanced)",
                "threshold_source": "oof_walkforward_folds_last_k",
                "saved_model_refit_on": "trainval_years (all non-test years)",
                "speed_notes": {
                    "optuna_trials": int(N_TRIALS),
                    "search_space": "tight",
                    "n_jobs": int(RF_N_JOBS),
                    "warm_start_stages": True,
                },
            },
            f,
            indent=2,
        )

    # Simple console output to confirm where artifacts are written
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
