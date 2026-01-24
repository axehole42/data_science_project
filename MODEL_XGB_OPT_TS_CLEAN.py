
# train_xgb_roa_improvement_optuna_timeseries_cv.py
# Time-series Cross-Validation (walk-forward) + Optuna (Bayesian/TPE)
#
# Purpose:
#   Train an XGBoost binary classifier to predict the target (ROA improvement label),
#   using a strict time-series setup:
#     - The last N years are held out as an untouched test set.
#     - Hyperparameters are tuned via walk-forward CV (expanding window).
#     - All preprocessing is fitted ONLY on each split's training portion to avoid leakage.
#
# Run:
#   pip install optuna
#   python train_xgb_roa_improvement_optuna_timeseries_cv.py
#
# Outputs (in task_data/models_optuna_tscv_clean/):
#   xgb_model.json            -> trained final model (XGBoost Booster)
#   metrics.json              -> evaluation metrics on final val and test
#   config.json               -> run configuration + split definition
#   feature_importance.csv    -> feature importance (gain)
#   optuna_trials.csv         -> all Optuna trials and their scores
#   best_params.json          -> best hyperparameters and best CV score

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------- PATHS ----------
# Project structure: assume script is run from its own folder (PROJECT_ROOT).
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

# Raw data and intermediate data products used by our pipeline.
RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

# Dedicated output folder for this run so we do not overwrite other experiments.
MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv_clean"
MODEL_JSON = MODEL_DIR / "xgb_model.json"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
OPTUNA_TRIALS_CSV = MODEL_DIR / "optuna_trials.csv"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"

# ---------- COLUMNS ----------
# Target is a binary classification label.
TARGET_COL = "target"

# Year column used for time-based splitting.
YEAR_COL = "fyear"

# Group key (firm identifier). Not used for modeling directly, but exists in data.
GROUP_COL = "gvkey"

# ---------- SETTINGS ----------
SEED = 42

# Preprocessing choices:
#   - We optionally add missingness indicators per feature.
#   - We winsorize numeric features to reduce the influence of outliers.
ADD_MISSING_INDICATORS = True
WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

# Test-Set: the last N years are kept completely untouched until final evaluation.
TEST_LAST_N_YEARS = 2

# Walk-forward CV configuration:
#   - MIN_TRAIN_YEARS: minimum number of years needed before we start validating.
#   - VAL_WINDOW: how many years per validation fold (1 year = typical walk-forward).
#   - N_FOLDS: optionally only take the last N folds for tuning (focus on recent periods).
MIN_TRAIN_YEARS = 4
VAL_WINDOW = 1
N_FOLDS = 3

# Optuna tuning configuration:
#   - TPE sampler approximates Bayesian optimization over hyperparameters.
N_TRIALS = 50
TIMEOUT_SECONDS = None
OPTIMIZE_METRIC = "auc"   # choose what Optuna maximizes: "auc" or "ap"

# Threshold selection (reporting only):
#   - Model is trained as probabilistic classifier.
#   - We later select a threshold on the final validation years for reporting metrics.
THRESHOLD_CRITERION = "max_accuracy"  # can be "max_f1" or "max_accuracy"

# Feature group mapping file to select engineered features consistently.
FEATURE_MAP_FILE = "task_data/feature_groups.json"

# Optionally restrict to a subset of feature groups.
USE_GROUPS = [
    "Liquidity_&_CashFlow",
    "Leverage_&_CapitalStructure",
    "Profitability_&_Returns",
    "Efficiency_/_Activity",
    "FirmCharacteristics_&_Dynamics",
]


def load_engineered_features(path=FEATURE_MAP_FILE, use_groups=None):
    """
    Load a feature-group mapping from JSON and return a de-duplicated list of features.

    The JSON is expected to look like:
      { "GroupName": ["feat1", "feat2", ...], ... }

    use_groups:
      - None -> use all groups
      - list -> only use selected groups
    """
    with open(path, "r") as f:
        groups = json.load(f)

    if use_groups is None:
        use_groups = list(groups.keys())

    feats = []
    for g in use_groups:
        feats.extend(groups.get(g, []))

    # Remove duplicates while preserving order.
    seen = set()
    feats = [x for x in feats if not (x in seen or seen.add(x))]
    return feats


# Final feature list used by this script (from the JSON mapping).
ENGINEERED_FEATURES = load_engineered_features(use_groups=USE_GROUPS)


# ---------- HELPERS ----------
def ensure_features_exist(force: bool = False) -> None:
    """
    Make sure the cleaned data and engineered features exist.
    If missing (or force=True), run the project's cleaning and feature engineering steps.

    Important:
      - This script assumes those two functions exist in the project:
          data_cleanup.clean_data()
          OLD_feature_engineering.construct_features()
    """
    from data_cleanup import clean_data
    from OLD_feature_engineering import construct_features

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
    Ensure all columns are numeric (coerce non-numeric to NaN) and cast to float32.

    This is defensive:
      - XGBoost expects numeric input.
      - Any unexpected string/object columns are converted.
    """
    Xn = X.copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn.astype(np.float32)


def predict_proba_booster(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """
    Robust probability prediction that works across XGBoost versions.

    Some versions use:
      - iteration_range=(0, best_iteration+1)
    Others use:
      - ntree_limit=best_iteration+1

    We detect which signature is supported and apply early-stopping limits if available.
    """
    pred_params = inspect.signature(booster.predict).parameters
    best_iter = getattr(booster, "best_iteration", None)
    best_ntree = getattr(booster, "best_ntree_limit", None)

    if "iteration_range" in pred_params:
        if best_iter is not None:
            return booster.predict(dmat, iteration_range=(0, int(best_iter) + 1))
        if best_ntree is not None and int(best_ntree) > 0:
            return booster.predict(dmat, iteration_range=(0, int(best_ntree)))
        return booster.predict(dmat)

    if "ntree_limit" in pred_params:
        if best_iter is not None:
            return booster.predict(dmat, ntree_limit=int(best_iter) + 1)
        if best_ntree is not None and int(best_ntree) > 0:
            return booster.predict(dmat, ntree_limit=int(best_ntree))
        return booster.predict(dmat)

    return booster.predict(dmat)


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """
    Compute standard classification metrics from probabilities p using a fixed threshold.

    Notes:
      - AUC/AP require both classes to be present; otherwise we return NaN.
      - Confusion matrix is returned as tn/fp/fn/tp for easier reporting.
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
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def best_threshold(y_true: np.ndarray, p: np.ndarray, criterion: str) -> tuple[float, float]:
    """
    Choose a decision threshold based on a simple grid search on validation probabilities.

    criterion:
      - "max_accuracy": select threshold that maximizes accuracy
      - "max_f1": select threshold that maximizes F1

    We restrict thresholds to [0.05, 0.95] to avoid degenerate extremes.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        yhat = (p >= t).astype(int)
        if criterion == "max_accuracy":
            score = accuracy_score(y_true, yhat) if len(y_true) > 0 else float("nan")
        elif criterion == "max_f1":
            score = f1_score(y_true, yhat, zero_division=0) if len(y_true) > 0 else float("nan")
        else:
            raise ValueError("criterion must be 'max_accuracy' or 'max_f1'")
        if score > best_score:
            best_score, best_t = float(score), float(t)
    return best_t, best_score


def build_walkforward_folds(years_trainval: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build walk-forward (expanding window) folds over the train/validation years.

    Input:
      years_trainval: sorted years excluding the final test years.

    Output:
      list of (train_years, val_years) pairs.

    Example with VAL_WINDOW=1:
      Train: [y1..y4], Val: [y5]
      Train: [y1..y5], Val: [y6]
      ...

    We optionally keep only the last N_FOLDS folds to tune on recent history.
    """
    years_trainval = np.array(sorted(years_trainval))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # First validation year occurs only after we have at least MIN_TRAIN_YEARS in training.
    start = MIN_TRAIN_YEARS
    for i in range(start, len(years_trainval) - VAL_WINDOW + 1):
        train_years = years_trainval[:i]
        val_years = years_trainval[i : i + VAL_WINDOW]
        folds.append((train_years, val_years))

    if not folds:
        raise ValueError("Zu wenige Jahre für Walk-forward CV. MIN_TRAIN_YEARS/VAL_WINDOW anpassen.")

    # Keep only the most recent folds for tuning if requested.
    if N_FOLDS is not None and len(folds) > N_FOLDS:
        folds = folds[-N_FOLDS:]

    return folds


def make_Xy_raw(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract raw feature matrix X and label vector y from a dataframe slice.

    We enforce that all engineered features exist (otherwise feature engineering is inconsistent).
    """
    missing = [c for c in ENGINEERED_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Missing engineered features in dataframe: {missing}")
    X = d[ENGINEERED_FEATURES].copy()
    y = d[TARGET_COL].astype(int).to_numpy()
    return X, y


class SplitPreprocessor:
    """
    Preprocessing that is trained ONLY on the training subset of a split (no leakage).

    Steps in fit():
      1) winsorize each feature by clipping to [p_low, p_high] quantiles
      2) compute mean per feature after winsorization (for imputation)

    Steps in transform():
      1) convert to numeric float32
      2) optionally create missingness indicators from original NaNs
      3) clip with the learned quantile bounds
      4) impute missing values with learned means
      5) return final float32 dataframe (optionally with indicator columns appended)

    This keeps preprocessing consistent across train/val/test, and ensures that
    validation/test information never influences preprocessing parameters.
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

        # Learned parameters (per feature).
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.mean_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "SplitPreprocessor":
        # Convert to numeric (coerce errors to NaN) and keep fixed column order.
        Xn = to_numeric_matrix(X[self.feature_cols])

        # Quantile bounds for winsorization (computed on training data only).
        lower = Xn.quantile(self.p_low, numeric_only=True).reindex(self.feature_cols)
        upper = Xn.quantile(self.p_high, numeric_only=True).reindex(self.feature_cols)

        # If a column is all-NaN in train, quantiles become NaN.
        # In that case we disable clipping by using +/- infinity.
        lower = lower.fillna(-np.inf)
        upper = upper.fillna(np.inf)

        # Clip outliers.
        Xc = Xn.clip(lower=lower, upper=upper, axis=1)

        # Mean imputation values (computed after winsorization).
        mean = Xc.mean(axis=0, skipna=True).reindex(self.feature_cols)
        mean = mean.fillna(0.0)  # fallback if column is entirely NaN in train

        self.lower_ = lower
        self.upper_ = upper
        self.mean_ = mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.lower_ is None or self.upper_ is None or self.mean_ is None:
            raise RuntimeError("Preprocessor not fitted.")

        Xn = to_numeric_matrix(X[self.feature_cols])

        # Missingness indicators are based on original missing values (before imputation).
        if self.add_missing_indicators:
            miss = Xn.isna().astype(np.int8)
            miss.columns = [f"{c}__is_missing" for c in self.feature_cols]
        else:
            miss = None

        # Apply clipping and imputation using parameters learned on training split.
        Xc = Xn.clip(lower=self.lower_, upper=self.upper_, axis=1)
        Xi = Xc.fillna(self.mean_)

        # Append missing indicators if enabled.
        out = pd.concat([Xi, miss], axis=1) if miss is not None else Xi
        return out.astype(np.float32)


# ---------- MAIN ----------
def main() -> None:
    import os

    # Make relative paths consistent (assume running from project root).
    os.chdir(PROJECT_ROOT)

    # Ensure cleaned + engineered feature datasets exist.
    ensure_features_exist(force=False)

    # Load the engineered feature dataset.
    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' nicht in {FEATURES_PARQUET} gefunden.")

    # Columns we do NOT feed into the model (target/IDs/meta/leakage-prone fields).
    # We keep them conceptually for bookkeeping, but features used for training are ENGINEERED_FEATURES only.
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

    # Determine all available years and create a strict time split:
    #   - last TEST_LAST_N_YEARS years -> test set
    #   - remaining years -> train/validation pool
    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Zu wenige Jahre insgesamt für MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS.")

    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    # Build walk-forward folds only within train/validation years (test remains untouched).
    folds = build_walkforward_folds(trainval_years)

    # For the final model fit we reserve the last validation window from trainval as the early-stopping set.
    # This is separate from the untouched test years.
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    # Fix feature column order explicitly.
    # If missing indicators are enabled, we append one indicator column per base feature.
    base_feature_cols = list(ENGINEERED_FEATURES)
    feature_cols = (
        base_feature_cols + [f"{c}__is_missing" for c in base_feature_cols]
        if ADD_MISSING_INDICATORS
        else base_feature_cols
    )

    # Base XGBoost parameters shared across trials.
    # Note: scale_pos_weight is NOT fixed here; we compute it per fold from that fold's training labels
    # to account for class imbalance in each time period.
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": SEED,
        # scale_pos_weight is added inside score_fold based on the fold's training data
    }

    # Enable Optuna logging and set up the sampler.
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        """
        Train an XGBoost model on a single walk-forward fold and return the fold score.

        Key leakage control:
          - Preprocessor is fit on df_tr only and applied to df_va.
          - scale_pos_weight is computed from y_tr only (fold-specific imbalance).
        """
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

        X_tr_raw, y_tr = make_Xy_raw(df_tr)
        X_va_raw, y_va = make_Xy_raw(df_va)

        # Compute class imbalance weight from THIS fold's training data only.
        pos = int(y_tr.sum())
        neg = int((y_tr == 0).sum())
        scale_pos_weight = (neg / pos) if (pos > 0 and neg > 0) else 1.0

        # Copy params and insert fold-specific scale_pos_weight.
        fold_params = dict(params)
        fold_params["scale_pos_weight"] = scale_pos_weight

        # Fit preprocessing on training fold only.
        pp = SplitPreprocessor(
            feature_cols=ENGINEERED_FEATURES,
            p_low=WINSOR_P_LOW,
            p_high=WINSOR_P_HIGH,
            add_missing_indicators=ADD_MISSING_INDICATORS,
        ).fit(X_tr_raw)

        # Transform train/val and align to the fixed feature order (defensive reindex).
        X_tr = pp.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
        X_va = pp.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

        # Create XGBoost DMatrix objects.
        dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
        dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)

        # Train with early stopping on validation fold.
        booster = xgb.train(
            params=fold_params,
            dtrain=dtr,
            num_boost_round=8000,
            evals=[(dva, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Predict probabilities and compute the chosen fold score.
        p = predict_proba_booster(booster, dva)
        if OPTIMIZE_METRIC == "auc":
            return float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
        if OPTIMIZE_METRIC == "ap":
            return float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
        raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective:
          - Sample hyperparameters
          - Evaluate average score across walk-forward folds
          - Return mean fold score for maximization
        """
        params = dict(base_params)
        params.update(
            {
                # Learning rate (eta): log-scale search, small values for smoother training.
                "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),

                # Tree complexity: depth controls model capacity.
                "max_depth": trial.suggest_int("max_depth", 2, 8),

                # Regularization on leaf weights (helps prevent overfitting).
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),

                # Row and column subsampling for regularization.
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

                # L2 and L1 regularization.
                "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),

                # Minimum loss reduction required to make a split (controls conservativeness).
                "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            }
        )

        # Evaluate params on each fold and average.
        scores = []
        for tr_years, va_years in folds:
            scores.append(score_fold(params, tr_years, va_years))

        return float(np.mean(scores))

    def compute_oof_threshold(params: dict) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute a single out-of-fold (OOF) threshold using the same walk-forward folds
        used in Optuna (respecting N_FOLDS). Returns (thr, thr_score, y_oof, p_oof).
        """
        y_oof_list = []
        p_oof_list = []

        for tr_years, va_years in folds:
            df_tr = df[df[YEAR_COL].isin(tr_years)]
            df_va = df[df[YEAR_COL].isin(va_years)]

            X_tr_raw, y_tr = make_Xy_raw(df_tr)
            X_va_raw, y_va = make_Xy_raw(df_va)

            # Compute class imbalance weight from THIS fold's training data only.
            pos = int(y_tr.sum())
            neg = int((y_tr == 0).sum())
            scale_pos_weight = (neg / pos) if (pos > 0 and neg > 0) else 1.0

            fold_params = dict(params)
            fold_params["scale_pos_weight"] = scale_pos_weight

            # Fit preprocessing on training fold only.
            pp = SplitPreprocessor(
                feature_cols=ENGINEERED_FEATURES,
                p_low=WINSOR_P_LOW,
                p_high=WINSOR_P_HIGH,
                add_missing_indicators=ADD_MISSING_INDICATORS,
            ).fit(X_tr_raw)

            X_tr = pp.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
            X_va = pp.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

            dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
            dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)

            booster = xgb.train(
                params=fold_params,
                dtrain=dtr,
                num_boost_round=8000,
                evals=[(dva, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            p = predict_proba_booster(booster, dva)
            y_oof_list.append(y_va)
            p_oof_list.append(p)

        y_oof = np.concatenate(y_oof_list) if len(y_oof_list) > 0 else np.array([], dtype=int)
        p_oof = np.concatenate(p_oof_list) if len(p_oof_list) > 0 else np.array([], dtype=float)

        if len(y_oof) == 0:
            return 0.5, float("nan"), y_oof, p_oof

        thr, thr_score = best_threshold(y_oof, p_oof, THRESHOLD_CRITERION)
        return thr, thr_score, y_oof, p_oof

    # Create and run the Optuna study.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    # Construct final best parameter dict from base params + best Optuna trial params.
    best_params = dict(base_params)
    best_params.update(study.best_params)

    # Compute single OOF threshold using the same walk-forward folds as Optuna (respecting N_FOLDS).
    thr, thr_score, y_oof, p_oof = compute_oof_threshold(best_params)

    # ----- FINAL TRAIN (train=final_train_years, eval=final_val_years) -----
    # After tuning we train a final model:
    #   - training on final_train_years
    #   - early stopping using final_val_years
    # Test years remain completely untouched until evaluation.
    df_final_train = df[df[YEAR_COL].isin(final_train_years)]
    df_final_val = df[df[YEAR_COL].isin(final_val_years)]

    X_tr_raw, y_tr = make_Xy_raw(df_final_train)
    X_va_raw, y_va = make_Xy_raw(df_final_val)

    # Compute scale_pos_weight on FINAL TRAIN only (consistent with CV leakage control).
    pos_final = int(y_tr.sum())
    neg_final = int((y_tr == 0).sum())
    scale_pos_weight_final = (neg_final / pos_final) if (pos_final > 0 and neg_final > 0) else 1.0
    best_params = dict(best_params)
    best_params["scale_pos_weight"] = scale_pos_weight_final

    # Fit preprocessing on final training set only.
    pp_final = SplitPreprocessor(
        feature_cols=ENGINEERED_FEATURES,
        p_low=WINSOR_P_LOW,
        p_high=WINSOR_P_HIGH,
        add_missing_indicators=ADD_MISSING_INDICATORS,
    ).fit(X_tr_raw)

    # Transform final train/val sets and align feature order.
    X_tr = pp_final.transform(X_tr_raw).reindex(columns=feature_cols, fill_value=0.0)
    X_va = pp_final.transform(X_va_raw).reindex(columns=feature_cols, fill_value=0.0)

    # Prepare test set (transformed using preprocessing fitted on final train only).
    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test_raw, y_test = make_Xy_raw(df_test)
    X_test = pp_final.transform(X_test_raw).reindex(columns=feature_cols, fill_value=0.0)

    # Build DMatrices.
    dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
    dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)
    dte = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=feature_cols)

    # Train final model with early stopping (monitoring both train and val).
    booster = xgb.train(
        params=best_params,
        dtrain=dtr,
        num_boost_round=8000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Evaluate on final_val and test using the single OOF threshold.
    p_val = predict_proba_booster(booster, dva)
    p_test = predict_proba_booster(booster, dte)

    # ----- FINAL REFIT (deployment model) on ALL non-test years -----
    # Refit FINAL saved model on ALL trainval years (all years except last TEST_LAST_N_YEARS).
    df_refit = df[df[YEAR_COL].isin(trainval_years)]
    X_refit_raw, y_refit = make_Xy_raw(df_refit)

    # Compute scale_pos_weight on refit data (trainval years).
    pos_refit = int(y_refit.sum())
    neg_refit = int((y_refit == 0).sum())
    scale_pos_weight_refit = (neg_refit / pos_refit) if (pos_refit > 0 and neg_refit > 0) else 1.0

    refit_params = dict(best_params)
    refit_params["scale_pos_weight"] = scale_pos_weight_refit

    # Fit preprocessing on refit (trainval) only.
    pp_refit = SplitPreprocessor(
        feature_cols=ENGINEERED_FEATURES,
        p_low=WINSOR_P_LOW,
        p_high=WINSOR_P_HIGH,
        add_missing_indicators=ADD_MISSING_INDICATORS,
    ).fit(X_refit_raw)

    X_refit = pp_refit.transform(X_refit_raw).reindex(columns=feature_cols, fill_value=0.0)
    drefit = xgb.DMatrix(X_refit, label=y_refit, missing=np.nan, feature_names=feature_cols)

    booster_refit = xgb.train(
        params=refit_params,
        dtrain=drefit,
        num_boost_round=getattr(booster, "best_iteration", None) + 1 if getattr(booster, "best_iteration", None) is not None else getattr(booster, "best_ntree_limit", None) if getattr(booster, "best_ntree_limit", None) is not None else 8000,
        evals=[(drefit, "train")],
        verbose_eval=False,
    )

    # ---------- SAVE ARTIFACTS ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model in XGBoost native format.
    booster_refit.save_model(str(MODEL_JSON))

    # Save all Optuna trials for transparency / reproducibility.
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    # Save best params including fold-aware scale_pos_weight used in final fit.
    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": refit_params}, f, indent=2)

    # ---------- FEATURE IMPORTANCE ----------
    # CHANGE (only): use booster_refit (the actually saved model) for feature importance
    score = booster_refit.get_score(importance_type="gain")
    if len(score) == 0:
        fi = pd.DataFrame({"feature": [], "importance": []})
    else:
        keys = list(score.keys())
        vals = [float(score[k]) for k in keys]
        # If keys are f0, f1, ... map them back to our explicit feature names.
        if all(k.startswith("f") and k[1:].isdigit() for k in keys):
            idx = np.array([int(k[1:]) for k in keys], dtype=int)
            names = np.array(feature_cols)[idx]
        else:
            names = np.array(keys)
        fi = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # ---------- METRICS + CONFIG ----------
    # Store detailed split info (years/folds), tuning info, preprocessing info, and evaluation metrics.
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
            "n_trials": int(N_TRIALS),
            "optimize_metric": OPTIMIZE_METRIC,
            "best_value_mean_cv": float(study.best_value),
            "best_params_final_fit": best_params,
        },
        "preprocessing": {
            "winsor_p_low": float(WINSOR_P_LOW),
            "winsor_p_high": float(WINSOR_P_HIGH),
            "mean_imputation": True,
            "missing_indicators": bool(ADD_MISSING_INDICATORS),
            "fit_scope": "fit on each split's training data only; applied to train/val/test accordingly",
        },
        "threshold_selection": {
            "optimized_on": "oof_walkforward_folds",
            "criterion": THRESHOLD_CRITERION,
            "best_threshold": float(thr),
            "best_score_on_oof": float(thr_score),
        },
        "final_val_metrics_at_threshold": compute_metrics(y_va, p_val, thr),
        "test_metrics_at_val_threshold": compute_metrics(y_test, p_test, thr),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Store a compact configuration file that captures the main run settings.
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
                "saved_model_fit_scope": "refit_on_trainval_years",
            },
            f,
            indent=2,
        )

    # Console output for quick verification of produced files.
    print("Saved:")
    print(f"  {MODEL_DIR}")
    print(f"  {MODEL_JSON}")
    print(f"  {METRICS_JSON}")
    print(f"  {CONFIG_JSON}")
    print(f"  {FEATURE_IMPORTANCE_CSV}")
    print(f"  {OPTUNA_TRIALS_CSV}")
    print(f"  {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()

