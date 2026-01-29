# train_xgb_roa_improvement_optuna_timeseries_cv.py
# Train an XGBoost binary classifier with proper time-series validation:
# - Keep the most recent years as a true hold-out test set (never touched during tuning)
# - Use walk-forward (rolling) cross-validation on the remaining years for hyperparameter tuning
# - Use Optuna (TPE sampler) to search hyperparameters efficiently
#
# Run:
#   pip install optuna
#   python train_xgb_roa_improvement_optuna_timeseries_cv.py
#
# Outputs (written to task_data/models_optuna_tscv/):
#   xgb_model.json            -> trained XGBoost model (REFIT on all non-test years)
#   metrics.json              -> evaluation metrics on final validation + test
#   config.json               -> run configuration (splits, settings)
#   feature_importance.csv    -> gain-based feature importance (from saved refit model)
#   optuna_trials.csv         -> all Optuna trials (params + scores)
#   best_params.json          -> best hyperparameters + best CV score (+ refit scale_pos_weight)

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
# Project structure: we keep raw/processed data inside task_data/, and write model artifacts there too.
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

# Input files produced by our preprocessing / feature engineering pipeline
RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

# Dedicated output directory for this run (to avoid overwriting other experiments)
MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv"
MODEL_JSON = MODEL_DIR / "xgb_model.json"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
OPTUNA_TRIALS_CSV = MODEL_DIR / "optuna_trials.csv"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"

# ---------- COLUMNS ----------
# Target is a binary label; fyear is the time axis; gvkey identifies firms (panel data)
TARGET_COL = "target"
YEAR_COL = "fyear"
GROUP_COL = "gvkey"

# ---------- SETTINGS ----------
SEED = 42

# If True, add explicit missing-value indicator columns for each feature
# (keep OFF in the original model)
ADD_MISSING_INDICATORS = False

# Test set = last N years (completely untouched by tuning and threshold selection)
TEST_LAST_N_YEARS = 2

# Walk-forward CV settings:
# - We only start validating once we have at least MIN_TRAIN_YEARS of training history.
# - VAL_WINDOW is the number of years per validation fold (1 = classic next-year validation).
# - N_FOLDS optionally limits tuning to the last N folds (closer to the test period).
MIN_TRAIN_YEARS = 4
VAL_WINDOW = 1
N_FOLDS = 3  # None = use all possible folds

# Optuna tuning setup
N_TRIALS = 50
TIMEOUT_SECONDS = None
OPTIMIZE_METRIC = "auc"  # either "auc" (ROC AUC) or "ap" (Average Precision)

# Classification threshold selection is used only for reporting (tuning is threshold-independent)
THRESHOLD_CRITERION = "max_accuracy"  # either "max_f1" or "max_accuracy"

# Feature group definition file: maps group names -> list of engineered feature names
FEATURE_MAP_FILE = "task_data/feature_groups.json"

# Optional: restrict model inputs to selected feature groups (None = use all groups in file)
USE_GROUPS = [
    "Liquidity_&_CashFlow",
    "Leverage_&_CapitalStructure",
    "Profitability_&_Returns",
    "Efficiency_/_Activity",
    "FirmCharacteristics_&_Dynamics",
]


def load_engineered_features(path=FEATURE_MAP_FILE, use_groups=None):
    """
    Load engineered feature names from a JSON file that stores them by thematic groups.
    We build one flat feature list, keep group order, and remove duplicates while preserving order.
    """
    with open(path, "r") as f:
        groups = json.load(f)

    if use_groups is None:
        use_groups = list(groups.keys())

    # Flatten features in the chosen group order
    feats = []
    for g in use_groups:
        feats.extend(groups.get(g, []))

    # De-duplicate while keeping original order
    seen = set()
    feats = [x for x in feats if not (x in seen or seen.add(x))]
    return feats


# This is the final list of input columns used for modeling
ENGINEERED_FEATURES = load_engineered_features(use_groups=USE_GROUPS)


# ---------- HELPERS ----------
def ensure_features_exist(force: bool = False) -> None:
    """
    Make sure the cleaned dataset and engineered features exist.
    If they do not exist (or if force=True), run the pipeline steps that generate them.
    """
    from data_cleanup import clean_data
    from feature_engineering import construct_features

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
    Convert a feature frame to a purely numeric float32 matrix.
    Non-numeric values are coerced to NaN, which XGBoost can handle via 'missing=np.nan'.
    """
    Xn = X.copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn.astype(np.float32)


def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary indicator columns showing whether each original feature is missing.
    This sometimes helps tree models use "missingness patterns" as information.
    """
    miss = X.isna().astype(np.int8)
    miss.columns = [f"{c}__is_missing" for c in miss.columns]
    return pd.concat([X, miss], axis=1)


def predict_proba_booster(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """
    Robust probability prediction for different XGBoost versions/APIs.
    We try to respect early-stopping info (best_iteration / best_ntree_limit) if available.
    """
    pred_params = inspect.signature(booster.predict).parameters
    best_iter = getattr(booster, "best_iteration", None)
    best_ntree = getattr(booster, "best_ntree_limit", None)

    # Newer XGBoost uses iteration_range
    if "iteration_range" in pred_params:
        if best_iter is not None:
            return booster.predict(dmat, iteration_range=(0, int(best_iter) + 1))
        if best_ntree is not None and int(best_ntree) > 0:
            return booster.predict(dmat, iteration_range=(0, int(best_ntree)))
        return booster.predict(dmat)

    # Older XGBoost uses ntree_limit
    if "ntree_limit" in pred_params:
        if best_iter is not None:
            return booster.predict(dmat, ntree_limit=int(best_iter) + 1)
        if best_ntree is not None and int(best_ntree) > 0:
            return booster.predict(dmat, ntree_limit=int(best_ntree))
        return booster.predict(dmat)

    # Fallback: plain predict
    return booster.predict(dmat)


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """
    Compute classification metrics given true labels and predicted probabilities.
    Threshold converts probabilities into class predictions (0/1).

    Robustness:
      - AUC/AP require both classes -> else NaN
      - confusion matrix uses labels=[0,1] to always return tn/fp/fn/tp
      - empty input -> NaNs/0s
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
    Choose a probability threshold by brute-force scanning.
    This is only for reporting (final model tuning uses AUC/AP, which are threshold-free).
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
    Create walk-forward CV folds from the available training/validation years (test years excluded).
    """
    years_trainval = np.array(sorted(years_trainval))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # First validation starts only after we have enough training years
    start = MIN_TRAIN_YEARS
    for i in range(start, len(years_trainval) - VAL_WINDOW + 1):
        train_years = years_trainval[:i]
        val_years = years_trainval[i : i + VAL_WINDOW]
        folds.append((train_years, val_years))

    if not folds:
        raise ValueError("Not enough years for walk-forward CV. Adjust MIN_TRAIN_YEARS/VAL_WINDOW.")

    # Optionally focus tuning on the last N folds (closest to the test period)
    if N_FOLDS is not None and len(folds) > N_FOLDS:
        folds = folds[-N_FOLDS:]

    return folds


def make_Xy(d: pd.DataFrame, drop_cols: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the feature matrix X and target vector y from a dataframe.

    We explicitly check that all engineered features exist, then:
    - select columns
    - coerce to numeric float32
    - optionally add missing indicators (OFF by request)
    """
    missing = [c for c in ENGINEERED_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Missing engineered features in dataframe: {missing}")
    X = d[ENGINEERED_FEATURES].copy()
    y = d[TARGET_COL].astype(int).to_numpy()
    X = to_numeric_matrix(X)
    if ADD_MISSING_INDICATORS:
        X = add_missing_indicators(X)
    return X, y


# ---------- MAIN ----------
def main() -> None:
    """
    End-to-end pipeline:
    1) Ensure features exist
    2) Split by year into train/validation and untouched test
    3) Build walk-forward folds for tuning
    4) Run Optuna to tune XGBoost hyperparameters (mean CV AUC/AP)
    5) Compute OOF threshold on the walk-forward folds (reporting only)
    6) Train final model with early stopping on the last train/val split
    7) Evaluate on the untouched test set (using the OOF threshold)
    8) Refit deployment model on ALL non-test years and save it
    9) Save artifacts (model, metrics, params, feature importance, trials)
    """
    import os
    os.chdir(PROJECT_ROOT)

    ensure_features_exist(force=False)

    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' not found in {FEATURES_PARQUET}.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' not found in {FEATURES_PARQUET}.")

    # Columns to exclude from features to avoid leakage and remove identifiers/meta info
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

    # Determine chronological split by year
    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Not enough total years for the requested split and CV settings.")

    # Hold out the last years as test set (untouched)
    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    # Walk-forward folds for tuning on the train/val range
    folds = build_walkforward_folds(trainval_years)

    # Final training uses all trainval years except the last VAL_WINDOW years,
    # and the last VAL_WINDOW years as validation for early stopping.
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    # Prepare one "reference" trainval dataset to fix the feature column order
    df_trainval = df[df[YEAR_COL].isin(trainval_years)].copy()
    X_all, y_all = make_Xy(df_trainval, drop_cols)
    feature_cols = list(X_all.columns)

    # Prepare test set with identical feature columns/order
    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test, y_test = make_Xy(df_test, drop_cols)
    X_test = X_test.reindex(columns=feature_cols, fill_value=np.nan)

    # Base XGBoost params that remain fixed throughout tuning
    # (scale_pos_weight is computed per fold / per final train / per refit to avoid leakage)
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": SEED,
    }

    # Configure Optuna: log progress and use TPE sampler (Bayesian optimization-style)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        """
        Train XGBoost on one walk-forward fold and return the fold score (AUC or AP).
        Important: scale_pos_weight is computed per fold using only that fold's training data.
        """
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

        # ---- fold-specific class weight computed from fold train only ----
        pos_f = int(df_tr[TARGET_COL].sum())
        neg_f = int((df_tr[TARGET_COL] == 0).sum())
        spw_fold = (neg_f / pos_f) if (pos_f > 0 and neg_f > 0) else 1.0

        fold_params = dict(params)
        fold_params["scale_pos_weight"] = spw_fold
        # ----------------------------------------------------------------

        # Build X/y for train and validation and align feature columns
        X_tr, y_tr = make_Xy(df_tr, drop_cols)
        X_va, y_va = make_Xy(df_va, drop_cols)

        X_tr = X_tr.reindex(columns=feature_cols, fill_value=np.nan)
        X_va = X_va.reindex(columns=feature_cols, fill_value=np.nan)

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
        if OPTIMIZE_METRIC == "auc":
            return float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
        if OPTIMIZE_METRIC == "ap":
            return float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
        raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective:
        - sample hyperparameters
        - score them across all walk-forward folds
        - return mean performance (maximize)
        """
        params = dict(base_params)
        params.update(
            {
                "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            }
        )

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

            X_tr, y_tr = make_Xy(df_tr, drop_cols)
            X_va, y_va = make_Xy(df_va, drop_cols)

            X_tr = X_tr.reindex(columns=feature_cols, fill_value=np.nan)
            X_va = X_va.reindex(columns=feature_cols, fill_value=np.nan)

            pos_f = int(y_tr.sum())
            neg_f = int((y_tr == 0).sum())
            spw_fold = (neg_f / pos_f) if (pos_f > 0 and neg_f > 0) else 1.0

            fold_params = dict(params)
            fold_params["scale_pos_weight"] = spw_fold

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

    # Run hyperparameter search
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    # Assemble best params = base + best hyperparameters found by Optuna
    best_params = dict(base_params)
    best_params.update(study.best_params)

    # Compute OOF threshold (reporting only)
    thr, thr_score, y_oof, p_oof = compute_oof_threshold(best_params)

    # ----- FINAL TRAIN (train=final_train_years, eval=final_val_years) -----
    df_final_train = df[df[YEAR_COL].isin(final_train_years)]
    df_final_val = df[df[YEAR_COL].isin(final_val_years)]

    X_tr, y_tr = make_Xy(df_final_train, drop_cols)
    X_va, y_va = make_Xy(df_final_val, drop_cols)

    X_tr = X_tr.reindex(columns=feature_cols, fill_value=np.nan)
    X_va = X_va.reindex(columns=feature_cols, fill_value=np.nan)

    # scale_pos_weight computed on FINAL TRAIN only
    pos_final = int(y_tr.sum())
    neg_final = int((y_tr == 0).sum())
    spw_final = (neg_final / pos_final) if (pos_final > 0 and neg_final > 0) else 1.0

    final_params = dict(best_params)
    final_params["scale_pos_weight"] = spw_final

    dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
    dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)
    dte = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=feature_cols)

    booster = xgb.train(
        params=final_params,
        dtrain=dtr,
        num_boost_round=8000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Evaluate using the OOF threshold
    p_val = predict_proba_booster(booster, dva)
    p_test = predict_proba_booster(booster, dte)

    # ----- FINAL REFIT (deployment model) on ALL non-test years -----
    df_refit = df[df[YEAR_COL].isin(trainval_years)].copy()
    X_refit, y_refit = make_Xy(df_refit, drop_cols)
    X_refit = X_refit.reindex(columns=feature_cols, fill_value=np.nan)

    pos_refit = int(y_refit.sum())
    neg_refit = int((y_refit == 0).sum())
    spw_refit = (neg_refit / pos_refit) if (pos_refit > 0 and neg_refit > 0) else 1.0

    refit_params = dict(best_params)
    refit_params["scale_pos_weight"] = spw_refit

    drefit = xgb.DMatrix(X_refit, label=y_refit, missing=np.nan, feature_names=feature_cols)

    # Use best_iteration from final model for a consistent number of trees in refit
    best_iter = getattr(booster, "best_iteration", None)
    best_ntree = getattr(booster, "best_ntree_limit", None)
    if best_iter is not None:
        refit_num_round = int(best_iter) + 1
    elif best_ntree is not None and int(best_ntree) > 0:
        refit_num_round = int(best_ntree)
    else:
        refit_num_round = 8000

    booster_refit = xgb.train(
        params=refit_params,
        dtrain=drefit,
        num_boost_round=refit_num_round,
        evals=[(drefit, "train")],
        verbose_eval=False,
    )

    # ---------- SAVE ARTIFACTS ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save REFIT (deployment) model
    booster_refit.save_model(str(MODEL_JSON))

    # Save Optuna trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    # Save best parameters + refit scale_pos_weight
    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": refit_params}, f, indent=2)

    # Feature importance from saved model (refit)
    score = booster_refit.get_score(importance_type="gain")
    if len(score) == 0:
        fi = pd.DataFrame({"feature": [], "importance": []})
    else:
        keys = list(score.keys())
        vals = [float(score[k]) for k in keys]
        if all(k.startswith("f") and k[1:].isdigit() for k in keys):
            idx = np.array([int(k[1:]) for k in keys], dtype=int)
            names = np.array(feature_cols)[idx]
        else:
            names = np.array(keys)
        fi = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    # Metrics and config
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
            "best_params_final_fit": final_params,
            "best_params_refit_saved_model": refit_params,
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

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "year_col": YEAR_COL,
                "drop_cols": sorted([c for c in drop_cols if c in df.columns]),
                "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
                "test_last_n_years": int(TEST_LAST_N_YEARS),
                "min_train_years": int(MIN_TRAIN_YEARS),
                "val_window": int(VAL_WINDOW),
                "n_folds": None if N_FOLDS is None else int(N_FOLDS),
                "threshold_criterion": THRESHOLD_CRITERION,
                "saved_model_fit_scope": "refit_on_trainval_years",
            },
            f,
            indent=2,
        )

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
