# train_xgb_roa_improvement_optuna_timeseries_cv.py
# Time-series Cross-Validation (walk-forward) + Optuna (Bayesian/TPE)
#
# Run:
#   pip install optuna
#   python train_xgb_roa_improvement_optuna_timeseries_cv.py
#
# Outputs (in task_data/models_optuna_tscv/):
#   xgb_model.json
#   metrics.json
#   config.json
#   feature_importance.csv
#   optuna_trials.csv
#   best_params.json

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
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

# eigener Output-Ordner (überschreibt NICHT eure anderen Runs)
MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv"
MODEL_JSON = MODEL_DIR / "xgb_model.json"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
OPTUNA_TRIALS_CSV = MODEL_DIR / "optuna_trials.csv"
BEST_PARAMS_JSON = MODEL_DIR / "best_params.json"

# ---------- COLUMNS ----------
TARGET_COL = "target"
YEAR_COL = "fyear"
GROUP_COL = "gvkey"

# ---------- SETTINGS ----------
SEED = 42
ADD_MISSING_INDICATORS = False

# Test-Set: die letzten N Jahre (untouched)
TEST_LAST_N_YEARS = 2

# Walk-forward CV:
MIN_TRAIN_YEARS = 4   # erstes Val-Jahr erst nachdem mindestens so viele Train-Jahre existieren
VAL_WINDOW = 1        # wie viele Jahre pro Val-Fold (1 = klassisch)
N_FOLDS = 3           # wie viele letzte Folds im Train/Val-Bereich fürs Tuning (None = alle möglichen)

# Optuna
N_TRIALS = 50
TIMEOUT_SECONDS = None
OPTIMIZE_METRIC = "auc"   # "auc" oder "ap"

# Threshold (nur fürs Reporting; Modell-Tuning passiert threshold-unabhängig)
THRESHOLD_CRITERION = "max_f1"  # "max_f1" oder "max_accuracy"


# ---------- HELPERS ----------
def ensure_features_exist(force: bool = False) -> None:
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
    Xn = X.copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn.astype(np.float32)


def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    miss = X.isna().astype(np.int8)
    miss.columns = [f"{c}__is_missing" for c in miss.columns]
    return pd.concat([X, miss], axis=1)


def predict_proba_booster(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
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
    yhat = (p >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def best_threshold(y_true: np.ndarray, p: np.ndarray, criterion: str) -> tuple[float, float]:
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
    years_trainval: sortierte Jahre OHNE Testjahre
    Returns: Liste (train_years, val_years) walk-forward
    """
    years_trainval = np.array(sorted(years_trainval))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # Val startet bei Index MIN_TRAIN_YEARS
    start = MIN_TRAIN_YEARS
    for i in range(start, len(years_trainval) - VAL_WINDOW + 1):
        train_years = years_trainval[:i]
        val_years = years_trainval[i : i + VAL_WINDOW]
        folds.append((train_years, val_years))

    if not folds:
        raise ValueError("Zu wenige Jahre für Walk-forward CV. MIN_TRAIN_YEARS/VAL_WINDOW anpassen.")

    # optional: nur die letzten N Folds
    if N_FOLDS is not None and len(folds) > N_FOLDS:
        folds = folds[-N_FOLDS:]

    return folds


def make_Xy(df: pd.DataFrame, drop_cols: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_COL].astype(int).to_numpy()
    X = to_numeric_matrix(X)
    if ADD_MISSING_INDICATORS:
        X = add_missing_indicators(X)
    return X, y


# ---------- MAIN ----------
def main() -> None:
    import os
    os.chdir(PROJECT_ROOT)

    ensure_features_exist(force=False)

    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' nicht in {FEATURES_PARQUET} gefunden.")

    # drop leakage/IDs/meta
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

    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Zu wenige Jahre insgesamt für MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS.")

    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    folds = build_walkforward_folds(trainval_years)

    # Für finalen Fit mit Early Stopping nehmen wir als eval das letzte TrainVal-Jahr (z.B. 2021)
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    # Daten vorbereiten (einmal alle Spalten fixieren)
    df_trainval = df[df[YEAR_COL].isin(trainval_years)].copy()
    X_all, y_all = make_Xy(df_trainval, drop_cols)
    # Spalten sichern
    feature_cols = list(X_all.columns)

    # Test vorbereiten
    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test, y_test = make_Xy(df_test, drop_cols)
    X_test = X_test.reindex(columns=feature_cols, fill_value=np.nan)

    # class weight aus final_train (nicht aus test/val)
    df_final_train = df[df[YEAR_COL].isin(final_train_years)]
    pos = int(df_final_train[TARGET_COL].sum())
    neg = int((df_final_train[TARGET_COL] == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    # Optuna live Fortschritt
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

        X_tr, y_tr = make_Xy(df_tr, drop_cols)
        X_va, y_va = make_Xy(df_va, drop_cols)

        X_tr = X_tr.reindex(columns=feature_cols, fill_value=np.nan)
        X_va = X_va.reindex(columns=feature_cols, fill_value=np.nan)

        dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
        dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)

        booster = xgb.train(
            params=params,
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
        params = dict(base_params)
        params.update(
            {
                "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),  # mind. 0.01
                "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            }
        )

        scores = []
        for tr_years, va_years in folds:
            scores.append(score_fold(params, tr_years, va_years))

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    best_params = dict(base_params)
    best_params.update(study.best_params)

    # ----- FINAL TRAIN (train=final_train_years, eval=final_val_years) -----
    df_final_train = df[df[YEAR_COL].isin(final_train_years)]
    df_final_val = df[df[YEAR_COL].isin(final_val_years)]

    X_tr, y_tr = make_Xy(df_final_train, drop_cols)
    X_va, y_va = make_Xy(df_final_val, drop_cols)

    X_tr = X_tr.reindex(columns=feature_cols, fill_value=np.nan)
    X_va = X_va.reindex(columns=feature_cols, fill_value=np.nan)

    dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_cols)
    dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_cols)
    dte = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=feature_cols)

    booster = xgb.train(
        params=best_params,
        dtrain=dtr,
        num_boost_round=8000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Threshold auf final_val bestimmen (nur fürs Reporting)
    p_val = predict_proba_booster(booster, dva)
    thr, thr_score = best_threshold(y_va, p_val, THRESHOLD_CRITERION)

    p_test = predict_proba_booster(booster, dte)

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_JSON))

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": best_params}, f, indent=2)

    # Feature importance (gain), robust
    score = booster.get_score(importance_type="gain")
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
            "best_params": best_params,
        },
        "threshold_selection": {
            "optimized_on": "final_val_years",
            "criterion": THRESHOLD_CRITERION,
            "best_threshold": float(thr),
            "best_score_on_final_val": float(thr_score),
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
