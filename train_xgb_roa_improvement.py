# train_xgb_roa_improvement.py
# Run im Projekt-Root:
#   python train_xgb_roa_improvement.py
#
# Outputs:
#   task_data/models/xgb_model.json
#   task_data/models/metrics.json
#   task_data/models/config.json
#   task_data/models/feature_importance.csv

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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

MODEL_DIR = TASK_DATA_DIR / "models"
MODEL_JSON = MODEL_DIR / "xgb_model.json"
METRICS_JSON = MODEL_DIR / "metrics.json"
CONFIG_JSON = MODEL_DIR / "config.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"

TARGET_COL = "target"
YEAR_COL = "fyear"
GROUP_COL = "gvkey"  # nur ID (nicht als Feature)

SEED = 42
EVAL_TEST = True
ADD_MISSING_INDICATORS = False


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


def time_split_by_year(df: pd.DataFrame, year_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = np.sort(df[year_col].dropna().unique())
    if len(years) < 6:
        df_sorted = df.sort_values(year_col)
        n = len(df_sorted)
        i1 = int(0.7 * n)
        i2 = int(0.85 * n)
        return df_sorted.iloc[:i1], df_sorted.iloc[i1:i2], df_sorted.iloc[i2:]

    y1 = years[int(0.7 * len(years)) - 1]
    y2 = years[int(0.85 * len(years)) - 1]

    train = df[df[year_col] <= y1]
    val = df[(df[year_col] > y1) & (df[year_col] <= y2)]
    test = df[df[year_col] > y2]
    return train, val, test


def best_threshold_by_f1(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        yhat = (p >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


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


def main() -> None:
    import os
    os.chdir(PROJECT_ROOT)

    ensure_features_exist(force=False)

    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")

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

    train_df, val_df, test_df = time_split_by_year(df, YEAR_COL)

    def make_xy(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = d.drop(columns=[c for c in drop_cols if c in d.columns])
        y = d[TARGET_COL].astype(int).to_numpy()
        X = to_numeric_matrix(X)
        if ADD_MISSING_INDICATORS:
            X = add_missing_indicators(X)
        return X, y

    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=np.nan)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "seed": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=y_val, missing=np.nan, feature_names=list(X_train.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan, feature_names=list(X_train.columns))

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=6000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    p_val = predict_proba_booster(booster, dval)
    # best_t, best_f1 = best_threshold_by_f1(y_val, p_val) comment this in to go back

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_acc = 0.5, -1.0
    for t in thresholds:
        acc = accuracy_score(y_val, (p_val >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = float(acc), float(t)

    metrics = {
        "research_question": "Predict ROA improvement: y=1 if ROA(t+1)>ROA(t), else 0; using year-t financial ratios",
        "split": {
            "method": "time_split_by_fyear",
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "years_train_minmax": [int(train_df[YEAR_COL].min()), int(train_df[YEAR_COL].max())],
            "years_val_minmax": [int(val_df[YEAR_COL].min()), int(val_df[YEAR_COL].max())] if len(val_df) else None,
            "years_test_minmax": [int(test_df[YEAR_COL].min()), int(test_df[YEAR_COL].max())] if len(test_df) else None,
        },
        "class_balance": {
            "pos_rate_train": float(train_df[TARGET_COL].mean()),
            "pos_rate_val": float(val_df[TARGET_COL].mean()) if len(val_df) else float("nan"),
            "pos_rate_test": float(test_df[TARGET_COL].mean()) if len(test_df) else float("nan"),
            "scale_pos_weight_used": float(scale_pos_weight),
        },
        "missing_handling": {
            "imputation": "none",
            "xgboost_missing_value": "np.nan",
            "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
        },
        "model": {
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
            "params": params,
        }, # comment this in to go back
        """threshold_selection": {
            "optimized_on": "validation",
            "criterion": "max_f1",
            "best_threshold": float(best_t),
            "best_f1_on_val": float(best_f1),
        },
        "val_metrics_at_best_threshold": compute_metrics(y_val, p_val, best_t),"""
        "threshold_selection": {
            "optimized_on": "validation",
            "criterion": "max_accuracy",
            "best_threshold": float(best_t),
            "best_accuracy_on_val": float(best_acc),
        },
        "val_metrics_at_best_threshold": compute_metrics(y_val, p_val, best_t),

    }

    if EVAL_TEST and len(test_df) > 0:
        p_test = predict_proba_booster(booster, dtest)
        metrics["test_metrics_at_val_threshold"] = compute_metrics(y_test, p_test, best_t)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_JSON))

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(best_t),
                "target_col": TARGET_COL,
                "year_col": YEAR_COL,
                "drop_cols": sorted([c for c in drop_cols if c in df.columns]),
                "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
            },
            f,
            indent=2,
        )

    # Feature importance (gain) -> robust: Keys können "f0" ODER echte Feature-Namen sein
    score = booster.get_score(importance_type="gain")

    if len(score) == 0:
        fi = pd.DataFrame({"feature": [], "importance": []})
    else:
        keys = list(score.keys())
        values = [float(score[k]) for k in keys]

        if all(k.startswith("f") and k[1:].isdigit() for k in keys):
            idx = np.array([int(k[1:]) for k in keys], dtype=int)
            names = np.array(X_train.columns)[idx]
        else:
            names = np.array(keys)

        fi = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)

    fi.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print("Saved:")
    print(f"  {MODEL_JSON}")
    print(f"  {METRICS_JSON}")
    print(f"  {CONFIG_JSON}")
    print(f"  {FEATURE_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()


