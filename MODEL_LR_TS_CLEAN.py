# train_logreg_roa_improvement_optuna_timeseries_cv.py
# Logistic Regression with:
#   - Time-series Cross-Validation (walk-forward)
#   - Optuna (TPE/Bayesian) hyperparameter tuning
#   - Split-wise preprocessing (IDENTICAL to your XGBoost script):
#       winsorize (1%/99%), mean impute, missing indicators
#
# Changes to avoid sklearn warnings you saw:
#   - suppress sklearn FutureWarnings + UserWarnings
#   - removed n_jobs (no effect in sklearn>=1.8)
#   - ensure l1_ratio is ONLY passed when penalty == "elasticnet"
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
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv_logreg_clean"
MODEL_FILE = MODEL_DIR / "logreg_model.joblib"
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

# preprocessing identical to XGB script:
ADD_MISSING_INDICATORS = True
WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

# keep false to match original exactly (no scaling)
STANDARDIZE = False  # unused here, kept just for clarity

TEST_LAST_N_YEARS = 2

MIN_TRAIN_YEARS = 4
VAL_WINDOW = 1
N_FOLDS = 3

N_TRIALS = 50
TIMEOUT_SECONDS = None
OPTIMIZE_METRIC = "auc"  # "auc" oder "ap"

THRESHOLD_CRITERION = "max_accuracy"  # "max_f1" oder "max_accuracy"

FEATURE_MAP_FILE = "task_data/feature_groups.json"

USE_GROUPS = [
    "Liquidity_&_CashFlow",
    "Leverage_&_CapitalStructure",
    "Profitability_&_Returns",
    "Efficiency_/_Activity",
    "FirmCharacteristics_&_Dynamics",
]


def load_engineered_features(path=FEATURE_MAP_FILE, use_groups=None):
    with open(path, "r") as f:
        groups = json.load(f)

    if use_groups is None:
        use_groups = list(groups.keys())

    feats = []
    for g in use_groups:
        feats.extend(groups.get(g, []))

    seen = set()
    feats = [x for x in feats if not (x in seen or seen.add(x))]
    return feats


ENGINEERED_FEATURES = load_engineered_features(use_groups=USE_GROUPS)


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
    missing = [c for c in ENGINEERED_FEATURES if c not in d.columns]
    if missing:
        raise KeyError(f"Missing engineered features in dataframe: {missing}")
    X = d[ENGINEERED_FEATURES].copy()
    y = d[TARGET_COL].astype(int).to_numpy()
    return X, y


class SplitPreprocessor:
    """
    IDENTICAL to your XGB script:
      - winsorize per feature at 1% / 99% (ignoring NaNs)
      - mean impute per feature (means computed AFTER winsorization)
      - create missing indicators from original missingness
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

        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.mean_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "SplitPreprocessor":
        Xn = to_numeric_matrix(X[self.feature_cols])

        lower = Xn.quantile(self.p_low, numeric_only=True).reindex(self.feature_cols)
        upper = Xn.quantile(self.p_high, numeric_only=True).reindex(self.feature_cols)

        lower = lower.fillna(-np.inf)
        upper = upper.fillna(np.inf)

        Xc = Xn.clip(lower=lower, upper=upper, axis=1)

        mean = Xc.mean(axis=0, skipna=True).reindex(self.feature_cols)
        mean = mean.fillna(0.0)

        self.lower_ = lower
        self.upper_ = upper
        self.mean_ = mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.lower_ is None or self.upper_ is None or self.mean_ is None:
            raise RuntimeError("Preprocessor not fitted.")

        Xn = to_numeric_matrix(X[self.feature_cols])

        if self.add_missing_indicators:
            miss = Xn.isna().astype(np.int8)
            miss.columns = [f"{c}__is_missing" for c in self.feature_cols]
        else:
            miss = None

        Xc = Xn.clip(lower=self.lower_, upper=self.upper_, axis=1)
        Xi = Xc.fillna(self.mean_)

        out = pd.concat([Xi, miss], axis=1) if miss is not None else Xi
        return out.astype(np.float32)


def safe_predict_proba(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)


def clean_logreg_params(params: dict) -> dict:
    """
    sklearn>=1.8 warnings happen if l1_ratio is present with penalty != elasticnet.
    This function enforces: l1_ratio ONLY when penalty == "elasticnet".
    """
    p = dict(params)
    if p.get("penalty") != "elasticnet":
        p.pop("l1_ratio", None)
    return p


def main() -> None:
    import os

    os.chdir(PROJECT_ROOT)
    ensure_features_exist(force=False)

    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' nicht in {FEATURES_PARQUET} gefunden.")

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

    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    base_feature_cols = list(ENGINEERED_FEATURES)
    feature_cols = (
        base_feature_cols + [f"{c}__is_missing" for c in base_feature_cols]
        if ADD_MISSING_INDICATORS
        else base_feature_cols
    )

    # XGB-like positive weighting factor computed on final_train_years
    df_final_train_for_weight = df[df[YEAR_COL].isin(final_train_years)]
    pos = int(df_final_train_for_weight[TARGET_COL].sum())
    neg = int((df_final_train_for_weight[TARGET_COL] == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    def make_sample_weight(y: np.ndarray, spw: float) -> np.ndarray:
        w = np.ones_like(y, dtype=np.float64)
        w[y == 1] = float(spw)
        return w

    # removed n_jobs to avoid warning (no effect in sklearn>=1.8)
    base_params = {
        "random_state": SEED,
        "max_iter": 5000,
    }

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    def score_fold(params: dict, train_years: np.ndarray, val_years: np.ndarray) -> float:
        df_tr = df[df[YEAR_COL].isin(train_years)]
        df_va = df[df[YEAR_COL].isin(val_years)]

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

        params_clean = clean_logreg_params(params)
        model = LogisticRegression(**params_clean)

        sw_tr = make_sample_weight(y_tr, scale_pos_weight)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        p = safe_predict_proba(model, X_va)

        if OPTIMIZE_METRIC == "auc":
            return float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
        if OPTIMIZE_METRIC == "ap":
            return float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
        raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

        if penalty == "elasticnet":
            solver = "saga"
        else:
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])

        C = trial.suggest_float("C", 1e-3, 1e2, log=True)

        params = dict(base_params)
        params.update({"penalty": penalty, "solver": solver, "C": C})

        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)

        scores = []
        for tr_years, va_years in folds:
            scores.append(score_fold(params, tr_years, va_years))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    best_params = dict(base_params)
    best_params.update(study.best_params)

    # ensure solver for elasticnet
    if best_params.get("penalty") == "elasticnet":
        best_params["solver"] = "saga"

    best_params_clean = clean_logreg_params(best_params)

    # ----- FINAL TRAIN -----
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

    df_test = df[df[YEAR_COL].isin(test_years)].copy()
    X_test_raw, y_test = make_Xy_raw(df_test)
    X_test = pp_final.transform(X_test_raw).reindex(columns=feature_cols, fill_value=0.0)

    model = LogisticRegression(**best_params_clean)
    sw_tr = make_sample_weight(y_tr, scale_pos_weight)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)

    p_val = safe_predict_proba(model, X_va)
    thr, thr_score = best_threshold(y_va, p_val, THRESHOLD_CRITERION)

    p_test = safe_predict_proba(model, X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "preprocessor": pp_final,
            "feature_cols": feature_cols,
            "engineered_features": ENGINEERED_FEATURES,
            "scale_pos_weight_like": scale_pos_weight,
            "best_params": best_params_clean,
        },
        MODEL_FILE,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": float(study.best_value), "best_params": best_params_clean}, f, indent=2)

    coef = model.coef_.reshape(-1)
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": np.abs(coef).astype(float), "coef": coef.astype(float)}
    ).sort_values("importance", ascending=False)
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
            "standardize": False,
            "fit_scope": "fit on each split's training data only; applied to train/val/test accordingly",
        },
        "class_imbalance_handling": {
            "xgb_like": True,
            "scale_pos_weight": float(scale_pos_weight),
            "implementation": "sample_weight: positives weighted by scale_pos_weight (estimated on final_train_years)",
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
                "test_last_n_years": int(TEST_LAST_N_YEARS),
                "min_train_years": int(MIN_TRAIN_YEARS),
                "val_window": int(VAL_WINDOW),
                "n_folds": None if N_FOLDS is None else int(N_FOLDS),
                "threshold_criterion": THRESHOLD_CRITERION,
                "winsor_p_low": float(WINSOR_P_LOW),
                "winsor_p_high": float(WINSOR_P_HIGH),
                "add_missing_indicators": bool(ADD_MISSING_INDICATORS),
                "imputation": "mean",
                "standardize": False,
                "class_imbalance": {
                    "xgb_scale_pos_weight": float(scale_pos_weight),
                    "logreg_equivalent": "sample_weight positives=scale_pos_weight",
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
