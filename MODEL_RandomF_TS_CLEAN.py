# train_rf_roa_improvement_optuna_timeseries_cv_fast.py
# Time-series Cross-Validation (walk-forward) + Optuna (TPE) + RandomForest (speed-optimized)
# Same data handling as your XGBoost script:
#  - same parquet inputs
#  - same engineered feature selection (feature_groups.json)
#  - same time split (last N years test)
#  - same walk-forward folds
#  - winsorize + mean impute + missing indicators fit ONLY on train of each split
#
# Speed changes vs a "naive" RF:
#  - tighter search space
#  - fewer trials by default
#  - cap threads (laptop friendly)
#  - warm_start staged fitting + early abandon
#  - BUT: avoid sklearn warning by using fixed class_weight dict from y_tr (compute_class_weight)

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

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parent
TASK_DATA_DIR = PROJECT_ROOT / "task_data"

RAW_CSV = TASK_DATA_DIR / "itaiif_compustat_data_24112025.csv"
CLEANED_PARQUET = TASK_DATA_DIR / "cleaned_data.parquet"
FEATURES_PARQUET = TASK_DATA_DIR / "features.parquet"

MODEL_DIR = TASK_DATA_DIR / "models_optuna_tscv_clean_rf_fast"
MODEL_FILE = MODEL_DIR / "rf_model.joblib"
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

# Requested preprocessing:
ADD_MISSING_INDICATORS = True
WINSOR_P_LOW = 0.01
WINSOR_P_HIGH = 0.99

# Test-Set: die letzten N Jahre (untouched)
TEST_LAST_N_YEARS = 2

# Walk-forward CV:
MIN_TRAIN_YEARS = 4
VAL_WINDOW = 1
N_FOLDS = 3  # None = all (slower)

# Optuna (speed)
N_TRIALS = 25
TIMEOUT_SECONDS = 1800  # 30 min cap; set None if you don't want a cap
OPTIMIZE_METRIC = "auc"  # "auc" oder "ap"

# Threshold (nur fürs Reporting)
THRESHOLD_CRITERION = "max_accuracy"  # "max_f1" oder "max_accuracy"

# CPU usage (laptop-friendly)
RF_N_JOBS = 4

# Early-abandon cutoff after first stage
AUC_BAD_CUT = 0.52
AP_BAD_CUT = 0.05

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


def class_weight_dict_from_y(y: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1], dtype=int)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {0: float(w[0]), 1: float(w[1])}


class SplitPreprocessor:
    """
    Fit ONLY on training data of a split:
      - winsorize per feature at 1% / 99% (ignoring NaNs)
      - mean impute per feature (means computed AFTER winsorization)
      - create missing indicators from original missingness
    Then transform any dataset with the learned params.
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


def main() -> None:
    import os

    os.chdir(PROJECT_ROOT)
    ensure_features_exist(force=False)

    df = pd.read_parquet(FEATURES_PARQUET)
    if TARGET_COL not in df.columns:
        raise KeyError(f"'{TARGET_COL}' nicht in {FEATURES_PARQUET} gefunden.")
    if YEAR_COL not in df.columns:
        raise KeyError(f"'{YEAR_COL}' nicht in {FEATURES_PARQUET} gefunden.")

    years_all = np.array(sorted(df[YEAR_COL].dropna().unique()))
    if len(years_all) < (MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS):
        raise ValueError("Zu wenige Jahre insgesamt für MIN_TRAIN_YEARS + VAL_WINDOW + TEST_LAST_N_YEARS.")

    test_years = years_all[-TEST_LAST_N_YEARS:]
    trainval_years = years_all[:-TEST_LAST_N_YEARS]

    folds = build_walkforward_folds(trainval_years)

    # final split analog zu XGB: eval = letztes TrainVal-Jahr
    final_val_years = trainval_years[-VAL_WINDOW:]
    final_train_years = trainval_years[:-VAL_WINDOW]

    base_feature_cols = list(ENGINEERED_FEATURES)
    feature_cols = (
        base_feature_cols + [f"{c}__is_missing" for c in base_feature_cols]
        if ADD_MISSING_INDICATORS
        else base_feature_cols
    )

    # for logging
    df_final_train_for_stats = df[df[YEAR_COL].isin(final_train_years)]
    pos = int(df_final_train_for_stats[TARGET_COL].sum())
    neg = int((df_final_train_for_stats[TARGET_COL] == 0).sum())

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

        cw = class_weight_dict_from_y(y_tr)

        # staged warm-start: quickly reject bad trials
        n_final = int(params["n_estimators"])
        if n_final >= 350:
            stages = [150, 300, n_final]
        elif n_final > 150:
            stages = [150, n_final]
        else:
            stages = [n_final]

        base = dict(params)
        base.pop("n_estimators", None)

        clf = RandomForestClassifier(
            random_state=SEED,
            n_jobs=RF_N_JOBS,
            class_weight=cw,
            warm_start=True,
            **base,
            n_estimators=stages[0],
        )

        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_va)[:, 1]

        if OPTIMIZE_METRIC == "auc":
            best = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
            bad_cut = AUC_BAD_CUT
        elif OPTIMIZE_METRIC == "ap":
            best = float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0
            bad_cut = AP_BAD_CUT
        else:
            raise ValueError("OPTIMIZE_METRIC must be 'auc' or 'ap'")

        # early abandon if clearly bad after first stage
        if best < bad_cut and len(stages) > 1:
            return best

        for s in stages[1:]:
            clf.set_params(n_estimators=s)
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_va)[:, 1]

            if OPTIMIZE_METRIC == "auc":
                score = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
            else:
                score = float(average_precision_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.0

            if score > best:
                best = score

        return best

    def objective(trial: optuna.Trial) -> float:
        # speed-focused search space; keep depth closer to XGB typical range
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

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

    best_params = dict(study.best_params)

    # ----- FINAL TRAIN (train=final_train_years, eval=final_val_years) -----
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

    cw_final = class_weight_dict_from_y(y_tr)

    clf = RandomForestClassifier(
        random_state=SEED,
        n_jobs=RF_N_JOBS,
        class_weight=cw_final,
        **best_params,
    )
    clf.fit(X_tr, y_tr)

    p_val = clf.predict_proba(X_va)[:, 1]
    thr, thr_score = best_threshold(y_va, p_val, THRESHOLD_CRITERION)

    p_test = clf.predict_proba(X_test)[:, 1]

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feature_cols,
            "engineered_features": ENGINEERED_FEATURES,
            "preprocessor": pp_final,
            "class_weight": cw_final,
        },
        MODEL_FILE,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(OPTUNA_TRIALS_CSV, index=False)

    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": best_params}, f, indent=2)

    # Feature importance (Gini / impurity-based)
    importances = getattr(clf, "feature_importances_", None)
    if importances is None or len(importances) == 0:
        fi = pd.DataFrame({"feature": [], "importance": []})
    else:
        fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
            "importance", ascending=False
        )
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
