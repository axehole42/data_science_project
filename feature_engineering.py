import pandas as pd
import numpy as np
import os
import json

INPUT_FILE = "task_data/cleaned_data.parquet"
OUTPUT_FILE = "task_data/features.parquet"
FEATURE_MAP_FILE = "task_data/feature_groups.json"

# ------------------------------------------------------------
# Helpers (more robust, without changing the feature definitions)
# ------------------------------------------------------------
# We track lag columns separately:
# - helper_lag_cols: lag columns used only as intermediate steps; we will drop them later
# - feature_lag_cols: lag columns that we intentionally keep as model features
helper_lag_cols = []
feature_lag_cols = []

def as_series(x, index=None):
    """
    Ensure we always operate on a pandas Series (to preserve index alignment).
    If x is already a Series, return it. Otherwise convert it to a Series.
    """
    if isinstance(x, pd.Series):
        return x
    if index is None:
        return pd.Series(x)
    return pd.Series(x, index=index)

def safe_div(num, den):
    """
    Robust, index-preserving division.
    - Replaces 0 in the denominator with NaN to avoid division-by-zero.
    - Divides elementwise and keeps pandas index alignment.
    """
    num_s = as_series(num)
    den_s = as_series(den, index=num_s.index)
    den_s = den_s.replace(0, np.nan)
    return num_s / den_s

def safe_log(x):
    """
    Index-preserving logarithm.
    - Computes log only for x > 0.
    - For non-positive values, returns NaN.
    """
    x_s = as_series(x)
    out = pd.Series(np.nan, index=x_s.index)
    m = x_s > 0
    out.loc[m] = np.log(x_s.loc[m])
    return out

def avg_if_contiguous(curr, lag1, lag_mask):
    """
    Average of current and t-1 only if the time series is contiguous (t-1 exists).
    Otherwise returns NaN (to avoid mixing values across time gaps).
    """
    curr_s = as_series(curr)
    lag1_s = as_series(lag1, index=curr_s.index)
    mask_s = as_series(lag_mask, index=curr_s.index).astype(bool)

    out = pd.Series(np.nan, index=curr_s.index)
    out.loc[mask_s] = (curr_s.loc[mask_s] + lag1_s.loc[mask_s]) / 2.0
    return out

def growth(curr, lag1, lag_mask):
    """
    Growth rate (curr - lag1) / lag1, only if t-1 is contiguous.
    Otherwise returns NaN.
    """
    curr_s = as_series(curr)
    lag1_s = as_series(lag1, index=curr_s.index)
    mask_s = as_series(lag_mask, index=curr_s.index).astype(bool)

    out = pd.Series(np.nan, index=curr_s.index)
    out.loc[mask_s] = safe_div(curr_s.loc[mask_s] - lag1_s.loc[mask_s], lag1_s.loc[mask_s])
    return out

def delta(curr, lag1, lag_mask):
    """
    Difference (curr - lag1), only if t-1 is contiguous.
    Otherwise returns NaN.
    """
    curr_s = as_series(curr)
    lag1_s = as_series(lag1, index=curr_s.index)
    mask_s = as_series(lag_mask, index=curr_s.index).astype(bool)

    out = pd.Series(np.nan, index=curr_s.index)
    out.loc[mask_s] = curr_s.loc[mask_s] - lag1_s.loc[mask_s]
    return out

def require_cols(df, cols):
    """
    Convenience check: return True only if all columns are present in df.
    This makes the feature construction robust to missing Compustat items.
    """
    return all(c in df.columns for c in cols)

def _append_unique(lst, item):
    """
    Append to a list only if the item is not already present.
    This helps in notebook/repeated runs and avoids duplicates.
    """
    if item not in lst:
        lst.append(item)

def add_lag(df, col, group="gvkey", suffix="_lag1", keep=False):
    """
    Create a within-firm lag-1 column for `col` (grouped by gvkey).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Column name to lag.
    group : str
        Grouping key (default: gvkey).
    suffix : str
        Suffix for the lagged column name (default: _lag1).
    keep : bool
        If True, this lag column is a real feature and should NOT be dropped.
        If False, it is a helper lag used only for intermediate computations.
    """
    if col in df.columns:
        newc = f"{col}{suffix}"
        df[newc] = df.groupby(group)[col].shift(1)
        if keep:
            _append_unique(feature_lag_cols, newc)
        else:
            _append_unique(helper_lag_cols, newc)

# ---------------------------
# Main feature construction
# ---------------------------
def construct_features():
    print("==========================================")
    print("      FEATURE ENGINEERING (ROA TARGET)    ")
    print("==========================================")

    # Important for notebook / repeated runs: reset lag trackers
    helper_lag_cols.clear()
    feature_lag_cols.clear()

    # Basic I/O check
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Load cleaned panel data (firm-year level)
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    # Normalize column names to lowercase (robust against case differences)
    df.columns = [c.lower() for c in df.columns]

    # Sort by firm and fiscal year for time-series operations
    df.sort_values(by=["gvkey", "fyear"], inplace=True)

    # Create a strict contiguity mask: True only if previous row is exactly last year (t-1 exists)
    df["fyear_lag1"] = df.groupby("gvkey")["fyear"].shift(1)
    lag_mask = (df["fyear_lag1"] == df["fyear"] - 1)

    # Create lag columns only for variables used in subsequent features (as helpers by default)
    lag_cols = [
        "at", "act", "lct", "invt", "rect", "ap", "ppent", "lt", "seq",
        "niadj", "oancf", "capx", "dltt", "dlc", "oibdp", "ebit"
    ]
    for c in lag_cols:
        add_lag(df, c, keep=False)

    # Create averages of balance-sheet variables (only if t-1 is contiguous)
    if require_cols(df, ["at", "at_lag1"]):
        df["avg_at"] = avg_if_contiguous(df["at"], df["at_lag1"], lag_mask)
    if require_cols(df, ["act", "act_lag1"]):
        df["avg_act"] = avg_if_contiguous(df["act"], df["act_lag1"], lag_mask)
    if require_cols(df, ["lct", "lct_lag1"]):
        df["avg_lct"] = avg_if_contiguous(df["lct"], df["lct_lag1"], lag_mask)
    if require_cols(df, ["invt", "invt_lag1"]):
        df["avg_invt"] = avg_if_contiguous(df["invt"], df["invt_lag1"], lag_mask)
    if require_cols(df, ["rect", "rect_lag1"]):
        df["avg_rect"] = avg_if_contiguous(df["rect"], df["rect_lag1"], lag_mask)
    if require_cols(df, ["ap", "ap_lag1"]):
        df["avg_ap"] = avg_if_contiguous(df["ap"], df["ap_lag1"], lag_mask)
    if require_cols(df, ["ppent", "ppent_lag1"]):
        df["avg_ppent"] = avg_if_contiguous(df["ppent"], df["ppent_lag1"], lag_mask)
    if require_cols(df, ["seq", "seq_lag1"]):
        df["avg_seq"] = avg_if_contiguous(df["seq"], df["seq_lag1"], lag_mask)
    if require_cols(df, ["lt", "lt_lag1"]):
        df["avg_lt"] = avg_if_contiguous(df["lt"], df["lt_lag1"], lag_mask)

    # ------------------------------------------------------------
    # Feature groups: we store which features belong to which category
    # (useful for analysis, reporting, and model interpretation)
    # ------------------------------------------------------------
    feature_groups = {
        "Liquidity_&_CashFlow": [],
        "Leverage_&_CapitalStructure": [],
        "Profitability_&_Returns": [],
        "Efficiency_/_Activity": [],
        "FirmCharacteristics_&_Dynamics": [],
    }

    print("Constructing ratios (only calculable with available columns)...")

    # ============================================================
    # 1) Liquidity & Cash Flow
    # ============================================================
    if require_cols(df, ["act", "lct"]):
        df["current_ratio"] = safe_div(df["act"], df["lct"])
        feature_groups["Liquidity_&_CashFlow"].append("current_ratio")

    if require_cols(df, ["act", "lct", "invt"]):
        # Quick ratio excludes inventories (less liquid)
        df["quick_ratio"] = safe_div(df["act"] - df["invt"], df["lct"])
        feature_groups["Liquidity_&_CashFlow"].append("quick_ratio")

    if require_cols(df, ["che", "lct"]):
        df["cash_ratio"] = safe_div(df["che"], df["lct"])
        feature_groups["Liquidity_&_CashFlow"].append("cash_ratio")

    if require_cols(df, ["act", "lct", "at"]):
        df["working_cap_to_assets"] = safe_div(df["act"] - df["lct"], df["at"])
        feature_groups["Liquidity_&_CashFlow"].append("working_cap_to_assets")

    if require_cols(df, ["oancf", "lct"]):
        df["cfo_to_current_liab"] = safe_div(df["oancf"], df["lct"])
        feature_groups["Liquidity_&_CashFlow"].append("cfo_to_current_liab")

    if require_cols(df, ["oancf", "lt"]):
        df["cfo_to_total_liab"] = safe_div(df["oancf"], df["lt"])
        feature_groups["Liquidity_&_CashFlow"].append("cfo_to_total_liab")

    # total_debt = long-term debt (dltt) + short-term debt (dlc), if both exist
    has_debt = require_cols(df, ["dltt", "dlc"])
    if has_debt:
        df["total_debt"] = df["dltt"].fillna(0) + df["dlc"].fillna(0)
    else:
        df["total_debt"] = np.nan

    if has_debt and "oancf" in df.columns:
        df["cfo_to_total_debt"] = safe_div(df["oancf"], df["total_debt"])
        feature_groups["Liquidity_&_CashFlow"].append("cfo_to_total_debt")

    if require_cols(df, ["oancf", "at"]):
        df["cfo_to_assets"] = safe_div(df["oancf"], df["at"])
        feature_groups["Liquidity_&_CashFlow"].append("cfo_to_assets")

    # Free cash flow proxy: CFO - CapEx
    if require_cols(df, ["oancf", "capx"]):
        df["fcf"] = df["oancf"] - df["capx"]
    else:
        df["fcf"] = np.nan

    if require_cols(df, ["fcf", "at"]):
        df["fcf_to_assets"] = safe_div(df["fcf"], df["at"])
        feature_groups["Liquidity_&_CashFlow"].append("fcf_to_assets")

    if "fcf" in df.columns and has_debt:
        df["fcf_to_debt"] = safe_div(df["fcf"], df["total_debt"])
        feature_groups["Liquidity_&_CashFlow"].append("fcf_to_debt")

    # ============================================================
    # 2) Leverage & Capital Structure
    # ============================================================
    if require_cols(df, ["dltt", "dlc", "at"]):
        df["debt_to_assets"] = safe_div(df["dltt"].fillna(0) + df["dlc"].fillna(0), df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("debt_to_assets")

    if require_cols(df, ["lt", "seq"]):
        df["debt_to_equity"] = safe_div(df["lt"], df["seq"])
        feature_groups["Leverage_&_CapitalStructure"].append("debt_to_equity")

    if require_cols(df, ["lt", "at"]):
        df["total_liab_to_assets"] = safe_div(df["lt"], df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("total_liab_to_assets")

    if require_cols(df, ["seq", "at"]):
        df["equity_to_assets"] = safe_div(df["seq"], df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("equity_to_assets")

    if require_cols(df, ["seq", "lt"]):
        df["equity_to_liab"] = safe_div(df["seq"], df["lt"])
        feature_groups["Leverage_&_CapitalStructure"].append("equity_to_liab")

    if require_cols(df, ["dltt", "at"]):
        df["lt_debt_to_assets"] = safe_div(df["dltt"].fillna(0), df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("lt_debt_to_assets")

    if require_cols(df, ["dlc", "at"]):
        df["st_debt_to_assets"] = safe_div(df["dlc"].fillna(0), df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("st_debt_to_assets")

    if require_cols(df, ["lct", "at"]):
        df["current_liab_to_assets"] = safe_div(df["lct"], df["at"])
        feature_groups["Leverage_&_CapitalStructure"].append("current_liab_to_assets")

    # Interest coverage ratios (require interest expense xint)
    if require_cols(df, ["ebit", "xint"]):
        df["interest_coverage_ebit"] = safe_div(df["ebit"], df["xint"])
        feature_groups["Leverage_&_CapitalStructure"].append("interest_coverage_ebit")

    if require_cols(df, ["oibdp", "xint"]):
        df["interest_coverage_ebitda"] = safe_div(df["oibdp"], df["xint"])
        feature_groups["Leverage_&_CapitalStructure"].append("interest_coverage_ebitda")

    if require_cols(df, ["ppent", "dltt"]):
        df["fixed_assets_to_lt_debt"] = safe_div(df["ppent"], df["dltt"])
        feature_groups["Leverage_&_CapitalStructure"].append("fixed_assets_to_lt_debt")

    # ============================================================
    # 3) Profitability & Returns
    # ============================================================
    if require_cols(df, ["niadj", "at"]):
        # ROA based on adjusted net income
        df["roa"] = safe_div(df["niadj"], df["at"])
        feature_groups["Profitability_&_Returns"].append("roa")

    if require_cols(df, ["niadj", "avg_at"]):
        # ROA using average assets (more standard in accounting)
        df["roa_avg_assets"] = safe_div(df["niadj"], df["avg_at"])
        feature_groups["Profitability_&_Returns"].append("roa_avg_assets")

    if require_cols(df, ["niadj", "seq"]):
        df["roe"] = safe_div(df["niadj"], df["seq"])
        feature_groups["Profitability_&_Returns"].append("roe")

    if require_cols(df, ["niadj", "avg_seq"]):
        df["roe_avg_equity"] = safe_div(df["niadj"], df["avg_seq"])
        feature_groups["Profitability_&_Returns"].append("roe_avg_equity")

    if require_cols(df, ["ebit", "at", "lct"]):
        # Simple ROIC proxy: EBIT / (Assets - Current liabilities)
        df["roic_proxy"] = safe_div(df["ebit"], (df["at"] - df["lct"]))
        feature_groups["Profitability_&_Returns"].append("roic_proxy")

    if require_cols(df, ["niadj", "ppent"]):
        df["return_on_fixed_assets"] = safe_div(df["niadj"], df["ppent"])
        feature_groups["Profitability_&_Returns"].append("return_on_fixed_assets")

    if require_cols(df, ["niadj", "act"]):
        df["return_on_current_assets"] = safe_div(df["niadj"], df["act"])
        feature_groups["Profitability_&_Returns"].append("return_on_current_assets")

    # ============================================================
    # 4) Efficiency / Activity (balance-sheet composition proxies)
    # ============================================================
    if require_cols(df, ["invt", "at"]):
        df["inventory_to_assets"] = safe_div(df["invt"], df["at"])
        feature_groups["Efficiency_/_Activity"].append("inventory_to_assets")

    if require_cols(df, ["rect", "at"]):
        df["receivables_to_assets"] = safe_div(df["rect"], df["at"])
        feature_groups["Efficiency_/_Activity"].append("receivables_to_assets")

    if require_cols(df, ["ap", "at"]):
        df["payables_to_assets"] = safe_div(df["ap"], df["at"])
        feature_groups["Efficiency_/_Activity"].append("payables_to_assets")

    if require_cols(df, ["ppent", "at"]):
        df["fixed_assets_to_assets"] = safe_div(df["ppent"], df["at"])
        feature_groups["Efficiency_/_Activity"].append("fixed_assets_to_assets")

    if require_cols(df, ["act", "at"]):
        df["current_assets_to_assets"] = safe_div(df["act"], df["at"])
        feature_groups["Efficiency_/_Activity"].append("current_assets_to_assets")

    if require_cols(df, ["ppent", "act"]):
        df["fixed_to_current_assets"] = safe_div(df["ppent"], df["act"])
        feature_groups["Efficiency_/_Activity"].append("fixed_to_current_assets")

    if require_cols(df, ["ppent", "act"]):
        # Share of fixed assets in (fixed + current assets) as a rough intensity proxy
        df["capital_intensity_proxy"] = safe_div(df["ppent"], (df["ppent"] + df["act"]))
        feature_groups["Efficiency_/_Activity"].append("capital_intensity_proxy")

    # ============================================================
    # 5) Firm characteristics & dynamics (size, growth, changes)
    # ============================================================
    if "at" in df.columns:
        df["log_assets"] = safe_log(df["at"])
        feature_groups["FirmCharacteristics_&_Dynamics"].append("log_assets")

    if "seq" in df.columns:
        df["log_book_equity"] = safe_log(df["seq"])
        feature_groups["FirmCharacteristics_&_Dynamics"].append("log_book_equity")

    if require_cols(df, ["at", "at_lag1"]):
        df["asset_growth"] = growth(df["at"], df["at_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("asset_growth")

    if require_cols(df, ["seq", "seq_lag1"]):
        df["equity_growth"] = growth(df["seq"], df["seq_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("equity_growth")

    if require_cols(df, ["niadj", "niadj_lag1"]):
        df["ni_growth"] = growth(df["niadj"], df["niadj_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("ni_growth")

    if require_cols(df, ["niadj", "oancf", "at"]):
        # Accruals proxy: (Net income - CFO) scaled by assets
        df["accruals"] = safe_div(df["niadj"] - df["oancf"], df["at"])
        feature_groups["FirmCharacteristics_&_Dynamics"].append("accruals")

        # Also keep the unscaled version (can be useful for diagnostics)
        df["total_accruals"] = df["niadj"] - df["oancf"]
        feature_groups["FirmCharacteristics_&_Dynamics"].append("total_accruals")

    if require_cols(df, ["act", "lct", "act_lag1", "lct_lag1", "at"]):
        # Change in net working capital scaled by assets (only if contiguous)
        nwc = df["act"] - df["lct"]
        nwc_lag1 = df["act_lag1"] - df["lct_lag1"]
        out = pd.Series(np.nan, index=df.index)
        m = lag_mask.astype(bool)
        out.loc[m] = safe_div((nwc.loc[m] - nwc_lag1.loc[m]), df.loc[m, "at"])
        df["delta_nwc_to_assets"] = out
        feature_groups["FirmCharacteristics_&_Dynamics"].append("delta_nwc_to_assets")

    if require_cols(df, ["capx", "at"]):
        df["capx_to_assets"] = safe_div(df["capx"], df["at"])
        feature_groups["FirmCharacteristics_&_Dynamics"].append("capx_to_assets")

    if require_cols(df, ["ppent", "ppent_lag1", "at"]):
        # Change in fixed assets scaled by assets (only if contiguous)
        out = pd.Series(np.nan, index=df.index)
        m = lag_mask.astype(bool)
        out.loc[m] = safe_div(df.loc[m, "ppent"] - df.loc[m, "ppent_lag1"], df.loc[m, "at"])
        df["delta_fixed_assets_to_assets"] = out
        feature_groups["FirmCharacteristics_&_Dynamics"].append("delta_fixed_assets_to_assets")

    # Keep selected lagged ratios as explicit model features
    for c in ["roa", "debt_to_assets", "current_ratio"]:
        add_lag(df, c, keep=True)

    # IMPORTANT: ensure kept lag features are only valid if t-1 is contiguous.
    # This prevents leaking information across year gaps (e.g., 2018 -> 2020).
    for c in ["roa_lag1", "debt_to_assets_lag1", "current_ratio_lag1"]:
        if c in df.columns:
            df.loc[~lag_mask.astype(bool), c] = np.nan
            feature_groups["FirmCharacteristics_&_Dynamics"].append(c)

    # Deltas (changes) for selected key ratios
    if require_cols(df, ["roa", "roa_lag1"]):
        df["delta_roa"] = delta(df["roa"], df["roa_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("delta_roa")

    if require_cols(df, ["debt_to_assets", "debt_to_assets_lag1"]):
        df["delta_debt_to_assets"] = delta(df["debt_to_assets"], df["debt_to_assets_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("delta_debt_to_assets")

    if require_cols(df, ["current_ratio", "current_ratio_lag1"]):
        df["delta_current_ratio"] = delta(df["current_ratio"], df["current_ratio_lag1"], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append("delta_current_ratio")

    # Replace infinities created by divisions with NaN
    print("Handling Infinities (Replacing with NaN)...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ------------------------------------------------------------
    # Target variable: predict whether ROA increases next year (t+1)
    # Only keep rows where next year exists and is contiguous (strict t+1).
    # ------------------------------------------------------------
    print("Creating Target Variable (Strict t+1 Check)...")

    if "roa" not in df.columns:
        print("Error: 'roa' could not be computed (missing niadj/at).")
        return

    # Next-year ROA and next-year fiscal year
    df["roa_next"] = df.groupby("gvkey")["roa"].shift(-1)
    df["fyear_next"] = df.groupby("gvkey")["fyear"].shift(-1)

    # Strictly require t+1 to be the next fiscal year (avoid gaps)
    valid_target_mask = (df["fyear_next"] == df["fyear"] + 1)

    # Keep only rows with valid next-year target
    df_valid = df[valid_target_mask].copy()
    dropped_rows = len(df) - len(df_valid)
    print(f"Dropped {dropped_rows} rows (Last year of data OR gaps in time series).")

    # Binary classification target: 1 if ROA increases from t to t+1, else 0
    df_valid["target"] = (df_valid["roa_next"] > df_valid["roa"]).astype(int)

    # Save feature group map (for later reporting / plotting)
    os.makedirs(os.path.dirname(FEATURE_MAP_FILE), exist_ok=True)
    with open(FEATURE_MAP_FILE, "w") as f:
        json.dump(feature_groups, f, indent=2)

    # Identify ID/meta columns we want to keep if present
    id_cols = [c for c in ["gvkey", "fyear", "datadate", "conm", "indfmt", "datafmt", "consol", "ismod"]
               if c in df_valid.columns]

    # Determine the ordered feature columns (by our defined group ordering)
    ordered_feature_cols = []
    for g in [
        "Liquidity_&_CashFlow",
        "Leverage_&_CapitalStructure",
        "Profitability_&_Returns",
        "Efficiency_/_Activity",
        "FirmCharacteristics_&_Dynamics",
    ]:
        ordered_feature_cols.extend([c for c in feature_groups[g] if c in df_valid.columns])

    # Drop ONLY helper columns (including helper lags); keep lag features we flagged as features
    helper_cols = set(helper_lag_cols + [
        "fyear_lag1", "roa_next", "fyear_next",
        "avg_at", "avg_act", "avg_lct", "avg_invt", "avg_rect", "avg_ap", "avg_ppent", "avg_seq", "avg_lt",
    ])

    cols_to_drop = [c for c in helper_cols if c in df_valid.columns]
    df_out = df_valid.drop(columns=cols_to_drop)

    # Reorder columns: IDs + target + ordered engineered features; keep any remaining columns at the end
    important_cols = id_cols + ["target"] + ordered_feature_cols
    other_cols = [c for c in df_out.columns if c not in important_cols]
    df_final = df_out[important_cols + other_cols].copy()

    # Save final dataset for modeling
    print(f"Saving {len(df_final)} rows to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, index=False)

    # Print a short summary of the created feature groups
    print("\nFeature groups written to:", FEATURE_MAP_FILE)
    for k, v in feature_groups.items():
        print(f"{k}: {len(v)} features")

    # Basic diagnostics: descriptive statistics and missingness for target + engineered features
    diag_cols = ["target"] + ordered_feature_cols
    diag_cols = [c for c in diag_cols if c in df_final.columns]

    print("\nFeature Statistics (Target + Engineered Features):")
    if diag_cols:
        print(df_final[diag_cols].describe().T[["count", "mean", "std", "min", "max"]])

        print("\nMissing Values (NaN) Summary (Target + Engineered Features):")
        print(df_final[diag_cols].isna().sum())

    # Target class balance (useful to anticipate imbalance issues)
    print("\nClass Balance (Target):")
    print(df_final["target"].value_counts(normalize=True))

    print("==========================================")
    print("        FEATURE ENGINEERING DONE          ")
    print("==========================================")


if __name__ == "__main__":
    construct_features()

