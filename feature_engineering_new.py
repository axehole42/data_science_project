import pandas as pd
import numpy as np
import os
import json

INPUT_FILE = 'task_data/cleaned_data.parquet'
OUTPUT_FILE = 'task_data/features.parquet'
FEATURE_MAP_FILE = 'task_data/feature_groups.json'

# ---------------------------
# helpers
# ---------------------------
def safe_div(num, den):
    """Elementwise safe division: returns NaN when den is 0/NaN."""
    den = den.replace(0, np.nan)
    return num / den

def safe_log(x):
    return np.where(x > 0, np.log(x), np.nan)

def avg_if_contiguous(curr, lag1, lag_mask):
    """Average(curr, lag1) only if the prior year exists; else NaN."""
    return np.where(lag_mask, (curr + lag1) / 2.0, np.nan)

def growth(curr, lag1, lag_mask):
    return np.where(lag_mask, safe_div(curr - lag1, lag1), np.nan)

def delta(curr, lag1, lag_mask):
    return np.where(lag_mask, curr - lag1, np.nan)

def require_cols(df, cols):
    return all(c in df.columns for c in cols)

# ---------------------------
# main
# ---------------------------
def construct_features():
    print("==========================================")
    print("      FEATURE ENGINEERING (ROA TARGET)    ")
    print("==========================================")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    # Sort for time series ops
    df.sort_values(by=['gvkey', 'fyear'], inplace=True)

    # Strict lag mask (t-1 exists)
    df['fyear_lag1'] = df.groupby('gvkey')['fyear'].shift(1)
    lag_mask = (df['fyear_lag1'] == df['fyear'] - 1)

    # Create common lags needed for averages/growth
    lag_cols = [
        'at','act','lct','invt','rect','ap','ppent','lt','seq','sale','niadj','oancf','capx','dltt','dlc','oibdp','oiadp','ebit'
    ]
    for c in lag_cols:
        if c in df.columns:
            df[f'{c}_lag1'] = df.groupby('gvkey')[c].shift(1)

    # Common averages (only if contiguous)
    if 'at' in df.columns and 'at_lag1' in df.columns:
        df['avg_at'] = avg_if_contiguous(df['at'], df['at_lag1'], lag_mask)
    if 'act' in df.columns and 'act_lag1' in df.columns:
        df['avg_act'] = avg_if_contiguous(df['act'], df['act_lag1'], lag_mask)
    if 'lct' in df.columns and 'lct_lag1' in df.columns:
        df['avg_lct'] = avg_if_contiguous(df['lct'], df['lct_lag1'], lag_mask)
    if 'invt' in df.columns and 'invt_lag1' in df.columns:
        df['avg_invt'] = avg_if_contiguous(df['invt'], df['invt_lag1'], lag_mask)
    if 'rect' in df.columns and 'rect_lag1' in df.columns:
        df['avg_rect'] = avg_if_contiguous(df['rect'], df['rect_lag1'], lag_mask)
    if 'ap' in df.columns and 'ap_lag1' in df.columns:
        df['avg_ap'] = avg_if_contiguous(df['ap'], df['ap_lag1'], lag_mask)
    if 'ppent' in df.columns and 'ppent_lag1' in df.columns:
        df['avg_ppent'] = avg_if_contiguous(df['ppent'], df['ppent_lag1'], lag_mask)
    if 'seq' in df.columns and 'seq_lag1' in df.columns:
        df['avg_seq'] = avg_if_contiguous(df['seq'], df['seq_lag1'], lag_mask)
    if 'lt' in df.columns and 'lt_lag1' in df.columns:
        df['avg_lt'] = avg_if_contiguous(df['lt'], df['lt_lag1'], lag_mask)

    # ---------------------------
    # feature groups (5 Kategorien)
    # ---------------------------
    feature_groups = {
        "Liquidity_&_CashFlow": [],
        "Leverage_&_CapitalStructure": [],
        "Profitability_&_Returns": [],
        "Efficiency_/_Activity": [],
        "FirmCharacteristics_&_Dynamics": []
    }

    print("Constructing ratios (as many as possible, depending on available columns)...")

    # ========= 1) Liquidity & Cash Flow =========
    # Liquidity
    if require_cols(df, ['act','lct']):
        df['current_ratio'] = safe_div(df['act'], df['lct'])
        feature_groups["Liquidity_&_CashFlow"].append('current_ratio')

    if require_cols(df, ['act','lct','invt']):
        df['quick_ratio'] = safe_div(df['act'] - df['invt'], df['lct'])
        feature_groups["Liquidity_&_CashFlow"].append('quick_ratio')

    if require_cols(df, ['che','lct']):
        df['cash_ratio'] = safe_div(df['che'], df['lct'])
        feature_groups["Liquidity_&_CashFlow"].append('cash_ratio')

    if require_cols(df, ['act','lct','at']):
        df['working_cap_to_assets'] = safe_div(df['act'] - df['lct'], df['at'])
        feature_groups["Liquidity_&_CashFlow"].append('working_cap_to_assets')

    # Cash flow ratios
    if require_cols(df, ['oancf','lct']):
        df['cfo_to_current_liab'] = safe_div(df['oancf'], df['lct'])
        feature_groups["Liquidity_&_CashFlow"].append('cfo_to_current_liab')

    if require_cols(df, ['oancf','lt']):
        df['cfo_to_total_liab'] = safe_div(df['oancf'], df['lt'])
        feature_groups["Liquidity_&_CashFlow"].append('cfo_to_total_liab')

    if require_cols(df, ['oancf','dltt','dlc']):
        dltt0 = df['dltt'].fillna(0)
        dlc0  = df['dlc'].fillna(0)
        df['total_debt'] = dltt0 + dlc0
        df['cfo_to_total_debt'] = safe_div(df['oancf'], df['total_debt'])
        feature_groups["Liquidity_&_CashFlow"].append('cfo_to_total_debt')
    else:
        df['total_debt'] = np.nan  # keep column consistent for later steps if needed

    if require_cols(df, ['oancf','at']):
        df['cfo_to_assets'] = safe_div(df['oancf'], df['at'])
        feature_groups["Liquidity_&_CashFlow"].append('cfo_to_assets')

    if require_cols(df, ['oancf','sale']):
        df['cfo_to_sales'] = safe_div(df['oancf'], df['sale'])
        feature_groups["Liquidity_&_CashFlow"].append('cfo_to_sales')

    if require_cols(df, ['oancf','capx','at']):
        df['fcf'] = df['oancf'] - df['capx']
        df['fcf_to_assets'] = safe_div(df['fcf'], df['at'])
        feature_groups["Liquidity_&_CashFlow"].append('fcf_to_assets')

    if require_cols(df, ['oancf','capx','total_debt']):
        df['fcf'] = df.get('fcf', np.nan)
        df['fcf_to_debt'] = safe_div(df['fcf'], df['total_debt'])
        feature_groups["Liquidity_&_CashFlow"].append('fcf_to_debt')

    # ========= 2) Leverage & Capital Structure =========
    if require_cols(df, ['dltt','dlc','at']):
        dltt0 = df['dltt'].fillna(0)
        dlc0  = df['dlc'].fillna(0)
        df['debt_to_assets'] = safe_div(dltt0 + dlc0, df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('debt_to_assets')

    if require_cols(df, ['lt','seq']):
        df['debt_to_equity'] = safe_div(df['lt'], df['seq'])
        feature_groups["Leverage_&_CapitalStructure"].append('debt_to_equity')

    if require_cols(df, ['lt','at']):
        df['total_liab_to_assets'] = safe_div(df['lt'], df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('total_liab_to_assets')

    if require_cols(df, ['seq','at']):
        df['equity_to_assets'] = safe_div(df['seq'], df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('equity_to_assets')

    if require_cols(df, ['seq','lt']):
        df['equity_to_liab'] = safe_div(df['seq'], df['lt'])
        feature_groups["Leverage_&_CapitalStructure"].append('equity_to_liab')

    if require_cols(df, ['dltt','at']):
        df['lt_debt_to_assets'] = safe_div(df['dltt'].fillna(0), df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('lt_debt_to_assets')

    if require_cols(df, ['dlc','at']):
        df['st_debt_to_assets'] = safe_div(df['dlc'].fillna(0), df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('st_debt_to_assets')

    if require_cols(df, ['lct','at']):
        df['current_liab_to_assets'] = safe_div(df['lct'], df['at'])
        feature_groups["Leverage_&_CapitalStructure"].append('current_liab_to_assets')

    # Interest coverage
    if require_cols(df, ['ebit','xint']):
        df['interest_coverage_ebit'] = safe_div(df['ebit'], df['xint'])
        feature_groups["Leverage_&_CapitalStructure"].append('interest_coverage_ebit')

    if require_cols(df, ['oibdp','xint']):
        df['interest_coverage_ebitda'] = safe_div(df['oibdp'], df['xint'])
        feature_groups["Leverage_&_CapitalStructure"].append('interest_coverage_ebitda')

    # Fixed assets / long-term liabilities (proxy with DLTT if no dedicated long-term liabilities)
    if require_cols(df, ['ppent','dltt']):
        df['fixed_assets_to_lt_debt'] = safe_div(df['ppent'], df['dltt'].replace(0, np.nan))
        feature_groups["Leverage_&_CapitalStructure"].append('fixed_assets_to_lt_debt')

    # ========= 3) Profitability & Returns =========
    # Margins
    if require_cols(df, ['sale','cogs']):
        df['gross_margin'] = safe_div(df['sale'] - df['cogs'], df['sale'])
        feature_groups["Profitability_&_Returns"].append('gross_margin')

    if require_cols(df, ['sale','oiadp']):  # operating income after depreciation
        df['operating_margin'] = safe_div(df['oiadp'], df['sale'])
        feature_groups["Profitability_&_Returns"].append('operating_margin')

    if require_cols(df, ['sale','ebit']):
        df['ebit_margin'] = safe_div(df['ebit'], df['sale'])
        feature_groups["Profitability_&_Returns"].append('ebit_margin')

    if require_cols(df, ['sale','oibdp']):  # EBITDA proxy
        df['ebitda_margin'] = safe_div(df['oibdp'], df['sale'])
        feature_groups["Profitability_&_Returns"].append('ebitda_margin')

    if require_cols(df, ['sale','niadj']):
        df['net_margin'] = safe_div(df['niadj'], df['sale'])
        feature_groups["Profitability_&_Returns"].append('net_margin')

    # Returns (use average denominators if possible)
    if require_cols(df, ['niadj','at']):
        # keep your original definition; you can switch to avg_at if you prefer
        df['roa'] = safe_div(df['niadj'], df['at'])
        feature_groups["Profitability_&_Returns"].append('roa')

    if require_cols(df, ['niadj','avg_at']):
        df['roa_avg_assets'] = safe_div(df['niadj'], df['avg_at'])
        feature_groups["Profitability_&_Returns"].append('roa_avg_assets')

    if require_cols(df, ['niadj','seq']):
        df['roe'] = safe_div(df['niadj'], df['seq'])
        feature_groups["Profitability_&_Returns"].append('roe')

    if require_cols(df, ['niadj','avg_seq']):
        df['roe_avg_equity'] = safe_div(df['niadj'], df['avg_seq'])
        feature_groups["Profitability_&_Returns"].append('roe_avg_equity')

    # Simple ROIC / return on capital employed proxy: EBIT / (AT - LCT)
    if require_cols(df, ['ebit','at','lct']):
        df['roic_proxy'] = safe_div(df['ebit'], (df['at'] - df['lct']))
        feature_groups["Profitability_&_Returns"].append('roic_proxy')

    # Return on fixed/current assets
    if require_cols(df, ['niadj','ppent']):
        df['return_on_fixed_assets'] = safe_div(df['niadj'], df['ppent'])
        feature_groups["Profitability_&_Returns"].append('return_on_fixed_assets')

    if require_cols(df, ['niadj','act']):
        df['return_on_current_assets'] = safe_div(df['niadj'], df['act'])
        feature_groups["Profitability_&_Returns"].append('return_on_current_assets')

    # ========= 4) Efficiency / Activity =========
    # Turnovers use averages where possible
    if require_cols(df, ['sale','avg_at']):
        df['asset_turnover'] = safe_div(df['sale'], df['avg_at'])
        feature_groups["Efficiency_/_Activity"].append('asset_turnover')
    elif require_cols(df, ['sale','at']):
        df['asset_turnover'] = safe_div(df['sale'], df['at'])
        feature_groups["Efficiency_/_Activity"].append('asset_turnover')

    if require_cols(df, ['sale','avg_ppent']):
        df['fixed_asset_turnover'] = safe_div(df['sale'], df['avg_ppent'])
        feature_groups["Efficiency_/_Activity"].append('fixed_asset_turnover')
    elif require_cols(df, ['sale','ppent']):
        df['fixed_asset_turnover'] = safe_div(df['sale'], df['ppent'])
        feature_groups["Efficiency_/_Activity"].append('fixed_asset_turnover')

    if require_cols(df, ['sale','avg_act']):
        df['current_asset_turnover'] = safe_div(df['sale'], df['avg_act'])
        feature_groups["Efficiency_/_Activity"].append('current_asset_turnover')
    elif require_cols(df, ['sale','act']):
        df['current_asset_turnover'] = safe_div(df['sale'], df['act'])
        feature_groups["Efficiency_/_Activity"].append('current_asset_turnover')

    # Inventory turnover (COGS / avg inventory)
    if require_cols(df, ['cogs','avg_invt']):
        df['inventory_turnover'] = safe_div(df['cogs'], df['avg_invt'])
        feature_groups["Efficiency_/_Activity"].append('inventory_turnover')
    elif require_cols(df, ['cogs','invt']):
        df['inventory_turnover'] = safe_div(df['cogs'], df['invt'])
        feature_groups["Efficiency_/_Activity"].append('inventory_turnover')

    # Receivables turnover (Sales / avg receivables)
    if require_cols(df, ['sale','avg_rect']):
        df['receivables_turnover'] = safe_div(df['sale'], df['avg_rect'])
        feature_groups["Efficiency_/_Activity"].append('receivables_turnover')
    elif require_cols(df, ['sale','rect']):
        df['receivables_turnover'] = safe_div(df['sale'], df['rect'])
        feature_groups["Efficiency_/_Activity"].append('receivables_turnover')

    # Payables turnover (COGS / avg AP)
    if require_cols(df, ['cogs','avg_ap']):
        df['payables_turnover'] = safe_div(df['cogs'], df['avg_ap'])
        feature_groups["Efficiency_/_Activity"].append('payables_turnover')
    elif require_cols(df, ['cogs','ap']):
        df['payables_turnover'] = safe_div(df['cogs'], df['ap'])
        feature_groups["Efficiency_/_Activity"].append('payables_turnover')

    # Equity / liability turnovers (Sales over financing bases)
    if require_cols(df, ['sale','avg_seq']):
        df['equity_turnover'] = safe_div(df['sale'], df['avg_seq'])
        feature_groups["Efficiency_/_Activity"].append('equity_turnover')
    elif require_cols(df, ['sale','seq']):
        df['equity_turnover'] = safe_div(df['sale'], df['seq'])
        feature_groups["Efficiency_/_Activity"].append('equity_turnover')

    if require_cols(df, ['sale','avg_lct']):
        df['current_liab_turnover'] = safe_div(df['sale'], df['avg_lct'])
        feature_groups["Efficiency_/_Activity"].append('current_liab_turnover')
    elif require_cols(df, ['sale','lct']):
        df['current_liab_turnover'] = safe_div(df['sale'], df['lct'])
        feature_groups["Efficiency_/_Activity"].append('current_liab_turnover')

    if require_cols(df, ['sale','dltt','dltt_lag1']):
        avg_dltt = avg_if_contiguous(df['dltt'], df['dltt_lag1'], lag_mask)
        df['lt_debt_turnover'] = safe_div(df['sale'], avg_dltt)
        feature_groups["Efficiency_/_Activity"].append('lt_debt_turnover')
    elif require_cols(df, ['sale','dltt']):
        df['lt_debt_turnover'] = safe_div(df['sale'], df['dltt'])
        feature_groups["Efficiency_/_Activity"].append('lt_debt_turnover')

    if require_cols(df, ['sale','avg_lt']):
        df['total_liab_turnover'] = safe_div(df['sale'], df['avg_lt'])
        feature_groups["Efficiency_/_Activity"].append('total_liab_turnover')
    elif require_cols(df, ['sale','lt']):
        df['total_liab_turnover'] = safe_div(df['sale'], df['lt'])
        feature_groups["Efficiency_/_Activity"].append('total_liab_turnover')

    # ========= 5) Firm Characteristics & Dynamics =========
    # Size
    if 'at' in df.columns:
        df['log_assets'] = safe_log(df['at'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('log_assets')
    if 'sale' in df.columns:
        df['log_sales'] = safe_log(df['sale'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('log_sales')
    if 'seq' in df.columns:
        df['log_book_equity'] = safe_log(df['seq'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('log_book_equity')

    # Growth / change (YoY, strict)
    if require_cols(df, ['at','at_lag1']):
        df['asset_growth'] = growth(df['at'], df['at_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('asset_growth')

    if require_cols(df, ['sale','sale_lag1']):
        df['sales_growth'] = growth(df['sale'], df['sale_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('sales_growth')

    if require_cols(df, ['seq','seq_lag1']):
        df['equity_growth'] = growth(df['seq'], df['seq_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('equity_growth')

    if require_cols(df, ['niadj','niadj_lag1']):
        df['ni_growth'] = growth(df['niadj'], df['niadj_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('ni_growth')

    # Accruals
    if require_cols(df, ['niadj','oancf','at']):
        df['accruals'] = safe_div(df['niadj'] - df['oancf'], df['at'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('accruals')
        df['total_accruals'] = df['niadj'] - df['oancf']
        feature_groups["FirmCharacteristics_&_Dynamics"].append('total_accruals')

    # ΔNWC / Assets (proxy): Δ(ACT-LCT)/AT (strict)
    if require_cols(df, ['act','lct','act_lag1','lct_lag1','at']):
        nwc = df['act'] - df['lct']
        nwc_lag1 = df['act_lag1'] - df['lct_lag1']
        df['delta_nwc_to_assets'] = np.where(lag_mask, safe_div(nwc - nwc_lag1, df['at']), np.nan)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('delta_nwc_to_assets')

    # Investment intensity
    if require_cols(df, ['capx','at']):
        df['capx_to_assets'] = safe_div(df['capx'], df['at'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('capx_to_assets')
    if require_cols(df, ['capx','sale']):
        df['capx_to_sales'] = safe_div(df['capx'], df['sale'])
        feature_groups["FirmCharacteristics_&_Dynamics"].append('capx_to_sales')
    if require_cols(df, ['ppent','ppent_lag1','at']):
        df['delta_fixed_assets_to_assets'] = np.where(lag_mask, safe_div(df['ppent'] - df['ppent_lag1'], df['at']), np.nan)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('delta_fixed_assets_to_assets')

    # Trends (deltas)
    if require_cols(df, ['roa','roa_lag1']):
        df['delta_roa'] = delta(df['roa'], df['roa_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('delta_roa')

    if require_cols(df, ['debt_to_assets','debt_to_assets_lag1']):
        df['delta_debt_to_assets'] = delta(df['debt_to_assets'], df['debt_to_assets_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('delta_debt_to_assets')

    if require_cols(df, ['current_ratio','current_ratio_lag1']):
        df['delta_current_ratio'] = delta(df['current_ratio'], df['current_ratio_lag1'], lag_mask)
        feature_groups["FirmCharacteristics_&_Dynamics"].append('delta_current_ratio')

    # ---------------------------
    # cleanup infinities
    # ---------------------------
    print("Handling Infinities (Replacing with NaN)...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ---------------------------
    # target creation (strict t+1)
    # ---------------------------
    print("Creating Target Variable (Strict t+1 Check)...")

    if 'roa' not in df.columns:
        print("Error: 'roa' could not be computed (missing niadj/at).")
        return

    df['roa_next'] = df.groupby('gvkey')['roa'].shift(-1)
    df['fyear_next'] = df.groupby('gvkey')['fyear'].shift(-1)
    valid_target_mask = (df['fyear_next'] == df['fyear'] + 1)

    df_valid = df[valid_target_mask].copy()
    dropped_rows = len(df) - len(df_valid)
    print(f"Dropped {dropped_rows} rows (Last year of data OR gaps in time series).")

    df_valid['target'] = (df_valid['roa_next'] > df_valid['roa']).astype(int)

    # ---------------------------
    # save feature group map
    # ---------------------------
    os.makedirs(os.path.dirname(FEATURE_MAP_FILE), exist_ok=True)
    with open(FEATURE_MAP_FILE, 'w') as f:
        json.dump(feature_groups, f, indent=2)

    # ---------------------------
    # final cols: identifiers + target + features (grouped order)
    # ---------------------------
    id_cols = [c for c in ['gvkey','fyear','datadate','conm','indfmt','datafmt'] if c in df_valid.columns]
    ordered_feature_cols = []
    for g in ["Liquidity_&_CashFlow","Leverage_&_CapitalStructure","Profitability_&_Returns","Efficiency_/_Activity","FirmCharacteristics_&_Dynamics"]:
        ordered_feature_cols.extend([c for c in feature_groups[g] if c in df_valid.columns])

    # include any other non-helper columns you want to keep
    helper_cols = set([c for c in df_valid.columns if c.endswith('_lag1')] + ['fyear_lag1','roa_next','fyear_next','avg_at','avg_act','avg_lct','avg_invt','avg_rect','avg_ap','avg_ppent','avg_seq','avg_lt'])
    other_cols = [c for c in df_valid.columns if c not in set(id_cols + ['target'] + ordered_feature_cols) and c not in helper_cols]

    df_final = df_valid[id_cols + ['target'] + ordered_feature_cols + other_cols].copy()

    print(f"Saving {len(df_final)} rows to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, index=False)

    # quick diagnostics
    print("\nFeature groups written to:", FEATURE_MAP_FILE)
    for k, v in feature_groups.items():
        print(f"{k}: {len(v)} features")

    print("\nClass Balance (Target):")
    print(df_final['target'].value_counts(normalize=True))

    print("==========================================")
    print("        FEATURE ENGINEERING DONE          ")
    print("==========================================")

if __name__ == "__main__":
    construct_features()
