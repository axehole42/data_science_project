# Mapping of Compustat Codes to Human-Readable Labels

VAR_LABELS = {
    # Identifiers
    'gvkey': 'Company ID (GVKEY)',
    'datadate': 'Data Date',
    'fyear': 'Fiscal Year',
    'indfmt': 'Industry Format',
    'consol': 'Consolidation',
    'popsrc': 'Population Source',
    'datafmt': 'Data Format',
    'tic': 'Ticker Symbol',
    'conm': 'Company Name',
    'curcd': 'Currency',
    'fyrc': 'Fiscal Year-End Month',
    
    # Balance Sheet - Assets
    'at': 'Total Assets',
    'act': 'Current Assets',
    'che': 'Cash & Short-Term Inv',
    'rect': 'Receivables (Total)',
    'invt': 'Inventories (Total)',
    'aco': 'Current Assets (Other)',
    'ppent': 'PPE (Net)',
    'intan': 'Intangible Assets',
    'ao': 'Assets (Other)',
    
    # Balance Sheet - Liabilities & Equity
    'lt': 'Total Liabilities',
    'lct': 'Current Liabilities',
    'ap': 'Accounts Payable',
    'dlc': 'Debt in Current Liab',
    'txp': 'Income Taxes Payable',
    'lco': 'Current Liab (Other)',
    'dltt': 'Long-Term Debt',
    'lo': 'Liabilities (Other)',
    'txditc': 'Deferred Taxes & ITC',
    'seq': 'Stockholders Equity',
    'ceq': 'Common Equity',
    'pstk': 'Preferred Stock',
    
    # Income Statement
    'sale': 'Sales/Turnover (Net)',
    'revt': 'Revenue (Total)',
    'cogs': 'Cost of Goods Sold',
    'xsga': 'SG&A Expense',
    'oibdp': 'Operating Income Bef Deprec',
    'dp': 'Depreciation & Amort',
    'xint': 'Interest Expense',
    'nopi': 'Non-Operating Income',
    'txt': 'Income Taxes (Total)',
    'ib': 'Income Before Extra Items',
    'ni': 'Net Income (Loss)',
    'niadj': 'Net Income (Adjusted)',
    'epsfi': 'EPS (Basic) - Excl Extra',
    
    # Cash Flow
    'oancf': 'Operating Cash Flow',
    'ivncf': 'Investing Cash Flow',
    'fincf': 'Financing Cash Flow',
    'capx': 'Capital Expenditures',
    
    # Market Data
    'prcc_f': 'Price Close (Fiscal Year)',
    'csho': 'Common Shares Outstanding',
    'mkvalt': 'Market Value (Total)',

    # --- Engineered Features ---
    'target': 'Target: ROA Improvement next year (t+1)',
    'roa': 'Return on Assets (niadj / at)',
    'ocf_to_assets': 'Operating Cash Flow / Total Assets',
    'accruals': 'Accruals: (niadj - oancf) / at',
    'current_ratio': 'Current Ratio: act / lct',
    'cash_ratio': 'Cash Ratio: che / lct',
    'working_cap_to_assets': 'Working Capital / Total Assets',
    'debt_to_assets': 'Financial Debt / Total Assets',
    'debt_to_equity': 'Total Liabilities / Stockholders Equity',
    'total_liab_to_assets': 'Total Liabilities / Total Assets',
    'asset_growth': 'Asset Growth (percentage change)',
    'log_size': 'Firm Size (natural log of assets)',
    'delta_roa': 'Trend: Change in ROA from t-1',
    'delta_leverage': 'Trend: Change in Financial Debt ratio',
    'delta_curr_ratio': 'Trend: Change in Current Ratio',
}
