# GEMINI.md

This file provides context and instructions for the Gemini AI agent working on this project.

## Project Overview
**Goal**: Predict whether a firm will report positive or negative net income next year ($t+1$) using current-year ($t$) financial ratios.
**Deliverable**: A report (PDF) and the code used.
**Deadline**: Report & Code by Feb 1st, 2026.

## Research Question & Methodology
**Question**: Can we predict whether a firm will report positive or negative net income next year ($t+1$) using current-year ($t$) **Annual** financial ratios?
- **Target Variable**: Binary (1 if `niadj` in $t+1 > 0$, else 0).
- **Lecture-Based Standards**:
    - **Cleaning**: 
        - Standardize formats (INDL, STD, Domestic, Consol='C').
        - Drop rows with missing `niadj` (Target).
        - Drop rows with missing or zero `at` (Assets > 0 required for ratios).
        - Remove duplicates.
    - **Imputation**: Handle missing values (Mean/Median/Mode or Model-based).
    - **Outliers**: Apply Winsorization (capping extreme values).
    - **Features**: Construct financial ratios and lagged variables.
    - **Split Strategy**: Time-series split (Train on past, Test on future) to prevent look-ahead bias.

## Tech Stack
- **Language**: Python 3.x
- **Libraries**:
    - `pandas`: Data manipulation
    - `numpy`: Numerical operations
    - `scikit-learn`: Machine Learning (StandardScaler, Models, Metrics)
    - `matplotlib` / `seaborn`: Visualization
    - `yfinance`: (Optional/Supplementary)

## Data Source
- **Raw Input**: `task_data/itaiif_compustat_data_24112025.csv` (Original Source)
- **Working Data**: `task_data/cleaned_data.parquet` (Optimized for ML & Analysis)
- **Data Dictionary**: `task_data/balance_income_cashflow_ITAIIF.pdf`

### Variable Dictionary (Compustat Codes)
| Variable | Description |
| :--- | :--- |
| **Identifiers** | |
| `gvkey` | Global Company Key (Unique ID) |
| `datadate` | Data Date |
| `fyear` | Fiscal Year |
| `conm` | Company Name |
| `indfmt` | Industry Format |
| `datafmt` | Data Format |
| `consol` | Consolidation Code |
| `ismod` | Issue Modifier Code |
| **Balance Sheet** | |
| `at` | Assets - Total |
| `lt` | Liabilities - Total |
| `act` | Current Assets - Total |
| `lct` | Current Liabilities - Total |
| `che` | Cash and Short-Term Investments |
| `rect` | Receivables - Total |
| `invt` | Inventories - Total |
| `ppent` | Property, Plant and Equipment - Total (Net) |
| `intan` | Intangible Assets - Total |
| `dlc` | Debt in Current Liabilities |
| `dltt` | Long-Term Debt - Total |
| `ceq` | Common/Ordinary Equity - Total |
| `seq` | Stockholders' Equity - Total |
| `pstk` | Preferred/Preference Stock (Capital) - Total |
| `tstk` | Treasury Stock - Total |
| `ap` | Accounts Payable - Trade |
| `txp` | Income Taxes Payable |
| **Income Statement** | |
| `niadj` | Net Income (Adjusted) - *TARGET VARIABLE* |
| `sale` | Sales/Turnover (Net) - **[MISSING IN RAW DATA]** |
| `cogs` | Cost of Goods Sold - **[MISSING IN RAW DATA]** |
| `xint` | Interest and Related Expense - Total |
| `txt` | Income Taxes - Total |
| `dp` | Depreciation and Amortization |
| `oibdp` | Operating Income Before Depreciation |
| **Cash Flow / Other** | |
| `capx` | Capital Expenditures |
| `oancf` | Operating Activities - Net Cash Flow |
| `fincf` | Financing Activities - Net Cash Flow |
| `ivncf` | Investing Activities - Net Cash Flow |
| `prcc_f` | Price Close - Annual - Fiscal |
| `csho` | Common Shares Outstanding |

## Development Commands
- **Browse Data**: `python browse_data.py` (View top rows in terminal)
- **Run Cleanup**: `python data_cleanup.py`
