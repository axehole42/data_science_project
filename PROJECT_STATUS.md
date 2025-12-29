# Project Status & History

## [2025-12-27 15:30] Context Restoration
- **Session Status**: Successfully restored project context after session loss.
- **Action**: Created `PROJECT_STATUS.md` to maintain persistent state across sessions.

## [2025-12-27 15:35] Data Cleaning Refinement
- **Requirement**: Drop companies missing `at` (Total Assets) or `niadj` (Net Income).
- **Reasoning**:
    - `niadj`: Essential as the target variable. Rows without labels are unusable for classification.
    - `at`: Fundamental denominator for financial ratios. Prevents division-by-zero or missing features during normalization.
- **Action**: Updated `data_cleanup.py` and `documentation.tex`.

## [2025-12-27 15:45] Data Validation & Issue Discovery
- **Status Check**: Verified `cleaned_data.parquet`.
    - **Result**: 75,005 rows. 0 missing `at`. 0 missing `niadj`. **Cleanup was successful.**
- **Critical Issue**: The variables `sale` (Sales) and `cogs` (Cost of Goods Sold) are **missing from the raw input CSV**.
    - **Impact**: Cannot calculate Asset Turnover, Inventory Turnover, or Profit Margin.
    - **Decision**: Proceed with available metrics (ROA, ROE, Leverage, Liquidity, Cash Flow).

## [2025-12-27 15:50] Feature Engineering Strategy
- **Goal**: Create predictor variables (Ratios) and the Target variable.
- **Planned Features**:
    - **Profitability**: ROA (`niadj`/`at`), ROE (`niadj`/`ceq`).
    - **Liquidity**: Current Ratio (`act`/`lct`), Cash Ratio (`che`/`lct`).
    - **Leverage**: Debt-to-Assets (`lt`/`at`), Debt-to-Equity (`lt`/`ceq`).
    - **Cash Flow**: CFO-to-Assets (`oancf`/`at`).
    - **Growth**: Asset Growth (requires lag).
- **Target Creation**: Shift `niadj` by -1 year to predict t+1 status.

## [2025-12-27 15:55] Quality Check (Pre-Features)
- **Issue**: Found 309 rows where `at = 0.0` (e.g., shell companies).
- **Implication**: Causes division-by-zero for all asset-based ratios.
- **Decision**: Filter out rows where `at == 0`.

## [2025-12-27 16:15] Filtering Zero Assets
- **Action**: Modified `data_cleanup.py` to filter out rows where `at == 0`.
- **Clarification**: Confirmed that negative `niadj` values are **preserved**. Only NaNs and `at=0` are dropped.
- **Status**: Code and Documentation updated.

## [2025-12-27 16:25] Documentation Synchronization
- **Action**: Updated `README.md` and `GEMINI.md`.
- **Details**:
    - `README.md`: Added `feature_engineering.py`, clarified cleaning details, and fixed Git Workflow indentation/formatting.
    - `GEMINI.md`: Marked `sale` and `cogs` as missing; explicitly listed the `at > 0` rule.
- **Status**: Documentation synchronized.

## [2025-12-27 17:15] Feature Engineering Refinement
- **Enhancements**:
    - Renamed `working_cap` to `working_cap_to_assets` for precision.
    - Updated `log_size` calculation to handle non-positive values safely using `np.where`.
    - Cleaned up intermediate temporary columns (`dltt_temp`, `dlc_temp`) from the final dataset.
- **Decision**: Maintained `NaN` for missing features, noting that Winsorization and Imputation will be handled in the modeling phase.
- **Data Preservation**: Updated script to **keep all original columns** from `cleaned_data.parquet` in the final `features.parquet` file to allow for full traceability.
- **Status**: Ready to generate the final feature dataset.

## [2025-12-27 17:40] Feature Engineering Execution (Final)
- **Status**: Successful run.
- **Output**: `features.parquet` with 62,754 rows.
- **Data Reduction Logic (Explained)**:
    - **Total Drop**: ~11,942 rows.
    - **Reason**: Supervised learning requires a known target ($t+1$).
    - **Component A**: The last year of data for every firm (e.g., 2024) has no 2025 data to compare against, so it has no label. (~9.5k rows).
    - **Component B**: Years before a data gap (e.g., 2012 followed by 2014) cannot verify a 1-year change. (~2.5k rows).
- **Class Balance**: Perfectly balanced (49.6% Improvement / 50.4% Deterioration).
- **Next Phase**: Model Training (Preprocessing Pipeline).

