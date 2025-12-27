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
    - `README.md`: Added `feature_engineering.py` and clarified cleaning details.
    - `GEMINI.md`: Marked `sale` and `cogs` as missing; explicitly listed the `at > 0` rule.
- **Status**: Documentation synchronized.

## [2025-12-27 16:35] Feature Engineering Execution
- **Status**: `feature_engineering.py` is written and verified.
- **Action**: User is taking a break.
- **Next Step**: Execute `python feature_engineering.py` to generate ratios and target variables.

