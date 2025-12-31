# Project Work Log & Status

## [2025-12-27 15:30] Context Restoration
- **Status**: Successfully restored project context after session loss.
- **Action**: Created `PROJECT_STATUS.md` to maintain persistent state across sessions.

## [2025-12-27 15:35] Data Cleaning Refinement
- **Requirement**: Drop companies missing `at` (Total Assets) or `niadj` (Net Income).
- **Reasoning**: `niadj` is the target variable; `at` is the primary scaler for financial ratios.
- **Action**: Updated `data_cleanup.py` and `documentation.tex`.

## [2025-12-27 15:45] Data Validation & Issue Discovery
- **Status Check**: Verified `cleaned_data.parquet`. Result: 75,005 rows.
- **Critical Issue**: Discovered `sale` (Sales) and `cogs` (Cost of Goods Sold) are **missing from raw input CSV**.
- **Verification**: Confirmed via CSV header inspection.
- **Decision**: Proceed with ratios not requiring Sales (ROA, ROE, Leverage, Liquidity).

## [2025-12-27 15:55] Quality Check: Shell Companies
- **Issue**: Found 309 rows where `at = 0.0`.
- **Investigation**: Identified these as inactive shell companies (e.g., PETRO USA INC).
- **Implication**: `at=0` causes division-by-zero errors in all ratios.
- **Decision**: Filter out rows where `at == 0` in the cleanup script.

## [2025-12-27 16:15] Filtering Zero Assets
- **Action**: Modified `data_cleanup.py` to drop `at = 0`.
- **Clarification**: Confirmed negative `niadj` values are **preserved** (only NaNs dropped).
- **Status**: Cleanup code and LaTeX documentation synchronized.

## [2025-12-27 16:25] Documentation Synchronization
- **Action**: Updated `README.md` and `GEMINI.md`.
- **Details**: Added `feature_engineering.py` instructions and fixed Git workflow formatting (indentation).
- **Data Check**: Verified `niadj` types are correct (float64); identified extreme outliers as the cause of visualization issues in D-Tale.

## [2025-12-27 16:45] Pivot: New Research Question
- **Definition Change**: Redefined Target Variable based on project requirements.
- **Old Target**: Profitable vs. Loss (`niadj > 0`).
- **New Target**: Improvement vs. Deterioration ($ROA_{t+1} > ROA_t$).
- **Action**: Planned comprehensive feature set including Accruals and Financial Leverage.

## [2025-12-27 16:55] Methodology: Time Alignment
- **Issue**: Standard `shift(-1)` ignores fiscal year gaps (e.g., skipping 2013).
- **Fix**: Implemented strict check: Target is only valid if $FYEAR_{next} == FYEAR_{current} + 1$.
- **Lag Features**: Applied the same logic to trend features ($t-1$).
- **Action**: Updated `feature_engineering.py`.

## [2025-12-27 17:05] Feature Engineering Robustness
- **Issue**: Denominators like `lct` (current liabilities) can be zero.
- **Fix**: Added infinity handling (replaced with `NaN`) to prevent model distortion.
- **Debt Assumption**: Documented filling missing debt (`dltt`/`dlc`) with 0.

## [2025-12-27 17:15] Feature Engineering Refinement
- **Enhancements**: 
    - Renamed variables for clarity (`working_cap_to_assets`).
    - Added robust log calculations for firm size (`np.where` for `log_size`).
- **Data Preservation**: Updated script to **keep all original columns** in `features.parquet` for traceability.
- **Column Order**: Moved identifiers and new features to the very front of the file.

## [2025-12-27 17:35] UI Improvements
- **Action**: Updated `browse_data.py`.
- **New Feature**: Added command-line arguments and prompts to toggle between Cleaned Data (`C`) and Features (`F`) with auto-reload.

## [2025-12-27 17:40] Feature Engineering Execution (Final)
- **Output**: `features.parquet` generated with **62,754 rows**.
- **Class Balance**: 49.6% (1) vs 50.4% (0) - perfectly balanced.
- **Missing Data**: ~15% missingness in Liquidity/Trend features identified for future imputation.
- **Target Alignment**: Verified that Target for year $t$ correctly points to $t+1$ outcome.

## [2025-12-27 17:50] Labels Update
- **Action**: Updated `task_data/var_labels.py`.
- **Details**: Added human-readable labels for all 13 new ratios and the target variable.

## [2025-12-27 18:00] Lecture Requirements Verification
- **Audit**: Read all Lecture PDFs (Intro to Chapter 4).
- **Key Finding**: Chapter 3, Slide 12 strictly requires **TimeSeriesSplit** (Walk-Forward).
- **Prohibition**: Standard K-Fold is marked as incorrect for this data type.
- **Requirement**: Performance must be measured via ROC-AUC and Confusion Matrices.

## Current Project Status: Ready for Modeling
- **Next Phase**: Implementation of `model_training.py` using Time-Series Cross-Validation and Robust Preprocessing (Winsorization/Imputation).