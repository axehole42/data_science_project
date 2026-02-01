# AI in Finance: Net Income Prediction Project

This repo is for the preparation, training, testing, and validation of models for the Introduction to AI in Finance project. The goal is to predict whether a firm's **Return on Assets (ROA)** will **improve** in the next fiscal year ($ROA_{t+1} > ROA_t$) using **Annual** financial ratios and features from the current year ($t$).

## Git Workflow Guide

To contribute to this repository, ensure you have Git installed and configured in VS Code.

### Cloning the Repository
```bash
git clone https://github.com/axehole42/data_science_project
```

### Updating the Repository (Pushing Changes)
1.  **Stage your changes**:
    ```bash
    git add .
    ```
2.  **Commit your changes** (add a meaningful message):
    ```bash
    git commit -m "Description of your changes"
    ```
3.  **Push to GitHub**:
    ```bash
    git push origin main 
    ```
    *(Note: You will need to log in to GitHub the first time you push.)*

---

# ROA Improvement Prediction Project

This repository contains a machine learning pipeline for predicting Return on Assets (ROA) improvement using financial data (Compustat). The project follows a strict time-series cross-validation approach to prevent data leakage and ensure realistic performance estimates.

## Procedural Workflow

To reproduce the analysis and model training, execute the following steps in order:

1.  **Data Cleanup**
    *   **Command:** `python data_cleanup.py`
    *   **Action:** Loads the raw Compustat CSV, filters for industrial companies (INDL), and performs initial cleaning.
    *   **Output:** `task_data/cleaned_data.parquet`

2.  **Feature Engineering**
    *   **Command:** `python feature_engineering.py`
    *   **Action:** Loads the cleaned data and calculates financial ratios, lags, and other derived features defined in `task_data/feature_groups.json`.
    *   **Output:** `task_data/features.parquet`

3.  **Model Training (Main Model)**
    *   **Command:** `python MAIN_MODEL_XGB_OPT_TS.py` (or `MAIN_MODEL_XGB_OPT_TS.py`)
    *   **Action:** Trains an XGBoost classifier using time-series cross-validation and Optuna for hyperparameter optimization
    *   **Output:** `task_data/models_optuna_tscv/` (contains model artifacts, metrics, and best parameters)

4.  **Model Training (Baselines)**
    *   **Logistic Regression:** `python MODEL_LR_TS_CLEAN.py` -> Output: `task_data/models_optuna_tscv_logreg_clean/`
    *   **Random Forest:** `python MODEL_RandomF_TS_CLEAN.py` -> Output: `task_data/models_optuna_tscv_clean_rf_fast/`

5.  **Report Generation**
    *   Run specific scripts in the `latex/` folder to generate LaTeX tables and statistics for the final report.

## Main Code Modules

### Data Processing
*   **`data_cleanup.py`**:
    *   Handles the ingestion of the raw CSV file (`itaiif_compustat_data_24112025.csv`).
    *   Applies standard Compustat filters (e.g., keeping only standard industrial format 'INDL').
    *   Converts the data to efficient Parquet format.
*   **`feature_engineering.py`**:
    *   Central hub for feature creation.
    *   Computes financial ratios (liquidity, profitability, leverage, etc.).
    *   Handles lag generation (creating features from previous years).
    *   Uses robust mathematical operations (safe division, safe log) to handle financial edge cases (zeros, negatives).

### Model Training
All model scripts share a common "time-series cross-validation" architecture:
*   **`MAIN_MODEL_XGB_OPT_TS.py`**: The primary XGBoost model script. It uses Optuna to find the best hyperparameters by training on a rolling window of historical data and validating on the subsequent year. It saves feature importance and evaluation metrics.
*   **`MODEL_LR_TS_CLEAN.py`**: A Logistic Regression baseline. Includes preprocessing steps like winsorization and standardization within the cross-validation loop to prevent leakage.
*   **`MODEL_RandomF_TS_CLEAN.py`**: A Random Forest baseline optimized for speed ("fast" implementation with capped threads and warm starts).

## LaTeX Generation Tools (`latex/` folder)

These scripts generate `.tex` files or printed tables for the research paper/report:

*   **Missing Data Analysis**:
    *   `analyze_missing_mechanisms.py`, `formal_rubin_test.py`: Analyze why data is missing (Missing Completely at Random vs. Missing At Random).
    *   `generate_mar_latex.py`: Generates tables summarizing missing data patterns.
*   **Descriptive Statistics**:
    *   `generate_feature_stats.py`: summary statistics for the features.
    *   `generate_ratio_stats.py`: Statistics specifically for financial ratios.
*   **Model Results**:
    *   `generate_fi_latex.py`: Creates feature importance tables from the trained models.
    *   `generate_params_table.py`: Formats the best hyperparameters found by Optuna into a LaTeX table.

## Outputs

All intermediate and final outputs are stored in the `task_data/` directory:

*   **`task_data/cleaned_data.parquet`**: The cleaned raw dataset.
*   **`task_data/features.parquet`**: The final dataset with all engineering features, ready for training.
*   **`task_data/models_optuna_tscv_clean/`** (XGBoost Output):
    *   `best_params.json`: Optimal hyperparameters.
    *   `metrics.json`: Accuracy, AUC, F1, etc., on validation/test sets.
    *   `feature_importance.csv`: Global feature importance scores.
    *   `xgb_model.json`: The saved model object.
