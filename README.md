# AI in Finance: Net Income Prediction Project

This repo is for the preparation, training, testing, and validation of models for the Introduction to AI in Finance project. The goal is to predict whether a firm will report positive or negative Net Income in the next fiscal year ($t+1$) using **Annual** financial ratios and features from the current year ($t$).

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

## Project Structure

### Core Scripts
*   **`browse_data.py`**
    *   **Purpose**: Interactive data exploration.
    *   **Usage**: 
        *   `python browse_data.py` (Asks for choice)
        *   `python browse_data.py C` (Browse Cleaned Data)
        *   `python browse_data.py F` (Browse Features)
    *   **Details**: Launches a local D-Tale web server. It opens your default browser to a GUI where you can filter, sort, visualize, and analyze the raw dataset similar to STATA's browse command.
    
    > [!IMPORTANT]
    > Your web browser may display a "Not Secure" or "Unsafe" warning. This is expected behavior for local servers running on `localhost` without SSL (HTTPS) and is safe to ignore.
    
*   **`data_cleanup.py`**
    *   **Purpose**: Data ingestion and preprocessing pipeline.
    *   **Usage**: `python data_cleanup.py`
    *   **Details**: 
        *   Loads the raw CSV.
        *   Applies standard Compustat filters (INDL, STD, Domestic, Consolidated).
        *   Drops rows with missing `niadj` (Target) or missing/zero `at` (Assets).
        *   Removes duplicate entries.
        *   Parses dates into standard datetime objects.
        *   **Outputs**: Saves a processed `cleaned_data.parquet` file.

*   **`feature_engineering.py`**
    *   **Purpose**: Creation of financial ratios and target variables.
    *   **Usage**: `python feature_engineering.py`
    *   **Details**:
        *   Calculates key ratios: ROA, ROE, Liquidity, Leverage, Cash Flow.
        *   Handles missing Sales data by focusing on Balance Sheet/Income Statement ratios.
        *   Creates the Target Variable: `niadj` > 0 in $t+1$.
        *   **Outputs**: Saves `task_data/features.parquet`.

*   **`yfinancedownload.py`**
    *   **Purpose**: Supplementary data fetching.
    *   **Details**: Script for downloading additional market data via the `yfinance` API (currently a scratchpad/utility).

### Documentation & Config
*   **`GEMINI.md`**: Context file containing the project's Research Question, Variable Dictionary, and AI Agent instructions.
*   **`documentation.tex`**: LaTeX source file for the formal project report.
*   **`task_data/`**: Directory containing the raw input data (`itaiif_compustat_data_24112025.csv`) and lecture PDFs.

---

## Getting Started

### 1. Environment Setup
Ensure you have Python 3.x installed. It is recommended to work within a virtual environment.

```bash
# Windows
.\venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn dtale yfinance
```

### 2. Exploring the Data
To verify the data load and explore the variables interactively:

```bash
python browse_data.py
```

### 3. Running the Pipeline
To execute the cleaning steps defined so far:

```bash
python data_cleanup.py
```

---

## Methodology (Brief)
1.  **Data Cleaning**: 
    *   Confirm data frequency is **Annual**.
    *   Filter for standard industrial formats (`indfmt='INDL'`, `consol='C'`).
    *   Remove duplicate company-year entries.
2.  **Preprocessing**: Winsorize outliers and impute missing values (Median/Mode).3.  **Feature Engineering**: Construct financial ratios (e.g., Current Ratio, Debt/Equity) and lag variables.
4.  **Modeling**: Train a binary classifier on $t$ to predict $t+1$ outcome.
