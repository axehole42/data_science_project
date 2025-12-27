# AI in Finance: Net Income Prediction Project

This repo is for the preparation, training, testing, and validation of models for the **Introduction to AI in Finance** project. The goal is to predict whether a firm will report positive or negative Net Income in the next fiscal year ($t+1$) using financial ratios and features from the current year ($t$).

## 🛠️ Git Workflow Guide

To contribute to this repository, ensure you have [Git installed](https://git-scm.com/install/windows) and configured in VS Code.

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

## 📂 Project Structure

### 🐍 Core Scripts
*   **`browse_data.py`**
    *   **Purpose**: Interactive data exploration.
    *   **Usage**: `python browse_data.py`
    *   **Details**: Launches a local **D-Tale** web server. It opens your default browser to a GUI where you can filter, sort, visualize, and analyze the raw dataset similar to STATA's `browse` command.
    
*   **`data_cleanup.py`**
    *   **Purpose**: Data ingestion and preprocessing pipeline.
    *   **Usage**: `python data_cleanup.py`
    *   **Details**: 
        *   Loads the raw CSV.
        *   Sorts data by Company (`gvkey`) and Year (`fyear`).
        *   Removes duplicate entries.
        *   Parses dates into standard datetime objects.
        *   *(Future)*: Will handle Winsorization and Missing Value Imputation.

*   **`yfinancedownload.py`**
    *   **Purpose**: Supplementary data fetching.
    *   **Details**: Script for downloading additional market data via the `yfinance` API (currently a scratchpad/utility).

### 📄 Documentation & Config
*   **`GEMINI.md`**: Context file containing the project's Research Question, Variable Dictionary, and AI Agent instructions.
*   **`documentation.tex`**: LaTeX source file for the formal project report.
*   **`task_data/`**: Directory containing the raw input data (`itaiif_compustat_data_24112025.csv`) and lecture PDFs.

---

## 🚀 Getting Started

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

## 📊 Methodology (Brief)
1.  **Data Cleaning**: Standardize formats and remove duplicates.
2.  **Preprocessing**: Winsorize outliers and impute missing values (Median/Mode).
3.  **Feature Engineering**: Construct financial ratios (e.g., Current Ratio, Debt/Equity) and lag variables.
4.  **Modeling**: Train a binary classifier on $t$ to predict $t+1$ outcome.