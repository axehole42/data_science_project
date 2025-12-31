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
        *   Drops rows with missing $niadj$ (Target) or missing/zero $at$ (Assets).
        *   Removes duplicate entries.
        *   Parses dates into standard datetime objects.
        *   **Outputs**: Saves a processed `cleaned_data.parquet` file.

*   **`feature_engineering.py`**
    *   **Purpose**: Creation of financial ratios and target variables.
    *   **Usage**: `python feature_engineering.py`
    *   **Details**:
        *   Calculates key ratios: ROA, ROE, Liquidity, Leverage, Cash Flow.
        *   Handles missing Sales data by focusing on Balance Sheet/Income Statement ratios.
        *   Creates the Target Variable: $niadj > 0$ in $t+1$.
        *   **Outputs**: Saves `task_data/features.parquet`.

*   **`yfinancedownload.py`**
    *   **Purpose**: Supplementary data fetching.
    *   **Details**: Script for downloading additional market data via the `yfinance` API (currently a scratchpad/utility).

### Documentation & Config
*   **`GEMINI.md`**: Context file containing the project's Research Question, Variable Dictionary, and AI Agent instructions.
*   **`documentation.tex`**: LaTeX source file for the formal project report.
*   **`task_data/`**: Directory containing the raw input data (`itaiif_compustat_data_24112025.csv`) and lecture PDFs.

---

### Feature Definitions

#### Target Variable (Research Question)

**ROA Improvement Indicator ($Y_{t+1}$)**
> **Rationale**: We focus on the *direction* of change rather than the absolute value. Improvements in profitability are stronger drivers of stock returns and credit upgrades than static levels.
> **Compustat**: `niadj`, `at`
```math
Y_{i, t+1} = \begin{cases} 1 & \text{if } ROA_{i, t+1} > ROA_{i, t} \\ 0 & \text{otherwise} \end{cases}
```

#### Financial Ratios (Features at time $t$)

**1. Profitability & Earnings Quality**

**Return on Assets (ROA)**
> **Rationale**: The baseline measure of operational efficiency. It reflects how effectively a company uses its assets to generate earnings.
> **Compustat**: `niadj`, `at`
```math
ROA_{t} = \frac{\text{NIADJ}_{t}}{\text{AT}_{t}}
```

**Operating Cash Flow to Assets ($CFO_{ratio}$)**
> **Rationale**: Cash flow is harder to manipulate than Net Income. A high ratio of cash flow to assets indicates strong genuine earnings power, distinct from accounting adjustments.
> **Compustat**: `oancf`, `at`
```math
OCF\_Ratio_{t} = \frac{\text{OANCF}_{t}}{\text{AT}_{t}}
```

**Accruals**
> **Rationale**: Captures the non-cash component of earnings. According to Sloan (1996), high accruals are less persistent and often "reverse" in future periods, predicting a decline in profitability.
> **Compustat**: `niadj`, `oancf`, `at`
```math
\text{Accruals}_{t} = \frac{\text{NIADJ}_{t} - \text{OANCF}_{t}}{\text{AT}_{t}}
```

**2. Liquidity & Short-Term Risk**

**Current Ratio**
> **Rationale**: A standard metric of short-term solvency. Low ratios indicate potential distress and an inability to fund operations, while extremely high ratios may suggest inefficient use of capital.
> **Compustat**: `act`, `lct`
```math
\text{Current Ratio}_{t} = \frac{\text{ACT}_{t}}{\text{LCT}_{t}}
```

**Cash Ratio**
> **Rationale**: The most conservative liquidity measure, focusing solely on cash and equivalents. It signals the firm's immediate capacity to pay off debts without selling inventory.
> **Compustat**: `che`, `lct`
```math
\text{Cash Ratio}_{t} = \frac{\text{CHE}_{t}}{\text{LCT}_{t}}
```

**Working Capital to Assets**
> **Rationale**: Normalizes the net working capital cushion by firm size. Positive working capital provides a buffer against operational shocks.
> **Compustat**: `act`, `lct`, `at`
```math
\text{WCAP}_{t} = \frac{\text{ACT}_{t} - \text{LCT}_{t}}{\text{AT}_{t}}
```

**3. Leverage & Solvency**

**Debt to Assets**
> **Rationale**: Measures financial leverage. High leverage increases bankruptcy risk but can also boost ROE in good times. High debt often constrains future borrowing capacity.
> **Compustat**: `dltt`, `dlc`, `at`
```math
\text{Lev}_{t} = \frac{\text{DLTT}_{t} + \text{DLC}_{t}}{\text{AT}_{t}}
```

**Debt to Equity**
> **Rationale**: Assesses solvency relative to shareholder capital. A high D/E ratio implies aggressive financing and higher volatility in earnings.
> **Compustat**: `lt`, `seq`
```math
\text{D/E}_{t} = \frac{\text{LT}_{t}}{\text{SEQ}_{t}}
```

**Total Liabilities to Assets**
> **Rationale**: The broadest measure of indebtedness, capturing all obligations (including payables and deferred taxes), not just interest-bearing debt.
> **Compustat**: `lt`, `at`
```math
\text{Liab/Assets}_{t} = \frac{\text{LT}_{t}}{\text{AT}_{t}}
```

**4. Growth & Size**

**Firm Size (Log Assets)**
> **Rationale**: Larger firms tend to be more diversified, have better access to capital markets, and lower volatility. We use the logarithm to normalize the highly skewed distribution of raw assets.
> **Compustat**: `at`
```math
\text{Size}_{t} = \ln(\text{AT}_{t})
```

**Asset Growth**
> **Rationale**: Rapid asset expansion can be a sign of success but is often associated with lower future returns due to diminishing marginal utility of investment (the "Asset Growth Anomaly").
> **Compustat**: `at` ($t$ and $t-1$)
```math
\text{Growth}_{t} = \frac{\text{AT}_{t} - \text{AT}_{t-1}}{\text{AT}_{t-1}}
```

**5. Trends (Momentum vs. Reversion)**

**$\Delta$ ROA**
> **Rationale**: Captures momentum. A firm with rising profitability may continue to improve, though strong increases are also subject to mean reversion.
> **Compustat**: `niadj`, `at` ($t$ and $t-1$)
```math
\Delta ROA_{t} = ROA_{t} - ROA_{t-1}
```

**$\Delta$ Leverage**
> **Rationale**: An increasing debt burden can signal distress or aggressive expansion, potentially pressuring future margins through higher interest expense.
> **Compustat**: `dltt`, `dlc`, `at` ($t$ and $t-1$)
```math
\Delta \text{Lev}_{t} = \text{Lev}_{t} - \text{Lev}_{t-1}
```

**$\Delta$ Current Ratio**
> **Rationale**: Improvements in liquidity trends suggest a strengthening balance sheet and reduced operational risk.
> **Compustat**: `act`, `lct` ($t$ and $t-1$)
```math
\Delta \text{Current Ratio}_{t} = \text{Current Ratio}_{t} - \text{Current Ratio}_{t-1}
```

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
