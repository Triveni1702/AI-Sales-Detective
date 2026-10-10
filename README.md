# AI Sales Detective

**AI Sales Detective** is a Python and Streamlit dashboard for exploring sales data, identifying changes over time, estimating unit sales with a simple regression model, flagging potential outliers, and presenting investigation findings in one place.

> **Scope note:** This is a small, exploratory analytics project. Forecasts are estimates, outlier flags need review, and suggested explanations are hypotheses—not proof of why sales changed.

## Demonstration Video

Watch the AI Sales Detective demonstration

https://drive.google.com/file/d/1aFjhRThmxaaxMOHpQAL-RMWHHz5O5OCS/view?usp=sharing

The video file is included in this repository as `demo.mp4`. If you later upload the video to YouTube, Google Drive, or another hosting service, replace the link above with the shareable URL.

## Key Features

### 1. Sales overview
- Displays total revenue, units sold, website visitors, and average unit price.
- Shows the date range covered by the uploaded dataset.

### 2. Revenue trend visualization
- Plots revenue over time to make increases and decreases easier to spot.

### 3. Previous-versus-recent period comparison
- Splits the sorted data into earlier and recent records.
- Compares revenue, units sold, website visitors, and average unit price.
- Periods may contain different numbers of dates, so compare like-for-like durations when drawing business conclusions.

### 4. Machine-learning sales estimate
- Uses Scikit-learn's `LinearRegression` model to estimate units sold.
- Compares a date-only model against a model using selected features.
- Reports Mean Absolute Error (MAE), displays actual versus predicted sales, and provides an experimental next-day estimate.
- A small dataset can make model metrics unstable; MAE is meaningful only relative to the evaluation data and baseline.

### 5. Unusual sales detection
- Uses the Interquartile Range (IQR) rule to flag unusually high or low unit-sales values.
- Provides a downloadable CSV of flagged records when any are found.
- An outlier is a record for review, not automatically an error or fraud.

### 6. Sales investigation report
- Summarizes observed percentage changes.
- Offers possible areas to investigate—conversion rates, stock availability, customer demand, website/checkout issues, and marketing quality—when relevant.
- Clearly distinguishes observable trends from unverified root-cause hypotheses.

### 7. Data inspection and export
- Lets users view the processed records and download analyzed data.
- Checks for missing or invalid values, negative values, duplicate dates, and required columns.

## Project Structure

```text
AI-Sales-Detective/
├── app.py                 # Streamlit web dashboard
├── main.py                # Command-line sales analysis and investigation summary
├── train_model.py         # Standalone time-based regression training/evaluation demo
├── requirements.txt       # Python dependencies
├── demo.mp4               # Project demonstration video
├── DEMO_SCRIPT.txt        # Suggested demonstration flow
└── data/
    ├── sales.csv          # Main sample dataset
    ├── test_sales.csv     # Additional sample data
    ├── bad_sales.csv      # Missing/invalid numeric values for validation testing
    ├── bad_date.csv       # Invalid date for validation testing
    └── negative_sales.csv # Negative sales value for validation testing
```

## Dataset Format

The Streamlit app expects a CSV file with these required columns:

| Column | Description | Example |
|---|---|---:|
| `date` | Date of the sales observation | `2026-10-01` |
| `units_sold` | Number of units sold | `120` |
| `unit_price` | Price per unit | `500` |
| `website_visitors` | Website visitor count | `5000` |

Use one row per date. Dates must be valid, numeric fields must contain valid non-negative values, and all required columns must be present. The included CSV files are sample/testing data and should not be treated as real business results.

## Installation and Usage

### 1. Install Python

Use Python 3.10 or newer.

### 2. Open a terminal in the project directory

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The dashboard waits for a CSV upload; it does not automatically display the sample dataset before a file is selected.

### 5. Run the standalone analysis scripts (optional)

```bash
python main.py
python train_model.py
```

These two scripts use `data/sales.csv` directly. `main.py` prints a comparison and investigation summary to the terminal. `train_model.py` demonstrates a simple chronological train/test split, linear regression, MAE, and a next-day estimate.

## How the Analysis Works

1. **Read and validate:** load the CSV, check required fields, convert dates/numbers, and reject invalid or negative values.
2. **Calculate metrics:** compute row-level revenue as `units_sold × unit_price`, then aggregate summary statistics.
3. **Compare periods:** split records into earlier and recent groups and calculate percentage changes.
4. **Estimate sales:** fit a linear regression model using time and optional selected features; evaluate using MAE on later records.
5. **Flag potential outliers:** calculate quartiles and IQR for `units_sold`, then apply the conventional 1.5 × IQR thresholds.
6. **Summarize observations:** display measured changes and suggest further checks without claiming to prove causation.

## Limitations

- The supplied sample has very few rows, which is not enough for dependable business forecasting.
- Historical patterns may not persist; the linear model is a baseline, not a production forecasting system.
- A low MAE on a tiny test set does not guarantee future accuracy.
- Period totals can be misleading when the compared periods have different lengths.
- The app's root-cause section does not establish causation. Confirming the cause would require more context such as orders, conversion rates, inventory, campaigns, pricing changes, and checkout logs.
- The standalone `main.py` and `train_model.py` scripts read the bundled sample file, whereas the Streamlit dashboard analyzes the CSV uploaded by the user.

## Conclusion

AI Sales Detective demonstrates how Python data processing, interactive dashboards, basic regression, IQR-based outlier detection, and readable reporting can be combined into one workflow. It helps users see **what changed** in a sales dataset and identify **what to investigate next**. Its findings should be treated as exploratory decision support, with conclusions verified using more complete data before making business decisions.

## Author

**Triveni Reddyboina**  
Computer Science and Engineering (CSE)

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly (dependency available for visualizations)

---

Feedback and improvements are welcome. Potential next steps include more historical data, stronger time-series validation, comparable-length period comparisons, more robust data quality checks, and evaluation of multiple forecasting baselines.
