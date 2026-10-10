from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PROJECT_FOLDER = Path(__file__).parent
DATA_FILE = PROJECT_FOLDER / "data" / "sales.csv"


def main():
    """Train a simple time-based regression model and report its error."""
    df = pd.read_csv(DATA_FILE)
    required = {"date", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if len(df) < 4:
        raise ValueError("At least four records are required for this train/test demonstration.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df = df.dropna(subset=["date", "units_sold"]).sort_values("date").reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Not enough valid records remain after data cleaning.")

    df["day_number"] = (df["date"] - df["date"].min()).dt.days
    X = df[["day_number"]]
    y = df["units_sold"]
    split = int(len(df) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_train) < 2 or X_test.empty:
        raise ValueError("Not enough data to train and test the model.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("\n=== AI Sales Detective: ML Model ===")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Actual units sold: {y_test.tolist()}")
    print(f"Predicted units sold: {[round(value, 2) for value in predictions]}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} units")

    next_day = X["day_number"].max() + 1
    next_prediction = model.predict(pd.DataFrame({"day_number": [next_day]}))[0]
    print(f"Predicted units sold for next day: {max(0, round(next_prediction))}")
    print("Note: This is an experimental estimate, not a guaranteed forecast.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nError: {error}")
