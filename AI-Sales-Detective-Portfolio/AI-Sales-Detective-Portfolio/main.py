from pathlib import Path
import pandas as pd

PROJECT_FOLDER = Path(__file__).parent
DATA_FILE = PROJECT_FOLDER / "data" / "sales.csv"


def load_data():
    """Load and validate the sample sales data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Sales CSV file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required_columns = ["date", "units_sold", "unit_price", "website_visitors"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("The sales CSV file is empty.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("The CSV contains invalid dates.")

    numeric_columns = ["units_sold", "unit_price", "website_visitors"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Invalid numeric values in column: {column}")
        if (df[column] < 0).any():
            raise ValueError(f"Negative values are not allowed in: {column}")

    df["revenue"] = df["units_sold"] * df["unit_price"]
    return df.sort_values("date").reset_index(drop=True)


def split_periods(df):
    """Split data into earlier and recent groups."""
    if len(df) < 2:
        raise ValueError("At least two rows are needed for comparison.")
    midpoint = len(df) // 2
    return df.iloc[:midpoint], df.iloc[midpoint:]


def percentage_change(old_value, new_value):
    """Calculate percentage change, including a zero baseline."""
    if old_value == 0:
        return 0.0 if new_value == 0 else None
    return ((new_value - old_value) / abs(old_value)) * 100


def analyze_sales():
    """Print a simple console-based sales investigation summary."""
    df = load_data()
    previous, recent = split_periods(df)
    metrics = {
        "Revenue": (previous["revenue"].sum(), recent["revenue"].sum()),
        "Units sold": (previous["units_sold"].sum(), recent["units_sold"].sum()),
        "Website visitors": (previous["website_visitors"].sum(), recent["website_visitors"].sum()),
        "Average unit price": (previous["unit_price"].mean(), recent["unit_price"].mean()),
    }

    print("\n" + "=" * 42)
    print("             AI SALES DETECTIVE")
    print("=" * 42)
    print(f"Previous period: {previous['date'].min().date()} to {previous['date'].max().date()}")
    print(f"Recent period:   {recent['date'].min().date()} to {recent['date'].max().date()}")
    print("\n--- Period comparison ---")
    changes = {}
    for label, (old, new) in metrics.items():
        change = percentage_change(old, new)
        changes[label] = change
        change_text = "N/A (previous value was zero)" if change is None else f"{change:+.2f}%"
        print(f"{label}: {old:,.2f} -> {new:,.2f} ({change_text})")

    revenue_change = changes["Revenue"]
    print("\n--- Investigation summary ---")
    if revenue_change is None:
        print("Previous-period revenue was zero, so a percentage change cannot be calculated.")
    elif revenue_change < 0:
        print(f"Revenue decreased by {abs(revenue_change):.2f}% across the selected periods.")
        print("Review units sold, website traffic, pricing, stock availability, and conversion data.")
    elif revenue_change > 0:
        print(f"Revenue increased by {revenue_change:.2f}% across the selected periods.")
        print("Check whether the increase aligns with units sold, traffic, or pricing.")
    else:
        print("Revenue remained unchanged across the selected periods.")
    print("Conclusion: These are observed patterns, not proof of what caused the change.")


if __name__ == "__main__":
    try:
        analyze_sales()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nError: {error}")
