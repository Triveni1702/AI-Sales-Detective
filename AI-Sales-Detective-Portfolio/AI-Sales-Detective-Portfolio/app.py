from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PROJECT_FOLDER = Path(__file__).parent

st.set_page_config(
    page_title="AI Sales Detective",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Sales Detective")
st.write("Investigate sales trends and discover what changed.")

REQUIRED_COLUMNS = ["date", "units_sold", "unit_price", "website_visitors"]


def analyze_sales(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["units_sold", "unit_price", "website_visitors"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if df.isnull().any().any():
        raise ValueError(
            "Some values are missing or invalid. Check your date, "
            "units_sold, unit_price and website_visitors columns."
        )

    if (df[["units_sold", "unit_price", "website_visitors"]] < 0).any().any():
        raise ValueError("Sales values cannot be negative.")

    if df["date"].duplicated().any():
        raise ValueError("Please use one row per date.")

    df = df.sort_values("date").reset_index(drop=True)
    df["revenue"] = df["units_sold"] * df["unit_price"]
    return df


def percent_change(old, new):
    if old == 0:
        return 0.0 if new == 0 else None
    return (new - old) / abs(old) * 100


# Sidebar: upload data
st.sidebar.header("📁 Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a sales CSV file using the sidebar to get started.")
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
    data = analyze_sales(raw_data)
except (FileNotFoundError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Unable to load sales data: {error}")
    st.info("Required CSV columns: date, units_sold, unit_price, website_visitors.")
    st.stop()

st.sidebar.success(f"Loaded: {source_name}")
st.sidebar.write(f"Records: {len(data)}")

if len(data) < 2:
    st.warning("Please provide at least two dated sales records.")
    st.dataframe(data)
    st.stop()

# Sidebar: flexible comparison settings
st.sidebar.header("⚙️ Analysis Settings")
split_position = st.sidebar.slider(
    "Records in previous period",
    min_value=1,
    max_value=len(data) - 1,
    value=max(1, len(data) // 2),
    help="Choose how many of the earliest records belong to the previous period.",
)

previous = data.iloc[:split_position]
recent = data.iloc[split_position:]

# Calculate metrics
total_revenue = data["revenue"].sum()
total_units = data["units_sold"].sum()
total_visitors = data["website_visitors"].sum()
average_price = data["unit_price"].mean()

previous_revenue = previous["revenue"].sum()
recent_revenue = recent["revenue"].sum()
previous_units = previous["units_sold"].sum()
recent_units = recent["units_sold"].sum()
previous_visitors = previous["website_visitors"].sum()
recent_visitors = recent["website_visitors"].sum()
previous_price = previous["unit_price"].mean()
recent_price = recent["unit_price"].mean()

# Overview
st.subheader("📊 Sales Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,.0f}")
c3.metric("Website Visitors", f"{total_visitors:,.0f}")
c4.metric("Average Unit Price", f"₹{average_price:,.2f}")

st.caption(
    f"Data period: {data['date'].min().date()} to "
    f"{data['date'].max().date()}"
)

st.divider()

# Revenue trend
st.subheader("📈 Revenue Trend")
st.line_chart(data.set_index("date")[["revenue"]])

# Period comparison
st.subheader("🔍 Previous Period vs Recent Period")
comparison = pd.DataFrame(
    {
        "Previous Period": [
            previous_revenue,
            previous_units,
            previous_visitors,
            previous_price,
        ],
        "Recent Period": [
            recent_revenue,
            recent_units,
            recent_visitors,
            recent_price,
        ],
    },
    index=[
        "Revenue (₹)",
        "Units Sold",
        "Website Visitors",
        "Average Unit Price (₹)",
    ],
)
st.bar_chart(comparison)

left, right = st.columns(2)
with left:
    st.markdown("#### Previous Period")
    st.write(f"{previous['date'].min().date()} to {previous['date'].max().date()}")
    st.metric("Revenue", f"₹{previous_revenue:,.2f}")
    st.metric("Units Sold", f"{previous_units:,.0f}")
    st.metric("Website Visitors", f"{previous_visitors:,.0f}")

with right:
    st.markdown("#### Recent Period")
    st.write(f"{recent['date'].min().date()} to {recent['date'].max().date()}")
    st.metric("Revenue", f"₹{recent_revenue:,.2f}")
    st.metric("Units Sold", f"{recent_units:,.0f}")
    st.metric("Website Visitors", f"{recent_visitors:,.0f}")

st.divider()

# Machine learning
st.subheader("🤖 Machine Learning: Sales Prediction")
st.write("Choose which information the model should use to predict units sold.")

if len(data) < 5:
    st.warning(
        "At least five records are needed for this demonstration. "
        "More historical data is recommended."
    )
else:
    ml_data = data.copy()
    ml_data["day_number"] = (ml_data["date"] - ml_data["date"].min()).dt.days

    st.markdown("#### Choose Model Features")
    use_visitors = st.checkbox("Use website visitors", value=True)
    use_price = st.checkbox("Use unit price", value=True)

    feature_columns = ["day_number"]
    if use_visitors:
        feature_columns.append("website_visitors")
    if use_price:
        feature_columns.append("unit_price")

    st.caption(
        "The date is always included. Select or deselect the other "
        "features to see how model performance changes."
    )

    X = ml_data[feature_columns]
    X_date = ml_data[["day_number"]]
    y = ml_data["units_sold"]

    # Use earlier records for training and later records for testing
    split = int(len(ml_data) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    X_date_train = X_date.iloc[:split]
    X_date_test = X_date.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train the date-only comparison model
    date_model = LinearRegression()
    date_model.fit(X_date_train, y_train)
    date_predictions = date_model.predict(X_date_test)

    # Train the model using the selected features
    selected_model = LinearRegression()
    selected_model.fit(X_train, y_train)
    selected_predictions = selected_model.predict(X_test)

    date_mae = mean_absolute_error(y_test, date_predictions)
    selected_mae = mean_absolute_error(y_test, selected_predictions)

    st.write("**Features currently used:** " + ", ".join(feature_columns))
    m1, m2 = st.columns(2)
    m1.metric("Date-only model MAE", f"{date_mae:.2f} units")
    m2.metric(
        "Selected-feature model MAE",
        f"{selected_mae:.2f} units",
        delta=f"{date_mae - selected_mae:+.2f} units vs date-only",
    )

    if selected_mae < date_mae:
        st.success("The selected-feature model had lower error on these test records.")
    elif selected_mae > date_mae:
        st.info("The date-only model had lower error on these test records.")
    else:
        st.info("Both models had the same error on these test records.")

    st.caption(
        "MAE is the average absolute prediction error; lower is better. "
        "Small datasets can give misleading results, so treat this "
        "comparison as experimental."
    )

    # Display actual sales against both model predictions
    test_results = pd.DataFrame(
        {
            "Actual units": y_test.to_numpy(),
            "Date-only prediction": date_predictions,
            "Selected-feature prediction": selected_predictions,
        },
        index=ml_data["date"].iloc[split:],
    )
    st.markdown("#### Actual vs Predicted Sales")
    st.line_chart(test_results)

    with st.expander("View test predictions"):
        st.dataframe(test_results.round(2))

    # Customizable next-day forecast
    st.markdown("#### Customize Next-Day Prediction")
    latest_visitors = float(data["website_visitors"].iloc[-1])
    latest_price = float(data["unit_price"].iloc[-1])

    p1, p2 = st.columns(2)
    with p1:
        forecast_visitors = st.number_input(
            "Expected website visitors",
            min_value=0,
            value=int(latest_visitors),
            step=100,
        )
    with p2:
        forecast_price = st.number_input(
            "Expected unit price (₹)",
            min_value=0.0,
            value=float(latest_price),
            step=10.0,
        )

    next_day_number = int(ml_data["day_number"].max()) + 1
    next_values = {
        "day_number": [next_day_number],
        "website_visitors": [forecast_visitors],
        "unit_price": [forecast_price],
    }
    next_features = pd.DataFrame(next_values)[feature_columns]
    next_prediction = float(selected_model.predict(next_features)[0])

    st.metric("Predicted units for next day", f"{max(0, next_prediction):.1f}")
    st.caption(
        "This estimate uses your selected features and the assumptions "
        "you entered. It is experimental, not a guaranteed forecast."
    )

st.divider()

# Sales anomaly detection
st.subheader("🚨 Unusual Sales Detection")
st.write(
    "Identify sales values that are unusually high or low "
    "compared with the rest of the uploaded dataset."
)

q1 = data["units_sold"].quantile(0.25)
q3 = data["units_sold"].quantile(0.75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

anomalies = data[
    (data["units_sold"] < lower_limit) | (data["units_sold"] > upper_limit)
].copy()

st.caption(
    f"Expected range using the IQR method: "
    f"{max(0, lower_limit):.1f} to {upper_limit:.1f} units."
)

if anomalies.empty:
    st.success("No unusually high or low sales values were detected.")
else:
    st.warning(
        f"{len(anomalies)} unusual sales record(s) detected. "
        "Review these dates to see whether they need investigation."
    )
    st.dataframe(
        anomalies[
            ["date", "units_sold", "unit_price", "website_visitors", "revenue"]
        ]
    )
    st.download_button(
        "Download unusual sales records",
        data=anomalies.to_csv(index=False).encode("utf-8"),
        file_name="sales_anomalies.csv",
        mime="text/csv",
    )

# Investigation report
st.subheader("🕵️ Sales Investigation Report")

revenue_change = percent_change(previous_revenue, recent_revenue)
units_change = percent_change(previous_units, recent_units)
visitors_change = percent_change(previous_visitors, recent_visitors)
price_change = percent_change(previous_price, recent_price)

if revenue_change is None:
    st.info("Previous-period revenue was zero, so a percentage change cannot be calculated.")
elif revenue_change < 0:
    st.error(f"Revenue declined by {abs(revenue_change):.2f}%.")
elif revenue_change > 0:
    st.success(f"Revenue increased by {revenue_change:.2f}%.")
else:
    st.info("Revenue remained unchanged between the two periods.")

st.markdown("#### Findings")


def describe_change(label, change):
    if change is None:
        return f"{label}: percentage change is unavailable because the previous-period value was zero."
    if change < -0.01:
        return f"{label} decreased by {abs(change):.2f}%."
    if change > 0.01:
        return f"{label} increased by {change:.2f}%."
    return f"{label} remained approximately stable."


st.write("- " + describe_change("Revenue", revenue_change))
st.write("- " + describe_change("Units sold", units_change))
st.write("- " + describe_change("Website visitors", visitors_change))
st.write("- " + describe_change("Average unit price", price_change))

st.markdown("#### 🔍 Root-Cause Analysis")

if revenue_change is not None and revenue_change < -0.01:
    st.error("Primary finding: Revenue declined because fewer units were sold.")

    if visitors_change is not None and abs(visitors_change) <= 0.01:
        st.write(
            "**Website traffic was stable.** The decline in total visitors "
            "does not explain the sales drop."
        )

    if price_change is not None and abs(price_change) <= 0.01:
        st.write(
            "**Average unit price was stable.** The revenue decline was "
            "not associated with a reduction in the average selling price."
        )

    if units_change is not None and units_change < -0.01:
        st.write(
            f"**Observed sales impact:** Units sold fell by "
            f"{abs(units_change):.2f}%, from {previous_units:,.0f} "
            f"to {recent_units:,.0f} units."
        )

    st.markdown("**Possible underlying causes to investigate:**")
    st.markdown(
        "- Lower conversion: visitors may be less likely to purchase.\n"
        "- Product availability: popular products may have been out of stock.\n"
        "- Customer demand: interest in the product may have weakened.\n"
        "- Website or checkout issues: customers may have had difficulty ordering.\n"
        "- Marketing quality: traffic may have included fewer purchase-ready visitors."
    )
    st.warning(
        "These are hypotheses, not confirmed causes. The current dataset "
        "does not contain orders, conversion rates, inventory, marketing "
        "campaigns, or checkout data needed to identify the actual cause."
    )

elif revenue_change is not None and revenue_change > 0.01:
    st.success("Revenue increased during the recent period.")

    if units_change is not None and units_change > 0.01:
        st.write(
            f"Units sold increased by {units_change:.2f}%, which contributed "
            "directly to the higher revenue."
        )

    if visitors_change is not None and visitors_change > 0.01:
        st.write(
            f"Website visitors increased by {visitors_change:.2f}%. "
            "This occurred alongside the revenue increase, but does not "
            "prove that additional traffic caused it."
        )

    if price_change is not None and price_change > 0.01:
        st.write(
            f"Average unit price increased by {price_change:.2f}%, "
            "which may also have contributed to higher revenue."
        )

else:
    st.info(
        "Revenue did not change substantially. Inspect the individual "
        "metrics and compare additional periods for more insight."
    )

st.markdown("#### Recommended checks")
st.markdown(
    "- Compare conversion rate (orders or units per visitor) if order-level data is available.\n"
    "- Check stock-outs, product mix, discounts, refunds, and checkout issues.\n"
    "- Compare equivalent-length periods and collect more historical data before making decisions."
)

st.caption(
    "This report describes patterns in the uploaded data, not proven causes. "
    "The dataset is small, so treat the findings as exploratory."
)

# View and download
with st.expander("View sales data"):
    st.dataframe(data)

csv_data = data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download analyzed sales data",
    data=csv_data,
    file_name="analyzed_sales.csv",
    mime="text/csv",
)
