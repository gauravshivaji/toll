import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

# --------------------------
# Load and preprocess data
# --------------------------
def load_data(uploaded_file):
    # Accept file-like uploaded_file from Streamlit or a local path string
    if hasattr(uploaded_file, "read"):
        name = getattr(uploaded_file, "name", "")
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        # path string
        if str(uploaded_file).lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    df = df.rename(columns=lambda c: c.strip())
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Safe column selection for counts and amounts
    count_cols = [c for c in df.columns if "Count" in c and "Total" not in c]
    amt_cols = [c for c in df.columns if "Amount" in c]

    if len(count_cols) == 0:
        st.warning("No '*Count*' columns detected — ensure your file has columns containing 'Count'.")
    if len(amt_cols) == 0:
        st.warning("No '*Amount*' columns detected — ensure your file has columns containing 'Amount'.")

    df["Total_Count"] = df[count_cols].sum(axis=1) if len(count_cols) > 0 else 0
    df["Total_Revenue"] = df[amt_cols].sum(axis=1) if len(amt_cols) > 0 else 0

    if "Toll Plaza Location" not in df.columns:
        df["Toll Plaza Location"] = "Unknown"

    return df

def make_features(df):
    df = df.sort_values(["Toll Plaza Location", "Date"]).copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df["Date"].dt.month

    # Lag features (1) and rolling mean (3)
    df["total_count_lag1"] = df.groupby("Toll Plaza Location")["Total_Count"].shift(1)
    df["total_revenue_lag1"] = df.groupby("Toll Plaza Location")["Total_Revenue"].shift(1)

    df["total_count_roll3"] = (
        df.groupby("Toll Plaza Location")["Total_Count"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["total_revenue_roll3"] = (
        df.groupby("Toll Plaza Location")["Total_Revenue"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # Drop rows where lag features are missing
    df = df.dropna(subset=["total_count_lag1", "total_revenue_lag1"])
    return df

# --------------------------
# Train models
# --------------------------
def train_models(df):
    df_fe = make_features(df)
    feature_cols = [c for c in df_fe.columns if c not in ["Date", "Toll Plaza Location", "Total_Count", "Total_Revenue"]]

    X = df_fe[feature_cols].fillna(0)
    y_count = df_fe["Total_Count"]
    y_revenue = df_fe["Total_Revenue"]

    # Simple random split (for more rigorous eval use time-based split per plaza)
    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_count, y_revenue, test_size=0.2, random_state=42
    )

    model_count = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rev = RandomForestRegressor(n_estimators=100, random_state=42)

    model_count.fit(X_train, y_train_c)
    model_rev.fit(X_train, y_train_r)

    pred_c = model_count.predict(X_test)
    pred_r = model_rev.predict(X_test)

    st.write("### Model performance (test split)")
    st.write(f"- Count MAE: {mean_absolute_error(y_test_c, pred_c):,.2f}")
    st.write(f"- Count RMSE: {mean_squared_error(y_test_c, pred_c, squared=False):,.2f}")
    st.write(f"- Revenue MAE: {mean_absolute_error(y_test_r, pred_r):,.2f}")
    st.write(f"- Revenue RMSE: {mean_squared_error(y_test_r, pred_r, squared=False):,.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model_count, "features": feature_cols}, "models/count_model.joblib")
    joblib.dump({"model": model_rev, "features": feature_cols}, "models/revenue_model.joblib")

    st.success("Models trained and saved to ./models")

# --------------------------
# Predict single future day
# --------------------------
def predict_future(df, date_input, plaza):
    if not os.path.exists("models/count_model.joblib") or not os.path.exists("models/revenue_model.joblib"):
        st.error("Models not found — train models first.")
        return

    count_info = joblib.load("models/count_model.joblib")
    rev_info = joblib.load("models/revenue_model.joblib")

    hist = df[df["Toll Plaza Location"] == plaza].sort_values("Date")
    if hist.shape[0] < 1:
        st.error("No historical data for selected toll plaza.")
        return

    last = hist.iloc[-1]

    feat = {
        "dayofweek": pd.to_datetime(date_input).dayofweek,
        "is_weekend": int(pd.to_datetime(date_input).dayofweek in [5, 6]),
        "month": pd.to_datetime(date_input).month,
        "total_count_lag1": last["Total_Count"],
        "total_revenue_lag1": last["Total_Revenue"],
        "total_count_roll3": hist["Total_Count"].tail(3).mean(),
        "total_revenue_roll3": hist["Total_Revenue"].tail(3).mean(),
    }

    X_pred = pd.DataFrame([feat]).reindex(columns=count_info["features"]).fillna(0)
    pred_count = count_info["model"].predict(X_pred)[0]
    pred_rev = rev_info["model"].predict(X_pred)[0]

    st.write("### Prediction")
    st.metric("Predicted Traffic Count", f"{int(pred_count):,}")
    st.metric("Predicted Revenue (₹)", f"{pred_rev:,.2f}")

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Toll Traffic & Revenue Predictor", layout="wide")
st.title("🚗 Toll Traffic & Revenue Prediction")

uploaded_file = st.file_uploader("Upload your Toll data (.xlsx or .csv)", type=["xlsx", "csv"])

# If you already have a local Excel file named Toll_Data_Jan2025.xlsx in the same folder,
# the app will fallback to it when no file is uploaded.
LOCAL_FALLBACK = "Toll_Data_Jan2025.xlsx"

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists(LOCAL_FALLBACK):
    df = load_data(LOCAL_FALLBACK)

if df is not None:
    st.write("### Data preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Models"):
            train_models(df)

    with col2:
        if os.path.exists("models/count_model.joblib") and os.path.exists("models/revenue_model.joblib"):
            st.success("Models are available in ./models")
        else:
            st.info("No trained models found. Train models to enable prediction.")

    st.markdown("---")
    st.subheader("Predict a future day")
    if os.path.exists("models/count_model.joblib"):
        date_input = st.date_input("Date for prediction")
        plaza = st.selectbox("Toll Plaza", df["Toll Plaza Location"].unique())
        if st.button("Predict"):
            predict_future(df, date_input, plaza)
    else:
        st.info("Train models first to use prediction.")
else:
    st.info("Upload your dataset (Excel or CSV) or place a file named 'Toll_Data_Jan2025.xlsx' in this folder.")
