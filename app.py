from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
STATS_PATH = BASE_DIR / "models" / "store_stats.pkl"
STORE_PATH = BASE_DIR / "data" / "store.csv"

STORE_TYPE_MAP = {"a": 1, "b": 2, "c": 3, "d": 4}
ASSORTMENT_MAP = {"a": 1, "b": 2, "c": 3}


# ============================================================
# LOAD FILES (cached so they load only once)
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_stats():
    return joblib.load(STATS_PATH)


@st.cache_data
def load_stores():
    return pd.read_csv(STORE_PATH).set_index("Store")


# ============================================================
# BUILD THE MODEL INPUT
# ============================================================

def _number(value, default=0):
    """Return value as a float, or default if it is missing."""
    return default if pd.isna(value) else float(value)


def build_features(store_id, expected_customers, promo, day,
                   stores, stats, feature_order):
    """
    Build one row of model input, using the SAME logic as train.py.
    Every store-specific value is looked up from store.csv and
    store_stats.pkl, so different stores give different inputs.
    """

    if store_id not in stores.index:
        raise ValueError(f"Store {store_id} was not found in store.csv")

    store = stores.loc[store_id]
    store_stats = stats["store_stats"]

    # --- date features ---
    year = day.year
    month = day.month
    week = day.isocalendar()[1]

    # --- store info from store.csv ---
    store_type = STORE_TYPE_MAP.get(str(store["StoreType"]).strip(), 0)
    assortment = ASSORTMENT_MAP.get(str(store["Assortment"]).strip(), 0)

    distance = store["CompetitionDistance"]
    if pd.isna(distance):
        distance = stores["CompetitionDistance"].median()

    comp_year = store.get("CompetitionOpenSinceYear", np.nan)
    comp_month = store.get("CompetitionOpenSinceMonth", np.nan)

    if pd.isna(comp_year) or pd.isna(comp_month):
        comp_open_months = 0
    else:
        comp_open_months = max(
            0, (year - comp_year) * 12 + (month - comp_month)
        )

    promo2 = int(_number(store["Promo2"]))
    promo2_year = store.get("Promo2SinceYear", np.nan)
    promo2_week = store.get("Promo2SinceWeek", np.nan)

    is_promo2_active = 0
    if promo2 == 1 and not pd.isna(promo2_year) and not pd.isna(promo2_week):
        if year > promo2_year or (year == promo2_year and week >= promo2_week):
            is_promo2_active = 1

    # --- store statistics from training (store_stats.pkl) ---
    if store_id in store_stats.index:
        store_mean_sales = store_stats.loc[store_id, "StoreMeanSales"]
        avg_customers = store_stats.loc[store_id, "AvgCustomers"]
    else:
        store_mean_sales = stats["global_mean_sales"]
        avg_customers = stats["global_mean_customers"]

    row = {
        "Store": store_id,
        "DayOfWeek": day.weekday() + 1,   # Monday = 1 ... Sunday = 7
        "Promo": promo,
        "SchoolHoliday": 0,
        "StateHoliday": 0,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": distance,
        "CompetitionOpenMonths": comp_open_months,
        "Promo2": promo2,
        "IsPromo2Active": is_promo2_active,
        "Year": year,
        "Month": month,
        "Day": day.day,
        "WeekOfYear": week,
        "Customers": expected_customers,
        "StoreMeanSales": store_mean_sales,
        "AvgCustomers": avg_customers,
    }

    # Same columns, same order as the model was trained with.
    return pd.DataFrame([row])[list(feature_order)].astype(np.float32)


# ============================================================
# APP
# ============================================================

def main():

    st.set_page_config(
        page_title="Retail Sales Forecasting",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Retail Sales Forecasting")
    st.write("Predict retail sales using your machine learning model.")

    for path in (MODEL_PATH, STATS_PATH, STORE_PATH):
        if not path.exists():
            st.error(f"File not found: {path}")
            st.stop()

    model = load_model()
    stats = load_stats()
    stores = load_stores()

    feature_order = getattr(model, "feature_names_in_", None)

    if feature_order is None:
        st.error("Model has no saved feature names. Retrain with train.py.")
        st.stop()

    # ---------- sidebar inputs ----------
    st.sidebar.header("Store Information")

    store_id = st.sidebar.number_input(
        "Store ID",
        min_value=int(stores.index.min()),
        max_value=int(stores.index.max()),
        value=int(stores.index.min()),
        step=1
    )

    expected_customers = st.sidebar.number_input(
        "Expected Customers",
        min_value=0,
        value=500,
        step=100
    )

    promo_choice = st.sidebar.selectbox("Promotion Active?", ["Yes", "No"])

    day = st.sidebar.date_input("Date", value=date.today())

    # ---------- model information ----------
    with st.expander("Model information"):
        st.write("Model type:", type(model).__name__)
        st.write("Model features:", list(feature_order))
        st.write(
            "Expected Customers is an input, so this tool answers: "
            "'if this many customers visit, what will sales be?'"
        )

    if "Customers" not in feature_order:
        st.error(
            "Your trained model was NOT trained with a 'Customers' "
            "feature. Retrain it with train.py."
        )
        st.stop()

    # ---------- prediction ----------
    if st.button("Predict Sales"):

        st.subheader("Prediction Results")
        st.write(f"Store ID: {int(store_id)}")
        st.write(f"Expected customers: {expected_customers}")
        st.write(f"Promotion: {promo_choice}")
        st.write(f"Date: {day}")

        try:
            features = build_features(
                int(store_id),
                expected_customers,
                1 if promo_choice == "Yes" else 0,
                day,
                stores,
                stats,
                feature_order
            )

            # Show exactly what is being sent to the model
            with st.expander("Input sent to model"):
                st.dataframe(features.T.rename(columns={0: "Value"}))

            prediction = float(model.predict(features)[0])

            st.success(f"Predicted Sales: {prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()