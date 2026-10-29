import gc
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/train.csv"
STORE_PATH = "data/store.csv"

MODEL_PATH = "models/best_model.pkl"
STATS_PATH = "models/store_stats.pkl"

TEST_FRACTION = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("====================================")
print("LOADING DATASET")
print("====================================")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. "
        "Make sure train.csv is inside the data folder."
    )

print(f"Reading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded.")
print(f"Shape: {df.shape}")
print("\nColumns found:")
print(df.columns.tolist())


# ============================================================
# MERGE STORE DATA IF NECESSARY
# ============================================================

# Rossmann keeps StoreType, Assortment, Competition and Promo2
# columns in store.csv. Merge them in if train.csv lacks them.

if "StoreType" not in df.columns:

    print("\nStore information not found in train.csv.")
    print("Loading store.csv...")

    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(f"{STORE_PATH} was not found.")

    store_df = pd.read_csv(STORE_PATH, low_memory=False)

    print(f"store.csv shape: {store_df.shape}")

    df = df.merge(store_df, on="Store", how="left")

    del store_df

    print("train.csv and store.csv merged successfully.")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Sales",
    "Customers",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2"
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\nAll required columns are present.")


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning data...")

# Only train on days when the store was open.
if "Open" in df.columns:

    before = len(df)

    df = df[df["Open"] == 1].copy()

    print(f"Removed closed-store rows: {before - len(df):,}")

# Sales is the target. Zero-sales rows are data errors / closures
# (even on "open" days) and pull predictions toward 0.
df = df[df["Sales"].notna() & (df["Sales"] > 0)].copy()

print(f"Rows after cleaning: {len(df):,}")


# ============================================================
# DATE FEATURES
# ============================================================

print("\n====================================")
print("CREATING DATE FEATURES")
print("====================================")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.dropna(subset=["Date"]).copy()

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Monday = 1 ... Sunday = 7
df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

df["WeekOfYear"] = (
    df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)


# ============================================================
# STATE HOLIDAY / STORE TYPE / ASSORTMENT
# ============================================================

print("Encoding StateHoliday, StoreType, Assortment...")

df["StateHoliday"] = (
    df["StateHoliday"]
    .astype(str)
    .str.strip()
    .map({"0": 0, "a": 1, "b": 2, "c": 3})
    .fillna(0)
)

df["StoreType"] = (
    df["StoreType"]
    .astype(str)
    .str.strip()
    .map({"a": 1, "b": 2, "c": 3, "d": 4})
    .fillna(0)
)

df["Assortment"] = (
    df["Assortment"]
    .astype(str)
    .str.strip()
    .map({"a": 1, "b": 2, "c": 3})
    .fillna(0)
)


# ============================================================
# COMPETITION DISTANCE
# ============================================================

print("Processing competition distance...")

df["CompetitionDistance"] = pd.to_numeric(
    df["CompetitionDistance"],
    errors="coerce"
)

competition_median = df["CompetitionDistance"].median()

if pd.isna(competition_median):
    competition_median = 0

df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
    competition_median
)


# ============================================================
# COMPETITION OPEN MONTHS
# ============================================================

print("Creating CompetitionOpenMonths...")

if (
    "CompetitionOpenSinceYear" in df.columns
    and
    "CompetitionOpenSinceMonth" in df.columns
):

    since_year = pd.to_numeric(
        df["CompetitionOpenSinceYear"],
        errors="coerce"
    )

    since_month = pd.to_numeric(
        df["CompetitionOpenSinceMonth"],
        errors="coerce"
    )

    df["CompetitionOpenMonths"] = (
        (df["Year"] - since_year) * 12
        +
        (df["Month"] - since_month)
    )

    df["CompetitionOpenMonths"] = (
        df["CompetitionOpenMonths"]
        .clip(lower=0)
        .fillna(0)
    )

else:

    df["CompetitionOpenMonths"] = 0


# ============================================================
# PROMO2 ACTIVE
# ============================================================

print("Creating IsPromo2Active...")

if (
    "Promo2SinceYear" in df.columns
    and
    "Promo2SinceWeek" in df.columns
):

    since_year = pd.to_numeric(
        df["Promo2SinceYear"],
        errors="coerce"
    )

    since_week = pd.to_numeric(
        df["Promo2SinceWeek"],
        errors="coerce"
    )

    df["IsPromo2Active"] = (
        (df["Promo2"] == 1)
        &
        (
            (df["Year"] > since_year)
            |
            (
                (df["Year"] == since_year)
                &
                (df["WeekOfYear"] >= since_week)
            )
        )
    ).astype(np.int8)

else:

    df["IsPromo2Active"] = 0


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n====================================")
print("CREATING TIME-BASED TRAIN/TEST SPLIT")
print("====================================")

df = df.sort_values("Date").reset_index(drop=True)

cutoff_index = int(len(df) * (1 - TEST_FRACTION))
cutoff_date = df["Date"].iloc[cutoff_index]

train_df = df[df["Date"] < cutoff_date].copy()
test_df = df[df["Date"] >= cutoff_date].copy()

# Free memory: the full frame is no longer needed.
del df
gc.collect()

print(f"Train rows : {len(train_df):,}")
print(f"Test rows  : {len(test_df):,}")
print(f"Cutoff date: {cutoff_date.date()}")


# ============================================================
# STORE STATISTICS (TRAIN ONLY, to avoid leakage)
# ============================================================

print("\n====================================")
print("CREATING STORE STATISTICS")
print("====================================")

store_stats = (
    train_df
    .groupby("Store")
    .agg(
        StoreMeanSales=("Sales", "mean"),
        AvgCustomers=("Customers", "mean")
    )
)

global_mean_sales = train_df["Sales"].mean()
global_mean_customers = train_df["Customers"].mean()


def add_store_statistics(frame):

    frame["StoreMeanSales"] = (
        frame["Store"]
        .map(store_stats["StoreMeanSales"])
        .fillna(global_mean_sales)
    )

    frame["AvgCustomers"] = (
        frame["Store"]
        .map(store_stats["AvgCustomers"])
        .fillna(global_mean_customers)
    )

    return frame


train_df = add_store_statistics(train_df)
test_df = add_store_statistics(test_df)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenMonths",
    "Promo2",
    "IsPromo2Active",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",

    # Customers is intentionally included so the app can predict
    # sales from the "Expected Customers" input.
    "Customers",

    "StoreMeanSales",
    "AvgCustomers"
]

TARGET = "Sales"


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\n====================================")
print("PREPARING TRAINING DATA")
print("====================================")


def prepare_features(frame):

    X = frame[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # float32 uses about half the RAM of float64.
    X = X.astype(np.float32)

    y = pd.to_numeric(frame[TARGET], errors="coerce")
    y = y.fillna(0).astype(np.float32)

    return X, y


X_train, y_train = prepare_features(train_df)
X_test, y_test = prepare_features(test_df)

del train_df, test_df
gc.collect()

print("\nFeatures:")

for feature in FEATURES:
    print("  ", feature)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

print(
    f"X_train memory: "
    f"{X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n====================================")
print("TRAINING MODEL")
print("====================================")

print("Using HistGradientBoostingRegressor...")

# Much more memory-efficient than a 200-tree Random Forest.

model = HistGradientBoostingRegressor(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42,
    early_stopping=False
)

print("\nStarting training...")
print("Please wait. Do NOT close the terminal.")

model.fit(X_train, y_train)

print("\nTraining finished successfully.")


# ============================================================
# EVALUATION
# ============================================================

print("\n====================================")
print("EVALUATING MODEL")
print("====================================")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n====================================")
print("FEATURE IMPORTANCE")
print("====================================")

# HistGradientBoostingRegressor has no feature_importances_,
# so use permutation importance on a RANDOM test sample.
# (X_test is sorted by date, so taking the first N rows would
# only cover the first couple of weeks of the test period.)

try:

    sample_size = min(10000, len(X_test))

    X_importance = X_test.sample(sample_size, random_state=42)
    y_importance = y_test.loc[X_importance.index]

    print(f"Calculating importance on {sample_size:,} test rows...")

    permutation = permutation_importance(
        model,
        X_importance,
        y_importance,
        n_repeats=2,
        random_state=42,
        n_jobs=2
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": permutation.importances_mean
    })

    importance = importance.sort_values("Importance", ascending=False)

    print(importance.to_string(index=False))

except Exception as error:

    print("Feature importance calculation skipped:")
    print(error)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n====================================")
print("SAVING MODEL")
print("====================================")

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH, compress=3)

# Statistics required by app.py at prediction time.
joblib.dump(
    {
        "store_stats": store_stats,
        "global_mean_sales": global_mean_sales,
        "global_mean_customers": global_mean_customers
    },
    STATS_PATH,
    compress=3
)

print("\n====================================")
print("MODEL SAVED SUCCESSFULLY")
print("====================================")

print(f"Model saved to : {MODEL_PATH}")
print(f"Stats saved to : {STATS_PATH}")


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\n====================================")
print("VERIFYING SAVED MODEL")
print("====================================")

loaded_model = joblib.load(MODEL_PATH)
loaded_stats = joblib.load(STATS_PATH)

print("Saved model and stats loaded successfully.")

# The reloaded model must reproduce the same predictions.
check_rows = X_test.iloc[:100]

if not np.allclose(loaded_model.predict(check_rows), y_pred[:100]):
    raise RuntimeError("Reloaded model gives different predictions!")

print("Reloaded model reproduces the original predictions.")

saved_features = getattr(loaded_model, "feature_names_in_", None)

print("\nModel expects:")

if saved_features is not None:
    for feature in saved_features:
        print("  ", feature)

print("\nCustomers included:")

if saved_features is not None:
    print("YES" if "Customers" in saved_features else "NO")
else:
    print("YES - Customers is part of FEATURES")

print("\n====================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("====================================")