# Seasonal Sales Forecasting (Rossmann)
# Run from the project root:  python seasonal_forecast.py

# %%
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"    # future dates (Aug-Sep 2015). It has no Sales column.
TEST_DAYS = 56                 # last 8 weeks are kept aside as the test set

os.makedirs("reports", exist_ok=True)

# %%
# 1. Load and prepare the data (total sales per day)
raw = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=["Date"])

# StateHoliday: "0" = no holiday, "a"/"b"/"c" = holiday  ->  1 / 0
raw["StateHoliday"] = raw["StateHoliday"].astype(str).ne("0").astype(int)


def to_daily(frame, with_sales=True):
    """One row per day, built from the store-level rows."""
    agg = {
        "PromoShare": ("Promo", "mean"),                  # share of stores running a promo
        "StateHolidayShare": ("StateHoliday", "mean"),    # share of stores on a state holiday
        "SchoolHolidayShare": ("SchoolHoliday", "mean"),
    }
    if with_sales:
        agg["Sales"] = ("Sales", "sum")
    return frame.groupby("Date").agg(**agg)


daily = to_daily(raw)

# make sure every calendar day exists
daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
daily.index.name = "Date"

print("Days:", len(daily), "|", daily.index.min().date(), "to", daily.index.max().date())
print("Missing days:", int(daily["Sales"].isna().sum()))

# %%
# 2. Visualize: sales over time
fig, ax = plt.subplots()
ax.plot(daily.index, daily["Sales"], lw=0.6, label="Daily sales")
ax.plot(daily.index, daily["Sales"].rolling(28).mean(), lw=2, label="28-day average")
ax.set_title("Total daily sales (all stores)")
ax.legend()
plt.savefig("reports/01_sales_over_time.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Weekly pattern
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_avg = daily["Sales"].groupby(daily.index.dayofweek).mean()
weekday_avg.index = weekday_names

weekday_avg.plot(kind="bar", rot=0, title="Average total sales by day of week")
plt.savefig("reports/02_weekly_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

sunday_ratio = weekday_avg["Sun"] / weekday_avg.drop("Sun").mean()
print(f"Sunday sales are {sunday_ratio:.0%} of an average Mon-Sat day (most stores are closed).")

# %%
# Seasonal pattern
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_avg = daily["Sales"].groupby(daily.index.month).mean()
month_avg.index = [month_names[m - 1] for m in month_avg.index]

month_avg.plot(kind="bar", rot=0, title="Average total daily sales by month")
plt.savefig("reports/03_seasonal_pattern.png", dpi=150, bbox_inches="tight")
plt.show()

print("Strongest month:", month_avg.idxmax(), "| Weakest month:", month_avg.idxmin())

# %%
# Promotion and holiday effects
open_rows = raw[(raw["Open"] == 1) & (raw["Sales"] > 0)]
promo_avg = open_rows.groupby("Promo")["Sales"].mean()

non_sunday = daily[daily.index.dayofweek != 6]          # remove the Sunday effect
holiday_flag = non_sunday["StateHolidayShare"] > 0.5
holiday_avg = non_sunday["Sales"].groupby(holiday_flag).mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
promo_avg.rename({0: "No promo", 1: "Promo"}).plot(
    kind="bar", rot=0, ax=axes[0], title="Avg sales per open store-day")
holiday_avg.rename({False: "Normal day", True: "State holiday"}).plot(
    kind="bar", rot=0, ax=axes[1], title="Avg total sales (Mon-Sat only)")
plt.savefig("reports/04_promo_holiday.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Promo lifts sales per open store-day by {promo_avg[1] / promo_avg[0] - 1:.0%}")
if True in holiday_avg.index and False in holiday_avg.index:
    print(f"State-holiday days are {holiday_avg[True] / holiday_avg[False] - 1:+.0%} vs normal Mon-Sat days")

# %%
# 3. Features: calendar + event flags + previous-sales (past days only)
LAGS = [7, 14, 28]
CALENDAR_COLS = ["DayOfWeek", "Month", "WeekOfYear", "DayOfMonth", "IsWeekend"]
EVENT_COLS = ["PromoShare", "StateHolidayShare", "SchoolHolidayShare"]
HISTORY_COLS = [f"Lag{lag}" for lag in LAGS] + ["Roll7"]


def add_calendar(frame):
    frame["DayOfWeek"] = frame.index.dayofweek + 1            # Monday = 1 ... Sunday = 7
    frame["Month"] = frame.index.month
    frame["WeekOfYear"] = frame.index.isocalendar().week.astype(int).values
    frame["DayOfMonth"] = frame.index.day
    frame["IsWeekend"] = (frame["DayOfWeek"] >= 6).astype(int)
    return frame


def add_history_features(frame):
    for lag in LAGS:
        frame[f"Lag{lag}"] = frame["Sales"].shift(lag)
    frame["Roll7"] = frame["Sales"].shift(1).rolling(7).mean()   # previous 7 days only
    return frame


def make_linear():
    """Improved Linear Regression: log-sales target + scaled features + light regularisation."""
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()),
                            ("ridge", Ridge(alpha=1.0))]),
        func=np.log1p, inverse_func=np.expm1)


def make_forest():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)


def make_X(frame, linear):
    """Model input. Linear Regression needs weekday and month as one-hot columns,
    because it would otherwise treat December as '12 times January'.
    Its lag columns are put on a log scale to match the log target."""
    X = frame[CALENDAR_COLS + EVENT_COLS + HISTORY_COLS].copy()
    if linear:
        X[HISTORY_COLS] = np.log1p(X[HISTORY_COLS])
        X["DayOfWeek"] = pd.Categorical(X["DayOfWeek"], categories=range(1, 8))
        X["Month"] = pd.Categorical(X["Month"], categories=range(1, 13))
        X = pd.get_dummies(X, columns=["DayOfWeek", "Month"], drop_first=True, dtype=float)
        X = X.drop(columns=["WeekOfYear", "DayOfMonth", "IsWeekend"])
    return X


data = add_history_features(add_calendar(daily.copy()))
data = data.dropna(subset=["Sales"] + EVENT_COLS + HISTORY_COLS)

print("Rows available for modelling:", len(data))

# %%
# 4. Chronological split (last 8 weeks = test)
train = data.iloc[:-TEST_DAYS]
test = data.iloc[-TEST_DAYS:]

print(f"Train: {train.index.min().date()} to {train.index.max().date()}  ({len(train)} days)")
print(f"Test : {test.index.min().date()} to {test.index.max().date()}  ({len(test)} days)")

# %%
# 5. Train and compare models (test A: real past sales used as lag inputs)
y_train, y_test = train["Sales"], test["Sales"]
NAIVE = "Naive (same day last week)"

predictions = {NAIVE: test["Lag7"].to_numpy()}

linear_model = make_linear().fit(make_X(train, linear=True), y_train)
predictions["Linear Regression"] = linear_model.predict(make_X(test, linear=True))

forest = make_forest().fit(make_X(train, linear=False), y_train)
predictions["Random Forest"] = forest.predict(make_X(test, linear=False))


def score_table(preds):
    rows = [{"Model": name,
             "MAE": mean_absolute_error(y_test, p),
             "RMSE": np.sqrt(mean_squared_error(y_test, p))}
            for name, p in preds.items()]
    return pd.DataFrame(rows).set_index("Model").sort_values("RMSE").round(1)


results = score_table(predictions)
results.to_csv("reports/model_comparison.csv")
print("\nTest A - one-step-ahead (uses real past sales as lags):")
print(results)

# %%
fig, ax = plt.subplots()
ax.plot(test.index, y_test, color="black", lw=2, label="Actual")
for name, pred in predictions.items():
    ax.plot(test.index, pred, lw=1.2, label=name)
ax.set_title("Test period: actual vs predicted total sales")
ax.legend()
plt.savefig("reports/05_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
importance = pd.Series(forest.feature_importances_, index=make_X(train, linear=False).columns)
importance.sort_values().tail(8).plot(kind="barh", title="Random Forest: top features")
plt.savefig("reports/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# 5b. Test B - the fair check: forecast the last 8 weeks day by day.
# Only data before the test period is known; after that the model
# uses its OWN predictions as lags, exactly like a real future forecast.
def recursive_eval(model, linear):
    start = len(daily) - TEST_DAYS
    frame = add_calendar(daily[["Sales"] + EVENT_COLS].copy())
    sales = frame["Sales"].to_numpy(dtype=float).copy()
    sales[start:] = np.nan                      # hide the test weeks
    for i in range(start, len(frame)):
        row = frame.iloc[[i]].copy()
        for lag in LAGS:
            row[f"Lag{lag}"] = sales[i - lag]
        row["Roll7"] = sales[i - 7:i].mean()
        sales[i] = model.predict(make_X(row, linear=linear))[0]
    return sales[start:]


rec_preds = {
    "Linear Regression": recursive_eval(linear_model, linear=True),
    "Random Forest": recursive_eval(forest, linear=False),
}
# naive: repeat the last known week
last_week = daily["Sales"].iloc[-TEST_DAYS - 7:-TEST_DAYS].to_numpy()
rec_preds[NAIVE] = np.tile(last_week, TEST_DAYS // 7 + 1)[:TEST_DAYS]

rec_results = score_table(rec_preds)
rec_results.to_csv("reports/model_comparison_recursive.csv")
print("\nTest B - recursive (real forecasting, own predictions as lags):")
print(rec_results)

fig, ax = plt.subplots()
ax.plot(test.index, y_test, color="black", lw=2, label="Actual")
for name, pred in rec_preds.items():
    ax.plot(test.index, pred, lw=1.2, label=name)
ax.set_title("Recursive check: 8 weeks forecast day by day")
ax.legend()
plt.savefig("reports/08_recursive_check.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# 6. Retrain the best model on all history
# The final forecast is recursive, so the best model is chosen on Test B.
naive_rmse = rec_results.loc[NAIVE, "RMSE"]
candidates = rec_results.drop(index=NAIVE)
best_name = candidates["RMSE"].idxmin()
best_linear = best_name == "Linear Regression"

print("\nBest model (recursive test):", best_name)
if candidates["RMSE"].min() >= naive_rmse:
    print("WARNING: no model beats the naive baseline. Be honest about this in your report.")

final_model = make_linear() if best_linear else make_forest()
final_model.fit(make_X(data, linear=best_linear), data["Sales"])

# %%
# Forecast future dates, one day at a time
assert os.path.exists(TEST_PATH), f"{TEST_PATH} not found. It is needed for the future promo / holiday flags."

future_raw = pd.read_csv(TEST_PATH, low_memory=False, parse_dates=["Date"])
future_raw["StateHoliday"] = future_raw["StateHoliday"].astype(str).ne("0").astype(int)
future_daily = to_daily(future_raw, with_sales=False)

future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), future_daily.index.max(), freq="D")
future_daily = future_daily.reindex(future_dates).ffill().fillna(0)
future_daily["Sales"] = np.nan

full = pd.concat([daily[["Sales"] + EVENT_COLS], future_daily[["Sales"] + EVENT_COLS]])
full.index.name = "Date"
full = add_calendar(full)

n_hist = len(daily)
sales = full["Sales"].to_numpy(dtype=float).copy()

for i in range(n_hist, len(full)):
    row = full.iloc[[i]].copy()
    for lag in LAGS:
        row[f"Lag{lag}"] = sales[i - lag]
    row["Roll7"] = sales[i - 7:i].mean()
    sales[i] = final_model.predict(make_X(row, linear=best_linear))[0]

full["Sales"] = sales
forecast = full.iloc[n_hist:].rename(columns={"Sales": "ForecastSales"})[["ForecastSales"] + EVENT_COLS].copy()

print(f"Forecast: {forecast.index.min().date()} to {forecast.index.max().date()} ({len(forecast)} days)")

# %%
# High / low demand days
# Demand index = forecast / typical sales for that weekday over the last 365 days.
# This removes the weekday effect, so Saturdays are not all labelled "low".
HIGH, LOW = 1.10, 0.90

last_year = daily["Sales"].iloc[-365:]
weekday_baseline = last_year.groupby(last_year.index.dayofweek).mean()

forecast["Weekday"] = forecast.index.day_name().str[:3]
forecast["DemandIndex"] = forecast["ForecastSales"] / forecast.index.dayofweek.map(weekday_baseline).to_numpy()
forecast["Demand"] = np.select(
    [forecast["DemandIndex"] >= HIGH, forecast["DemandIndex"] <= LOW],
    ["High", "Low"],
    default="Normal",
)
forecast.loc[forecast.index.dayofweek == 6, "Demand"] = "Sunday"   # stores mostly closed: not judged

print(forecast["Demand"].value_counts().to_string())
print("(Sundays are not judged because most stores are closed.)")
forecast.to_csv("reports/seasonal_forecast.csv")
print(forecast[forecast["Demand"] != "Normal"][
    ["Weekday", "ForecastSales", "PromoShare", "StateHolidayShare", "DemandIndex", "Demand"]].round(2))

# %%
recent = daily["Sales"].iloc[-90:]

fig, ax = plt.subplots()
ax.plot(recent.index, recent, label="Actual (last 90 days)")
ax.plot(forecast.index, forecast["ForecastSales"], color="tab:orange", label="Forecast")
for label, colour in [("High", "green"), ("Low", "red")]:
    points = forecast[forecast["Demand"] == label]
    ax.scatter(points.index, points["ForecastSales"], color=colour, zorder=3, label=f"{label} demand")
ax.set_title("Sales forecast with high / low demand days")
ax.legend()
plt.savefig("reports/06_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
