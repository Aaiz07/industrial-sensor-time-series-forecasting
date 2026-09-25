"""
07_feature_predictability.py

Feature predictability analysis for the industrial sensor dataset.

Purpose:
    Determine whether a sensor can be predicted from:
      - its own recent history
      - other sensor channels
      - rolling statistics
      - first differences
      - time-of-day

This is a DIAGNOSTIC script.
It does not train LSTM/Transformer models.

Models used only as interpretable diagnostic models:
    1. Persistence baseline
    2. Linear Ridge regression
    3. Random Forest
    4. XGBoost (if available)

IMPORTANT:
    All train/test operations are chronological.
    Feature statistics are calculated from the training period only
    where applicable.

Input:
    outputs/data/sensor_data_cleaned.csv

Outputs:
    outputs/reports/feature_predictability_metrics.csv
    outputs/reports/feature_importance_<sensor>.csv
    outputs/reports/feature_predictability_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)

TIMESTAMP_COL = "ts"

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]

# Keep the diagnostic model compact enough to run on CPU.
LAGS = [1, 2, 5, 10, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Use a subset for the expensive Random Forest.
RF_MAX_TRAIN_ROWS = 20000

RANDOM_STATE = 42


# ============================================================
# METRIC FUNCTION
# ============================================================

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("FEATURE PREDICTABILITY ANALYSIS")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run 01_data_pipeline.py first."
    )

df = pd.read_csv(INPUT_FILE)

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

df = (
    df.dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .drop_duplicates(
        subset=[TIMESTAMP_COL],
        keep="first"
    )
    .reset_index(drop=True)
)

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print(f"\nRows: {len(df):,}")
print(f"Start: {df[TIMESTAMP_COL].min()}")
print(f"End:   {df[TIMESTAMP_COL].max()}")


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating multivariate lag/rolling features...")

data = df.copy()

# Time-of-day features.
seconds_of_day = (
    data[TIMESTAMP_COL].dt.hour * 3600
    + data[TIMESTAMP_COL].dt.minute * 60
    + data[TIMESTAMP_COL].dt.second
)

data["time_sin"] = np.sin(
    2 * np.pi * seconds_of_day / 86400
)

data["time_cos"] = np.cos(
    2 * np.pi * seconds_of_day / 86400
)

# Current sensor values are deliberately NOT used as predictors.
# Only past values are allowed.
feature_columns = []

for sensor in SENSOR_COLS:

    for lag in LAGS:

        name = f"{sensor}_lag_{lag}"

        data[name] = data[sensor].shift(lag)

        feature_columns.append(name)

    for window in ROLLING_WINDOWS:

        name = f"{sensor}_rollmean_{window}"

        # Shift first so the current target cannot leak into the feature.
        data[name] = (
            data[sensor]
            .shift(1)
            .rolling(window)
            .mean()
        )

        feature_columns.append(name)

        name = f"{sensor}_rollstd_{window}"

        data[name] = (
            data[sensor]
            .shift(1)
            .rolling(window)
            .std()
        )

        feature_columns.append(name)

# Time features are also valid predictors.
feature_columns.extend([
    "time_sin",
    "time_cos",
])

# Remove rows that cannot have complete historical features.
model_data = data[
    [TIMESTAMP_COL] + SENSOR_COLS + feature_columns
].dropna().reset_index(drop=True)

print(
    f"Usable rows after feature construction: "
    f"{len(model_data):,}"
)

print(
    f"Number of predictor features: "
    f"{len(feature_columns):,}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(model_data)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (TRAIN_RATIO + VAL_RATIO)
)

train = model_data.iloc[
    :train_end
].copy()

val = model_data.iloc[
    train_end:val_end
].copy()

test = model_data.iloc[
    val_end:
].copy()

print("\nChronological split:")
print(f"Train: {len(train):,}")
print(f"Val:   {len(val):,}")
print(f"Test:  {len(test):,}")

X_train = train[feature_columns]
X_test = test[feature_columns]


# ============================================================
# DIAGNOSTIC MODELS
# ============================================================

all_metrics = []
all_importance = []

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

if XGB_AVAILABLE:
    print("\nXGBoost is available.")
else:
    print("\nXGBoost not available. Continuing with Ridge + Random Forest.")


# ============================================================
# PER SENSOR ANALYSIS
# ============================================================

for target in SENSOR_COLS:

    print("\n" + "=" * 75)
    print(f"TARGET: {target}")
    print("=" * 75)

    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistence_pred = test[
        f"{target}_lag_1"
    ].to_numpy()

    mae, rmse, r2 = metrics(
        y_test,
        persistence_pred
    )

    all_metrics.append({
        "sensor": target,
        "model": "Persistence",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(
        f"Persistence  | "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"R2={r2:.6f}"
    )

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    ridge = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=10.0
            )
        )
    ])

    ridge.fit(
        X_train,
        y_train
    )

    ridge_pred = ridge.predict(
        X_test
    )

    mae, rmse, r2 = metrics(
        y_test,
        ridge_pred
    )

    all_metrics.append({
        "sensor": target,
        "model": "Ridge_Multivariate",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(
        f"Ridge        | "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"R2={r2:.6f}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if len(X_train) > RF_MAX_TRAIN_ROWS:

        rf_indices = np.linspace(
            0,
            len(X_train) - 1,
            RF_MAX_TRAIN_ROWS
        ).astype(int)

        X_rf = X_train.iloc[
            rf_indices
        ]

        y_rf = y_train[
            rf_indices
        ]

    else:

        X_rf = X_train
        y_rf = y_train

    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=2
    )

    rf.fit(
        X_rf,
        y_rf
    )

    rf_pred = rf.predict(
        X_test
    )

    mae, rmse, r2 = metrics(
        y_test,
        rf_pred
    )

    all_metrics.append({
        "sensor": target,
        "model": "RandomForest_Multivariate",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(
        f"RandomForest | "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"R2={r2:.6f}"
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGB_AVAILABLE:

        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE,
        )

        xgb.fit(
            X_train,
            y_train,
            verbose=False
        )

        xgb_pred = xgb.predict(
            X_test
        )

        mae, rmse, r2 = metrics(
            y_test,
            xgb_pred
        )

        all_metrics.append({
            "sensor": target,
            "model": "XGBoost_Multivariate",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(
            f"XGBoost      | "
            f"MAE={mae:.6f} | "
            f"RMSE={rmse:.6f} | "
            f"R2={r2:.6f}"
        )

        importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": xgb.feature_importances_
        }).sort_values(
            "importance",
            ascending=False
        )

        importance["sensor"] = target

        all_importance.append(
            importance
        )

        importance.to_csv(
            os.path.join(
                REPORT_DIR,
                f"feature_importance_{target}.csv"
            ),
            index=False
        )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_file = os.path.join(
    REPORT_DIR,
    "feature_predictability_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# BEST MODEL PER SENSOR
# ============================================================

best_rows = []

for sensor in SENSOR_COLS:

    subset = metrics_df[
        metrics_df["sensor"] == sensor
    ].copy()

    if subset.empty:
        continue

    best = subset.loc[
        subset["RMSE"].idxmin()
    ]

    persistence = subset[
        subset["model"] == "Persistence"
    ]

    persistence_rmse = (
        persistence["RMSE"].iloc[0]
        if not persistence.empty
        else np.nan
    )

    improvement_percent = (
        (
            persistence_rmse
            - best["RMSE"]
        )
        / max(
            abs(persistence_rmse),
            1e-12
        )
        * 100
    )

    best_rows.append({
        "sensor": sensor,
        "best_model": best["model"],
        "best_RMSE": best["RMSE"],
        "best_MAE": best["MAE"],
        "best_R2": best["R2"],
        "persistence_RMSE": persistence_rmse,
        "improvement_vs_persistence_percent":
            improvement_percent,
    })

best_df = pd.DataFrame(
    best_rows
)

best_file = os.path.join(
    REPORT_DIR,
    "feature_predictability_summary.csv"
)

best_df.to_csv(
    best_file,
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

if all_importance:

    importance_df = pd.concat(
        all_importance,
        ignore_index=True
    )

    print("\n" + "=" * 75)
    print("TOP PREDICTIVE FEATURES")
    print("=" * 75)

    for sensor in SENSOR_COLS:

        subset = importance_df[
            importance_df["sensor"] == sensor
        ].head(10)

        if subset.empty:
            continue

        print(f"\n{sensor}:")

        print(
            subset[
                ["feature", "importance"]
            ].to_string(index=False)
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FEATURE PREDICTABILITY SUMMARY")
print("=" * 75)

print(
    best_df.to_string(index=False)
)

print("\nReports saved:")
print(
    f"  {metrics_file}"
)
print(
    f"  {best_file}"
)

print(
    f"\nFeature-importance files saved in:\n"
    f"  {REPORT_DIR}"
)

print("\nIMPORTANT:")
print(
    "This analysis tells us whether other sensor channels and "
    "historical features contain predictive information. "
    "It does not yet select the final production model."
)

print(
    "\nDo not run the final forecasting script yet. "
    "Use these results to decide the final feature set and "
    "model architecture."
)

print("\n" + "=" * 75)
print("END")
print("=" * 75)
