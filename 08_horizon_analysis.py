"""
08_horizon_analysis.py

Forecast-horizon diagnostic.

Goal:
    Measure how prediction difficulty changes as we forecast farther
    into the future.

Horizons tested:
    1, 5, 10, 30, 60, 300, 600 samples

IMPORTANT:
    This script uses the ORIGINAL cleaned data and evaluates
    direct horizon baselines. It does not train LSTM/Transformer
    models and does not use future values as features.

For each sensor/horizon we compare:
    1. Persistence:
       prediction = current/latest known value

    2. Recent Mean:
       prediction = mean of the previous 30 observations

    3. Linear Trend:
       fit a simple line to the previous 30 observations and
       extrapolate to the requested horizon

For each horizon:
    MAE, RMSE, R2

This tells us:
    - how quickly predictability decays
    - whether a simple baseline is sufficient
    - which sensors justify a more advanced model
    - whether a one-hour forecast is realistic at the current
      sampling rate

Input:
    outputs/data/sensor_data_cleaned.csv

Outputs:
    outputs/reports/horizon_analysis.csv
    outputs/reports/horizon_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

# Number of observations into the future.
HORIZONS = [
    1,
    5,
    10,
    30,
    60,
    300,
    600,
]

# History used by recent-mean and trend baselines.
LOOKBACK = 30

# To keep the script reasonably fast on CPU.
MAX_EVALUATION_POINTS = 10000


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

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

    try:
        r2 = r2_score(
            y_true,
            y_pred
        )
    except Exception:
        r2 = np.nan

    return mae, rmse, r2


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("FORECAST HORIZON ANALYSIS")
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
    df.dropna(
        subset=[TIMESTAMP_COL]
    )
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
# ACTUAL SAMPLING INFORMATION
# ============================================================

intervals = (
    df[TIMESTAMP_COL]
    .diff()
    .dt.total_seconds()
    .dropna()
)

median_interval = intervals.median()
mean_interval = intervals.mean()

print(
    f"\nMedian sampling interval: "
    f"{median_interval:.4f} seconds"
)

print(
    f"Mean sampling interval:   "
    f"{mean_interval:.4f} seconds"
)

print("\nApproximate future time represented by each horizon:")

for h in HORIZONS:

    approx_seconds = (
        h * median_interval
    )

    print(
        f"  {h:4d} samples ≈ "
        f"{approx_seconds:10.2f} seconds"
    )


# ============================================================
# EVALUATION INDICES
# ============================================================

# We need LOOKBACK observations before the forecast origin
# and HORIZON observations after it.

max_horizon = max(HORIZONS)

first_origin = LOOKBACK
last_origin = len(df) - max_horizon - 1

if last_origin <= first_origin:
    raise ValueError(
        "Dataset is too small for requested horizons."
    )

all_origins = np.arange(
    first_origin,
    last_origin + 1
)

if len(all_origins) > MAX_EVALUATION_POINTS:

    # Evenly sample through the entire testable period so that
    # we do not accidentally evaluate only one part of the day.
    positions = np.linspace(
        0,
        len(all_origins) - 1,
        MAX_EVALUATION_POINTS
    ).astype(int)

    origins = all_origins[
        positions
    ]

else:

    origins = all_origins

print(
    f"\nForecast origins evaluated: "
    f"{len(origins):,}"
)


# ============================================================
# HORIZON ANALYSIS
# ============================================================

records = []

for sensor in SENSOR_COLS:

    print("\n" + "=" * 75)
    print(f"SENSOR: {sensor}")
    print("=" * 75)

    values = df[sensor].to_numpy(
        dtype=float
    )

    for horizon in HORIZONS:

        print(
            f"\nHorizon: {horizon} samples"
        )

        y_true_list = {
            "Persistence": [],
            "RecentMean30": [],
            "LinearTrend30": [],
        }

        y_pred_list = {
            "Persistence": [],
            "RecentMean30": [],
            "LinearTrend30": [],
        }

        for origin in origins:

            history_start = (
                origin - LOOKBACK
            )

            history_end = origin

            future_index = (
                origin + horizon
            )

            if future_index >= len(values):
                continue

            history = values[
                history_start:history_end
            ]

            actual = values[
                future_index
            ]

            if (
                len(history) < LOOKBACK
                or not np.all(
                    np.isfinite(history)
                )
                or not np.isfinite(actual)
            ):
                continue

            # ------------------------------------------------
            # 1. Persistence
            # ------------------------------------------------

            persistence = history[-1]

            # ------------------------------------------------
            # 2. Recent mean
            # ------------------------------------------------

            recent_mean = np.mean(
                history
            )

            # ------------------------------------------------
            # 3. Linear trend
            # ------------------------------------------------

            x = np.arange(
                LOOKBACK,
                dtype=float
            )

            try:

                slope, intercept = np.polyfit(
                    x,
                    history,
                    1
                )

                trend_x = (
                    LOOKBACK - 1 + horizon
                )

                trend_prediction = (
                    intercept
                    + slope * trend_x
                )

            except Exception:

                trend_prediction = recent_mean

            predictions = {
                "Persistence": persistence,
                "RecentMean30": recent_mean,
                "LinearTrend30": trend_prediction,
            }

            for model_name, prediction in predictions.items():

                y_true_list[
                    model_name
                ].append(actual)

                y_pred_list[
                    model_name
                ].append(prediction)

        for model_name in y_true_list:

            mae, rmse, r2 = calculate_metrics(
                y_true_list[model_name],
                y_pred_list[model_name]
            )

            records.append({
                "sensor": sensor,
                "horizon_samples": horizon,
                "approx_horizon_seconds":
                    horizon * median_interval,
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "evaluation_points":
                    len(
                        y_true_list[model_name]
                    ),
            })

            print(
                f"  {model_name:15s} | "
                f"MAE={mae:.6f} | "
                f"RMSE={rmse:.6f} | "
                f"R2={r2:.6f}"
            )


# ============================================================
# SAVE FULL RESULTS
# ============================================================

results = pd.DataFrame(records)

results_file = os.path.join(
    REPORT_DIR,
    "horizon_analysis.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# BEST BASELINE PER SENSOR/HORIZON
# ============================================================

best_records = []

for sensor in SENSOR_COLS:

    for horizon in HORIZONS:

        subset = results[
            (results["sensor"] == sensor)
            &
            (
                results["horizon_samples"]
                == horizon
            )
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["RMSE"].idxmin()
        ]

        best_records.append({
            "sensor": sensor,
            "horizon_samples": horizon,
            "approx_horizon_seconds":
                horizon * median_interval,
            "best_baseline": best["model"],
            "best_RMSE": best["RMSE"],
            "best_MAE": best["MAE"],
            "best_R2": best["R2"],
        })

best_df = pd.DataFrame(
    best_records
)

summary_file = os.path.join(
    REPORT_DIR,
    "horizon_summary.csv"
)

best_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# ONE-HOUR HORIZON REFERENCE
# ============================================================

one_hour_samples = int(
    round(
        3600 / median_interval
    )
)

print("\n" + "=" * 75)
print("ONE-HOUR FORECAST REFERENCE")
print("=" * 75)

print(
    f"Median interval: "
    f"{median_interval:.4f} seconds"
)

print(
    f"Approximate samples in 1 hour: "
    f"{one_hour_samples:,}"
)

print(
    "\nNOTE:"
)

print(
    "The tested 600-sample horizon is only about "
    f"{600 * median_interval / 60:.1f} minutes."
)

print(
    "A true one-hour forecast is therefore substantially "
    "harder than the longest diagnostic horizon above."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("HORIZON ANALYSIS COMPLETED")
print("=" * 75)

print(
    f"\nFull report:\n{results_file}"
)

print(
    f"\nBest-baseline report:\n{summary_file}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Do not interpret a low RMSE alone as evidence that a "
    "sensor is predictable. Compare R2, persistence behavior, "
    "and how performance changes with horizon."
)

print(
    "\nNo final neural network has been trained by this script."
)

print("\n" + "=" * 75)
print("END")
print("=" * 75)
