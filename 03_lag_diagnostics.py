"""
03_lag_diagnostics.py

Purpose:
Determine how much predictive information is contained in previous
observations before choosing the final forecasting architecture.

Tests:
- Persistence / lag-1
- lag-2
- lag-5
- lag-10
- lag-30
- lag-60
- moving averages using past observations

Rules:
- Chronological split only.
- No future values are used in features.
- Metrics are calculated on the TEST portion.
- Vibration signals are NOT cleaned or clipped here.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

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

LAGS = [1, 2, 5, 10, 30, 60]
ROLLING_WINDOWS = [5, 15, 30, 60]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ============================================================
# HELPERS
# ============================================================

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-8

    if not np.any(mask):
        return 0.0

    return float(
        np.mean(
            2.0 * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        ) * 100.0
    )


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "sMAPE_percent": smape(y_true, y_pred),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("TIME-SERIES LAG / PREDICTABILITY DIAGNOSTIC")
print("=" * 75)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{DATA_FILE}\n\n"
        "Run 01_data_pipeline.py first."
    )

df = pd.read_csv(DATA_FILE)

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

df = df.dropna(subset=[TIMESTAMP_COL]).copy()
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

for sensor in SENSOR_COLS:
    df[sensor] = pd.to_numeric(
        df[sensor],
        errors="coerce"
    )

df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

print(f"\nRows: {len(df):,}")

# ============================================================
# CHRONOLOGICAL TEST SPLIT
# ============================================================

n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

test_df = df.iloc[val_end:].copy()

print(f"Train rows: {train_end:,}")
print(f"Validation rows: {val_end - train_end:,}")
print(f"Test rows: {len(test_df):,}")

# ============================================================
# BUILD DIAGNOSTICS
# ============================================================

results = []

for sensor in SENSOR_COLS:

    print("\n" + "-" * 75)
    print(f"SENSOR: {sensor}")
    print("-" * 75)

    # --------------------------------------------------------
    # LAG PREDICTORS
    # --------------------------------------------------------

    for lag in LAGS:

        prediction = df[sensor].shift(lag)

        # Only evaluate the TEST timestamps.
        y_true = df.loc[val_end:, sensor].to_numpy(dtype=float)
        y_pred = prediction.loc[val_end:].to_numpy(dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        m = metrics(
            y_true[mask],
            y_pred[mask]
        )

        results.append({
            "sensor": sensor,
            "method": f"Lag-{lag}",
            "lag": lag,
            "window": np.nan,
            **m
        })

        print(
            f"Lag-{lag:>2}: "
            f"MAE={m['MAE']:.6f} | "
            f"RMSE={m['RMSE']:.6f} | "
            f"R2={m['R2']:.6f}"
        )

    # --------------------------------------------------------
    # MOVING AVERAGE PREDICTORS
    # --------------------------------------------------------

    for window in ROLLING_WINDOWS:

        # shift(1) is critical:
        # the current target is never included in its own feature.
        prediction = (
            df[sensor]
            .shift(1)
            .rolling(window)
            .mean()
        )

        y_true = df.loc[val_end:, sensor].to_numpy(dtype=float)
        y_pred = prediction.loc[val_end:].to_numpy(dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        m = metrics(
            y_true[mask],
            y_pred[mask]
        )

        results.append({
            "sensor": sensor,
            "method": f"MovingAverage-{window}",
            "lag": np.nan,
            "window": window,
            **m
        })

        print(
            f"MA-{window:>2}:   "
            f"MAE={m['MAE']:.6f} | "
            f"RMSE={m['RMSE']:.6f} | "
            f"R2={m['R2']:.6f}"
        )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    REPORT_DIR,
    "lag_predictability_diagnostics.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

# ============================================================
# BEST METHOD PER SENSOR
# ============================================================

best_rows = []

for sensor in SENSOR_COLS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()

    # RMSE is primary because it penalizes missed spikes
    # more strongly than MAE.
    best = sensor_results.sort_values(
        ["RMSE", "MAE"],
        ascending=True
    ).iloc[0]

    best_rows.append(best)

best_df = pd.DataFrame(best_rows)

best_path = os.path.join(
    REPORT_DIR,
    "best_lag_method_per_sensor.csv"
)

best_df.to_csv(
    best_path,
    index=False
)

# ============================================================
# VISUALIZE RMSE BY LAG
# ============================================================

for sensor in SENSOR_COLS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ]

    lag_results = sensor_results[
        sensor_results["method"].str.startswith("Lag-")
    ].copy()

    lag_results = lag_results.sort_values("lag")

    plt.figure(figsize=(10, 5))

    plt.plot(
        lag_results["lag"],
        lag_results["RMSE"],
        marker="o",
        label="RMSE"
    )

    plt.xticks(LAGS)
    plt.xlabel("Lag (samples)")
    plt.ylabel("RMSE")
    plt.title(f"{sensor} - Predictability by Lag")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"{sensor}_lag_predictability.png"
        ),
        dpi=150
    )

    plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("BEST SIMPLE PREDICTOR FOR EACH SENSOR")
print("=" * 75)

for _, row in best_df.iterrows():

    print(
        f"{row['sensor']:15s} -> "
        f"{row['method']:20s} | "
        f"RMSE={row['RMSE']:.6f} | "
        f"MAE={row['MAE']:.6f} | "
        f"R2={row['R2']:.6f}"
    )

print("\nFiles created:")
print(f"  {results_path}")
print(f"  {best_path}")
print("  outputs/plots/*_lag_predictability.png")

print("\n" + "=" * 75)
print("DIAGNOSTIC COMPLETED")
print("=" * 75)

print(
    "\nNext step: send me the terminal output. "
    "We will use these results to design the final forecasting model."
)
