"""
06_temporal_resolution.py

Find the most useful temporal resolution for each sensor group.

We compare rolling/aggregated representations at:
    1s, 5s, 10s, 30s, 60s

The goal is NOT to train a forecasting model yet.

We measure:
    - number of samples
    - coverage
    - mean/std
    - variability
    - lag-1 autocorrelation
    - correlation with previous time window
    - persistence baseline difficulty

This helps decide whether magnetic sensors become more predictable
after temporal aggregation, while keeping vibration/temperature at
a resolution that preserves useful behavior.

Input:
    outputs/data/sensor_data_cleaned.csv

Output:
    outputs/reports/temporal_resolution_comparison.csv
    outputs/plots/temporal_resolution_<sensor>.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

PLOT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "plots"
)

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

RESOLUTIONS = [1, 5, 10, 30, 60]

# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("TEMPORAL RESOLUTION ANALYSIS")
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
    .set_index(TIMESTAMP_COL)
)

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print(f"\nRows: {len(df):,}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")


# ============================================================
# ANALYSIS
# ============================================================

records = []

for seconds in RESOLUTIONS:

    print("\n" + "-" * 75)
    print(f"Resolution: {seconds} second(s)")
    print("-" * 75)

    # Mean aggregation is appropriate for comparing the general
    # signal level across temporal resolutions.
    if seconds == 1:
        aggregated = df[SENSOR_COLS].resample("1s").mean()
    else:
        aggregated = df[SENSOR_COLS].resample(
            f"{seconds}s"
        ).mean()

    for col in SENSOR_COLS:

        series = aggregated[col].dropna()

        if len(series) < 3:
            continue

        # Lag-1 autocorrelation.
        lag_corr = series.autocorr(lag=1)

        # Persistence error:
        # predict current aggregated value from previous value.
        actual = series.iloc[1:].to_numpy()
        previous = series.iloc[:-1].to_numpy()

        errors = actual - previous

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        mean_value = series.mean()
        std_value = series.std()

        # Coefficient of variation of first differences.
        diff_std = series.diff().std()

        # Quantiles help identify whether aggregation destroys
        # important extremes.
        q99 = series.quantile(0.99)
        q01 = series.quantile(0.01)

        records.append({
            "resolution_seconds": seconds,
            "sensor": col,
            "samples": len(series),
            "mean": mean_value,
            "std": std_value,
            "q01": q01,
            "q99": q99,
            "lag1_autocorrelation": lag_corr,
            "persistence_MAE": mae,
            "persistence_RMSE": rmse,
            "difference_std": diff_std,
        })

        print(
            f"{col:15s} | "
            f"n={len(series):6d} | "
            f"lag1={lag_corr:8.4f} | "
            f"persist_RMSE={rmse:12.6f} | "
            f"diff_std={diff_std:12.6f}"
        )

results = pd.DataFrame(records)

output_file = os.path.join(
    REPORT_DIR,
    "temporal_resolution_comparison.csv"
)

results.to_csv(
    output_file,
    index=False
)


# ============================================================
# BEST RESOLUTION BY PERSISTENCE RMSE
# ============================================================

print("\n" + "=" * 75)
print("RESOLUTION COMPARISON")
print("=" * 75)

best_records = []

for col in SENSOR_COLS:

    sensor_results = results[
        results["sensor"] == col
    ].copy()

    if sensor_results.empty:
        continue

    best_row = sensor_results.loc[
        sensor_results["persistence_RMSE"].idxmin()
    ]

    best_records.append({
        "sensor": col,
        "best_resolution_seconds":
            int(best_row["resolution_seconds"]),
        "best_persistence_RMSE":
            best_row["persistence_RMSE"],
        "best_lag1_autocorrelation":
            best_row["lag1_autocorrelation"],
    })

best_df = pd.DataFrame(best_records)

print(
    best_df.to_string(index=False)
)

best_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "best_temporal_resolution.csv"
    ),
    index=False
)


# ============================================================
# PLOTS
# ============================================================

print("\nCreating temporal-resolution plots...")

for col in SENSOR_COLS:

    subset = results[
        results["sensor"] == col
    ].sort_values(
        "resolution_seconds"
    )

    # Persistence RMSE
    plt.figure(figsize=(9, 5))

    plt.plot(
        subset["resolution_seconds"],
        subset["persistence_RMSE"],
        marker="o"
    )

    plt.xlabel("Temporal resolution (seconds)")
    plt.ylabel("Persistence RMSE")
    plt.title(
        f"Persistence difficulty vs resolution — {col}"
    )

    plt.xticks(RESOLUTIONS)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"temporal_resolution_{col}.png"
        ),
        dpi=150
    )

    plt.close()

    # Lag-1 autocorrelation
    plt.figure(figsize=(9, 5))

    plt.plot(
        subset["resolution_seconds"],
        subset["lag1_autocorrelation"],
        marker="o"
    )

    plt.xlabel("Temporal resolution (seconds)")
    plt.ylabel("Lag-1 autocorrelation")
    plt.title(
        f"Temporal dependence vs resolution — {col}"
    )

    plt.xticks(RESOLUTIONS)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"temporal_autocorrelation_{col}.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# FINAL GUIDANCE
# ============================================================

print("\n" + "=" * 75)
print("TEMPORAL RESOLUTION ANALYSIS COMPLETED")
print("=" * 75)

print(
    f"\nMain report:\n{output_file}"
)

print(
    "\nInterpretation:"
)

print(
    "Higher lag-1 autocorrelation generally means the signal "
    "has stronger short-term temporal persistence."
)

print(
    "Lower persistence RMSE means a simple previous-value "
    "forecast is already more effective at that resolution."
)

print(
    "\nIMPORTANT:"
)

print(
    "Do not select a resolution using only one metric. "
    "We will consider predictability AND whether aggregation "
    "destroys meaningful vibration spikes or magnetic behavior."
)

print(
    "\nNo final model has been trained by this script."
)
