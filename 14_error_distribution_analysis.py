import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 14: ERROR & DISTRIBUTION ANALYSIS
# ============================================================
# Purpose:
#   1. Compare TRAIN / VALIDATION / TEST distributions
#   2. Analyze prediction errors
#   3. Detect distribution shift
#   4. Check vibration spikes
#   5. Check model performance across time
#   6. Investigate why validation and test differ
#
# NO MODEL TRAINING IS DONE HERE.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FORECAST_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "future_features_60s.csv"
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "true_forecasting_test_predictions_60s.csv"
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

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

FUTURE_TARGET_COLS = [
    f"future_{sensor}"
    for sensor in SENSOR_COLS
]

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 14: ERROR & DISTRIBUTION ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "ts",
        "target_ts"
    ]
)

pred = pd.read_csv(
    PREDICTION_FILE,
    parse_dates=[
        "ts",
        "target_ts"
    ]
)

df = df.sort_values(
    "ts"
).reset_index(drop=True)

pred = pred.sort_values(
    "ts"
).reset_index(drop=True)

print(
    f"\nForecast dataset rows: {len(df):,}"
)

print(
    f"Prediction rows:       {len(pred):,}"
)

# ============================================================
# 2. CREATE CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)

train = df.iloc[
    :train_end
].copy()

val = df.iloc[
    train_end:val_end
].copy()

test = df.iloc[
    val_end:
].copy()

print("\nDATA PERIODS")

print(
    f"Train:"
    f"      {train['ts'].iloc[0]}"
    f" → {train['ts'].iloc[-1]}"
)

print(
    f"Validation:"
    f" {val['ts'].iloc[0]}"
    f" → {val['ts'].iloc[-1]}"
)

print(
    f"Test:"
    f"       {test['ts'].iloc[0]}"
    f" → {test['ts'].iloc[-1]}"
)

# ============================================================
# 3. DISTRIBUTION SUMMARY
# ============================================================

distribution_rows = []

for sensor in SENSOR_COLS:

    target = f"future_{sensor}"

    for name, data in [
        ("Train", train),
        ("Validation", val),
        ("Test", test)
    ]:

        values = data[target].dropna()

        distribution_rows.append({
            "sensor": sensor,
            "split": name,
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q01": values.quantile(0.01),
            "q05": values.quantile(0.05),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "q95": values.quantile(0.95),
            "q99": values.quantile(0.99),
            "max": values.max()
        })

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_file = os.path.join(
    REPORT_DIR,
    "step14_distribution_summary.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False
)

print("\nDISTRIBUTION SUMMARY")

for sensor in SENSOR_COLS:

    print("\n" + "-" * 70)
    print(sensor)

    subset = distribution_df[
        distribution_df["sensor"] == sensor
    ]

    print(
        subset[
            [
                "split",
                "mean",
                "std",
                "q95",
                "q99",
                "max"
            ]
        ].to_string(index=False)
    )

# ============================================================
# 4. DISTRIBUTION SHIFT
# ============================================================

shift_rows = []

for sensor in SENSOR_COLS:

    sensor_data = distribution_df[
        distribution_df["sensor"] == sensor
    ].set_index("split")

    train_mean = sensor_data.loc[
        "Train", "mean"
    ]

    train_std = sensor_data.loc[
        "Train", "std"
    ]

    for split in [
        "Validation",
        "Test"
    ]:

        split_mean = sensor_data.loc[
            split, "mean"
        ]

        split_std = sensor_data.loc[
            split, "std"
        ]

        mean_change_percent = (
            (split_mean - train_mean)
            / (abs(train_mean) + 1e-12)
            * 100
        )

        std_change_percent = (
            (split_std - train_std)
            / (abs(train_std) + 1e-12)
            * 100
        )

        shift_rows.append({
            "sensor": sensor,
            "split": split,
            "train_mean": train_mean,
            "split_mean": split_mean,
            "mean_change_percent":
                mean_change_percent,
            "train_std": train_std,
            "split_std": split_std,
            "std_change_percent":
                std_change_percent
        })

shift_df = pd.DataFrame(
    shift_rows
)

shift_file = os.path.join(
    REPORT_DIR,
    "step14_distribution_shift.csv"
)

shift_df.to_csv(
    shift_file,
    index=False
)

print("\n" + "=" * 70)
print("DISTRIBUTION SHIFT")
print("=" * 70)

print(
    shift_df.to_string(index=False)
)

# ============================================================
# 5. SPIKE ANALYSIS
# ============================================================

spike_rows = []

for sensor in SENSOR_COLS:

    target = f"future_{sensor}"

    train_values = train[target].dropna()

    # Use TRAIN distribution only to define spike threshold.
    q95 = train_values.quantile(0.95)
    q99 = train_values.quantile(0.99)

    for name, data in [
        ("Train", train),
        ("Validation", val),
        ("Test", test)
    ]:

        values = data[target].dropna()

        spike_95 = (
            values > q95
        ).sum()

        spike_99 = (
            values > q99
        ).sum()

        spike_rows.append({
            "sensor": sensor,
            "split": name,
            "train_q95_threshold": q95,
            "train_q99_threshold": q99,
            "count_above_train_q95":
                spike_95,
            "percent_above_train_q95":
                spike_95 / len(values) * 100,
            "count_above_train_q99":
                spike_99,
            "percent_above_train_q99":
                spike_99 / len(values) * 100
        })

spike_df = pd.DataFrame(
    spike_rows
)

spike_file = os.path.join(
    REPORT_DIR,
    "step14_spike_analysis.csv"
)

spike_df.to_csv(
    spike_file,
    index=False
)

print("\n" + "=" * 70)
print("SPIKE ANALYSIS")
print("=" * 70)

print(
    spike_df.to_string(index=False)
)

# ============================================================
# 6. MATCH PREDICTIONS WITH ACTUAL VALUES
# ============================================================

# Prediction file contains actual future values and predictions.
# Analyze prediction errors.

error_rows = []

for sensor in SENSOR_COLS:

    actual_col = f"actual_{sensor}"
    pred_col = f"predicted_{sensor}"

    if (
        actual_col not in pred.columns
        or pred_col not in pred.columns
    ):
        print(
            f"\nWARNING: Missing prediction columns "
            f"for {sensor}"
        )
        continue

    actual = pred[actual_col]
    predicted = pred[pred_col]

    error = (
        actual - predicted
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    error_rows.append({
        "sensor": sensor,
        "mean_error": error.mean(),
        "mean_absolute_error":
            absolute_error.mean(),
        "rmse":
            np.sqrt(squared_error.mean()),
        "error_std": error.std(),
        "error_min": error.min(),
        "error_max": error.max(),
        "error_q50":
            error.quantile(0.50),
        "error_q90":
            error.quantile(0.90),
        "error_q95":
            error.quantile(0.95),
        "error_q99":
            error.quantile(0.99)
    })

error_df = pd.DataFrame(
    error_rows
)

error_file = os.path.join(
    REPORT_DIR,
    "step14_error_summary.csv"
)

error_df.to_csv(
    error_file,
    index=False
)

print("\n" + "=" * 70)
print("TEST ERROR SUMMARY")
print("=" * 70)

print(
    error_df.to_string(index=False)
)

# ============================================================
# 7. TEST PERFORMANCE BY TIME SEGMENT
# ============================================================
# This is especially important because the overall test score
# can hide periods where the model performs very differently.

print("\n" + "=" * 70)
print("TEST PERFORMANCE BY TIME SEGMENT")
print("=" * 70)

segment_rows = []

number_of_segments = 10

for sensor in SENSOR_COLS:

    actual_col = f"actual_{sensor}"
    pred_col = f"predicted_{sensor}"

    actual = pred[actual_col].values
    predicted = pred[pred_col].values

    total = len(actual)

    boundaries = np.linspace(
        0,
        total,
        number_of_segments + 1,
        dtype=int
    )

    for segment in range(
        number_of_segments
    ):

        start = boundaries[segment]
        end = boundaries[segment + 1]

        y_true = actual[start:end]
        y_pred = predicted[start:end]

        errors = (
            y_true - y_pred
        )

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        segment_rows.append({
            "sensor": sensor,
            "segment": segment + 1,
            "start_time":
                pred["ts"].iloc[start],
            "end_time":
                pred["ts"].iloc[end - 1],
            "MAE": mae,
            "RMSE": rmse,
            "actual_mean":
                np.mean(y_true),
            "actual_std":
                np.std(y_true),
            "actual_max":
                np.max(y_true)
        })

segment_df = pd.DataFrame(
    segment_rows
)

segment_file = os.path.join(
    REPORT_DIR,
    "step14_test_performance_by_time_segment.csv"
)

segment_df.to_csv(
    segment_file,
    index=False
)

print(
    segment_df.to_string(index=False)
)

# ============================================================
# 8. PLOT DISTRIBUTIONS
# ============================================================

print("\nCreating distribution plots...")

for sensor in SENSOR_COLS:

    target = f"future_{sensor}"

    plt.figure(
        figsize=(12, 6)
    )

    plt.hist(
        train[target].dropna(),
        bins=100,
        alpha=0.5,
        label="Train"
    )

    plt.hist(
        val[target].dropna(),
        bins=100,
        alpha=0.5,
        label="Validation"
    )

    plt.hist(
        test[target].dropna(),
        bins=100,
        alpha=0.5,
        label="Test"
    )

    plt.title(
        f"{sensor} - Train / Validation / Test Distribution"
    )

    plt.xlabel(sensor)
    plt.ylabel("Frequency")

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        PLOT_DIR,
        f"step14_distribution_{sensor}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

# ============================================================
# 9. PLOT ACTUAL VS PREDICTED
# ============================================================

print(
    "Creating actual-vs-predicted plots..."
)

for sensor in SENSOR_COLS:

    actual_col = f"actual_{sensor}"
    pred_col = f"predicted_{sensor}"

    if (
        actual_col not in pred.columns
        or pred_col not in pred.columns
    ):
        continue

    # Plot first 1000 test points to keep the graph readable.

    n_plot = min(
        1000,
        len(pred)
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        pred["ts"].iloc[:n_plot],
        pred[actual_col].iloc[:n_plot],
        label="Actual"
    )

    plt.plot(
        pred["ts"].iloc[:n_plot],
        pred[pred_col].iloc[:n_plot],
        label="Predicted"
    )

    plt.title(
        f"{sensor} - Actual vs Predicted "
        f"(First {n_plot} Test Samples)"
    )

    plt.xlabel("Time")
    plt.ylabel(sensor)

    plt.xticks(
        rotation=45
    )

    plt.legend()

    plt.tight_layout()

    filename = os.path.join(
        PLOT_DIR,
        f"step14_actual_vs_predicted_{sensor}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

# ============================================================
# 10. ERROR OVER TIME
# ============================================================

print(
    "Creating error-over-time plots..."
)

for sensor in SENSOR_COLS:

    actual_col = f"actual_{sensor}"
    pred_col = f"predicted_{sensor}"

    if (
        actual_col not in pred.columns
        or pred_col not in pred.columns
    ):
        continue

    n_plot = min(
        1000,
        len(pred)
    )

    errors = (
        pred[actual_col].iloc[:n_plot]
        -
        pred[pred_col].iloc[:n_plot]
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        pred["ts"].iloc[:n_plot],
        errors
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        f"{sensor} - Forecast Error Over Time"
    )

    plt.xlabel("Time")
    plt.ylabel("Actual - Predicted")

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    filename = os.path.join(
        PLOT_DIR,
        f"step14_error_over_time_{sensor}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

# ============================================================
# 11. EXTREME ERROR ANALYSIS
# ============================================================

extreme_rows = []

for sensor in SENSOR_COLS:

    actual_col = f"actual_{sensor}"
    pred_col = f"predicted_{sensor}"

    actual = pred[actual_col]
    predicted = pred[pred_col]

    errors = np.abs(
        actual - predicted
    )

    threshold = errors.quantile(
        0.99
    )

    extreme_mask = (
        errors >= threshold
    )

    extreme_rows.append({
        "sensor": sensor,
        "error_99th_percentile":
            threshold,
        "extreme_error_count":
            extreme_mask.sum(),
        "extreme_error_percent":
            extreme_mask.mean() * 100,
        "maximum_absolute_error":
            errors.max()
    })

extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_file = os.path.join(
    REPORT_DIR,
    "step14_extreme_error_analysis.csv"
)

extreme_df.to_csv(
    extreme_file,
    index=False
)

print("\n" + "=" * 70)
print("EXTREME ERROR ANALYSIS")
print("=" * 70)

print(
    extreme_df.to_string(index=False)
)

# ============================================================
# 12. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 OUTPUTS")
print("=" * 70)

print("\nReports:")

print(
    f"  {distribution_file}"
)

print(
    f"  {shift_file}"
)

print(
    f"  {spike_file}"
)

print(
    f"  {error_file}"
)

print(
    f"  {segment_file}"
)

print(
    f"  {extreme_file}"
)

print("\nPlots saved in:")

print(
    f"  {PLOT_DIR}"
)

print("\n" + "=" * 70)
print("STEP 14 COMPLETED")
print("=" * 70)

print(
    "\nNo models were trained in this step."
)

print(
    "The purpose was to diagnose distribution shift "
    "and forecasting errors."
)