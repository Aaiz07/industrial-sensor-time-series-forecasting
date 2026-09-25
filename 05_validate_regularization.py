"""
05_validate_regularization.py

Validate whether the 1-second regularized dataset preserves the
real sensor behavior.

Compares:
    Original cleaned observations
    vs
    Regularized 1-second data

Checks:
    1. Coverage and interpolation amount
    2. Original vs regularized descriptive statistics
    3. Quantiles
    4. Correlation matrices
    5. Vibration spike preservation
    6. Interpolation error using only artificially hidden REAL values
    7. Temperature / magnetic / vibration behavior
    8. Plots for visual inspection

IMPORTANT:
This script does NOT train any model and does NOT modify the data.

Input:
    outputs/data/sensor_data_cleaned.csv
    outputs/data/sensor_data_regular_1s.csv

Outputs:
    outputs/reports/regularization_validation_summary.csv
    outputs/reports/interpolation_holdout_metrics.csv
    outputs/reports/original_vs_regularized_stats.csv
    outputs/plots/regularization_comparison_*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEANED_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

REGULAR_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_regular_1s.csv"
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

# Number of points used in plots.
PLOT_POINTS = 5000

# For interpolation validation:
# randomly hide real observations only when they have a real
# observation immediately before and after them.
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.10


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("REGULARIZATION VALIDATION")
print("=" * 75)

if not os.path.exists(CLEANED_FILE):
    raise FileNotFoundError(
        f"Cleaned file not found:\n{CLEANED_FILE}\n"
        "Run 01_data_pipeline.py first."
    )

if not os.path.exists(REGULAR_FILE):
    raise FileNotFoundError(
        f"Regularized file not found:\n{REGULAR_FILE}\n"
        "Run 04_time_regularization.py first."
    )

original = pd.read_csv(CLEANED_FILE)
regular = pd.read_csv(REGULAR_FILE)

original[TIMESTAMP_COL] = pd.to_datetime(
    original[TIMESTAMP_COL],
    errors="coerce"
)

regular[TIMESTAMP_COL] = pd.to_datetime(
    regular[TIMESTAMP_COL],
    errors="coerce"
)

original = (
    original
    .dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)

regular = (
    regular
    .dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)

for col in SENSOR_COLS:
    original[col] = pd.to_numeric(
        original[col],
        errors="coerce"
    )
    regular[col] = pd.to_numeric(
        regular[col],
        errors="coerce"
    )

print(f"\nOriginal rows:    {len(original):,}")
print(f"Regularized rows: {len(regular):,}")


# ============================================================
# BASIC COVERAGE
# ============================================================

if "observed" in regular.columns:
    observed_rows = int(
        regular["observed"].sum()
    )
else:
    observed_rows = int(
        regular[TIMESTAMP_COL]
        .isin(original[TIMESTAMP_COL])
        .sum()
    )

if "is_missing_sensor_row" in regular.columns:
    missing_rows = int(
        regular["is_missing_sensor_row"].sum()
    )
else:
    missing_rows = int(
        regular[SENSOR_COLS]
        .isna()
        .any(axis=1)
        .sum()
    )

interpolated_rows = int(
    len(regular)
    - observed_rows
    - missing_rows
)

print("\nCoverage:")
print(
    f"Observed:       {observed_rows:,}"
)

print(
    f"Interpolated:   {interpolated_rows:,}"
)

print(
    f"Still missing:  {missing_rows:,}"
)

print(
    f"Observed %:     "
    f"{observed_rows / len(regular) * 100:.2f}%"
)

print(
    f"Usable %:       "
    f"{(len(regular) - missing_rows) / len(regular) * 100:.2f}%"
)


# ============================================================
# ORIGINAL VS REGULARIZED STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("STATISTICAL COMPARISON")
print("=" * 75)

stats_records = []

for col in SENSOR_COLS:

    o = original[col].dropna()
    r = regular[col].dropna()

    for dataset_name, values in [
        ("original", o),
        ("regularized", r),
    ]:

        stats_records.append({
            "sensor": col,
            "dataset": dataset_name,
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
            "max": values.max(),
        })

stats_df = pd.DataFrame(stats_records)

stats_file = os.path.join(
    REPORT_DIR,
    "original_vs_regularized_stats.csv"
)

stats_df.to_csv(
    stats_file,
    index=False
)

print(stats_df.to_string(index=False))


# ============================================================
# CORRELATION COMPARISON
# ============================================================

original_corr = original[
    SENSOR_COLS
].corr()

regular_corr = regular[
    SENSOR_COLS
].corr()

corr_difference = (
    regular_corr - original_corr
)

original_corr.to_csv(
    os.path.join(
        REPORT_DIR,
        "original_correlation_matrix.csv"
    )
)

regular_corr.to_csv(
    os.path.join(
        REPORT_DIR,
        "regularized_correlation_matrix.csv"
    )
)

corr_difference.to_csv(
    os.path.join(
        REPORT_DIR,
        "correlation_difference_matrix.csv"
    )
)

print("\nMaximum absolute correlation change:")

print(
    np.nanmax(
        np.abs(
            corr_difference.values
        )
    )
)


# ============================================================
# SPIKE PRESERVATION
# ============================================================

print("\n" + "=" * 75)
print("VIBRATION SPIKE PRESERVATION")
print("=" * 75)

spike_records = []

for col in [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
]:

    original_values = (
        original[col]
        .dropna()
    )

    regular_values = (
        regular[col]
        .dropna()
    )

    # Compare several high quantiles rather than only max.
    for q in [0.90, 0.95, 0.99, 0.995, 0.999]:

        original_q = original_values.quantile(q)
        regular_q = regular_values.quantile(q)

        relative_change = (
            (regular_q - original_q)
            / max(abs(original_q), 1e-12)
            * 100
        )

        spike_records.append({
            "sensor": col,
            "quantile": q,
            "original": original_q,
            "regularized": regular_q,
            "relative_change_percent":
                relative_change,
        })

spike_df = pd.DataFrame(
    spike_records
)

spike_file = os.path.join(
    REPORT_DIR,
    "vibration_spike_preservation.csv"
)

spike_df.to_csv(
    spike_file,
    index=False
)

print(
    spike_df.to_string(index=False)
)


# ============================================================
# INTERPOLATION QUALITY TEST
# ============================================================

print("\n" + "=" * 75)
print("INTERPOLATION QUALITY TEST")
print("=" * 75)

# We test interpolation honestly:
#
# 1. Find real observations where timestamp-1 second and
#    timestamp+1 second also exist.
# 2. Temporarily hide a random subset of those REAL values.
# 3. Interpolate between their neighbors.
# 4. Compare the interpolated estimate with the REAL value.
#
# This does not use the actual missing values as ground truth.

original_indexed = (
    original
    .set_index(TIMESTAMP_COL)
    .sort_index()
)

candidate_times = []

original_index = original_indexed.index

for ts in original_index:

    before = ts - pd.Timedelta(seconds=1)
    after = ts + pd.Timedelta(seconds=1)

    if (
        before in original_index
        and after in original_index
    ):
        candidate_times.append(ts)

candidate_times = pd.DatetimeIndex(
    candidate_times
)

print(
    f"\nPotential one-second interpolation "
    f"validation points: {len(candidate_times):,}"
)

rng = np.random.default_rng(
    RANDOM_STATE
)

if len(candidate_times) > 0:

    sample_size = max(
        1,
        int(
            len(candidate_times)
            * HOLDOUT_FRACTION
        )
    )

    sample_size = min(
        sample_size,
        5000
    )

    holdout_times = rng.choice(
        candidate_times,
        size=sample_size,
        replace=False
    )

    holdout_times = pd.DatetimeIndex(
        holdout_times
    )

else:

    holdout_times = pd.DatetimeIndex([])

interpolation_records = []

for col in SENSOR_COLS:

    true_values = []
    predicted_values = []

    for ts in holdout_times:

        before = ts - pd.Timedelta(
            seconds=1
        )

        after = ts + pd.Timedelta(
            seconds=1
        )

        y0 = original_indexed.loc[
            before,
            col
        ]

        y1 = original_indexed.loc[
            after,
            col
        ]

        y_true = original_indexed.loc[
            ts,
            col
        ]

        if (
            pd.isna(y0)
            or pd.isna(y1)
            or pd.isna(y_true)
        ):
            continue

        # Linear interpolation at the midpoint.
        y_pred = (
            y0 + y1
        ) / 2.0

        true_values.append(
            float(y_true)
        )

        predicted_values.append(
            float(y_pred)
        )

    true_values = np.asarray(
        true_values
    )

    predicted_values = np.asarray(
        predicted_values
    )

    if len(true_values) > 0:

        errors = (
            predicted_values
            - true_values
        )

        mae = np.mean(
            np.abs(errors)
        )

        rmse = np.sqrt(
            np.mean(
                errors ** 2
            )
        )

        bias = np.mean(errors)

        correlation = (
            np.corrcoef(
                true_values,
                predicted_values
            )[0, 1]
            if len(true_values) > 1
            else np.nan
        )

    else:

        mae = np.nan
        rmse = np.nan
        bias = np.nan
        correlation = np.nan

    interpolation_records.append({
        "sensor": col,
        "validation_points": len(
            true_values
        ),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Correlation": correlation,
    })

interpolation_df = pd.DataFrame(
    interpolation_records
)

interpolation_file = os.path.join(
    REPORT_DIR,
    "interpolation_holdout_metrics.csv"
)

interpolation_df.to_csv(
    interpolation_file,
    index=False
)

print(
    interpolation_df.to_string(
        index=False
    )
)


# ============================================================
# PLOT ORIGINAL VS REGULARIZED
# ============================================================

print("\nCreating comparison plots...")

# Use the first continuous portion of the day for plotting.
plot_regular = regular.copy()

if "segment_id" in plot_regular.columns:

    first_segment = (
        plot_regular["segment_id"]
        .min()
    )

    plot_regular = plot_regular[
        plot_regular["segment_id"]
        == first_segment
    ].copy()

plot_regular = plot_regular.head(
    PLOT_POINTS
)

original_plot = (
    original[
        original[TIMESTAMP_COL]
        .between(
            plot_regular[TIMESTAMP_COL].min(),
            plot_regular[TIMESTAMP_COL].max()
        )
    ]
    .copy()
)

for col in SENSOR_COLS:

    plt.figure(figsize=(14, 5))

    plt.plot(
        original_plot[TIMESTAMP_COL],
        original_plot[col],
        label="Original observations",
        linewidth=1
    )

    plt.plot(
        plot_regular[TIMESTAMP_COL],
        plot_regular[col],
        label="Regularized 1-second",
        linewidth=1
    )

    plt.xlabel("Timestamp")
    plt.ylabel(col)
    plt.title(
        f"Original vs Regularized — {col}"
    )

    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plot_file = os.path.join(
        PLOT_DIR,
        f"regularization_comparison_{col}.png"
    )

    plt.savefig(
        plot_file,
        dpi=150
    )

    plt.close()


# ============================================================
# SUMMARY DECISION METRICS
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

summary_records = []

for col in SENSOR_COLS:

    o = original[col].dropna()
    r = regular[col].dropna()

    original_mean = o.mean()
    regular_mean = r.mean()

    original_std = o.std()
    regular_std = r.std()

    mean_change = (
        (regular_mean - original_mean)
        / max(abs(original_mean), 1e-12)
        * 100
    )

    std_change = (
        (regular_std - original_std)
        / max(abs(original_std), 1e-12)
        * 100
    )

    summary_records.append({
        "sensor": col,
        "original_mean": original_mean,
        "regularized_mean": regular_mean,
        "mean_change_percent": mean_change,
        "original_std": original_std,
        "regularized_std": regular_std,
        "std_change_percent": std_change,
    })

summary_df = pd.DataFrame(
    summary_records
)

summary_file = os.path.join(
    REPORT_DIR,
    "regularization_validation_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print(
    summary_df.to_string(index=False)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION COMPLETED")
print("=" * 75)

print("\nReports created:")

print(
    f"  {summary_file}"
)

print(
    f"  {interpolation_file}"
)

print(
    f"  {stats_file}"
)

print(
    f"  {spike_file}"
)

print(
    f"  {os.path.join(REPORT_DIR, 'original_correlation_matrix.csv')}"
)

print(
    f"  {os.path.join(REPORT_DIR, 'regularized_correlation_matrix.csv')}"
)

print(
    f"  {os.path.join(REPORT_DIR, 'correlation_difference_matrix.csv')}"
)

print("\nPlots created in:")
print(f"  {PLOT_DIR}")

print("\nIMPORTANT:")
print(
    "Do not train the final forecasting models yet. "
    "Use the interpolation metrics, spike preservation, "
    "statistics, and plots to decide whether the 1-second "
    "representation is appropriate."
)

print("\n" + "=" * 75)
print("END")
print("=" * 75)
