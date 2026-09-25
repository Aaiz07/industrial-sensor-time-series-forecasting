# ============================================================
# 01_DATA_PIPELINE.PY
# Industrial Sensor Time-Series Data Pipeline
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "sensor_data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_DIR = os.path.join(OUTPUT_DIR, "data")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
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


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def print_section(title):
    """
    Print a clean section heading.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

print_section("LOADING DATA")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Input file : {INPUT_FILE}")
print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")

print("\nColumns:")
print(list(df.columns))


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

print_section("VALIDATING COLUMNS")

required_columns = [TIMESTAMP_COL] + SENSOR_COLS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ============================================================
# 5. TIMESTAMP PROCESSING
# ============================================================

print_section("PROCESSING TIMESTAMP")

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

invalid_timestamps = df[TIMESTAMP_COL].isna().sum()

print(
    f"Invalid timestamps: "
    f"{invalid_timestamps:,}"
)

if invalid_timestamps > 0:

    df = df.dropna(
        subset=[TIMESTAMP_COL]
    ).copy()

    print(
        f"Removed {invalid_timestamps:,} "
        f"rows with invalid timestamps."
    )


# ============================================================
# 6. SORT BY TIME
# ============================================================

print_section("SORTING DATA")

is_sorted = df[TIMESTAMP_COL].is_monotonic_increasing

print(f"Already sorted: {is_sorted}")

if not is_sorted:

    df = df.sort_values(
        TIMESTAMP_COL
    ).reset_index(drop=True)

    print("Data sorted chronologically.")

else:

    df = df.reset_index(drop=True)

    print("No sorting required.")


# ============================================================
# 7. DUPLICATE TIMESTAMP ANALYSIS
# ============================================================

print_section("DUPLICATE TIMESTAMP ANALYSIS")

duplicate_timestamp_count = (
    df[TIMESTAMP_COL]
    .duplicated()
    .sum()
)

print(
    f"Duplicate timestamp rows: "
    f"{duplicate_timestamp_count:,}"
)

if duplicate_timestamp_count > 0:

    duplicate_rows = df[
        df[TIMESTAMP_COL].duplicated(
            keep=False
        )
    ].copy()

    duplicate_rows.to_csv(
        os.path.join(
            REPORT_DIR,
            "duplicate_timestamps.csv"
        ),
        index=False
    )

    print(
        "Duplicate timestamps saved to:"
    )

    print(
        os.path.join(
            REPORT_DIR,
            "duplicate_timestamps.csv"
        )
    )


# ============================================================
# 8. SAMPLING INTERVAL ANALYSIS
# ============================================================

print_section("SAMPLING INTERVAL ANALYSIS")

sampling_analysis = pd.DataFrame({
    "timestamp": df[TIMESTAMP_COL],
    "previous_timestamp": (
        df[TIMESTAMP_COL].shift(1)
    )
})

sampling_analysis["gap_seconds"] = (
    sampling_analysis["timestamp"]
    .diff()
    .dt.total_seconds()
)

# Remove first row because it has no previous timestamp
valid_intervals = (
    sampling_analysis["gap_seconds"]
    .dropna()
)

if len(valid_intervals) > 0:

    median_interval = valid_intervals.median()
    mean_interval = valid_intervals.mean()
    min_interval = valid_intervals.min()
    max_interval = valid_intervals.max()

    print(
        f"Median interval : "
        f"{median_interval:.4f} seconds"
    )

    print(
        f"Mean interval   : "
        f"{mean_interval:.4f} seconds"
    )

    print(
        f"Minimum interval: "
        f"{min_interval:.4f} seconds"
    )

    print(
        f"Maximum interval: "
        f"{max_interval:.4f} seconds"
    )

else:

    median_interval = np.nan

    print("Not enough data to calculate sampling interval.")


# ============================================================
# 9. INTERVAL DISTRIBUTION
# ============================================================

if len(valid_intervals) > 0:

    interval_counts = (
        valid_intervals
        .round(3)
        .value_counts()
        .sort_index()
    )

    interval_report = pd.DataFrame({
        "interval_seconds": interval_counts.index,
        "count": interval_counts.values
    })

    interval_report.to_csv(
        os.path.join(
            REPORT_DIR,
            "sampling_interval_distribution.csv"
        ),
        index=False
    )

    print("\nMost common sampling intervals:")

    print(
        interval_report
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 10. LARGE GAP ANALYSIS
# ============================================================

print_section("LARGE TIME GAP ANALYSIS")

if (
    len(valid_intervals) > 0
    and median_interval > 0
):

    # A gap larger than 1.5 × median
    # is considered irregular.
    large_gap_threshold = (
        median_interval * 1.5
    )

    large_gaps = sampling_analysis[
        sampling_analysis["gap_seconds"]
        > large_gap_threshold
    ].copy()

    large_gaps.to_csv(
        os.path.join(
            REPORT_DIR,
            "large_time_gaps.csv"
        ),
        index=False
    )

    print(
        f"Large-gap threshold: "
        f"{large_gap_threshold:.4f} seconds"
    )

    print(
        f"Large gaps detected: "
        f"{len(large_gaps):,}"
    )

    if len(large_gaps) > 0:

        print("\nLargest gaps:")

        print(
            large_gaps
            .sort_values(
                "gap_seconds",
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

else:

    print(
        "Sampling interval could not be analyzed."
    )


# ============================================================
# 11. MISSING VALUE ANALYSIS
# ============================================================

print_section("MISSING VALUE ANALYSIS")

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_count": [
        df[col].isna().sum()
        for col in df.columns
    ]
})

missing_report["missing_percentage"] = (
    missing_report["missing_count"]
    / len(df)
    * 100
)

print(
    missing_report.to_string(index=False)
)

missing_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "missing_values_report.csv"
    ),
    index=False
)


# ============================================================
# 12. CONVERT SENSOR COLUMNS TO NUMERIC
# ============================================================

print_section("NUMERIC VALIDATION")

for col in SENSOR_COLS:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("Sensor columns converted to numeric.")


# ============================================================
# 13. INVALID NUMERIC VALUES
# ============================================================

invalid_numeric_report = []

for col in SENSOR_COLS:

    invalid_count = df[col].isna().sum()

    invalid_numeric_report.append({
        "column": col,
        "invalid_or_missing": invalid_count
    })

invalid_numeric_report = pd.DataFrame(
    invalid_numeric_report
)

print(
    invalid_numeric_report.to_string(
        index=False
    )
)


# ============================================================
# 14. DESCRIPTIVE STATISTICS
# ============================================================

print_section("DESCRIPTIVE STATISTICS")

stats = df[SENSOR_COLS].describe().T

stats["missing"] = (
    df[SENSOR_COLS]
    .isna()
    .sum()
)

stats["missing_percentage"] = (
    stats["missing"]
    / len(df)
    * 100
)

print(
    stats.to_string()
)

stats.to_csv(
    os.path.join(
        REPORT_DIR,
        "descriptive_statistics.csv"
    )
)


# ============================================================
# 15. OUTLIER ANALYSIS
# ============================================================

print_section("OUTLIER ANALYSIS")

outlier_results = []

for col in SENSOR_COLS:

    series = df[col].dropna()

    if len(series) == 0:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower)
        | (series > upper)
    )

    outlier_count = outlier_mask.sum()

    outlier_percentage = (
        outlier_count
        / len(series)
        * 100
    )

    outlier_results.append({
        "column": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_percentage":
            outlier_percentage
    })

outlier_report = pd.DataFrame(
    outlier_results
)

print(
    outlier_report.to_string(
        index=False
    )
)

outlier_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "outlier_analysis.csv"
    ),
    index=False
)

print(
    "\nIMPORTANT:"
    "\nOutliers are NOT automatically removed."
    "\nSensor spikes may represent real machine events."
)


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================

print_section("CORRELATION ANALYSIS")

correlation = df[
    SENSOR_COLS
].corr()

print(
    correlation.to_string()
)

correlation.to_csv(
    os.path.join(
        REPORT_DIR,
        "correlation_matrix.csv"
    )
)


# ============================================================
# 17. RAW TIME-SERIES PLOTS
# ============================================================

print_section("CREATING TIME-SERIES PLOTS")

for col in SENSOR_COLS:

    plt.figure(figsize=(14, 5))

    plt.plot(
        df[TIMESTAMP_COL],
        df[col],
        linewidth=0.8
    )

    plt.title(
        f"{col} - Raw Time Series"
    )

    plt.xlabel("Time")
    plt.ylabel(col)

    plt.xticks(rotation=30)

    save_plot(
        f"raw_{col}.png"
    )


# ============================================================
# 18. DISTRIBUTION PLOTS
# ============================================================

print_section("CREATING DISTRIBUTION PLOTS")

for col in SENSOR_COLS:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[col].dropna(),
        bins=80
    )

    plt.title(
        f"{col} - Distribution"
    )

    plt.xlabel(col)
    plt.ylabel("Frequency")

    save_plot(
        f"distribution_{col}.png"
    )


# ============================================================
# 19. CORRELATION HEATMAP
# ============================================================

print_section("CREATING CORRELATION HEATMAP")

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(SENSOR_COLS)),
    SENSOR_COLS,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(SENSOR_COLS)),
    SENSOR_COLS
)

plt.title(
    "Sensor Correlation Matrix"
)

save_plot(
    "correlation_heatmap.png"
)


# ============================================================
# 20. DATA CLEANING
# ============================================================

print_section("CLEANING DATA")

clean_df = df.copy()

# Remove rows where timestamp is invalid
clean_df = clean_df.dropna(
    subset=[TIMESTAMP_COL]
).copy()

# Make sure data remains chronological
clean_df = (
    clean_df
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)

print(
    f"Rows before cleaning: "
    f"{len(df):,}"
)

print(
    f"Rows after timestamp cleaning: "
    f"{len(clean_df):,}"
)


# ============================================================
# 21. HANDLE DUPLICATE TIMESTAMPS
# ============================================================

print_section("HANDLING DUPLICATE TIMESTAMPS")

duplicate_count_before = (
    clean_df[TIMESTAMP_COL]
    .duplicated()
    .sum()
)

print(
    f"Duplicate timestamps before handling: "
    f"{duplicate_count_before:,}"
)

if duplicate_count_before > 0:

    # Aggregate duplicate timestamp rows
    # using mean for sensor measurements.
    #
    # This prevents zero-second intervals
    # and gives us one observation per timestamp.

    clean_df = (
        clean_df
        .groupby(
            TIMESTAMP_COL,
            as_index=False
        )[SENSOR_COLS]
        .mean()
    )

    print(
        "Duplicate timestamps aggregated "
        "using sensor-wise mean."
    )

duplicate_count_after = (
    clean_df[TIMESTAMP_COL]
    .duplicated()
    .sum()
)

print(
    f"Duplicate timestamps after handling: "
    f"{duplicate_count_after:,}"
)


# ============================================================
# 22. SENSOR MISSING VALUES
# ============================================================

print_section("HANDLING SENSOR MISSING VALUES")

missing_before = (
    clean_df[SENSOR_COLS]
    .isna()
    .sum()
)

print("Missing values before interpolation:")

print(
    missing_before.to_string()
)


# ------------------------------------------------------------
# Important:
# We do NOT interpolate across arbitrary long time gaps.
#
# First interpolate only normal/local gaps.
# Long gaps remain visible and can be analyzed later.
# ------------------------------------------------------------

if clean_df[SENSOR_COLS].isna().sum().sum() > 0:

    clean_df = (
        clean_df
        .set_index(TIMESTAMP_COL)
    )

    # Time interpolation
    clean_df[SENSOR_COLS] = (
        clean_df[SENSOR_COLS]
        .interpolate(
            method="time",
            limit_direction="both"
        )
    )

    clean_df = (
        clean_df
        .reset_index()
    )

missing_after = (
    clean_df[SENSOR_COLS]
    .isna()
    .sum()
)

print("\nMissing values after interpolation:")

print(
    missing_after.to_string()
)


# ============================================================
# 23. FINAL NUMERIC VALIDATION
# ============================================================

print_section("FINAL DATA VALIDATION")

for col in SENSOR_COLS:

    clean_df[col] = pd.to_numeric(
        clean_df[col],
        errors="coerce"
    )

# Check infinite values
infinite_counts = {}

for col in SENSOR_COLS:

    infinite_counts[col] = np.isinf(
        clean_df[col].to_numpy(
            dtype=float
        )
    ).sum()

print("Infinite values:")

print(
    pd.Series(infinite_counts)
)


# ============================================================
# 24. FINAL CLEAN DATASET
# ============================================================

clean_df = clean_df[
    [TIMESTAMP_COL] + SENSOR_COLS
].copy()

clean_df = (
    clean_df
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)


# ============================================================
# 25. SAVE CLEAN DATA
# ============================================================

print_section("SAVING CLEAN DATA")

clean_file = os.path.join(
    DATA_DIR,
    "sensor_data_cleaned.csv"
)

clean_df.to_csv(
    clean_file,
    index=False
)

print(
    f"Clean dataset saved:\n"
    f"{clean_file}"
)


# ============================================================
# 26. CLEAN DATA SUMMARY
# ============================================================

print_section("FINAL QUALITY SUMMARY")

print(
    f"Final rows       : "
    f"{len(clean_df):,}"
)

print(
    f"Final columns    : "
    f"{len(clean_df.columns)}"
)

print(
    f"Duplicate times  : "
    f"{clean_df[TIMESTAMP_COL].duplicated().sum():,}"
)

print(
    f"Missing values   : "
    f"{clean_df[SENSOR_COLS].isna().sum().sum():,}"
)

print(
    f"Start timestamp  : "
    f"{clean_df[TIMESTAMP_COL].min()}"
)

print(
    f"End timestamp    : "
    f"{clean_df[TIMESTAMP_COL].max()}"
)


# ============================================================
# 27. FINAL CLEAN DATA PLOTS
# ============================================================

print_section("CREATING CLEAN DATA PLOTS")

for col in SENSOR_COLS:

    plt.figure(figsize=(14, 5))

    plt.plot(
        clean_df[TIMESTAMP_COL],
        clean_df[col],
        linewidth=0.8
    )

    plt.title(
        f"{col} - Cleaned Time Series"
    )

    plt.xlabel("Time")
    plt.ylabel(col)

    plt.xticks(rotation=30)

    save_plot(
        f"cleaned_{col}.png"
    )


# ============================================================
# 28. SAVE FINAL QUALITY REPORT
# ============================================================

quality_summary = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "duplicate_timestamps",
        "missing_sensor_values",
        "start_timestamp",
        "end_timestamp",
        "median_sampling_interval_seconds",
        "mean_sampling_interval_seconds",
    ],
    "value": [
        len(clean_df),
        len(clean_df.columns),
        clean_df[TIMESTAMP_COL]
            .duplicated()
            .sum(),
        clean_df[SENSOR_COLS]
            .isna()
            .sum()
            .sum(),
        str(
            clean_df[TIMESTAMP_COL].min()
        ),
        str(
            clean_df[TIMESTAMP_COL].max()
        ),
        (
            median_interval
            if not np.isnan(median_interval)
            else np.nan
        ),
        (
            mean_interval
            if len(valid_intervals) > 0
            else np.nan
        ),
    ]
})

quality_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "final_quality_summary.csv"
    ),
    index=False
)


# ============================================================
# 29. COMPLETION
# ============================================================

print_section("PIPELINE COMPLETED")

print(
    "Data pipeline completed successfully."
)

print(
    "\nClean data:"
)

print(clean_file)

print(
    "\nReports:"
)

print(REPORT_DIR)

print(
    "\nPlots:"
)

print(PLOT_DIR)

print(
    "\nNext step:"
)

print(
    "Use sensor_data_cleaned.csv "
    "for model preparation and training."
)