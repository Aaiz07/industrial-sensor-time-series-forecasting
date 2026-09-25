"""
04_time_regularization.py

Regularize the cleaned sensor data on a 1-second grid.

Important:
- The raw dataset is irregular: many 1-second and 2-second gaps.
- We do NOT fill the entire 1-second grid.
- Only a single missing second between two observations is
  interpolated.
- Longer gaps remain NaN.
- The large 1579-second outage is preserved.
- No outlier clipping/removal is performed.
- segment_id identifies continuous data regions.

Input:
    outputs/data/sensor_data_cleaned.csv

Outputs:
    outputs/data/sensor_data_regular_1s.csv
    outputs/reports/time_regularization_report.csv
    outputs/reports/large_gaps_regularization.csv
    outputs/plots/time_regularization_gap_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

for folder in [DATA_DIR, REPORT_DIR, PLOT_DIR]:
    os.makedirs(folder, exist_ok=True)

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

TARGET_FREQUENCY = "1s"

# Only one missing second may be interpolated.
MAX_INTERPOLATED_MISSING_POINTS = 1

# A gap > 2 seconds starts a new continuous segment.
SEGMENT_GAP_SECONDS = 2.0


# ============================================================
# LOAD AND VALIDATE
# ============================================================

print("=" * 75)
print("TIME-AXIS REGULARIZATION")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Run 01_data_pipeline.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [TIMESTAMP_COL] + SENSOR_COLS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[TIMESTAMP_COL]
).copy()

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=SENSOR_COLS
).copy()

df = (
    df.sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)

print(f"\nInput rows: {len(df):,}")
print(f"Start: {df[TIMESTAMP_COL].min()}")
print(f"End:   {df[TIMESTAMP_COL].max()}")


# ============================================================
# DUPLICATES
# ============================================================

duplicate_count = int(
    df[TIMESTAMP_COL].duplicated().sum()
)

print(
    f"\nDuplicate timestamps: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:

    print(
        "Aggregating duplicate timestamps "
        "using sensor-wise mean..."
    )

    df = (
        df.groupby(
            TIMESTAMP_COL,
            as_index=False
        )[SENSOR_COLS]
        .mean()
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )


# ============================================================
# ORIGINAL GAP ANALYSIS
# ============================================================

df["gap_seconds"] = (
    df[TIMESTAMP_COL]
    .diff()
    .dt.total_seconds()
)

gap_series = (
    df["gap_seconds"]
    .dropna()
)

print("\nOriginal sampling intervals:")
print(
    gap_series
    .value_counts()
    .sort_index()
    .head(20)
    .to_string()
)


# ============================================================
# FIND LARGE GAPS
# ============================================================

large_gap_indices = df.index[
    df["gap_seconds"] > SEGMENT_GAP_SECONDS
]

large_gap_records = []

for idx in large_gap_indices:

    previous_idx = idx - 1

    large_gap_records.append({
        "previous_timestamp": df.loc[
            previous_idx,
            TIMESTAMP_COL
        ],
        "timestamp": df.loc[
            idx,
            TIMESTAMP_COL
        ],
        "gap_seconds": df.loc[
            idx,
            "gap_seconds"
        ],
    })

large_gap_report = pd.DataFrame(
    large_gap_records,
    columns=[
        "previous_timestamp",
        "timestamp",
        "gap_seconds",
    ]
)

large_gap_file = os.path.join(
    REPORT_DIR,
    "large_gaps_regularization.csv"
)

large_gap_report.to_csv(
    large_gap_file,
    index=False
)

print(
    f"\nGaps > {SEGMENT_GAP_SECONDS:.0f} seconds: "
    f"{len(large_gap_report):,}"
)

if not large_gap_report.empty:

    print("\nLargest gaps:")

    print(
        large_gap_report
        .sort_values(
            "gap_seconds",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# CREATE SEGMENT IDS ON ORIGINAL DATA
# ============================================================

original_ts = df[TIMESTAMP_COL].copy()

original_gap = (
    original_ts
    .diff()
    .dt.total_seconds()
)

new_segment = (
    original_gap.isna()
    |
    (
        original_gap >
        SEGMENT_GAP_SECONDS
    )
)

df["segment_id"] = (
    new_segment
    .cumsum()
    .astype(int)
)

segment_map = df[
    [TIMESTAMP_COL, "segment_id"]
].copy()


# ============================================================
# CREATE 1-SECOND GRID
# ============================================================

print("\nCreating 1-second time grid...")

indexed = (
    df.set_index(TIMESTAMP_COL)[SENSOR_COLS]
    .sort_index()
)

regular = (
    indexed
    .resample(TARGET_FREQUENCY)
    .asfreq()
)

print(
    f"Regular-grid rows: "
    f"{len(regular):,}"
)


# ============================================================
# OBSERVED FLAG
# ============================================================

regular["observed"] = (
    regular.index.isin(
        segment_map[TIMESTAMP_COL]
    )
)


# ============================================================
# SEGMENT ID
# ============================================================

# Map segment IDs at actual observations.
# Then forward/backward fill only the label.

segment_dictionary = dict(
    zip(
        segment_map[TIMESTAMP_COL],
        segment_map["segment_id"]
    )
)

regular["segment_id"] = (
    pd.Series(
        regular.index.map(segment_dictionary),
        index=regular.index
    )
    .ffill()
    .bfill()
    .astype(int)
)


# ============================================================
# CONTROLLED INTERPOLATION
# ============================================================

missing_before = (
    regular[SENSOR_COLS]
    .isna()
    .sum()
)

print(
    "\nMissing values after 1-second resampling:"
)

print(
    missing_before.to_string()
)

# IMPORTANT:
# Keep DatetimeIndex while using method='time'.
#
# limit=1 means:
#     observed -> 1 missing second -> observed
# can be filled.
#
# A 2-second or larger missing run is NOT filled.

regular[SENSOR_COLS] = (
    regular[SENSOR_COLS]
    .interpolate(
        method="time",
        limit=MAX_INTERPOLATED_MISSING_POINTS,
        limit_direction="both",
        limit_area="inside"
    )
)

missing_after = (
    regular[SENSOR_COLS]
    .isna()
    .sum()
)

print(
    "\nMissing values after controlled "
    "interpolation:"
)

print(
    missing_after.to_string()
)


# ============================================================
# RETURN TIMESTAMP TO NORMAL COLUMN
# ============================================================

regular = regular.reset_index()

# Explicitly guarantee the timestamp column name.
if "index" in regular.columns:
    regular = regular.rename(
        columns={"index": TIMESTAMP_COL}
    )

if TIMESTAMP_COL not in regular.columns:
    first_column = regular.columns[0]
    regular = regular.rename(
        columns={
            first_column: TIMESTAMP_COL
        }
    )


# ============================================================
# FLAGS
# ============================================================

regular["is_missing_sensor_row"] = (
    regular[SENSOR_COLS]
    .isna()
    .any(axis=1)
)

regular["gap_boundary"] = False

for _, gap in large_gap_report.iterrows():

    timestamp = gap["timestamp"]

    regular.loc[
        regular[TIMESTAMP_COL] == timestamp,
        "gap_boundary"
    ] = True


# ============================================================
# STATISTICS
# ============================================================

observed_rows = int(
    regular["observed"].sum()
)

interpolated_rows = int(
    (
        (~regular["observed"])
        &
        (~regular["is_missing_sensor_row"])
    ).sum()
)

unfilled_rows = int(
    regular["is_missing_sensor_row"].sum()
)

total_grid_rows = len(regular)

coverage_percent = (
    observed_rows /
    total_grid_rows *
    100
)

usable_percent = (
    (total_grid_rows - unfilled_rows) /
    total_grid_rows *
    100
)


# ============================================================
# SAVE DATA
# ============================================================

output_columns = [
    TIMESTAMP_COL,
    *SENSOR_COLS,
    "observed",
    "is_missing_sensor_row",
    "segment_id",
    "gap_boundary",
]

output_df = regular[
    output_columns
].copy()

output_file = os.path.join(
    DATA_DIR,
    "sensor_data_regular_1s.csv"
)

output_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

report = pd.DataFrame([{
    "input_rows": len(df),
    "output_regular_rows": total_grid_rows,
    "duplicate_timestamps_removed": duplicate_count,
    "observed_rows": observed_rows,
    "interpolated_rows": interpolated_rows,
    "unfilled_missing_rows": unfilled_rows,
    "observed_coverage_percent": coverage_percent,
    "usable_data_percent": usable_percent,
    "large_gaps_over_2_seconds":
        len(large_gap_report),
    "target_frequency": TARGET_FREQUENCY,
    "max_interpolated_missing_points":
        MAX_INTERPOLATED_MISSING_POINTS,
    "segment_count": int(
        output_df["segment_id"]
        .nunique()
    ),
    "start_timestamp":
        output_df[TIMESTAMP_COL].min(),
    "end_timestamp":
        output_df[TIMESTAMP_COL].max(),
}])

report_file = os.path.join(
    REPORT_DIR,
    "time_regularization_report.csv"
)

report.to_csv(
    report_file,
    index=False
)


# ============================================================
# GAP DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(12, 5))

plot_gaps = gap_series[
    gap_series > 0
].clip(upper=10)

plt.hist(
    plot_gaps,
    bins=np.arange(
        0.5,
        10.5,
        1
    )
)

plt.xlabel(
    "Observed gap (seconds), clipped at 10 s"
)

plt.ylabel(
    "Number of intervals"
)

plt.title(
    "Original Sensor Sampling Gap Distribution"
)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "time_regularization_gap_plot.png"
)

plt.savefig(
    plot_file,
    dpi=150
)

plt.close()


# ============================================================
# FINAL VALIDATION
# ============================================================

final_ts = pd.to_datetime(
    output_df[TIMESTAMP_COL]
)

final_diffs = (
    final_ts
    .diff()
    .dt.total_seconds()
    .dropna()
)

print("\n" + "=" * 75)
print("REGULARIZATION SUMMARY")
print("=" * 75)

print(
    f"Original rows:           "
    f"{len(df):,}"
)

print(
    f"Regular 1-second rows:   "
    f"{total_grid_rows:,}"
)

print(
    f"Observed rows:           "
    f"{observed_rows:,}"
)

print(
    f"Observed coverage:       "
    f"{coverage_percent:.2f}%"
)

print(
    f"Interpolated rows:       "
    f"{interpolated_rows:,}"
)

print(
    f"Still-missing rows:      "
    f"{unfilled_rows:,}"
)

print(
    f"Usable data:             "
    f"{usable_percent:.2f}%"
)

print(
    f"Continuous segments:     "
    f"{output_df['segment_id'].nunique():,}"
)

print(
    f"Final minimum interval:  "
    f"{final_diffs.min():.1f} seconds"
)

print(
    f"Final maximum interval:  "
    f"{final_diffs.max():.1f} seconds"
)

print("\nFinal sensor missing counts:")

print(
    output_df[SENSOR_COLS]
    .isna()
    .sum()
    .to_string()
)

print("\nSaved:")

print(f"  {output_file}")
print(f"  {report_file}")
print(f"  {large_gap_file}")
print(f"  {plot_file}")

print("\nIMPORTANT:")
print(
    "Only short internal gaps are interpolated. "
    "The large outage is preserved as missing data. "
    "No vibration or magnetic spikes were removed."
)

print("\n" + "=" * 75)
print("TIME REGULARIZATION COMPLETED")
print("=" * 75)
