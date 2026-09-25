# ============================================================
# 15_regime_analysis.py
# Regime / Operating-State Analysis for Industrial Sensors
# ============================================================

import os
import json
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

VIBRATION_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms"
]

MAGNETIC_COLS = [
    "mag_x",
    "mag_y",
    "mag_z"
]

TEMPERATURE_COL = "temperature"

SENSOR_COLS = (
    VIBRATION_COLS
    + MAGNETIC_COLS
    + [TEMPERATURE_COL]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    return name.replace("/", "_").replace(" ", "_")


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print_section("STEP 15: REGIME / OPERATING-STATE ANALYSIS")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Input file: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. VALIDATE DATA
# ============================================================

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

df = df.dropna(subset=[TIMESTAMP_COL]).copy()

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

print(f"Valid rows after timestamp cleaning: {len(df):,}")

print(
    f"Start: {df[TIMESTAMP_COL].min()}"
)

print(
    f"End:   {df[TIMESTAMP_COL].max()}"
)


# ============================================================
# 3. DETECT LARGE TIME GAPS
# ============================================================

print_section("TIME GAP ANALYSIS")

df["gap_seconds"] = (
    df[TIMESTAMP_COL]
    .diff()
    .dt.total_seconds()
)

median_gap = df["gap_seconds"].dropna().median()

print(f"Median sampling interval: {median_gap:.3f} seconds")

# We consider gaps > 10 seconds as major gaps.
# This avoids treating normal 1-2 second sampling variation
# as an operating-state change.

MAJOR_GAP_THRESHOLD = 10

major_gaps = df[
    df["gap_seconds"] > MAJOR_GAP_THRESHOLD
].copy()

print(
    f"Major gaps > {MAJOR_GAP_THRESHOLD} seconds: "
    f"{len(major_gaps):,}"
)

if len(major_gaps) > 0:

    major_gap_report = major_gaps[
        [
            TIMESTAMP_COL,
            "gap_seconds"
        ]
    ].copy()

    major_gap_report["previous_timestamp"] = (
        df[TIMESTAMP_COL].shift(1)
        .loc[major_gap_report.index]
        .values
    )

    major_gap_report = major_gap_report[
        [
            "previous_timestamp",
            TIMESTAMP_COL,
            "gap_seconds"
        ]
    ]

    major_gap_report = major_gap_report.sort_values(
        "gap_seconds",
        ascending=False
    )

    print("\nLargest gaps:")

    print(
        major_gap_report.head(10).to_string(
            index=False
        )
    )

else:

    major_gap_report = pd.DataFrame(
        columns=[
            "previous_timestamp",
            TIMESTAMP_COL,
            "gap_seconds"
        ]
    )


major_gap_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_major_time_gaps.csv"
    ),
    index=False
)


# ============================================================
# 4. CREATE CONTINUOUS SEGMENTS
# ============================================================

print_section("CONTINUOUS SEGMENT IDENTIFICATION")

df["segment_id"] = (
    df["gap_seconds"]
    .gt(MAJOR_GAP_THRESHOLD)
    .cumsum()
)

segment_summary = (
    df.groupby("segment_id")
    .agg(
        start_time=(TIMESTAMP_COL, "min"),
        end_time=(TIMESTAMP_COL, "max"),
        rows=(TIMESTAMP_COL, "size"),
    )
    .reset_index()
)

segment_summary["duration_minutes"] = (
    (
        segment_summary["end_time"]
        - segment_summary["start_time"]
    )
    .dt.total_seconds()
    / 60
)

print(
    f"Continuous segments: "
    f"{len(segment_summary):,}"
)

print("\nSegment summary:")

print(
    segment_summary.to_string(
        index=False
    )
)

segment_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_segment_summary.csv"
    ),
    index=False
)


# ============================================================
# 5. CREATE TIME-BASED ROLLING FEATURES
# ============================================================

print_section("ROLLING OPERATING-STATE FEATURES")

print(
    "Creating time-based rolling statistics..."
)

# Use timestamp as index temporarily.
# Time-based windows are better here because the sampling
# interval is irregular.

indexed = df.set_index(TIMESTAMP_COL).copy()

# ------------------------------------------------------------
# Vibration combined energy
# ------------------------------------------------------------

indexed["vibration_energy"] = np.sqrt(
    indexed["vib_x_rms"] ** 2
    + indexed["vib_y_rms"] ** 2
    + indexed["vib_z_rms"] ** 2
)

# Rolling statistics
rolling_windows = [
    "1min",
    "5min",
    "10min",
    "30min"
]

for window in rolling_windows:

    indexed[
        f"vibration_energy_mean_{window}"
    ] = (
        indexed["vibration_energy"]
        .rolling(window)
        .mean()
    )

    indexed[
        f"vibration_energy_std_{window}"
    ] = (
        indexed["vibration_energy"]
        .rolling(window)
        .std()
    )

    indexed[
        f"vibration_energy_max_{window}"
    ] = (
        indexed["vibration_energy"]
        .rolling(window)
        .max()
    )

    indexed[
        f"temperature_mean_{window}"
    ] = (
        indexed[TEMPERATURE_COL]
        .rolling(window)
        .mean()
    )

    indexed[
        f"temperature_std_{window}"
    ] = (
        indexed[TEMPERATURE_COL]
        .rolling(window)
        .std()
    )


# ============================================================
# 6. VIBRATION REGIME SCORE
# ============================================================

print_section("VIBRATION REGIME DETECTION")

# We use a 5-minute rolling vibration energy level.
# This captures operating state rather than individual spikes.

vibration_score = (
    indexed["vibration_energy_mean_5min"]
)

valid_score = vibration_score.dropna()

print(
    f"Valid vibration regime-score rows: "
    f"{len(valid_score):,}"
)

# Robust quantile thresholds
q25 = valid_score.quantile(0.25)
q50 = valid_score.quantile(0.50)
q75 = valid_score.quantile(0.75)
q90 = valid_score.quantile(0.90)
q95 = valid_score.quantile(0.95)

print("\nVibration score thresholds:")

print(f"Q25: {q25:.6f}")
print(f"Q50: {q50:.6f}")
print(f"Q75: {q75:.6f}")
print(f"Q90: {q90:.6f}")
print(f"Q95: {q95:.6f}")


# ============================================================
# 7. ASSIGN OPERATING REGIMES
# ============================================================

def classify_vibration(value):

    if pd.isna(value):
        return "Unknown"

    if value <= q25:
        return "Low"

    elif value <= q75:
        return "Normal"

    elif value <= q90:
        return "Elevated"

    else:
        return "High"


indexed["vibration_regime"] = (
    indexed["vibration_energy_mean_5min"]
    .apply(classify_vibration)
)


# ============================================================
# 8. REGIME SUMMARY
# ============================================================

print_section("REGIME DISTRIBUTION")

regime_counts = (
    indexed["vibration_regime"]
    .value_counts()
    .rename_axis("regime")
    .reset_index(name="rows")
)

regime_counts["percentage"] = (
    regime_counts["rows"]
    / regime_counts["rows"].sum()
    * 100
)

print(
    regime_counts.to_string(
        index=False
    )
)

regime_counts.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_regime_distribution.csv"
    ),
    index=False
)


# ============================================================
# 9. REGIME STATISTICS FOR ALL SENSORS
# ============================================================

print_section("SENSOR STATISTICS BY REGIME")

regime_sensor_summary = (
    indexed
    .groupby("vibration_regime")[
        SENSOR_COLS + ["vibration_energy"]
    ]
    .agg(
        ["count", "mean", "std", "min", "max"]
    )
)

regime_sensor_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_sensor_statistics_by_regime.csv"
    )
)

print(
    "Saved sensor statistics by regime."
)


# ============================================================
# 10. FIND MAJOR REGIME CHANGES
# ============================================================

print_section("MAJOR REGIME CHANGES")

regime_series = (
    indexed["vibration_regime"]
)

regime_change_mask = (
    regime_series != regime_series.shift(1)
)

regime_changes = indexed.loc[
    regime_change_mask
].copy()

regime_changes = regime_changes[
    [
        "segment_id",
        "vibration_energy_mean_5min",
        "vibration_regime"
    ]
].copy()

regime_changes["previous_regime"] = (
    regime_series.shift(1)
    .loc[regime_changes.index]
    .values
)

regime_changes = regime_changes[
    regime_changes["previous_regime"].notna()
].copy()

regime_changes = regime_changes[
    [
        "segment_id",
        "previous_regime",
        "vibration_regime",
        "vibration_energy_mean_5min"
    ]
]

print(
    f"Regime transitions detected: "
    f"{len(regime_changes):,}"
)

print("\nFirst regime transitions:")

print(
    regime_changes.head(20).to_string(
        index=True
    )
)

regime_changes.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_regime_changes.csv"
    )
)


# ============================================================
# 11. FIND STRONG VIBRATION SHIFTS
# ============================================================

print_section("STRONG VIBRATION SHIFTS")

# Compare short-term and longer-term vibration levels.

indexed["vibration_mean_1min"] = (
    indexed["vibration_energy"]
    .rolling("1min")
    .mean()
)

indexed["vibration_mean_10min"] = (
    indexed["vibration_energy"]
    .rolling("10min")
    .mean()
)

indexed["vibration_shift_ratio"] = (
    indexed["vibration_mean_1min"]
    /
    indexed["vibration_mean_10min"]
)

# Avoid division problems
indexed["vibration_shift_ratio"] = (
    indexed["vibration_shift_ratio"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

# Strong upward changes
strong_up = indexed[
    indexed["vibration_shift_ratio"] > 1.5
].copy()

# Strong downward changes
strong_down = indexed[
    indexed["vibration_shift_ratio"] < 0.67
].copy()

print(
    f"Strong upward vibration shifts: "
    f"{len(strong_up):,}"
)

print(
    f"Strong downward vibration shifts: "
    f"{len(strong_down):,}"
)

shift_summary = pd.DataFrame(
    {
        "type": [
            "Strong upward shift",
            "Strong downward shift"
        ],
        "threshold": [
            "> 1.50",
            "< 0.67"
        ],
        "rows": [
            len(strong_up),
            len(strong_down)
        ]
    }
)

shift_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_strong_vibration_shifts.csv"
    ),
    index=False
)


# ============================================================
# 12. HOURLY OPERATING-STATE SUMMARY
# ============================================================

print_section("HOURLY OPERATING-STATE SUMMARY")

hourly = indexed.resample("1h").agg(
    vibration_mean=(
        "vibration_energy",
        "mean"
    ),
    vibration_std=(
        "vibration_energy",
        "std"
    ),
    vibration_max=(
        "vibration_energy",
        "max"
    ),
    temperature_mean=(
        TEMPERATURE_COL,
        "mean"
    ),
    temperature_std=(
        TEMPERATURE_COL,
        "std"
    ),
    mag_x_mean=(
        "mag_x",
        "mean"
    ),
    mag_y_mean=(
        "mag_y",
        "mean"
    ),
    mag_z_mean=(
        "mag_z",
        "mean"
    ),
    samples=(
        "vibration_energy",
        "count"
    )
)

hourly["vibration_regime"] = (
    hourly["vibration_mean"]
    .apply(classify_vibration)
)

hourly = hourly.reset_index()

print(
    hourly.to_string(
        index=False
    )
)

hourly.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_hourly_operating_state.csv"
    ),
    index=False
)


# ============================================================
# 13. PLOT 1 — VIBRATION ENERGY
# ============================================================

print_section("CREATING PLOTS")

plt.figure(figsize=(15, 6))

plt.plot(
    indexed.index,
    indexed["vibration_energy"],
    linewidth=0.7,
    alpha=0.5,
    label="Vibration Energy"
)

plt.plot(
    indexed.index,
    indexed["vibration_energy_mean_5min"],
    linewidth=2,
    label="5-min Rolling Mean"
)

plt.xlabel("Time")
plt.ylabel("Vibration Energy")
plt.title("Vibration Energy and Operating-State Trend")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "step15_vibration_regime_timeline.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 14. PLOT 2 — REGIME TIMELINE
# ============================================================

regime_numeric_map = {
    "Unknown": 0,
    "Low": 1,
    "Normal": 2,
    "Elevated": 3,
    "High": 4
}

regime_numeric = (
    indexed["vibration_regime"]
    .map(regime_numeric_map)
)

plt.figure(figsize=(15, 5))

plt.plot(
    indexed.index,
    regime_numeric,
    linewidth=1
)

plt.yticks(
    [0, 1, 2, 3, 4],
    [
        "Unknown",
        "Low",
        "Normal",
        "Elevated",
        "High"
    ]
)

plt.xlabel("Time")
plt.ylabel("Operating Regime")
plt.title("Detected Vibration Operating Regimes")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "step15_regime_timeline.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 15. PLOT 3 — TEMPERATURE
# ============================================================

plt.figure(figsize=(15, 5))

plt.plot(
    indexed.index,
    indexed[TEMPERATURE_COL],
    linewidth=0.8,
    label="Temperature"
)

plt.plot(
    indexed.index,
    indexed["temperature_mean_10min"],
    linewidth=2,
    label="10-min Rolling Mean"
)

plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Operating-State Trend")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "step15_temperature_regime_timeline.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 16. PLOT 4 — MAGNETIC SENSOR LEVELS
# ============================================================

plt.figure(figsize=(15, 6))

for col in MAGNETIC_COLS:

    plt.plot(
        indexed.index,
        indexed[col],
        linewidth=0.8,
        label=col
    )

plt.xlabel("Time")
plt.ylabel("Magnetic Value")
plt.title("Magnetic Sensor Operating Behavior")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "step15_magnetic_operating_behavior.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 17. SENSOR CORRELATION BY REGIME
# ============================================================

print_section("CORRELATION BY VIBRATION REGIME")

correlation_reports = []

for regime in [
    "Low",
    "Normal",
    "Elevated",
    "High"
]:

    regime_data = indexed[
        indexed["vibration_regime"] == regime
    ][SENSOR_COLS]

    if len(regime_data) < 10:
        continue

    corr = regime_data.corr()

    output_file = os.path.join(
        REPORT_DIR,
        f"step15_correlation_{safe_filename(regime)}.csv"
    )

    corr.to_csv(output_file)

    correlation_reports.append(
        {
            "regime": regime,
            "rows": len(regime_data),
            "mean_vibration_energy":
                regime_data.assign(
                    vibration_energy=(
                        np.sqrt(
                            regime_data["vib_x_rms"] ** 2
                            + regime_data["vib_y_rms"] ** 2
                            + regime_data["vib_z_rms"] ** 2
                        )
                    )
                )[
                    "vibration_energy"
                ].mean()
        }
    )


correlation_summary = pd.DataFrame(
    correlation_reports
)

correlation_summary.to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_correlation_regime_summary.csv"
    ),
    index=False
)


# ============================================================
# 18. IMPORTANT REGIME FINDINGS
# ============================================================

print_section("REGIME FINDINGS")

findings = {}

findings["input_rows"] = int(len(df))

findings["continuous_segments"] = int(
    len(segment_summary)
)

findings["major_time_gaps"] = int(
    len(major_gaps)
)

findings["regime_transitions"] = int(
    len(regime_changes)
)

findings["vibration_score_q25"] = float(q25)

findings["vibration_score_q50"] = float(q50)

findings["vibration_score_q75"] = float(q75)

findings["vibration_score_q90"] = float(q90)

findings["vibration_score_q95"] = float(q95)

findings["strong_upward_shifts"] = int(
    len(strong_up)
)

findings["strong_downward_shifts"] = int(
    len(strong_down)
)

findings["regime_distribution"] = (
    regime_counts
    .set_index("regime")["percentage"]
    .to_dict()
)

print(
    json.dumps(
        findings,
        indent=4,
        default=str
    )
)

with open(
    os.path.join(
        REPORT_DIR,
        "step15_regime_findings.json"
    ),
    "w"
) as f:

    json.dump(
        findings,
        f,
        indent=4,
        default=str
    )


# ============================================================
# 19. SAVE ANALYSIS DATA
# ============================================================

analysis_output = indexed.reset_index()

# Keep the most useful columns rather than saving
# every intermediate calculation.

analysis_columns = [
    TIMESTAMP_COL,
    "segment_id",
    "vibration_energy",
    "vibration_energy_mean_1min",
    "vibration_energy_mean_5min",
    "vibration_energy_mean_10min",
    "vibration_energy_mean_30min",
    "vibration_energy_std_5min",
    "vibration_energy_std_10min",
    "vibration_energy_max_5min",
    "temperature",
    "temperature_mean_5min",
    "temperature_mean_10min",
    "temperature_mean_30min",
    "vibration_regime",
    "vibration_shift_ratio"
]

analysis_columns = [
    col
    for col in analysis_columns
    if col in analysis_output.columns
]

analysis_output[
    analysis_columns
].to_csv(
    os.path.join(
        REPORT_DIR,
        "step15_regime_analysis_data.csv"
    ),
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print_section("STEP 15 COMPLETED")

print("Regime analysis completed successfully.")

print("\nReports saved:")

print(
    "  outputs/reports/"
    "step15_major_time_gaps.csv"
)

print(
    "  outputs/reports/"
    "step15_segment_summary.csv"
)

print(
    "  outputs/reports/"
    "step15_regime_distribution.csv"
)

print(
    "  outputs/reports/"
    "step15_sensor_statistics_by_regime.csv"
)

print(
    "  outputs/reports/"
    "step15_regime_changes.csv"
)

print(
    "  outputs/reports/"
    "step15_strong_vibration_shifts.csv"
)

print(
    "  outputs/reports/"
    "step15_hourly_operating_state.csv"
)

print(
    "  outputs/reports/"
    "step15_regime_findings.json"
)

print(
    "  outputs/reports/"
    "step15_regime_analysis_data.csv"
)

print("\nPlots saved:")

print(
    "  outputs/plots/"
    "step15_vibration_regime_timeline.png"
)

print(
    "  outputs/plots/"
    "step15_regime_timeline.png"
)

print(
    "  outputs/plots/"
    "step15_temperature_regime_timeline.png"
)

print(
    "  outputs/plots/"
    "step15_magnetic_operating_behavior.png"
)

print("\nIMPORTANT:")
print(
    "No forecasting model was trained in Step 15."
)
print(
    "This step only identifies operating regimes "
    "and temporal behavior."
)