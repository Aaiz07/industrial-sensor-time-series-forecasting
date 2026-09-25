import os
import pandas as pd

# ============================================================
# STEP 12: CREATE TRUE FUTURE TARGETS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_FILE = os.path.join(
    BASE_DIR, "outputs", "data", "final_features.csv"
)

CLEAN_FILE = os.path.join(
    BASE_DIR, "outputs", "data", "sensor_data_cleaned.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "data")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# First forecasting horizon
HORIZON_SECONDS = 60

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]

print("=" * 70)
print("STEP 12: TRUE FUTURE TARGET CREATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

features = pd.read_csv(
    FEATURE_FILE,
    parse_dates=["ts"]
)

clean = pd.read_csv(
    CLEAN_FILE,
    parse_dates=["ts"]
)

features = features.sort_values("ts").reset_index(drop=True)
clean = clean.sort_values("ts").reset_index(drop=True)

print(f"\nFeature rows: {len(features):,}")
print(f"Cleaned rows: {len(clean):,}")
print(f"Forecast horizon: {HORIZON_SECONDS} seconds")

# ------------------------------------------------------------
# 2. REMOVE OLD TARGET COLUMNS
# ------------------------------------------------------------

old_target_cols = [
    c for c in features.columns
    if c.startswith("target_")
]

feature_cols = [
    c for c in features.columns
    if c != "ts" and c not in old_target_cols
]

print(f"\nInput feature columns: {len(feature_cols)}")
print(f"Old target columns removed: {len(old_target_cols)}")

# ------------------------------------------------------------
# 3. CREATE FUTURE TARGET TABLE
# ------------------------------------------------------------

future_targets = clean[
    ["ts"] + SENSOR_COLS
].copy()

future_targets = future_targets.rename(
    columns={
        sensor: f"future_{sensor}"
        for sensor in SENSOR_COLS
    }
)

# For every feature time t, target must be t + 60 seconds.
future_targets["target_ts"] = (
    future_targets["ts"]
    - pd.Timedelta(seconds=HORIZON_SECONDS)
)

future_targets = future_targets.drop(columns=["ts"])

# ------------------------------------------------------------
# 4. ALIGN t WITH t + 60 SECONDS
# ------------------------------------------------------------

dataset = features[
    ["ts"] + feature_cols
].copy()

dataset["target_ts"] = (
    dataset["ts"]
    + pd.Timedelta(seconds=HORIZON_SECONDS)
)

dataset = dataset.merge(
    future_targets,
    on="target_ts",
    how="left"
)

rows_before = len(dataset)

dataset = dataset.dropna(
    subset=[
        f"future_{sensor}"
        for sensor in SENSOR_COLS
    ]
).reset_index(drop=True)

rows_after = len(dataset)

print("\nTRUE FUTURE TARGET ALIGNMENT")
print(f"Rows before matching: {rows_before:,}")
print(f"Rows after matching:  {rows_after:,}")

coverage = rows_after / rows_before * 100

print(f"Target match coverage: {coverage:.2f}%")

# ------------------------------------------------------------
# 5. VERIFY FUTURE TIME
# ------------------------------------------------------------

dataset["actual_horizon_seconds"] = (
    dataset["target_ts"] - dataset["ts"]
).dt.total_seconds()

print("\nHORIZON CHECK")

print(
    dataset["actual_horizon_seconds"]
    .describe()
    .to_string()
)

horizon_pass = (
    dataset["actual_horizon_seconds"] == HORIZON_SECONDS
).all()

print(
    "\nAll targets exactly 60 seconds in the future:",
    "PASS" if horizon_pass else "FAIL"
)

# ------------------------------------------------------------
# 6. CURRENT VS FUTURE CORRELATION
# ------------------------------------------------------------

correlation_rows = []

for sensor in SENSOR_COLS:

    future_col = f"future_{sensor}"

    check = dataset[
        ["ts", future_col]
    ].copy()

    check = check.merge(
        clean[["ts", sensor]],
        on="ts",
        how="left"
    )

    correlation = check[sensor].corr(
        check[future_col]
    )

    correlation_rows.append({
        "sensor": sensor,
        "horizon_seconds": HORIZON_SECONDS,
        "current_vs_future_correlation": correlation,
        "rows": len(check)
    })

correlation_report = pd.DataFrame(
    correlation_rows
)

print("\nCURRENT VS FUTURE CORRELATION")
print(
    correlation_report.to_string(index=False)
)

# ------------------------------------------------------------
# 7. LEAKAGE CHECK
# ------------------------------------------------------------

future_features = [
    c for c in feature_cols
    if c.startswith("future_")
]

print("\nLEAKAGE CHECK")
print(
    "Future target columns inside input features:",
    len(future_features)
)

if len(future_features) == 0:
    print(
        "PASS - future targets are NOT being used "
        "as input features."
    )
else:
    print("FAIL - future target leakage detected!")

# ------------------------------------------------------------
# 8. SAVE DATASET
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    f"future_features_{HORIZON_SECONDS}s.csv"
)

dataset.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# 9. SAVE REPORTS
# ------------------------------------------------------------

correlation_file = os.path.join(
    REPORT_DIR,
    f"future_target_alignment_{HORIZON_SECONDS}s.csv"
)

correlation_report.to_csv(
    correlation_file,
    index=False
)

summary = pd.DataFrame({
    "metric": [
        "input_feature_rows",
        "matched_future_target_rows",
        "unmatched_rows",
        "match_coverage_percent",
        "forecast_horizon_seconds"
    ],
    "value": [
        rows_before,
        rows_after,
        rows_before - rows_after,
        coverage,
        HORIZON_SECONDS
    ]
})

summary_file = os.path.join(
    REPORT_DIR,
    f"future_target_match_summary_{HORIZON_SECONDS}s.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ------------------------------------------------------------
# 10. FINAL OUTPUT
# ------------------------------------------------------------

print("\nSAVED:")
print(f"  {output_file}")
print(f"  {correlation_file}")
print(f"  {summary_file}")

print("\n" + "=" * 70)
print("STEP 12 COMPLETED")
print("=" * 70)

print(
    "\nNext step:"
    "\nTrain models using:"
    "\n    FEATURES at t"
    "\n        ↓"
    "\n    SENSOR VALUES at t + 60 seconds"
)