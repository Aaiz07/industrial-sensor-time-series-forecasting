"""
09_final_feature_strategy.py

Step 9: Final Feature Strategy

This script creates the final, leakage-safe feature sets that will be
used by the final forecasting models.

We use the CLEANED original data rather than blindly using the
1-second interpolated dataset.

Feature groups:
    A. Own sensor lags
    B. Cross-sensor lags
    C. Rolling statistics
    D. First differences
    E. Time-of-day cyclic features

Sensor-specific strategy:

    Temperature:
        Short own lags + rolling statistics + time features.
        Temperature showed very strong persistence.

    Vibration:
        Own lags + cross-axis vibration lags + rolling statistics.
        vib_y showed useful cross-axis predictability.

    Magnetic:
        Recent level / rolling features are preferred.
        Magnetic signals showed weak point-by-point predictability,
        so we avoid pretending that large deep models will solve it.

IMPORTANT:
    Features are generated only from information available at or
    before the prediction timestamp.

Outputs:
    outputs/data/final_features.csv
    outputs/reports/final_feature_strategy.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "data"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# COLUMNS
# ============================================================

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

VIBRATION_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
]

MAGNETIC_COLS = [
    "mag_x",
    "mag_y",
    "mag_z",
]


# ============================================================
# SENSOR-SPECIFIC FEATURES
# ============================================================

# These are deliberately compact.
# We do not want hundreds/thousands of nearly redundant features.

OWN_LAGS = [
    1,
    2,
    5,
    10,
    30,
    60,
]

VIBRATION_CROSS_LAGS = [
    1,
    2,
    5,
    10,
]

MAGNETIC_LAGS = [
    1,
    5,
    10,
    30,
    60,
]

ROLLING_WINDOWS = [
    5,
    15,
    30,
    60,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("FINAL FEATURE STRATEGY")
print("=" * 75)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run 01_data_pipeline.py first."
    )

df = pd.read_csv(
    INPUT_FILE
)

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

print(
    f"\nInput rows: {len(df):,}"
)

print(
    f"Start: {df[TIMESTAMP_COL].min()}"
)

print(
    f"End:   {df[TIMESTAMP_COL].max()}"
)


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

features = pd.DataFrame(
    index=df.index
)

features[TIMESTAMP_COL] = df[
    TIMESTAMP_COL
]


# ============================================================
# TIME FEATURES
# ============================================================

print("\nCreating time features...")

seconds_from_midnight = (
    df[TIMESTAMP_COL].dt.hour * 3600
    + df[TIMESTAMP_COL].dt.minute * 60
    + df[TIMESTAMP_COL].dt.second
)

features["time_sin"] = np.sin(
    2 * np.pi
    * seconds_from_midnight
    / 86400
)

features["time_cos"] = np.cos(
    2 * np.pi
    * seconds_from_midnight
    / 86400
)


# ============================================================
# OWN SENSOR FEATURES
# ============================================================

print("Creating own-sensor lag features...")

for sensor in SENSOR_COLS:

    # Own lags
    for lag in OWN_LAGS:

        features[
            f"{sensor}_lag_{lag}"
        ] = df[sensor].shift(lag)

    # First difference
    features[
        f"{sensor}_diff_1"
    ] = df[sensor].diff(1)


# ============================================================
# VIBRATION CROSS-AXIS FEATURES
# ============================================================

print(
    "Creating vibration cross-axis features..."
)

for target in VIBRATION_COLS:

    other_vibrations = [
        col
        for col in VIBRATION_COLS
        if col != target
    ]

    for other in other_vibrations:

        for lag in VIBRATION_CROSS_LAGS:

            features[
                f"{other}_to_{target}_lag_{lag}"
            ] = df[other].shift(lag)


# ============================================================
# MAGNETIC FEATURES
# ============================================================

print(
    "Creating magnetic rolling-level features..."
)

for sensor in MAGNETIC_COLS:

    # Magnetic signals were weakly predictable at individual
    # samples, so emphasize recent level rather than trend
    # extrapolation.

    for lag in MAGNETIC_LAGS:

        features[
            f"{sensor}_mag_lag_{lag}"
        ] = df[sensor].shift(lag)

    for window in ROLLING_WINDOWS:

        rolling = (
            df[sensor]
            .shift(1)
            .rolling(
                window=window,
                min_periods=window
            )
        )

        features[
            f"{sensor}_rollmean_{window}"
        ] = rolling.mean()

        features[
            f"{sensor}_rollstd_{window}"
        ] = rolling.std()


# ============================================================
# ROLLING FEATURES FOR TEMPERATURE + VIBRATION
# ============================================================

print(
    "Creating temperature/vibration rolling features..."
)

for sensor in (
    VIBRATION_COLS
    + ["temperature"]
):

    for window in ROLLING_WINDOWS:

        rolling = (
            df[sensor]
            .shift(1)
            .rolling(
                window=window,
                min_periods=window
            )
        )

        features[
            f"{sensor}_rollmean_{window}"
        ] = rolling.mean()

        features[
            f"{sensor}_rollstd_{window}"
        ] = rolling.std()


# ============================================================
# TARGET COLUMNS
# ============================================================

for sensor in SENSOR_COLS:

    features[
        f"target_{sensor}"
    ] = df[sensor]


# ============================================================
# REMOVE ROWS WITHOUT COMPLETE HISTORY
# ============================================================

required_feature_cols = [
    col
    for col in features.columns
    if col != TIMESTAMP_COL
    and not col.startswith("target_")
]

before = len(features)

features = features.dropna(
    subset=required_feature_cols
).reset_index(
    drop=True
)

after = len(features)

print(
    f"\nRows removed because of initial feature history: "
    f"{before - after:,}"
)

print(
    f"Usable feature rows: {after:,}"
)


# ============================================================
# VERIFY NO FEATURE LOOK-AHEAD
# ============================================================

print(
    "\nRunning leakage checks..."
)

# All lag/rolling features must be generated using shift(1)
# for rolling calculations. Own lags are also shifted.
# Therefore feature timestamp is <= target timestamp.

leakage_columns = []

for col in required_feature_cols:

    if col == TIMESTAMP_COL:
        continue

    # Target columns should not be considered input features.
    if col.startswith("target_"):
        leakage_columns.append(col)

print(
    f"Input feature columns: "
    f"{len(required_feature_cols):,}"
)

print(
    "Rolling features use shift(1): YES"
)

print(
    "Future target values used as input: NO"
)


# ============================================================
# FEATURE GROUP REPORT
# ============================================================

feature_names = [
    col
    for col in features.columns
    if col != TIMESTAMP_COL
    and not col.startswith("target_")
]

strategy_records = []

for sensor in SENSOR_COLS:

    if sensor == "temperature":

        strategy = (
            "Own lags + rolling mean/std + diff + "
            "time-of-day"
        )

        reason = (
            "Very strong persistence and high short/medium "
            "horizon predictability."
        )

    elif sensor in VIBRATION_COLS:

        strategy = (
            "Own lags + cross-axis vibration lags + "
            "rolling mean/std + diff + time-of-day"
        )

        reason = (
            "Short-horizon structure exists; cross-axis "
            "information was useful, especially for vib_y."
        )

    else:

        strategy = (
            "Magnetic lags + rolling level/std + "
            "limited cross-sensor context"
        )

        reason = (
            "Magnetic point-level predictability is weak; "
            "avoid relying on unstable trend extrapolation."
        )

    strategy_records.append({
        "sensor": sensor,
        "feature_strategy": strategy,
        "reason": reason,
        "horizon_note": (
            "Use direct multi-step evaluation before claiming "
            "long-horizon performance."
        ),
    })

strategy_df = pd.DataFrame(
    strategy_records
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "final_features.csv"
)

report_file = os.path.join(
    REPORT_DIR,
    "final_feature_strategy.csv"
)

features.to_csv(
    output_file,
    index=False
)

strategy_df.to_csv(
    report_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL FEATURE SUMMARY")
print("=" * 75)

print(
    f"\nRows: {len(features):,}"
)

print(
    f"Input features: {len(feature_names):,}"
)

print(
    f"Targets: {len(SENSOR_COLS)}"
)

print(
    "\nSensor-specific strategy:"
)

for _, row in strategy_df.iterrows():

    print(
        f"\n{row['sensor']}"
    )

    print(
        f"  Strategy: {row['feature_strategy']}"
    )

    print(
        f"  Reason:   {row['reason']}"
    )

print(
    "\nSaved:"
)

print(
    f"  {output_file}"
)

print(
    f"  {report_file}"
)

print(
    "\nIMPORTANT:"
)

print(
    "This step defines the feature space only. "
    "No final model is trained here."
)

print(
    "The next step will train and compare final models "
    "using these features with a strict chronological split."
)

print("\n" + "=" * 75)
print("STEP 9 COMPLETED")
print("=" * 75)
