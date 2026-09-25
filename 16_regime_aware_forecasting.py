# ============================================================
# 16_regime_aware_forecasting.py
#
# Purpose:
# Compare baseline forecasting features against
# regime-aware forecasting features for a TRUE 60-second
# future forecast.
#
# IMPORTANT:
#   Features at time t -> target at time t + 60 seconds
#
# No future information is used to create features.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FUTURE_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "future_features_60s.csv"
)

CLEAN_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

REGIME_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "step15_regime_analysis_data.csv"
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


TIMESTAMP_COL = "ts"

FORECAST_SECONDS = 60

TARGETS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

FUTURE_TARGETS = [
    "future_vib_x_rms",
    "future_vib_y_rms",
    "future_vib_z_rms",
    "future_mag_x",
    "future_mag_y",
    "future_mag_z",
    "future_temperature"
]

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "RMSE": rmse(
            y_true,
            y_pred
        ),
        "R2": r2_score(
            y_true,
            y_pred
        )
    }


def safe_filename(name):
    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
    )


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print_section(
    "STEP 16: REGIME-AWARE 60-SECOND FORECASTING"
)

for file_path in [
    FUTURE_FILE,
    CLEAN_FILE,
    REGIME_FILE
]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

print("All required input files found.")


# ============================================================
# 2. LOAD DATA
# ============================================================

future_df = pd.read_csv(FUTURE_FILE)
clean_df = pd.read_csv(CLEAN_FILE)
regime_df = pd.read_csv(REGIME_FILE)

print(
    f"Future-target dataset rows: "
    f"{len(future_df):,}"
)

print(
    f"Cleaned dataset rows: "
    f"{len(clean_df):,}"
)

print(
    f"Regime-analysis rows: "
    f"{len(regime_df):,}"
)


# ============================================================
# 3. PARSE TIMESTAMPS
# ============================================================

for df_name, data in [
    ("future_df", future_df),
    ("clean_df", clean_df),
    ("regime_df", regime_df)
]:

    if TIMESTAMP_COL not in data.columns:
        raise ValueError(
            f"{TIMESTAMP_COL} missing from {df_name}"
        )

    data[TIMESTAMP_COL] = pd.to_datetime(
        data[TIMESTAMP_COL],
        errors="coerce"
    )


future_df = future_df.dropna(
    subset=[TIMESTAMP_COL]
).copy()

clean_df = clean_df.dropna(
    subset=[TIMESTAMP_COL]
).copy()

regime_df = regime_df.dropna(
    subset=[TIMESTAMP_COL]
).copy()


# ============================================================
# 4. VERIFY TRUE FUTURE TARGETS
# ============================================================

print_section(
    "TRUE FUTURE TARGET VERIFICATION"
)

required_future_columns = (
    [TIMESTAMP_COL]
    + FUTURE_TARGETS
)

missing_future = [
    col
    for col in required_future_columns
    if col not in future_df.columns
]

if missing_future:
    raise ValueError(
        f"Missing future-target columns: "
        f"{missing_future}"
    )

if "target_ts" in future_df.columns:

    future_df["target_ts"] = pd.to_datetime(
        future_df["target_ts"],
        errors="coerce"
    )

    future_df["actual_horizon_seconds"] = (
        future_df["target_ts"]
        - future_df[TIMESTAMP_COL]
    ).dt.total_seconds()

    valid_horizons = (
        future_df["actual_horizon_seconds"]
        .dropna()
    )

    print(
        f"Median actual horizon: "
        f"{valid_horizons.median():.2f} seconds"
    )

    if not np.allclose(
        valid_horizons,
        FORECAST_SECONDS,
        atol=0.001
    ):
        raise ValueError(
            "Future targets are not exactly "
            "60 seconds ahead."
        )

    print(
        "60-second target alignment: PASS"
    )


# ============================================================
# 5. MERGE CURRENT SENSOR VALUES
#
# Needed for persistence baseline and regime features.
# These are values at time t only.
# ============================================================

print_section(
    "CURRENT SENSOR ALIGNMENT"
)

current_columns = [
    TIMESTAMP_COL
] + TARGETS

current_values = clean_df[
    current_columns
].copy()

# If duplicate timestamps somehow exist,
# average them safely.
current_values = (
    current_values
    .groupby(TIMESTAMP_COL, as_index=False)[TARGETS]
    .mean()
)

df = future_df.merge(
    current_values,
    on=TIMESTAMP_COL,
    how="inner",
    suffixes=("", "_current")
)

print(
    f"Rows after current-value alignment: "
    f"{len(df):,}"
)


# ============================================================
# 6. MERGE REGIME INFORMATION
# ============================================================

print_section(
    "REGIME INFORMATION ALIGNMENT"
)

regime_columns = [
    TIMESTAMP_COL,
    "vibration_energy",
    "vibration_energy_mean_1min",
    "vibration_energy_mean_5min",
    "vibration_energy_mean_10min",
    "vibration_energy_mean_30min",
    "vibration_energy_std_5min",
    "vibration_energy_std_10min",
    "vibration_energy_max_5min",
    "temperature_mean_5min",
    "temperature_mean_10min",
    "temperature_mean_30min",
    "vibration_regime",
    "vibration_shift_ratio"
]

regime_columns = [
    col
    for col in regime_columns
    if col in regime_df.columns
]

regime_features = regime_df[
    regime_columns
].copy()

regime_features = (
    regime_features
    .drop_duplicates(
        subset=[TIMESTAMP_COL]
    )
)

df = df.merge(
    regime_features,
    on=TIMESTAMP_COL,
    how="left"
)

print(
    f"Rows after regime alignment: "
    f"{len(df):,}"
)


# ============================================================
# 7. CREATE REGIME FEATURES
# ============================================================

print_section(
    "CREATING REGIME-AWARE FEATURES"
)

# ------------------------------------------------------------
# Numeric regime encoding
# ------------------------------------------------------------

regime_map = {
    "Low": 0,
    "Normal": 1,
    "Elevated": 2,
    "High": 3,
    "Unknown": -1
}

df["regime_code"] = (
    df["vibration_regime"]
    .map(regime_map)
    .fillna(-1)
)


# ------------------------------------------------------------
# Time-of-day
# ------------------------------------------------------------

df["hour"] = (
    df[TIMESTAMP_COL]
    .dt.hour
    +
    df[TIMESTAMP_COL]
    .dt.minute / 60.0
)

df["time_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24.0
)

df["time_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24.0
)


# ------------------------------------------------------------
# Regime-change indicator
# ------------------------------------------------------------

df["previous_regime"] = (
    df["vibration_regime"]
    .shift(1)
)

df["regime_changed"] = (
    df["vibration_regime"]
    != df["previous_regime"]
).astype(int)


# ------------------------------------------------------------
# Regime persistence
# ------------------------------------------------------------

# Number of consecutive rows in the same regime.
# This is based ONLY on information available at time t.

regime_group = (
    df["vibration_regime"]
    != df["vibration_regime"].shift(1)
).cumsum()

df["regime_duration_rows"] = (
    df.groupby(regime_group)
    .cumcount()
    + 1
)


# ------------------------------------------------------------
# Vibration balance between axes
# ------------------------------------------------------------

df["vibration_energy_ratio_x"] = (
    df["vib_x_rms"]
    /
    (df["vibration_energy"] + 1e-8)
)

df["vibration_energy_ratio_y"] = (
    df["vib_y_rms"]
    /
    (df["vibration_energy"] + 1e-8)
)

df["vibration_energy_ratio_z"] = (
    df["vib_z_rms"]
    /
    (df["vibration_energy"] + 1e-8)
)


# ------------------------------------------------------------
# Vibration spread
# ------------------------------------------------------------

df["vibration_axis_std"] = (
    df[VIBRATION_COLS]
    .std(axis=1)
)

df["vibration_axis_mean"] = (
    df[VIBRATION_COLS]
    .mean(axis=1)
)

df["vibration_axis_max"] = (
    df[VIBRATION_COLS]
    .max(axis=1)
)


# ------------------------------------------------------------
# Temperature relative to recent operating state
# ------------------------------------------------------------

if "temperature_mean_10min" in df.columns:

    df["temperature_deviation_10min"] = (
        df[TEMPERATURE_COL]
        -
        df["temperature_mean_10min"]
    )

else:

    df["temperature_deviation_10min"] = 0.0


# ------------------------------------------------------------
# Magnetic magnitude
# ------------------------------------------------------------

df["magnetic_magnitude"] = np.sqrt(
    df["mag_x"] ** 2
    + df["mag_y"] ** 2
    + df["mag_z"] ** 2
)


# ============================================================
# 8. DEFINE BASELINE FEATURES
# ============================================================

print_section(
    "BASELINE FEATURE SET"
)

# These are features already known to work from our
# previous diagnostics.
#
# IMPORTANT:
# All are values available at time t or earlier.

baseline_features = []

# Own historical values
for sensor in TARGETS:

    for lag in [
        1,
        2,
        5,
        10,
        30,
        60
    ]:

        feature = f"{sensor}_lag_{lag}"

        if feature in df.columns:
            baseline_features.append(feature)


# Existing rolling features
for sensor in TARGETS:

    for window in [
        5,
        15,
        30,
        60
    ]:

        for suffix in [
            "rollmean",
            "rollstd"
        ]:

            feature = (
                f"{sensor}_{suffix}_{window}"
            )

            if feature in df.columns:
                baseline_features.append(
                    feature
                )


# Difference features
for sensor in TARGETS:

    feature = f"{sensor}_diff_1"

    if feature in df.columns:
        baseline_features.append(feature)


# Time features
for feature in [
    "time_sin",
    "time_cos"
]:

    if feature in df.columns:
        baseline_features.append(feature)


baseline_features = list(
    dict.fromkeys(baseline_features)
)

print(
    f"Baseline features found: "
    f"{len(baseline_features)}"
)


# ============================================================
# 9. DEFINE REGIME-AWARE FEATURES
# ============================================================

print_section(
    "REGIME-AWARE FEATURE SET"
)

regime_specific_features = [
    "vibration_energy",
    "vibration_energy_mean_1min",
    "vibration_energy_mean_5min",
    "vibration_energy_mean_10min",
    "vibration_energy_mean_30min",
    "vibration_energy_std_5min",
    "vibration_energy_std_10min",
    "vibration_energy_max_5min",
    "temperature_mean_5min",
    "temperature_mean_10min",
    "temperature_mean_30min",
    "regime_code",
    "regime_changed",
    "regime_duration_rows",
    "vibration_energy_ratio_x",
    "vibration_energy_ratio_y",
    "vibration_energy_ratio_z",
    "vibration_axis_std",
    "vibration_axis_mean",
    "vibration_axis_max",
    "temperature_deviation_10min",
    "magnetic_magnitude"
]

regime_specific_features = [
    col
    for col in regime_specific_features
    if col in df.columns
]

regime_features_final = (
    baseline_features
    +
    regime_specific_features
)

regime_features_final = list(
    dict.fromkeys(regime_features_final)
)

print(
    f"Regime-aware features: "
    f"{len(regime_features_final)}"
)


# ============================================================
# 10. LEAKAGE CHECK
# ============================================================

print_section(
    "LEAKAGE CHECK"
)

forbidden_features = (
    FUTURE_TARGETS
    + [
        "target_ts",
        "actual_horizon_seconds"
    ]
)

leakage_found = [
    feature
    for feature in regime_features_final
    if feature in forbidden_features
]

if leakage_found:

    raise ValueError(
        "LEAKAGE DETECTED:\n"
        + "\n".join(leakage_found)
    )

print(
    "Future target columns in features: 0"
)

print(
    "Future timestamp information in features: 0"
)

print(
    "LEAKAGE CHECK: PASS"
)


# ============================================================
# 11. CLEAN DATA
# ============================================================

required_columns = (
    regime_features_final
    + FUTURE_TARGETS
    + TARGETS
)

df_model = df[
    [
        TIMESTAMP_COL
    ]
    + list(
        dict.fromkeys(required_columns)
    )
].copy()

df_model = df_model.replace(
    [np.inf, -np.inf],
    np.nan
)

before_drop = len(df_model)

df_model = df_model.dropna(
    subset=(
        regime_features_final
        + FUTURE_TARGETS
    )
).reset_index(drop=True)

print(
    f"Rows before missing-value removal: "
    f"{before_drop:,}"
)

print(
    f"Rows after cleaning: "
    f"{len(df_model):,}"
)


# ============================================================
# 12. CHRONOLOGICAL SPLIT
# ============================================================

print_section(
    "CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT"
)

n = len(df_model)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)

train_df = df_model.iloc[
    :train_end
].copy()

val_df = df_model.iloc[
    train_end:val_end
].copy()

test_df = df_model.iloc[
    val_end:
].copy()

print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)

print(
    f"\nTrain period:"
    f"\n  {train_df[TIMESTAMP_COL].min()}"
    f"\n  {train_df[TIMESTAMP_COL].max()}"
)

print(
    f"\nValidation period:"
    f"\n  {val_df[TIMESTAMP_COL].min()}"
    f"\n  {val_df[TIMESTAMP_COL].max()}"
)

print(
    f"\nTest period:"
    f"\n  {test_df[TIMESTAMP_COL].min()}"
    f"\n  {test_df[TIMESTAMP_COL].max()}"
)


# ============================================================
# 13. PREPARE MATRICES
# ============================================================

X_train_base = train_df[
    baseline_features
].values

X_val_base = val_df[
    baseline_features
].values

X_test_base = test_df[
    baseline_features
].values


X_train_regime = train_df[
    regime_features_final
].values

X_val_regime = val_df[
    regime_features_final
].values

X_test_regime = test_df[
    regime_features_final
].values


# ============================================================
# 14. RIDGE SCALING
# ============================================================

print_section(
    "RIDGE MODEL PREPARATION"
)

base_scaler = StandardScaler()

X_train_base_scaled = (
    base_scaler.fit_transform(
        X_train_base
    )
)

X_val_base_scaled = (
    base_scaler.transform(
        X_val_base
    )
)

X_test_base_scaled = (
    base_scaler.transform(
        X_test_base
    )
)


regime_scaler = StandardScaler()

X_train_regime_scaled = (
    regime_scaler.fit_transform(
        X_train_regime
    )
)

X_val_regime_scaled = (
    regime_scaler.transform(
        X_val_regime
    )
)

X_test_regime_scaled = (
    regime_scaler.transform(
        X_test_regime
    )
)

print(
    "Scalers fitted on training data only."
)


# ============================================================
# 15. TRAIN MODELS
# ============================================================

print_section(
    "TRAINING BASELINE VS REGIME-AWARE MODELS"
)

results = []

test_predictions = []

model_objects = {}


for sensor, future_target in zip(
    TARGETS,
    FUTURE_TARGETS
):

    print("\n" + "-" * 72)
    print(f"SENSOR: {sensor}")
    print("-" * 72)

    y_train = train_df[
        future_target
    ].values

    y_val = val_df[
        future_target
    ].values

    y_test = test_df[
        future_target
    ].values


    # ========================================================
    # PERSISTENCE BASELINE
    # ========================================================

    persistence_val = val_df[
        sensor
    ].values

    persistence_test = test_df[
        sensor
    ].values

    p_val_metrics = calculate_metrics(
        y_val,
        persistence_val
    )

    p_test_metrics = calculate_metrics(
        y_test,
        persistence_test
    )

    results.append(
        {
            "sensor": sensor,
            "feature_set": "Persistence",
            "model": "Persistence",
            "validation_MAE":
                p_val_metrics["MAE"],
            "validation_RMSE":
                p_val_metrics["RMSE"],
            "validation_R2":
                p_val_metrics["R2"],
            "test_MAE":
                p_test_metrics["MAE"],
            "test_RMSE":
                p_test_metrics["RMSE"],
            "test_R2":
                p_test_metrics["R2"]
        }
    )


    # ========================================================
    # BASELINE RIDGE
    # ========================================================

    baseline_ridge = Ridge(
        alpha=1.0
    )

    baseline_ridge.fit(
        X_train_base_scaled,
        y_train
    )

    base_val_pred = baseline_ridge.predict(
        X_val_base_scaled
    )

    base_test_pred = baseline_ridge.predict(
        X_test_base_scaled
    )

    base_val_metrics = calculate_metrics(
        y_val,
        base_val_pred
    )

    base_test_metrics = calculate_metrics(
        y_test,
        base_test_pred
    )

    results.append(
        {
            "sensor": sensor,
            "feature_set": "Baseline",
            "model": "Ridge",
            "validation_MAE":
                base_val_metrics["MAE"],
            "validation_RMSE":
                base_val_metrics["RMSE"],
            "validation_R2":
                base_val_metrics["R2"],
            "test_MAE":
                base_test_metrics["MAE"],
            "test_RMSE":
                base_test_metrics["RMSE"],
            "test_R2":
                base_test_metrics["R2"]
        }
    )


    # ========================================================
    # REGIME-AWARE RIDGE
    # ========================================================

    regime_ridge = Ridge(
        alpha=1.0
    )

    regime_ridge.fit(
        X_train_regime_scaled,
        y_train
    )

    regime_val_pred = regime_ridge.predict(
        X_val_regime_scaled
    )

    regime_test_pred = regime_ridge.predict(
        X_test_regime_scaled
    )

    regime_val_metrics = calculate_metrics(
        y_val,
        regime_val_pred
    )

    regime_test_metrics = calculate_metrics(
        y_test,
        regime_test_pred
    )

    results.append(
        {
            "sensor": sensor,
            "feature_set": "RegimeAware",
            "model": "Ridge",
            "validation_MAE":
                regime_val_metrics["MAE"],
            "validation_RMSE":
                regime_val_metrics["RMSE"],
            "validation_R2":
                regime_val_metrics["R2"],
            "test_MAE":
                regime_test_metrics["MAE"],
            "test_RMSE":
                regime_test_metrics["RMSE"],
            "test_R2":
                regime_test_metrics["R2"]
        }
    )


    # ========================================================
    # BASELINE XGBOOST
    # ========================================================

    baseline_xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=2,
        random_state=42
    )

    baseline_xgb.fit(
        X_train_base,
        y_train
    )

    xgb_base_val_pred = (
        baseline_xgb.predict(
            X_val_base
        )
    )

    xgb_base_test_pred = (
        baseline_xgb.predict(
            X_test_base
        )
    )

    xgb_base_val_metrics = calculate_metrics(
        y_val,
        xgb_base_val_pred
    )

    xgb_base_test_metrics = calculate_metrics(
        y_test,
        xgb_base_test_pred
    )

    results.append(
        {
            "sensor": sensor,
            "feature_set": "Baseline",
            "model": "XGBoost",
            "validation_MAE":
                xgb_base_val_metrics["MAE"],
            "validation_RMSE":
                xgb_base_val_metrics["RMSE"],
            "validation_R2":
                xgb_base_val_metrics["R2"],
            "test_MAE":
                xgb_base_test_metrics["MAE"],
            "test_RMSE":
                xgb_base_test_metrics["RMSE"],
            "test_R2":
                xgb_base_test_metrics["R2"]
        }
    )


    # ========================================================
    # REGIME-AWARE XGBOOST
    # ========================================================

    regime_xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=2,
        random_state=42
    )

    regime_xgb.fit(
        X_train_regime,
        y_train
    )

    xgb_regime_val_pred = (
        regime_xgb.predict(
            X_val_regime
        )
    )

    xgb_regime_test_pred = (
        regime_xgb.predict(
            X_test_regime
        )
    )

    xgb_regime_val_metrics = calculate_metrics(
        y_val,
        xgb_regime_val_pred
    )

    xgb_regime_test_metrics = calculate_metrics(
        y_test,
        xgb_regime_test_pred
    )

    results.append(
        {
            "sensor": sensor,
            "feature_set": "RegimeAware",
            "model": "XGBoost",
            "validation_MAE":
                xgb_regime_val_metrics["MAE"],
            "validation_RMSE":
                xgb_regime_val_metrics["RMSE"],
            "validation_R2":
                xgb_regime_val_metrics["R2"],
            "test_MAE":
                xgb_regime_test_metrics["MAE"],
            "test_RMSE":
                xgb_regime_test_metrics["RMSE"],
            "test_R2":
                xgb_regime_test_metrics["R2"]
        }
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_block = pd.DataFrame(
        {
            TIMESTAMP_COL:
                test_df[TIMESTAMP_COL].values,

            "sensor":
                sensor,

            "actual":
                y_test,

            "persistence":
                persistence_test,

            "baseline_ridge":
                base_test_pred,

            "regime_ridge":
                regime_test_pred,

            "baseline_xgb":
                xgb_base_test_pred,

            "regime_xgb":
                xgb_regime_test_pred
        }
    )

    test_predictions.append(
        prediction_block
    )


    # ========================================================
    # PRINT VALIDATION RESULTS
    # ========================================================

    print(
        f"Persistence:"
        f"  RMSE={p_val_metrics['RMSE']:.6f}"
    )

    print(
        f"Baseline Ridge:"
        f"  RMSE={base_val_metrics['RMSE']:.6f}"
    )

    print(
        f"Regime Ridge:"
        f"  RMSE={regime_val_metrics['RMSE']:.6f}"
    )

    print(
        f"Baseline XGBoost:"
        f"  RMSE={xgb_base_val_metrics['RMSE']:.6f}"
    )

    print(
        f"Regime XGBoost:"
        f"  RMSE={xgb_regime_val_metrics['RMSE']:.6f}"
    )


# ============================================================
# 16. RESULTS TABLE
# ============================================================

print_section(
    "MODEL COMPARISON"
)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    [
        "sensor",
        "validation_RMSE"
    ]
)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step16_regime_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 17. SELECT BEST MODEL USING VALIDATION ONLY
# ============================================================

print_section(
    "BEST MODEL SELECTION"
)

best_rows = []

for sensor in TARGETS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()

    best_idx = (
        sensor_results[
            "validation_RMSE"
        ].idxmin()
    )

    best = sensor_results.loc[
        best_idx
    ]

    best_rows.append(
        {
            "sensor": sensor,
            "best_feature_set":
                best["feature_set"],
            "best_model":
                best["model"],
            "validation_RMSE":
                best["validation_RMSE"],
            "validation_MAE":
                best["validation_MAE"],
            "validation_R2":
                best["validation_R2"],
            "test_RMSE":
                best["test_RMSE"],
            "test_MAE":
                best["test_MAE"],
            "test_R2":
                best["test_R2"]
        }
    )

best_models_df = pd.DataFrame(
    best_rows
)

print(
    best_models_df.to_string(
        index=False
    )
)

best_models_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step16_best_regime_models.csv"
    ),
    index=False
)


# ============================================================
# 18. CALCULATE REGIME IMPROVEMENT
# ============================================================

print_section(
    "IMPACT OF REGIME FEATURES"
)

improvement_rows = []

for sensor in TARGETS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ]

    baseline_ridge_row = sensor_results[
        (
            sensor_results["feature_set"]
            == "Baseline"
        )
        &
        (
            sensor_results["model"]
            == "Ridge"
        )
    ]

    regime_ridge_row = sensor_results[
        (
            sensor_results["feature_set"]
            == "RegimeAware"
        )
        &
        (
            sensor_results["model"]
            == "Ridge"
        )
    ]

    baseline_xgb_row = sensor_results[
        (
            sensor_results["feature_set"]
            == "Baseline"
        )
        &
        (
            sensor_results["model"]
            == "XGBoost"
        )
    ]

    regime_xgb_row = sensor_results[
        (
            sensor_results["feature_set"]
            == "RegimeAware"
        )
        &
        (
            sensor_results["model"]
            == "XGBoost"
        )
    ]

    base_ridge_rmse = (
        baseline_ridge_row[
            "validation_RMSE"
        ].iloc[0]
    )

    regime_ridge_rmse = (
        regime_ridge_row[
            "validation_RMSE"
        ].iloc[0]
    )

    base_xgb_rmse = (
        baseline_xgb_row[
            "validation_RMSE"
        ].iloc[0]
    )

    regime_xgb_rmse = (
        regime_xgb_row[
            "validation_RMSE"
        ].iloc[0]
    )

    ridge_improvement = (
        (
            base_ridge_rmse
            -
            regime_ridge_rmse
        )
        /
        base_ridge_rmse
        * 100
    )

    xgb_improvement = (
        (
            base_xgb_rmse
            -
            regime_xgb_rmse
        )
        /
        base_xgb_rmse
        * 100
    )

    improvement_rows.append(
        {
            "sensor": sensor,
            "ridge_baseline_RMSE":
                base_ridge_rmse,
            "ridge_regime_RMSE":
                regime_ridge_rmse,
            "ridge_improvement_percent":
                ridge_improvement,
            "xgb_baseline_RMSE":
                base_xgb_rmse,
            "xgb_regime_RMSE":
                regime_xgb_rmse,
            "xgb_improvement_percent":
                xgb_improvement
        }
    )

improvement_df = pd.DataFrame(
    improvement_rows
)

print(
    improvement_df.to_string(
        index=False
    )
)

improvement_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step16_regime_feature_impact.csv"
    ),
    index=False
)


# ============================================================
# 19. CREATE ACTUAL VS PREDICTED PLOTS
# ============================================================

print_section(
    "CREATING TEST PREDICTION PLOTS"
)

predictions_df = pd.concat(
    test_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step16_test_predictions.csv"
    ),
    index=False
)


for sensor in TARGETS:

    sensor_pred = predictions_df[
        predictions_df["sensor"] == sensor
    ].copy()

    # Plot only a manageable section so the graph
    # remains readable.
    plot_data = sensor_pred.head(1500)

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        plot_data[TIMESTAMP_COL],
        plot_data["actual"],
        label="Actual",
        linewidth=1.5
    )

    plt.plot(
        plot_data[TIMESTAMP_COL],
        plot_data["regime_xgb"],
        label="Regime XGBoost",
        linewidth=1
    )

    plt.plot(
        plot_data[TIMESTAMP_COL],
        plot_data["regime_ridge"],
        label="Regime Ridge",
        linewidth=1
    )

    plt.xlabel("Time")
    plt.ylabel(sensor)

    plt.title(
        f"{sensor}: 60-Second Forecast "
        "with Regime-Aware Features"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"step16_regime_forecast_{safe_filename(sensor)}.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 20. SAVE CONFIGURATION
# ============================================================

config = {
    "forecast_horizon_seconds":
        FORECAST_SECONDS,

    "input_rows":
        int(len(df_model)),

    "train_rows":
        int(len(train_df)),

    "validation_rows":
        int(len(val_df)),

    "test_rows":
        int(len(test_df)),

    "baseline_feature_count":
        len(baseline_features),

    "regime_aware_feature_count":
        len(regime_features_final),

    "baseline_features":
        baseline_features,

    "regime_specific_features":
        regime_specific_features,

    "leakage_check":
        "PASS",

    "selection_rule":
        "Lowest validation RMSE",

    "test_set_used_for_selection":
        False
}

with open(
    os.path.join(
        REPORT_DIR,
        "step16_config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print_section(
    "STEP 16 COMPLETED"
)

print(
    "Regime-aware forecasting experiment completed."
)

print(
    "\nImportant:"
)

print(
    "This experiment predicts sensor values "
    "60 seconds into the future."
)

print(
    "Features only use information available "
    "at prediction time."
)

print(
    "No future target leakage detected."
)

print(
    "\nReports:"
)

print(
    "  outputs/reports/"
    "step16_regime_model_comparison.csv"
)

print(
    "  outputs/reports/"
    "step16_best_regime_models.csv"
)

print(
    "  outputs/reports/"
    "step16_regime_feature_impact.csv"
)

print(
    "  outputs/reports/"
    "step16_test_predictions.csv"
)

print(
    "  outputs/reports/"
    "step16_config.json"
)

print(
    "\nPlots:"
)

print(
    "  outputs/plots/"
    "step16_regime_forecast_*.png"
)

print(
    "\nDO NOT train another model yet."
)

print(
    "First inspect the Step 16 results."
)