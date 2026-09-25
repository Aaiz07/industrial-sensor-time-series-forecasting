"""
18_robust_regime_evaluation.py

STEP 18 - ROBUST REGIME-BASED EVALUATION

Purpose:
    Evaluate 60-second forecasting performance across
    different operating regimes.

Inputs:
    1. outputs/data/sensor_data_cleaned.csv
    2. outputs/data/future_features_60s.csv

Models:
    - Persistence
    - Ridge
    - XGBoost

Regimes:
    - Low / Medium / High vibration
    - Low / Medium / High temperature
    - Combined operating regime

Important:
    - Exact 60-second future targets are used.
    - Future targets are never model inputs.
    - Current raw sensor values are used only to define regimes.
    - Current raw sensor values are NOT model features.
    - Chronological train/validation/test split is used.
"""

import os
import warnings

import numpy as np
import pandas as pd

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
# STEP 1: PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CLEANED_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "future_features_60s.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

TIMESTAMP_COL = "ts"

HORIZON_SECONDS = 60

SENSOR_COLS = [

    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"

]


# Future target mapping from Step 12

FUTURE_TARGET_MAP = {

    "vib_x_rms":
        "future_vib_x_rms",

    "vib_y_rms":
        "future_vib_y_rms",

    "vib_z_rms":
        "future_vib_z_rms",

    "mag_x":
        "future_mag_x",

    "mag_y":
        "future_mag_y",

    "mag_z":
        "future_mag_z",

    "temperature":
        "future_temperature"

}


# ============================================================
# STEP 3: METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )

    y_true = y_true[valid]

    y_pred = y_pred[valid]

    if len(y_true) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    if len(np.unique(y_true)) > 1:

        r2 = r2_score(
            y_true,
            y_pred
        )

    else:

        r2 = np.nan

    return (
        mae,
        rmse,
        r2
    )


# ============================================================
# START
# ============================================================

print("=" * 80)

print(
    "STEP 18: ROBUST REGIME-BASED EVALUATION"
)

print("=" * 80)


# ============================================================
# STEP 4: CHECK FILES
# ============================================================

if not os.path.exists(
    CLEANED_FILE
):

    raise FileNotFoundError(

        f"\nCleaned sensor file not found:\n"
        f"{CLEANED_FILE}\n\n"
        f"Run Step 1 first."

    )


if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(

        f"\nFuture feature file not found:\n"
        f"{INPUT_FILE}\n\n"
        f"Run Step 12 first."

    )


# ============================================================
# STEP 5: LOAD FUTURE FEATURE DATA
# ============================================================

print(
    "\nLoading Step 12 future-target dataset..."
)


future_df = pd.read_csv(
    INPUT_FILE
)


future_df[TIMESTAMP_COL] = pd.to_datetime(
    future_df[TIMESTAMP_COL],
    errors="coerce"
)


future_df = (

    future_df

    .dropna(
        subset=[TIMESTAMP_COL]
    )

    .sort_values(
        TIMESTAMP_COL
    )

    .reset_index(
        drop=True
    )

)


print(
    f"Future-target rows : "
    f"{len(future_df):,}"
)


# ============================================================
# STEP 6: LOAD CLEANED RAW SENSOR DATA
# ============================================================

print(
    "\nLoading cleaned sensor data..."
)


raw_df = pd.read_csv(
    CLEANED_FILE
)


raw_df[TIMESTAMP_COL] = pd.to_datetime(
    raw_df[TIMESTAMP_COL],
    errors="coerce"
)


raw_df = (

    raw_df

    .dropna(
        subset=[TIMESTAMP_COL]
    )

    .sort_values(
        TIMESTAMP_COL
    )

    .reset_index(
        drop=True
    )

)


print(
    f"Cleaned sensor rows : "
    f"{len(raw_df):,}"
)


# ============================================================
# STEP 7: CHECK RAW SENSOR COLUMNS
# ============================================================

missing_raw = [

    col

    for col in SENSOR_COLS

    if col not in raw_df.columns

]


if missing_raw:

    raise ValueError(

        "\nMissing raw sensor columns:\n"
        +
        "\n".join(
            missing_raw
        )

    )


# ============================================================
# STEP 8: CHECK FUTURE TARGET COLUMNS
# ============================================================

print(
    "\nChecking future target columns..."
)


missing_targets = []


for sensor, target in FUTURE_TARGET_MAP.items():

    if target not in future_df.columns:

        missing_targets.append(
            target
        )


if missing_targets:

    raise ValueError(

        "\nMissing future target columns:\n"
        +
        "\n".join(
            missing_targets
        )

    )


print(
    "All 7 future target columns found."
)


print(
    "\nTarget mapping:"
)


for sensor, target in FUTURE_TARGET_MAP.items():

    print(
        f"{sensor:15s} -> {target}"
    )


# ============================================================
# STEP 9: MERGE CURRENT SENSOR VALUES
# ============================================================

print(
    "\nMerging current sensor values "
    "with future-target dataset..."
)


current_sensor_df = raw_df[
    ["ts"] + SENSOR_COLS
].copy()


df = pd.merge(

    future_df,

    current_sensor_df,

    on=TIMESTAMP_COL,

    how="inner"

)


df = (

    df

    .sort_values(
        TIMESTAMP_COL
    )

    .reset_index(
        drop=True
    )

)


print(
    f"Merged rows : {len(df):,}"
)


coverage = (

    len(df)

    /

    len(future_df)

    *

    100

)


print(
    f"Merge coverage : "
    f"{coverage:.2f}%"
)


if len(df) == 0:

    raise ValueError(

        "\nNo timestamps matched between "
        "future_features_60s.csv and "
        "sensor_data_cleaned.csv."

    )


# ============================================================
# STEP 10: HORIZON VALIDATION
# ============================================================

print(
    "\nChecking actual forecast horizon..."
)


if "actual_horizon_seconds" not in df.columns:

    raise ValueError(

        "\nactual_horizon_seconds column "
        "was not found."

    )


horizon_values = (

    pd.to_numeric(

        df[
            "actual_horizon_seconds"
        ],

        errors="coerce"

    )

    .dropna()

)


print(
    "\nActual horizon statistics:"
)

print(
    horizon_values.describe()
)


unique_horizons = (
    horizon_values
    .unique()
)


if not np.allclose(
    unique_horizons,
    HORIZON_SECONDS
):

    raise ValueError(

        "\nUnexpected forecast horizon detected:\n"
        f"{unique_horizons}"

    )


print(
    f"\nPASS - All targets are exactly "
    f"{HORIZON_SECONDS} seconds ahead."
)


# ============================================================
# STEP 11: IDENTIFY MODEL FEATURES
# ============================================================

print(
    "\nIdentifying model features..."
)


EXCLUDED_COLUMNS = {

    TIMESTAMP_COL,

    "target_ts",

    "actual_horizon_seconds"

}


# Current raw sensor values must NOT be
# used as model inputs.

for sensor in SENSOR_COLS:

    EXCLUDED_COLUMNS.add(
        sensor
    )


# Future targets must NOT be model inputs.

for target in FUTURE_TARGET_MAP.values():

    EXCLUDED_COLUMNS.add(
        target
    )


MODEL_FEATURES = [

    col

    for col in df.columns

    if col not in EXCLUDED_COLUMNS

]


print(
    f"Total model features: "
    f"{len(MODEL_FEATURES)}"
)


# ============================================================
# STEP 12: LEAKAGE CHECK
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "LEAKAGE CHECK"
)

print(
    "=" * 80
)


leakage_features = []


for col in MODEL_FEATURES:

    col_lower = col.lower()

    if (

        "future_" in col_lower

        or

        "target_" in col_lower

        or

        "t_plus" in col_lower

    ):

        leakage_features.append(
            col
        )


if leakage_features:

    print(
        "WARNING: Possible leakage features:"
    )

    for col in leakage_features:

        print(
            f"  {col}"
        )


    raise ValueError(

        "\nFuture/target columns detected "
        "inside model features."

    )


print(
    "PASS - No future target columns "
    "are being used as features."
)


# ============================================================
# STEP 13: CONVERT FEATURES TO NUMERIC
# ============================================================

for col in MODEL_FEATURES:

    df[col] = pd.to_numeric(

        df[col],

        errors="coerce"

    )


# Convert targets

for target in FUTURE_TARGET_MAP.values():

    df[target] = pd.to_numeric(

        df[target],

        errors="coerce"

    )


# Convert current sensor values

for sensor in SENSOR_COLS:

    df[sensor] = pd.to_numeric(

        df[sensor],

        errors="coerce"

    )


# ============================================================
# STEP 14: REMOVE MISSING VALUES
# ============================================================

required_columns = (

    MODEL_FEATURES

    +

    list(
        FUTURE_TARGET_MAP.values()
    )

    +

    SENSOR_COLS

)


df = (

    df

    .dropna(
        subset=required_columns
    )

    .reset_index(
        drop=True
    )

)


print(
    "\nRows after removing missing "
    "feature/target/current values: "
    f"{len(df):,}"
)


# ============================================================
# STEP 15: CREATE VIBRATION MAGNITUDE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CREATING OPERATING REGIMES"
)

print(
    "=" * 80
)


df[
    "vibration_magnitude"
] = np.sqrt(

    df["vib_x_rms"] ** 2

    +

    df["vib_y_rms"] ** 2

    +

    df["vib_z_rms"] ** 2

)


# ============================================================
# STEP 16: VIBRATION REGIME
# ============================================================

vib_q33 = df[
    "vibration_magnitude"
].quantile(
    0.33
)


vib_q66 = df[
    "vibration_magnitude"
].quantile(
    0.66
)


def get_vibration_regime(
    value
):

    if value <= vib_q33:

        return "Low_Vibration"

    elif value <= vib_q66:

        return "Medium_Vibration"

    else:

        return "High_Vibration"


df[
    "vibration_regime"
] = (

    df[
        "vibration_magnitude"
    ]

    .apply(
        get_vibration_regime
    )

)


# ============================================================
# STEP 17: TEMPERATURE REGIME
# ============================================================

temp_q33 = df[
    "temperature"
].quantile(
    0.33
)


temp_q66 = df[
    "temperature"
].quantile(
    0.66
)


def get_temperature_regime(
    value
):

    if value <= temp_q33:

        return "Low_Temperature"

    elif value <= temp_q66:

        return "Medium_Temperature"

    else:

        return "High_Temperature"


df[
    "temperature_regime"
] = (

    df[
        "temperature"
    ]

    .apply(
        get_temperature_regime
    )

)


# ============================================================
# STEP 18: COMBINED OPERATING REGIME
# ============================================================

df[
    "operating_regime"
] = (

    df[
        "vibration_regime"
    ]

    +

    "__"

    +

    df[
        "temperature_regime"
    ]

)


print(
    "\nVibration regime thresholds:"
)

print(
    f"33rd percentile : "
    f"{vib_q33:.6f}"
)

print(
    f"66th percentile : "
    f"{vib_q66:.6f}"
)


print(
    "\nTemperature regime thresholds:"
)

print(
    f"33rd percentile : "
    f"{temp_q33:.6f}"
)

print(
    f"66th percentile : "
    f"{temp_q66:.6f}"
)


print(
    "\nVibration regimes:"
)

print(
    df[
        "vibration_regime"
    ].value_counts()
)


print(
    "\nTemperature regimes:"
)

print(
    df[
        "temperature_regime"
    ].value_counts()
)


# ============================================================
# STEP 19: CHRONOLOGICAL SPLIT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT"
)

print(
    "=" * 80
)


n = len(df)


train_end = int(
    n * 0.70
)


val_end = int(
    n * 0.85
)


train_df = df.iloc[
    :train_end
].copy()


val_df = df.iloc[
    train_end:val_end
].copy()


test_df = df.iloc[
    val_end:
].copy()


print(
    f"Train      : {len(train_df):,}"
)

print(
    f"Validation : {len(val_df):,}"
)

print(
    f"Test       : {len(test_df):,}"
)


print(
    "\nTrain period:"
)

print(
    f"{train_df[TIMESTAMP_COL].min()}"
)

print(
    f"to"
)

print(
    f"{train_df[TIMESTAMP_COL].max()}"
)


print(
    "\nValidation period:"
)

print(
    f"{val_df[TIMESTAMP_COL].min()}"
)

print(
    f"to"
)

print(
    f"{val_df[TIMESTAMP_COL].max()}"
)


print(
    "\nTest period:"
)

print(
    f"{test_df[TIMESTAMP_COL].min()}"
)

print(
    f"to"
)

print(
    f"{test_df[TIMESTAMP_COL].max()}"
)


# ============================================================
# STEP 20: TRAIN MODELS SENSOR BY SENSOR
# ============================================================

results = []


for sensor in SENSOR_COLS:

    print(
        "\n" + "-" * 80
    )

    print(
        f"SENSOR: {sensor}"
    )

    print(
        "-" * 80
    )


    target_column = (
        FUTURE_TARGET_MAP[
            sensor
        ]
    )


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    X_train = train_df[
        MODEL_FEATURES
    ]

    X_val = val_df[
        MODEL_FEATURES
    ]

    X_test = test_df[
        MODEL_FEATURES
    ]


    y_train = train_df[
        target_column
    ]

    y_val = val_df[
        target_column
    ]

    y_test = test_df[
        target_column
    ]


    # ========================================================
    # PERSISTENCE
    # ========================================================

    persistence_test = (

        test_df[
            sensor
        ]

        .values

    )


    # ========================================================
    # RIDGE
    # ========================================================

    print(
        "Training Ridge..."
    )


    scaler = StandardScaler()


    X_train_scaled = (

        scaler

        .fit_transform(
            X_train
        )

    )


    X_val_scaled = (

        scaler

        .transform(
            X_val
        )

    )


    X_test_scaled = (

        scaler

        .transform(
            X_test
        )

    )


    ridge = Ridge(
        alpha=10.0
    )


    ridge.fit(

        X_train_scaled,

        y_train

    )


    ridge_test = ridge.predict(
        X_test_scaled
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    print(
        "Training XGBoost..."
    )


    xgb = XGBRegressor(

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


    xgb.fit(

        X_train,

        y_train,

        verbose=False

    )


    xgb_test = xgb.predict(
        X_test
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = {

        "Persistence":
            persistence_test,

        "Ridge":
            ridge_test,

        "XGBoost":
            xgb_test

    }


    # ========================================================
    # REGIME EVALUATION
    # ========================================================

    regime_types = [

        "vibration_regime",

        "temperature_regime",

        "operating_regime"

    ]


    for regime_type in regime_types:

        print(
            f"\nEvaluating "
            f"{regime_type}..."
        )


        regimes = (

            test_df[
                regime_type
            ]

            .dropna()

            .unique()

        )


        for regime in regimes:

            mask = (

                test_df[
                    regime_type
                ]

                .values

                ==

                regime

            )


            sample_count = int(
                mask.sum()
            )


            # Ignore extremely small groups.

            if sample_count < 20:

                continue


            actual = (

                y_test

                .values[
                    mask
                ]

            )


            for model_name, prediction in predictions.items():

                pred = (

                    prediction[
                        mask
                    ]

                )


                mae, rmse, r2 = (

                    calculate_metrics(

                        actual,

                        pred

                    )

                )


                results.append({

                    "sensor":
                        sensor,

                    "horizon_seconds":
                        HORIZON_SECONDS,

                    "regime_type":
                        regime_type,

                    "regime":
                        regime,

                    "model":
                        model_name,

                    "samples":
                        sample_count,

                    "MAE":
                        mae,

                    "RMSE":
                        rmse,

                    "R2":
                        r2

                })


# ============================================================
# STEP 21: CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise ValueError(
        "\nNo regime evaluation results "
        "were generated."
    )


# ============================================================
# STEP 22: SAVE DETAILED RESULTS
# ============================================================

results_file = os.path.join(

    REPORT_DIR,

    "step18_regime_evaluation.csv"

)


results_df.to_csv(

    results_file,

    index=False

)


# ============================================================
# STEP 23: REGIME SUMMARY
# ============================================================

summary_df = (

    results_df

    .groupby(

        [

            "sensor",

            "horizon_seconds",

            "regime_type",

            "model"

        ],

        as_index=False

    )

    .agg(

        mean_MAE=(
            "MAE",
            "mean"
        ),

        mean_RMSE=(
            "RMSE",
            "mean"
        ),

        mean_R2=(
            "R2",
            "mean"
        ),

        regimes_evaluated=(
            "regime",
            "nunique"
        ),

        total_samples=(
            "samples",
            "sum"
        )

    )

)


summary_file = os.path.join(

    REPORT_DIR,

    "step18_regime_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


# ============================================================
# STEP 24: BEST MODEL BY REGIME
# ============================================================

best_model_df = (

    results_df

    .sort_values(

        [

            "sensor",

            "regime_type",

            "RMSE"

        ]

    )

    .groupby(

        [

            "sensor",

            "regime_type"

        ],

        as_index=False

    )

    .first()

)


best_model_file = os.path.join(

    REPORT_DIR,

    "step18_best_model_by_regime.csv"

)


best_model_df.to_csv(

    best_model_file,

    index=False

)


# ============================================================
# STEP 25: OVERALL MODEL SUMMARY
# ============================================================

overall_summary = (

    results_df

    .groupby(

        [

            "sensor",

            "model"

        ],

        as_index=False

    )

    .agg(

        average_MAE=(
            "MAE",
            "mean"
        ),

        average_RMSE=(
            "RMSE",
            "mean"
        ),

        average_R2=(
            "R2",
            "mean"
        )

    )

    .sort_values(

        [

            "sensor",

            "average_RMSE"

        ]

    )

)


overall_file = os.path.join(

    REPORT_DIR,

    "step18_overall_model_summary.csv"

)


overall_summary.to_csv(

    overall_file,

    index=False

)


# ============================================================
# STEP 26: PRINT DETAILED RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 18 RESULTS"
)

print(
    "=" * 80
)


display_columns = [

    "sensor",

    "regime_type",

    "regime",

    "model",

    "samples",

    "MAE",

    "RMSE",

    "R2"

]


print(

    results_df[

        display_columns

    ]

    .sort_values(

        [

            "sensor",

            "regime_type",

            "RMSE"

        ]

    )

    .to_string(
        index=False
    )

)


# ============================================================
# STEP 27: PRINT OVERALL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "AVERAGE PERFORMANCE ACROSS REGIMES"
)

print(
    "=" * 80
)


print(

    overall_summary.to_string(
        index=False
    )

)


# ============================================================
# STEP 28: PRINT BEST MODELS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "BEST MODEL BY SENSOR AND REGIME TYPE"
)

print(
    "=" * 80
)


best_display = [

    "sensor",

    "regime_type",

    "regime",

    "model",

    "RMSE",

    "R2",

    "samples"

]


print(

    best_model_df[
        best_display
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# STEP 29: FINISH
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 18 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 80
)


print(
    "\nGenerated files:"
)

print(
    f"  {results_file}"
)

print(
    f"  {summary_file}"
)

print(
    f"  {best_model_file}"
)

print(
    f"  {overall_file}"
)


print(
    "\nStep 18 answered:"
)

print(
    "Does forecasting performance remain "
    "consistent across different operating regimes?"
)


print(
    "\nNext:"
)

print(
    "STEP 19 - Improve Features / "
    "Forecasting Strategy"
)