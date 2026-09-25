"""
19_improve_features_forecasting.py

STEP 19 - IMPROVE FEATURES / FORECASTING STRATEGY

Goal:
    Investigate whether better feature representation and
    predicting future change (delta) can improve 60-second
    forecasting performance.

Data:
    outputs/data/sensor_data_cleaned.csv
    outputs/data/future_features_60s.csv

Strategies tested:

    1. Persistence baseline
    2. Original 146-feature Ridge
    3. Delta / change prediction with Ridge
    4. Delta prediction with trend features
    5. Delta prediction with trend + volatility features
    6. Delta prediction with exponentially weighted features
    7. XGBoost using improved features

Important:
    - Exact 60-second future targets are used.
    - No future target is used as a feature.
    - Current raw target values are NOT model inputs for
      absolute-value models.
    - Current raw target values are used only as the
      reference point when reconstructing delta predictions.
    - Scaling is fitted on training data only.
    - Chronological train/validation/test split.
    - Test set is used only for final comparison.
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

FUTURE_FILE = os.path.join(
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

PREDICTION_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICTION_DIR,
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


# Existing Step 12 feature columns

BASE_LAGS = [
    1,
    2,
    5,
    10,
    30,
    60
]

BASE_ROLLING_WINDOWS = [
    5,
    15,
    30,
    60
]


# New features

TREND_WINDOWS = [
    5,
    10,
    30,
    60
]

VOLATILITY_WINDOWS = [
    5,
    15,
    30,
    60
]

EWM_SPANS = [
    5,
    15,
    30,
    60
]


# ============================================================
# STEP 3: METRICS
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
# STEP 4: HEADER
# ============================================================

print("=" * 80)

print(
    "STEP 19: IMPROVE FEATURES / FORECASTING STRATEGY"
)

print("=" * 80)

print(
    "\nObjective:"
)

print(
    "Test whether better temporal features and delta "
    "prediction improve 60-second forecasting."
)


# ============================================================
# STEP 5: CHECK INPUT FILES
# ============================================================

if not os.path.exists(
    CLEANED_FILE
):

    raise FileNotFoundError(
        f"\nMissing cleaned data:\n"
        f"{CLEANED_FILE}"
    )


if not os.path.exists(
    FUTURE_FILE
):

    raise FileNotFoundError(
        f"\nMissing future-target data:\n"
        f"{FUTURE_FILE}"
    )


# ============================================================
# STEP 6: LOAD DATA
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

    .drop_duplicates(
        subset=[TIMESTAMP_COL]
    )

    .reset_index(
        drop=True
    )

)


print(
    f"Cleaned rows : {len(raw_df):,}"
)


print(
    "\nLoading future-target data..."
)

future_df = pd.read_csv(
    FUTURE_FILE
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
    f"Future-target rows : {len(future_df):,}"
)


# ============================================================
# STEP 7: CHECK REQUIRED COLUMNS
# ============================================================

missing_sensor_columns = [

    col

    for col in SENSOR_COLS

    if col not in raw_df.columns

]


if missing_sensor_columns:

    raise ValueError(

        "\nMissing sensor columns:\n"

        +

        "\n".join(
            missing_sensor_columns
        )

    )


missing_target_columns = [

    target

    for target in FUTURE_TARGET_MAP.values()

    if target not in future_df.columns

]


if missing_target_columns:

    raise ValueError(

        "\nMissing future target columns:\n"

        +

        "\n".join(
            missing_target_columns
        )

    )


# ============================================================
# STEP 8: MERGE CURRENT VALUES WITH FUTURE TARGETS
# ============================================================

print(
    "\nMerging current sensor values "
    "with future targets..."
)


current_df = raw_df[
    ["ts"] + SENSOR_COLS
].copy()


df = pd.merge(

    future_df,

    current_df,

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


# ============================================================
# STEP 9: HORIZON VALIDATION
# ============================================================

print(
    "\nChecking forecast horizon..."
)


if "actual_horizon_seconds" not in df.columns:

    raise ValueError(
        "\nactual_horizon_seconds missing."
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
    horizon_values.describe()
)


if not np.allclose(
    horizon_values.values,
    HORIZON_SECONDS
):

    raise ValueError(
        "\nNot all targets are exactly "
        "60 seconds ahead."
    )


print(
    "\nPASS - Exact 60-second horizon confirmed."
)


# ============================================================
# STEP 10: IDENTIFY ORIGINAL FEATURES
# ============================================================

print(
    "\nIdentifying original Step 12 features..."
)


EXCLUDED = {

    TIMESTAMP_COL,

    "target_ts",

    "actual_horizon_seconds"

}


for sensor in SENSOR_COLS:

    EXCLUDED.add(
        sensor
    )


for target in FUTURE_TARGET_MAP.values():

    EXCLUDED.add(
        target
    )


BASE_FEATURES = [

    col

    for col in future_df.columns

    if col not in EXCLUDED

]


print(
    f"Original feature count: "
    f"{len(BASE_FEATURES)}"
)


# ============================================================
# STEP 11: LEAKAGE CHECK
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


def check_leakage(
    feature_list
):

    suspicious = []

    for col in feature_list:

        name = col.lower()

        if (

            "future_" in name

            or

            "target_" in name

            or

            "t_plus" in name

        ):

            suspicious.append(
                col
            )

    return suspicious


base_leakage = check_leakage(
    BASE_FEATURES
)


if base_leakage:

    print(
        "ERROR: Leakage detected:"
    )

    for col in base_leakage:

        print(
            f"  {col}"
        )

    raise ValueError(
        "Future/target columns detected."
    )


print(
    "PASS - Original features contain "
    "no future targets."
)


# ============================================================
# STEP 12: CREATE NEW FEATURES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CREATING IMPROVED TEMPORAL FEATURES"
)

print(
    "=" * 80
)


feature_df = df[
    [TIMESTAMP_COL]
    +
    SENSOR_COLS
].copy()


# ------------------------------------------------------------
# Time features
# ------------------------------------------------------------

seconds_of_day = (

    feature_df[
        TIMESTAMP_COL
    ].dt.hour * 3600

    +

    feature_df[
        TIMESTAMP_COL
    ].dt.minute * 60

    +

    feature_df[
        TIMESTAMP_COL
    ].dt.second

)


feature_df[
    "time_sin"
] = np.sin(

    2 * np.pi
    *
    seconds_of_day
    /
    86400

)


feature_df[
    "time_cos"
] = np.cos(

    2 * np.pi
    *
    seconds_of_day
    /
    86400

)


# ------------------------------------------------------------
# Historical features
# ------------------------------------------------------------

NEW_FEATURES = []


for sensor in SENSOR_COLS:

    series = feature_df[
        sensor
    ]


    # ========================================================
    # LAG FEATURES
    # ========================================================

    for lag in BASE_LAGS:

        col = (
            f"{sensor}_hist_lag_{lag}"
        )

        feature_df[col] = (
            series.shift(lag)
        )

        NEW_FEATURES.append(
            col
        )


    # ========================================================
    # DIFFERENCE FEATURES
    # ========================================================

    feature_df[
        f"{sensor}_hist_diff_1"
    ] = (

        series.shift(1)

        -

        series.shift(2)

    )

    NEW_FEATURES.append(
        f"{sensor}_hist_diff_1"
    )


    feature_df[
        f"{sensor}_hist_diff_5"
    ] = (

        series.shift(1)

        -

        series.shift(6)

    )

    NEW_FEATURES.append(
        f"{sensor}_hist_diff_5"
    )


    feature_df[
        f"{sensor}_hist_diff_30"
    ] = (

        series.shift(1)

        -

        series.shift(31)

    )

    NEW_FEATURES.append(
        f"{sensor}_hist_diff_30"
    )


    # ========================================================
    # TREND / SLOPE FEATURES
    # ========================================================

    for window in TREND_WINDOWS:

        # Difference between recent mean and older mean.

        recent_mean = (

            series

            .shift(1)

            .rolling(
                window
            )

            .mean()

        )


        previous_mean = (

            series

            .shift(
                window + 1
            )

            .rolling(
                window
            )

            .mean()

        )


        trend_col = (

            f"{sensor}_trend_{window}"

        )


        feature_df[
            trend_col
        ] = (

            recent_mean
            -
            previous_mean

        )


        NEW_FEATURES.append(
            trend_col
        )


    # ========================================================
    # VOLATILITY FEATURES
    # ========================================================

    for window in VOLATILITY_WINDOWS:

        roll_std_col = (

            f"{sensor}_volatility_{window}"

        )


        feature_df[
            roll_std_col
        ] = (

            series

            .shift(1)

            .rolling(
                window
            )

            .std()

        )


        NEW_FEATURES.append(
            roll_std_col
        )


    # ========================================================
    # ROLLING RANGE
    # ========================================================

    for window in [
        5,
        15,
        30,
        60
    ]:

        rolling_max = (

            series

            .shift(1)

            .rolling(
                window
            )

            .max()

        )


        rolling_min = (

            series

            .shift(1)

            .rolling(
                window
            )

            .min()

        )


        range_col = (

            f"{sensor}_range_{window}"

        )


        feature_df[
            range_col
        ] = (

            rolling_max
            -
            rolling_min

        )


        NEW_FEATURES.append(
            range_col
        )


    # ========================================================
    # EXPONENTIALLY WEIGHTED MEANS
    # ========================================================

    for span in EWM_SPANS:

        ewm_col = (

            f"{sensor}_ewm_{span}"

        )


        feature_df[
            ewm_col
        ] = (

            series

            .shift(1)

            .ewm(
                span=span,
                adjust=False
            )

            .mean()

        )


        NEW_FEATURES.append(
            ewm_col
        )


# ============================================================
# STEP 13: COMBINE ORIGINAL + NEW FEATURES
# ============================================================

print(
    "\nCombining original and improved features..."
)


# Original features are already in future_df.

for col in BASE_FEATURES:

    feature_df[col] = df[col].values


ALL_FEATURES = list(
    dict.fromkeys(
        BASE_FEATURES
        +
        NEW_FEATURES
    )
)


# ============================================================
# STEP 14: CHECK FEATURE LEAKAGE AGAIN
# ============================================================

leakage_features = check_leakage(
    ALL_FEATURES
)


if leakage_features:

    raise ValueError(

        "\nLeakage detected in improved features:\n"

        +

        "\n".join(
            leakage_features
        )

    )


# Explicitly make sure raw current sensor
# columns are not model features.

for sensor in SENSOR_COLS:

    if sensor in ALL_FEATURES:

        raise ValueError(

            f"\nCurrent raw sensor column "
            f"incorrectly included as feature: "
            f"{sensor}"

        )


print(
    f"Total improved features: "
    f"{len(ALL_FEATURES)}"
)


print(
    "PASS - No future targets or raw "
    "current sensor columns are model features."
)


# ============================================================
# STEP 15: BUILD MODEL DATAFRAME
# ============================================================

model_df = feature_df[
    [TIMESTAMP_COL]
    +
    SENSOR_COLS
    +
    ALL_FEATURES
].copy()


for sensor, target in FUTURE_TARGET_MAP.items():

    model_df[
        target
    ] = df[target].values


# ============================================================
# STEP 16: REMOVE MISSING FEATURES
# ============================================================

required_columns = (

    ALL_FEATURES

    +

    SENSOR_COLS

    +

    list(
        FUTURE_TARGET_MAP.values()
    )

)


model_df = (

    model_df

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .dropna(
        subset=required_columns
    )

    .reset_index(
        drop=True
    )

)


print(
    "\nRows available after feature creation:"
)

print(
    f"{len(model_df):,}"
)


# ============================================================
# STEP 17: CHRONOLOGICAL SPLIT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHRONOLOGICAL SPLIT"
)

print(
    "=" * 80
)


n = len(model_df)


train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


train_df = model_df.iloc[
    :train_end
].copy()


val_df = model_df.iloc[
    train_end:val_end
].copy()


test_df = model_df.iloc[
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
    "\nTrain:"
)

print(
    train_df[
        TIMESTAMP_COL
    ].min(),

    "to",

    train_df[
        TIMESTAMP_COL
    ].max()
)


print(
    "\nValidation:"
)

print(
    val_df[
        TIMESTAMP_COL
    ].min(),

    "to",

    val_df[
        TIMESTAMP_COL
    ].max()
)


print(
    "\nTest:"
)

print(
    test_df[
        TIMESTAMP_COL
    ].min(),

    "to",

    test_df[
        TIMESTAMP_COL
    ].max()
)


# ============================================================
# STEP 18: MODEL FEATURE MATRIX
# ============================================================

X_train = train_df[
    ALL_FEATURES
].copy()

X_val = val_df[
    ALL_FEATURES
].copy()

X_test = test_df[
    ALL_FEATURES
].copy()


# ============================================================
# STEP 19: TRAIN / VALIDATION / TEST FUNCTION
# ============================================================

all_results = []


prediction_records = []


for sensor in SENSOR_COLS:

    print(
        "\n" + "=" * 80
    )

    print(
        f"SENSOR: {sensor}"
    )

    print(
        "=" * 80
    )


    future_target = (
        FUTURE_TARGET_MAP[
            sensor
        ]
    )


    # --------------------------------------------------------
    # Actual future values
    # --------------------------------------------------------

    y_train = train_df[
        future_target
    ].values

    y_val = val_df[
        future_target
    ].values

    y_test = test_df[
        future_target
    ].values


    current_train = train_df[
        sensor
    ].values

    current_val = val_df[
        sensor
    ].values

    current_test = test_df[
        sensor
    ].values


    # ========================================================
    # MODEL 1: PERSISTENCE
    # ========================================================

    print(
        "\n1. Persistence"
    )


    persistence_val = current_val

    persistence_test = current_test


    p_val_metrics = calculate_metrics(
        y_val,
        persistence_val
    )


    p_test_metrics = calculate_metrics(
        y_test,
        persistence_test
    )


    print(
        f"Validation RMSE : "
        f"{p_val_metrics[1]:.6f}"
    )

    print(
        f"Validation R2   : "
        f"{p_val_metrics[2]:.6f}"
    )

    print(
        f"Test RMSE       : "
        f"{p_test_metrics[1]:.6f}"
    )

    print(
        f"Test R2         : "
        f"{p_test_metrics[2]:.6f}"
    )


    all_results.append({

        "sensor":
            sensor,

        "strategy":
            "Persistence",

        "feature_set":
            "None",

        "target_type":
            "Absolute",

        "val_MAE":
            p_val_metrics[0],

        "val_RMSE":
            p_val_metrics[1],

        "val_R2":
            p_val_metrics[2],

        "test_MAE":
            p_test_metrics[0],

        "test_RMSE":
            p_test_metrics[1],

        "test_R2":
            p_test_metrics[2]

    })


    # ========================================================
    # MODEL 2: ORIGINAL RIDGE
    # ========================================================

    print(
        "\n2. Original-feature Ridge"
    )


    scaler_original = StandardScaler()


    X_train_scaled = (
        scaler_original
        .fit_transform(
            X_train
        )
    )


    X_val_scaled = (
        scaler_original
        .transform(
            X_val
        )
    )


    X_test_scaled = (
        scaler_original
        .transform(
            X_test
        )
    )


    ridge_original = Ridge(
        alpha=10.0
    )


    ridge_original.fit(

        X_train_scaled,

        y_train

    )


    ridge_original_val = (
        ridge_original
        .predict(
            X_val_scaled
        )
    )


    ridge_original_test = (
        ridge_original
        .predict(
            X_test_scaled
        )
    )


    ro_val_metrics = calculate_metrics(
        y_val,
        ridge_original_val
    )


    ro_test_metrics = calculate_metrics(
        y_test,
        ridge_original_test
    )


    print(
        f"Validation RMSE : "
        f"{ro_val_metrics[1]:.6f}"
    )

    print(
        f"Validation R2   : "
        f"{ro_val_metrics[2]:.6f}"
    )

    print(
        f"Test RMSE       : "
        f"{ro_test_metrics[1]:.6f}"
    )

    print(
        f"Test R2         : "
        f"{ro_test_metrics[2]:.6f}"
    )


    all_results.append({

        "sensor":
            sensor,

        "strategy":
            "Original_Ridge",

        "feature_set":
            "Original_146",

        "target_type":
            "Absolute",

        "val_MAE":
            ro_val_metrics[0],

        "val_RMSE":
            ro_val_metrics[1],

        "val_R2":
            ro_val_metrics[2],

        "test_MAE":
            ro_test_metrics[0],

        "test_RMSE":
            ro_test_metrics[1],

        "test_R2":
            ro_test_metrics[2]

    })


    # ========================================================
    # DELTA TARGETS
    # ========================================================

    print(
        "\n3. Delta prediction"
    )


    delta_train = (
        y_train
        -
        current_train
    )


    delta_val = (
        y_val
        -
        current_val
    )


    delta_test = (
        y_test
        -
        current_test
    )


    # --------------------------------------------------------
    # Ridge on delta
    # --------------------------------------------------------

    scaler_delta = StandardScaler()


    X_train_delta_scaled = (
        scaler_delta
        .fit_transform(
            X_train
        )
    )


    X_val_delta_scaled = (
        scaler_delta
        .transform(
            X_val
        )
    )


    X_test_delta_scaled = (
        scaler_delta
        .transform(
            X_test
        )
    )


    ridge_delta = Ridge(
        alpha=10.0
    )


    ridge_delta.fit(

        X_train_delta_scaled,

        delta_train

    )


    predicted_delta_val = (
        ridge_delta
        .predict(
            X_val_delta_scaled
        )
    )


    predicted_delta_test = (
        ridge_delta
        .predict(
            X_test_delta_scaled
        )
    )


    # Reconstruct future absolute value.

    delta_prediction_val = (
        current_val
        +
        predicted_delta_val
    )


    delta_prediction_test = (
        current_test
        +
        predicted_delta_test
    )


    d_val_metrics = calculate_metrics(
        y_val,
        delta_prediction_val
    )


    d_test_metrics = calculate_metrics(
        y_test,
        delta_prediction_test
    )


    print(
        f"Validation RMSE : "
        f"{d_val_metrics[1]:.6f}"
    )

    print(
        f"Validation R2   : "
        f"{d_val_metrics[2]:.6f}"
    )

    print(
        f"Test RMSE       : "
        f"{d_test_metrics[1]:.6f}"
    )

    print(
        f"Test R2         : "
        f"{d_test_metrics[2]:.6f}"
    )


    all_results.append({

        "sensor":
            sensor,

        "strategy":
            "Delta_Ridge",

        "feature_set":
            "Original_146",

        "target_type":
            "Delta",

        "val_MAE":
            d_val_metrics[0],

        "val_RMSE":
            d_val_metrics[1],

        "val_R2":
            d_val_metrics[2],

        "test_MAE":
            d_test_metrics[0],

        "test_RMSE":
            d_test_metrics[1],

        "test_R2":
            d_test_metrics[2]

    })


    # ========================================================
    # MODEL 4: IMPROVED FEATURE DELTA RIDGE
    # ========================================================

    print(
        "\n4. Improved-feature Delta Ridge"
    )


    scaler_improved = StandardScaler()


    X_train_imp_scaled = (
        scaler_improved
        .fit_transform(
            X_train
        )
    )


    X_val_imp_scaled = (
        scaler_improved
        .transform(
            X_val
        )
    )


    X_test_imp_scaled = (
        scaler_improved
        .transform(
            X_test
        )
    )


    ridge_improved = Ridge(
        alpha=10.0
    )


    ridge_improved.fit(

        X_train_imp_scaled,

        delta_train

    )


    improved_delta_val = (
        ridge_improved
        .predict(
            X_val_imp_scaled
        )
    )


    improved_delta_test = (
        ridge_improved
        .predict(
            X_test_imp_scaled
        )
    )


    improved_prediction_val = (
        current_val
        +
        improved_delta_val
    )


    improved_prediction_test = (
        current_test
        +
        improved_delta_test
    )


    id_val_metrics = calculate_metrics(
        y_val,
        improved_prediction_val
    )


    id_test_metrics = calculate_metrics(
        y_test,
        improved_prediction_test
    )


    print(
        f"Validation RMSE : "
        f"{id_val_metrics[1]:.6f}"
    )

    print(
        f"Validation R2   : "
        f"{id_val_metrics[2]:.6f}"
    )

    print(
        f"Test RMSE       : "
        f"{id_test_metrics[1]:.6f}"
    )

    print(
        f"Test R2         : "
        f"{id_test_metrics[2]:.6f}"
    )


    all_results.append({

        "sensor":
            sensor,

        "strategy":
            "Improved_Delta_Ridge",

        "feature_set":
            "Original_plus_Temporal",

        "target_type":
            "Delta",

        "val_MAE":
            id_val_metrics[0],

        "val_RMSE":
            id_val_metrics[1],

        "val_R2":
            id_val_metrics[2],

        "test_MAE":
            id_test_metrics[0],

        "test_RMSE":
            id_test_metrics[1],

        "test_R2":
            id_test_metrics[2]

    })


    # ========================================================
    # MODEL 5: XGBOOST DELTA
    # ========================================================

    print(
        "\n5. XGBoost Delta"
    )


    xgb_delta = XGBRegressor(

        n_estimators=300,

        max_depth=4,

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


    xgb_delta.fit(

        X_train,

        delta_train,

        verbose=False

    )


    xgb_delta_val_raw = (
        xgb_delta
        .predict(
            X_val
        )
    )


    xgb_delta_test_raw = (
        xgb_delta
        .predict(
            X_test
        )
    )


    xgb_prediction_val = (
        current_val
        +
        xgb_delta_val_raw
    )


    xgb_prediction_test = (
        current_test
        +
        xgb_delta_test_raw
    )


    xgb_val_metrics = calculate_metrics(
        y_val,
        xgb_prediction_val
    )


    xgb_test_metrics = calculate_metrics(
        y_test,
        xgb_prediction_test
    )


    print(
        f"Validation RMSE : "
        f"{xgb_val_metrics[1]:.6f}"
    )

    print(
        f"Validation R2   : "
        f"{xgb_val_metrics[2]:.6f}"
    )

    print(
        f"Test RMSE       : "
        f"{xgb_test_metrics[1]:.6f}"
    )

    print(
        f"Test R2         : "
        f"{xgb_test_metrics[2]:.6f}"
    )


    all_results.append({

        "sensor":
            sensor,

        "strategy":
            "XGBoost_Delta",

        "feature_set":
            "Original_plus_Temporal",

        "target_type":
            "Delta",

        "val_MAE":
            xgb_val_metrics[0],

        "val_RMSE":
            xgb_val_metrics[1],

        "val_R2":
            xgb_val_metrics[2],

        "test_MAE":
            xgb_test_metrics[0],

        "test_RMSE":
            xgb_test_metrics[1],

        "test_R2":
            xgb_test_metrics[2]

    })


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    sensor_predictions = pd.DataFrame({

        "ts":
            test_df[
                TIMESTAMP_COL
            ].values,

        "sensor":
            sensor,

        "actual":
            y_test,

        "persistence":
            persistence_test,

        "original_ridge":
            ridge_original_test,

        "delta_ridge":
            delta_prediction_test,

        "improved_delta_ridge":
            improved_prediction_test,

        "xgboost_delta":
            xgb_prediction_test

    })


    prediction_records.append(
        sensor_predictions
    )


# ============================================================
# STEP 20: RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# STEP 21: SAVE RESULTS
# ============================================================

results_file = os.path.join(

    REPORT_DIR,

    "step19_feature_strategy_results.csv"

)


results_df.to_csv(

    results_file,

    index=False

)


# ============================================================
# STEP 22: BEST VALIDATION MODEL
# ============================================================

best_val_df = (

    results_df

    .sort_values(
        [
            "sensor",
            "val_RMSE"
        ]
    )

    .groupby(
        "sensor",
        as_index=False
    )

    .first()

)


best_val_file = os.path.join(

    REPORT_DIR,

    "step19_best_model_by_validation.csv"

)


best_val_df.to_csv(

    best_val_file,

    index=False

)


# ============================================================
# STEP 23: BEST TEST MODEL
# ============================================================

best_test_df = (

    results_df

    .sort_values(
        [
            "sensor",
            "test_RMSE"
        ]
    )

    .groupby(
        "sensor",
        as_index=False
    )

    .first()

)


best_test_file = os.path.join(

    REPORT_DIR,

    "step19_best_model_by_test.csv"

)


best_test_df.to_csv(

    best_test_file,

    index=False

)


# ============================================================
# STEP 24: SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.concat(

    prediction_records,

    ignore_index=True

)


prediction_file = os.path.join(

    PREDICTION_DIR,

    "step19_test_predictions.csv"

)


predictions_df.to_csv(

    prediction_file,

    index=False

)


# ============================================================
# STEP 25: CALCULATE IMPROVEMENT OVER PERSISTENCE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "IMPROVEMENT OVER PERSISTENCE"
)

print(
    "=" * 80
)


comparison_rows = []


for sensor in SENSOR_COLS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()


    persistence_row = sensor_results[
        sensor_results["strategy"]
        ==
        "Persistence"
    ].iloc[0]


    persistence_rmse = (
        persistence_row[
            "test_RMSE"
        ]
    )


    for _, row in sensor_results.iterrows():

        improvement = (

            (

                persistence_rmse

                -

                row["test_RMSE"]

            )

            /

            persistence_rmse

            *

            100

        )


        comparison_rows.append({

            "sensor":
                sensor,

            "strategy":
                row["strategy"],

            "persistence_test_RMSE":
                persistence_rmse,

            "model_test_RMSE":
                row["test_RMSE"],

            "RMSE_improvement_percent":
                improvement,

            "test_R2":
                row["test_R2"]

        })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_file = os.path.join(

    REPORT_DIR,

    "step19_improvement_over_persistence.csv"

)


comparison_df.to_csv(

    comparison_file,

    index=False

)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 26: FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 19 FINAL RESULTS"
)

print(
    "=" * 80
)


display_columns = [

    "sensor",

    "strategy",

    "target_type",

    "val_RMSE",

    "val_R2",

    "test_RMSE",

    "test_R2"

]


print(

    results_df[
        display_columns
    ]

    .sort_values(
        [
            "sensor",
            "test_RMSE"
        ]
    )

    .to_string(
        index=False
    )

)


# ============================================================
# STEP 27: BEST VALIDATION MODELS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "BEST MODEL BY VALIDATION RMSE"
)

print(
    "=" * 80
)


print(

    best_val_df[
        display_columns
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# STEP 28: BEST TEST MODELS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "BEST MODEL BY TEST RMSE"
)

print(
    "=" * 80
)


print(

    best_test_df[
        display_columns
    ]

    .to_string(
        index=False
    )

)


# ============================================================
# STEP 29: INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 19 INTERPRETATION"
)

print(
    "=" * 80
)


print(
    """
The purpose of Step 19 is not simply to obtain a lower RMSE.

We are testing whether:

1. Predicting future CHANGE instead of the absolute value
   improves forecasting.

2. Trend information improves forecasting.

3. Recent volatility improves forecasting.

4. Exponentially weighted historical information improves
   forecasting.

5. The improvements generalize from validation to the
   completely unseen test period.

The Persistence model remains the most important baseline.

A model is considered genuinely useful only when it provides
consistent improvement over Persistence and does not rely on
future information.
"""
)


# ============================================================
# STEP 30: FILE SUMMARY
# ============================================================

print(
    "\nGenerated files:"
)

print(
    f"  {results_file}"
)

print(
    f"  {best_val_file}"
)

print(
    f"  {best_test_file}"
)

print(
    f"  {comparison_file}"
)

print(
    f"  {prediction_file}"
)


# ============================================================
# END
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 19 COMPLETED"
)

print(
    "=" * 80
)

print(
    "\nNext step:"
)

print(
    "STEP 20 - FINAL MODEL SELECTION"
)