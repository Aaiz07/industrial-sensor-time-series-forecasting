"""
10_final_model_training.py

STEP 10 - FINAL MODEL TRAINING

Purpose:
    Train and compare final candidate models using the leakage-safe
    feature set created in Step 9.

Models:
    1. Persistence
    2. Recent Mean
    3. Ridge
    4. XGBoost

Why no large LSTM here?
    We already tested LSTM in Step 2. It did not consistently beat
    simple baselines, and the horizon analysis showed that
    predictability is strongly sensor/horizon dependent.

The final model is selected PER SENSOR using validation RMSE.
The untouched TEST set is used only once for final reporting.

Split:
    70% TRAIN
    15% VALIDATION
    15% TEST

Outputs:
    outputs/models/
        ridge_<sensor>.joblib
        xgb_<sensor>.joblib

    outputs/reports/
        final_model_validation.csv
        final_model_test.csv
        final_model_selection.csv

    outputs/data/
        final_test_predictions.csv
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

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
    "final_features.csv"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
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

TARGET_PREFIX = "target_"


# ============================================================
# MODEL SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RIDGE_ALPHA = 10.0

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 42,
}


# ============================================================
# METRICS
# ============================================================

def metrics(y_true, y_pred):

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
        return np.nan, np.nan, np.nan

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

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 10 - FINAL MODEL TRAINING")
print("=" * 75)

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run 09_final_feature_strategy.py first."
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
    .reset_index(drop=True)
)

print(
    f"\nRows: {len(df):,}"
)


# ============================================================
# INPUT FEATURES
# ============================================================

feature_cols = [
    col
    for col in df.columns
    if col != TIMESTAMP_COL
    and not col.startswith(TARGET_PREFIX)
]

print(
    f"Input features: {len(feature_cols):,}"
)

print(
    f"Targets: {len(SENSOR_COLS)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (TRAIN_RATIO + VAL_RATIO)
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

print("\nChronological split:")
print(
    f"Train: {len(train_df):,} "
    f"({train_df[TIMESTAMP_COL].min()} → "
    f"{train_df[TIMESTAMP_COL].max()})"
)

print(
    f"Validation: {len(val_df):,} "
    f"({val_df[TIMESTAMP_COL].min()} → "
    f"{val_df[TIMESTAMP_COL].max()})"
)

print(
    f"Test: {len(test_df):,} "
    f"({test_df[TIMESTAMP_COL].min()} → "
    f"{test_df[TIMESTAMP_COL].max()})"
)


# ============================================================
# NUMERIC ARRAYS
# ============================================================

X_train_raw = train_df[
    feature_cols
].to_numpy(
    dtype=float
)

X_val_raw = val_df[
    feature_cols
].to_numpy(
    dtype=float
)

X_test_raw = test_df[
    feature_cols
].to_numpy(
    dtype=float
)


# ============================================================
# HANDLE NUMERIC FEATURE ISSUES
# ============================================================

# Feature generation should already have removed NaNs.
# This is only a safety guard.

feature_medians = np.nanmedian(
    X_train_raw,
    axis=0
)

feature_medians = np.where(
    np.isfinite(feature_medians),
    feature_medians,
    0.0
)

def fill_features(X):

    X = np.asarray(
        X,
        dtype=float
    )

    bad = ~np.isfinite(X)

    if bad.any():

        X = X.copy()

        rows, cols = np.where(bad)

        X[rows, cols] = feature_medians[
            cols
        ]

    return X


X_train_raw = fill_features(
    X_train_raw
)

X_val_raw = fill_features(
    X_val_raw
)

X_test_raw = fill_features(
    X_test_raw
)


# ============================================================
# RIDGE SCALING
# ============================================================

ridge_scaler = StandardScaler()

X_train_scaled = ridge_scaler.fit_transform(
    X_train_raw
)

X_val_scaled = ridge_scaler.transform(
    X_val_raw
)

X_test_scaled = ridge_scaler.transform(
    X_test_raw
)

joblib.dump(
    ridge_scaler,
    os.path.join(
        MODEL_DIR,
        "ridge_feature_scaler.joblib"
    )
)


# ============================================================
# TRAIN
# ============================================================

validation_records = []
test_records = []

test_prediction_df = pd.DataFrame({
    TIMESTAMP_COL:
        test_df[TIMESTAMP_COL].values
})


for sensor in SENSOR_COLS:

    target_col = (
        TARGET_PREFIX
        + sensor
    )

    y_train = train_df[
        target_col
    ].to_numpy(
        dtype=float
    )

    y_val = val_df[
        target_col
    ].to_numpy(
        dtype=float
    )

    y_test = test_df[
        target_col
    ].to_numpy(
        dtype=float
    )

    print("\n" + "=" * 75)
    print(f"SENSOR: {sensor}")
    print("=" * 75)

    # --------------------------------------------------------
    # BASELINE 1: PERSISTENCE
    # --------------------------------------------------------

    # The first row cannot have a previous target inside this
    # feature file, but the feature corresponding to lag_1 is
    # available.

    lag1_name = f"{sensor}_lag_1"

    if lag1_name in val_df.columns:

        pred_val_persistence = val_df[
            lag1_name
        ].to_numpy(
            dtype=float
        )

        pred_test_persistence = test_df[
            lag1_name
        ].to_numpy(
            dtype=float
        )

    else:

        pred_val_persistence = np.repeat(
            y_train[-1],
            len(y_val)
        )

        pred_test_persistence = np.repeat(
            y_val[-1],
            len(y_test)
        )

    mae, rmse, r2 = metrics(
        y_val,
        pred_val_persistence
    )

    validation_records.append({
        "sensor": sensor,
        "model": "Persistence",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    mae, rmse, r2 = metrics(
        y_test,
        pred_test_persistence
    )

    test_records.append({
        "sensor": sensor,
        "model": "Persistence",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    # --------------------------------------------------------
    # BASELINE 2: RECENT MEAN
    # --------------------------------------------------------

    mean_candidates = [
        f"{sensor}_rollmean_30",
        f"{sensor}_rollmean_15",
        f"{sensor}_rollmean_5",
    ]

    mean_feature = None

    for candidate in mean_candidates:

        if candidate in val_df.columns:

            mean_feature = candidate
            break

    if mean_feature is not None:

        pred_val_mean = val_df[
            mean_feature
        ].to_numpy(
            dtype=float
        )

        pred_test_mean = test_df[
            mean_feature
        ].to_numpy(
            dtype=float
        )

        mae, rmse, r2 = metrics(
            y_val,
            pred_val_mean
        )

        validation_records.append({
            "sensor": sensor,
            "model": "RecentMean",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        mae, rmse, r2 = metrics(
            y_test,
            pred_test_mean
        )

        test_records.append({
            "sensor": sensor,
            "model": "RecentMean",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    # --------------------------------------------------------
    # MODEL 1: RIDGE
    # --------------------------------------------------------

    print("\nTraining Ridge...")

    ridge = Ridge(
        alpha=RIDGE_ALPHA
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    pred_val_ridge = ridge.predict(
        X_val_scaled
    )

    pred_test_ridge = ridge.predict(
        X_test_scaled
    )

    mae, rmse, r2 = metrics(
        y_val,
        pred_val_ridge
    )

    validation_records.append({
        "sensor": sensor,
        "model": "Ridge",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    mae, rmse, r2 = metrics(
        y_test,
        pred_test_ridge
    )

    test_records.append({
        "sensor": sensor,
        "model": "Ridge",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    joblib.dump(
        ridge,
        os.path.join(
            MODEL_DIR,
            f"ridge_{sensor}.joblib"
        )
    )

    # --------------------------------------------------------
    # MODEL 2: XGBOOST
    # --------------------------------------------------------

    print(
        "Training XGBoost..."
    )

    xgb = XGBRegressor(
        **XGB_PARAMS
    )

    xgb.fit(
        X_train_raw,
        y_train
    )

    pred_val_xgb = xgb.predict(
        X_val_raw
    )

    pred_test_xgb = xgb.predict(
        X_test_raw
    )

    mae, rmse, r2 = metrics(
        y_val,
        pred_val_xgb
    )

    validation_records.append({
        "sensor": sensor,
        "model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    mae, rmse, r2 = metrics(
        y_test,
        pred_test_xgb
    )

    test_records.append({
        "sensor": sensor,
        "model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    joblib.dump(
        xgb,
        os.path.join(
            MODEL_DIR,
            f"xgb_{sensor}.joblib"
        )
    )

    # --------------------------------------------------------
    # TEMPORARY TEST PREDICTIONS
    # --------------------------------------------------------

    test_prediction_df[
        f"{sensor}_actual"
    ] = y_test

    test_prediction_df[
        f"{sensor}_persistence"
    ] = pred_test_persistence

    if mean_feature is not None:

        test_prediction_df[
            f"{sensor}_recent_mean"
        ] = pred_test_mean

    test_prediction_df[
        f"{sensor}_ridge"
    ] = pred_test_ridge

    test_prediction_df[
        f"{sensor}_xgb"
    ] = pred_test_xgb

    # --------------------------------------------------------
    # PRINT VALIDATION
    # --------------------------------------------------------

    sensor_val = pd.DataFrame(
        validation_records
    )

    sensor_val = sensor_val[
        sensor_val["sensor"] == sensor
    ].sort_values(
        "RMSE"
    )

    print("\nValidation ranking:")

    print(
        sensor_val[
            [
                "model",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE REPORTS
# ============================================================

validation_df = pd.DataFrame(
    validation_records
)

test_df_results = pd.DataFrame(
    test_records
)

validation_file = os.path.join(
    REPORT_DIR,
    "final_model_validation.csv"
)

test_file = os.path.join(
    REPORT_DIR,
    "final_model_test.csv"
)

prediction_file = os.path.join(
    DATA_DIR,
    "final_test_predictions.csv"
)

validation_df.to_csv(
    validation_file,
    index=False
)

test_df_results.to_csv(
    test_file,
    index=False
)

test_prediction_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# SELECT BEST MODEL PER SENSOR
# ============================================================

selection_records = []

for sensor in SENSOR_COLS:

    subset = validation_df[
        validation_df["sensor"] == sensor
    ].copy()

    subset = subset.sort_values(
        "RMSE"
    )

    best = subset.iloc[0]

    selection_records.append({
        "sensor": sensor,
        "selected_model": best["model"],
        "validation_MAE": best["MAE"],
        "validation_RMSE": best["RMSE"],
        "validation_R2": best["R2"],
    })

selection_df = pd.DataFrame(
    selection_records
)

selection_file = os.path.join(
    REPORT_DIR,
    "final_model_selection.csv"
)

selection_df.to_csv(
    selection_file,
    index=False
)


# ============================================================
# FINAL TEST REPORT
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SELECTION")
print("=" * 75)

print(
    selection_df.to_string(
        index=False
    )
)

print("\n" + "=" * 75)
print("UNTOUCHED TEST PERFORMANCE")
print("=" * 75)

for sensor in SENSOR_COLS:

    selected = selection_df.loc[
        selection_df["sensor"] == sensor,
        "selected_model"
    ].iloc[0]

    row = test_df_results[
        (test_df_results["sensor"] == sensor)
        &
        (test_df_results["model"] == selected)
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(
        f"{sensor:15s} -> "
        f"{selected:12s} | "
        f"MAE={row['MAE']:.6f} | "
        f"RMSE={row['RMSE']:.6f} | "
        f"R2={row['R2']:.6f}"
    )


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "ridge_alpha": RIDGE_ALPHA,
    "xgb_params": XGB_PARAMS,
    "feature_count": len(feature_cols),
    "sensor_count": len(SENSOR_COLS),
}

joblib.dump(
    config,
    os.path.join(
        MODEL_DIR,
        "final_training_config.joblib"
    )
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("STEP 10 COMPLETED")
print("=" * 75)

print(
    "\nValidation report:"
)

print(
    f"  {validation_file}"
)

print(
    "\nTest report:"
)

print(
    f"  {test_file}"
)

print(
    "\nTest predictions:"
)

print(
    f"  {prediction_file}"
)

print(
    "\nModels saved in:"
)

print(
    f"  {MODEL_DIR}"
)

print(
    "\nIMPORTANT:"
)

print(
    "Model selection was made using validation RMSE."
)

print(
    "The test set was not used for model selection."
)

print(
    "The next step will analyze actual vs predicted values "
    "and forecast behavior."
)

print("\n" + "=" * 75)
