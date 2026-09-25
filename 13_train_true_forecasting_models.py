import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================================================
# STEP 13: TRAIN TRUE FORECASTING MODELS
# ============================================================
# FEATURES at time t
#          ↓
# TARGET at time t + 60 seconds
#
# Models:
#   1. Persistence
#   2. Recent Mean
#   3. Ridge
#   4. XGBoost
#
# Chronological split:
#   70% Train
#   15% Validation
#   15% Test
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FORECAST_FILE = os.path.join(
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "data"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

FUTURE_TARGET_COLS = [
    f"future_{sensor}"
    for sensor in SENSOR_COLS
]


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

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
# 1. LOAD TRUE FORECASTING DATA
# ============================================================

print("=" * 70)
print("STEP 13: TRUE FORECASTING MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(
    FORECAST_FILE,
    parse_dates=[
        "ts",
        "target_ts"
    ]
)

clean = pd.read_csv(
    CLEAN_FILE,
    parse_dates=["ts"]
)

df = df.sort_values(
    "ts"
).reset_index(drop=True)

clean = clean.sort_values(
    "ts"
).reset_index(drop=True)

print(
    f"\nForecast dataset rows: {len(df):,}"
)

print(
    f"Original cleaned rows: {len(clean):,}"
)

print(
    f"Forecast horizon: {HORIZON_SECONDS} seconds"
)


# ============================================================
# 2. ALIGN CURRENT SENSOR VALUES
# ============================================================
# The forecasting dataset contains:
#
#     t → t+60 sec
#
# but does not contain the raw current sensor columns.
#
# We therefore bring the CURRENT sensor value at time t
# from sensor_data_cleaned.csv.
# ============================================================

current_values = clean[
    ["ts"] + SENSOR_COLS
].copy()

df = df.merge(
    current_values,
    on="ts",
    how="left"
)

print(
    f"\nRows after current-value alignment: {len(df):,}"
)


# ============================================================
# 3. IDENTIFY INPUT FEATURES
# ============================================================

excluded_cols = (
    [
        "ts",
        "target_ts",
        "actual_horizon_seconds"
    ]
    + FUTURE_TARGET_COLS
    + SENSOR_COLS
)

feature_cols = [
    c
    for c in df.columns
    if c not in excluded_cols
]

print(
    f"Input features: {len(feature_cols)}"
)

print(
    f"Forecast targets: {len(FUTURE_TARGET_COLS)}"
)


# ============================================================
# 4. CHECK FOR MISSING VALUES
# ============================================================

required_cols = (
    feature_cols
    + SENSOR_COLS
    + FUTURE_TARGET_COLS
)

missing_before = df[required_cols].isna().sum().sum()

print(
    f"\nMissing values before cleaning: "
    f"{missing_before:,}"
)

df = df.dropna(
    subset=required_cols
).reset_index(drop=True)

print(
    f"Usable rows: {len(df):,}"
)


# ============================================================
# 5. CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)

train = df.iloc[
    :train_end
].copy()

val = df.iloc[
    train_end:val_end
].copy()

test = df.iloc[
    val_end:
].copy()

print("\nCHRONOLOGICAL SPLIT")

print(
    f"Train:      {len(train):,} "
    f"({train['ts'].iloc[0]} → "
    f"{train['ts'].iloc[-1]})"
)

print(
    f"Validation: {len(val):,} "
    f"({val['ts'].iloc[0]} → "
    f"{val['ts'].iloc[-1]})"
)

print(
    f"Test:       {len(test):,} "
    f"({test['ts'].iloc[0]} → "
    f"{test['ts'].iloc[-1]})"
)


# ============================================================
# 6. STORAGE
# ============================================================

validation_results = []
test_results = []

selected_models = {}

test_predictions = test[
    [
        "ts",
        "target_ts"
    ]
].copy()


# ============================================================
# 7. TRAIN SENSOR BY SENSOR
# ============================================================

for sensor in SENSOR_COLS:

    future_target = (
        f"future_{sensor}"
    )

    print("\n" + "=" * 70)
    print(f"SENSOR: {sensor}")
    print("=" * 70)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y_train = train[
        future_target
    ].values

    y_val = val[
        future_target
    ].values

    y_test = test[
        future_target
    ].values

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    X_train = train[
        feature_cols
    ].values

    X_val = val[
        feature_cols
    ].values

    X_test = test[
        feature_cols
    ].values


    # ========================================================
    # MODEL 1: PERSISTENCE
    # ========================================================
    #
    # Prediction:
    #
    # future(t+60) = current(t)
    #
    # This is a genuine forecasting baseline.
    # ========================================================

    persistence_val = val[
        sensor
    ].values

    persistence_test = test[
        sensor
    ].values

    p_mae, p_rmse, p_r2 = calculate_metrics(
        y_val,
        persistence_val
    )

    validation_results.append({
        "sensor": sensor,
        "model": "Persistence",
        "MAE": p_mae,
        "RMSE": p_rmse,
        "R2": p_r2
    })


    # ========================================================
    # MODEL 2: RECENT MEAN
    # ========================================================
    #
    # Mean of the previous 30 observations.
    #
    # shift(1) guarantees that the current value itself
    # is not used in calculating the mean.
    # ========================================================

    combined = pd.concat(
        [
            train[
                ["ts", sensor]
            ],
            val[
                ["ts", sensor]
            ],
            test[
                ["ts", sensor]
            ]
        ],
        axis=0
    ).reset_index(drop=True)

    recent_mean = (
        combined[sensor]
        .shift(1)
        .rolling(30)
        .mean()
    )

    val_start = len(train)

    val_end_index = (
        val_start + len(val)
    )

    recent_val = recent_mean.iloc[
        val_start:val_end_index
    ].values

    test_start = val_end_index

    recent_test = recent_mean.iloc[
        test_start:
    ].values

    val_mask = ~np.isnan(
        recent_val
    )

    rmean_mae, rmean_rmse, rmean_r2 = calculate_metrics(
        y_val[val_mask],
        recent_val[val_mask]
    )

    validation_results.append({
        "sensor": sensor,
        "model": "RecentMean30",
        "MAE": rmean_mae,
        "RMSE": rmean_rmse,
        "R2": rmean_r2
    })


    # ========================================================
    # MODEL 3: RIDGE
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    ridge = Ridge(
        alpha=10.0
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    ridge_val = ridge.predict(
        X_val_scaled
    )

    ridge_test = ridge.predict(
        X_test_scaled
    )

    ridge_mae, ridge_rmse, ridge_r2 = calculate_metrics(
        y_val,
        ridge_val
    )

    validation_results.append({
        "sensor": sensor,
        "model": "Ridge",
        "MAE": ridge_mae,
        "RMSE": ridge_rmse,
        "R2": ridge_r2
    })


    # ========================================================
    # MODEL 4: XGBOOST
    # ========================================================

    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
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
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False
    )

    xgb_val = xgb.predict(
        X_val
    )

    xgb_test = xgb.predict(
        X_test
    )

    xgb_mae, xgb_rmse, xgb_r2 = calculate_metrics(
        y_val,
        xgb_val
    )

    validation_results.append({
        "sensor": sensor,
        "model": "XGBoost",
        "MAE": xgb_mae,
        "RMSE": xgb_rmse,
        "R2": xgb_r2
    })


    # ========================================================
    # PRINT VALIDATION RESULTS
    # ========================================================

    print("\nValidation Results:")

    print(
        f"Persistence  | "
        f"MAE={p_mae:.6f} | "
        f"RMSE={p_rmse:.6f} | "
        f"R2={p_r2:.6f}"
    )

    print(
        f"RecentMean30 | "
        f"MAE={rmean_mae:.6f} | "
        f"RMSE={rmean_rmse:.6f} | "
        f"R2={rmean_r2:.6f}"
    )

    print(
        f"Ridge        | "
        f"MAE={ridge_mae:.6f} | "
        f"RMSE={ridge_rmse:.6f} | "
        f"R2={ridge_r2:.6f}"
    )

    print(
        f"XGBoost      | "
        f"MAE={xgb_mae:.6f} | "
        f"RMSE={xgb_rmse:.6f} | "
        f"R2={xgb_r2:.6f}"
    )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    candidates = {
        "Persistence": (
            p_rmse,
            persistence_test
        ),

        "RecentMean30": (
            rmean_rmse,
            recent_test
        ),

        "Ridge": (
            ridge_rmse,
            ridge_test
        ),

        "XGBoost": (
            xgb_rmse,
            xgb_test
        )
    }

    best_model = min(
        candidates,
        key=lambda name:
        candidates[name][0]
    )

    best_rmse = candidates[
        best_model
    ][0]

    best_prediction = candidates[
        best_model
    ][1]

    selected_models[sensor] = {
        "model": best_model,
        "validation_rmse": float(
            best_rmse
        )
    }

    print(
        f"\nBEST MODEL: {best_model}"
    )

    print(
        f"Validation RMSE: "
        f"{best_rmse:.6f}"
    )


    # ========================================================
    # TEST RESULTS
    # ========================================================

    test_mae, test_rmse, test_r2 = calculate_metrics(
        y_test,
        best_prediction
    )

    test_results.append({
        "sensor": sensor,
        "selected_model": best_model,
        "MAE": test_mae,
        "RMSE": test_rmse,
        "R2": test_r2
    })

    print(
        f"Test | "
        f"MAE={test_mae:.6f} | "
        f"RMSE={test_rmse:.6f} | "
        f"R2={test_r2:.6f}"
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_predictions[
        f"actual_{sensor}"
    ] = y_test

    test_predictions[
        f"predicted_{sensor}"
    ] = best_prediction

    test_predictions[
        f"selected_model_{sensor}"
    ] = best_model


    # ========================================================
    # SAVE SELECTED MODEL
    # ========================================================

    if best_model == "Ridge":

        joblib.dump(
            ridge,
            os.path.join(
                MODEL_DIR,
                f"ridge_{sensor}_60s.joblib"
            )
        )

        joblib.dump(
            scaler,
            os.path.join(
                MODEL_DIR,
                f"ridge_scaler_{sensor}_60s.joblib"
            )
        )

    elif best_model == "XGBoost":

        joblib.dump(
            xgb,
            os.path.join(
                MODEL_DIR,
                f"xgboost_{sensor}_60s.joblib"
            )
        )


# ============================================================
# 8. SAVE VALIDATION REPORT
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_file = os.path.join(
    REPORT_DIR,
    "true_forecasting_validation_60s.csv"
)

validation_df.to_csv(
    validation_file,
    index=False
)


# ============================================================
# 9. SAVE TEST REPORT
# ============================================================

test_df = pd.DataFrame(
    test_results
)

test_file = os.path.join(
    REPORT_DIR,
    "true_forecasting_test_60s.csv"
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_DIR,
    "true_forecasting_test_predictions_60s.csv"
)

test_predictions.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# 11. SAVE MODEL SELECTION
# ============================================================

selection_df = pd.DataFrame([
    {
        "sensor": sensor,
        "selected_model": info["model"],
        "validation_rmse": info[
            "validation_rmse"
        ]
    }
    for sensor, info
    in selected_models.items()
])

selection_file = os.path.join(
    REPORT_DIR,
    "true_forecasting_model_selection_60s.csv"
)

selection_df.to_csv(
    selection_file,
    index=False
)


# ============================================================
# 12. SAVE CONFIG
# ============================================================

config = {
    "forecast_horizon_seconds": HORIZON_SECONDS,
    "number_of_features": len(feature_cols),
    "number_of_sensors": len(SENSOR_COLS),
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "models": [
        "Persistence",
        "RecentMean30",
        "Ridge",
        "XGBoost"
    ],
    "selection_metric": "Validation RMSE",
    "test_used_for_selection": False
}

config_file = os.path.join(
    REPORT_DIR,
    "true_forecasting_config_60s.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL PER SENSOR")
print("=" * 70)

for sensor, info in selected_models.items():

    print(
        f"{sensor:<15} -> "
        f"{info['model']:<15} | "
        f"Validation RMSE = "
        f"{info['validation_rmse']:.6f}"
    )


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    test_df.to_string(
        index=False
    )
)


print("\nSAVED:")
print(f"  {validation_file}")
print(f"  {test_file}")
print(f"  {prediction_file}")
print(f"  {selection_file}")
print(f"  {config_file}")


print("\n" + "=" * 70)
print("STEP 13 COMPLETED")
print("=" * 70)

print(
    "\nThese are TRUE 60-second-ahead "
    "forecasting results."
)

print(
    "Test data was not used for model selection."
)