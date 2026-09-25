# ============================================================
# STEP 21: FINAL MODEL TRAINING + FINAL FORECAST
# ============================================================
#
# Purpose:
#   Train the final selected forecasting models using the
#   conclusions from Step 20.
#
# Final models selected from walk-forward validation:
#
#   vib_x_rms    -> XGBoost
#   vib_y_rms    -> XGBoost
#   vib_z_rms    -> XGBoost
#   mag_x        -> Ridge
#   mag_y        -> Ridge
#   mag_z        -> Ridge
#   temperature  -> Persistence
#
# Forecast horizon:
#   Exactly 60 seconds
#
# Outputs:
#   outputs/models/step21_final_<sensor>.joblib
#   outputs/reports/step21_final_model_performance.csv
#   outputs/reports/step21_final_model_selection.csv
#   outputs/forecasts/step21_final_predictions.csv
#
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

CLEANED_DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

FUTURE_DATA = os.path.join(
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

FORECAST_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

HORIZON_SECONDS = 60

TARGETS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

FUTURE_TARGETS = {
    "vib_x_rms": "future_vib_x_rms",
    "vib_y_rms": "future_vib_y_rms",
    "vib_z_rms": "future_vib_z_rms",
    "mag_x": "future_mag_x",
    "mag_y": "future_mag_y",
    "mag_z": "future_mag_z",
    "temperature": "future_temperature"
}

FINAL_MODELS = {
    "vib_x_rms": "XGBoost",
    "vib_y_rms": "XGBoost",
    "vib_z_rms": "XGBoost",
    "mag_x": "Ridge",
    "mag_y": "Ridge",
    "mag_z": "Ridge",
    "temperature": "Persistence"
}


# ============================================================
# 3. METRIC FUNCTION
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
# 4. START
# ============================================================

print("=" * 75)
print("STEP 21: FINAL MODEL TRAINING + FINAL FORECAST")
print("=" * 75)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n[1/8] Loading data...")

if not os.path.exists(CLEANED_DATA):
    raise FileNotFoundError(
        f"Cleaned data not found:\n{CLEANED_DATA}"
    )

if not os.path.exists(FUTURE_DATA):
    raise FileNotFoundError(
        f"Future data not found:\n{FUTURE_DATA}"
    )

cleaned = pd.read_csv(
    CLEANED_DATA
)

future = pd.read_csv(
    FUTURE_DATA
)

cleaned["ts"] = pd.to_datetime(
    cleaned["ts"]
)

future["ts"] = pd.to_datetime(
    future["ts"]
)

cleaned = (
    cleaned
    .sort_values("ts")
    .reset_index(drop=True)
)

future = (
    future
    .sort_values("ts")
    .reset_index(drop=True)
)

print(
    f"Cleaned data shape: {cleaned.shape}"
)

print(
    f"Future feature data shape: {future.shape}"
)


# ============================================================
# 6. VERIFY HORIZON
# ============================================================

print("\n[2/8] Verifying forecasting horizon...")

if "actual_horizon_seconds" in future.columns:

    horizon_values = (
        future["actual_horizon_seconds"]
        .dropna()
        .unique()
    )

    print(
        "Horizon values:",
        horizon_values
    )

    if not np.allclose(
        horizon_values,
        HORIZON_SECONDS
    ):
        raise ValueError(
            "Future data does not contain an exact "
            "60-second forecasting horizon."
        )

print(
    "PASS: Exact 60-second forecasting horizon confirmed."
)


# ============================================================
# 7. VERIFY FUTURE TARGETS
# ============================================================

print("\n[3/8] Checking future target columns...")

for sensor, future_column in FUTURE_TARGETS.items():

    if future_column not in future.columns:

        raise ValueError(
            f"Missing future target for {sensor}: "
            f"{future_column}"
        )

    print(
        f"  {sensor} -> {future_column}"
    )

print(
    "PASS: All future target columns are present."
)


# ============================================================
# 8. MERGE CURRENT SENSOR VALUES
# ============================================================

print("\n[4/8] Merging current sensor values...")

current_columns = [
    "ts"
] + TARGETS

current_data = cleaned[
    current_columns
].copy()

df = future.merge(
    current_data,
    on="ts",
    how="inner"
)

print(
    f"Rows after merge: {len(df):,}"
)


# ============================================================
# 9. IDENTIFY MODEL FEATURES
# ============================================================

print("\n[5/8] Preparing model features...")

# These columns must NOT be model inputs.

EXCLUDE_COLUMNS = [
    "ts",
    "target_ts",
    "actual_horizon_seconds"
]

# Future targets
EXCLUDE_COLUMNS += list(
    FUTURE_TARGETS.values()
)

# Current raw sensor values
EXCLUDE_COLUMNS += TARGETS

EXCLUDE_COLUMNS = list(
    dict.fromkeys(
        EXCLUDE_COLUMNS
    )
)

feature_columns = []

for column in df.columns:

    if column in EXCLUDE_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        feature_columns.append(
            column
        )

print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)


# ============================================================
# 10. LEAKAGE CHECK
# ============================================================

print("\nRunning final leakage check...")

leakage_columns = []

for feature in feature_columns:

    feature_lower = feature.lower()

    # Future targets
    if feature in FUTURE_TARGETS.values():

        leakage_columns.append(
            feature
        )

    # Raw target columns
    if feature in TARGETS:

        leakage_columns.append(
            feature
        )

    # Future-looking feature names
    if feature_lower.startswith(
        "future_"
    ):

        leakage_columns.append(
            feature
        )

if leakage_columns:

    raise ValueError(
        "LEAKAGE DETECTED:\n"
        + "\n".join(
            leakage_columns
        )
    )

print(
    "PASS: No obvious target leakage detected."
)


# ============================================================
# 11. CLEAN DATA
# ============================================================

print("\nCleaning model data...")

required_columns = (
    feature_columns
    + list(
        FUTURE_TARGETS.values()
    )
    + TARGETS
)

df_model = df[
    ["ts"] + required_columns
].copy()

df_model = df_model.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(df_model)

df_model = df_model.dropna(
    subset=required_columns
).reset_index(
    drop=True
)

after = len(df_model)

print(
    f"Rows removed: {before - after:,}"
)

print(
    f"Final usable rows: {after:,}"
)


# ============================================================
# 12. FINAL MODEL TRAINING
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL MODELS")
print("=" * 75)

all_predictions = []

performance_results = []

trained_models = {}

X = df_model[
    feature_columns
].values

print(
    f"\nTraining rows: {len(X):,}"
)

print(
    f"Features: {len(feature_columns)}"
)


# ============================================================
# 13. TRAIN EACH SENSOR
# ============================================================

for sensor in TARGETS:

    model_name = FINAL_MODELS[
        sensor
    ]

    future_target = FUTURE_TARGETS[
        sensor
    ]

    print("\n" + "-" * 75)

    print(
        f"SENSOR: {sensor}"
    )

    print(
        f"MODEL : {model_name}"
    )

    y = df_model[
        future_target
    ].values

    current = df_model[
        sensor
    ].values

    timestamps = df_model[
        "ts"
    ].values


    # ========================================================
    # PERSISTENCE
    # ========================================================

    if model_name == "Persistence":

        print(
            "Using persistence baseline."
        )

        prediction = current.copy()

        model = None


    # ========================================================
    # RIDGE
    # ========================================================

    elif model_name == "Ridge":

        print(
            "Training Ridge Regression..."
        )

        model = Ridge(
            alpha=10.0
        )

        model.fit(
            X,
            y
        )

        prediction = model.predict(
            X
        )


    # ========================================================
    # XGBOOST
    # ========================================================

    elif model_name == "XGBoost":

        print(
            "Training XGBoost..."
        )

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X,
            y
        )

        prediction = model.predict(
            X
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


    # ========================================================
    # METRICS
    # ========================================================

    mae, rmse, r2 = calculate_metrics(
        y,
        prediction
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R2   : {r2:.6f}"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"step21_final_{sensor}.joblib"
    )

    if model is not None:

        joblib.dump(
            {
                "model": model,
                "sensor": sensor,
                "model_name": model_name,
                "features": feature_columns,
                "horizon_seconds":
                    HORIZON_SECONDS
            },
            model_path
        )

        print(
            f"Saved model: {model_path}"
        )

    else:

        # Save configuration for persistence.
        joblib.dump(
            {
                "model": None,
                "sensor": sensor,
                "model_name": "Persistence",
                "features": [],
                "horizon_seconds":
                    HORIZON_SECONDS
            },
            model_path
        )

        print(
            f"Saved persistence configuration: "
            f"{model_path}"
        )


    trained_models[
        sensor
    ] = model


    # ========================================================
    # SAVE PERFORMANCE
    # ========================================================

    performance_results.append(
        {
            "sensor": sensor,
            "model": model_name,
            "training_rows": len(X),
            "features": len(feature_columns),
            "horizon_seconds":
                HORIZON_SECONDS,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    for i in range(len(y)):

        all_predictions.append(
            {
                "ts": timestamps[i],
                "sensor": sensor,
                "model": model_name,
                "actual": y[i],
                "prediction": prediction[i],
                "error":
                    y[i] - prediction[i],
                "absolute_error":
                    abs(
                        y[i] - prediction[i]
                    ),
                "horizon_seconds":
                    HORIZON_SECONDS
            }
        )


# ============================================================
# 14. SAVE PERFORMANCE REPORT
# ============================================================

performance_df = pd.DataFrame(
    performance_results
)

performance_path = os.path.join(
    REPORT_DIR,
    "step21_final_model_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)


# ============================================================
# 15. SAVE FINAL MODEL SELECTION
# ============================================================

selection_rows = []

for sensor in TARGETS:

    selection_rows.append(
        {
            "sensor": sensor,
            "final_model":
                FINAL_MODELS[sensor],
            "forecast_horizon_seconds":
                HORIZON_SECONDS,
            "reason":
                "Selected from Step 20 walk-forward validation"
        }
    )

selection_df = pd.DataFrame(
    selection_rows
)

selection_path = os.path.join(
    REPORT_DIR,
    "step21_final_model_selection.csv"
)

selection_df.to_csv(
    selection_path,
    index=False
)


# ============================================================
# 16. SAVE FINAL PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

predictions_path = os.path.join(
    FORECAST_DIR,
    "step21_final_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 17. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL PERFORMANCE")
print("=" * 75)

print(
    performance_df.to_string(
        index=False
    )
)


# ============================================================
# 18. FINAL MODEL TABLE
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SELECTION")
print("=" * 75)

print(
    selection_df.to_string(
        index=False
    )
)


# ============================================================
# 19. IMPORTANT INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("IMPORTANT INTERPRETATION")
print("=" * 75)

print(
    """
These metrics are training-data metrics because the final models
were trained using all currently available labeled forecasting rows.

Therefore:

DO NOT report these numbers as the unbiased final test accuracy.

The unbiased evidence comes from Step 20 walk-forward validation.

Step 21 is primarily responsible for producing the final trained
model artifacts and the final forecast pipeline.

The Step 20 results remain the main evidence for model reliability.
"""
)


# ============================================================
# 20. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODELS")
print("=" * 75)

for sensor in TARGETS:

    print(
        f"{sensor:15s} -> "
        f"{FINAL_MODELS[sensor]}"
    )


# ============================================================
# 21. OUTPUT FILES
# ============================================================

print("\n" + "=" * 75)
print("FILES CREATED")
print("=" * 75)

print(
    f"Performance report:\n"
    f"{performance_path}"
)

print(
    f"\nModel selection:\n"
    f"{selection_path}"
)

print(
    f"\nFinal predictions:\n"
    f"{predictions_path}"
)

print(
    "\nFinal trained models:"
)

for sensor in TARGETS:

    model_path = os.path.join(
        MODEL_DIR,
        f"step21_final_{sensor}.joblib"
    )

    print(
        f"  {model_path}"
    )


# ============================================================
# 22. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("STEP 21 COMPLETE")
print("=" * 75)

print(
    """
The final model artifacts have been created.

Next step:
Analyze the final predictions and create the final
visualization/report for the forecasting project.
"""
)

print("\nDone.")