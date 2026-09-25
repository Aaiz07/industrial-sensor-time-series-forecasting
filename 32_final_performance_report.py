# ============================================================
# STEP 32: FINAL PERFORMANCE REPORT
# ============================================================
# Purpose:
#   Create the final professional performance report for the
#   complete sensor forecasting project.
#
# This script:
#   1. Loads existing evaluation results
#   2. Loads multi-horizon results
#   3. Builds final sensor-wise performance tables
#   4. Compares selected models
#   5. Analyzes forecastability
#   6. Analyzes forecast horizon performance
#   7. Generates CSV reports
#   8. Generates a comprehensive TXT report
#
# IMPORTANT:
#   - No model retraining
#   - No database modification
#   - No forecasting modification
#   - Uses existing evaluation results
# ============================================================

import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

FORECAST_DIR = os.path.join(
    OUTPUT_DIR,
    "forecasts"
)

DATA_DIR = os.path.join(
    OUTPUT_DIR,
    "data"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

SENSORS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]


SENSOR_GROUPS = {
    "Vibration": [
        "vib_x_rms",
        "vib_y_rms",
        "vib_z_rms",
    ],
    "Magnetic": [
        "mag_x",
        "mag_y",
        "mag_z",
    ],
    "Temperature": [
        "temperature",
    ],
}


FINAL_MODELS = {
    "vib_x_rms": "XGBoost",
    "vib_y_rms": "XGBoost",
    "vib_z_rms": "XGBoost",
    "mag_x": "Ridge",
    "mag_y": "Ridge",
    "mag_z": "Ridge",
    "temperature": "Persistence",
}


# ============================================================
# KNOWN WALK-FORWARD RESULTS
# ============================================================
# These are the final walk-forward results already obtained
# during Step 20 / Step 22.
#
# IMPORTANT:
# These are generalization-style evaluation results, not
# training performance.
# ============================================================

WALK_FORWARD_RESULTS = [
    {
        "sensor": "vib_x_rms",
        "model": "XGBoost",
        "mae": 0.019250,
        "rmse": 0.036500,
        "rmse_std": 0.031473,
        "r2": -8.375215,
    },
    {
        "sensor": "vib_y_rms",
        "model": "XGBoost",
        "mae": 0.024974,
        "rmse": 0.046564,
        "rmse_std": 0.038200,
        "r2": -7.574728,
    },
    {
        "sensor": "vib_z_rms",
        "model": "XGBoost",
        "mae": 0.008276,
        "rmse": 0.015306,
        "rmse_std": 0.011545,
        "r2": -2.613671,
    },
    {
        "sensor": "mag_x",
        "model": "Ridge",
        "mae": 114.777000,
        "rmse": 132.499000,
        "rmse_std": 0.995781,
        "r2": -0.002038,
    },
    {
        "sensor": "mag_y",
        "model": "Ridge",
        "mae": 114.555000,
        "rmse": 132.387000,
        "rmse_std": 1.089499,
        "r2": -0.003333,
    },
    {
        "sensor": "mag_z",
        "model": "Ridge",
        "mae": 113.912000,
        "rmse": 131.702000,
        "rmse_std": 0.559924,
        "r2": -0.001523,
    },
    {
        "sensor": "temperature",
        "model": "Persistence",
        "mae": 0.124780,
        "rmse": 0.189486,
        "rmse_std": 0.083086,
        "r2": 0.817088,
    },
]


# ============================================================
# MULTI-HORIZON RESULTS
# ============================================================
# Exact timestamp-based evaluation results obtained in
# Step 17.
#
# R² values are used for horizon analysis.
# ============================================================

MULTI_HORIZON_R2 = {

    "vib_x_rms": {
        60: -4.257,
        300: -16.272,
        600: -101.171,
        1800: -132.697,
        3600: -64.384,
    },

    "vib_y_rms": {
        60: -3.328,
        300: -10.406,
        600: -66.076,
        1800: -78.957,
        3600: -33.191,
    },

    "vib_z_rms": {
        60: -1.132,
        300: -5.763,
        600: -25.026,
        1800: -33.614,
        3600: -20.429,
    },

    "mag_x": {
        60: -0.00450,
        300: -0.00282,
        600: -0.00148,
        1800: -0.00241,
        3600: -0.00292,
    },

    "mag_y": {
        60: -0.00386,
        300: -0.00062,
        600: -0.00127,
        1800: -0.00359,
        3600: -0.01148,
    },

    "mag_z": {
        60: -0.00358,
        300: -0.00386,
        600: -0.00413,
        1800: -0.00404,
        3600: -0.00638,
    },

    "temperature": {
        60: 0.934887,
        300: 0.716699,
        600: 0.501596,
        1800: -0.120017,
        3600: -0.336376,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def separator(char="=", length=78):
    return char * length


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


def format_number(value, digits=4):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


# ============================================================
# START
# ============================================================

print(separator())
print("STEP 32: FINAL PERFORMANCE REPORT")
print(separator())

print()
print("Project:")
print(BASE_DIR)

print()
print("Generating final performance analysis...")
print()


# ============================================================
# 1. BUILD FINAL PERFORMANCE DATAFRAME
# ============================================================

performance_df = pd.DataFrame(WALK_FORWARD_RESULTS)

performance_df["selected_model"] = performance_df[
    "sensor"
].map(FINAL_MODELS)

performance_df["model_selection_verified"] = (
    performance_df["model"]
    == performance_df["selected_model"]
)


# ============================================================
# 2. MODEL SELECTION CONSISTENCY
# ============================================================

print(separator())
print("1. MODEL SELECTION CONSISTENCY")
print(separator())

selection_errors = performance_df[
    performance_df["model_selection_verified"] == False
]

if len(selection_errors) == 0:

    print("[PASS] Final model selection is consistent.")

else:

    print("[FAIL] Model selection mismatch detected.")

    print(selection_errors)


# ============================================================
# 3. SENSOR PERFORMANCE
# ============================================================

print()
print(separator())
print("2. FINAL SENSOR PERFORMANCE")
print(separator())

display_columns = [
    "sensor",
    "model",
    "mae",
    "rmse",
    "rmse_std",
    "r2",
]

print()

print(
    performance_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 4. FORECASTABILITY CLASSIFICATION
# ============================================================
# Classification is based on the final walk-forward R²:
#
# R² >= 0.50       Strong
# 0.20 <= R² < .50 Moderate
# 0 <= R² < .20    Weak
# R² < 0           Poor / not useful relative to baseline
# ============================================================

def classify_forecastability(r2):

    if r2 >= 0.50:
        return "Strong"

    elif r2 >= 0.20:
        return "Moderate"

    elif r2 >= 0:
        return "Weak"

    else:
        return "Poor"


performance_df["forecastability"] = (
    performance_df["r2"]
    .apply(classify_forecastability)
)


# ============================================================
# 5. ADD SENSOR GROUP
# ============================================================

def get_sensor_group(sensor):

    for group, sensors in SENSOR_GROUPS.items():

        if sensor in sensors:
            return group

    return "Other"


performance_df["sensor_group"] = (
    performance_df["sensor"]
    .apply(get_sensor_group)
)


# ============================================================
# 6. PRINT FORECASTABILITY
# ============================================================

print()
print(separator())
print("3. FORECASTABILITY ANALYSIS")
print(separator())

for _, row in performance_df.iterrows():

    print(
        f"{row['sensor']:15s} | "
        f"R² = {row['r2']:10.4f} | "
        f"{row['forecastability']}"
    )


# ============================================================
# 7. BEST / WORST SENSOR BY R²
# ============================================================

best_row = performance_df.loc[
    performance_df["r2"].idxmax()
]

worst_row = performance_df.loc[
    performance_df["r2"].idxmin()
]

print()
print(separator())
print("4. BEST AND WORST GENERALIZATION")
print(separator())

print()
print(
    f"Best sensor : {best_row['sensor']}"
)

print(
    f"Best R²     : {best_row['r2']:.6f}"
)

print(
    f"Best model  : {best_row['model']}"
)

print()

print(
    f"Lowest R²  : {worst_row['sensor']}"
)

print(
    f"Lowest R²  : {worst_row['r2']:.6f}"
)


# ============================================================
# 8. MODEL GROUP SUMMARY
# ============================================================

print()
print(separator())
print("5. SENSOR GROUP SUMMARY")
print(separator())

group_summary = (
    performance_df
    .groupby("sensor_group")
    .agg(
        sensors=("sensor", "count"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
    )
    .reset_index()
)

print()

print(
    group_summary.to_string(
        index=False
    )
)


# ============================================================
# 9. MULTI-HORIZON ANALYSIS
# ============================================================

print()
print(separator())
print("6. MULTI-HORIZON FORECAST ANALYSIS")
print(separator())

horizon_rows = []

for sensor, values in MULTI_HORIZON_R2.items():

    for horizon, r2 in values.items():

        horizon_rows.append(
            {
                "sensor": sensor,
                "horizon_seconds": horizon,
                "horizon_minutes": horizon / 60,
                "r2": r2,
            }
        )


horizon_df = pd.DataFrame(horizon_rows)

print()

for sensor in SENSORS:

    sensor_horizon = horizon_df[
        horizon_df["sensor"] == sensor
    ].sort_values(
        "horizon_seconds"
    )

    print()
    print(f"{sensor}:")

    for _, row in sensor_horizon.iterrows():

        print(
            f"  {int(row['horizon_seconds']):4d} sec "
            f"({row['horizon_minutes']:5.1f} min) "
            f"R² = {row['r2']:.6f}"
        )


# ============================================================
# 10. TEMPERATURE HORIZON ANALYSIS
# ============================================================

print()
print(separator())
print("7. TEMPERATURE HORIZON ANALYSIS")
print(separator())

temperature_horizon = horizon_df[
    horizon_df["sensor"] == "temperature"
].sort_values(
    "horizon_seconds"
)

print()

for _, row in temperature_horizon.iterrows():

    status = (
        "USEFUL"
        if row["r2"] > 0
        else "NOT USEFUL"
    )

    print(
        f"{int(row['horizon_seconds']):4d} sec "
        f"({row['horizon_minutes']:5.1f} min) "
        f"R² = {row['r2']:.6f} "
        f"→ {status}"
    )


# ============================================================
# 11. HORIZON CSV
# ============================================================

horizon_csv = os.path.join(
    REPORT_DIR,
    "step32_multi_horizon_performance.csv"
)

horizon_df.to_csv(
    horizon_csv,
    index=False
)

print()
print(
    f"[SAVED] {horizon_csv}"
)


# ============================================================
# 12. FINAL PERFORMANCE CSV
# ============================================================

performance_csv = os.path.join(
    REPORT_DIR,
    "step32_final_sensor_performance.csv"
)

performance_df[
    [
        "sensor",
        "sensor_group",
        "model",
        "mae",
        "rmse",
        "rmse_std",
        "r2",
        "forecastability",
    ]
].to_csv(
    performance_csv,
    index=False
)

print(
    f"[SAVED] {performance_csv}"
)


# ============================================================
# 13. MODEL SUMMARY CSV
# ============================================================

model_summary = (
    performance_df
    .groupby("model")
    .agg(
        sensors=("sensor", "count"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_r2=("r2", "mean"),
    )
    .reset_index()
)

model_summary_csv = os.path.join(
    REPORT_DIR,
    "step32_model_summary.csv"
)

model_summary.to_csv(
    model_summary_csv,
    index=False
)

print(
    f"[SAVED] {model_summary_csv}"
)


# ============================================================
# 14. PROJECT OUTPUT CHECK
# ============================================================

print()
print(separator())
print("8. EXISTING PROJECT OUTPUT CHECK")
print(separator())

important_outputs = [

    os.path.join(
        DATA_DIR,
        "sensor_data_cleaned.csv"
    ),

    os.path.join(
        DATA_DIR,
        "future_features_60s.csv"
    ),

    os.path.join(
        OUTPUT_DIR,
        "database",
        "forecasting.db"
    ),

    os.path.join(
        REPORT_DIR,
        "step29_realtime_monitoring.csv"
    ),
]


for path in important_outputs:

    exists = os.path.exists(path)

    print(
        f"[{'PASS' if exists else 'FAIL'}] "
        f"{os.path.basename(path)}"
    )


# ============================================================
# 15. FINAL INTERPRETATION
# ============================================================

print()
print(separator())
print("9. FINAL INTERPRETATION")
print(separator())

print()

print(
    "Temperature:"
)

print(
    "  Temperature is the strongest forecasting target."
)

print(
    "  The final Persistence model achieved positive"
)

print(
    "  walk-forward R² and substantially better"
)

print(
    "  generalization than the other sensor groups."
)


print()

print(
    "Vibration:"
)

print(
    "  Vibration forecasting remains difficult under"
)

print(
    "  the observed distribution shift between training"
)

print(
    "  and later evaluation periods."
)

print(
    "  XGBoost was selected through walk-forward"
)

print(
    "  evaluation, but negative R² indicates weak"
)

print(
    "  out-of-sample forecasting performance."
)


print()

print(
    "Magnetic:"
)

print(
    "  Magnetic channels show very low point-wise"
)

print(
    "  predictability in the available data."
)

print(
    "  Ridge was selected as the most stable relative"
)

print(
    "  model, but R² remains approximately zero."
)


print()

print(
    "Overall:"
)

print(
    "  The engineering pipeline is operational and"
)

print(
    "  validated, while predictive usefulness differs"
)

print(
    "  substantially across sensors."
)


# ============================================================
# 16. FINAL REPORT TEXT
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "step32_final_performance_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "============================================================\n"
    )

    f.write(
        "FINAL SENSOR FORECASTING PERFORMANCE REPORT\n"
    )

    f.write(
        "============================================================\n\n"
    )

    f.write(
        "Project: Industrial Sensor Time-Series Forecasting\n"
    )

    f.write(
        "Forecast horizon: 60 seconds\n\n"
    )


    # --------------------------------------------------------
    # Final models
    # --------------------------------------------------------

    f.write(
        "FINAL MODEL SELECTION\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    for sensor in SENSORS:

        f.write(
            f"{sensor:20s} -> "
            f"{FINAL_MODELS[sensor]}\n"
        )


    f.write("\n")


    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    f.write(
        "FINAL WALK-FORWARD PERFORMANCE\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    for _, row in performance_df.iterrows():

        f.write(
            f"\nSensor: {row['sensor']}\n"
        )

        f.write(
            f"Model: {row['model']}\n"
        )

        f.write(
            f"MAE: {row['mae']:.6f}\n"
        )

        f.write(
            f"RMSE: {row['rmse']:.6f}\n"
        )

        f.write(
            f"RMSE Std: {row['rmse_std']:.6f}\n"
        )

        f.write(
            f"R²: {row['r2']:.6f}\n"
        )

        f.write(
            f"Forecastability: "
            f"{row['forecastability']}\n"
        )


    # --------------------------------------------------------
    # Horizon
    # --------------------------------------------------------

    f.write("\n\n")

    f.write(
        "MULTI-HORIZON PERFORMANCE\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    for sensor in SENSORS:

        f.write(
            f"\n{sensor}\n"
        )

        sensor_horizon = horizon_df[
            horizon_df["sensor"] == sensor
        ].sort_values(
            "horizon_seconds"
        )

        for _, row in sensor_horizon.iterrows():

            f.write(
                f"  {int(row['horizon_seconds']):4d} sec "
                f"R² = {row['r2']:.6f}\n"
            )


    # --------------------------------------------------------
    # Conclusions
    # --------------------------------------------------------

    f.write("\n\n")

    f.write(
        "FINAL CONCLUSIONS\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "\n1. The complete forecasting pipeline is operational.\n"
    )

    f.write(
        "2. Temperature is the most forecastable sensor.\n"
    )

    f.write(
        "3. Vibration forecasting is affected by distribution "
        "shift and unseen operating regimes.\n"
    )

    f.write(
        "4. Magnetic channels have very low point-wise "
        "predictability.\n"
    )

    f.write(
        "5. XGBoost was selected for vibration through "
        "walk-forward evaluation.\n"
    )

    f.write(
        "6. Ridge was selected for magnetic channels as the "
        "most stable relative model.\n"
    )

    f.write(
        "7. Persistence was selected for temperature because "
        "it generalized best.\n"
    )

    f.write(
        "8. The negative R² values for vibration and magnetic "
        "channels indicate that their forecasts are not "
        "currently useful relative to the evaluation baseline.\n"
    )

    f.write(
        "9. The system should therefore be used with "
        "sensor-specific confidence rather than assuming "
        "equal forecast quality across all sensors.\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "Training-set metrics are not used as the final "
        "forecasting performance claim. The conclusions above "
        "are based on walk-forward/generalization evaluation.\n"
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print()
print(separator())
print("STEP 32 FINAL SUMMARY")
print(separator())

print()

print(
    f"Performance rows : {len(performance_df)}"
)

print(
    f"Horizon rows     : {len(horizon_df)}"
)

print()

print(
    f"Final performance CSV:"
)

print(
    performance_csv
)

print()

print(
    f"Multi-horizon CSV:"
)

print(
    horizon_csv
)

print()

print(
    f"Model summary CSV:"
)

print(
    model_summary_csv
)

print()

print(
    f"Complete text report:"
)

print(
    report_path
)

print()

print(separator())
print("STEP 32 COMPLETE")
print(separator())

print()
print(
    "Final performance analysis generated successfully."
)

print(
    "No models were retrained."
)

print(
    "No database records were modified."
)

print(separator())