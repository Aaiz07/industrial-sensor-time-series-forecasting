# ============================================================
# STEP 33: FINAL VISUALIZATIONS
# ============================================================
# Purpose:
#   Generate professional final plots for the sensor
#   forecasting project.
#
# Outputs:
#   1. Final R2 by sensor
#   2. Final RMSE by sensor
#   3. Final MAE by sensor
#   4. Multi-horizon R2
#   5. Temperature horizon performance
#   6. Sensor-group R2
#   7. Sensor-group RMSE
#
# Input:
#   outputs/reports/step32_final_sensor_performance.csv
#   outputs/reports/step32_multi_horizon_performance.csv
#
# IMPORTANT:
#   - No model retraining
#   - No database modification
#   - Uses existing evaluation results
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

PERFORMANCE_FILE = os.path.join(
    REPORT_DIR,
    "step32_final_sensor_performance.csv"
)

HORIZON_FILE = os.path.join(
    REPORT_DIR,
    "step32_multi_horizon_performance.csv"
)


# ============================================================
# SENSOR ORDER
# ============================================================

SENSOR_ORDER = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]


SENSOR_LABELS = {
    "vib_x_rms": "Vibration X",
    "vib_y_rms": "Vibration Y",
    "vib_z_rms": "Vibration Z",
    "mag_x": "Magnetic X",
    "mag_y": "Magnetic Y",
    "mag_z": "Magnetic Z",
    "temperature": "Temperature",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("STEP 33: FINAL VISUALIZATIONS")
print("=" * 75)

print()
print("Project:")
print(BASE_DIR)

print()


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 75)
print("1. INPUT FILE CHECK")
print("=" * 75)

if not os.path.exists(PERFORMANCE_FILE):

    print("[FAIL] Final performance file not found:")
    print(PERFORMANCE_FILE)

    raise SystemExit


if not os.path.exists(HORIZON_FILE):

    print("[FAIL] Multi-horizon file not found:")
    print(HORIZON_FILE)

    raise SystemExit


print("[PASS] Final performance CSV found")
print("[PASS] Multi-horizon CSV found")


# ============================================================
# LOAD DATA
# ============================================================

performance_df = pd.read_csv(
    PERFORMANCE_FILE
)

horizon_df = pd.read_csv(
    HORIZON_FILE
)


# ============================================================
# SORT DATA
# ============================================================

performance_df["sensor_order"] = (
    performance_df["sensor"]
    .map(
        {
            sensor: i
            for i, sensor in enumerate(SENSOR_ORDER)
        }
    )
)

performance_df = performance_df.sort_values(
    "sensor_order"
).reset_index(drop=True)


horizon_df["sensor_order"] = (
    horizon_df["sensor"]
    .map(
        {
            sensor: i
            for i, sensor in enumerate(SENSOR_ORDER)
        }
    )
)

horizon_df = horizon_df.sort_values(
    [
        "sensor_order",
        "horizon_seconds"
    ]
).reset_index(drop=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):

    path = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")


# ============================================================
# 2. FINAL R2 BY SENSOR
# ============================================================

print()
print("=" * 75)
print("2. FINAL R2 BY SENSOR")
print("=" * 75)

labels = [
    SENSOR_LABELS[sensor]
    for sensor in performance_df["sensor"]
]

r2_values = performance_df["r2"]


plt.figure(
    figsize=(11, 6)
)

plt.bar(
    labels,
    r2_values
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Final Walk-Forward R² by Sensor"
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "R²"
)

plt.xticks(
    rotation=30,
    ha="right"
)

save_plot(
    "step33_final_r2_by_sensor.png"
)


# ============================================================
# 3. FINAL RMSE BY SENSOR
# ============================================================

print()
print("=" * 75)
print("3. FINAL RMSE BY SENSOR")
print("=" * 75)

rmse_values = performance_df["rmse"]


plt.figure(
    figsize=(11, 6)
)

plt.bar(
    labels,
    rmse_values
)

plt.title(
    "Final Walk-Forward RMSE by Sensor"
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "RMSE"
)

plt.xticks(
    rotation=30,
    ha="right"
)

save_plot(
    "step33_final_rmse_by_sensor.png"
)


# ============================================================
# 4. FINAL MAE BY SENSOR
# ============================================================

print()
print("=" * 75)
print("4. FINAL MAE BY SENSOR")
print("=" * 75)

mae_values = performance_df["mae"]


plt.figure(
    figsize=(11, 6)
)

plt.bar(
    labels,
    mae_values
)

plt.title(
    "Final Walk-Forward MAE by Sensor"
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "MAE"
)

plt.xticks(
    rotation=30,
    ha="right"
)

save_plot(
    "step33_final_mae_by_sensor.png"
)


# ============================================================
# 5. MULTI-HORIZON R2 — ALL SENSORS
# ============================================================

print()
print("=" * 75)
print("5. MULTI-HORIZON R2")
print("=" * 75)

plt.figure(
    figsize=(12, 7)
)

for sensor in SENSOR_ORDER:

    sensor_data = horizon_df[
        horizon_df["sensor"] == sensor
    ].sort_values(
        "horizon_seconds"
    )

    if sensor_data.empty:
        continue

    plt.plot(
        sensor_data["horizon_minutes"],
        sensor_data["r2"],
        marker="o",
        label=SENSOR_LABELS[sensor]
    )


plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Forecast Performance Across Horizons"
)

plt.xlabel(
    "Forecast Horizon (minutes)"
)

plt.ylabel(
    "R²"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

save_plot(
    "step33_multi_horizon_r2_all_sensors.png"
)


# ============================================================
# 6. TEMPERATURE HORIZON PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("6. TEMPERATURE HORIZON PERFORMANCE")
print("=" * 75)

temperature_df = horizon_df[
    horizon_df["sensor"] == "temperature"
].sort_values(
    "horizon_seconds"
)


plt.figure(
    figsize=(9, 6)
)

plt.plot(
    temperature_df["horizon_minutes"],
    temperature_df["r2"],
    marker="o",
    linewidth=2
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Temperature Forecastability vs Horizon"
)

plt.xlabel(
    "Forecast Horizon (minutes)"
)

plt.ylabel(
    "R²"
)

plt.grid(
    True,
    alpha=0.3
)

save_plot(
    "step33_temperature_horizon_r2.png"
)


# ============================================================
# 7. SENSOR GROUP SUMMARY
# ============================================================

print()
print("=" * 75)
print("7. SENSOR GROUP PERFORMANCE")
print("=" * 75)

group_summary = (
    performance_df
    .groupby("sensor_group")
    .agg(
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# GROUP R2
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    group_summary["sensor_group"],
    group_summary["mean_r2"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Average Walk-Forward R² by Sensor Group"
)

plt.xlabel(
    "Sensor Group"
)

plt.ylabel(
    "Mean R²"
)

save_plot(
    "step33_sensor_group_r2.png"
)


# ------------------------------------------------------------
# GROUP RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    group_summary["sensor_group"],
    group_summary["mean_rmse"]
)

plt.title(
    "Average Walk-Forward RMSE by Sensor Group"
)

plt.xlabel(
    "Sensor Group"
)

plt.ylabel(
    "Mean RMSE"
)

save_plot(
    "step33_sensor_group_rmse.png"
)


# ============================================================
# 8. MODEL PERFORMANCE SUMMARY
# ============================================================

print()
print("=" * 75)
print("8. MODEL PERFORMANCE SUMMARY")
print("=" * 75)

model_summary = (
    performance_df
    .groupby("model")
    .agg(
        mean_r2=("r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        sensors=("sensor", "count"),
    )
    .reset_index()
)


print()

print(
    model_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# MODEL R2
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    model_summary["model"],
    model_summary["mean_r2"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    "Average Walk-Forward R² by Selected Model"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Mean R²"
)

save_plot(
    "step33_model_r2.png"
)


# ============================================================
# 9. TEMPERATURE USEFUL HORIZON
# ============================================================

print()
print("=" * 75)
print("9. TEMPERATURE USEFUL HORIZON")
print("=" * 75)

useful_temperature = temperature_df[
    temperature_df["r2"] > 0
]

if not useful_temperature.empty:

    maximum_useful_horizon = (
        useful_temperature[
            "horizon_minutes"
        ].max()
    )

    print(
        f"Maximum tested useful temperature horizon: "
        f"{maximum_useful_horizon:.1f} minutes"
    )

else:

    maximum_useful_horizon = None

    print(
        "No tested temperature horizon had positive R²."
    )


# ============================================================
# 10. SAVE GROUP SUMMARY
# ============================================================

group_summary_path = os.path.join(
    REPORT_DIR,
    "step33_sensor_group_summary.csv"
)

group_summary.to_csv(
    group_summary_path,
    index=False
)

print()
print(
    f"[SAVED] {group_summary_path}"
)


# ============================================================
# 11. SAVE MODEL SUMMARY
# ============================================================

model_summary_path = os.path.join(
    REPORT_DIR,
    "step33_model_summary.csv"
)

model_summary.to_csv(
    model_summary_path,
    index=False
)

print(
    f"[SAVED] {model_summary_path}"
)


# ============================================================
# 12. FINAL VISUALIZATION SUMMARY
# ============================================================

plot_files = [
    "step33_final_r2_by_sensor.png",
    "step33_final_rmse_by_sensor.png",
    "step33_final_mae_by_sensor.png",
    "step33_multi_horizon_r2_all_sensors.png",
    "step33_temperature_horizon_r2.png",
    "step33_sensor_group_r2.png",
    "step33_sensor_group_rmse.png",
    "step33_model_r2.png",
]


print()
print("=" * 75)
print("STEP 33 VISUALIZATION SUMMARY")
print("=" * 75)

existing_plots = 0

for filename in plot_files:

    path = os.path.join(
        PLOT_DIR,
        filename
    )

    if os.path.exists(path):

        existing_plots += 1

        print(
            f"[PASS] {filename}"
        )

    else:

        print(
            f"[FAIL] {filename}"
        )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STEP 33 FINAL SUMMARY")
print("=" * 75)

print()

print(
    f"Plots generated : {existing_plots}/{len(plot_files)}"
)

print(
    f"Plot directory  : {PLOT_DIR}"
)

print(
    f"Report directory: {REPORT_DIR}"
)

print()

if existing_plots == len(plot_files):

    print("=" * 75)
    print("FINAL VISUALIZATION CHECK: PASSED")
    print("=" * 75)

    print()
    print(
        "All final project visualizations were generated successfully."
    )

else:

    print("=" * 75)
    print("FINAL VISUALIZATION CHECK: FAILED")
    print("=" * 75)

    print()
    print(
        "One or more visualizations were not generated."
    )


print()
print("=" * 75)
print("STEP 33 COMPLETE")
print("=" * 75)

print()
print(
    "No models were retrained."
)

print(
    "No database records were modified."
)

print(
    "All visualizations are based on the final evaluation results."
)

print("=" * 75)