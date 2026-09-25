# ============================================================
# STEP 22: FINAL ANALYSIS AND VISUALIZATION
# ============================================================
#
# Purpose:
#   Analyze the final forecasting results and create
#   presentation-ready plots and reports.
#
# Main evidence:
#   Step 20 -> Walk-forward validation
#   Step 21 -> Final trained models and predictions
#
# Outputs:
#
#   outputs/reports/
#       step22_final_analysis.csv
#       step22_sensor_summary.csv
#       step22_walk_forward_summary.csv
#       step22_final_report.txt
#
#   outputs/plots/
#       step22_walk_forward_rmse.png
#       step22_walk_forward_r2.png
#       step22_sensor_rmse.png
#       step22_sensor_r2.png
#       step22_<sensor>_actual_vs_predicted.png
#       step22_<sensor>_error_distribution.png
#
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

STEP20_RESULTS = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "step20_walk_forward_results.csv"
)

STEP20_SUMMARY = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "step20_model_summary.csv"
)

STEP20_SELECTION = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "step20_final_model_selection.csv"
)

STEP20_PREDICTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "step20_walk_forward_predictions.csv"
)

STEP21_PREDICTIONS = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "step21_final_predictions.csv"
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

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. SENSOR CONFIGURATION
# ============================================================

TARGETS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

MODEL_SELECTION = {
    "vib_x_rms": "XGBoost",
    "vib_y_rms": "XGBoost",
    "vib_z_rms": "XGBoost",
    "mag_x": "Ridge",
    "mag_y": "Ridge",
    "mag_z": "Ridge",
    "temperature": "Persistence"
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (
            ss_res / ss_tot
        )

    return mae, rmse, r2


def safe_filename(name):

    return (
        name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ============================================================
# 4. START
# ============================================================

print("=" * 75)
print("STEP 22: FINAL ANALYSIS AND VISUALIZATION")
print("=" * 75)


# ============================================================
# 5. CHECK FILES
# ============================================================

print("\n[1/8] Checking input files...")

required_files = [
    STEP20_RESULTS,
    STEP20_SUMMARY,
    STEP20_SELECTION,
    STEP20_PREDICTIONS,
    STEP21_PREDICTIONS
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        f"FOUND: {file_path}"
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n[2/8] Loading results...")

walk_results = pd.read_csv(
    STEP20_RESULTS
)

walk_summary = pd.read_csv(
    STEP20_SUMMARY
)

model_selection = pd.read_csv(
    STEP20_SELECTION
)

walk_predictions = pd.read_csv(
    STEP20_PREDICTIONS
)

final_predictions = pd.read_csv(
    STEP21_PREDICTIONS
)

if "ts" in walk_predictions.columns:

    walk_predictions["ts"] = pd.to_datetime(
        walk_predictions["ts"]
    )

if "ts" in final_predictions.columns:

    final_predictions["ts"] = pd.to_datetime(
        final_predictions["ts"]
    )

print(
    f"Walk-forward result rows: "
    f"{len(walk_results):,}"
)

print(
    f"Walk-forward prediction rows: "
    f"{len(walk_predictions):,}"
)

print(
    f"Final prediction rows: "
    f"{len(final_predictions):,}"
)


# ============================================================
# 7. WALK-FORWARD ANALYSIS
# ============================================================

print("\n[3/8] Analyzing walk-forward performance...")

analysis_rows = []

for sensor in TARGETS:

    sensor_data = walk_summary[
        walk_summary["sensor"] == sensor
    ].copy()

    for _, row in sensor_data.iterrows():

        analysis_rows.append(
            {
                "sensor": sensor,
                "model": row["model"],
                "mean_mae": row["mean_mae"],
                "std_mae": row["std_mae"],
                "mean_rmse": row["mean_rmse"],
                "std_rmse": row["std_rmse"],
                "mean_r2": row["mean_r2"],
                "std_r2": row["std_r2"],
                "folds": row["folds"]
            }
        )

analysis_df = pd.DataFrame(
    analysis_rows
)

analysis_path = os.path.join(
    REPORT_DIR,
    "step22_final_analysis.csv"
)

analysis_df.to_csv(
    analysis_path,
    index=False
)


# ============================================================
# 8. FINAL SENSOR SUMMARY
# ============================================================

print("\n[4/8] Creating sensor summary...")

sensor_summary_rows = []

for sensor in TARGETS:

    selected_model = MODEL_SELECTION[
        sensor
    ]

    selected = analysis_df[
        (
            analysis_df["sensor"]
            == sensor
        )
        &
        (
            analysis_df["model"]
            == selected_model
        )
    ]

    if selected.empty:
        continue

    row = selected.iloc[0]

    sensor_summary_rows.append(
        {
            "sensor": sensor,
            "selected_model":
                selected_model,
            "mean_mae":
                row["mean_mae"],
            "mean_rmse":
                row["mean_rmse"],
            "rmse_std":
                row["std_rmse"],
            "mean_r2":
                row["mean_r2"],
            "r2_std":
                row["std_r2"],
            "folds":
                row["folds"]
        }
    )

sensor_summary = pd.DataFrame(
    sensor_summary_rows
)

sensor_summary_path = os.path.join(
    REPORT_DIR,
    "step22_sensor_summary.csv"
)

sensor_summary.to_csv(
    sensor_summary_path,
    index=False
)


# ============================================================
# 9. WALK-FORWARD SUMMARY
# ============================================================

print("\n[5/8] Creating fold-level summary...")

fold_summary = (
    walk_results
    .groupby(
        [
            "sensor",
            "model"
        ]
    )
    .agg(
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        rmse_std=("rmse", "std"),
        r2_std=("r2", "std")
    )
    .reset_index()
)

fold_summary_path = os.path.join(
    REPORT_DIR,
    "step22_walk_forward_summary.csv"
)

fold_summary.to_csv(
    fold_summary_path,
    index=False
)


# ============================================================
# 10. PLOT 1: WALK-FORWARD RMSE
# ============================================================

print("\n[6/8] Creating visualizations...")


# ------------------------------------------------------------
# Walk-forward RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

for sensor in TARGETS:

    sensor_data = walk_results[
        walk_results["sensor"] == sensor
    ]

    for model in sensor_data[
        "model"
    ].unique():

        model_data = sensor_data[
            sensor_data["model"] == model
        ]

        plt.plot(
            model_data["fold"],
            model_data["rmse"],
            marker="o",
            label=f"{sensor} - {model}"
        )

plt.xlabel(
    "Walk-forward fold"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "Walk-Forward RMSE Across Time"
)

plt.legend(
    fontsize=7,
    ncol=2
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

rmse_plot_path = os.path.join(
    PLOT_DIR,
    "step22_walk_forward_rmse.png"
)

plt.savefig(
    rmse_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# 11. PLOT 2: WALK-FORWARD R2
# ============================================================

plt.figure(
    figsize=(12, 7)
)

for sensor in TARGETS:

    sensor_data = walk_results[
        walk_results["sensor"] == sensor
    ]

    for model in sensor_data[
        "model"
    ].unique():

        model_data = sensor_data[
            sensor_data["model"] == model
        ]

        plt.plot(
            model_data["fold"],
            model_data["r2"],
            marker="o",
            label=f"{sensor} - {model}"
        )

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Walk-forward fold"
)

plt.ylabel(
    "R²"
)

plt.title(
    "Walk-Forward R² Across Time"
)

plt.legend(
    fontsize=7,
    ncol=2
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

r2_plot_path = os.path.join(
    PLOT_DIR,
    "step22_walk_forward_r2.png"
)

plt.savefig(
    r2_plot_path,
    dpi=200
)

plt.close()


# ============================================================
# 12. PLOT 3: SELECTED MODEL RMSE
# ============================================================

selected_rmse = sensor_summary[
    [
        "sensor",
        "selected_model",
        "mean_rmse"
    ]
].copy()

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    selected_rmse["sensor"],
    selected_rmse["mean_rmse"]
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "Mean walk-forward RMSE"
)

plt.title(
    "Selected Model Performance by Sensor"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

sensor_rmse_path = os.path.join(
    PLOT_DIR,
    "step22_sensor_rmse.png"
)

plt.savefig(
    sensor_rmse_path,
    dpi=200
)

plt.close()


# ============================================================
# 13. PLOT 4: SELECTED MODEL R2
# ============================================================

selected_r2 = sensor_summary[
    [
        "sensor",
        "selected_model",
        "mean_r2"
    ]
].copy()

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    selected_r2["sensor"],
    selected_r2["mean_r2"]
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "Mean walk-forward R²"
)

plt.title(
    "Selected Model R² by Sensor"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

sensor_r2_path = os.path.join(
    PLOT_DIR,
    "step22_sensor_r2.png"
)

plt.savefig(
    sensor_r2_path,
    dpi=200
)

plt.close()


# ============================================================
# 14. ACTUAL VS PREDICTED PLOTS
# ============================================================

print(
    "\nCreating actual-vs-predicted plots..."
)

for sensor in TARGETS:

    selected_model = MODEL_SELECTION[
        sensor
    ]

    data = walk_predictions[
        (
            walk_predictions["sensor"]
            == sensor
        )
        &
        (
            walk_predictions["model"]
            == selected_model
        )
    ].copy()

    data = data.sort_values(
        "ts"
    )

    if data.empty:
        continue

    # Use only the latest 2500 observations
    # so the plot remains readable.

    plot_data = data.tail(
        min(2500, len(data))
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        plot_data["ts"],
        plot_data["actual"],
        label="Actual"
    )

    plt.plot(
        plot_data["ts"],
        plot_data["prediction"],
        label="Predicted"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        sensor
    )

    plt.title(
        f"{sensor}: Actual vs Predicted "
        f"({selected_model})"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"step22_"
        f"{safe_filename(sensor)}"
        f"_actual_vs_predicted.png"
    )

    output_path = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# 15. ERROR DISTRIBUTION PLOTS
# ============================================================

print(
    "Creating error-distribution plots..."
)

for sensor in TARGETS:

    selected_model = MODEL_SELECTION[
        sensor
    ]

    data = walk_predictions[
        (
            walk_predictions["sensor"]
            == sensor
        )
        &
        (
            walk_predictions["model"]
            == selected_model
        )
    ].copy()

    if data.empty:
        continue

    errors = (
        data["actual"]
        - data["prediction"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        errors,
        bins=60
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Prediction error"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{sensor}: Prediction Error Distribution"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    filename = (
        f"step22_"
        f"{safe_filename(sensor)}"
        f"_error_distribution.png"
    )

    output_path = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# 16. FINAL REPORT GENERATION
# ============================================================

print("\n[7/8] Generating final report...")


report_lines = []

report_lines.append(
    "============================================================"
)

report_lines.append(
    "TIME-SERIES SENSOR FORECASTING PROJECT"
)

report_lines.append(
    "FINAL ANALYSIS REPORT"
)

report_lines.append(
    "============================================================"
)

report_lines.append("")

report_lines.append(
    "Forecast horizon: 60 seconds"
)

report_lines.append(
    "Validation method: 5-fold chronological walk-forward validation"
)

report_lines.append(
    "Final models selected from Step 20"
)

report_lines.append("")


# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

report_lines.append(
    "FINAL MODEL SELECTION"
)

report_lines.append(
    "------------------------------------------------------------"
)

for sensor in TARGETS:

    report_lines.append(
        f"{sensor:15s} -> "
        f"{MODEL_SELECTION[sensor]}"
    )

report_lines.append("")


# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

report_lines.append(
    "WALK-FORWARD PERFORMANCE"
)

report_lines.append(
    "------------------------------------------------------------"
)

for _, row in sensor_summary.iterrows():

    report_lines.append(
        f"\nSensor: {row['sensor']}"
    )

    report_lines.append(
        f"Model: {row['selected_model']}"
    )

    report_lines.append(
        f"Mean MAE: {row['mean_mae']:.6f}"
    )

    report_lines.append(
        f"Mean RMSE: {row['mean_rmse']:.6f}"
    )

    report_lines.append(
        f"RMSE std: {row['rmse_std']:.6f}"
    )

    report_lines.append(
        f"Mean R2: {row['mean_r2']:.6f}"
    )

    report_lines.append(
        f"R2 std: {row['r2_std']:.6f}"
    )


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

report_lines.append("")

report_lines.append(
    "============================================================"
)

report_lines.append(
    "FINAL INTERPRETATION"
)

report_lines.append(
    "============================================================"
)

report_lines.append("")

report_lines.append(
    "1. Temperature"
)

report_lines.append(
    "Persistence is the strongest forecasting strategy for "
    "temperature at the 60-second horizon."
)

report_lines.append("")

report_lines.append(
    "2. Magnetic sensors"
)

report_lines.append(
    "Ridge provides substantially lower RMSE than persistence, "
    "but the walk-forward R2 remains close to zero. Therefore "
    "the magnetic channels have limited predictable pointwise "
    "variation at this horizon."
)

report_lines.append("")

report_lines.append(
    "3. Vibration sensors"
)

report_lines.append(
    "XGBoost has the lowest average RMSE among the tested "
    "strategies, but its walk-forward R2 remains strongly "
    "negative. Therefore vibration forecasting should not be "
    "described as reliable point prediction under the current "
    "data and evaluation setup."
)

report_lines.append("")

report_lines.append(
    "4. Generalization"
)

report_lines.append(
    "The difference between training performance and "
    "walk-forward performance demonstrates that temporal "
    "distribution shift is a major challenge in this dataset."
)

report_lines.append("")

report_lines.append(
    "5. Model complexity"
)

report_lines.append(
    "Increasing model complexity did not consistently solve "
    "the forecasting problem. The persistence baseline remains "
    "strong for slow-moving temperature data."
)

report_lines.append("")

report_lines.append(
    "6. Overall conclusion"
)

report_lines.append(
    "The project successfully established a leakage-controlled "
    "60-second forecasting pipeline and evaluated model "
    "generalization across multiple chronological windows."
)

report_lines.append("")

report_lines.append(
    "IMPORTANT:"
)

report_lines.append(
    "The Step 21 training metrics must not be presented as "
    "unbiased forecasting accuracy. The Step 20 walk-forward "
    "results are the primary evidence of generalization."
)

report_lines.append("")


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

report_lines.append(
    "============================================================"
)

report_lines.append(
    "OUTPUT FILES"
)

report_lines.append(
    "============================================================"
)

report_lines.append(
    f"Analysis CSV:\n{analysis_path}"
)

report_lines.append(
    f"Sensor summary:\n{sensor_summary_path}"
)

report_lines.append(
    f"Walk-forward summary:\n{fold_summary_path}"
)

report_lines.append(
    f"Plots:\n{PLOT_DIR}"
)


report_text = "\n".join(
    report_lines
)

report_path = os.path.join(
    REPORT_DIR,
    "step22_final_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report_text
    )


# ============================================================
# 17. PRINT FINAL TABLE
# ============================================================

print("\n[8/8] Final analysis complete.")

print(
    "\n" + "=" * 75
)

print(
    "FINAL SENSOR PERFORMANCE"
)

print(
    "=" * 75
)

print(
    sensor_summary.to_string(
        index=False
    )
)


# ============================================================
# 18. OUTPUT LIST
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FILES CREATED"
)

print(
    "=" * 75
)

print(
    f"\n1. {analysis_path}"
)

print(
    f"2. {sensor_summary_path}"
)

print(
    f"3. {fold_summary_path}"
)

print(
    f"4. {report_path}"
)

print(
    f"5. {rmse_plot_path}"
)

print(
    f"6. {r2_plot_path}"
)

print(
    f"7. {sensor_rmse_path}"
)

print(
    f"8. {sensor_r2_path}"
)

print(
    f"\nAll sensor plots saved in:"
)

print(
    PLOT_DIR
)


# ============================================================
# 19. COMPLETION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 22 COMPLETE"
)

print(
    "=" * 75
)

print(
    """
The forecasting results have now been converted into
analysis tables, visualizations, and a final report.

Do NOT change the models based only on the Step 21
training metrics.

The next stage is to inspect the generated plots and
final report and then build the real-time forecasting
pipeline.
"""
)

print("\nDone.")