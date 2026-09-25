import os
import pandas as pd


# ============================================================
# STEP 34 - FINAL PROJECT DOCUMENTATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)


print("=" * 80)
print("STEP 34 - FINAL PROJECT DOCUMENTATION")
print("=" * 80)


# ============================================================
# FILE PATHS
# ============================================================

performance_file = os.path.join(
    REPORT_DIR,
    "step32_final_sensor_performance.csv"
)

horizon_file = os.path.join(
    REPORT_DIR,
    "step32_multi_horizon_performance.csv"
)

group_file = os.path.join(
    REPORT_DIR,
    "step33_sensor_group_summary.csv"
)

model_file = os.path.join(
    REPORT_DIR,
    "step33_model_summary.csv"
)

output_file = os.path.join(
    REPORT_DIR,
    "FINAL_PROJECT_DOCUMENTATION.md"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    performance_file,
    horizon_file,
    group_file,
    model_file,
]

print("\nChecking required report files...")

for file_path in required_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        "[PASS]",
        os.path.basename(file_path)
    )


# ============================================================
# LOAD REPORTS
# ============================================================

performance = pd.read_csv(performance_file)
horizon = pd.read_csv(horizon_file)
groups = pd.read_csv(group_file)
models = pd.read_csv(model_file)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

# This prevents errors such as:
# MAE vs mae
# RMSE vs rmse
# R2 vs r2

for df in [
    performance,
    horizon,
    groups,
    models
]:

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]


print("\nLoaded report columns:")

print(
    "Performance:",
    list(performance.columns)
)

print(
    "Horizon:",
    list(horizon.columns)
)

print(
    "Groups:",
    list(groups.columns)
)

print(
    "Models:",
    list(models.columns)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt(value, digits=6):

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):.{digits}f}"

    except Exception:

        return str(value)


def get_value(
    row,
    possible_names,
    default="N/A"
):

    for name in possible_names:

        if name in row.index:

            value = row[name]

            if not pd.isna(value):
                return value

    return default


# ============================================================
# FINAL SENSOR PERFORMANCE TABLE
# ============================================================

performance_table_lines = []

performance_table_lines.append(
    "| Sensor | Final Model | MAE | RMSE | RMSE Std | R² |"
)

performance_table_lines.append(
    "|---|---|---:|---:|---:|---:|"
)


for _, row in performance.iterrows():

    sensor = get_value(
        row,
        ["sensor"]
    )

    model = get_value(
        row,
        [
            "model",
            "selected_model",
            "final_model"
        ]
    )

    mae = get_value(
        row,
        [
            "mae",
            "mean_mae"
        ]
    )

    rmse = get_value(
        row,
        [
            "rmse",
            "mean_rmse"
        ]
    )

    rmse_std = get_value(
        row,
        [
            "rmse_std",
            "std_rmse",
            "rmse_standard_deviation"
        ]
    )

    r2 = get_value(
        row,
        [
            "r2",
            "mean_r2",
            "r_squared"
        ]
    )

    line = (
        f"| {sensor} "
        f"| {model} "
        f"| {fmt(mae)} "
        f"| {fmt(rmse)} "
        f"| {fmt(rmse_std)} "
        f"| {fmt(r2)} |"
    )

    performance_table_lines.append(line)


performance_table = "\n".join(
    performance_table_lines
)


# ============================================================
# MULTI-HORIZON TABLE
# ============================================================

horizon_table_lines = []

horizon_table_lines.append(
    "| Sensor | Horizon | Model | R² |"
)

horizon_table_lines.append(
    "|---|---:|---|---:|"
)


for _, row in horizon.iterrows():

    sensor = get_value(
        row,
        ["sensor"]
    )

    horizon_seconds = get_value(
        row,
        [
            "horizon_seconds",
            "horizon",
            "forecast_horizon_seconds"
        ]
    )

    model = get_value(
        row,
        [
            "model",
            "selected_model",
            "best_model"
        ]
    )

    r2 = get_value(
        row,
        [
            "r2",
            "mean_r2",
            "test_r2",
            "r_squared"
        ]
    )

    try:

        horizon_text = (
            f"{int(float(horizon_seconds))} sec"
        )

    except Exception:

        horizon_text = str(
            horizon_seconds
        )

    line = (
        f"| {sensor} "
        f"| {horizon_text} "
        f"| {model} "
        f"| {fmt(r2)} |"
    )

    horizon_table_lines.append(line)


horizon_table = "\n".join(
    horizon_table_lines
)


# ============================================================
# SENSOR GROUP SUMMARY TABLE
# ============================================================

group_table_lines = []

group_table_lines.append(
    "| Sensor Group | MAE | RMSE | R² |"
)

group_table_lines.append(
    "|---|---:|---:|---:|"
)


for _, row in groups.iterrows():

    group_name = get_value(
        row,
        [
            "sensor_group",
            "group",
            "category"
        ]
    )

    mae = get_value(
        row,
        [
            "mae",
            "mean_mae"
        ]
    )

    rmse = get_value(
        row,
        [
            "rmse",
            "mean_rmse"
        ]
    )

    r2 = get_value(
        row,
        [
            "r2",
            "mean_r2",
            "r_squared"
        ]
    )

    line = (
        f"| {group_name} "
        f"| {fmt(mae)} "
        f"| {fmt(rmse)} "
        f"| {fmt(r2)} |"
    )

    group_table_lines.append(line)


group_table = "\n".join(
    group_table_lines
)


# ============================================================
# MODEL SUMMARY TABLE
# ============================================================

model_table_lines = []

model_table_lines.append(
    "| Model | Sensor / Group | MAE | RMSE | R² |"
)

model_table_lines.append(
    "|---|---|---:|---:|---:|"
)


for _, row in models.iterrows():

    model = get_value(
        row,
        [
            "model",
            "selected_model"
        ]
    )

    sensor = get_value(
        row,
        [
            "sensor",
            "sensor_group",
            "group"
        ],
        default="Summary"
    )

    mae = get_value(
        row,
        [
            "mae",
            "mean_mae"
        ]
    )

    rmse = get_value(
        row,
        [
            "rmse",
            "mean_rmse"
        ]
    )

    r2 = get_value(
        row,
        [
            "r2",
            "mean_r2",
            "r_squared"
        ]
    )

    line = (
        f"| {model} "
        f"| {sensor} "
        f"| {fmt(mae)} "
        f"| {fmt(rmse)} "
        f"| {fmt(r2)} |"
    )

    model_table_lines.append(line)


model_table = "\n".join(
    model_table_lines
)


# ============================================================
# BUILD DOCUMENTATION
# ============================================================

sections = []


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

sections.append(
    "# Industrial Sensor Time-Series Forecasting System"
)


# ------------------------------------------------------------
# 1
# ------------------------------------------------------------

sections.append(
    "## 1. Project Overview\n\n"
    "This project develops an end-to-end industrial sensor "
    "time-series forecasting system.\n\n"
    "The system performs data validation, cleaning, exploratory "
    "analysis, feature engineering, future-target construction, "
    "model training, chronological validation, multi-horizon "
    "forecasting, real-time prediction, database storage, "
    "forecast monitoring, visualization, and system validation."
)


# ------------------------------------------------------------
# 2
# ------------------------------------------------------------

sections.append(
    "## 2. Problem Statement\n\n"
    "The objective of the project is to determine whether "
    "historical industrial sensor measurements contain enough "
    "temporal information to predict future sensor values.\n\n"
    "A major goal is not only to train forecasting models, but "
    "also to determine which sensor signals are genuinely "
    "forecastable under realistic future conditions."
)


# ------------------------------------------------------------
# 3
# ------------------------------------------------------------

sections.append(
    "## 3. Sensor Channels\n\n"
    "The project contains seven sensor channels:\n\n"
    "- vib_x_rms\n"
    "- vib_y_rms\n"
    "- vib_z_rms\n"
    "- mag_x\n"
    "- mag_y\n"
    "- mag_z\n"
    "- temperature"
)


# ------------------------------------------------------------
# 4
# ------------------------------------------------------------

sections.append(
    "## 4. Dataset\n\n"
    "The original dataset contains 59,230 observations.\n\n"
    "The original columns are:\n\n"
    "- ts\n"
    "- vib_x_rms\n"
    "- vib_y_rms\n"
    "- vib_z_rms\n"
    "- mag_x\n"
    "- mag_y\n"
    "- mag_z\n"
    "- temperature\n\n"
    "After duplicate timestamp handling, the cleaned dataset "
    "contains 59,195 observations."
)


# ------------------------------------------------------------
# 5
# ------------------------------------------------------------

sections.append(
    "## 5. Data Quality Analysis\n\n"
    "The data-quality pipeline evaluates:\n\n"
    "- Timestamp validity\n"
    "- Duplicate timestamps\n"
    "- Missing values\n"
    "- Sampling intervals\n"
    "- Temporal gaps\n"
    "- Sensor distributions\n"
    "- Extreme sensor measurements\n\n"
    "Duplicate timestamps were aggregated using sensor-wise "
    "means rather than arbitrarily deleting measurements."
)


# ------------------------------------------------------------
# 6
# ------------------------------------------------------------

sections.append(
    "## 6. Sampling and Time Regularization\n\n"
    "The raw data has irregular sampling intervals. The median "
    "sampling interval is approximately one second.\n\n"
    "A one-second regularized timeline was investigated. The "
    "regularized timeline contains 86,399 timestamps.\n\n"
    "The analysis identified a major outage in the data. Large "
    "outages were preserved rather than blindly interpolated."
)


# ------------------------------------------------------------
# 7
# ------------------------------------------------------------

sections.append(
    "## 7. Temporal Resolution Analysis\n\n"
    "Different temporal resolutions were investigated because "
    "sensor predictability can depend on the aggregation level.\n\n"
    "The analysis showed that vibration, magnetic, and temperature "
    "signals have very different temporal characteristics.\n\n"
    "Aggregation can reduce noise and RMSE, but lower RMSE alone "
    "does not prove that a sensor is genuinely predictable."
)


# ------------------------------------------------------------
# 8
# ------------------------------------------------------------

sections.append(
    "## 8. Feature Engineering\n\n"
    "The final forecasting pipeline uses 146 historical "
    "features.\n\n"
    "Important feature groups include:\n\n"
    "- Historical lag features\n"
    "- First-difference features\n"
    "- Cross-axis vibration features\n"
    "- Magnetic lag features\n"
    "- Rolling means\n"
    "- Rolling standard deviations\n"
    "- Time-of-day sine and cosine features\n\n"
    "Rolling features use shifted historical values so that "
    "future observations cannot leak into the feature set."
)


# ------------------------------------------------------------
# 9
# ------------------------------------------------------------

sections.append(
    "## 9. Target Alignment\n\n"
    "An important discovery during the project was the difference "
    "between current-value prediction and true future forecasting.\n\n"
    "Early experiments produced extremely high R² scores because "
    "the target represented the current timestamp rather than a "
    "future timestamp.\n\n"
    "The issue was diagnosed and corrected before final model "
    "evaluation."
)


# ------------------------------------------------------------
# 10
# ------------------------------------------------------------

sections.append(
    "## 10. True Future Targets\n\n"
    "True future targets were created using exact timestamp "
    "alignment.\n\n"
    "For the primary 60-second forecasting problem, each feature "
    "timestamp is matched with the sensor observation exactly "
    "60 seconds in the future.\n\n"
    "The resulting future-target dataset contains 41,881 matched "
    "observations."
)


# ------------------------------------------------------------
# 11
# ------------------------------------------------------------

sections.append(
    "## 11. Leakage Prevention\n\n"
    "The final forecasting workflow includes explicit leakage "
    "checks.\n\n"
    "The checks verify that:\n\n"
    "- Current raw target values are not used as predictor features\n"
    "- Lag-1 represents historical information\n"
    "- Rolling features use shifted data\n"
    "- Future target columns are excluded from model inputs\n"
    "- Target timestamps occur after feature timestamps"
)


# ------------------------------------------------------------
# 12
# ------------------------------------------------------------

sections.append(
    "## 12. Models Evaluated\n\n"
    "The project investigated several forecasting approaches:\n\n"
    "- Persistence baseline\n"
    "- Moving-average baselines\n"
    "- Ridge Regression\n"
    "- XGBoost\n"
    "- LSTM\n\n"
    "More complex models were not automatically assumed to be "
    "better. Baselines were retained throughout the evaluation."
)


# ------------------------------------------------------------
# 13
# ------------------------------------------------------------

sections.append(
    "## 13. Chronological Validation\n\n"
    "Random train-test splitting was avoided because it can "
    "produce misleading results for time-series forecasting.\n\n"
    "The project uses chronological splitting and later performs "
    "five-fold walk-forward validation for final model selection."
)


# ------------------------------------------------------------
# 14
# ------------------------------------------------------------

sections.append(
    "## 14. Final Model Selection\n\n"
    "Walk-forward validation selected the following models:\n\n"
    "- vib_x_rms -> XGBoost\n"
    "- vib_y_rms -> XGBoost\n"
    "- vib_z_rms -> XGBoost\n"
    "- mag_x -> Ridge\n"
    "- mag_y -> Ridge\n"
    "- mag_z -> Ridge\n"
    "- temperature -> Persistence\n\n"
    "A selected model should not automatically be interpreted as "
    "a highly accurate model. Selection means that it performed "
    "best relative to the alternatives under the validation "
    "procedure."
)


# ------------------------------------------------------------
# 15
# ------------------------------------------------------------

sections.append(
    "## 15. Final Walk-Forward Performance\n\n"
    "The following results represent the primary generalization "
    "performance of the final forecasting system.\n\n"
    + performance_table
)


# ------------------------------------------------------------
# 16
# ------------------------------------------------------------

sections.append(
    "## 16. Sensor Group Performance\n\n"
    + group_table
)


# ------------------------------------------------------------
# 17
# ------------------------------------------------------------

sections.append(
    "## 17. Model Summary\n\n"
    + model_table
)


# ------------------------------------------------------------
# 18
# ------------------------------------------------------------

sections.append(
    "## 18. Distribution Shift\n\n"
    "The project identified significant temporal distribution "
    "shift between training, validation, and test periods.\n\n"
    "The vibration test period is substantially quieter and more "
    "stable than earlier portions of the dataset.\n\n"
    "Temperature also operates in a different range during the "
    "test period.\n\n"
    "This distribution shift is a major reason why vibration "
    "models that perform well during training can generalize "
    "poorly to future operating regimes."
)


# ------------------------------------------------------------
# 19
# ------------------------------------------------------------

sections.append(
    "## 19. Vibration Forecasting\n\n"
    "XGBoost was selected for all three vibration channels through "
    "walk-forward validation.\n\n"
    "However, the final walk-forward R² values remain negative.\n\n"
    "Therefore, the vibration models should be interpreted as "
    "weak forecasting models under unseen operating regimes, "
    "rather than as reliable high-accuracy predictors."
)


# ------------------------------------------------------------
# 20
# ------------------------------------------------------------

sections.append(
    "## 20. Magnetic Forecasting\n\n"
    "Ridge Regression was selected for the three magnetic "
    "channels.\n\n"
    "The magnetic R² values remain approximately zero, indicating "
    "that the available historical features provide very limited "
    "point-forecasting information.\n\n"
    "Magnetic predictions are therefore treated as low-confidence "
    "outputs in the monitoring system."
)


# ------------------------------------------------------------
# 21
# ------------------------------------------------------------

sections.append(
    "## 21. Temperature Forecasting\n\n"
    "Temperature is the strongest forecastable sensor in the "
    "project.\n\n"
    "Persistence provides strong performance because temperature "
    "changes gradually over short forecasting horizons.\n\n"
    "The final walk-forward temperature R² is approximately "
    "0.817, demonstrating useful future predictive performance."
)


# ------------------------------------------------------------
# 22
# ------------------------------------------------------------

sections.append(
    "## 22. Multi-Horizon Forecasting\n\n"
    "The project evaluates exact future timestamps at:\n\n"
    "- 60 seconds\n"
    "- 300 seconds\n"
    "- 600 seconds\n"
    "- 1800 seconds\n"
    "- 3600 seconds\n\n"
    "Temperature remains useful through the tested 10-minute "
    "horizon, while its performance becomes poor at the tested "
    "30-minute and 60-minute horizons.\n\n"
    + horizon_table
)


# ------------------------------------------------------------
# 23
# ------------------------------------------------------------

sections.append(
    "## 23. Forecastability Conclusion\n\n"
    "The project demonstrates that forecastability is "
    "sensor-specific.\n\n"
    "**Temperature:** Strong short-term forecastability.\n\n"
    "**Vibration:** Weak generalization under regime shift.\n\n"
    "**Magnetic:** Very limited point forecastability with the "
    "available data.\n\n"
    "This is an important result because a professional "
    "forecasting project should identify when a signal cannot be "
    "reliably predicted rather than forcing every sensor to show "
    "positive model performance."
)


# ------------------------------------------------------------
# 24
# ------------------------------------------------------------

sections.append(
    "## 24. Final Model Training\n\n"
    "After walk-forward model selection, the selected model for "
    "each sensor was trained using the available labeled "
    "forecasting data and saved as a deployment artifact.\n\n"
    "Training-set metrics are not used as unbiased forecasting "
    "performance estimates. Walk-forward results remain the main "
    "generalization metrics."
)


# ------------------------------------------------------------
# 25
# ------------------------------------------------------------

sections.append(
    "## 25. Real-Time Forecasting\n\n"
    "The real-time forecasting pipeline reconstructs the exact "
    "146-feature schema required by the final model artifacts.\n\n"
    "The service generates forecasts 60 seconds ahead for all "
    "seven sensors."
)


# ------------------------------------------------------------
# 26
# ------------------------------------------------------------

sections.append(
    "## 26. Simulated Real-Time Forecasting\n\n"
    "Historical observations are replayed sequentially to simulate "
    "a real-time sensor stream.\n\n"
    "This verifies that the deployment pipeline can repeatedly "
    "construct features and generate future predictions."
)


# ------------------------------------------------------------
# 27
# ------------------------------------------------------------

sections.append(
    "## 27. Prediction Database\n\n"
    "Forecasts are stored in a SQLite database located at:\n\n"
    "`outputs/database/forecasting.db`\n\n"
    "The `forecast_predictions` table stores one record per "
    "sensor forecast.\n\n"
    "Important fields include:\n\n"
    "- feature_timestamp\n"
    "- forecast_timestamp\n"
    "- horizon_seconds\n"
    "- sensor\n"
    "- model\n"
    "- current_value\n"
    "- predicted_value\n"
    "- change_from_current\n"
    "- created_at"
)


# ------------------------------------------------------------
# 28
# ------------------------------------------------------------

sections.append(
    "## 28. Prediction Monitoring\n\n"
    "Forecasts are passed through a monitoring layer.\n\n"
    "Vibration monitoring uses warning and critical thresholds.\n\n"
    "Temperature monitoring uses temperature thresholds.\n\n"
    "Magnetic predictions are explicitly marked as low-confidence "
    "because their forecasting R² is approximately zero. Large "
    "magnetic percentage changes are therefore not automatically "
    "treated as reliable anomaly alerts."
)


# ------------------------------------------------------------
# 29
# ------------------------------------------------------------

sections.append(
    "## 29. Streamlit Dashboard\n\n"
    "A Streamlit dashboard provides an interactive interface for "
    "viewing forecast predictions and monitoring results.\n\n"
    "The dashboard reads directly from the SQLite forecasting "
    "database."
)


# ------------------------------------------------------------
# 30
# ------------------------------------------------------------

sections.append(
    "## 30. End-to-End Validation\n\n"
    "The complete system was validated after integrating the "
    "forecasting service, database, and monitoring pipeline.\n\n"
    "The end-to-end validation completed successfully and verified "
    "consistency between forecasting and monitoring records."
)


# ------------------------------------------------------------
# 31
# ------------------------------------------------------------

sections.append(
    "## 31. Final System Health Check\n\n"
    "The final system health check verifies model artifacts, "
    "reports, database tables, predictions, monitoring outputs, "
    "and project components.\n\n"
    "The temperature Persistence model intentionally does not "
    "require a trained machine-learning model object because the "
    "latest observation itself is used as the forecast."
)


# ------------------------------------------------------------
# 32
# ------------------------------------------------------------

sections.append(
    "## 32. System Architecture\n\n"
    "```text\n"
    "Raw Sensor Data\n"
    "       |\n"
    "       v\n"
    "Data Validation\n"
    "       |\n"
    "       v\n"
    "Cleaning & EDA\n"
    "       |\n"
    "       v\n"
    "Sampling / Gap Analysis\n"
    "       |\n"
    "       v\n"
    "Feature Engineering\n"
    "       |\n"
    "       v\n"
    "Future Target Construction\n"
    "       |\n"
    "       v\n"
    "Leakage Validation\n"
    "       |\n"
    "       v\n"
    "Model Training\n"
    "       |\n"
    "       v\n"
    "Walk-Forward Validation\n"
    "       |\n"
    "       v\n"
    "Final Model Selection\n"
    "       |\n"
    "       v\n"
    "Real-Time Forecasting\n"
    "       |\n"
    "       v\n"
    "SQLite Prediction Database\n"
    "       |\n"
    "       v\n"
    "Prediction Monitoring\n"
    "       |\n"
    "       v\n"
    "Streamlit Dashboard\n"
    "```"
)


# ------------------------------------------------------------
# 33
# ------------------------------------------------------------

sections.append(
    "## 33. Professional Project Workflow\n\n"
    "```text\n"
    "REAL-WORLD PROBLEM\n"
    "       |\n"
    "       v\n"
    "DEFINE TARGET\n"
    "       |\n"
    "       v\n"
    "DATA\n"
    "       |\n"
    "       v\n"
    "CLEANING + EDA\n"
    "       |\n"
    "       v\n"
    "PREPROCESSING\n"
    "       |\n"
    "       v\n"
    "TRAIN / VALIDATION / TEST\n"
    "       |\n"
    "       v\n"
    "BASELINE\n"
    "       |\n"
    "       v\n"
    "MODEL SELECTION\n"
    "       |\n"
    "       v\n"
    "TRAINING\n"
    "       |\n"
    "       v\n"
    "VALIDATION\n"
    "       |\n"
    "       v\n"
    "ERROR ANALYSIS\n"
    "       |\n"
    "       v\n"
    "IMPROVEMENT\n"
    "       |\n"
    "       v\n"
    "FINAL MODEL\n"
    "       |\n"
    "       v\n"
    "DEPLOYMENT\n"
    "```"
)


# ------------------------------------------------------------
# 34
# ------------------------------------------------------------

sections.append(
    "## 34. Key Lessons\n\n"
    "The main lessons from this project are:\n\n"
    "1. High training accuracy does not guarantee future "
    "forecasting accuracy.\n\n"
    "2. Target alignment must be explicitly validated in "
    "time-series problems.\n\n"
    "3. Random splitting can produce misleading forecasting "
    "results.\n\n"
    "4. Persistence is an important forecasting baseline.\n\n"
    "5. More complex models do not automatically solve "
    "distribution shift.\n\n"
    "6. Different sensors can have fundamentally different "
    "forecastability.\n\n"
    "7. Negative R² is a meaningful result when a model fails to "
    "beat a simple baseline.\n\n"
    "8. Deployment monitoring should account for model reliability."
)


# ------------------------------------------------------------
# 35
# ------------------------------------------------------------

sections.append(
    "## 35. Project Limitations\n\n"
    "- The dataset covers approximately one day.\n"
    "- Operating regimes vary significantly through the day.\n"
    "- Vibration generalization is weak under regime shift.\n"
    "- Magnetic channels have very low point forecastability.\n"
    "- External machine operating variables are unavailable.\n"
    "- Current forecasts are point predictions rather than "
    "probabilistic forecasts."
)


# ------------------------------------------------------------
# 36
# ------------------------------------------------------------

sections.append(
    "## 36. Future Improvements\n\n"
    "Future versions of the system could include:\n\n"
    "- Longer historical datasets\n"
    "- Multiple days or months of operation\n"
    "- Machine RPM\n"
    "- Machine load\n"
    "- Operating-state information\n"
    "- Maintenance events\n"
    "- Regime-specific forecasting models\n"
    "- Concept-drift detection\n"
    "- Online retraining\n"
    "- Probabilistic forecasting\n"
    "- Prediction intervals\n"
    "- Additional industrial context features"
)


# ------------------------------------------------------------
# 37
# ------------------------------------------------------------

sections.append(
    "## 37. Important Project Components\n\n"
    "```text\n"
    "01_data_pipeline.py\n"
    "02_train_models.py\n"
    "03_forecast_and_evaluate.py\n"
    "03_lag_diagnostics.py\n"
    "04_time_regularization.py\n"
    "05_validate_regularization.py\n"
    "06_temporal_resolution.py\n"
    "07_feature_predictability.py\n"
    "08_horizon_analysis.py\n"
    "09_final_feature_strategy.py\n"
    "10_final_model_training.py\n"
    "11_validate_target_alignment.py\n"
    "12_create_future_targets.py\n"
    "13_train_true_forecasting_models.py\n"
    "14_error_distribution_analysis.py\n"
    "15_regime_analysis.py\n"
    "16_regime_aware_forecasting.py\n"
    "17_multi_horizon_forecasting.py\n"
    "18_robust_regime_evaluation.py\n"
    "19_improve_features_forecasting.py\n"
    "20_walk_forward_final_selection.py\n"
    "21_final_model_training.py\n"
    "22_final_analysis_and_visualization.py\n"
    "23_realtime_forecasting.py\n"
    "24_simulated_realtime.py\n"
    "25_prediction_database.py\n"
    "26_prediction_monitoring.py\n"
    "27_forecasting_dashboard.py\n"
    "28_realtime_forecasting_service.py\n"
    "29_realtime_monitoring_service.py\n"
    "30_end_to_end_validation.py\n"
    "31_final_system_health_check.py\n"
    "32_final_performance_report.py\n"
    "33_final_visualizations.py\n"
    "34_final_project_documentation.py\n"
    "```"
)


# ------------------------------------------------------------
# 38
# ------------------------------------------------------------

sections.append(
    "## 38. Output Structure\n\n"
    "```text\n"
    "outputs/\n"
    "|\n"
    "+-- data/\n"
    "|\n"
    "+-- forecasts/\n"
    "|\n"
    "+-- models/\n"
    "|\n"
    "+-- plots/\n"
    "|\n"
    "+-- reports/\n"
    "|\n"
    "+-- database/\n"
    "    |\n"
    "    +-- forecasting.db\n"
    "```"
)


# ------------------------------------------------------------
# 39
# ------------------------------------------------------------

sections.append(
    "## 39. Final Project Status\n\n"
    "**Data pipeline:** Complete\n\n"
    "**Feature engineering:** Complete\n\n"
    "**Target alignment validation:** Complete\n\n"
    "**True forecasting models:** Complete\n\n"
    "**Walk-forward validation:** Complete\n\n"
    "**Final model training:** Complete\n\n"
    "**Real-time forecasting:** Complete\n\n"
    "**SQLite database:** Complete\n\n"
    "**Prediction monitoring:** Complete\n\n"
    "**Streamlit dashboard:** Complete\n\n"
    "**End-to-end validation:** Passed\n\n"
    "**System health validation:** Passed\n\n"
    "**Performance reporting:** Complete\n\n"
    "**Final visualizations:** Complete\n\n"
    "**Final documentation:** Complete"
)


# ------------------------------------------------------------
# 40
# ------------------------------------------------------------

sections.append(
    "## 40. Final Conclusion\n\n"
    "This project demonstrates a complete industrial sensor "
    "time-series forecasting workflow from raw data to an "
    "operational forecasting and monitoring system.\n\n"
    "The strongest forecasting result is temperature, which shows "
    "useful short-term predictability and remains useful through "
    "the tested 10-minute horizon.\n\n"
    "Magnetic channels demonstrate very limited point "
    "forecastability, while vibration forecasting is strongly "
    "affected by temporal regime shift.\n\n"
    "These results demonstrate an important machine-learning "
    "principle: the objective of a forecasting project is not to "
    "force every model to produce a high accuracy score. The goal "
    "is to measure genuine future predictive ability, understand "
    "model limitations, and build a system that uses predictions "
    "according to their reliability.\n\n"
    "The final project therefore combines forecasting research, "
    "time-series validation, deployment engineering, database "
    "integration, monitoring, and visualization into one "
    "end-to-end system."
)


# ============================================================
# COMBINE DOCUMENTATION
# ============================================================

documentation = "\n\n".join(
    sections
)


# ============================================================
# SAVE DOCUMENTATION
# ============================================================

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        documentation
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

if not os.path.exists(output_file):

    raise RuntimeError(
        "Documentation file was not created."
    )


file_size = os.path.getsize(
    output_file
)


if file_size == 0:

    raise RuntimeError(
        "Documentation file is empty."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINAL DOCUMENTATION RESULT")
print("=" * 80)

print(
    f"\nSections generated: "
    f"{len(sections)}"
)

print(
    f"Performance rows: "
    f"{len(performance)}"
)

print(
    f"Multi-horizon rows: "
    f"{len(horizon)}"
)

print(
    f"Sensor-group rows: "
    f"{len(groups)}"
)

print(
    f"Model-summary rows: "
    f"{len(models)}"
)

print(
    f"Documentation size: "
    f"{file_size:,} bytes"
)

print(
    "\n[PASS] Documentation generated successfully."
)

print(
    "\nDocumentation file:"
)

print(
    output_file
)

print("\n" + "=" * 80)
print("STEP 34 COMPLETE")
print("=" * 80)

print(
    "\nFinal project documentation is ready."
)