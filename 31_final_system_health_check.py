# ============================================================
# STEP 31: FINAL SYSTEM HEALTH CHECK
# ============================================================
# Purpose:
#   Perform a final health check of the complete forecasting
#   system after successful end-to-end validation.
#
# Checks:
#   1. Project files
#   2. Dataset
#   3. Model artifacts
#   4. Database
#   5. Forecast records
#   6. Monitoring records
#   7. Forecast/monitoring relationship
#   8. Data integrity
#   9. Dashboard script
#  10. Reports and outputs
#
# This script DOES NOT modify models or forecasting data.
# ============================================================

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "outputs", "data")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
DB_DIR = os.path.join(BASE_DIR, "outputs", "database")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FORECAST_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")

DB_PATH = os.path.join(DB_DIR, "forecasting.db")


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = [

    # Dataset
    os.path.join(BASE_DIR, "sensor_data.csv"),
    os.path.join(DATA_DIR, "sensor_data_cleaned.csv"),
    os.path.join(DATA_DIR, "future_features_60s.csv"),

    # Main pipeline
    os.path.join(BASE_DIR, "01_data_pipeline.py"),
    os.path.join(BASE_DIR, "13_train_true_forecasting_models.py"),
    os.path.join(BASE_DIR, "20_walk_forward_final_selection.py"),
    os.path.join(BASE_DIR, "21_final_model_training.py"),
    os.path.join(BASE_DIR, "23_realtime_forecasting.py"),
    os.path.join(BASE_DIR, "25_prediction_database.py"),
    os.path.join(BASE_DIR, "26_prediction_monitoring.py"),
    os.path.join(BASE_DIR, "27_forecasting_dashboard.py"),
    os.path.join(BASE_DIR, "28_realtime_forecasting_service.py"),
    os.path.join(BASE_DIR, "29_realtime_monitoring_service.py"),
    os.path.join(BASE_DIR, "30_end_to_end_validation.py"),

    # Database
    DB_PATH,
]


# ============================================================
# MODEL CONFIGURATION
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

EXPECTED_MODELS = {
    "vib_x_rms": "XGBoost",
    "vib_y_rms": "XGBoost",
    "vib_z_rms": "XGBoost",
    "mag_x": "Ridge",
    "mag_y": "Ridge",
    "mag_z": "Ridge",
    "temperature": "Persistence",
}


# ============================================================
# COUNTERS
# ============================================================

total_checks = 0
passed_checks = 0
failed_checks = 0


# ============================================================
# CHECK FUNCTION
# ============================================================

def check(name, condition, details=""):

    global total_checks
    global passed_checks
    global failed_checks

    total_checks += 1

    if condition:

        passed_checks += 1

        print(f"[PASS] {name}")

        if details:
            print(f"       {details}")

    else:

        failed_checks += 1

        print(f"[FAIL] {name}")

        if details:
            print(f"       {details}")


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("STEP 31: FINAL SYSTEM HEALTH CHECK")
print("=" * 75)

print()
print("Project:")
print(BASE_DIR)

print()
print("Purpose:")
print("Final verification of the complete forecasting system.")

print()


# ============================================================
# 1. REQUIRED FILES
# ============================================================

print("=" * 75)
print("1. PROJECT FILE CHECK")
print("=" * 75)

missing_files = []

for file_path in REQUIRED_FILES:

    if not os.path.exists(file_path):
        missing_files.append(file_path)


check(
    "Required project files",
    len(missing_files) == 0,
    f"{len(REQUIRED_FILES) - len(missing_files)}/{len(REQUIRED_FILES)} files present"
)

if missing_files:

    for file_path in missing_files:
        print(f"       Missing: {file_path}")


# ============================================================
# 2. MODEL ARTIFACTS
# ============================================================

print()
print("=" * 75)
print("2. MODEL ARTIFACT CHECK")
print("=" * 75)

loaded_models = {}

for sensor in SENSORS:

    model_path = os.path.join(
        MODEL_DIR,
        f"step21_final_{sensor}.joblib"
    )

    exists = os.path.exists(model_path)

    check(
        f"{sensor} model file",
        exists,
        os.path.basename(model_path)
    )

    if not exists:
        continue

    try:

        artifact = joblib.load(model_path)

        loaded_models[sensor] = artifact

        check(
            f"{sensor} model loading",
            artifact is not None,
            "Artifact loaded successfully"
        )

        if isinstance(artifact, dict):

            model = artifact.get("model")
            model_name = artifact.get("model_name")

            # ------------------------------------------------
            # Persistence model is intentionally different.
            # It does not require a trained model object.
            # ------------------------------------------------

            if sensor == "temperature" and model_name == "Persistence":

                check(
                    f"{sensor} model object",
                    True,
                    "Persistence model - no trained model object required"
                )

            else:

                check(
                    f"{sensor} model object",
                    model is not None,
                    f"Model: {model_name}"
                )

        else:

            check(
                f"{sensor} model object",
                True,
                f"Artifact type: {type(artifact).__name__}"
            )

    except Exception as e:

        check(
            f"{sensor} model loading",
            False,
            str(e)
        )


# ============================================================
# 3. DATABASE CONNECTION
# ============================================================

print()
print("=" * 75)
print("3. DATABASE CHECK")
print("=" * 75)

db_exists = os.path.exists(DB_PATH)

check(
    "Forecasting database exists",
    db_exists,
    DB_PATH
)


if not db_exists:

    print()
    print("=" * 75)
    print("STEP 31 CANNOT CONTINUE")
    print("=" * 75)

    raise SystemExit


try:

    conn = sqlite3.connect(DB_PATH)

    check(
        "Database connection",
        True,
        "SQLite connection successful"
    )

except Exception as e:

    check(
        "Database connection",
        False,
        str(e)
    )

    raise SystemExit


# ============================================================
# 4. DATABASE TABLES
# ============================================================

print()
print("=" * 75)
print("4. DATABASE TABLE CHECK")
print("=" * 75)

tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    """,
    conn
)

table_names = set(tables["name"].tolist())

check(
    "forecast_predictions table",
    "forecast_predictions" in table_names
)

check(
    "prediction_monitoring table",
    "prediction_monitoring" in table_names
)


# ============================================================
# 5. FORECAST DATABASE
# ============================================================

print()
print("=" * 75)
print("5. FORECAST DATABASE CHECK")
print("=" * 75)

if "forecast_predictions" in table_names:

    forecast_df = pd.read_sql_query(
        "SELECT * FROM forecast_predictions",
        conn
    )

    forecast_count = len(forecast_df)

    check(
        "Forecast records exist",
        forecast_count > 0,
        f"{forecast_count:,} records"
    )

    check(
        "Expected forecast record count",
        forecast_count == 3500,
        f"Expected 3,500 | Found {forecast_count:,}"
    )

    if forecast_count > 0:

        sensors_in_db = set(
            forecast_df["sensor"].dropna().unique()
        )

        check(
            "All seven sensors present",
            set(SENSORS).issubset(sensors_in_db),
            f"Sensors found: {len(sensors_in_db)}"
        )

        check(
            "Forecast values are finite",
            np.isfinite(
                forecast_df["predicted_value"].astype(float)
            ).all(),
            "No NaN or infinite predictions"
        )

        check(
            "Current values are finite",
            np.isfinite(
                forecast_df["current_value"].astype(float)
            ).all(),
            "No NaN or infinite current values"
        )

        horizons = set(
            forecast_df["horizon_seconds"]
            .dropna()
            .astype(int)
            .unique()
        )

        check(
            "Forecast horizon is 60 seconds",
            horizons == {60},
            f"Horizons found: {sorted(horizons)}"
        )

        duplicate_count = forecast_df.duplicated(
            subset=[
                "feature_timestamp",
                "forecast_timestamp",
                "sensor"
            ]
        ).sum()

        check(
            "No duplicate forecasts",
            duplicate_count == 0,
            f"Duplicates: {duplicate_count}"
        )

else:

    forecast_df = pd.DataFrame()
    forecast_count = 0


# ============================================================
# 6. MODEL ASSIGNMENTS
# ============================================================

print()
print("=" * 75)
print("6. MODEL ASSIGNMENT CHECK")
print("=" * 75)

if not forecast_df.empty:

    for sensor in SENSORS:

        sensor_df = forecast_df[
            forecast_df["sensor"] == sensor
        ]

        if sensor_df.empty:

            check(
                f"{sensor} forecast model",
                False,
                "No forecast records found"
            )

            continue

        models_found = set(
            sensor_df["model"].dropna().astype(str).unique()
        )

        expected_model = EXPECTED_MODELS[sensor]

        check(
            f"{sensor} model assignment",
            expected_model in models_found,
            f"Expected: {expected_model} | Found: {sorted(models_found)}"
        )


# ============================================================
# 7. MONITORING DATABASE
# ============================================================

print()
print("=" * 75)
print("7. MONITORING DATABASE CHECK")
print("=" * 75)

if "prediction_monitoring" in table_names:

    monitoring_df = pd.read_sql_query(
        "SELECT * FROM prediction_monitoring",
        conn
    )

    monitoring_count = len(monitoring_df)

    check(
        "Monitoring records exist",
        monitoring_count > 0,
        f"{monitoring_count:,} records"
    )

    check(
        "Monitoring/forecast record count",
        monitoring_count == forecast_count,
        f"Forecasts: {forecast_count:,} | Monitoring: {monitoring_count:,}"
    )

    required_monitoring_columns = [
        "forecast_id",
        "feature_timestamp",
        "forecast_timestamp",
        "horizon_seconds",
        "sensor",
        "model",
        "current_value",
        "predicted_value",
        "status",
        "severity",
        "alert_type",
        "reason",
        "model_reliability",
    ]

    missing_monitoring_columns = [
        col
        for col in required_monitoring_columns
        if col not in monitoring_df.columns
    ]

    check(
        "Monitoring table schema",
        len(missing_monitoring_columns) == 0,
        (
            "All required monitoring columns present"
            if not missing_monitoring_columns
            else f"Missing: {missing_monitoring_columns}"
        )
    )

else:

    monitoring_df = pd.DataFrame()
    monitoring_count = 0


# ============================================================
# 8. FORECAST → MONITORING RELATIONSHIP
# ============================================================

print()
print("=" * 75)
print("8. FORECAST-MONITORING RELATIONSHIP")
print("=" * 75)

if (
    not forecast_df.empty
    and not monitoring_df.empty
    and "forecast_id" in monitoring_df.columns
):

    forecast_ids = set(
        forecast_df["id"].astype(int)
    )

    monitoring_ids = set(
        monitoring_df["forecast_id"].dropna().astype(int)
    )

    missing_relationships = (
        forecast_ids - monitoring_ids
    )

    check(
        "Every forecast has monitoring record",
        len(missing_relationships) == 0,
        f"Missing relationships: {len(missing_relationships)}"
    )

else:

    check(
        "Forecast-monitoring relationship",
        False,
        "Required data unavailable"
    )


# ============================================================
# 9. MONITORING STATUS
# ============================================================

print()
print("=" * 75)
print("9. MONITORING STATUS CHECK")
print("=" * 75)

if not monitoring_df.empty:

    valid_statuses = {
        "NORMAL",
        "WARNING",
        "CRITICAL",
        "LOW_CONFIDENCE",
        "INFO",
        "MODEL_RELIABILITY_WARNING",
    }

    statuses = set(
        monitoring_df["status"]
        .dropna()
        .astype(str)
        .unique()
    )

    invalid_statuses = statuses - valid_statuses

    check(
        "Monitoring statuses valid",
        len(invalid_statuses) == 0,
        f"Statuses found: {sorted(statuses)}"
    )


# ============================================================
# 10. SENSOR COVERAGE
# ============================================================

print()
print("=" * 75)
print("10. SENSOR COVERAGE CHECK")
print("=" * 75)

if not forecast_df.empty:

    for sensor in SENSORS:

        count = (
            forecast_df["sensor"] == sensor
        ).sum()

        check(
            f"{sensor} forecast coverage",
            count > 0,
            f"{count:,} records"
        )


# ============================================================
# 11. DATABASE DUPLICATE CHECK
# ============================================================

print()
print("=" * 75)
print("11. DATABASE DUPLICATE CHECK")
print("=" * 75)

if not forecast_df.empty:

    duplicate_forecasts = forecast_df.duplicated(
        subset=[
            "feature_timestamp",
            "forecast_timestamp",
            "sensor"
        ]
    ).sum()

    check(
        "Forecast database has no duplicates",
        duplicate_forecasts == 0,
        f"Duplicates: {duplicate_forecasts}"
    )

if (
    not monitoring_df.empty
    and "forecast_id" in monitoring_df.columns
):

    duplicate_monitoring = monitoring_df[
        "forecast_id"
    ].duplicated().sum()

    check(
        "Monitoring database has no duplicate forecast IDs",
        duplicate_monitoring == 0,
        f"Duplicates: {duplicate_monitoring}"
    )


# ============================================================
# 12. REPORT CHECK
# ============================================================

print()
print("=" * 75)
print("12. REPORT CHECK")
print("=" * 75)

expected_reports = [
    "step26_prediction_monitoring.csv",
    "step29_realtime_monitoring.csv",
]

for report in expected_reports:

    path = os.path.join(
        REPORT_DIR,
        report
    )

    check(
        f"Report: {report}",
        os.path.exists(path)
    )


# ============================================================
# 13. DASHBOARD CHECK
# ============================================================

print()
print("=" * 75)
print("13. DASHBOARD CHECK")
print("=" * 75)

dashboard_path = os.path.join(
    BASE_DIR,
    "27_forecasting_dashboard.py"
)

check(
    "Dashboard script exists",
    os.path.exists(dashboard_path),
    "27_forecasting_dashboard.py"
)


# ============================================================
# 14. REALTIME SERVICE CHECK
# ============================================================

print()
print("=" * 75)
print("14. REALTIME SERVICE CHECK")
print("=" * 75)

realtime_scripts = [
    "23_realtime_forecasting.py",
    "28_realtime_forecasting_service.py",
    "29_realtime_monitoring_service.py",
]

for script in realtime_scripts:

    path = os.path.join(
        BASE_DIR,
        script
    )

    check(
        f"{script} exists",
        os.path.exists(path)
    )


# ============================================================
# 15. SQLITE INTEGRITY
# ============================================================

print()
print("=" * 75)
print("15. SQLITE INTEGRITY CHECK")
print("=" * 75)

try:

    integrity_result = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    check(
        "SQLite database integrity",
        integrity_result == "ok",
        f"Result: {integrity_result}"
    )

except Exception as e:

    check(
        "SQLite database integrity",
        False,
        str(e)
    )


# ============================================================
# 16. DATABASE STORAGE
# ============================================================

print()
print("=" * 75)
print("16. DATABASE STORAGE CHECK")
print("=" * 75)

if os.path.exists(DB_PATH):

    db_size_mb = os.path.getsize(DB_PATH) / (
        1024 * 1024
    )

    check(
        "Database file is readable",
        db_size_mb > 0,
        f"Size: {db_size_mb:.2f} MB"
    )


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("STEP 31 FINAL SUMMARY")
print("=" * 75)

print()
print(f"Total checks : {total_checks}")
print(f"Passed       : {passed_checks}")
print(f"Failed       : {failed_checks}")

print()

if failed_checks == 0:

    print("=" * 75)
    print("SYSTEM HEALTH CHECK: PASSED")
    print("=" * 75)

    print()
    print("The forecasting system is healthy and ready for")
    print("final documentation and project presentation.")

else:

    print("=" * 75)
    print("SYSTEM HEALTH CHECK: FAILED")
    print("=" * 75)

    print()
    print("Review the failed checks above before proceeding.")

print()
print("=" * 75)
print("STEP 31 COMPLETE")
print("=" * 75)