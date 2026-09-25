# ============================================================
# STEP 30 — END-TO-END SYSTEM VALIDATION
# ============================================================
#
# Purpose:
#   Validate the complete forecasting system:
#
#   Sensor Data
#       ↓
#   Future Features
#       ↓
#   Final Models
#       ↓
#   Forecast Database
#       ↓
#   Monitoring Database
#       ↓
#   Dashboard Data
#
# This script DOES NOT train models.
#
# ============================================================

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "outputs" / "data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
DB_PATH = BASE_DIR / "outputs" / "database" / "forecasting.db"

CLEANED_DATA_PATH = (
    DATA_DIR / "sensor_data_cleaned.csv"
)

FUTURE_DATA_PATH = (
    DATA_DIR / "future_features_60s.csv"
)


# ============================================================
# 2. SENSOR CONFIGURATION
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


VIBRATION_SENSORS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
]


MAGNETIC_SENSORS = [
    "mag_x",
    "mag_y",
    "mag_z",
]


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "vib_x_rms":
        (
            "step21_final_vib_x_rms.joblib",
            "XGBoost",
            146,
        ),

    "vib_y_rms":
        (
            "step21_final_vib_y_rms.joblib",
            "XGBoost",
            146,
        ),

    "vib_z_rms":
        (
            "step21_final_vib_z_rms.joblib",
            "XGBoost",
            146,
        ),

    "mag_x":
        (
            "step21_final_mag_x.joblib",
            "Ridge",
            146,
        ),

    "mag_y":
        (
            "step21_final_mag_y.joblib",
            "Ridge",
            146,
        ),

    "mag_z":
        (
            "step21_final_mag_z.joblib",
            "Ridge",
            146,
        ),

    "temperature":
        (
            "step21_final_temperature.joblib",
            "Persistence",
            0,
        ),
}


# ============================================================
# 4. VALIDATION STORAGE
# ============================================================

results = []


def check(
    category,
    name,
    passed,
    details=""
):

    status = (
        "PASS"
        if passed
        else
        "FAIL"
    )

    results.append(
        {
            "category": category,
            "check": name,
            "status": status,
            "details": details,
        }
    )

    symbol = (
        "[PASS]"
        if passed
        else
        "[FAIL]"
    )

    print(
        f"{symbol} {name}"
    )

    if details:

        print(
            f"       {details}"
        )


# ============================================================
# 5. START
# ============================================================

print()
print("=" * 75)
print("STEP 30 — END-TO-END SYSTEM VALIDATION")
print("=" * 75)


# ============================================================
# 6. FILE VALIDATION
# ============================================================

print()
print("[1/8] Checking project files...")
print("-" * 75)


required_files = {

    "Cleaned sensor data":
        CLEANED_DATA_PATH,

    "Future forecasting data":
        FUTURE_DATA_PATH,

    "Model directory":
        MODEL_DIR,

    "Database":
        DB_PATH,
}


for name, path in required_files.items():

    check(
        "Files",
        name,
        path.exists(),
        str(path),
    )


# ============================================================
# 7. SENSOR DATA VALIDATION
# ============================================================

print()
print("[2/8] Validating sensor data...")
print("-" * 75)


sensor_df = None


if CLEANED_DATA_PATH.exists():

    try:

        sensor_df = pd.read_csv(
            CLEANED_DATA_PATH
        )

        required_columns = [
            "ts",
            *SENSORS,
        ]


        missing_columns = [
            column
            for column in required_columns
            if column not in sensor_df.columns
        ]


        check(
            "Sensor Data",
            "Required columns",
            len(missing_columns) == 0,
            (
                "All required columns found."
                if not missing_columns
                else
                f"Missing: {missing_columns}"
            ),
        )


        if "ts" in sensor_df.columns:

            sensor_df["ts"] = pd.to_datetime(
                sensor_df["ts"],
                errors="coerce",
            )


            invalid_ts = (
                sensor_df["ts"]
                .isna()
                .sum()
            )


            check(
                "Sensor Data",
                "Valid timestamps",
                invalid_ts == 0,
                f"Invalid timestamps: {invalid_ts}",
            )


            duplicates = (
                sensor_df["ts"]
                .duplicated()
                .sum()
            )


            check(
                "Sensor Data",
                "No duplicate timestamps",
                duplicates == 0,
                f"Duplicate timestamps: {duplicates}",
            )


        if not missing_columns:

            missing_values = (
                sensor_df[SENSORS]
                .isna()
                .sum()
                .sum()
            )


            check(
                "Sensor Data",
                "No missing sensor values",
                missing_values == 0,
                f"Missing sensor values: {missing_values}",
            )


            numeric_ok = all(
                pd.api.types.is_numeric_dtype(
                    sensor_df[column]
                )
                for column in SENSORS
            )


            check(
                "Sensor Data",
                "Sensor columns are numeric",
                numeric_ok,
            )


        print()
        print(
            f"Sensor rows: "
            f"{len(sensor_df):,}"
        )


        print(
            f"Time range: "
            f"{sensor_df['ts'].min()} → "
            f"{sensor_df['ts'].max()}"
        )


    except Exception as e:

        check(
            "Sensor Data",
            "Sensor data readable",
            False,
            str(e),
        )


# ============================================================
# 8. FUTURE FEATURE VALIDATION
# ============================================================

print()
print("[3/8] Validating forecasting data...")
print("-" * 75)


future_df = None


if FUTURE_DATA_PATH.exists():

    try:

        future_df = pd.read_csv(
            FUTURE_DATA_PATH
        )


        future_df["ts"] = pd.to_datetime(
            future_df["ts"],
            errors="coerce",
        )


        future_df["target_ts"] = pd.to_datetime(
            future_df["target_ts"],
            errors="coerce",
        )


        check(
            "Forecast Data",
            "Future feature file readable",
            True,
            f"Rows: {len(future_df):,}",
        )


        expected_targets = [

            "future_vib_x_rms",
            "future_vib_y_rms",
            "future_vib_z_rms",

            "future_mag_x",
            "future_mag_y",
            "future_mag_z",

            "future_temperature",
        ]


        missing_targets = [

            column
            for column in expected_targets
            if column not in future_df.columns
        ]


        check(
            "Forecast Data",
            "All future target columns exist",
            len(missing_targets) == 0,
            (
                "All 7 future targets found."
                if not missing_targets
                else
                f"Missing: {missing_targets}"
            ),
        )


        if "actual_horizon_seconds" in future_df.columns:

            horizon = pd.to_numeric(
                future_df[
                    "actual_horizon_seconds"
                ],
                errors="coerce",
            )


            invalid_horizon = (
                horizon.isna()
                | (horizon != 60)
            ).sum()


            check(
                "Forecast Data",
                "All forecast horizons are exactly 60 seconds",
                invalid_horizon == 0,
                f"Invalid horizon rows: {invalid_horizon}",
            )


        target_ts_invalid = (
            future_df["target_ts"]
            .isna()
            .sum()
        )


        check(
            "Forecast Data",
            "Valid target timestamps",
            target_ts_invalid == 0,
            f"Invalid target timestamps: {target_ts_invalid}",
        )


        excluded_columns = {

            "ts",
            "target_ts",
            "actual_horizon_seconds",

            "future_vib_x_rms",
            "future_vib_y_rms",
            "future_vib_z_rms",

            "future_mag_x",
            "future_mag_y",
            "future_mag_z",

            "future_temperature",
        }


        model_features = [

            column
            for column in future_df.columns
            if column not in excluded_columns
        ]


        check(
            "Forecast Data",
            "Expected model feature count",
            len(model_features) == 146,
            (
                f"Found {len(model_features)} "
                f"features; expected 146."
            ),
        )


        leakage_columns = [

            column
            for column in expected_targets
            if column in model_features
        ]


        check(
            "Forecast Data",
            "Future targets excluded from model features",
            len(leakage_columns) == 0,
            (
                "No future target leakage."
                if not leakage_columns
                else
                f"Leaking columns: {leakage_columns}"
            ),
        )


        feature_nan = (
            future_df[model_features]
            .isna()
            .sum()
            .sum()
        )


        check(
            "Forecast Data",
            "No missing model features",
            feature_nan == 0,
            f"Missing feature values: {feature_nan}",
        )


    except Exception as e:

        check(
            "Forecast Data",
            "Forecast data validation",
            False,
            str(e),
        )


# ============================================================
# 9. MODEL VALIDATION
# ============================================================

print()
print("[4/8] Validating final models...")
print("-" * 75)


loaded_models = {}


for sensor in SENSORS:

    filename, expected_type, expected_features = (
        MODEL_CONFIG[sensor]
    )


    model_path = (
        MODEL_DIR / filename
    )


    if not model_path.exists():

        check(
            "Models",
            f"{sensor} model loads",
            False,
            f"Missing: {model_path}",
        )

        continue


    try:

        artifact = joblib.load(
            model_path
        )


        check(
            "Models",
            f"{sensor} model loads",
            True,
            filename,
        )


        structure_ok = (
            isinstance(artifact, dict)
        )


        check(
            "Models",
            f"{sensor} artifact structure",
            structure_ok,
            "Dictionary artifact."
            if structure_ok
            else
            "Unexpected artifact type.",
        )


        if not structure_ok:
            continue


        required_keys = {

            "model",
            "sensor",
            "model_name",
            "features",
            "horizon_seconds",
        }


        missing_keys = (
            required_keys
            - set(artifact.keys())
        )


        check(
            "Models",
            f"{sensor} artifact metadata",
            len(missing_keys) == 0,
            (
                "Required metadata present."
                if not missing_keys
                else
                f"Missing: {missing_keys}"
            ),
        )


        sensor_ok = (
            artifact.get("sensor")
            == sensor
        )


        check(
            "Models",
            f"{sensor} sensor metadata",
            sensor_ok,
            f"Stored sensor: {artifact.get('sensor')}",
        )


        model_name_ok = (
            artifact.get("model_name")
            == expected_type
        )


        check(
            "Models",
            f"{sensor} model type",
            model_name_ok,
            (
                f"Expected: {expected_type}, "
                f"Found: {artifact.get('model_name')}"
            ),
        )


        horizon_ok = (
            artifact.get("horizon_seconds")
            == 60
        )


        check(
            "Models",
            f"{sensor} horizon",
            horizon_ok,
            (
                f"Horizon: "
                f"{artifact.get('horizon_seconds')} seconds"
            ),
        )


        features = artifact.get(
            "features",
            []
        )


        feature_count_ok = (
            len(features)
            == expected_features
        )


        check(
            "Models",
            f"{sensor} feature schema",
            feature_count_ok,
            (
                f"Found {len(features)}; "
                f"expected {expected_features}."
            ),
        )


        loaded_models[sensor] = artifact


    except Exception as e:

        check(
            "Models",
            f"{sensor} model loading",
            False,
            str(e),
        )


# ============================================================
# 10. DATABASE VALIDATION
# ============================================================

print()
print("[5/8] Validating SQLite database...")
print("-" * 75)


forecast_db = pd.DataFrame()
monitoring_db = pd.DataFrame()


if DB_PATH.exists():

    try:

        conn = sqlite3.connect(
            DB_PATH
        )


        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """,
            conn,
        )


        table_names = set(
            tables["name"].tolist()
        )


        forecast_exists = (
            "forecast_predictions"
            in table_names
        )


        monitoring_exists = (
            "prediction_monitoring"
            in table_names
        )


        check(
            "Database",
            "forecast_predictions table",
            forecast_exists,
        )


        check(
            "Database",
            "prediction_monitoring table",
            monitoring_exists,
        )


        # ----------------------------------------------------
        # Forecast database
        # ----------------------------------------------------

        if forecast_exists:

            forecast_db = pd.read_sql_query(
                """
                SELECT *
                FROM forecast_predictions
                """,
                conn,
            )


            print()
            print(
                f"Forecast records: "
                f"{len(forecast_db):,}"
            )


            required_forecast_columns = {

                "id",
                "feature_timestamp",
                "forecast_timestamp",
                "horizon_seconds",
                "sensor",
                "model",
                "current_value",
                "predicted_value",
                "change_from_current",
            }


            missing_forecast_columns = (
                required_forecast_columns
                - set(forecast_db.columns)
            )


            check(
                "Database",
                "Forecast table schema",
                len(missing_forecast_columns) == 0,
                (
                    "Required columns present."
                    if not missing_forecast_columns
                    else
                    f"Missing: {missing_forecast_columns}"
                ),
            )


            if not missing_forecast_columns:

                db_sensors = set(
                    forecast_db["sensor"]
                    .dropna()
                    .unique()
                )


                missing_db_sensors = (
                    set(SENSORS)
                    - db_sensors
                )


                check(
                    "Database",
                    "All 7 sensors have forecasts",
                    len(missing_db_sensors) == 0,
                    (
                        "All sensors found."
                        if not missing_db_sensors
                        else
                        f"Missing: {missing_db_sensors}"
                    ),
                )


                predicted = pd.to_numeric(
                    forecast_db[
                        "predicted_value"
                    ],
                    errors="coerce",
                )


                invalid_predictions = (
                    ~np.isfinite(predicted)
                ).sum()


                check(
                    "Database",
                    "Forecast predictions are finite",
                    invalid_predictions == 0,
                    (
                        f"Invalid predictions: "
                        f"{invalid_predictions}"
                    ),
                )


                horizons = pd.to_numeric(
                    forecast_db[
                        "horizon_seconds"
                    ],
                    errors="coerce",
                )


                invalid_horizons = (
                    horizons.isna()
                    | (horizons != 60)
                ).sum()


                check(
                    "Database",
                    "Database forecasts use 60-sec horizon",
                    invalid_horizons == 0,
                    (
                        f"Invalid horizon records: "
                        f"{invalid_horizons}"
                    ),
                )


                duplicate_forecasts = (
                    forecast_db
                    .duplicated(
                        subset=[
                            "feature_timestamp",
                            "forecast_timestamp",
                            "sensor",
                        ]
                    )
                    .sum()
                )


                check(
                    "Database",
                    "No duplicate forecast records",
                    duplicate_forecasts == 0,
                    (
                        f"Duplicate records: "
                        f"{duplicate_forecasts}"
                    ),
                )


        # ----------------------------------------------------
        # Monitoring database
        # ----------------------------------------------------

        if monitoring_exists:

            monitoring_db = pd.read_sql_query(
                """
                SELECT *
                FROM prediction_monitoring
                """,
                conn,
            )


            print()
            print(
                f"Monitoring records: "
                f"{len(monitoring_db):,}"
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # We explicitly inspect the actual schema.
            # This prevents KeyError crashes.
            # ------------------------------------------------

            actual_monitoring_columns = set(
                monitoring_db.columns
            )


            required_monitoring_columns = {

                "forecast_id",
                "feature_timestamp",
                "forecast_timestamp",
                "horizon_seconds",
                "sensor",
                "model",
                "current_value",
                "predicted_value",
                "change_from_current",
                "status",
                "severity",
                "alert_type",
                "reason",
                "model_reliability",
                "created_at",
            }


            missing_monitoring_columns = (
                required_monitoring_columns
                - actual_monitoring_columns
            )


            if not missing_monitoring_columns:

                check(
                    "Database",
                    "Monitoring table schema",
                    True,
                    "All required columns present.",
                )


                monitoring_schema_ok = True


            else:

                check(
                    "Database",
                    "Monitoring table schema",
                    False,
                    (
                        "Missing: "
                        f"{sorted(missing_monitoring_columns)}"
                    ),
                )


                monitoring_schema_ok = False


            # ------------------------------------------------
            # Sensor coverage can still be checked even if
            # the old schema is present.
            # ------------------------------------------------

            if "sensor" in monitoring_db.columns:

                monitored_sensors = set(
                    monitoring_db["sensor"]
                    .dropna()
                    .unique()
                )


                missing_monitoring_sensors = (
                    set(SENSORS)
                    - monitored_sensors
                )


                check(
                    "Database",
                    "All 7 sensors monitored",
                    len(missing_monitoring_sensors) == 0,
                    (
                        "All sensors monitored."
                        if not missing_monitoring_sensors
                        else
                        f"Missing: {missing_monitoring_sensors}"
                    ),
                )


            # ------------------------------------------------
            # Status validation
            # ------------------------------------------------

            if "status" in monitoring_db.columns:

                allowed_statuses = {

                    "NORMAL",
                    "WARNING",
                    "ANOMALY",
                    "LOW_CONFIDENCE",
                    "UNKNOWN",
                }


                actual_statuses = set(
                    monitoring_db["status"]
                    .dropna()
                    .unique()
                )


                invalid_statuses = (
                    actual_statuses
                    - allowed_statuses
                )


                check(
                    "Database",
                    "Monitoring statuses are valid",
                    len(invalid_statuses) == 0,
                    (
                        "All statuses valid."
                        if not invalid_statuses
                        else
                        f"Invalid: {invalid_statuses}"
                    ),
                )


            # ------------------------------------------------
            # Duplicate monitoring check
            # ------------------------------------------------

            if "forecast_id" in monitoring_db.columns:

                duplicate_monitoring = (
                    monitoring_db
                    .duplicated(
                        subset=[
                            "forecast_id"
                        ]
                    )
                    .sum()
                )


                check(
                    "Database",
                    "No duplicate monitoring records",
                    duplicate_monitoring == 0,
                    (
                        f"Duplicate records: "
                        f"{duplicate_monitoring}"
                    ),
                )

            else:

                # Old schema cannot perform this check.

                print(
                    "       Relationship check deferred "
                    "until monitoring schema is upgraded."
                )


        conn.close()


    except Exception as e:

        check(
            "Database",
            "Database validation",
            False,
            str(e),
        )


# ============================================================
# 11. FORECAST → MONITORING RELATIONSHIP
# ============================================================

print()
print("[6/8] Validating forecast-monitoring relationship...")
print("-" * 75)


if (
    not forecast_db.empty
    and not monitoring_db.empty
    and "forecast_id" in monitoring_db.columns
):

    forecast_ids = set(
        forecast_db["id"]
    )


    monitoring_forecast_ids = set(
        monitoring_db["forecast_id"]
    )


    missing_monitoring = (
        forecast_ids
        - monitoring_forecast_ids
    )


    orphan_monitoring = (
        monitoring_forecast_ids
        - forecast_ids
    )


    check(
        "Integration",
        "Every forecast has monitoring result",
        len(missing_monitoring) == 0,
        (
            f"Unmonitored forecasts: "
            f"{len(missing_monitoring):,}"
        ),
    )


    check(
        "Integration",
        "No orphan monitoring records",
        len(orphan_monitoring) == 0,
        (
            f"Orphan records: "
            f"{len(orphan_monitoring):,}"
        ),
    )


else:

    check(
        "Integration",
        "Forecast-monitoring relationship",
        False,
        (
            "Cannot validate relationship because "
            "prediction_monitoring.forecast_id is missing."
        ),
    )


# ============================================================
# 12. MONITORING LOGIC
# ============================================================

print()
print("[7/8] Validating monitoring logic...")
print("-" * 75)


if not monitoring_db.empty:

    # --------------------------------------------------------
    # Magnetic logic
    # --------------------------------------------------------

    if (
        "sensor" in monitoring_db.columns
        and "status" in monitoring_db.columns
    ):

        magnetic_df = monitoring_db[
            monitoring_db["sensor"].isin(
                MAGNETIC_SENSORS
            )
        ]


        if not magnetic_df.empty:

            magnetic_status_ok = (
                magnetic_df["status"]
                .eq("LOW_CONFIDENCE")
                .all()
            )


            check(
                "Monitoring",
                "Magnetic sensors use LOW_CONFIDENCE",
                magnetic_status_ok,
                (
                    "Magnetic predictions are not "
                    "treated as physical anomalies."
                ),
            )


    # --------------------------------------------------------
    # Vibration logic
    # --------------------------------------------------------

    if (
        "sensor" in monitoring_db.columns
        and "status" in monitoring_db.columns
        and "predicted_value"
        in monitoring_db.columns
    ):

        vibration_df = monitoring_db[
            monitoring_db["sensor"].isin(
                VIBRATION_SENSORS
            )
        ].copy()


        if not vibration_df.empty:

            vibration_predictions = pd.to_numeric(
                vibration_df[
                    "predicted_value"
                ],
                errors="coerce",
            )


            expected_status = np.select(

                [

                    vibration_predictions
                    >= 0.20,

                    vibration_predictions
                    >= 0.10,
                ],

                [

                    "ANOMALY",

                    "WARNING",
                ],

                default="NORMAL",
            )


            actual_status = (
                vibration_df[
                    "status"
                ]
                .to_numpy()
            )


            vibration_ok = np.array_equal(
                expected_status,
                actual_status,
            )


            check(
                "Monitoring",
                "Vibration threshold logic",
                vibration_ok,
                (
                    "Threshold classification is correct."
                    if vibration_ok
                    else
                    "Threshold classification mismatch."
                ),
            )


    # --------------------------------------------------------
    # Temperature logic
    # --------------------------------------------------------

    if (
        "sensor" in monitoring_db.columns
        and "status" in monitoring_db.columns
        and "predicted_value"
        in monitoring_db.columns
    ):

        temperature_df = monitoring_db[
            monitoring_db["sensor"]
            == "temperature"
        ].copy()


        if not temperature_df.empty:

            temperature_predictions = pd.to_numeric(
                temperature_df[
                    "predicted_value"
                ],
                errors="coerce",
            )


            expected_status = np.select(

                [

                    temperature_predictions
                    >= 50.0,

                    temperature_predictions
                    >= 48.0,
                ],

                [

                    "ANOMALY",

                    "WARNING",
                ],

                default="NORMAL",
            )


            actual_status = (
                temperature_df[
                    "status"
                ]
                .to_numpy()
            )


            temperature_ok = np.array_equal(
                expected_status,
                actual_status,
            )


            check(
                "Monitoring",
                "Temperature threshold logic",
                temperature_ok,
                (
                    "Threshold classification is correct."
                    if temperature_ok
                    else
                    "Threshold classification mismatch."
                ),
            )


# ============================================================
# 13. DASHBOARD VALIDATION
# ============================================================

print()
print("[8/8] Validating dashboard data...")
print("-" * 75)


if DB_PATH.exists():

    try:

        conn = sqlite3.connect(
            DB_PATH
        )


        dashboard_query = """
        SELECT
            fp.id,
            fp.feature_timestamp,
            fp.forecast_timestamp,
            fp.sensor,
            fp.current_value,
            fp.predicted_value,
            fp.model,
            pm.status,
            pm.severity,
            pm.alert_type
        FROM forecast_predictions fp
        LEFT JOIN prediction_monitoring pm
            ON fp.id = pm.forecast_id
        ORDER BY fp.forecast_timestamp DESC
        LIMIT 100
        """


        # ----------------------------------------------------
        # If old schema is present, the JOIN cannot work.
        # We test this safely.
        # ----------------------------------------------------

        if (
            not monitoring_db.empty
            and "forecast_id"
            in monitoring_db.columns
        ):

            dashboard_df = pd.read_sql_query(
                dashboard_query,
                conn,
            )


            check(
                "Dashboard",
                "Dashboard query executes",
                True,
                f"Rows returned: {len(dashboard_df)}",
            )


            missing_status = (
                dashboard_df["status"]
                .isna()
                .sum()
            )


            check(
                "Dashboard",
                "Dashboard forecasts have monitoring status",
                missing_status == 0,
                (
                    f"Missing monitoring status: "
                    f"{missing_status}"
                ),
            )


        else:

            check(
                "Dashboard",
                "Dashboard query executes",
                False,
                (
                    "Monitoring table does not yet contain "
                    "forecast_id required by the dashboard."
                ),
            )


        conn.close()


    except Exception as e:

        check(
            "Dashboard",
            "Dashboard validation",
            False,
            str(e),
        )


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)


results_df = pd.DataFrame(
    results
)


total_checks = len(
    results_df
)


passed_checks = (
    results_df["status"]
    .eq("PASS")
    .sum()
)


failed_checks = (
    results_df["status"]
    .eq("FAIL")
    .sum()
)


print()
print(
    f"Total checks : {total_checks}"
)

print(
    f"Passed       : {passed_checks}"
)

print(
    f"Failed       : {failed_checks}"
)


# ============================================================
# 15. SAVE REPORT
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REPORT_PATH = (
    REPORT_DIR
    / "step30_end_to_end_validation.csv"
)


results_df.to_csv(
    REPORT_PATH,
    index=False
)


print()
print(
    "Validation report:"
)

print(
    f"  {REPORT_PATH}"
)


# ============================================================
# 16. FINAL SYSTEM STATUS
# ============================================================

print()
print("=" * 75)


if failed_checks == 0:

    print(
        "SYSTEM VALIDATION: PASSED"
    )

    print("=" * 75)

    print(
        "\nAll validation checks passed."
    )

else:

    print(
        "SYSTEM VALIDATION: ATTENTION REQUIRED"
    )

    print("=" * 75)

    print(
        "\nSome checks require attention."
    )

    print(
        "Review the [FAIL] entries above."
    )


print()
print("=" * 75)
print("STEP 30 COMPLETE")
print("=" * 75)
