# ============================================================
# STEP 29 — REAL-TIME MONITORING SERVICE
# ============================================================
#
# Purpose:
#   Connect the forecasting database with the monitoring system.
#
# Pipeline:
#
#   Forecasts
#       ↓
#   forecast_predictions
#       ↓
#   Monitoring Engine
#       ↓
#   prediction_monitoring
#       ↓
#   Dashboard
#
# Important:
#   - Automatically fixes the old Step 26 monitoring schema.
#   - Keeps forecast_predictions untouched.
#   - Rebuilds monitoring results from current forecasts.
#   - Does NOT use percentage-change alerts for magnetic sensors.
#
# ============================================================

import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = (
    BASE_DIR
    / "outputs"
    / "database"
    / "forecasting.db"
)

REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_PATH = (
    REPORT_DIR
    / "step29_realtime_monitoring.csv"
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
# 3. MONITORING THRESHOLDS
# ============================================================

VIBRATION_WARNING = 0.10
VIBRATION_CRITICAL = 0.20

TEMPERATURE_WARNING = 48.0
TEMPERATURE_CRITICAL = 50.0


# ============================================================
# 4. MODEL RELIABILITY
# ============================================================

MODEL_RELIABILITY = {

    "vib_x_rms": "MODERATE",
    "vib_y_rms": "MODERATE",
    "vib_z_rms": "MODERATE",

    "mag_x": "LOW",
    "mag_y": "LOW",
    "mag_z": "LOW",

    "temperature": "HIGH",
}


# ============================================================
# 5. DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {

    "vib_x_rms": "Vibration X",
    "vib_y_rms": "Vibration Y",
    "vib_z_rms": "Vibration Z",

    "mag_x": "Magnetic X",
    "mag_y": "Magnetic Y",
    "mag_z": "Magnetic Z",

    "temperature": "Temperature",
}


# ============================================================
# 6. START
# ============================================================

print()
print("=" * 75)
print("STEP 29 — REAL-TIME MONITORING SERVICE")
print("=" * 75)


# ============================================================
# 7. DATABASE CHECK
# ============================================================

if not DB_PATH.exists():

    raise FileNotFoundError(
        f"\nDatabase not found:\n{DB_PATH}"
    )


print()
print("Database:")
print(f"  {DB_PATH}")


# ============================================================
# 8. CONNECT DATABASE
# ============================================================

conn = sqlite3.connect(
    DB_PATH
)

cursor = conn.cursor()


# ============================================================
# 9. VERIFY FORECAST TABLE
# ============================================================

print()
print("[1/7] Checking forecast database...")
print("-" * 75)


cursor.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    AND name='forecast_predictions'
    """
)


if cursor.fetchone() is None:

    conn.close()

    raise RuntimeError(
        "forecast_predictions table does not exist."
    )


# ============================================================
# 10. LOAD FORECASTS
# ============================================================

forecasts = pd.read_sql_query(
    """
    SELECT
        id,
        feature_timestamp,
        forecast_timestamp,
        horizon_seconds,
        sensor,
        model,
        current_value,
        predicted_value,
        change_from_current,
        created_at
    FROM forecast_predictions
    ORDER BY forecast_timestamp ASC
    """,
    conn
)


if forecasts.empty:

    conn.close()

    print(
        "\nNo forecasts found."
    )

    raise SystemExit(0)


print(
    f"Forecast records found: "
    f"{len(forecasts):,}"
)


# ============================================================
# 11. VERIFY FORECAST DATA
# ============================================================

required_forecast_columns = [

    "id",
    "feature_timestamp",
    "forecast_timestamp",
    "horizon_seconds",
    "sensor",
    "model",
    "current_value",
    "predicted_value",
    "change_from_current",
    "created_at",
]


missing_forecast_columns = [

    column
    for column in required_forecast_columns
    if column not in forecasts.columns
]


if missing_forecast_columns:

    conn.close()

    raise RuntimeError(
        "Missing forecast columns: "
        + str(missing_forecast_columns)
    )


# ============================================================
# 12. CLEAN TIMESTAMPS
# ============================================================

forecasts["feature_timestamp"] = pd.to_datetime(
    forecasts["feature_timestamp"],
    errors="coerce"
)

forecasts["forecast_timestamp"] = pd.to_datetime(
    forecasts["forecast_timestamp"],
    errors="coerce"
)


# ============================================================
# 13. CHECK FORECAST SENSORS
# ============================================================

forecast_sensors = set(
    forecasts["sensor"]
    .dropna()
    .unique()
)


missing_sensors = (
    set(SENSORS)
    - forecast_sensors
)


if missing_sensors:

    print(
        f"WARNING: Missing sensors: "
        f"{sorted(missing_sensors)}"
    )

else:

    print(
        "All 7 sensors found."
    )


# ============================================================
# 14. MONITORING TABLE MIGRATION
# ============================================================

print()
print("[2/7] Preparing monitoring database...")
print("-" * 75)


# ------------------------------------------------------------
# Check whether old monitoring table exists
# ------------------------------------------------------------

cursor.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    AND name='prediction_monitoring'
    """
)

monitoring_exists = (
    cursor.fetchone() is not None
)


if monitoring_exists:

    cursor.execute(
        """
        PRAGMA table_info(prediction_monitoring)
        """
    )

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }


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


    missing_columns = (
        required_monitoring_columns
        - existing_columns
    )


    if missing_columns:

        print(
            "Old Step 26 monitoring schema detected."
        )

        print(
            "Missing columns:"
        )

        for column in sorted(
            missing_columns
        ):

            print(
                f"  - {column}"
            )


        print(
            "\nRebuilding monitoring table..."
        )


        # ----------------------------------------------------
        # Remove old table
        #
        # Forecast predictions are NOT touched.
        # ----------------------------------------------------

        cursor.execute(
            """
            DROP TABLE prediction_monitoring
            """
        )

        conn.commit()

        print(
            "Old monitoring table removed."
        )

    else:

        print(
            "Monitoring table already has "
            "the required schema."
        )


# ============================================================
# 15. CREATE CORRECT MONITORING TABLE
# ============================================================

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS prediction_monitoring (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        forecast_id INTEGER UNIQUE,

        feature_timestamp TEXT NOT NULL,

        forecast_timestamp TEXT NOT NULL,

        horizon_seconds INTEGER NOT NULL,

        sensor TEXT NOT NULL,

        model TEXT NOT NULL,

        current_value REAL,

        predicted_value REAL,

        change_from_current REAL,

        status TEXT NOT NULL,

        severity TEXT NOT NULL,

        alert_type TEXT NOT NULL,

        reason TEXT,

        model_reliability TEXT,

        created_at TEXT NOT NULL,

        FOREIGN KEY (forecast_id)
            REFERENCES forecast_predictions(id)
    )
    """
)

conn.commit()


# ============================================================
# 16. INDEXES
# ============================================================

cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_monitoring_forecast_id
    ON prediction_monitoring(forecast_id)
    """
)


cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_monitoring_sensor
    ON prediction_monitoring(sensor)
    """
)


cursor.execute(
    """
    CREATE INDEX IF NOT EXISTS
    idx_monitoring_timestamp
    ON prediction_monitoring(forecast_timestamp)
    """
)


conn.commit()


print(
    "Monitoring table is ready."
)


# ============================================================
# 17. MONITORING FUNCTION
# ============================================================

def evaluate_prediction(row):

    sensor = row["sensor"]

    predicted = float(
        row["predicted_value"]
    )

    reliability = MODEL_RELIABILITY.get(
        sensor,
        "UNKNOWN"
    )


    # ========================================================
    # MAGNETIC SENSORS
    # ========================================================
    #
    # Magnetic models have LOW predictive reliability.
    #
    # Therefore:
    #
    #   We do NOT treat a large current→prediction change
    #   as an anomaly.
    #
    #   We report LOW_CONFIDENCE instead.
    #
    # ========================================================

    if sensor in MAGNETIC_SENSORS:

        return {

            "status":
                "LOW_CONFIDENCE",

            "severity":
                "INFO",

            "alert_type":
                "MODEL_RELIABILITY_WARNING",

            "reason":
                (
                    "Magnetic forecasting model has low "
                    "predictive reliability based on "
                    "walk-forward evaluation."
                ),

            "model_reliability":
                reliability,
        }


    # ========================================================
    # VIBRATION
    # ========================================================

    if sensor in VIBRATION_SENSORS:

        if predicted >= VIBRATION_CRITICAL:

            return {

                "status":
                    "ANOMALY",

                "severity":
                    "CRITICAL",

                "alert_type":
                    "HIGH_VIBRATION",

                "reason":
                    (
                        f"Predicted vibration "
                        f"{predicted:.4f} exceeds "
                        f"critical threshold "
                        f"{VIBRATION_CRITICAL:.2f}."
                    ),

                "model_reliability":
                    reliability,
            }


        elif predicted >= VIBRATION_WARNING:

            return {

                "status":
                    "WARNING",

                "severity":
                    "MEDIUM",

                "alert_type":
                    "ELEVATED_VIBRATION",

                "reason":
                    (
                        f"Predicted vibration "
                        f"{predicted:.4f} exceeds "
                        f"warning threshold "
                        f"{VIBRATION_WARNING:.2f}."
                    ),

                "model_reliability":
                    reliability,
            }


        else:

            return {

                "status":
                    "NORMAL",

                "severity":
                    "INFO",

                "alert_type":
                    "NONE",

                "reason":
                    (
                        f"Predicted vibration "
                        f"{predicted:.4f} is below "
                        f"warning threshold."
                    ),

                "model_reliability":
                    reliability,
            }


    # ========================================================
    # TEMPERATURE
    # ========================================================

    if sensor == "temperature":

        if predicted >= TEMPERATURE_CRITICAL:

            return {

                "status":
                    "ANOMALY",

                "severity":
                    "CRITICAL",

                "alert_type":
                    "HIGH_TEMPERATURE",

                "reason":
                    (
                        f"Predicted temperature "
                        f"{predicted:.2f} °C exceeds "
                        f"critical threshold "
                        f"{TEMPERATURE_CRITICAL:.1f} °C."
                    ),

                "model_reliability":
                    reliability,
            }


        elif predicted >= TEMPERATURE_WARNING:

            return {

                "status":
                    "WARNING",

                "severity":
                    "MEDIUM",

                "alert_type":
                    "ELEVATED_TEMPERATURE",

                "reason":
                    (
                        f"Predicted temperature "
                        f"{predicted:.2f} °C exceeds "
                        f"warning threshold "
                        f"{TEMPERATURE_WARNING:.1f} °C."
                    ),

                "model_reliability":
                    reliability,
            }


        else:

            return {

                "status":
                    "NORMAL",

                "severity":
                    "INFO",

                "alert_type":
                    "NONE",

                "reason":
                    (
                        f"Predicted temperature "
                        f"{predicted:.2f} °C is below "
                        f"warning threshold."
                    ),

                "model_reliability":
                    reliability,
            }


    # ========================================================
    # UNKNOWN SENSOR
    # ========================================================

    return {

        "status":
            "UNKNOWN",

        "severity":
            "INFO",

        "alert_type":
            "UNKNOWN_SENSOR",

        "reason":
            "Sensor does not have a monitoring rule.",

        "model_reliability":
            reliability,
    }


# ============================================================
# 18. EVALUATE ALL FORECASTS
# ============================================================

print()
print("[3/7] Evaluating forecasts...")
print("-" * 75)


monitoring_records = []


for _, row in forecasts.iterrows():

    result = evaluate_prediction(
        row
    )


    monitoring_records.append(
        {

            "forecast_id":
                int(row["id"]),

            "feature_timestamp":
                str(row["feature_timestamp"]),

            "forecast_timestamp":
                str(row["forecast_timestamp"]),

            "horizon_seconds":
                int(row["horizon_seconds"]),

            "sensor":
                row["sensor"],

            "model":
                row["model"],

            "current_value":
                float(row["current_value"]),

            "predicted_value":
                float(row["predicted_value"]),

            "change_from_current":
                float(row["change_from_current"]),

            "status":
                result["status"],

            "severity":
                result["severity"],

            "alert_type":
                result["alert_type"],

            "reason":
                result["reason"],

            "model_reliability":
                result["model_reliability"],

            "created_at":
                str(pd.Timestamp.now()),
        }
    )


monitoring_df = pd.DataFrame(
    monitoring_records
)


print(
    f"Forecasts evaluated: "
    f"{len(monitoring_df):,}"
)


# ============================================================
# 19. SAVE MONITORING RESULTS
# ============================================================

print()
print("[4/7] Saving monitoring results...")
print("-" * 75)


# Clear existing monitoring results.
#
# This makes Step 29 reproducible and guarantees that the
# monitoring table represents the current forecast database.

cursor.execute(
    """
    DELETE FROM prediction_monitoring
    """
)

conn.commit()


insert_query = """
INSERT INTO prediction_monitoring
(
    forecast_id,
    feature_timestamp,
    forecast_timestamp,
    horizon_seconds,
    sensor,
    model,
    current_value,
    predicted_value,
    change_from_current,
    status,
    severity,
    alert_type,
    reason,
    model_reliability,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


records = [

    (

        row["forecast_id"],

        row["feature_timestamp"],

        row["forecast_timestamp"],

        row["horizon_seconds"],

        row["sensor"],

        row["model"],

        row["current_value"],

        row["predicted_value"],

        row["change_from_current"],

        row["status"],

        row["severity"],

        row["alert_type"],

        row["reason"],

        row["model_reliability"],

        row["created_at"],

    )

    for _, row
    in monitoring_df.iterrows()
]


cursor.executemany(
    insert_query,
    records
)


conn.commit()


print(
    f"Records inserted: "
    f"{len(records):,}"
)


# ============================================================
# 20. DATABASE COUNTS
# ============================================================

cursor.execute(
    """
    SELECT COUNT(*)
    FROM prediction_monitoring
    """
)


monitoring_count = (
    cursor.fetchone()[0]
)


cursor.execute(
    """
    SELECT COUNT(*)
    FROM forecast_predictions
    """
)


forecast_count = (
    cursor.fetchone()[0]
)


print(
    f"Forecast records: "
    f"{forecast_count:,}"
)

print(
    f"Monitoring records: "
    f"{monitoring_count:,}"
)


# ============================================================
# 21. CHECK ONE-TO-ONE RELATIONSHIP
# ============================================================

relationship_ok = (
    forecast_count
    == monitoring_count
)


if relationship_ok:

    print(
        "Forecast → monitoring relationship: PASS"
    )

else:

    print(
        "Forecast → monitoring relationship: FAIL"
    )


# ============================================================
# 22. STATUS SUMMARY
# ============================================================

print()
print("[5/7] Monitoring summary")
print("-" * 75)


status_summary = (
    monitoring_df[
        "status"
    ]
    .value_counts()
)


for status, count in status_summary.items():

    print(
        f"  {status:20s} "
        f"{count:6,}"
    )


# ============================================================
# 23. SENSOR SUMMARY
# ============================================================

print()
print("Sensor-wise monitoring:")
print("-" * 75)


sensor_summary = (
    monitoring_df
    .groupby(
        [
            "sensor",
            "status",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


for _, row in sensor_summary.iterrows():

    sensor_name = DISPLAY_NAMES.get(
        row["sensor"],
        row["sensor"]
    )


    print(
        f"  {sensor_name:20s} "
        f"{row['status']:20s} "
        f"{row['count']:6,}"
    )


# ============================================================
# 24. IMPORTANT EVENTS
# ============================================================

print()
print("[6/7] Important events")
print("-" * 75)


important_events = monitoring_df[
    monitoring_df["status"].isin(
        [
            "WARNING",
            "ANOMALY",
        ]
    )
]


if important_events.empty:

    print(
        "  No vibration or temperature "
        "warnings/anomalies detected."
    )

else:

    print(
        f"  Important events: "
        f"{len(important_events):,}"
    )


    recent_events = (
        important_events
        .sort_values(
            "forecast_timestamp",
            ascending=False
        )
        .head(10)
    )


    for _, row in recent_events.iterrows():

        sensor_name = DISPLAY_NAMES.get(
            row["sensor"],
            row["sensor"]
        )


        print(
            f"  {row['forecast_timestamp']} | "
            f"{sensor_name:15s} | "
            f"{row['status']:8s} | "
            f"predicted="
            f"{row['predicted_value']:.4f}"
        )


# ============================================================
# 25. SAVE REPORT
# ============================================================

monitoring_df.to_csv(
    REPORT_PATH,
    index=False
)


print()
print("[7/7] Report saved")
print("-" * 75)

print(
    f"  {REPORT_PATH}"
)


# ============================================================
# 26. CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# 27. FINAL STATUS
# ============================================================

print()
print("=" * 75)
print("STEP 29 COMPLETE")
print("=" * 75)

print(
    f"Forecasts monitored: "
    f"{len(monitoring_df):,}"
)

print(
    f"Monitoring records inserted: "
    f"{monitoring_count:,}"
)

print(
    f"Forecast records in database: "
    f"{forecast_count:,}"
)

print(
    f"\nDatabase:"
)

print(
    f"  {DB_PATH}"
)

print(
    f"\nReport:"
)

print(
    f"  {REPORT_PATH}"
)

print("=" * 75)

print(
    "\nReal-time monitoring service completed successfully."
)