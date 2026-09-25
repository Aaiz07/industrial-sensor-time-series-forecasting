import os
import sqlite3
import pandas as pd


# ============================================================
# STEP 26 - PREDICTION MONITORING & ANOMALY DETECTION
# ============================================================

print("=" * 70)
print("STEP 26 - PREDICTION MONITORING & ANOMALY DETECTION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

DB_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "database",
    "forecasting.db"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

REPORT_FILE = os.path.join(
    REPORT_DIR,
    "step26_prediction_monitoring.csv"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# CHECK DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(
        f"Database not found:\n{DB_FILE}\n"
        "Run Step 25 first."
    )


# ============================================================
# CONNECT TO DATABASE
# ============================================================

conn = sqlite3.connect(DB_FILE)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading predictions from database...")

df = pd.read_sql_query(
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
        change_from_current
    FROM forecast_predictions
    ORDER BY feature_timestamp, sensor
    """,
    conn
)

print(f"Loaded predictions: {len(df):,}")


if df.empty:
    raise ValueError("No prediction records found.")


# ============================================================
# MONITORING THRESHOLDS
# ============================================================

# These are project monitoring thresholds.
# They are NOT machine failure limits.

VIBRATION_WARNING = {
    "vib_x_rms": 0.10,
    "vib_y_rms": 0.10,
    "vib_z_rms": 0.10
}

VIBRATION_CRITICAL = {
    "vib_x_rms": 0.20,
    "vib_y_rms": 0.20,
    "vib_z_rms": 0.20
}

TEMPERATURE_WARNING = 48.0

TEMPERATURE_CRITICAL = 50.0


# ============================================================
# MODEL RELIABILITY
# ============================================================

# Based on the walk-forward evaluation from Step 20.

MODEL_RELIABILITY = {

    "vib_x_rms": "MODERATE",
    "vib_y_rms": "MODERATE",
    "vib_z_rms": "MODERATE",

    "mag_x": "LOW",
    "mag_y": "LOW",
    "mag_z": "LOW",

    "temperature": "HIGH"
}


df["model_reliability"] = df["sensor"].map(
    MODEL_RELIABILITY
)


# ============================================================
# INITIALIZE MONITORING RESULTS
# ============================================================

df["status"] = "NORMAL"

df["severity"] = "NORMAL"

df["alert_type"] = "NONE"

df["alert_reason"] = ""


# ============================================================
# MONITOR EACH PREDICTION
# ============================================================

for index, row in df.iterrows():

    sensor = row["sensor"]

    predicted = row["predicted_value"]

    # --------------------------------------------------------
    # VIBRATION
    # --------------------------------------------------------

    if sensor in VIBRATION_WARNING:

        warning_limit = VIBRATION_WARNING[sensor]

        critical_limit = VIBRATION_CRITICAL[sensor]

        if predicted >= critical_limit:

            df.at[index, "status"] = "ANOMALY"

            df.at[index, "severity"] = "CRITICAL"

            df.at[index, "alert_type"] = "HIGH_VIBRATION"

            df.at[index, "alert_reason"] = (
                f"Predicted vibration "
                f"{predicted:.6f} exceeds critical limit "
                f"{critical_limit:.2f}"
            )

        elif predicted >= warning_limit:

            df.at[index, "status"] = "WARNING"

            df.at[index, "severity"] = "MEDIUM"

            df.at[index, "alert_type"] = "ELEVATED_VIBRATION"

            df.at[index, "alert_reason"] = (
                f"Predicted vibration "
                f"{predicted:.6f} exceeds warning limit "
                f"{warning_limit:.2f}"
            )


    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    elif sensor == "temperature":

        if predicted >= TEMPERATURE_CRITICAL:

            df.at[index, "status"] = "ANOMALY"

            df.at[index, "severity"] = "CRITICAL"

            df.at[index, "alert_type"] = "HIGH_TEMPERATURE"

            df.at[index, "alert_reason"] = (
                f"Predicted temperature "
                f"{predicted:.2f} exceeds critical limit "
                f"{TEMPERATURE_CRITICAL:.2f}"
            )

        elif predicted >= TEMPERATURE_WARNING:

            df.at[index, "status"] = "WARNING"

            df.at[index, "severity"] = "MEDIUM"

            df.at[index, "alert_type"] = "ELEVATED_TEMPERATURE"

            df.at[index, "alert_reason"] = (
                f"Predicted temperature "
                f"{predicted:.2f} exceeds warning limit "
                f"{TEMPERATURE_WARNING:.2f}"
            )


    # --------------------------------------------------------
    # MAGNETIC SENSORS
    # --------------------------------------------------------

    elif sensor in ["mag_x", "mag_y", "mag_z"]:

        # We deliberately DO NOT use percentage change here.
        #
        # The magnetic Ridge models have approximately zero
        # forecasting R² in walk-forward evaluation.
        #
        # Therefore their output should be stored but marked
        # as low-confidence rather than treated as an anomaly.

        df.at[index, "status"] = "LOW_CONFIDENCE"

        df.at[index, "severity"] = "INFO"

        df.at[index, "alert_type"] = (
            "MODEL_RELIABILITY_WARNING"
        )

        df.at[index, "alert_reason"] = (
            "Magnetic forecasting model has low predictive "
            "reliability based on walk-forward evaluation."
        )


# ============================================================
# CREATE MONITORING TABLE
# ============================================================

print("\nCreating monitoring table...")

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS prediction_monitoring (

        monitoring_id INTEGER PRIMARY KEY AUTOINCREMENT,

        prediction_id INTEGER NOT NULL,

        feature_timestamp TEXT NOT NULL,

        forecast_timestamp TEXT NOT NULL,

        sensor TEXT NOT NULL,

        model TEXT NOT NULL,

        current_value REAL,

        predicted_value REAL,

        change_from_current REAL,

        status TEXT NOT NULL,

        severity TEXT NOT NULL,

        alert_type TEXT NOT NULL,

        model_reliability TEXT,

        alert_reason TEXT,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP,

        UNIQUE(prediction_id)

    )
    """
)


# ============================================================
# CLEAR OLD STEP 26 RESULTS
# ============================================================

# This makes Step 26 reproducible.
#
# If you modify monitoring thresholds and rerun the script,
# the database will contain only the latest Step 26 results.

conn.execute(
    "DELETE FROM prediction_monitoring"
)

conn.commit()


# ============================================================
# INSERT MONITORING RESULTS
# ============================================================

print("\nSaving monitoring results...")

inserted = 0

for _, row in df.iterrows():

    cursor = conn.execute(
        """
        INSERT INTO prediction_monitoring
        (
            prediction_id,
            feature_timestamp,
            forecast_timestamp,
            sensor,
            model,
            current_value,
            predicted_value,
            change_from_current,
            status,
            severity,
            alert_type,
            model_reliability,
            alert_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(row["id"]),
            row["feature_timestamp"],
            row["forecast_timestamp"],
            row["sensor"],
            row["model"],
            float(row["current_value"]),
            float(row["predicted_value"]),
            float(row["change_from_current"]),
            row["status"],
            row["severity"],
            row["alert_type"],
            row["model_reliability"],
            row["alert_reason"]
        )
    )

    inserted += cursor.rowcount


conn.commit()


# ============================================================
# SAVE COMPLETE MONITORING REPORT
# ============================================================

report_columns = [
    "id",
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
    "model_reliability",
    "alert_reason"
]

df[report_columns].to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MONITORING SUMMARY")
print("=" * 70)


summary = pd.read_sql_query(
    """
    SELECT
        status,
        COUNT(*) AS count
    FROM prediction_monitoring
    GROUP BY status
    ORDER BY count DESC
    """,
    conn
)

print(
    summary.to_string(index=False)
)


# ============================================================
# SENSOR SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY BY SENSOR")
print("=" * 70)


sensor_summary = pd.read_sql_query(
    """
    SELECT
        sensor,
        status,
        COUNT(*) AS count
    FROM prediction_monitoring
    GROUP BY sensor, status
    ORDER BY sensor, status
    """,
    conn
)

print(
    sensor_summary.to_string(index=False)
)


# ============================================================
# CRITICAL / WARNING EVENTS
# ============================================================

print("\n" + "=" * 70)
print("WARNING / CRITICAL EVENTS")
print("=" * 70)


important_events = pd.read_sql_query(
    """
    SELECT
        monitoring_id,
        feature_timestamp,
        forecast_timestamp,
        sensor,
        predicted_value,
        status,
        severity,
        alert_type,
        alert_reason
    FROM prediction_monitoring
    WHERE severity IN ('CRITICAL', 'MEDIUM')
    ORDER BY feature_timestamp
    """,
    conn
)

if important_events.empty:

    print("No warning or critical events detected.")

else:

    print(
        important_events.to_string(index=False)
    )


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("STEP 26 COMPLETE")
print("=" * 70)

print(
    f"Predictions monitored : {len(df):,}"
)

print(
    f"Records inserted      : {inserted:,}"
)

print(
    f"\nMonitoring report:\n{REPORT_FILE}"
)

print(
    f"\nDatabase:\n{DB_FILE}"
)

print(
    "\nStep 26 monitoring system successfully updated."
)