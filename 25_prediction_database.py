import os
import sqlite3
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts",
    "step24_simulated_realtime_predictions.csv"
)

DB_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "database"
)

DB_FILE = os.path.join(
    DB_DIR,
    "forecasting.db"
)


# ============================================================
# CREATE DATABASE DIRECTORY
# ============================================================

os.makedirs(DB_DIR, exist_ok=True)


# ============================================================
# LOAD STEP 24 FORECASTS
# ============================================================

print("=" * 70)
print("STEP 25 - PREDICTION DATABASE")
print("=" * 70)

print("\nLoading Step 24 predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded rows: {len(df):,}")


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
    "temperature"
]


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

model_info = {}

for sensor in SENSORS:

    model_file = os.path.join(
        BASE_DIR,
        "outputs",
        "models",
        f"step21_final_{sensor}.joblib"
    )

    artifact = joblib.load(model_file)

    model_info[sensor] = {
        "model_name": artifact.get("model_name", "Unknown"),
        "horizon_seconds": artifact.get("horizon_seconds", 60)
    }


# ============================================================
# CONNECT TO SQLITE
# ============================================================

print("\nCreating database...")

conn = sqlite3.connect(DB_FILE)

cursor = conn.cursor()


# ============================================================
# CREATE TABLE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS forecast_predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    feature_timestamp TEXT NOT NULL,

    forecast_timestamp TEXT NOT NULL,

    horizon_seconds INTEGER NOT NULL,

    sensor TEXT NOT NULL,

    model TEXT NOT NULL,

    current_value REAL,

    predicted_value REAL,

    change_from_current REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(feature_timestamp, forecast_timestamp, sensor)

)
""")


# ============================================================
# INSERT FORECASTS
# ============================================================

print("\nInserting predictions...")

inserted = 0

for _, row in df.iterrows():

    feature_timestamp = row["feature_timestamp"]
    forecast_timestamp = row["forecast_timestamp"]

    for sensor in SENSORS:

        current_column = f"{sensor}_current"
        prediction_column = f"{sensor}_prediction"
        change_column = f"{sensor}_change"

        current_value = row[current_column]
        predicted_value = row[prediction_column]
        change_value = row[change_column]

        model_name = model_info[sensor]["model_name"]
        horizon_seconds = model_info[sensor]["horizon_seconds"]

        cursor.execute("""
        INSERT OR IGNORE INTO forecast_predictions
        (
            feature_timestamp,
            forecast_timestamp,
            horizon_seconds,
            sensor,
            model,
            current_value,
            predicted_value,
            change_from_current
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feature_timestamp,
            forecast_timestamp,
            horizon_seconds,
            sensor,
            model_name,
            float(current_value),
            float(predicted_value),
            float(change_value)
        ))

        inserted += cursor.rowcount


# ============================================================
# COMMIT
# ============================================================

conn.commit()


# ============================================================
# DATABASE VALIDATION
# ============================================================

print("\nValidating database...")

cursor.execute("""
SELECT COUNT(*)
FROM forecast_predictions
""")

total_records = cursor.fetchone()[0]


cursor.execute("""
SELECT COUNT(DISTINCT feature_timestamp)
FROM forecast_predictions
""")

forecast_points = cursor.fetchone()[0]


cursor.execute("""
SELECT DISTINCT sensor
FROM forecast_predictions
ORDER BY sensor
""")

sensors_in_db = [row[0] for row in cursor.fetchall()]


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\nSample database records:")

sample = pd.read_sql_query("""
SELECT
    id,
    feature_timestamp,
    forecast_timestamp,
    sensor,
    model,
    current_value,
    predicted_value,
    change_from_current
FROM forecast_predictions
ORDER BY id
LIMIT 10
""", conn)

print(sample.to_string(index=False))


# ============================================================
# SENSOR COUNTS
# ============================================================

print("\nRecords per sensor:")

sensor_counts = pd.read_sql_query("""
SELECT
    sensor,
    COUNT(*) AS records
FROM forecast_predictions
GROUP BY sensor
ORDER BY sensor
""", conn)

print(sensor_counts.to_string(index=False))


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)

print(f"Database: {DB_FILE}")
print(f"Total records: {total_records:,}")
print(f"Forecast timestamps: {forecast_points:,}")
print(f"Sensors: {len(sensors_in_db)}")

print("\nSensors stored:")

for sensor in sensors_in_db:
    print(f"  - {sensor}")

print("\nDatabase successfully created.")