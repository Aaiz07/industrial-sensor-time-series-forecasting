# ============================================================
# STEP 28 — REAL-TIME FORECASTING SERVICE
# ============================================================
#
# Purpose:
#   Production-style simulated real-time forecasting service.
#
# Flow:
#
#   Historical / Incoming Sensor Data
#                 ↓
#          Feature Generation
#                 ↓
#          Exact 146 Features
#                 ↓
#           Final ML Models
#                 ↓
#          60-sec Forecast
#                 ↓
#           SQLite Database
#
# Models selected by walk-forward validation:
#
#   vib_x_rms    -> XGBoost
#   vib_y_rms    -> XGBoost
#   vib_z_rms    -> XGBoost
#   mag_x        -> Ridge
#   mag_y        -> Ridge
#   mag_z        -> Ridge
#   temperature  -> Persistence
#
# ============================================================

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "outputs"
    / "data"
    / "sensor_data_cleaned.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "models"
)

DB_PATH = (
    BASE_DIR
    / "outputs"
    / "database"
    / "forecasting.db"
)


# ============================================================
# 2. SERVICE CONFIGURATION
# ============================================================

HORIZON_SECONDS = 60

# Our feature schema requires lag_60.
HISTORY_REQUIRED = 60

# Number of simulated real-time observations.
MAX_RECORDS = 500


SENSORS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]


LEARNED_SENSORS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
]


MODEL_FILES = {

    "vib_x_rms":
        MODEL_DIR / "step21_final_vib_x_rms.joblib",

    "vib_y_rms":
        MODEL_DIR / "step21_final_vib_y_rms.joblib",

    "vib_z_rms":
        MODEL_DIR / "step21_final_vib_z_rms.joblib",

    "mag_x":
        MODEL_DIR / "step21_final_mag_x.joblib",

    "mag_y":
        MODEL_DIR / "step21_final_mag_y.joblib",

    "mag_z":
        MODEL_DIR / "step21_final_mag_z.joblib",
}


# ============================================================
# 3. START
# ============================================================

print("=" * 75)
print("STEP 28 — REAL-TIME FORECASTING SERVICE")
print("=" * 75)


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nSensor data not found:\n{DATA_PATH}"
    )


if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"\nModel directory not found:\n{MODEL_DIR}"
    )


DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. LOAD SENSOR DATA
# ============================================================

print("\n[1/7] Loading sensor data...")

df = pd.read_csv(DATA_PATH)

df["ts"] = pd.to_datetime(
    df["ts"]
)

df = (
    df
    .sort_values("ts")
    .drop_duplicates("ts")
    .reset_index(drop=True)
)


required_columns = [
    "ts",
    *SENSORS
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing sensor columns: "
        f"{missing_columns}"
    )


print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Time range: "
    f"{df['ts'].min()} → {df['ts'].max()}"
)


# ============================================================
# 6. LOAD FINAL MODELS
# ============================================================

print("\n[2/7] Loading final models...")

models = {}


for sensor, model_path in MODEL_FILES.items():

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )


    artifact = joblib.load(
        model_path
    )


    if not isinstance(
        artifact,
        dict
    ):

        raise ValueError(
            f"{sensor}: "
            f"Expected model artifact dictionary."
        )


    if "model" not in artifact:

        raise ValueError(
            f"{sensor}: "
            f"'model' missing from artifact."
        )


    if "features" not in artifact:

        raise ValueError(
            f"{sensor}: "
            f"'features' missing from artifact."
        )


    models[sensor] = {

        "model":
            artifact["model"],

        "features":
            artifact["features"],

        "model_name":
            artifact.get(
                "model_name",
                type(
                    artifact["model"]
                ).__name__
            ),

        "horizon_seconds":
            artifact.get(
                "horizon_seconds",
                HORIZON_SECONDS
            )
    }


    print(
        f"  {sensor:15s} "
        f"model={models[sensor]['model_name']:12s} "
        f"features={len(models[sensor]['features'])}"
    )


# ============================================================
# 7. MODEL VALIDATION
# ============================================================

print("\n[3/7] Validating model configuration...")


for sensor in LEARNED_SENSORS:

    feature_count = len(
        models[sensor]["features"]
    )


    if feature_count != 146:

        raise ValueError(
            f"{sensor}: "
            f"Expected 146 features, "
            f"found {feature_count}"
        )


    horizon = models[sensor][
        "horizon_seconds"
    ]


    if horizon != HORIZON_SECONDS:

        raise ValueError(
            f"{sensor}: "
            f"Expected {HORIZON_SECONDS}-second horizon, "
            f"found {horizon}"
        )


    print(
        f"  {sensor}: "
        f"146 features → OK | "
        f"horizon={horizon}s"
    )


print(
    "  temperature: "
    "Persistence → OK"
)


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

def create_features(history):
    """
    Generate the same 146-feature schema used by Step 21.

    All features are calculated using historical/current
    observations only.

    No future target is used.
    """

    data = history.copy()


    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    seconds_of_day = (
        data["ts"].dt.hour * 3600
        + data["ts"].dt.minute * 60
        + data["ts"].dt.second
    )


    time_features = pd.DataFrame(
        {
            "time_sin":
                np.sin(
                    2
                    * np.pi
                    * seconds_of_day
                    / 86400
                ),

            "time_cos":
                np.cos(
                    2
                    * np.pi
                    * seconds_of_day
                    / 86400
                )
        },
        index=data.index
    )


    # --------------------------------------------------------
    # BASIC LAGS + DIFFERENCES
    # --------------------------------------------------------

    lag_features = {}


    for sensor in SENSORS:

        for lag in [
            1,
            2,
            5,
            10,
            30,
            60
        ]:

            lag_features[
                f"{sensor}_lag_{lag}"
            ] = (
                data[sensor]
                .shift(lag)
            )


        lag_features[
            f"{sensor}_diff_1"
        ] = (
            data[sensor]
            .diff(1)
        )


    lag_features = pd.DataFrame(
        lag_features,
        index=data.index
    )


    # --------------------------------------------------------
    # CROSS-AXIS VIBRATION RATIOS
    # --------------------------------------------------------

    ratio_features = {}


    vibration_pairs = [

        (
            "vib_y_rms",
            "vib_x_rms"
        ),

        (
            "vib_z_rms",
            "vib_x_rms"
        ),

        (
            "vib_x_rms",
            "vib_y_rms"
        ),

        (
            "vib_z_rms",
            "vib_y_rms"
        ),

        (
            "vib_x_rms",
            "vib_z_rms"
        ),

        (
            "vib_y_rms",
            "vib_z_rms"
        )
    ]


    for numerator, denominator in vibration_pairs:

        ratio_name = (
            f"{numerator}_to_{denominator}"
        )


        numerator_values = (
            data[numerator]
        )


        denominator_values = (
            data[denominator]
            .replace(
                0,
                np.nan
            )
        )


        for lag in [
            1,
            2,
            5,
            10
        ]:

            ratio_features[
                f"{ratio_name}_lag_{lag}"
            ] = (

                numerator_values
                .shift(lag)

                /

                denominator_values
                .shift(lag)
            )


    ratio_features = pd.DataFrame(
        ratio_features,
        index=data.index
    )


    # --------------------------------------------------------
    # MAGNETIC FEATURES
    # --------------------------------------------------------

    magnetic_features = {}


    for sensor in [
        "mag_x",
        "mag_y",
        "mag_z"
    ]:

        shifted = (
            data[sensor]
            .shift(1)
        )


        # Magnetic lag features

        for lag in [
            1,
            5,
            10,
            30,
            60
        ]:

            magnetic_features[
                f"{sensor}_mag_lag_{lag}"
            ] = (
                data[sensor]
                .shift(lag)
            )


        # Rolling statistics

        for window in [
            5,
            15,
            30,
            60
        ]:

            magnetic_features[
                f"{sensor}_rollmean_{window}"
            ] = (
                shifted
                .rolling(window)
                .mean()
            )


            magnetic_features[
                f"{sensor}_rollstd_{window}"
            ] = (
                shifted
                .rolling(window)
                .std()
            )


    magnetic_features = pd.DataFrame(
        magnetic_features,
        index=data.index
    )


    # --------------------------------------------------------
    # VIBRATION ROLLING FEATURES
    # --------------------------------------------------------

    vibration_features = {}


    for sensor in [
        "vib_x_rms",
        "vib_y_rms",
        "vib_z_rms"
    ]:

        shifted = (
            data[sensor]
            .shift(1)
        )


        for window in [
            5,
            15,
            30,
            60
        ]:

            vibration_features[
                f"{sensor}_rollmean_{window}"
            ] = (
                shifted
                .rolling(window)
                .mean()
            )


            vibration_features[
                f"{sensor}_rollstd_{window}"
            ] = (
                shifted
                .rolling(window)
                .std()
            )


    vibration_features = pd.DataFrame(
        vibration_features,
        index=data.index
    )


    # --------------------------------------------------------
    # TEMPERATURE ROLLING FEATURES
    # --------------------------------------------------------

    temperature_shifted = (
        data["temperature"]
        .shift(1)
    )


    temperature_features = {}


    for window in [
        5,
        15,
        30,
        60
    ]:

        temperature_features[
            f"temperature_rollmean_{window}"
        ] = (
            temperature_shifted
            .rolling(window)
            .mean()
        )


        temperature_features[
            f"temperature_rollstd_{window}"
        ] = (
            temperature_shifted
            .rolling(window)
            .std()
        )


    temperature_features = pd.DataFrame(
        temperature_features,
        index=data.index
    )


    # --------------------------------------------------------
    # COMBINE ALL FEATURES AT ONCE
    # --------------------------------------------------------
    #
    # This is the important fix for the previous
    # "DataFrame is highly fragmented" warning.
    #

    feature_data = pd.concat(
        [
            time_features,
            lag_features,
            ratio_features,
            magnetic_features,
            vibration_features,
            temperature_features
        ],
        axis=1
    )


    return feature_data


# ============================================================
# 9. CREATE DATABASE
# ============================================================

print("\n[4/7] Preparing SQLite database...")


conn = sqlite3.connect(
    DB_PATH
)


cursor = conn.cursor()


cursor.execute(
    """
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

        created_at TEXT NOT NULL,

        UNIQUE(
            feature_timestamp,
            forecast_timestamp,
            sensor
        )
    )
    """
)


conn.commit()


print(
    f"Database ready:\n{DB_PATH}"
)


# ============================================================
# 10. SERVICE SETTINGS
# ============================================================

available_records = (
    len(df)
    - HISTORY_REQUIRED
)


records_to_process = min(
    MAX_RECORDS,
    available_records
)


if records_to_process <= 0:

    conn.close()

    raise ValueError(
        "Not enough sensor history "
        "to create forecasts."
    )


start_index = HISTORY_REQUIRED


print(
    f"\n[5/7] Service configuration"
)

print(
    f"  History required: "
    f"{HISTORY_REQUIRED} observations"
)

print(
    f"  Records to process: "
    f"{records_to_process}"
)

print(
    f"  Sensors per forecast: "
    f"{len(SENSORS)}"
)

print(
    f"  Forecast horizon: "
    f"{HORIZON_SECONDS} seconds"
)


# ============================================================
# 11. FORECASTING LOOP
# ============================================================

print(
    "\n[6/7] Starting forecasting service..."
)

print(
    "-" * 75
)


processed = 0
inserted = 0
skipped = 0


for index in range(
    start_index,
    start_index + records_to_process
):

    try:

        # ----------------------------------------------------
        # Historical data available at current time
        # ----------------------------------------------------

        history = (
            df
            .iloc[: index + 1]
            .copy()
        )


        feature_timestamp = (
            df.iloc[index]["ts"]
        )


        # ----------------------------------------------------
        # Generate complete feature matrix
        # ----------------------------------------------------

        feature_data = create_features(
            history
        )


        # ----------------------------------------------------
        # Latest feature row
        # ----------------------------------------------------

        latest_features = (
            feature_data
            .iloc[[-1]]
            .copy()
        )


        # ----------------------------------------------------
        # Forecast timestamp
        # ----------------------------------------------------

        forecast_timestamp = (
            feature_timestamp
            + pd.Timedelta(
                seconds=HORIZON_SECONDS
            )
        )


        predictions_for_timestamp = []


        # ====================================================
        # SENSOR PREDICTIONS
        # ====================================================

        for sensor in SENSORS:

            current_value = float(
                df.iloc[index][sensor]
            )


            # ------------------------------------------------
            # TEMPERATURE
            # ------------------------------------------------

            if sensor == "temperature":

                predicted_value = (
                    current_value
                )

                model_name = (
                    "Persistence"
                )


            # ------------------------------------------------
            # LEARNED MODELS
            # ------------------------------------------------

            else:

                model_info = (
                    models[sensor]
                )


                model = (
                    model_info["model"]
                )


                feature_names = (
                    model_info["features"]
                )


                # --------------------------------------------
                # Verify exact feature schema
                # --------------------------------------------

                missing_features = [
                    feature
                    for feature in feature_names
                    if feature
                    not in latest_features.columns
                ]


                if missing_features:

                    raise ValueError(
                        f"{sensor}: "
                        f"Missing features: "
                        f"{missing_features[:5]}"
                    )


                if len(feature_names) != 146:

                    raise ValueError(
                        f"{sensor}: "
                        f"Expected 146 features."
                    )


                # --------------------------------------------
                # IMPORTANT:
                #
                # Select features in EXACT order.
                # --------------------------------------------

                X_df = (
                    latest_features[
                        feature_names
                    ]
                    .copy()
                )


                # --------------------------------------------
                # Convert to NumPy.
                #
                # This avoids:
                #
                #   X has feature names...
                #
                # and avoids the XGBoost pandas
                # compatibility issue.
                # --------------------------------------------

                X = (
                    X_df
                    .to_numpy(
                        dtype=np.float64
                    )
                )


                # --------------------------------------------
                # Validate dimensions
                # --------------------------------------------

                if X.shape != (
                    1,
                    146
                ):

                    raise ValueError(
                        f"{sensor}: "
                        f"Invalid feature shape "
                        f"{X.shape}"
                    )


                # --------------------------------------------
                # Validate numerical values
                # --------------------------------------------

                if not np.isfinite(
                    X
                ).all():

                    raise ValueError(
                        f"{sensor}: "
                        f"NaN or infinite "
                        f"value detected."
                    )


                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = (
                    model.predict(X)
                )


                if len(prediction) != 1:

                    raise ValueError(
                        f"{sensor}: "
                        f"Unexpected prediction output."
                    )


                predicted_value = float(
                    prediction[0]
                )


                model_name = (
                    model_info["model_name"]
                )


            # ------------------------------------------------
            # Change from current
            # ------------------------------------------------

            change_from_current = (
                predicted_value
                - current_value
            )


            predictions_for_timestamp.append(
                (
                    str(feature_timestamp),
                    str(forecast_timestamp),
                    HORIZON_SECONDS,
                    sensor,
                    model_name,
                    current_value,
                    predicted_value,
                    change_from_current,
                    str(pd.Timestamp.now())
                )
            )


        # ====================================================
        # SAVE TO DATABASE
        # ====================================================

        cursor.executemany(
            """
            INSERT OR IGNORE INTO forecast_predictions
            (
                feature_timestamp,
                forecast_timestamp,
                horizon_seconds,
                sensor,
                model,
                current_value,
                predicted_value,
                change_from_current,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions_for_timestamp
        )


        conn.commit()


        # ====================================================
        # COUNTERS
        # ====================================================

        processed += 1

        inserted += len(
            predictions_for_timestamp
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            f"\n[{processed:04d}/"
            f"{records_to_process:04d}] "
            f"{feature_timestamp}"
        )


        print(
            f"  Forecast → "
            f"{forecast_timestamp}"
        )


        for row in predictions_for_timestamp:

            (
                feature_ts,
                forecast_ts,
                horizon,
                sensor,
                model_name,
                current,
                predicted,
                change,
                created
            ) = row


            print(
                f"  {sensor:15s} "
                f"current={current:10.4f} "
                f"predicted={predicted:10.4f}"
            )


    except Exception as error:

        skipped += 1


        print(
            f"\nERROR at index {index}:"
        )

        print(
            f"  {error}"
        )


        continue


# ============================================================
# 12. CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("STEP 28 COMPLETE")
print("=" * 75)


print(
    f"Records processed: "
    f"{processed:,}"
)


print(
    f"Forecast records attempted: "
    f"{processed * len(SENSORS):,}"
)


print(
    f"Database: "
    f"{DB_PATH}"
)


print(
    f"Skipped records: "
    f"{skipped:,}"
)


print(
    f"Forecast horizon: "
    f"{HORIZON_SECONDS} seconds"
)


print(
    "\nFinal models used:"
)


for sensor in LEARNED_SENSORS:

    print(
        f"  {sensor:15s} → "
        f"{models[sensor]['model_name']}"
    )


print(
    "  temperature     → Persistence"
)


print("=" * 75)


if skipped == 0:

    print(
        "\nSUCCESS: "
        "Real-time forecasting service "
        "completed without errors."
    )

else:

    print(
        "\nWARNING: "
        f"{skipped} records were skipped."
    )