# ============================================================
# STEP 24: SIMULATED REAL-TIME FORECASTING
# ============================================================
#
# Replays the cleaned sensor CSV as if observations are
# arriving in real time.
#
# No model training is performed.
#
# Uses the exact Step-21 model artifacts and feature schema.
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

DATA_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "step24_simulated_realtime_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SENSORS
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


MODEL_FILES = {
    "vib_x_rms": "step21_final_vib_x_rms.joblib",
    "vib_y_rms": "step21_final_vib_y_rms.joblib",
    "vib_z_rms": "step21_final_vib_z_rms.joblib",
    "mag_x": "step21_final_mag_x.joblib",
    "mag_y": "step21_final_mag_y.joblib",
    "mag_z": "step21_final_mag_z.joblib",
    "temperature": "step21_final_temperature.joblib"
}


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 75)
print("STEP 24: SIMULATED REAL-TIME FORECASTING")
print("=" * 75)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n[1/6] Loading sensor data...")

df = pd.read_csv(DATA_FILE)

df["ts"] = pd.to_datetime(df["ts"])

df = (
    df
    .sort_values("ts")
    .reset_index(drop=True)
)

print(f"Rows: {len(df):,}")
print(f"Start: {df['ts'].min()}")
print(f"End:   {df['ts'].max()}")


# ============================================================
# 5. LOAD MODELS
# ============================================================

print("\n[2/6] Loading models...")

artifacts = {}

for sensor in SENSORS:

    path = os.path.join(
        MODEL_DIR,
        MODEL_FILES[sensor]
    )

    artifact = joblib.load(path)

    artifacts[sensor] = artifact

    print(
        f"{sensor:15s} -> "
        f"{artifact['model_name']:12s} | "
        f"features = {len(artifact['features'])}"
    )


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\n[3/6] Creating feature pipeline...")


def create_features(data):

    data = data.copy()

    data = (
        data
        .sort_values("ts")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    seconds_since_midnight = (
        data["ts"].dt.hour * 3600
        + data["ts"].dt.minute * 60
        + data["ts"].dt.second
    )

    data["time_sin"] = np.sin(
        2 * np.pi *
        seconds_since_midnight / 86400
    )

    data["time_cos"] = np.cos(
        2 * np.pi *
        seconds_since_midnight / 86400
    )


    # --------------------------------------------------------
    # Standard lags
    # --------------------------------------------------------

    lags = [
        1,
        2,
        5,
        10,
        30,
        60
    ]

    for sensor in SENSORS:

        for lag in lags:

            data[
                f"{sensor}_lag_{lag}"
            ] = data[sensor].shift(lag)

        data[
            f"{sensor}_diff_1"
        ] = (
            data[sensor]
            - data[sensor].shift(1)
        )


    # --------------------------------------------------------
    # Cross-vibration features
    # --------------------------------------------------------

    vibration_sensors = [
        "vib_x_rms",
        "vib_y_rms",
        "vib_z_rms"
    ]

    cross_lags = [
        1,
        2,
        5,
        10
    ]

    for target in vibration_sensors:

        for source in vibration_sensors:

            if source == target:
                continue

            for lag in cross_lags:

                data[
                    f"{source}_to_{target}_lag_{lag}"
                ] = data[source].shift(lag)


    # --------------------------------------------------------
    # Magnetic features
    # --------------------------------------------------------

    magnetic_sensors = [
        "mag_x",
        "mag_y",
        "mag_z"
    ]

    magnetic_lags = [
        1,
        5,
        10,
        30,
        60
    ]

    rolling_windows = [
        5,
        15,
        30,
        60
    ]

    for sensor in magnetic_sensors:

        shifted = data[sensor].shift(1)

        for lag in magnetic_lags:

            data[
                f"{sensor}_mag_lag_{lag}"
            ] = data[sensor].shift(lag)

        for window in rolling_windows:

            data[
                f"{sensor}_rollmean_{window}"
            ] = (
                shifted
                .rolling(window)
                .mean()
            )

            data[
                f"{sensor}_rollstd_{window}"
            ] = (
                shifted
                .rolling(window)
                .std()
            )


    # --------------------------------------------------------
    # Vibration rolling features
    # --------------------------------------------------------

    for sensor in vibration_sensors:

        shifted = data[sensor].shift(1)

        for window in rolling_windows:

            data[
                f"{sensor}_rollmean_{window}"
            ] = (
                shifted
                .rolling(window)
                .mean()
            )

            data[
                f"{sensor}_rollstd_{window}"
            ] = (
                shifted
                .rolling(window)
                .std()
            )


    # --------------------------------------------------------
    # Temperature rolling features
    # --------------------------------------------------------

    shifted = data["temperature"].shift(1)

    for window in rolling_windows:

        data[
            f"temperature_rollmean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        data[
            f"temperature_rollstd_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )


    return data


# ============================================================
# 7. CREATE FEATURES
# ============================================================

features_df = create_features(df)

print(
    f"Feature table generated: "
    f"{features_df.shape[1]} columns"
)


# ============================================================
# 8. SIMULATE REAL-TIME FORECASTING
# ============================================================

print("\n[4/6] Starting simulated real-time forecasting...")

print(
    "\nThe system will process the historical observations "
    "in chronological order."
)

print(
    "For demonstration, only the first 500 valid forecasting "
    "points will be generated."
)

print()


MAX_FORECASTS = 500

results = []

forecast_count = 0


# Need enough history for lag 60.

START_INDEX = 60


for index in range(
    START_INDEX,
    len(features_df)
):

    timestamp = features_df.loc[
        index,
        "ts"
    ]


    # --------------------------------------------------------
    # Stop after requested number of forecasts
    # --------------------------------------------------------

    if forecast_count >= MAX_FORECASTS:
        break


    row_results = {
        "feature_timestamp":
            timestamp,

        "forecast_timestamp":
            timestamp
            + pd.Timedelta(
                seconds=60
            )
    }


    valid_row = True


    # --------------------------------------------------------
    # Generate each sensor forecast
    # --------------------------------------------------------

    for sensor in SENSORS:

        artifact = artifacts[sensor]

        model_name = artifact[
            "model_name"
        ]


        current_value = float(
            df.loc[
                index,
                sensor
            ]
        )


        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        if model_name == "Persistence":

            prediction = current_value


        # ----------------------------------------------------
        # ML models
        # ----------------------------------------------------

        else:

            required_features = artifact[
                "features"
            ]


            X = features_df.loc[
                [index],
                required_features
            ]


            # ------------------------------------------------
            # Validate input
            # ------------------------------------------------

            if X.isnull().any().any():

                valid_row = False
                break


            prediction_array = (
                artifact["model"]
                .predict(X)
            )


            prediction = float(
                np.asarray(
                    prediction_array
                ).ravel()[0]
            )


        # ----------------------------------------------------
        # Store sensor prediction
        # ----------------------------------------------------

        row_results[
            f"{sensor}_current"
        ] = current_value


        row_results[
            f"{sensor}_prediction"
        ] = prediction


        row_results[
            f"{sensor}_change"
        ] = (
            prediction
            -
            current_value
        )


    # --------------------------------------------------------
    # Store only complete rows
    # --------------------------------------------------------

    if valid_row:

        results.append(
            row_results
        )

        forecast_count += 1


    # --------------------------------------------------------
    # Progress display
    # --------------------------------------------------------

    if forecast_count % 100 == 0:

        print(
            f"Forecasts generated: "
            f"{forecast_count}"
        )


# ============================================================
# 9. RESULTS DATAFRAME
# ============================================================

print("\n[5/6] Creating prediction table...")

predictions = pd.DataFrame(
    results
)


if predictions.empty:

    raise ValueError(
        "No valid forecasts were generated."
    )


print(
    f"Forecasts generated: "
    f"{len(predictions):,}"
)


# ============================================================
# 10. SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. DISPLAY EXAMPLE
# ============================================================

print("\n[6/6] Simulation complete.")

print("\n" + "=" * 75)
print("FIRST FORECAST")
print("=" * 75)

first = predictions.iloc[0]

print(
    f"\nFeature time : "
    f"{first['feature_timestamp']}"
)

print(
    f"Forecast time: "
    f"{first['forecast_timestamp']}"
)


for sensor in SENSORS:

    print(
        f"\n{sensor}"
    )

    print(
        f"  Current    : "
        f"{first[f'{sensor}_current']:.6f}"
    )

    print(
        f"  Prediction : "
        f"{first[f'{sensor}_prediction']:.6f}"
    )

    print(
        f"  Change     : "
        f"{first[f'{sensor}_change']:.6f}"
    )


print("\n" + "=" * 75)
print("LAST FORECAST")
print("=" * 75)

last = predictions.iloc[-1]

print(
    f"\nFeature time : "
    f"{last['feature_timestamp']}"
)

print(
    f"Forecast time: "
    f"{last['forecast_timestamp']}"
)


for sensor in SENSORS:

    print(
        f"\n{sensor}"
    )

    print(
        f"  Current    : "
        f"{last[f'{sensor}_current']:.6f}"
    )

    print(
        f"  Prediction : "
        f"{last[f'{sensor}_prediction']:.6f}"
    )

    print(
        f"  Change     : "
        f"{last[f'{sensor}_change']:.6f}"
    )


print("\n" + "=" * 75)
print("STEP 24 COMPLETE")
print("=" * 75)

print(
    f"\nPrediction file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nThe trained models were not modified."
)

print(
    "The CSV was replayed chronologically as a "
    "simulated sensor stream."
)