# ============================================================
# STEP 23: REAL-TIME 60-SECOND FORECASTING
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
    "step23_realtime_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SENSOR LIST
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
# 3. START
# ============================================================

print("=" * 75)
print("STEP 23: REAL-TIME 60-SECOND FORECASTING")
print("=" * 75)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n[1/7] Loading sensor data...")

df = pd.read_csv(DATA_FILE)

df["ts"] = pd.to_datetime(df["ts"])

df = (
    df
    .sort_values("ts")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df):,}")
print(f"Latest timestamp: {df['ts'].max()}")


# ============================================================
# 5. LOAD MODEL ARTIFACTS
# ============================================================

print("\n[2/7] Loading final models...")

artifacts = {}
models = {}

for sensor in SENSORS:

    path = os.path.join(
        MODEL_DIR,
        MODEL_FILES[sensor]
    )

    artifact = joblib.load(path)

    artifacts[sensor] = artifact

    # Temperature is Persistence and has no estimator.
    # We handle it separately below.
    if artifact["model"] is not None:
        models[sensor] = artifact["model"]

    print(
        f"{sensor:15s} -> "
        f"{artifact['model_name']:12s} | "
        f"features = {len(artifact['features'])}"
    )


# ============================================================
# 6. CREATE EXACT TRAINING FEATURES
# ============================================================

print("\n[3/7] Creating exact 146-feature schema...")


def create_training_features(data):

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
    # Standard sensor lags + difference
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
    # Cross-axis vibration features
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The trained models expect:
    #
    # vib_y -> vib_x
    # vib_z -> vib_x
    # vib_x -> vib_y
    # vib_z -> vib_y
    # vib_x -> vib_z
    # vib_y -> vib_z
    #
    # at lags 1,2,5,10.
    #
    # --------------------------------------------------------

    cross_lags = [
        1,
        2,
        5,
        10
    ]

    vibration_sensors = [
        "vib_x_rms",
        "vib_y_rms",
        "vib_z_rms"
    ]

    for target in vibration_sensors:

        for source in vibration_sensors:

            if source == target:
                continue

            for lag in cross_lags:

                feature_name = (
                    f"{source}_to_{target}_lag_{lag}"
                )

                data[feature_name] = (
                    data[source]
                    .shift(lag)
                )


    # --------------------------------------------------------
    # Magnetic-specific features
    # --------------------------------------------------------
    #
    # Exact names expected by Step 21:
    #
    # mag_x_mag_lag_*
    # mag_x_rollmean_*
    # mag_x_rollstd_*
    #
    # Same for mag_y and mag_z.
    #
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

        # Magnetic lags
        for lag in magnetic_lags:

            data[
                f"{sensor}_mag_lag_{lag}"
            ] = data[sensor].shift(lag)

        # Rolling features
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


features_df = create_training_features(df)


print(
    f"Total columns generated: "
    f"{features_df.shape[1]}"
)


# ============================================================
# 7. VERIFY FEATURE SCHEMAS
# ============================================================

print("\n[4/7] Verifying model feature schemas...")


for sensor in SENSORS:

    required = artifacts[
        sensor
    ]["features"]

    if len(required) == 0:
        continue

    missing = [
        feature
        for feature in required
        if feature not in features_df.columns
    ]

    if missing:

        print(
            f"\nERROR: {sensor}"
        )

        print(
            f"Missing {len(missing)} features:"
        )

        for feature in missing:
            print(
                f"  {feature}"
            )

        raise ValueError(
            f"{sensor}: feature schema mismatch."
        )

    print(
        f"{sensor:15s}: "
        f"{len(required)} features -> OK"
    )


# ============================================================
# 8. GET LATEST ROW
# ============================================================

print("\n[5/7] Preparing latest observation...")


latest_index = features_df.index[-1]

latest_timestamp = features_df.loc[
    latest_index,
    "ts"
]

print(
    f"Feature timestamp: "
    f"{latest_timestamp}"
)


# ============================================================
# 9. GENERATE FORECASTS
# ============================================================

print("\n[6/7] Generating 60-second forecasts...")


results = []


for sensor in SENSORS:

    artifact = artifacts[sensor]

    model_name = artifact[
        "model_name"
    ]


    current_value = float(
        df.loc[
            latest_index,
            sensor
        ]
    )


    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    if model_name == "Persistence":

        predicted_value = current_value


    # --------------------------------------------------------
    # Learned model
    # --------------------------------------------------------

    else:

        required_features = artifact[
            "features"
        ]


        X = features_df.loc[
            [latest_index],
            required_features
        ].copy()


        # Check shape
        if X.shape[1] != len(
            required_features
        ):

            raise ValueError(
                f"{sensor}: wrong feature count."
            )


        # Check NaNs
        if X.isnull().any().any():

            bad_features = (
                X.columns[
                    X.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                f"{sensor}: NaN in features: "
                f"{bad_features}"
            )


        prediction = model = artifact[
            "model"
        ].predict(X)


        predicted_value = float(
            np.asarray(
                prediction
            ).ravel()[0]
        )


    results.append(
        {
            "feature_timestamp":
                latest_timestamp,

            "forecast_timestamp":
                latest_timestamp
                + pd.Timedelta(
                    seconds=60
                ),

            "horizon_seconds":
                artifact[
                    "horizon_seconds"
                ],

            "sensor":
                sensor,

            "model":
                model_name,

            "current_value":
                current_value,

            "predicted_value":
                predicted_value,

            "change_from_current":
                predicted_value
                -
                current_value
        }
    )


predictions = pd.DataFrame(
    results
)


# ============================================================
# 10. SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. DISPLAY
# ============================================================

print("\n" + "=" * 75)
print("60-SECOND FORECAST")
print("=" * 75)

print(
    f"\nCurrent timestamp : "
    f"{latest_timestamp}"
)

print(
    f"Forecast timestamp: "
    f"{latest_timestamp + pd.Timedelta(seconds=60)}"
)

print()

print(
    predictions[
        [
            "sensor",
            "model",
            "current_value",
            "predicted_value",
            "change_from_current"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 75)
print("STEP 23 COMPLETE")
print("=" * 75)

print(
    f"\nPredictions saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The forecasts are 60-second-ahead model outputs."
)

print(
    "Temperature has the strongest demonstrated "
    "forecast reliability."
)

print(
    "Magnetic channels have approximately zero "
    "walk-forward R2."
)

print(
    "Vibration models were selected by walk-forward "
    "RMSE but showed negative walk-forward R2, so "
    "their forecasts should be treated cautiously."
)