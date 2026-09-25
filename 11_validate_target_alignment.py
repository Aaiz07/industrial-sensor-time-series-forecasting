"""
11_validate_target_alignment.py

STEP 11 - TARGET ALIGNMENT + LEAKAGE VALIDATION
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "final_features.csv"
)

CLEANED_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "final_test_predictions.csv"
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
# COLUMNS
# ============================================================

TIMESTAMP_COL = "ts"

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]

TARGET_PREFIX = "target_"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 11 - TARGET ALIGNMENT + LEAKAGE VALIDATION")
print("=" * 75)


for file_path in [
    FEATURE_FILE,
    CLEANED_FILE,
    PREDICTION_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Missing file:\n{file_path}"
        )


features = pd.read_csv(
    FEATURE_FILE
)

features[TIMESTAMP_COL] = pd.to_datetime(
    features[TIMESTAMP_COL],
    errors="coerce"
)

features = (
    features
    .dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)


cleaned = pd.read_csv(
    CLEANED_FILE
)

cleaned[TIMESTAMP_COL] = pd.to_datetime(
    cleaned[TIMESTAMP_COL],
    errors="coerce"
)

cleaned = (
    cleaned
    .dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)


pred = pd.read_csv(
    PREDICTION_FILE
)

pred[TIMESTAMP_COL] = pd.to_datetime(
    pred[TIMESTAMP_COL],
    errors="coerce"
)

pred = (
    pred
    .dropna(subset=[TIMESTAMP_COL])
    .sort_values(TIMESTAMP_COL)
    .reset_index(drop=True)
)


print(
    f"\nFeature rows:  {len(features):,}"
)

print(
    f"Cleaned rows:  {len(cleaned):,}"
)

print(
    f"Prediction rows: {len(pred):,}"
)


# ============================================================
# 1. TARGET CHECK
# ============================================================

print("\n" + "=" * 75)
print("1. TARGET COLUMN CHECK")
print("=" * 75)


target_cols = [
    f"{TARGET_PREFIX}{sensor}"
    for sensor in SENSOR_COLS
]

missing_targets = [
    col
    for col in target_cols
    if col not in features.columns
]


print(
    f"Expected target columns: {len(target_cols)}"
)

print(
    f"Found target columns:    "
    f"{len(target_cols) - len(missing_targets)}"
)


if missing_targets:

    print("\nMissing targets:")

    for col in missing_targets:
        print(f"  {col}")

else:

    print(
        "All target columns found."
    )


# ============================================================
# 2. DIRECT FEATURE LEAKAGE
# ============================================================

print("\n" + "=" * 75)
print("2. FEATURE / TARGET NAME LEAKAGE CHECK")
print("=" * 75)


feature_cols = [
    col
    for col in features.columns
    if col != TIMESTAMP_COL
    and not col.startswith(TARGET_PREFIX)
]


direct_target_leaks = [
    sensor
    for sensor in SENSOR_COLS
    if sensor in feature_cols
]


print(
    f"Feature columns: {len(feature_cols):,}"
)


if direct_target_leaks:

    print(
        "\nFAIL - current sensor columns found "
        "as features:"
    )

    for col in direct_target_leaks:
        print(f"  {col}")

else:

    print(
        "PASS - raw current target columns are "
        "not present as features."
    )


# ============================================================
# 3. LAG VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("3. LAG FEATURE CHECK")
print("=" * 75)


# We use the original cleaned data here because
# final_features.csv contains the engineered features,
# not the raw sensor columns.

# Align original sensor values to the feature timestamps.

original_lookup = (
    cleaned
    .set_index(TIMESTAMP_COL)
)


lag_results = []


for sensor in SENSOR_COLS:

    lag1_name = f"{sensor}_lag_1"


    if lag1_name not in features.columns:

        print(
            f"{sensor:15s} -> MISSING"
        )

        lag_results.append({
            "sensor": sensor,
            "feature": lag1_name,
            "status": "MISSING",
            "max_absolute_error": np.nan
        })

        continue


    # --------------------------------------------------------
    # Reconstruct expected lag_1
    # --------------------------------------------------------

    # For each feature timestamp t, find the previous
    # observation in the original cleaned dataset.

    sensor_series = (
        cleaned[
            [
                TIMESTAMP_COL,
                sensor
            ]
        ]
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )


    expected_lag = (
        sensor_series[sensor]
        .shift(1)
    )


    # Feature rows correspond to cleaned rows after
    # dropping the first 60 rows.
    #
    # Instead of assuming exact row positions, merge by
    # timestamp and reconstruct previous timestamp.

    temp = features[
        [
            TIMESTAMP_COL,
            lag1_name
        ]
    ].copy()


    previous_values = sensor_series[
        [
            TIMESTAMP_COL,
            sensor
        ]
    ].copy()


    previous_values[
        "previous_timestamp"
    ] = previous_values[
        TIMESTAMP_COL
    ].shift(1)


    previous_values[
        "expected_lag"
    ] = previous_values[
        sensor
    ].shift(1)


    temp = temp.merge(
        previous_values[
            [
                TIMESTAMP_COL,
                "expected_lag"
            ]
        ],
        on=TIMESTAMP_COL,
        how="left"
    )


    valid = (
        temp[lag1_name].notna()
        &
        temp["expected_lag"].notna()
    )


    if valid.sum() == 0:

        status = "NO_VALID_ROWS"
        max_error = np.nan

    else:

        error = (
            temp.loc[
                valid,
                lag1_name
            ].to_numpy(
                dtype=float
            )
            -
            temp.loc[
                valid,
                "expected_lag"
            ].to_numpy(
                dtype=float
            )
        )

        max_error = np.max(
            np.abs(error)
        )

        status = (
            "PASS"
            if max_error < 1e-10
            else "CHECK"
        )


    lag_results.append({
        "sensor": sensor,
        "feature": lag1_name,
        "status": status,
        "max_absolute_error": max_error
    })


    if np.isfinite(max_error):

        print(
            f"{sensor:15s} -> "
            f"{status:5s} | "
            f"max error = "
            f"{max_error:.12g}"
        )

    else:

        print(
            f"{sensor:15s} -> "
            f"{status}"
        )


lag_report = pd.DataFrame(
    lag_results
)


# ============================================================
# 4. ROLLING FEATURE CHECK
# ============================================================

print("\n" + "=" * 75)
print("4. ROLLING FEATURE CHECK")
print("=" * 75)


rolling_results = []


for sensor in SENSOR_COLS:

    candidates = [
        col
        for col in features.columns
        if col.startswith(
            sensor + "_rollmean_"
        )
    ]


    for col in candidates:

        try:

            window = int(
                col.split("_")[-1]
            )

        except Exception:

            continue


        # Build expected rolling feature from
        # original cleaned sensor data.

        expected_series = (
            cleaned[sensor]
            .shift(1)
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
        )


        expected_df = pd.DataFrame({
            TIMESTAMP_COL:
                cleaned[TIMESTAMP_COL],
            "expected":
                expected_series
        })


        actual_df = features[
            [
                TIMESTAMP_COL,
                col
            ]
        ].copy()


        merged = actual_df.merge(
            expected_df,
            on=TIMESTAMP_COL,
            how="left"
        )


        valid = (
            merged[col].notna()
            &
            merged["expected"].notna()
        )


        if valid.sum() == 0:

            max_error = np.nan
            status = "NO_VALID_ROWS"

        else:

            error = (
                merged.loc[
                    valid,
                    col
                ].to_numpy(
                    dtype=float
                )
                -
                merged.loc[
                    valid,
                    "expected"
                ].to_numpy(
                    dtype=float
                )
            )

            max_error = np.max(
                np.abs(error)
            )

            status = (
                "PASS"
                if max_error < 1e-8
                else "CHECK"
            )


        rolling_results.append({
            "sensor": sensor,
            "feature": col,
            "window": window,
            "status": status,
            "max_absolute_error": max_error
        })


        if np.isfinite(max_error):

            print(
                f"{col:30s} -> "
                f"{status:5s} | "
                f"max error = "
                f"{max_error:.12g}"
            )

        else:

            print(
                f"{col:30s} -> "
                f"{status}"
            )


rolling_report = pd.DataFrame(
    rolling_results
)


# ============================================================
# 5. FUTURE-CORRELATION DIAGNOSTIC
# ============================================================

print("\n" + "=" * 75)
print("5. CURRENT VS FUTURE ALIGNMENT DIAGNOSTIC")
print("=" * 75)


alignment_records = []


HORIZONS = [
    1,
    5,
    10,
    30,
    60
]


for sensor in SENSOR_COLS:

    current = cleaned[sensor]


    for horizon in HORIZONS:

        future = current.shift(
            -horizon
        )


        valid = (
            current.notna()
            &
            future.notna()
        )


        if valid.sum() > 2:

            corr = np.corrcoef(
                current[valid].to_numpy(),
                future[valid].to_numpy()
            )[0, 1]

        else:

            corr = np.nan


        alignment_records.append({
            "sensor": sensor,
            "horizon_samples": horizon,
            "current_to_future_correlation": corr
        })


        print(
            f"{sensor:15s} "
            f"h={horizon:3d} -> "
            f"corr={corr:.6f}"
        )


alignment_report = pd.DataFrame(
    alignment_records
)


alignment_file = os.path.join(
    REPORT_DIR,
    "target_alignment_report.csv"
)


alignment_report.to_csv(
    alignment_file,
    index=False
)


# ============================================================
# 6. PREDICTION SANITY CHECK
# ============================================================

print("\n" + "=" * 75)
print("6. TEST PREDICTION SANITY CHECK")
print("=" * 75)


prediction_records = []


for sensor in SENSOR_COLS:

    actual_col = (
        f"{sensor}_actual"
    )

    if actual_col not in pred.columns:
        continue


    actual = pred[
        actual_col
    ].to_numpy(
        dtype=float
    )


    model_columns = [
        f"{sensor}_ridge",
        f"{sensor}_xgb"
    ]


    for model_col in model_columns:

        if model_col not in pred.columns:
            continue


        prediction = pred[
            model_col
        ].to_numpy(
            dtype=float
        )


        valid = (
            np.isfinite(actual)
            &
            np.isfinite(prediction)
        )


        if valid.sum() == 0:
            continue


        error = (
            prediction[valid]
            -
            actual[valid]
        )


        mae = np.mean(
            np.abs(error)
        )


        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )


        prediction_records.append({
            "sensor": sensor,
            "model": model_col.replace(
                f"{sensor}_",
                ""
            ),
            "MAE": mae,
            "RMSE": rmse,
            "prediction_min":
                np.min(
                    prediction[valid]
                ),
            "prediction_max":
                np.max(
                    prediction[valid]
                ),
            "actual_min":
                np.min(
                    actual[valid]
                ),
            "actual_max":
                np.max(
                    actual[valid]
                )
        })


        print(
            f"{sensor:15s} "
            f"{model_col.replace(f'{sensor}_', ''):12s} "
            f"MAE={mae:.6f} "
            f"RMSE={rmse:.6f}"
        )


prediction_report = pd.DataFrame(
    prediction_records
)


# ============================================================
# 7. ACTUAL VS PREDICTED PLOTS
# ============================================================

print("\n" + "=" * 75)
print("7. CREATING ACTUAL VS PREDICTED PLOTS")
print("=" * 75)


for sensor in SENSOR_COLS:

    actual_col = (
        f"{sensor}_actual"
    )

    ridge_col = (
        f"{sensor}_ridge"
    )


    if (
        actual_col not in pred.columns
        or ridge_col not in pred.columns
    ):
        continue


    actual = pred[
        actual_col
    ].to_numpy(
        dtype=float
    )


    prediction = pred[
        ridge_col
    ].to_numpy(
        dtype=float
    )


    timestamps = pred[
        TIMESTAMP_COL
    ]


    n_plot = min(
        len(pred),
        3000
    )


    plt.figure(
        figsize=(14, 5)
    )


    plt.plot(
        timestamps.iloc[:n_plot],
        actual[:n_plot],
        label="Actual"
    )


    plt.plot(
        timestamps.iloc[:n_plot],
        prediction[:n_plot],
        label="Ridge Prediction"
    )


    plt.title(
        f"{sensor} - Actual vs Predicted"
    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        sensor
    )


    plt.legend()


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    output_file = os.path.join(
        PLOT_DIR,
        f"actual_vs_predicted_{sensor}.png"
    )


    plt.savefig(
        output_file,
        dpi=150
    )


    plt.close()


    print(
        f"Saved: {output_file}"
    )


# ============================================================
# 8. FINAL LEAKAGE REPORT
# ============================================================

leakage_records = []


leakage_records.append({
    "check":
        "Raw current target in feature list",
    "status":
        "FAIL"
        if direct_target_leaks
        else "PASS",
    "details":
        ", ".join(
            direct_target_leaks
        )
        if direct_target_leaks
        else
        "No raw current sensor columns found."
})


if not lag_report.empty:

    lag_failures = lag_report[
        lag_report["status"] != "PASS"
    ]


    leakage_records.append({
        "check":
            "Lag-1 features",
        "status":
            "CHECK"
            if not lag_failures.empty
            else "PASS",
        "details":
            f"{len(lag_failures)} problematic lag features"
    })


if not rolling_report.empty:

    rolling_failures = rolling_report[
        rolling_report["status"] != "PASS"
    ]


    leakage_records.append({
        "check":
            "Rolling features",
        "status":
            "CHECK"
            if not rolling_failures.empty
            else "PASS",
        "details":
            f"{len(rolling_failures)} problematic rolling features"
    })


leakage_file = os.path.join(
    REPORT_DIR,
    "leakage_check_report.csv"
)


pd.DataFrame(
    leakage_records
).to_csv(
    leakage_file,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("STEP 11 COMPLETED")
print("=" * 75)

print("\nReports:")
print(
    f"  {alignment_file}"
)

print(
    f"  {leakage_file}"
)

print(
    "\nDo NOT proceed to one-hour forecasting yet."
)

print(
    "Send the complete terminal output."
)

print("\n" + "=" * 75)