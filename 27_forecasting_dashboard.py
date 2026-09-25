# ============================================================
# STEP 27 — FORECASTING DASHBOARD
# ============================================================
#
# Purpose:
#   Interactive Streamlit dashboard for the industrial
#   sensor forecasting and monitoring system.
#
# Data source:
#   outputs/database/forecasting.db
#
# Run:
#   streamlit run 27_forecasting_dashboard.py
#
# ============================================================

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "outputs" / "database" / "forecasting.db"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Industrial Sensor Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        margin-bottom: 20px;
    }

    .status-normal {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
    }

    .status-warning {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
    }

    .status-critical {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. DATABASE FUNCTIONS
# ============================================================

def get_connection():
    """Create SQLite database connection."""
    return sqlite3.connect(DB_PATH)


@st.cache_data(ttl=5)
def load_predictions():
    """Load forecast predictions from database."""

    conn = get_connection()

    query = """
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
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    if not df.empty:
        df["feature_timestamp"] = pd.to_datetime(
            df["feature_timestamp"]
        )

        df["forecast_timestamp"] = pd.to_datetime(
            df["forecast_timestamp"]
        )

        df["created_at"] = pd.to_datetime(
            df["created_at"]
        )

    return df


@st.cache_data(ttl=5)
def load_monitoring():
    """Load monitoring results from database."""

    conn = get_connection()

    query = """
        SELECT *
        FROM prediction_monitoring
        ORDER BY forecast_timestamp ASC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    if not df.empty:

        if "feature_timestamp" in df.columns:
            df["feature_timestamp"] = pd.to_datetime(
                df["feature_timestamp"]
            )

        if "forecast_timestamp" in df.columns:
            df["forecast_timestamp"] = pd.to_datetime(
                df["forecast_timestamp"]
            )

        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(
                df["created_at"]
            )

    return df


# ============================================================
# 5. SENSOR DISPLAY INFORMATION
# ============================================================

SENSOR_NAMES = {
    "vib_x_rms": "Vibration X",
    "vib_y_rms": "Vibration Y",
    "vib_z_rms": "Vibration Z",
    "mag_x": "Magnetic X",
    "mag_y": "Magnetic Y",
    "mag_z": "Magnetic Z",
    "temperature": "Temperature"
}


def display_sensor_name(sensor):
    return SENSOR_NAMES.get(sensor, sensor)


def format_value(sensor, value):

    if pd.isna(value):
        return "N/A"

    if sensor == "temperature":
        return f"{value:.2f} °C"

    if sensor.startswith("vib"):
        return f"{value:.4f}"

    if sensor.startswith("mag"):
        return f"{value:.2f}"

    return f"{value:.4f}"


# ============================================================
# 6. LOAD DATA
# ============================================================

if not DB_PATH.exists():

    st.error(
        f"Database not found:\n\n{DB_PATH}"
    )

    st.stop()


predictions = load_predictions()
monitoring = load_monitoring()


if predictions.empty:

    st.warning(
        "No forecast predictions found in the database."
    )

    st.stop()


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Industrial Sensor Forecasting Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Real-time-style forecasting, monitoring and anomaly visualization",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

# Sensor filter
available_sensors = sorted(
    predictions["sensor"].unique()
)

selected_sensor = st.sidebar.selectbox(
    "Select Sensor",
    ["All Sensors"] + available_sensors
)


# Status filter
if not monitoring.empty and "status" in monitoring.columns:

    available_statuses = sorted(
        monitoring["status"].dropna().unique()
    )

else:

    available_statuses = []


selected_status = st.sidebar.selectbox(
    "Status Filter",
    ["All Statuses"] + available_statuses
)


# Number of recent records
recent_count = st.sidebar.slider(
    "Recent Records",
    min_value=20,
    max_value=500,
    value=100,
    step=20
)


# ============================================================
# 9. FILTER DATA
# ============================================================

filtered_predictions = predictions.copy()

if selected_sensor != "All Sensors":

    filtered_predictions = filtered_predictions[
        filtered_predictions["sensor"] == selected_sensor
    ]


if (
    selected_status != "All Statuses"
    and not monitoring.empty
    and "status" in monitoring.columns
):

    status_ids = monitoring[
        monitoring["status"] == selected_status
    ]["id"]

    filtered_predictions = filtered_predictions[
        filtered_predictions["id"].isin(status_ids)
    ]


# ============================================================
# 10. LATEST FORECASTS
# ============================================================

st.header("🔮 Latest Forecasts")

latest_rows = []

for sensor in available_sensors:

    sensor_data = predictions[
        predictions["sensor"] == sensor
    ].sort_values("forecast_timestamp")

    if sensor_data.empty:
        continue

    latest = sensor_data.iloc[-1]

    latest_rows.append(
        {
            "sensor": sensor,
            "feature_timestamp": latest["feature_timestamp"],
            "forecast_timestamp": latest["forecast_timestamp"],
            "current_value": latest["current_value"],
            "predicted_value": latest["predicted_value"],
            "model": latest["model"]
        }
    )


latest_df = pd.DataFrame(latest_rows)


# ============================================================
# 11. DISPLAY SENSOR CARDS
# ============================================================

if not latest_df.empty:

    cols = st.columns(4)

    for i, row in latest_df.iterrows():

        sensor = row["sensor"]

        with cols[i % 4]:

            st.metric(
                label=display_sensor_name(sensor),
                value=format_value(
                    sensor,
                    row["predicted_value"]
                ),
                delta=(
                    row["predicted_value"]
                    - row["current_value"]
                )
            )

            st.caption(
                f"Model: {row['model']}"
            )


st.divider()


# ============================================================
# 12. SYSTEM SUMMARY
# ============================================================

st.header("📌 System Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Forecast Records",
        f"{len(predictions):,}"
    )

with col2:

    st.metric(
        "Sensors",
        predictions["sensor"].nunique()
    )

with col3:

    st.metric(
        "Forecast Horizon",
        f"{predictions['horizon_seconds'].iloc[0]} sec"
    )

with col4:

    latest_time = predictions[
        "forecast_timestamp"
    ].max()

    st.metric(
        "Latest Forecast",
        latest_time.strftime("%H:%M:%S")
    )


# ============================================================
# 13. MONITORING SUMMARY
# ============================================================

st.header("🚦 Monitoring Status")

if monitoring.empty:

    st.info("No monitoring records available.")

else:

    status_counts = (
        monitoring["status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "status",
        "count"
    ]

    c1, c2, c3, c4 = st.columns(4)

    normal_count = int(
        status_counts.loc[
            status_counts["status"] == "NORMAL",
            "count"
        ].sum()
    )

    warning_count = int(
        status_counts.loc[
            status_counts["status"] == "WARNING",
            "count"
        ].sum()
    )

    anomaly_count = int(
        status_counts.loc[
            status_counts["status"] == "ANOMALY",
            "count"
        ].sum()
    )

    low_conf_count = int(
        status_counts.loc[
            status_counts["status"] == "LOW_CONFIDENCE",
            "count"
        ].sum()
    )

    with c1:
        st.metric(
            "NORMAL",
            f"{normal_count:,}"
        )

    with c2:
        st.metric(
            "WARNING",
            f"{warning_count:,}"
        )

    with c3:
        st.metric(
            "ANOMALY",
            f"{anomaly_count:,}"
        )

    with c4:
        st.metric(
            "LOW CONFIDENCE",
            f"{low_conf_count:,}"
        )


# ============================================================
# 14. STATUS DISTRIBUTION
# ============================================================

if not monitoring.empty:

    st.subheader("Status Distribution")

    status_chart = (
        monitoring["status"]
        .value_counts()
    )

    st.bar_chart(status_chart)


# ============================================================
# 15. SENSOR FORECAST TREND
# ============================================================

st.header("📈 Forecast Trend")

if selected_sensor == "All Sensors":

    chart_sensor = st.selectbox(
        "Choose sensor for trend",
        available_sensors,
        key="trend_sensor"
    )

else:

    chart_sensor = selected_sensor


trend_data = predictions[
    predictions["sensor"] == chart_sensor
].copy()

trend_data = trend_data.sort_values(
    "forecast_timestamp"
).tail(recent_count)


if not trend_data.empty:

    chart_data = trend_data[
        [
            "forecast_timestamp",
            "current_value",
            "predicted_value"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "forecast_timestamp"
    )

    st.line_chart(
        chart_data
    )


# ============================================================
# 16. CURRENT VS PREDICTED
# ============================================================

st.header("🔄 Current vs Predicted")

comparison_data = latest_df.copy()

if not comparison_data.empty:

    comparison_display = comparison_data[
        [
            "sensor",
            "current_value",
            "predicted_value",
            "model"
        ]
    ].copy()

    comparison_display["sensor"] = (
        comparison_display["sensor"]
        .map(display_sensor_name)
    )

    comparison_display.columns = [
        "Sensor",
        "Current",
        "Predicted",
        "Model"
    ]

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 17. RECENT MONITORING EVENTS
# ============================================================

st.header("⚠️ Recent Monitoring Events")

if monitoring.empty:

    st.info("No monitoring events available.")

else:

    events = monitoring.copy()

    if selected_sensor != "All Sensors":

        events = events[
            events["sensor"] == selected_sensor
        ]

    if selected_status != "All Statuses":

        events = events[
            events["status"] == selected_status
        ]

    events = events.sort_values(
        "forecast_timestamp",
        ascending=False
    ).head(50)

    display_columns = [
        "forecast_timestamp",
        "sensor",
        "predicted_value",
        "status",
        "severity",
        "alert_type",
        "reason"
    ]

    available_columns = [
        col for col in display_columns
        if col in events.columns
    ]

    event_display = events[
        available_columns
    ].copy()

    if "sensor" in event_display.columns:

        event_display["sensor"] = (
            event_display["sensor"]
            .map(display_sensor_name)
        )

    st.dataframe(
        event_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 18. SENSOR-WISE MONITORING
# ============================================================

st.header("📊 Sensor-wise Monitoring")

if not monitoring.empty:

    sensor_status = pd.crosstab(
        monitoring["sensor"],
        monitoring["status"]
    )

    sensor_status.index = [
        display_sensor_name(sensor)
        for sensor in sensor_status.index
    ]

    st.dataframe(
        sensor_status,
        use_container_width=True
    )


# ============================================================
# 19. MODEL INFORMATION
# ============================================================

st.header("🤖 Model Information")

model_info = (
    predictions[
        ["sensor", "model", "horizon_seconds"]
    ]
    .drop_duplicates()
    .sort_values("sensor")
)

model_info["sensor"] = (
    model_info["sensor"]
    .map(display_sensor_name)
)

model_info.columns = [
    "Sensor",
    "Model",
    "Horizon (seconds)"
]

st.dataframe(
    model_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 20. DATASET INFORMATION
# ============================================================

st.header("🗄️ Database Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write(
        f"**Database:** `{DB_PATH}`"
    )

    st.write(
        f"**Prediction records:** {len(predictions):,}"
    )

with info_col2:

    st.write(
        f"**First forecast:** "
        f"{predictions['forecast_timestamp'].min()}"
    )

    st.write(
        f"**Last forecast:** "
        f"{predictions['forecast_timestamp'].max()}"
    )


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.caption(
    "Industrial Sensor Forecasting System | "
    "Step 27 Dashboard | "
    "Forecast horizon: 60 seconds | "
    "Current demonstration uses simulated historical sensor replay"
)