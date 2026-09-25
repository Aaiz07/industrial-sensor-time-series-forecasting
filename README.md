from pathlib import Path


# ============================================================
# STEP 35 - CREATE FINAL README
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
README_PATH = PROJECT_DIR / "README.md"


README = r"""# Industrial Sensor Time-Series Forecasting System

## 1. Project Overview

This project is an end-to-end industrial sensor time-series forecasting system designed to predict future sensor values from historical machine data.

The system works with:

- Vibration sensors:
  - `vib_x_rms`
  - `vib_y_rms`
  - `vib_z_rms`
- Magnetic field sensors:
  - `mag_x`
  - `mag_y`
  - `mag_z`
- Temperature:
  - `temperature`

The complete workflow is:

```text
Raw Sensor Data
      ↓
Data Validation & Cleaning
      ↓
Exploratory Data Analysis
      ↓
Time-Series Preprocessing
      ↓
Feature Engineering
      ↓
Future Target Creation
      ↓
Chronological Validation
      ↓
Walk-Forward Validation
      ↓
Final Model Selection
      ↓
60-Second Forecasting
      ↓
Real-Time Forecasting
      ↓
SQLite Database
      ↓
Monitoring & Alerts
      ↓
Streamlit Dashboard