"""Project configuration.

Edit this file to customize:
- feature sets per target
- CatBoost hyperparameters per target & quantile
- precipitation thresholds for 'Rain Classes'
- AQI thresholds (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

# -----------------------------
# Feature sets (as in notebook)
# -----------------------------

FEATURES_PM25: List[str] = [
    'PREV_pm2.5',
    'WIND direction gust',
    'AIR TEMP T min',
    'Rain Classes',
    'PREV_RMEAN_pm10',
    'PREV_WIND speed gust',
    'PREV_so2',
    'AIR TEMP time max',
    'AIR TEMP time min',
    'DAYLIGHT HOURS',
    'HUMIDITY avg',
    'PREV_pm10',
    'RMEAN_HUMIDITY avg',
    'AIR TEMP T avg',
    'HUMIDITY max',
]

FEATURES_O3: List[str] = [
    'PREV_o3',
    'HUMIDITY avg',
    'AIR TEMP T max',
    'PREV_AIR TEMP T avg',
    'AIR TEMP time min',
    'RMEAN_AIR TEMP T avg',
    'DAYLIGHT HOURS',
    'PREV_AIR TEMP T max',
    'PREV_pm10',
    'AIR TEMP time max',
]

# Categorical columns (will be cast to pandas 'category' when present)
CATEGORICAL_COLS: List[str] = [
    'WIND direction gust',
    'Rain Classes',
    'AIR TEMP time min',
    'AIR TEMP time max',
    'HUMIDITY time min',
    'HUMIDITY time max',
    'WIND time gust',
    # In the original notebook PREV_so2 is treated as string -> categorical
    'PREV_so2',
    'month of year',
    'year',
]

# Required lag columns for prediction validity (length-preserving predict)
REQUIRED_COLS_BY_TARGET = {
    "pm25": ["PREV_pm2.5"],
    "o3": ["PREV_o3"],
}

# -----------------------------
# Precipitation binning
# -----------------------------
# Use your fixed thresholds here (domain-based).
# The bins are interpreted as:
#   0 -> 'no rain'
#   (0, b1] -> 'very light rain'
#   (b1, b2] -> 'light rain'
#   (b2, b3] -> 'normal rain'
#   (b3, b4] -> 'heavy rain'
#   (b4, b5] -> 'very heavy rain'
#   > b5 -> 'exceptional rain'
#
# IMPORTANT: Replace the placeholder values below with the thresholds you identified.
RAIN_BINS: Tuple[float, float, float, float, float] = (
    0.5,   # b1
    2.0,   # b2
    5.0,   # b3
    10.0,  # b4
    20.0,  # b5
)

RAIN_LABELS: Tuple[str, ...] = (
    "no rain",
    "very light rain",
    "light rain",
    "normal rain",
    "heavy rain",
    "very heavy rain",
    "exceptional rain",
)

# -----------------------------
# CatBoost hyperparameters
# -----------------------------
# Hyperparameters copied from the notebook for the *tuned* models.
# 'loss_function' and 'eval_metric' are set at runtime based on quantile alpha.

CATBOOST_PARAMS: Dict[str, Dict[float, Dict]] = {
    "pm25": {
        0.25: dict(n_estimators=850, depth=3),
        0.50: dict(bootstrap_type="MVS", n_estimators=890, subsample=0.9),
        0.75: dict(colsample_bylevel=0.9, n_estimators=775, depth=3),
    },
    "o3": {
        0.25: dict(n_estimators=560, depth=3),
        0.50: dict(colsample_bylevel=0.8, n_estimators=800, depth=3),
        0.75: dict(n_estimators=640, depth=3),
    },
}

# Global defaults for all CatBoost models
CATBOOST_BASE: Dict = dict(
    random_state=0,
    verbose=0,
)

