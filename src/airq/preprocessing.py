from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, RAIN_BINS, RAIN_LABELS
from .utils import safe_copy


# Wind degrees -> cardinal directions (as in notebook)
WIND_DIRECTIONS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW','NW', 'NNW', 'N']
DEGREES_DIRECTION = [d * 22.5 for d in range(17)]


def cardinal_direction(degrees: float) -> str:
    if degrees is None or (isinstance(degrees, float) and math.isnan(degrees)):
        return "UNKNOWN"
    try:
        deg = float(degrees)
    except Exception:
        return "UNKNOWN"
    if deg > 360:
        deg -= 360
    closest_match = 0
    delta = abs(DEGREES_DIRECTION[0] - deg)
    for i in range(1, len(DEGREES_DIRECTION)):
        diff = abs(DEGREES_DIRECTION[i] - deg)
        if diff < delta:
            delta = diff
            closest_match = i
    return WIND_DIRECTIONS[closest_match]


def day_moment(time: pd.Timestamp) -> str:
    if pd.isna(time):
        return "empty"
    h = getattr(time, "hour", None)
    if h is None:
        return "empty"
    if 0 <= h < 6:
        return "night"
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    return "evening"


def rain_classes(value: float) -> str:
    if pd.isna(value):
        # keep a dedicated missing label (CatBoost handles NaN, but this is categorical)
        return "missing"
    v = float(value)
    if v == 0.0:
        return RAIN_LABELS[0]
    b1, b2, b3, b4, b5 = RAIN_BINS
    if v <= b1:
        return RAIN_LABELS[1]
    if v <= b2:
        return RAIN_LABELS[2]
    if v <= b3:
        return RAIN_LABELS[3]
    if v <= b4:
        return RAIN_LABELS[4]
    if v <= b5:
        return RAIN_LABELS[5]
    return RAIN_LABELS[6]


@dataclass
class AirQualityPreprocessor:
    """Stateless preprocessor that *never* drops rows.

    It mirrors the transformations used in the original notebook:
    - datetime parsing
    - time-of-day bucketing for various *time* columns
    - wind direction conversion (degrees -> cardinal)
    - precipitation -> Rain Classes using fixed thresholds
    - creation of lagged and rolling features
    - simple imputations for wind speed columns
    - categorical casting
    """

    date_col: str = "date"
    precipitation_col: str = "PRECIPITATION TOTAL"
    wind_dir_col: str = "WIND direction gust"

    pollutant_cols: Sequence[str] = ("pm2.5", "pm10", "no2", "so2", "o3")
    weather_prev_cols: Sequence[str] = (
        "HUMIDITY avg",
        "HUMIDITY min",
        "AIR TEMP T avg",
        "AIR TEMP T max",
        "WIND speed gust",
        "Rain Classes",
    )

    time_cols: Sequence[str] = (
        "AIR TEMP time min",
        "AIR TEMP time max",
        "HUMIDITY time min",
        "HUMIDITY time max",
        "WIND time gust",
    )

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = safe_copy(df_raw)

        # --- date handling ---
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
            df["month of year"] = df[self.date_col].dt.strftime("%B")
            df["year"] = df[self.date_col].dt.year.astype("Int64").astype(str)
            df = df.set_index(self.date_col)
        # if already indexed by date, keep it

        # --- convert time columns to datetime then bucket into moments ---
        for c in self.time_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], format="%H:%M:%S", errors="coerce")
                df[c] = df[c].apply(day_moment)

        # --- wind direction degrees -> cardinal ---
        if self.wind_dir_col in df.columns:
            df[self.wind_dir_col] = df[self.wind_dir_col].apply(cardinal_direction)

        # --- precipitation -> Rain Classes ---
        if self.precipitation_col in df.columns:
            df["Rain Classes"] = df[self.precipitation_col].apply(rain_classes)
            # replicate notebook behavior: drop numeric precipitation after categorization
            df.drop(columns=[self.precipitation_col], inplace=True)

        # --- numeric coercions for some weather columns (as in notebook) ---
        for c in ["WIND speed avg", "WIND speed gust", "DAYLIGHT HOURS"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- lag + rolling for pollutants (first 5 columns in notebook) ---
        for c in self.pollutant_cols:
            if c in df.columns:
                df[f"PREV_{c}"] = df[c].shift(1)
                # interpolate then rolling then shift (as in notebook)
                df[f"PREV_RMEAN_{c}"] = df[c].interpolate().rolling(window=3, min_periods=1).mean().shift(1)

        # --- lag for weather cols + rolling (not shifted) ---
        for c in self.weather_prev_cols:
            if c in df.columns:
                df[f"PREV_{c}"] = df[c].shift(1)
                if c not in ("Rain Classes", "HUMIDITY time min"):
                    # note: in notebook this rolling is not interpolated
                    df[f"RMEAN_{c}"] = pd.to_numeric(df[c], errors="coerce").rolling(window=3, min_periods=1).mean()

        # --- imputations (mirror notebook intent; do NOT drop rows) ---
        if "WIND speed avg" in df.columns:
            df["WIND speed avg"] = df["WIND speed avg"].fillna(0)
        if "WIND speed gust" in df.columns and "WIND speed avg" in df.columns:
            df["WIND speed gust"] = df["WIND speed gust"].fillna(df["WIND speed avg"])
        if "PREV_WIND speed gust" in df.columns and "WIND speed avg" in df.columns:
            df["PREV_WIND speed gust"] = df["PREV_WIND speed gust"].fillna(df["WIND speed avg"].shift(1))

        # replicate notebook quirk: PREV_so2 cast to string (then category)
        if "PREV_so2" in df.columns:
            df["PREV_so2"] = df["PREV_so2"].astype("string")

        # --- cast categoricals ---
        for c in CATEGORICAL_COLS:
            if c in df.columns:
                df[c] = df[c].astype("category")

        return df

