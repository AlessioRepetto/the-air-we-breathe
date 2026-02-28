"""EU AQI classification utilities.

This is optional for modeling but useful for analysis/inference outputs.
The thresholds mirror the notebook logic (Good/Fair/Moderate/Poor/Very Poor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "pm2.5": [10, 20, 25, 50],
    "pm10": [20, 40, 50, 100],
    "no2": [40, 90, 120, 230],
    "o3": [50, 100, 130, 240],
    "so2": [100, 200, 350, 500],
}

LEVELS = ["Good", "Fair", "Moderate", "Poor", "Very Poor"]


def classify_level(value: float, thresholds: List[float]) -> str:
    if pd.isna(value):
        return "missing_vals"
    if value <= thresholds[0]:
        return "Good"
    elif value <= thresholds[1]:
        return "Fair"
    elif value <= thresholds[2]:
        return "Moderate"
    elif value <= thresholds[3]:
        return "Poor"
    else:
        return "Very Poor"


def eu_aqi_category(row: pd.Series, pollutant_order: List[str]) -> str:
    # Worst pollutant among those available; if any required is missing -> missing_vals
    cats = []
    for p in pollutant_order:
        col = f"{p} CLASS"
        if col not in row.index:
            continue
        cats.append(row[col])
    if not cats:
        return "missing_vals"
    if any(c == "missing_vals" for c in cats):
        return "missing_vals"
    # pick worst according to LEVELS ordering
    worst_idx = max(LEVELS.index(c) for c in cats if c in LEVELS)
    return LEVELS[worst_idx]


@dataclass
class AqiCalculator:
    thresholds: Dict[str, List[float]] = None
    pollutant_order: Tuple[str, ...] = ("pm2.5", "pm10", "no2", "o3", "so2")

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {k: v[:] for k, v in DEFAULT_THRESHOLDS.items()}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for p, thr in self.thresholds.items():
            if p in df.columns:
                out[f"{p} CLASS"] = df[p].apply(lambda v: classify_level(v, thr))
        # compute worst category
        out["eu aqi"] = out.apply(lambda r: eu_aqi_category(r, list(self.pollutant_order)), axis=1)
        return out

