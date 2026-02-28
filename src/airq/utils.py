from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def mean_pinball_loss(y_true: Sequence[float], y_pred: Sequence[float], alpha: float) -> float:
    """Mean pinball loss for a given quantile alpha."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def enforce_quantile_monotonicity(q25: np.ndarray, q50: np.ndarray, q75: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enforce q25 <= q50 <= q75 elementwise.

    Returns corrected arrays and a boolean mask indicating where corrections occurred.
    """
    q25c = q25.copy()
    q50c = q50.copy()
    q75c = q75.copy()

    corrected = np.zeros_like(q25c, dtype=bool)

    # If median is below q25, set to q25
    m1 = q50c < q25c
    if np.any(m1):
        q50c[m1] = q25c[m1]
        corrected[m1] = True

    # If q75 below median, set to median
    m2 = q75c < q50c
    if np.any(m2):
        q75c[m2] = q50c[m2]
        corrected[m2] = True

    # If q25 above q75 (rare after steps, but just in case), set both to min/max ordering
    m3 = q25c > q75c
    if np.any(m3):
        lo = np.minimum(q25c[m3], q75c[m3])
        hi = np.maximum(q25c[m3], q75c[m3])
        q25c[m3] = lo
        q75c[m3] = hi
        # also keep median within bounds
        q50c[m3] = np.clip(q50c[m3], lo, hi)
        corrected[m3] = True

    return q25c, q50c, q75c, corrected


def clamp_non_negative(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Clamp arrays to be non-negative."""
    out = []
    for a in arrays:
        aa = a.copy()
        aa[aa < 0] = 0.0
        out.append(aa)
    return tuple(out)


def compute_iqr_limits(q25: np.ndarray, q75: np.ndarray, whisker: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute IQR and Tukey-style limits."""
    iqr = q75 - q25
    low = q25 - whisker * iqr
    high = q75 + whisker * iqr
    return iqr, low, high


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    # avoid view-related surprises
    return df.copy(deep=True)

