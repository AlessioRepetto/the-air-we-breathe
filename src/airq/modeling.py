from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import (
    CATBOOST_BASE,
    CATBOOST_PARAMS,
    FEATURES_O3,
    FEATURES_PM25,
    REQUIRED_COLS_BY_TARGET,
)
from .preprocessing import AirQualityPreprocessor
from .utils import (
    clamp_non_negative,
    compute_iqr_limits,
    enforce_quantile_monotonicity,
    ensure_columns,
    safe_copy,
)


Target = Literal["pm25", "o3"]


def _features_for_target(target: Target) -> List[str]:
    if target == "pm25":
        return FEATURES_PM25[:]
    if target == "o3":
        return FEATURES_O3[:]
    raise ValueError(f"Unknown target: {target}")


def _cat_cols_from_X(X: pd.DataFrame) -> List[str]:
    return X.select_dtypes(include=["category"]).columns.tolist()


def time_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    n = len(X)
    cut = int(round(n * train_frac))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    return X_train, y_train, X_test, y_test


def build_training_matrix(df_features: pd.DataFrame, y: pd.Series, target: Target) -> Tuple[pd.DataFrame, pd.Series]:
    """Select target features and drop rows ONLY for training.

    Mirrors the notebook practice of dropping rows where key lag columns are missing,
    and ensures the target has no NaNs (CatBoost Quantile requires non-NaN y).
    """
    feats = _features_for_target(target)
    ensure_columns(df_features, feats)
    X = df_features[feats].copy()

    # Ensure numeric target (safe) and align on index
    y_num = pd.to_numeric(y, errors="coerce")

    required = REQUIRED_COLS_BY_TARGET[target]

    mask = np.ones(len(X), dtype=bool)

    # Drop rows with missing required lag columns (training only)
    for c in required:
        if c in X.columns:
            mask &= X[c].notna().to_numpy()

    # Drop rows with missing target values (training only)
    mask &= y_num.loc[X.index].notna().to_numpy()

    X = X.loc[mask]
    y2 = y_num.loc[X.index]
    return X, y2

@dataclass
class QuantileEnsembleCatBoost:
    """Three CatBoost quantile regressors (q25/q50/q75) with coherence correction."""

    target: Target
    quantiles: Tuple[float, float, float] = (0.25, 0.50, 0.75)
    models: Dict[float, CatBoostRegressor] = None
    feature_names_: List[str] = None
    cat_features_: List[str] = None

    def __post_init__(self):
        self.models = {} if self.models is None else self.models

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileEnsembleCatBoost":
        self.feature_names_ = X.columns.tolist()
        self.cat_features_ = _cat_cols_from_X(X)

        for q in self.quantiles:
            params = {}
            params.update(CATBOOST_BASE)
            params.update(CATBOOST_PARAMS[self.target].get(q, {}))
            model = CatBoostRegressor(
                loss_function=f"Quantile:alpha={q}",
                eval_metric=f"Quantile:alpha={q}",
                cat_features=self.cat_features_,
                **params,
            )
            model.fit(X, y, eval_set=[(X, y)], verbose=0)
            self.models[q] = model
        return self

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise RuntimeError("Model is not fitted.")
        ensure_columns(X, self.feature_names_)
        X2 = X[self.feature_names_]
        out = {}
        for q, m in self.models.items():
            out[f"q{int(q*100):02d}"] = m.predict(X2)
        return pd.DataFrame(out, index=X2.index)

    def predict_bands(self, X: pd.DataFrame, non_negative: bool = True, whisker: float = 1.5) -> pd.DataFrame:
        raw = self.predict_raw(X)
        q25 = raw["q25"].to_numpy(dtype=float)
        q50 = raw["q50"].to_numpy(dtype=float)
        q75 = raw["q75"].to_numpy(dtype=float)

        if non_negative:
            q25, q50, q75 = clamp_non_negative(q25, q50, q75)

        q25c, q50c, q75c, corrected = enforce_quantile_monotonicity(q25, q50, q75)
        iqr, low, high = compute_iqr_limits(q25c, q75c, whisker=whisker)
        if non_negative:
            low, = clamp_non_negative(low)

        out = pd.DataFrame(
            {
                "q25": q25c,
                "q50": q50c,
                "q75": q75c,
                "iqr": iqr,
                "low_limit": low,
                "high_limit": high,
                "corrected": corrected,
                "q25_raw": q25,
                "q50_raw": q50,
                "q75_raw": q75,
            },
            index=X.index,
        )
        return out


@dataclass
class AirQualityModel:
    """End-to-end model wrapper (preprocessing + per-target ensemble)."""

    preprocessor: AirQualityPreprocessor
    ensembles: Dict[Target, QuantileEnsembleCatBoost]

    @classmethod
    def create_empty(cls) -> "AirQualityModel":
        return cls(preprocessor=AirQualityPreprocessor(), ensembles={})

    def fit(self, df_raw: pd.DataFrame, target: Target) -> "AirQualityModel":
        df_feat = self.preprocessor.transform(df_raw)

        # target column must exist for training
        y_col = "pm2.5" if target == "pm25" else "o3"
        ensure_columns(df_feat, [y_col])
        y = pd.to_numeric(df_feat[y_col], errors="coerce")

        X_train_all, y_train_all = build_training_matrix(df_feat, y, target=target)

        # CatBoost can handle NaN; just ensure categorical dtype is preserved
        ensemble = QuantileEnsembleCatBoost(target=target)
        ensemble.fit(X_train_all, y_train_all)
        self.ensembles[target] = ensemble
        return self

    def predict(self, df_raw: pd.DataFrame, target: Target) -> pd.DataFrame:
        if target not in self.ensembles:
            raise RuntimeError(f"Target '{target}' is not fitted/loaded.")
        df_feat = self.preprocessor.transform(df_raw)

        feats = _features_for_target(target)
        ensure_columns(df_feat, feats)
        X_all = df_feat[feats].copy()

        # validity mask: do NOT drop rows, but mark invalid ones
        required = REQUIRED_COLS_BY_TARGET[target]
        is_valid = np.ones(len(X_all), dtype=bool)
        for c in required:
            if c in X_all.columns:
                is_valid &= X_all[c].notna().to_numpy()

        # run prediction only on valid subset
        pred = pd.DataFrame(index=X_all.index)
        pred["is_valid"] = is_valid
        pred["reason"] = np.where(is_valid, "", "missing_required_lags")

        if is_valid.any():
            bands_valid = self.ensembles[target].predict_bands(X_all.loc[is_valid])
            pred = pred.join(bands_valid, how="left")

        return pred

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "AirQualityModel":
        return joblib.load(Path(path))

