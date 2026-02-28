"""AirQ — The Air We Breathe

Core reusable code extracted from the original notebook/script.

Main entrypoint: :class:`airq.modeling.AirQualityModel`
"""

from .modeling import AirQualityModel, QuantileEnsembleCatBoost
from .preprocessing import AirQualityPreprocessor
from .aqi import AqiCalculator

__all__ = [
    "AirQualityModel",
    "QuantileEnsembleCatBoost",
    "AirQualityPreprocessor",
    "AqiCalculator",
]
