"""Statistical AI-text detection via empirical types and large deviations."""

from .detector import SanovDetector
from .types import EmpiricalType, TypeEstimator, kl_divergence
from .watermark import GreenlistWatermarker

__all__ = ["EmpiricalType", "GreenlistWatermarker",
           "SanovDetector", "TypeEstimator", "kl_divergence"]
