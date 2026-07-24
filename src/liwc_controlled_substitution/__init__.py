"""Auditable controlled-substitution analysis for tabular LIWC features."""

from .controls import make_liwc_control
from .statistics import benjamini_hochberg, exact_sign_flip_test

__all__ = ["make_liwc_control", "benjamini_hochberg", "exact_sign_flip_test"]
__version__ = "0.1.0-rc1"
