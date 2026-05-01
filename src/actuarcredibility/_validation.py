"""Internal input validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )


def require_fitted(fitted: bool, cls_name: str) -> None:
    if not fitted:
        raise RuntimeError(
            f"{cls_name} has not been fitted yet. Call .fit(...) before "
            f"requesting credibility factors or premiums."
        )


def require_positive_weights(weights: np.ndarray, weight_col: str) -> None:
    if np.any(weights <= 0):
        raise ValueError(
            f"All weights in column '{weight_col}' must be strictly positive; "
            f"found {(weights <= 0).sum()} non-positive value(s)."
        )


def require_finite(values: np.ndarray, col: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Column '{col}' contains non-finite values (NaN or inf). "
            f"Drop or impute them before fitting."
        )
