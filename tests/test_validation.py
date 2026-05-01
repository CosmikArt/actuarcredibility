"""Tests for internal validation helpers."""

import numpy as np
import pandas as pd
import pytest

from actuarcredibility._validation import (
    require_columns,
    require_finite,
    require_fitted,
    require_positive_weights,
)


def test_require_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_columns(df, ["a", "b"])


def test_require_columns_lists_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match=r"Missing required column"):
        require_columns(df, ["a", "b", "c"])


def test_require_fitted_passes_when_true():
    require_fitted(True, "Foo")


def test_require_fitted_raises_when_false():
    with pytest.raises(RuntimeError, match=r"Foo has not been fitted"):
        require_fitted(False, "Foo")


def test_require_positive_weights_passes():
    require_positive_weights(np.array([1.0, 2.0, 3.0]), "w")


def test_require_positive_weights_raises_on_zero():
    with pytest.raises(ValueError, match=r"strictly positive"):
        require_positive_weights(np.array([1.0, 0.0]), "w")


def test_require_positive_weights_raises_on_negative():
    with pytest.raises(ValueError, match=r"strictly positive"):
        require_positive_weights(np.array([-1.0, 1.0]), "w")


def test_require_finite_passes():
    require_finite(np.array([1.0, 2.0, 3.0]), "x")


def test_require_finite_raises_on_nan():
    with pytest.raises(ValueError, match=r"non-finite"):
        require_finite(np.array([1.0, np.nan]), "x")


def test_require_finite_raises_on_inf():
    with pytest.raises(ValueError, match=r"non-finite"):
        require_finite(np.array([1.0, np.inf]), "x")
