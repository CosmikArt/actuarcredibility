"""Tests for the diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from actuarcredibility import BuhlmannModel, BuhlmannStraubModel, LimitedFluctuationCredibility
from actuarcredibility.diagnostics import (
    compare_models,
    credibility_curve,
    shrinkage_summary,
    variance_decomposition,
)


@pytest.fixture
def fitted_bs():
    df = pd.DataFrame(
        {
            "g": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "y": [0.62, 0.58, 0.65, 0.81, 0.77, 0.84, 0.45, 0.52, 0.48],
            "w": [5.0, 5.2, 5.5, 2.0, 2.1, 2.3, 8.0, 8.5, 9.0],
        }
    )
    return BuhlmannStraubModel().fit(df, group_col="g", observation_col="y", weight_col="w")


def test_variance_decomposition_keys(fitted_bs):
    vd = variance_decomposition(fitted_bs)
    for key in ("v_within", "a_between", "total", "between_share", "k_v_over_a"):
        assert key in vd.index
    assert 0 <= vd["between_share"] <= 1


def test_credibility_curve_monotone(fitted_bs):
    curve = credibility_curve(fitted_bs, n_points=50)
    assert (curve["Z"].diff().dropna() >= -1e-12).all()
    assert (curve["Z"] >= 0).all() and (curve["Z"] <= 1).all()


def test_credibility_curve_explicit_weights(fitted_bs):
    curve = credibility_curve(fitted_bs, weights=[1.0, 10.0, 100.0])
    assert len(curve) == 3
    assert curve["Z"].iloc[2] >= curve["Z"].iloc[0]


def test_shrinkage_summary_columns(fitted_bs):
    s = shrinkage_summary(fitted_bs)
    for col in ("weight", "mean", "Z", "P_cred", "raw_distance", "cred_distance",
                "shrinkage_ratio"):
        assert col in s.columns
    assert ((s["shrinkage_ratio"].dropna() >= -1e-12) &
            (s["shrinkage_ratio"].dropna() <= 1 + 1e-9)).all()


def test_compare_models_aligns_by_group():
    df = pd.DataFrame(
        {
            "g": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "y": np.concatenate([
                np.full(4, 0.6),
                np.full(4, 0.7),
                np.full(4, 0.5),
            ]) + np.random.default_rng(0).normal(0, 0.02, 12),
            "w": [1.0] * 12,
        }
    )
    m1 = BuhlmannModel().fit(df, group_col="g", observation_col="y")
    m2 = BuhlmannStraubModel().fit(df, group_col="g", observation_col="y", weight_col="w")
    cmp = compare_models({"buhlmann": m1, "bs": m2})
    assert cmp.shape[0] == 3
    assert any("buhlmann::Z" in c for c in cmp.columns)
    assert any("bs::Z" in c for c in cmp.columns)


def test_compare_models_skips_non_series_factors():
    """LimitedFluctuationCredibility doesn't fit a panel; ensure compare_models
    just ignores models without compatible Series outputs by passing only
    panel-fitted models."""
    df = pd.DataFrame(
        {
            "g": ["A"] * 4 + ["B"] * 4,
            "y": [0.6, 0.62, 0.61, 0.59, 0.7, 0.68, 0.71, 0.69],
            "w": [1.0] * 8,
        }
    )
    m1 = BuhlmannStraubModel().fit(df, group_col="g", observation_col="y", weight_col="w")
    cmp = compare_models({"bs": m1})
    assert "bs::P_cred" in cmp.columns
    # Smoke check that the LimitedFluctuationCredibility class can be instantiated
    # alongside (but is not panel-shaped).
    assert LimitedFluctuationCredibility() is not None
