"""Tests for BuhlmannModel and BuhlmannStraubModel."""

import math

import numpy as np
import pandas as pd
import pytest

from actuarcredibility import BuhlmannModel, BuhlmannStraubModel


@pytest.fixture
def panel():
    """Three-group, three-year panel with clear between-group separation."""
    return pd.DataFrame(
        {
            "g": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "t": [1, 2, 3] * 3,
            "y": [0.62, 0.58, 0.65, 0.81, 0.77, 0.84, 0.45, 0.52, 0.48],
            "w": [
                5_000_000.0,
                5_200_000.0,
                5_500_000.0,
                2_000_000.0,
                2_100_000.0,
                2_300_000.0,
                8_000_000.0,
                8_500_000.0,
                9_000_000.0,
            ],
        }
    )


def test_buhlmann_fit_returns_self(panel):
    m = BuhlmannModel()
    out = m.fit(panel, group_col="g", observation_col="y")
    assert out is m


def test_buhlmann_factor_in_unit_interval(panel):
    m = BuhlmannModel().fit(panel, group_col="g", observation_col="y")
    z = m.credibility_factor()
    assert ((z >= 0) & (z <= 1)).all()


def test_buhlmann_premium_blends_individual_and_collective(panel):
    m = BuhlmannModel().fit(panel, group_col="g", observation_col="y")
    z = m.credibility_factor()
    premium = m.credibility_premium()
    summary = m.summary()
    mu = m.structural_parameters["mu"]
    expected = z * summary["mean"] + (1.0 - z) * mu
    np.testing.assert_allclose(premium.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_buhlmann_balanced_matches_textbook_formula():
    """In the balanced case Z = n / (n + v/a) with v = mean(s^2_i) and
    a = sample variance of group means - v / n."""
    rng = np.random.default_rng(42)
    n = 8
    means = np.array([0.5, 0.7, 0.9])
    rows = []
    for g, mu in zip(["A", "B", "C"], means, strict=True):
        for t in range(n):
            rows.append({"g": g, "t": t, "y": mu + rng.normal(0, 0.05)})
    df = pd.DataFrame(rows)
    m = BuhlmannModel().fit(df, group_col="g", observation_col="y")
    params = m.structural_parameters

    s2 = df.groupby("g")["y"].var(ddof=1).mean()
    expected_v = float(s2)
    sample_var_means = df.groupby("g")["y"].mean().var(ddof=1)
    expected_a = sample_var_means - expected_v / n

    assert math.isclose(params["v"], expected_v, rel_tol=1e-9)
    assert math.isclose(params["a"], expected_a, rel_tol=1e-6)
    expected_z = n / (n + expected_v / expected_a)
    z = m.credibility_factor()
    np.testing.assert_allclose(z.to_numpy(), [expected_z] * 3, rtol=1e-9)


def test_buhlmann_straub_weights_drive_credibility(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    summary = m.summary()
    # Group C has the largest weight → must have the highest Z.
    assert summary.loc["C", "Z"] >= summary.loc["A", "Z"]
    assert summary.loc["A", "Z"] >= summary.loc["B", "Z"]


def test_buhlmann_straub_premium_blends(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    z = m.credibility_factor()
    summary = m.summary()
    mu = m.structural_parameters["mu"]
    expected = z * summary["mean"] + (1 - z) * mu
    np.testing.assert_allclose(
        m.credibility_premium().to_numpy(), expected.to_numpy(), rtol=1e-12
    )


def test_buhlmann_straub_predict_matches_premium(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    p = m.credibility_premium()
    for g in ["A", "B", "C"]:
        assert math.isclose(m.predict(g), p.loc[g], rel_tol=1e-12)


def test_buhlmann_straub_no_signal_collapses_to_grand_mean():
    """When all groups have identical observations, ``a`` clips to 0 and the
    credibility premium is the grand mean for every group."""
    # Same group means by construction → between-variance == 0, within > 0 →
    # a_raw is negative and gets clipped to 0.
    rows = []
    for g in ["A", "B", "C", "D"]:
        for y in [0.5, 0.7, 0.6, 0.8]:
            rows.append({"g": g, "y": y, "w": 1.0})
    df = pd.DataFrame(rows)
    m = BuhlmannStraubModel().fit(
        df, group_col="g", observation_col="y", weight_col="w"
    )
    z = m.credibility_factor()
    assert m.structural_parameters["a"] == 0.0
    assert (z == 0).all()
    p = m.credibility_premium()
    grand = df["y"].mean()
    np.testing.assert_allclose(p.to_numpy(), [grand] * 4, atol=1e-9)


def test_buhlmann_straub_rejects_non_positive_weights(panel):
    bad = panel.copy()
    bad.loc[0, "w"] = 0
    m = BuhlmannStraubModel()
    with pytest.raises(ValueError, match="positive"):
        m.fit(bad, group_col="g", observation_col="y", weight_col="w")


def test_buhlmann_rejects_missing_columns(panel):
    m = BuhlmannModel()
    with pytest.raises(KeyError):
        m.fit(panel, group_col="missing", observation_col="y")


def test_buhlmann_rejects_single_group():
    df = pd.DataFrame({"g": ["A"] * 5, "y": [0.5, 0.6, 0.55, 0.62, 0.58]})
    m = BuhlmannModel()
    with pytest.raises(ValueError, match="2 risk groups"):
        m.fit(df, group_col="g", observation_col="y")


def test_unfitted_raises():
    m = BuhlmannStraubModel()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.credibility_factor()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.credibility_premium()


def test_predict_unknown_group_raises(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    with pytest.raises(KeyError, match="not present"):
        m.predict("Z")


def test_buhlmann_rejects_single_period_per_group():
    """If every group has only one observation period, within-group variance
    cannot be estimated and fit must raise."""
    df = pd.DataFrame({"g": ["A", "B", "C"], "y": [0.5, 0.7, 0.9]})
    m = BuhlmannModel()
    with pytest.raises(ValueError, match="within-group variance"):
        m.fit(df, group_col="g", observation_col="y")


def test_buhlmann_straub_rejects_nan_observations(panel):
    bad = panel.copy()
    bad.loc[0, "y"] = float("nan")
    m = BuhlmannStraubModel()
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(bad, group_col="g", observation_col="y", weight_col="w")


def test_buhlmann_summary_columns(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    summary = m.summary()
    assert set(summary.columns) == {"weight", "mean", "Z", "P_cred"}
    assert list(summary.index) == ["A", "B", "C"]


def test_buhlmann_structural_parameters_unfitted():
    m = BuhlmannStraubModel()
    with pytest.raises(RuntimeError):
        _ = m.structural_parameters


def test_buhlmann_summary_unfitted():
    m = BuhlmannStraubModel()
    with pytest.raises(RuntimeError):
        m.summary()


def test_buhlmann_predict_unfitted():
    m = BuhlmannStraubModel()
    with pytest.raises(RuntimeError):
        m.predict("A")


def test_buhlmann_structural_parameters_keys(panel):
    m = BuhlmannStraubModel().fit(
        panel, group_col="g", observation_col="y", weight_col="w"
    )
    params = m.structural_parameters
    assert set(params.keys()) == {"mu", "v", "a", "a_raw", "k"}
