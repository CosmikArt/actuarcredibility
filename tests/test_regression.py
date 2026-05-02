"""Tests for HachemeisterRegression."""

import numpy as np
import pandas as pd
import pytest

from actuarcredibility import HachemeisterRegression


@pytest.fixture
def trend_panel():
    rng = np.random.default_rng(123)
    rows = []
    for g in range(4):
        intercept = 0.6 + 0.05 * g
        slope = 0.02 + 0.005 * g
        for t in range(8):
            rows.append(
                {
                    "g": g,
                    "t": float(t),
                    "y": intercept + slope * t + rng.normal(0, 0.01),
                    "w": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_hachemeister_fit_returns_self(trend_panel):
    m = HachemeisterRegression()
    out = m.fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    assert out is m


def test_hachemeister_coefficients_shape(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    coefs = m.coefficients()
    assert coefs.shape == (4, 2)
    assert "_intercept" in coefs.columns
    assert "t" in coefs.columns


def test_hachemeister_structural_a_is_psd(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    A = m.structural_parameters["A"]
    eig = np.linalg.eigvalsh(A)
    assert (eig >= -1e-9).all()
    assert m.structural_parameters["sigma2"] >= 0


def test_hachemeister_credibility_matrices_finite(trend_panel):
    """Z_i = A (A + sigma^2 S_i)^{-1} is generally non-symmetric; just
    ensure all entries are finite and free of pathological values."""
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    for g in range(4):
        Z = m.credibility_matrix(g).to_numpy()
        assert np.all(np.isfinite(Z))


def test_hachemeister_predict_at_average_time_matches_premium(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    premium = m.credibility_premium()
    # The premium uses the within-group weighted-mean of t (here = 3.5).
    avg_t = trend_panel.groupby("g")["t"].mean()
    for g in range(4):
        assert m.predict(g, t=avg_t.loc[g]) == pytest.approx(premium.loc[g], rel=1e-12)


def test_hachemeister_high_signal_recovers_individual_coefs():
    """With low residual noise and strong between-group separation, sigma^2
    becomes tiny relative to A, so Z_i ≈ I and credibility coefficients
    should be close to the individual WLS estimates."""
    rng = np.random.default_rng(7)
    rows = []
    for g in range(5):
        intercept = 0.4 + 0.1 * g
        slope = 0.05 * g
        for t in range(20):
            rows.append(
                {"g": g, "t": float(t),
                 "y": intercept + slope * t + rng.normal(0, 1e-4),
                 "w": 1.0}
            )
    df = pd.DataFrame(rows)
    m = HachemeisterRegression().fit(
        df, group_col="g", observation_col="y", time_col="t", weight_col="w",
    )
    raw = m.coefficients_individual().to_numpy()
    cred = m.coefficients().to_numpy()
    np.testing.assert_allclose(cred, raw, atol=1e-3)


def test_hachemeister_requires_two_groups():
    df = pd.DataFrame({"g": [0] * 6, "t": list(range(6)), "y": list(range(6)), "w": [1.0] * 6})
    m = HachemeisterRegression()
    with pytest.raises(ValueError, match="2 risk groups"):
        m.fit(df, group_col="g", observation_col="y", time_col="t", weight_col="w")


def test_hachemeister_singular_design_rejected():
    # Two groups but each only has one observation → cannot fit p=2 regression.
    df = pd.DataFrame({
        "g": [0, 1],
        "t": [0.0, 0.0],
        "y": [1.0, 2.0],
        "w": [1.0, 1.0],
    })
    m = HachemeisterRegression()
    with pytest.raises(ValueError):
        m.fit(df, group_col="g", observation_col="y", time_col="t", weight_col="w")


def test_hachemeister_predict_unknown_group_raises(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    with pytest.raises(KeyError):
        m.predict(99, t=1.0)


def test_hachemeister_predict_missing_covariate_raises(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    with pytest.raises(KeyError, match="Missing value"):
        m.predict(0)


def test_hachemeister_requires_design_inputs():
    df = pd.DataFrame({"g": [0, 0, 1, 1], "y": [0.5, 0.6, 0.7, 0.8], "w": [1.0] * 4})
    m = HachemeisterRegression()
    with pytest.raises(ValueError, match="time_col or covariates"):
        m.fit(df, group_col="g", observation_col="y", weight_col="w")


def test_hachemeister_singular_xtwx_rejected():
    """Three obs per group but all at the same t → rank-deficient X'WX."""
    df = pd.DataFrame({
        "g": [0, 0, 0, 1, 1, 1],
        "t": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "y": [0.5, 0.6, 0.55, 0.7, 0.72, 0.71],
        "w": [1.0] * 6,
    })
    m = HachemeisterRegression()
    with pytest.raises(ValueError, match=r"[Ss]ingular"):
        m.fit(df, group_col="g", observation_col="y", time_col="t", weight_col="w")


def test_hachemeister_no_weight_col_uses_unit_weights(trend_panel):
    df = trend_panel.drop(columns=["w"])
    m = HachemeisterRegression().fit(df, group_col="g", observation_col="y", time_col="t")
    assert m.coefficients().shape == (4, 2)


def test_hachemeister_with_explicit_covariates(trend_panel):
    df = trend_panel.copy()
    df["t2"] = df["t"] ** 2
    m = HachemeisterRegression().fit(
        df, group_col="g", observation_col="y",
        covariates=["t", "t2"], weight_col="w",
    )
    coefs = m.coefficients()
    assert list(coefs.columns) == ["_intercept", "t", "t2"]


def test_hachemeister_a_clipped_to_zero_when_groups_identical():
    """Identical groups → between-coefficient covariance is zero, A clips to 0,
    triggers the Z_sum_inv fallback through information-weighted collective."""
    rows = []
    base = [0.5, 0.55, 0.6, 0.65, 0.7]
    for g in range(4):
        for t, y in enumerate(base):
            rows.append({"g": g, "t": float(t), "y": y, "w": 1.0})
    df = pd.DataFrame(rows)
    m = HachemeisterRegression().fit(
        df, group_col="g", observation_col="y", time_col="t", weight_col="w",
    )
    # All groups have identical coefficients → A_raw <= 0 → A = 0.
    A = m.structural_parameters["A"]
    assert np.allclose(A, 0.0)
    # Credibility coefficients should equal the collective for every group.
    cred = m.coefficients().to_numpy()
    coll = m.coefficients_collective().to_numpy()
    for i in range(cred.shape[0]):
        np.testing.assert_allclose(cred[i], coll, atol=1e-9)


def test_hachemeister_credibility_factor_long_dataframe(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    long_df = m.credibility_factor()
    # 4 groups × 2 design columns = 8 rows
    assert long_df.shape[0] == 8
    assert "row" in long_df.columns
    assert "_intercept" in long_df.columns


def test_hachemeister_credibility_matrix_unknown_group(trend_panel):
    m = HachemeisterRegression().fit(
        trend_panel, group_col="g", observation_col="y",
        time_col="t", weight_col="w",
    )
    with pytest.raises(KeyError, match="not present"):
        m.credibility_matrix(999)


def test_hachemeister_unfitted_methods_raise():
    m = HachemeisterRegression()
    for fn in (m.coefficients, m.coefficients_individual,
               m.coefficients_collective, m.credibility_factor,
               m.credibility_premium):
        with pytest.raises(RuntimeError):
            fn()
    with pytest.raises(RuntimeError):
        m.credibility_matrix(0)
    with pytest.raises(RuntimeError):
        m.predict(0, t=1.0)
    with pytest.raises(RuntimeError):
        _ = m.structural_parameters


def test_hachemeister_zero_dof_rejected():
    """With p=2 design and exactly 2 observations per group, sum(n_i - p) = 0
    and residual variance cannot be estimated, so fit must raise."""
    df = pd.DataFrame({
        "g": [0, 0, 1, 1, 2, 2],
        "t": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "y": [0.5, 0.6, 0.7, 0.8, 0.6, 0.7],
        "w": [1.0] * 6,
    })
    m = HachemeisterRegression()
    with pytest.raises(ValueError, match="degrees of freedom"):
        m.fit(df, group_col="g", observation_col="y", time_col="t", weight_col="w")


def test_hachemeister_rejects_nan_observations():
    df = pd.DataFrame({
        "g": [0, 0, 0, 1, 1, 1],
        "t": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "y": [0.5, np.nan, 0.6, 0.7, 0.72, 0.71],
        "w": [1.0] * 6,
    })
    m = HachemeisterRegression()
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(df, group_col="g", observation_col="y", time_col="t", weight_col="w")
