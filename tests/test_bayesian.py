"""Tests for the Bayesian credibility bridge.

Most tests skip when PyMC is not installed.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from actuarcredibility import BayesianCredibility

pymc_available = importlib.util.find_spec("pymc") is not None
pytestmark = pytest.mark.skipif(not pymc_available, reason="PyMC not installed")


@pytest.fixture
def panel():
    return pd.DataFrame(
        {
            "g": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "y": [0.62, 0.58, 0.65, 0.61, 0.81, 0.77, 0.84, 0.79, 0.45, 0.52, 0.48, 0.50],
            "w": [1.0] * 12,
        }
    )


def test_bayesian_fit_and_summarize(panel):
    m = BayesianCredibility(prior_config={"mu_prior_sd": 5.0}).fit(
        panel, group_col="g", observation_col="y", weight_col="w",
        samples=200, chains=2, tune=200, random_seed=0, progressbar=False,
    )
    z = m.credibility_factor()
    p = m.credibility_premium()
    assert set(z.index) == {"A", "B", "C"}
    assert ((z > 0) & (z < 1)).all()
    assert (p > 0).all()
    summary = m.posterior_summary()
    assert "mean" in summary.columns


def test_bayesian_unfitted_raises():
    m = BayesianCredibility()
    with pytest.raises(RuntimeError):
        m.credibility_premium()


def test_bayesian_missing_pymc(monkeypatch):
    """If PyMC is missing, fit should raise an informative ImportError."""
    import sys
    monkeypatch.setitem(sys.modules, "pymc", None)
    m = BayesianCredibility()
    with pytest.raises((ImportError, TypeError)):
        m._check_pymc()


def test_bayesian_fit_without_weight_col(panel):
    df = panel.drop(columns=["w"])
    m = BayesianCredibility().fit(
        df, group_col="g", observation_col="y",
        samples=100, chains=2, tune=100, random_seed=0, progressbar=False,
    )
    assert len(m.credibility_premium()) == 3


def test_bayesian_trace_property(panel):
    m = BayesianCredibility().fit(
        panel, group_col="g", observation_col="y", weight_col="w",
        samples=100, chains=2, tune=100, random_seed=0, progressbar=False,
    )
    trace = m.trace
    assert "posterior" in trace.groups()


def test_bayesian_trace_property_unfitted():
    m = BayesianCredibility()
    with pytest.raises(RuntimeError):
        _ = m.trace


def test_bayesian_factor_unfitted():
    m = BayesianCredibility()
    with pytest.raises(RuntimeError):
        m.credibility_factor()


def test_bayesian_summary_unfitted():
    m = BayesianCredibility()
    with pytest.raises(RuntimeError):
        m.posterior_summary()


def test_bayesian_constant_observations_uses_fallback_sd():
    """When all observations are identical, data_sd is 0 and the prior code
    swaps in 1.0 as a default scale."""
    df = pd.DataFrame({
        "g": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "y": [0.6] * 12,
        "w": [1.0] * 12,
    })
    m = BayesianCredibility().fit(
        df, group_col="g", observation_col="y", weight_col="w",
        samples=80, chains=2, tune=80, random_seed=1, progressbar=False,
    )
    # All groups have the same observation, so all posterior premiums hover
    # around 0.6.
    p = m.credibility_premium()
    assert ((p - 0.6).abs() < 0.5).all()


def test_bayesian_summary_returns_arviz_frame(panel):
    m = BayesianCredibility().fit(
        panel, group_col="g", observation_col="y", weight_col="w",
        samples=80, chains=2, tune=80, random_seed=0, progressbar=False,
    )
    summary = m.posterior_summary()
    assert "mean" in summary.columns
    assert "sd" in summary.columns


def test_bayesian_rejects_missing_columns():
    m = BayesianCredibility()
    with pytest.raises(KeyError):
        m.fit(
            pd.DataFrame({"y": [1.0, 2.0]}),
            group_col="missing", observation_col="y",
        )


def test_bayesian_rejects_missing_weight_col():
    m = BayesianCredibility()
    with pytest.raises(KeyError):
        m.fit(
            pd.DataFrame({"g": ["A", "B"], "y": [1.0, 2.0]}),
            group_col="g", observation_col="y", weight_col="missing",
        )
