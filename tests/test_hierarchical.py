"""Tests for JewellHierarchical."""

import numpy as np
import pandas as pd
import pytest

from actuarcredibility import JewellHierarchical


@pytest.fixture
def two_level_panel():
    """Two-level panel: 2 regions × 2 risks each × 3 years."""
    rng = np.random.default_rng(0)
    rows = []
    region_offset = {"N": -0.05, "S": 0.05}
    risk_offset = {"A": -0.02, "B": 0.02, "C": 0.0, "D": 0.04}
    base = 0.65
    for region, risks in [("N", ["A", "B"]), ("S", ["C", "D"])]:
        for risk in risks:
            for year in range(3):
                y = base + region_offset[region] + risk_offset[risk] + rng.normal(0, 0.01)
                rows.append(
                    {
                        "region": region,
                        "risk": risk,
                        "year": year,
                        "y": y,
                        "w": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_jewell_fit_returns_self(two_level_panel):
    m = JewellHierarchical()
    out = m.fit(
        two_level_panel,
        hierarchy_cols=["region", "risk"],
        observation_col="y",
        weight_col="w",
    )
    assert out is m


def test_jewell_premium_one_per_leaf(two_level_panel):
    m = JewellHierarchical().fit(
        two_level_panel,
        hierarchy_cols=["region", "risk"],
        observation_col="y",
        weight_col="w",
    )
    p = m.credibility_premium()
    assert len(p) == 4
    assert set(p.index.get_level_values("risk")) == {"A", "B", "C", "D"}


def test_jewell_credibility_factor_each_level(two_level_panel):
    m = JewellHierarchical().fit(
        two_level_panel,
        hierarchy_cols=["region", "risk"],
        observation_col="y",
        weight_col="w",
    )
    z_risk = m.credibility_factor("risk")
    z_region = m.credibility_factor("region")
    assert ((z_risk >= 0) & (z_risk <= 1)).all()
    assert ((z_region >= 0) & (z_region <= 1)).all()


def test_jewell_unknown_level_raises(two_level_panel):
    m = JewellHierarchical().fit(
        two_level_panel,
        hierarchy_cols=["region", "risk"],
        observation_col="y",
        weight_col="w",
    )
    with pytest.raises(KeyError, match="not part of"):
        m.credibility_factor("country")


def test_jewell_no_weights_uses_ones(two_level_panel):
    df = two_level_panel.drop(columns=["w"])
    m = JewellHierarchical().fit(
        df, hierarchy_cols=["region", "risk"], observation_col="y"
    )
    p = m.credibility_premium()
    assert len(p) == 4


def test_jewell_single_level_falls_back_to_grand_mean():
    df = pd.DataFrame(
        {
            "risk": ["A", "A", "B", "B", "C", "C"],
            "y": [0.6, 0.62, 0.55, 0.58, 0.7, 0.72],
            "w": [1.0] * 6,
        }
    )
    m = JewellHierarchical().fit(
        df, hierarchy_cols=["risk"], observation_col="y", weight_col="w"
    )
    # Single level: every leaf "premium" equals the grand mean (no within-parent
    # B-S step is possible because there is no parent).
    p = m.credibility_premium()
    np.testing.assert_allclose(p.to_numpy(), [m.grand_mean] * len(p), rtol=1e-12)


def test_jewell_unfitted_raises():
    m = JewellHierarchical()
    with pytest.raises(RuntimeError):
        m.credibility_premium()


def test_jewell_empty_hierarchy_rejected():
    m = JewellHierarchical()
    with pytest.raises(ValueError, match="at least one"):
        m.fit(
            pd.DataFrame({"y": [1.0]}),
            hierarchy_cols=[],
            observation_col="y",
        )
