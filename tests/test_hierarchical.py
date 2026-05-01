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


def test_jewell_single_child_per_parent():
    """A parent with only one child cannot estimate between-child variance,
    triggering the Z=0 fallback."""
    df = pd.DataFrame({
        "region": ["N", "N", "S", "S"],
        "risk":   ["A", "A", "B", "B"],
        "y":      [0.6, 0.62, 0.8, 0.78],
        "w":      [1.0, 1.0, 1.0, 1.0],
    })
    m = JewellHierarchical().fit(
        df, hierarchy_cols=["region", "risk"],
        observation_col="y", weight_col="w",
    )
    z_risk = m.credibility_factor("risk")
    # Each region has only one child risk → Z = 0 for every leaf.
    assert (z_risk == 0).all()


def test_jewell_zero_between_variance_at_inner_level():
    """When the between-leaf variance within a parent is zero (identical
    leaf means), a is clipped to 0 and Z is 0 for those leaves."""
    df = pd.DataFrame({
        "region": ["N"] * 4 + ["S"] * 4,
        "risk":   ["A", "A", "B", "B", "C", "C", "D", "D"],
        "y":      [0.6, 0.62, 0.6, 0.62,   # both N risks have identical means
                   0.7, 0.72, 0.7, 0.72],  # both S risks identical too
        "w":      [1.0] * 8,
    })
    m = JewellHierarchical().fit(
        df, hierarchy_cols=["region", "risk"],
        observation_col="y", weight_col="w",
    )
    z_risk = m.credibility_factor("risk")
    assert (z_risk == 0).all()


def test_jewell_rejects_missing_hierarchy_column():
    df = pd.DataFrame({"y": [0.5, 0.6], "w": [1.0, 1.0]})
    m = JewellHierarchical()
    with pytest.raises(KeyError):
        m.fit(df, hierarchy_cols=["region"], observation_col="y", weight_col="w")


def test_jewell_rejects_missing_weight_column():
    df = pd.DataFrame({
        "region": ["N", "S"], "y": [0.5, 0.6],
    })
    m = JewellHierarchical()
    with pytest.raises(KeyError):
        m.fit(
            df, hierarchy_cols=["region"], observation_col="y", weight_col="missing",
        )


def test_jewell_credibility_factor_unfitted():
    m = JewellHierarchical()
    with pytest.raises(RuntimeError):
        m.credibility_factor()


def test_jewell_grand_mean_unfitted():
    m = JewellHierarchical()
    with pytest.raises(RuntimeError):
        _ = m.grand_mean


def test_jewell_three_level_hierarchy():
    """Full three-level hierarchy: region → territory → risk."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["N", "S"]:
        for terr in ["T1", "T2"]:
            for risk in ["R1", "R2"]:
                for year in range(3):
                    base = (
                        0.6
                        + (0.05 if region == "N" else -0.05)
                        + (0.02 if terr == "T1" else -0.02)
                        + (0.01 if risk == "R1" else -0.01)
                    )
                    rows.append({
                        "region": region, "territory": terr, "risk": risk,
                        "year": year,
                        "y": base + rng.normal(0, 0.005),
                        "w": 1.0,
                    })
    df = pd.DataFrame(rows)
    m = JewellHierarchical().fit(
        df,
        hierarchy_cols=["region", "territory", "risk"],
        observation_col="y", weight_col="w",
    )
    z_risk = m.credibility_factor("risk")
    z_terr = m.credibility_factor("territory")
    z_region = m.credibility_factor("region")
    assert len(z_risk) == 8
    assert len(z_terr) == 4
    assert len(z_region) == 2
    assert len(m.credibility_premium()) == 8
