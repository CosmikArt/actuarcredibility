"""Tests for LimitedFluctuationCredibility."""

import math

import pytest

from actuarcredibility import LimitedFluctuationCredibility


def test_full_credibility_classical_standard():
    """k=0.05, p=0.90, Poisson → 1082 claims (Longley-Cook 1962)."""
    cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    assert cred.full_credibility_standard() == 1082


def test_z_value_at_p_90():
    cred = LimitedFluctuationCredibility(p=0.90)
    assert math.isclose(cred.z_value, 1.6448536269514722, rel_tol=1e-9)


def test_full_credibility_scales_with_cv_squared():
    cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    n_poisson = cred.full_credibility_standard(cv=1.0)
    n_doubled = cred.full_credibility_standard(cv=math.sqrt(2.0))
    assert n_doubled == pytest.approx(2 * n_poisson, abs=1)


def test_partial_credibility_square_root_rule():
    cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    n_full = cred.full_credibility_standard()
    z_500 = cred.credibility_factor(n_claims=500)
    assert math.isclose(z_500, math.sqrt(500 / n_full), rel_tol=1e-12)


def test_credibility_factor_capped_at_one():
    cred = LimitedFluctuationCredibility()
    assert cred.credibility_factor(n_claims=10_000) == 1.0


def test_credibility_premium_blends():
    cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    z = cred.credibility_factor(n_claims=500)
    expected = z * 0.72 + (1 - z) * 0.65
    assert math.isclose(
        cred.credibility_premium(observed=0.72, prior=0.65, n_claims=500),
        expected,
        rel_tol=1e-12,
    )


def test_invalid_k():
    with pytest.raises(ValueError):
        LimitedFluctuationCredibility(k=0.0)
    with pytest.raises(ValueError):
        LimitedFluctuationCredibility(k=1.5)


def test_invalid_p():
    with pytest.raises(ValueError):
        LimitedFluctuationCredibility(p=0.0)
    with pytest.raises(ValueError):
        LimitedFluctuationCredibility(p=1.0)


def test_invalid_cv():
    cred = LimitedFluctuationCredibility()
    with pytest.raises(ValueError):
        cred.full_credibility_standard(cv=0.0)


def test_negative_n_claims_rejected():
    cred = LimitedFluctuationCredibility()
    with pytest.raises(ValueError):
        cred.credibility_factor(n_claims=-10)


def test_at_p_99_higher_threshold():
    cred = LimitedFluctuationCredibility(k=0.05, p=0.99)
    cred_90 = LimitedFluctuationCredibility(k=0.05, p=0.90)
    assert cred.full_credibility_standard() > cred_90.full_credibility_standard()
