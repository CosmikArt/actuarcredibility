"""Smoke tests — verify imports, version, and basic instantiation."""

import actuarcredibility
from actuarcredibility import (
    BayesianCredibility,
    BuhlmannModel,
    BuhlmannStraubModel,
    HachemeisterRegression,
    JewellHierarchical,
    LimitedFluctuationCredibility,
)


def test_version_exists():
    assert hasattr(actuarcredibility, "__version__")
    assert isinstance(actuarcredibility.__version__, str)


def test_buhlmann_instantiation():
    assert BuhlmannModel() is not None


def test_buhlmann_straub_instantiation():
    assert BuhlmannStraubModel() is not None


def test_jewell_instantiation():
    assert JewellHierarchical() is not None


def test_hachemeister_instantiation():
    assert HachemeisterRegression() is not None


def test_limited_fluctuation_instantiation():
    cred = LimitedFluctuationCredibility()
    assert cred.k == 0.05
    assert cred.p == 0.90


def test_limited_fluctuation_custom_params():
    cred = LimitedFluctuationCredibility(k=0.10, p=0.95)
    assert cred.k == 0.10
    assert cred.p == 0.95


def test_bayesian_instantiation():
    model = BayesianCredibility()
    assert model is not None


def test_bayesian_custom_prior():
    model = BayesianCredibility(prior_config={"mu_prior_sd": 10.0})
    assert model._prior_config == {"mu_prior_sd": 10.0}
