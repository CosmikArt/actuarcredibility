"""Smoke tests — verify imports and class instantiation."""

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
    assert actuarcredibility.__version__ == "0.0.1"


def test_buhlmann_instantiation():
    model = BuhlmannModel()
    assert model is not None
    assert model._fitted is False


def test_buhlmann_straub_instantiation():
    model = BuhlmannStraubModel()
    assert model is not None
    assert model._fitted is False


def test_jewell_instantiation():
    model = JewellHierarchical()
    assert model is not None
    assert model._fitted is False


def test_hachemeister_instantiation():
    model = HachemeisterRegression()
    assert model is not None
    assert model._fitted is False


def test_limited_fluctuation_instantiation():
    cred = LimitedFluctuationCredibility()
    assert cred is not None
    assert cred.k == 0.05
    assert cred.p == 0.90


def test_limited_fluctuation_custom_params():
    cred = LimitedFluctuationCredibility(k=0.10, p=0.95)
    assert cred.k == 0.10
    assert cred.p == 0.95


def test_bayesian_instantiation():
    model = BayesianCredibility()
    assert model is not None
    assert model._fitted is False


def test_bayesian_custom_prior():
    model = BayesianCredibility(prior_config={"mu_sd": 10.0})
    assert model._prior_config == {"mu_sd": 10.0}
