"""actuarcredibility — Credibility models for actuarial pricing.

Public API
----------
- :class:`BuhlmannModel` — non-parametric Bühlmann credibility (equal weights).
- :class:`BuhlmannStraubModel` — Bühlmann-Straub with exposure weights.
- :class:`JewellHierarchical` — multi-level hierarchical credibility.
- :class:`HachemeisterRegression` — regression credibility with covariates.
- :class:`LimitedFluctuationCredibility` — classical full/partial standards.
- :class:`BayesianCredibility` — PyMC bridge for MCMC-based credibility.

Diagnostics live in :mod:`actuarcredibility.diagnostics`.
"""

from actuarcredibility.bayesian import BayesianCredibility
from actuarcredibility.buhlmann import BuhlmannModel, BuhlmannStraubModel
from actuarcredibility.classical import LimitedFluctuationCredibility
from actuarcredibility.hierarchical import JewellHierarchical
from actuarcredibility.regression import HachemeisterRegression

__version__ = "0.1.0"

__all__ = [
    "BayesianCredibility",
    "BuhlmannModel",
    "BuhlmannStraubModel",
    "HachemeisterRegression",
    "JewellHierarchical",
    "LimitedFluctuationCredibility",
    "__version__",
]
