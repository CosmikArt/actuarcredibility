"""Core credibility models for actuarial pricing.

This module implements the major credibility frameworks used in property &
casualty insurance ratemaking:

- **Bühlmann** — non-parametric empirical Bayes credibility
- **Bühlmann-Straub** — extension with varying volume/exposure weights
- **Jewell** — multi-level hierarchical credibility for nested portfolios
- **Hachemeister** — regression credibility with covariate adjustment
- **Limited Fluctuation** — classical full and partial credibility standards
- **Bayesian** — MCMC-based credibility estimation via optional PyMC bridge

References
----------
- Bühlmann, H. (1967). "Experience Rating and Credibility." ASTIN Bulletin.
- Bühlmann, H. & Gisler, A. (2005). A Course in Credibility Theory. Springer.
- Jewell, W.S. (1975). "The Use of Collateral Data in Credibility Theory."
- Hachemeister, C.A. (1975). "Credibility for Regression Models with
  Application to Trend."
- Klugman, Panjer & Willmot. Loss Models: From Data to Decisions. Wiley.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class BuhlmannModel:
    """Non-parametric Bühlmann credibility model.

    The Bühlmann model estimates the credibility premium as a weighted average
    of the individual risk's experience and the collective (grand) mean. The
    credibility factor Z is derived from the ratio of the between-variance
    (process variance of the hypothetical means) to the total variance.

    The key structural parameters are:

    - **mu** — grand mean (collective premium)
    - **v** — expected value of the process variance (within-risk variance)
    - **a** — variance of the hypothetical means (between-risk variance)

    The credibility factor for a risk with n observation periods is:

        Z = n / (n + v/a) = n / (n + k),  where k = v/a

    And the credibility premium is:

        P_cred = Z * X_bar_i + (1 - Z) * mu

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate structural parameters.

    Examples
    --------
    >>> model = BuhlmannModel()
    >>> model.fit(data, group_col="risk", observation_col="loss_ratio")
    >>> model.credibility_factor()
    >>> model.credibility_premium()

    References
    ----------
    Bühlmann, H. (1967). "Experience Rating and Credibility." ASTIN Bulletin,
    4(3), 199-207.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._data: pd.DataFrame | None = None
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._mu: float | None = None
        self._v: float | None = None
        self._a: float | None = None

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
    ) -> BuhlmannModel:
        """Estimate structural parameters from observed data.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with one row per (group, period) observation.
        group_col : str
            Column identifying the risk group (e.g., risk class, territory).
        observation_col : str
            Column containing the observed quantity (e.g., loss ratio, pure premium).

        Returns
        -------
        BuhlmannModel
            The fitted model instance (for method chaining).

        Raises
        ------
        NotImplementedError
            This is a scaffold — estimation logic is not yet implemented.
        """
        raise NotImplementedError(
            "BuhlmannModel.fit() is not yet implemented. "
            "This scaffold defines the interface; estimation logic is forthcoming."
        )

    def credibility_factor(self) -> pd.Series:
        """Return the credibility factor Z for each risk group.

        Returns
        -------
        pd.Series
            Credibility factor indexed by group, with values in [0, 1].

        Raises
        ------
        NotImplementedError
            This is a scaffold.
        """
        raise NotImplementedError(
            "BuhlmannModel.credibility_factor() is not yet implemented."
        )

    def credibility_premium(self) -> pd.Series:
        """Return the credibility-weighted premium estimate for each risk group.

        The credibility premium blends the group's own experience with the
        collective mean:

            P_cred_i = Z_i * X_bar_i + (1 - Z_i) * mu

        Returns
        -------
        pd.Series
            Credibility premium indexed by group.

        Raises
        ------
        NotImplementedError
            This is a scaffold.
        """
        raise NotImplementedError(
            "BuhlmannModel.credibility_premium() is not yet implemented."
        )


class BuhlmannStraubModel:
    """Bühlmann-Straub credibility model with exposure (volume) weights.

    An extension of the basic Bühlmann model that accounts for varying
    exposure or volume across observation periods. Each observation is
    weighted by an exposure measure (e.g., earned premium, number of
    policies, payroll), so that periods with more information contribute
    proportionally more to the credibility estimate.

    The credibility factor becomes:

        Z_i = w_i / (w_i + v/a) = w_i / (w_i + k)

    where w_i is the total weight (exposure) for risk group i.

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate structural parameters.

    Examples
    --------
    >>> model = BuhlmannStraubModel()
    >>> model.fit(
    ...     data,
    ...     group_col="risk_class",
    ...     observation_col="loss_ratio",
    ...     weight_col="earned_premium",
    ... )
    >>> model.credibility_factor()
    >>> model.credibility_premium()

    References
    ----------
    Bühlmann, H. & Straub, E. (1970). "Glaubwürdigkeit für Schadensätze."
    Mitteilungen der Vereinigung Schweizerischer Versicherungsmathematiker.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._data: pd.DataFrame | None = None
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._weight_col: str | None = None
        self._mu: float | None = None
        self._v: float | None = None
        self._a: float | None = None

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        weight_col: str,
    ) -> BuhlmannStraubModel:
        """Estimate structural parameters from exposure-weighted data.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with one row per (group, period) observation.
        group_col : str
            Column identifying the risk group.
        observation_col : str
            Column containing the observed quantity (e.g., loss ratio).
        weight_col : str
            Column containing the exposure/volume weight (e.g., earned premium).

        Returns
        -------
        BuhlmannStraubModel
            The fitted model instance.

        Raises
        ------
        NotImplementedError
            This is a scaffold.
        """
        raise NotImplementedError(
            "BuhlmannStraubModel.fit() is not yet implemented. "
            "This scaffold defines the interface; estimation logic is forthcoming."
        )

    def credibility_factor(self) -> pd.Series:
        """Return the credibility factor Z for each risk group.

        Returns
        -------
        pd.Series
            Credibility factor indexed by group, with values in [0, 1].
        """
        raise NotImplementedError(
            "BuhlmannStraubModel.credibility_factor() is not yet implemented."
        )

    def credibility_premium(self) -> pd.Series:
        """Return the credibility-weighted premium estimate for each group.

        Returns
        -------
        pd.Series
            Credibility premium indexed by group.
        """
        raise NotImplementedError(
            "BuhlmannStraubModel.credibility_premium() is not yet implemented."
        )


class JewellHierarchical:
    """Jewell multi-level hierarchical credibility model.

    Extends Bühlmann-Straub credibility to nested hierarchical structures,
    such as company > region > territory or line of business > sub-line > state.
    Each level in the hierarchy introduces its own between-variance component,
    and credibility factors are computed at every level.

    This is essential for large insurance portfolios where risk classification
    is naturally hierarchical and credibility should flow between related
    groups at different levels of aggregation.

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate structural parameters.

    Examples
    --------
    >>> model = JewellHierarchical()
    >>> model.fit(
    ...     data,
    ...     hierarchy_cols=["region", "territory", "risk_class"],
    ...     observation_col="loss_ratio",
    ...     weight_col="earned_premium",
    ... )
    >>> model.credibility_factor(level="territory")
    >>> model.credibility_premium()

    References
    ----------
    Jewell, W.S. (1975). "The Use of Collateral Data in Credibility Theory:
    A Hierarchical Model." Giornale dell'Istituto Italiano degli Attuari, 38.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._data: pd.DataFrame | None = None
        self._hierarchy_cols: list[str] | None = None
        self._observation_col: str | None = None
        self._weight_col: str | None = None

    def fit(
        self,
        data: pd.DataFrame,
        hierarchy_cols: list[str],
        observation_col: str,
        weight_col: str | None = None,
    ) -> JewellHierarchical:
        """Estimate hierarchical structural parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with columns for each hierarchy level.
        hierarchy_cols : list[str]
            Columns defining the hierarchy from coarsest to finest level.
            E.g., ``["region", "territory", "risk_class"]``.
        observation_col : str
            Column containing the observed quantity.
        weight_col : str, optional
            Column containing the exposure weight. If None, equal weights.

        Returns
        -------
        JewellHierarchical
            The fitted model instance.
        """
        raise NotImplementedError(
            "JewellHierarchical.fit() is not yet implemented. "
            "Multi-level variance decomposition logic is forthcoming."
        )

    def credibility_factor(self, level: str | None = None) -> pd.Series:
        """Return credibility factors, optionally at a specific hierarchy level.

        Parameters
        ----------
        level : str, optional
            Hierarchy level to report. If None, returns finest-level factors.

        Returns
        -------
        pd.Series
            Credibility factors indexed by the groups at the requested level.
        """
        raise NotImplementedError(
            "JewellHierarchical.credibility_factor() is not yet implemented."
        )

    def credibility_premium(self) -> pd.DataFrame:
        """Return credibility premiums at the finest hierarchy level.

        Returns
        -------
        pd.DataFrame
            Credibility premiums with hierarchy columns as index.
        """
        raise NotImplementedError(
            "JewellHierarchical.credibility_premium() is not yet implemented."
        )


class HachemeisterRegression:
    """Hachemeister regression credibility model.

    Extends Bühlmann-Straub credibility by incorporating regression
    structure (covariates) into the credibility estimation. This is
    particularly useful when trends or other systematic effects must
    be modeled alongside the credibility weighting.

    The most common application is **trend credibility**: each risk
    group's loss experience is fit with a linear trend, and the
    credibility-weighted estimate blends individual and collective
    trend coefficients.

    The model estimates:

        Y_i = X_i * beta_i + epsilon_i

    where beta_i is the vector of regression coefficients for group i,
    and the credibility estimate of beta_i is:

        beta_cred_i = Z_i * beta_hat_i + (I - Z_i) * beta_hat_collective

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate parameters.

    Examples
    --------
    >>> model = HachemeisterRegression()
    >>> model.fit(
    ...     data,
    ...     group_col="state",
    ...     observation_col="avg_claim_cost",
    ...     time_col="year",
    ...     weight_col="claim_count",
    ... )
    >>> model.credibility_factor()
    >>> model.credibility_premium()

    References
    ----------
    Hachemeister, C.A. (1975). "Credibility for Regression Models with
    Application to Trend." In Credibility: Theory and Applications.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._data: pd.DataFrame | None = None
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._time_col: str | None = None
        self._weight_col: str | None = None
        self._covariates: list[str] | None = None

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        time_col: str | None = None,
        weight_col: str | None = None,
        covariates: list[str] | None = None,
    ) -> HachemeisterRegression:
        """Fit the regression credibility model.

        If ``time_col`` is provided and ``covariates`` is None, a simple
        linear trend model is used (intercept + time). Otherwise, the
        user-supplied covariates define the design matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with one row per (group, period) observation.
        group_col : str
            Column identifying the risk group.
        observation_col : str
            Column containing the observed quantity.
        time_col : str, optional
            Column containing the time index (for trend models).
        weight_col : str, optional
            Column containing the exposure weight.
        covariates : list[str], optional
            Columns to use as regression covariates. Overrides time_col if
            both are provided.

        Returns
        -------
        HachemeisterRegression
            The fitted model instance.
        """
        raise NotImplementedError(
            "HachemeisterRegression.fit() is not yet implemented. "
            "Regression credibility estimation logic is forthcoming."
        )

    def credibility_factor(self) -> pd.DataFrame:
        """Return the credibility matrix Z for each risk group.

        In regression credibility, Z is a matrix (not a scalar) that blends
        individual and collective regression coefficients.

        Returns
        -------
        pd.DataFrame
            Credibility matrices indexed by group.
        """
        raise NotImplementedError(
            "HachemeisterRegression.credibility_factor() is not yet implemented."
        )

    def credibility_premium(self) -> pd.Series:
        """Return credibility-weighted predicted values.

        Returns
        -------
        pd.Series
            Credibility premium estimates indexed by group.
        """
        raise NotImplementedError(
            "HachemeisterRegression.credibility_premium() is not yet implemented."
        )


class LimitedFluctuationCredibility:
    """Classical limited-fluctuation (full and partial) credibility.

    The oldest and simplest form of credibility. Full credibility is
    granted when the volume of experience is large enough that the
    observed statistic is unlikely to deviate from the true value by
    more than a specified percentage, with a given probability.

    The classical standard for full credibility of claim frequency is
    **1,082 claims** (for k=0.05 tolerance, P=0.90 probability, Poisson).

    Partial credibility is then:

        Z = sqrt(n / n_full)  capped at 1.0

    This class supports:
    - Full credibility standards for frequency and severity
    - Partial credibility calculation
    - Custom tolerance (k) and probability (P) parameters
    - Adjustment for non-Poisson frequency distributions

    Parameters
    ----------
    k : float, default 0.05
        Maximum tolerable deviation from the true value (5% = 0.05).
    p : float, default 0.90
        Required probability that the observed value is within k of the
        true value.

    Examples
    --------
    >>> cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    >>> cred.full_credibility_standard()
    1082
    >>> cred.credibility_factor(n_claims=500)
    0.6797...
    >>> cred.credibility_premium(observed=0.72, prior=0.65, n_claims=500)
    0.6976...
    """

    def __init__(self, k: float = 0.05, p: float = 0.90) -> None:
        self.k = k
        self.p = p

    def full_credibility_standard(
        self,
        cv: float = 1.0,
    ) -> int:
        """Compute the number of claims required for full credibility.

        For Poisson frequency (CV=1), the classical standard is:

            n_full = (z_{(1+P)/2} / k)^2 * CV^2

        With k=0.05 and P=0.90: n_full = (1.645 / 0.05)^2 = 1082.

        Parameters
        ----------
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.
            CV=1.0 corresponds to the Poisson assumption.

        Returns
        -------
        int
            Number of claims required for full credibility.
        """
        raise NotImplementedError(
            "LimitedFluctuationCredibility.full_credibility_standard() "
            "is not yet implemented."
        )

    def credibility_factor(self, n_claims: int, cv: float = 1.0) -> float:
        """Compute the partial credibility factor.

        Z = min(1, sqrt(n / n_full))

        Parameters
        ----------
        n_claims : int
            Observed number of claims.
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.

        Returns
        -------
        float
            Credibility factor in [0, 1].
        """
        raise NotImplementedError(
            "LimitedFluctuationCredibility.credibility_factor() is not yet implemented."
        )

    def credibility_premium(
        self,
        observed: float,
        prior: float,
        n_claims: int,
        cv: float = 1.0,
    ) -> float:
        """Compute the credibility-weighted premium.

        P_cred = Z * observed + (1 - Z) * prior

        Parameters
        ----------
        observed : float
            Observed loss ratio, pure premium, or other target statistic.
        prior : float
            Prior (manual rate, industry average, or collective mean).
        n_claims : int
            Observed number of claims.
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.

        Returns
        -------
        float
            The credibility-weighted premium estimate.
        """
        raise NotImplementedError(
            "LimitedFluctuationCredibility.credibility_premium() is not yet implemented."
        )


class BayesianCredibility:
    """Bayesian credibility estimation via PyMC (MCMC bridge).

    Provides a modern Bayesian alternative to the classical non-parametric
    credibility estimators. Instead of estimating structural parameters via
    method-of-moments (as in Bühlmann), this class formulates the credibility
    model as a hierarchical Bayesian model and estimates posterior distributions
    via Markov Chain Monte Carlo (MCMC) sampling.

    Advantages over classical approaches:

    - Full posterior distributions, not just point estimates
    - Natural handling of small-sample uncertainty
    - Flexible prior specification for domain knowledge
    - Model comparison via WAIC, LOO-CV, etc.

    Requires the optional ``pymc`` dependency:

        pip install actuarcredibility[bayesian]

    Parameters
    ----------
    prior_config : dict, optional
        Configuration for prior distributions. Keys depend on the model
        type. If None, weakly informative defaults are used.

    Examples
    --------
    >>> model = BayesianCredibility()
    >>> model.fit(
    ...     data,
    ...     group_col="risk_class",
    ...     observation_col="loss_ratio",
    ...     weight_col="earned_premium",
    ...     samples=2000,
    ... )
    >>> model.credibility_factor()  # posterior mean of Z
    >>> model.posterior_summary()   # full posterior summaries

    References
    ----------
    Bühlmann, H. & Gisler, A. (2005). A Course in Credibility Theory. Springer.
    (Chapter on Bayesian interpretation of credibility.)
    """

    def __init__(self, prior_config: dict[str, Any] | None = None) -> None:
        self._prior_config = prior_config or {}
        self._fitted: bool = False
        self._trace: Any = None

    @staticmethod
    def _check_pymc() -> None:
        """Verify that PyMC is installed."""
        try:
            import pymc  # noqa: F401
        except ImportError:
            raise ImportError(
                "BayesianCredibility requires PyMC. "
                "Install it with: pip install actuarcredibility[bayesian]"
            )

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        weight_col: str | None = None,
        samples: int = 2000,
        chains: int = 4,
        random_seed: int | None = None,
    ) -> BayesianCredibility:
        """Fit the Bayesian credibility model via MCMC.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with one row per (group, period) observation.
        group_col : str
            Column identifying the risk group.
        observation_col : str
            Column containing the observed quantity.
        weight_col : str, optional
            Column containing the exposure weight.
        samples : int, default 2000
            Number of MCMC samples per chain (after tuning).
        chains : int, default 4
            Number of MCMC chains.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        BayesianCredibility
            The fitted model instance.
        """
        self._check_pymc()
        raise NotImplementedError(
            "BayesianCredibility.fit() is not yet implemented. "
            "PyMC model construction and sampling logic is forthcoming."
        )

    def credibility_factor(self) -> pd.Series:
        """Return posterior mean credibility factors.

        Returns
        -------
        pd.Series
            Posterior mean of the credibility factor for each group.
        """
        raise NotImplementedError(
            "BayesianCredibility.credibility_factor() is not yet implemented."
        )

    def credibility_premium(self) -> pd.Series:
        """Return posterior mean credibility premiums.

        Returns
        -------
        pd.Series
            Posterior mean credibility premium for each group.
        """
        raise NotImplementedError(
            "BayesianCredibility.credibility_premium() is not yet implemented."
        )

    def posterior_summary(self) -> pd.DataFrame:
        """Return a summary of posterior distributions for all parameters.

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, sd, HDI) for each parameter.
        """
        raise NotImplementedError(
            "BayesianCredibility.posterior_summary() is not yet implemented."
        )
