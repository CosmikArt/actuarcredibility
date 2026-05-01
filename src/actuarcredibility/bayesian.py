"""Bayesian credibility via PyMC.

Hierarchical normal model:

    mu     ~ Normal(prior_mu, prior_mu_sd)
    sigma_a ~ HalfNormal(prior_sigma_a)
    sigma_v ~ HalfNormal(prior_sigma_v)
    theta_i ~ Normal(mu, sigma_a)
    Y_{it}  ~ Normal(theta_i, sigma_v / sqrt(w_{it}))

This corresponds to the Bayesian interpretation of Bühlmann-Straub
credibility (Bühlmann & Gisler, 2005, Ch. 2). Posterior summaries of
``theta_i`` give the credibility premium; the credibility factor is
recovered as ``Z_i = a / (a + v / w_{i.})`` averaged over the posterior.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from actuarcredibility._validation import (
    require_columns,
    require_finite,
    require_fitted,
    require_positive_weights,
)


class BayesianCredibility:
    """Hierarchical Bayesian credibility model.

    Requires the optional ``pymc`` dependency::

        pip install actuarcredibility[bayesian]

    Parameters
    ----------
    prior_config : dict, optional
        Optional overrides for the priors. Recognized keys:

        - ``mu_prior_mean`` (default: data mean)
        - ``mu_prior_sd`` (default: 10 * data sd or 1.0)
        - ``sigma_a_prior`` (default: data sd or 1.0)
        - ``sigma_v_prior`` (default: data sd or 1.0)

    Examples
    --------
    >>> # doctest: +SKIP
    >>> model = BayesianCredibility().fit(
    ...     data, group_col="risk", observation_col="y", weight_col="w",
    ...     samples=500, chains=2, random_seed=0,
    ... )
    >>> model.credibility_premium()
    """

    def __init__(self, prior_config: dict[str, Any] | None = None) -> None:
        self._prior_config: dict[str, Any] = dict(prior_config or {})
        self._fitted: bool = False
        self._trace: Any = None
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._weight_col: str | None = None
        self._groups: np.ndarray | None = None
        self._group_weights: np.ndarray | None = None
        self._group_means: np.ndarray | None = None

    @staticmethod
    def _check_pymc() -> Any:
        try:
            import pymc  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "BayesianCredibility requires PyMC. "
                "Install it with: pip install actuarcredibility[bayesian]"
            ) from exc
        return pymc

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        weight_col: str | None = None,
        samples: int = 2000,
        chains: int = 4,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        progressbar: bool = False,
    ) -> BayesianCredibility:
        pm = self._check_pymc()

        require_columns(data, [group_col, observation_col])
        if weight_col is not None:
            require_columns(data, [weight_col])

        df = data.copy()
        if weight_col is None:
            df["_w"] = 1.0
            weight_col_eff = "_w"
        else:
            weight_col_eff = weight_col

        obs = df[observation_col].to_numpy(dtype=float)
        weights = df[weight_col_eff].to_numpy(dtype=float)
        require_finite(obs, observation_col)
        require_finite(weights, weight_col_eff)
        require_positive_weights(weights, weight_col_eff)
        groups = df[group_col].to_numpy()
        unique_groups, group_idx = np.unique(groups, return_inverse=True)

        # Per-group weighted aggregates (used later for posterior Z).
        r = unique_groups.size
        group_weights = np.bincount(group_idx, weights=weights, minlength=r)
        group_obs_weighted = np.bincount(
            group_idx, weights=weights * obs, minlength=r
        )
        group_means = group_obs_weighted / group_weights

        # Sensible default priors.
        data_mean = float(np.average(obs, weights=weights))
        data_sd = float(np.sqrt(np.average((obs - data_mean) ** 2, weights=weights)))
        if data_sd == 0:
            data_sd = 1.0
        cfg = {
            "mu_prior_mean": data_mean,
            "mu_prior_sd": 10.0 * data_sd,
            "sigma_a_prior": data_sd,
            "sigma_v_prior": data_sd,
            **self._prior_config,
        }

        with pm.Model():
            mu = pm.Normal("mu", mu=cfg["mu_prior_mean"], sigma=cfg["mu_prior_sd"])
            sigma_a = pm.HalfNormal("sigma_a", sigma=cfg["sigma_a_prior"])
            sigma_v = pm.HalfNormal("sigma_v", sigma=cfg["sigma_v_prior"])
            theta = pm.Normal("theta", mu=mu, sigma=sigma_a, shape=r)
            sd_obs = sigma_v / np.sqrt(weights)
            pm.Normal(
                "y",
                mu=theta[group_idx],
                sigma=sd_obs,
                observed=obs,
            )
            trace = pm.sample(
                draws=samples,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
            )

        self._trace = trace
        self._fitted = True
        self._group_col = group_col
        self._observation_col = observation_col
        self._weight_col = weight_col
        self._groups = unique_groups
        self._group_weights = group_weights
        self._group_means = group_means
        return self

    def _posterior_samples(self, name: str) -> np.ndarray:
        trace = self._trace
        # Compatible with PyMC 5 ArviZ InferenceData.
        return np.asarray(trace.posterior[name].values)

    def credibility_premium(self) -> pd.Series:
        """Posterior-mean credibility premium for each group."""
        require_fitted(self._fitted, type(self).__name__)
        theta = self._posterior_samples("theta")  # (chains, draws, r)
        post_mean = theta.reshape(-1, theta.shape[-1]).mean(axis=0)
        return pd.Series(
            post_mean,
            index=pd.Index(self._groups, name=self._group_col),
            name="P_cred",
        )

    def credibility_factor(self) -> pd.Series:
        """Posterior-mean credibility factor ``Z_i = a / (a + v / w_i)``."""
        require_fitted(self._fitted, type(self).__name__)
        sigma_a = self._posterior_samples("sigma_a")  # (chains, draws)
        sigma_v = self._posterior_samples("sigma_v")
        a = sigma_a**2
        v = sigma_v**2
        w = cast(np.ndarray, self._group_weights)
        a_flat = a.reshape(-1)
        v_flat = v.reshape(-1)
        z = a_flat[:, None] / (a_flat[:, None] + v_flat[:, None] / w[None, :])
        z_mean = z.mean(axis=0)
        return pd.Series(
            z_mean,
            index=pd.Index(self._groups, name=self._group_col),
            name="Z",
        )

    def posterior_summary(self) -> pd.DataFrame:
        """ArviZ-style summary of the posterior (mean, sd, HDI 94%, r_hat)."""
        require_fitted(self._fitted, type(self).__name__)
        try:
            import arviz as az  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "posterior_summary requires arviz (installed automatically with PyMC)."
            ) from exc
        return az.summary(self._trace)

    @property
    def trace(self) -> Any:
        """Underlying ArviZ InferenceData object."""
        require_fitted(self._fitted, type(self).__name__)
        return self._trace
