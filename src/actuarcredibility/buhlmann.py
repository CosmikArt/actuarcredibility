"""Bühlmann and Bühlmann-Straub non-parametric credibility models.

Estimators follow Bühlmann & Gisler (2005), *A Course in Credibility Theory
and its Applications*, Springer.

Notation
--------
- ``r`` risk groups, indexed ``i``.
- ``n_i`` observation periods for group ``i``, indexed ``t``.
- ``X_{it}``: observed quantity (loss ratio, pure premium, frequency).
- ``w_{it}``: exposure weight (earned premium, policy count, payroll).
- ``v = E[Var(X | Theta)]``: expected process variance.
- ``a = Var(E[X | Theta])``: variance of the hypothetical means.

The credibility factor is ``Z_i = w_{i.} / (w_{i.} + v / a)``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from actuarcredibility._validation import (
    require_columns,
    require_finite,
    require_fitted,
    require_positive_weights,
)


def _bs_structural_estimates(
    groups: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    np.ndarray,  # group labels
    np.ndarray,  # group weighted means X_i.
    np.ndarray,  # group total weights w_i.
    float,       # collective weighted mean X..
    float,       # v_hat
    float,       # a_hat (>= 0, clipped)
    float,       # a_hat_raw (may be negative)
]:
    """Bühlmann-Gisler unbiased moment estimators of v and a.

    Implements equations 4.27–4.32 of Bühlmann & Gisler (2005).
    """
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    r = unique_groups.size
    if r < 2:
        raise ValueError(
            f"At least 2 risk groups are required to estimate credibility; "
            f"found {r}."
        )

    w_sum_per_group = np.bincount(inverse, weights=weights, minlength=r)
    wx_sum_per_group = np.bincount(inverse, weights=weights * obs, minlength=r)
    if np.any(w_sum_per_group <= 0):  # pragma: no cover - guarded by require_positive_weights
        raise ValueError("Every group must have a strictly positive total weight.")
    group_means = wx_sum_per_group / w_sum_per_group
    grand_total_weight = float(w_sum_per_group.sum())
    grand_mean = float((w_sum_per_group * group_means).sum() / grand_total_weight)

    # Within-group sum of squared deviations:
    #   SS_i = sum_t w_{it} (X_{it} - X_{i.})^2
    centered_sq = weights * (obs - group_means[inverse]) ** 2
    ss_per_group = np.bincount(inverse, weights=centered_sq, minlength=r)
    n_per_group = np.bincount(inverse, minlength=r)

    # v_hat: mean across groups of unbiased within-group variance estimates.
    # For groups with n_i = 1 we cannot estimate within-group variance — drop.
    multi_obs_mask = n_per_group > 1
    if not np.any(multi_obs_mask):
        raise ValueError(
            "At least one group must have more than one observation period to "
            "estimate within-group variance v."
        )
    sigma2_per_group = np.zeros(r)
    sigma2_per_group[multi_obs_mask] = (
        ss_per_group[multi_obs_mask] / (n_per_group[multi_obs_mask] - 1)
    )
    v_hat = float(sigma2_per_group[multi_obs_mask].mean())

    # a_hat: between-group variance, Bühlmann-Gisler unbiased estimator
    # (equation 4.31).
    between_ss = float((w_sum_per_group * (group_means - grand_mean) ** 2).sum())
    denom = grand_total_weight - float((w_sum_per_group**2).sum()) / grand_total_weight
    # denom > 0 follows from Cauchy-Schwarz whenever r >= 2 with positive
    # weights, both of which are enforced above.
    if denom <= 0:  # pragma: no cover - mathematically unreachable under our guards
        raise ValueError(
            "Degenerate weight configuration: cannot compute between-group "
            "variance estimate (denominator non-positive)."
        )
    a_hat_raw = (between_ss - (r - 1) * v_hat) / denom
    a_hat = max(a_hat_raw, 0.0)

    return (
        unique_groups,
        group_means,
        w_sum_per_group,
        grand_mean,
        v_hat,
        a_hat,
        a_hat_raw,
    )


class BuhlmannStraubModel:
    """Bühlmann-Straub credibility model with exposure weights.

    Each observation ``X_{it}`` carries a weight ``w_{it}`` (e.g., earned
    premium, policy count). Structural parameters ``v`` and ``a`` are
    estimated by the Bühlmann-Gisler unbiased moment estimators. The
    credibility factor for group ``i`` is

        Z_i = w_{i.} / (w_{i.} + v / a)

    and the credibility premium is

        P_cred_i = Z_i * X_{i.} + (1 - Z_i) * mu_hat,

    where ``mu_hat`` is the credibility-weighted collective mean

        mu_hat = sum_i Z_i * X_{i.} / sum_i Z_i.

    When the unbiased estimator of ``a`` is negative (a "no signal" situation),
    it is clipped to zero. In that case ``Z_i = 0`` for every group and the
    credibility premium collapses to the grand weighted mean.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "g": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    ...     "y": [0.62, 0.58, 0.65, 0.81, 0.77, 0.84, 0.45, 0.52, 0.48],
    ...     "w": [5.0, 5.2, 5.5, 2.0, 2.1, 2.3, 8.0, 8.5, 9.0],
    ... })
    >>> m = BuhlmannStraubModel().fit(data, group_col="g",
    ...                                observation_col="y", weight_col="w")
    >>> z = m.credibility_factor()
    >>> bool((z > 0).all() and (z <= 1).all())
    True
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._weight_col: str | None = None
        self._groups: np.ndarray | None = None
        self._group_means: np.ndarray | None = None
        self._group_weights: np.ndarray | None = None
        self._mu: float | None = None
        self._v: float | None = None
        self._a: float | None = None
        self._a_raw: float | None = None
        self._k: float | None = None  # v / a (np.inf when a == 0)

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        weight_col: str,
    ) -> BuhlmannStraubModel:
        require_columns(data, [group_col, observation_col, weight_col])
        obs = data[observation_col].to_numpy(dtype=float)
        weights = data[weight_col].to_numpy(dtype=float)
        require_finite(obs, observation_col)
        require_finite(weights, weight_col)
        require_positive_weights(weights, weight_col)
        groups = data[group_col].to_numpy()

        (
            unique_groups,
            group_means,
            group_weights,
            grand_mean,
            v_hat,
            a_hat,
            a_hat_raw,
        ) = _bs_structural_estimates(groups, obs, weights)

        self._group_col = group_col
        self._observation_col = observation_col
        self._weight_col = weight_col
        self._groups = unique_groups
        self._group_means = group_means
        self._group_weights = group_weights
        self._v = v_hat
        self._a = a_hat
        self._a_raw = a_hat_raw
        self._k = (v_hat / a_hat) if a_hat > 0 else float("inf")

        # Credibility-weighted collective mean (per Bühlmann-Gisler).
        z = self._compute_z(group_weights)
        z_sum = float(z.sum())
        if z_sum > 0:
            self._mu = float((z * group_means).sum() / z_sum)
        else:
            # a == 0: no signal, fall back to grand weighted mean.
            self._mu = grand_mean

        self._fitted = True
        return self

    def _compute_z(self, w: np.ndarray) -> np.ndarray:
        if self._a is None or self._k is None:  # pragma: no cover - internal guard
            raise RuntimeError("internal error: model parameters not initialized")
        if self._a == 0.0:
            return np.zeros_like(w)
        return w / (w + self._k)

    def credibility_factor(self) -> pd.Series:
        require_fitted(self._fitted, type(self).__name__)
        z = self._compute_z(cast(np.ndarray, self._group_weights))
        return pd.Series(z, index=pd.Index(self._groups, name=self._group_col), name="Z")

    def credibility_premium(self) -> pd.Series:
        require_fitted(self._fitted, type(self).__name__)
        z = self._compute_z(cast(np.ndarray, self._group_weights))
        means = cast(np.ndarray, self._group_means)
        prem = z * means + (1.0 - z) * cast(float, self._mu)
        return pd.Series(
            prem,
            index=pd.Index(self._groups, name=self._group_col),
            name="P_cred",
        )

    def predict(self, group: object) -> float:
        """Credibility premium for a single (already fitted) group."""
        require_fitted(self._fitted, type(self).__name__)
        groups = cast(np.ndarray, self._groups)
        idx = np.where(groups == group)[0]
        if idx.size == 0:
            raise KeyError(f"Group {group!r} was not present in the fitted data.")
        i = int(idx[0])
        z_i = self._compute_z(cast(np.ndarray, self._group_weights))[i]
        means = cast(np.ndarray, self._group_means)
        return float(z_i * means[i] + (1.0 - z_i) * cast(float, self._mu))

    @property
    def structural_parameters(self) -> dict[str, float]:
        """Estimated structural parameters: ``mu``, ``v``, ``a``, ``k = v/a``.

        Also includes ``a_raw`` — the unbiased estimator before clipping at 0.
        Negative ``a_raw`` indicates the data are consistent with a single
        underlying risk (no heterogeneity signal).
        """
        require_fitted(self._fitted, type(self).__name__)
        return {
            "mu": cast(float, self._mu),
            "v": cast(float, self._v),
            "a": cast(float, self._a),
            "a_raw": cast(float, self._a_raw),
            "k": cast(float, self._k),
        }

    def summary(self) -> pd.DataFrame:
        """Per-group summary: weight, mean, Z, credibility premium."""
        require_fitted(self._fitted, type(self).__name__)
        z = self.credibility_factor().to_numpy()
        return pd.DataFrame(
            {
                "weight": self._group_weights,
                "mean": self._group_means,
                "Z": z,
                "P_cred": z * cast(np.ndarray, self._group_means)
                + (1.0 - z) * cast(float, self._mu),
            },
            index=pd.Index(self._groups, name=self._group_col),
        )


class BuhlmannModel(BuhlmannStraubModel):
    """Non-parametric Bühlmann credibility (equal weights).

    Special case of :class:`BuhlmannStraubModel` with all weights equal to 1.
    The credibility factor reduces to ``Z = n_i / (n_i + v / a)`` where
    ``n_i`` is the number of observation periods for group ``i``.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "g": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    ...     "y": [0.62, 0.58, 0.65, 0.81, 0.77, 0.84, 0.45, 0.52, 0.48],
    ... })
    >>> m = BuhlmannModel().fit(data, group_col="g", observation_col="y")
    >>> bool((m.credibility_factor() > 0).all())
    True
    """

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
    ) -> BuhlmannModel:
        require_columns(data, [group_col, observation_col])
        data = data.assign(_unit_weight=1.0)
        super().fit(
            data,
            group_col=group_col,
            observation_col=observation_col,
            weight_col="_unit_weight",
        )
        self._weight_col = None
        return self
