"""Diagnostics for credibility models.

Includes:

- :func:`variance_decomposition`: splits total variance into within-risk
  (process) and between-risk (structural) components.
- :func:`credibility_curve` returns the credibility factor as a function of
  weight, given fitted ``v`` and ``a``.
- :func:`compare_models`. Side-by-side comparison of credibility-weighted
  premiums and factors across fitted models.
- :func:`shrinkage_summary` (distance from group mean to grand mean before
  vs. after credibility weighting; a "how much did credibility move me?"
  diagnostic).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from actuarcredibility.buhlmann import BuhlmannStraubModel


def variance_decomposition(model: BuhlmannStraubModel) -> pd.Series:
    """Return the variance decomposition of a fitted Bühlmann-Straub model.

    Provides ``v`` (process variance, within-risk), ``a`` (variance of
    hypothetical means, between-risk), the implied total variance ``v + a``,
    and the share of variance attributable to risk heterogeneity
    (``a / (v + a)``). High shares mean credibility weights toward the
    individual; low shares mean credibility weights toward the collective.
    """
    params = model.structural_parameters
    v = params["v"]
    a = params["a"]
    total = v + a
    share = a / total if total > 0 else 0.0
    return pd.Series(
        {
            "v_within": v,
            "a_between": a,
            "total": total,
            "between_share": share,
            "k_v_over_a": params["k"],
        },
        name="variance_decomposition",
    )


def credibility_curve(
    model: BuhlmannStraubModel,
    weights: np.ndarray | list[float] | None = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Tabulate the credibility factor across a range of weights.

    Useful for sense-checking the credibility curve implied by the fitted
    structural parameters and identifying the weight at which a target
    credibility level (e.g., 0.50, 0.90) is reached.

    Parameters
    ----------
    model : BuhlmannStraubModel
        Fitted model.
    weights : array-like, optional
        Specific weights to evaluate. If ``None``, a logarithmic grid spanning
        ``[w_min/10, w_max*10]`` of the fitted groups is generated.
    n_points : int, default 100
        Number of grid points when ``weights`` is None.

    Returns
    -------
    pd.DataFrame
        Columns ``weight`` and ``Z``.
    """
    params = model.structural_parameters
    a = params["a"]
    v = params["v"]
    if weights is None:
        fitted = model.summary()["weight"].to_numpy()
        lo = max(fitted.min() / 10.0, 1e-6)
        hi = fitted.max() * 10.0
        weights = np.logspace(np.log10(lo), np.log10(hi), n_points)
    w = np.asarray(weights, dtype=float)
    if a <= 0:
        z = np.zeros_like(w)
    else:
        z = w / (w + v / a)
    return pd.DataFrame({"weight": w, "Z": z})


def shrinkage_summary(model: BuhlmannStraubModel) -> pd.DataFrame:
    """Per-group raw vs. credibility-weighted estimate, with shrinkage ratio.

    Shrinkage ratio = ``|P_cred - mu| / |X_bar - mu|`` ∈ ``[0, 1]``. A
    ratio close to 1 means almost no shrinkage; close to 0 means the
    credibility estimate has been pulled fully to the collective mean.
    """
    summary = model.summary().copy()
    mu = model.structural_parameters["mu"]
    summary["mu"] = mu
    summary["raw_distance"] = (summary["mean"] - mu).abs()
    summary["cred_distance"] = (summary["P_cred"] - mu).abs()
    summary["shrinkage_ratio"] = np.where(
        summary["raw_distance"] > 0,
        summary["cred_distance"] / summary["raw_distance"],
        np.nan,
    )
    return summary


def compare_models(
    models: dict[str, Any],
    label_factor: str = "Z",
    label_premium: str = "P_cred",
) -> pd.DataFrame:
    """Side-by-side credibility factors and premiums across fitted models.

    Each value in ``models`` must implement ``credibility_factor()`` and
    ``credibility_premium()`` returning ``pd.Series`` over a common group
    index (the function aligns by index).
    """
    factor_frames: list[pd.Series] = []
    prem_frames: list[pd.Series] = []
    for name, m in models.items():
        z = cast(pd.Series, m.credibility_factor())
        # If z is multi-dimensional (e.g., regression credibility matrix as
        # DataFrame), skip it for the factor comparison.
        if isinstance(z, pd.Series):
            z = z.rename(f"{name}::{label_factor}")
            factor_frames.append(z)
        p = m.credibility_premium()
        if isinstance(p, pd.Series):
            p = p.rename(f"{name}::{label_premium}")
            prem_frames.append(p)
    pieces = [*factor_frames, *prem_frames]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)
