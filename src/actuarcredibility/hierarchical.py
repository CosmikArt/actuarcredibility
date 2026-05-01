"""Jewell hierarchical credibility — multi-level extension of Bühlmann-Straub.

Implements the iterative variance-decomposition approach of
Jewell (1975) and Bühlmann-Gisler (2005, Chapter 6). For each level of
the hierarchy, a Bühlmann-Straub problem is solved on the
credibility-weighted residuals from the level below.

References
----------
- Jewell, W.S. (1975). "The Use of Collateral Data in Credibility Theory:
  A Hierarchical Model." *Giornale dell'Istituto Italiano degli Attuari* 38.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory*. Springer.
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
from actuarcredibility.buhlmann import _bs_structural_estimates


class JewellHierarchical:
    """Multi-level hierarchical credibility model (Jewell, 1975).

    Supports an arbitrary number of nesting levels listed from coarsest to
    finest, e.g. ``["region", "territory", "risk_class"]``. The model fits a
    Bühlmann-Straub problem at every level by iterating bottom-up: the
    finest-level groups are aggregated into their parents, the resulting
    parent-level credibility-weighted means become observations at the next
    level, and so on.

    The credibility premium at the finest level recombines top-down:

        P[level_k] = Z[level_k] * X[level_k] + (1 - Z[level_k]) * P[level_{k-1}]

    starting from the grand mean at the top.

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate the model.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "region":   ["N","N","N","N","S","S","S","S"],
    ...     "risk":     ["A","A","B","B","C","C","D","D"],
    ...     "year":     [2022, 2023]*4,
    ...     "y":        [0.6, 0.62, 0.55, 0.58, 0.80, 0.78, 0.82, 0.84],
    ...     "w":        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ... })
    >>> m = JewellHierarchical().fit(
    ...     data, hierarchy_cols=["region", "risk"],
    ...     observation_col="y", weight_col="w",
    ... )
    >>> "risk" in m.credibility_factor().index.names
    True
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._hierarchy_cols: list[str] | None = None
        self._observation_col: str | None = None
        self._weight_col: str | None = None
        self._levels: list[dict] = []  # one dict per level, fine -> coarse
        self._mu: float | None = None
        self._leaf_premiums: pd.Series | None = None

    def fit(
        self,
        data: pd.DataFrame,
        hierarchy_cols: list[str],
        observation_col: str,
        weight_col: str | None = None,
    ) -> JewellHierarchical:
        if not hierarchy_cols:
            raise ValueError("hierarchy_cols must contain at least one column.")
        require_columns(data, [*hierarchy_cols, observation_col])
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

        # Bottom-up iteration. At each level, the "groups" are the columns
        # from the top down to the current level; the "subgroups" used to
        # compute the credibility step are level-current itself.
        levels_info: list[dict] = []
        # Start with leaf-level aggregation: for each finest combination,
        # weighted-mean the observations and total the weights.
        finest_keys = hierarchy_cols
        agg = (
            df.assign(_wx=weights * obs, _w_=weights)
            .groupby(finest_keys, sort=False)[["_wx", "_w_"]]
            .sum()
        )
        leaf_means = agg["_wx"] / agg["_w_"]
        leaf_weights = agg["_w_"]

        # Compute within-leaf variance contribution (period-level).
        # We delegate this to the bottom level by using the panel-level data.
        # The standard Jewell approach treats period-within-leaf as the
        # innermost B-S problem.
        # Build a synthetic group key joining all hierarchy_cols.
        leaf_key = pd.MultiIndex.from_frame(df[finest_keys])
        leaf_codes = leaf_key.factorize()[0]

        current_means = leaf_means.values.astype(float)
        current_weights = leaf_weights.values.astype(float)
        current_index = leaf_means.index  # MultiIndex over hierarchy_cols

        # First, run a B-S step over (leaf, period) to get the leaf-level Z
        # and the within-leaf variance v that propagates up.
        unique_groups, group_means, group_weights, _, v_hat, _, _ = _bs_structural_estimates(
            leaf_codes, obs, weights
        )
        # The leaf-level structural v_hat applies; for between-leaf-within-parent
        # variance we estimate at each higher level using B-S with v_hat fixed
        # from the period level.

        # For the leaf level, between-variance ``a`` is what we estimate by
        # running B-S over leaves grouped under each parent.
        # Build levels from finest (hierarchy_cols[-1]) up to coarsest (hierarchy_cols[0]).
        # The loop always exits via `break` at the top-level case; the natural
        # `range`-exhaustion branch is unreachable because we reject empty
        # hierarchies upstream.
        for depth in range(len(hierarchy_cols), 0, -1):  # pragma: no branch
            current_levels = hierarchy_cols[:depth]
            parent_levels = hierarchy_cols[: depth - 1]  # may be empty

            # Build a frame indexed by current_levels with current means and weights.
            level_frame = pd.DataFrame(
                {"mean": current_means, "weight": current_weights},
                index=current_index,
            )
            if parent_levels:
                # For each parent group, run a B-S step over its children.
                z_vals = np.empty(len(level_frame))
                parent_means: dict[tuple, float] = {}
                parent_weights_next: dict[tuple, float] = {}
                a_vals: list[float] = []
                a_raw_vals: list[float] = []
                # Group children by parent.
                level_frame_reset = level_frame.reset_index()
                grouped = level_frame_reset.groupby(parent_levels, sort=False)
                # For groups with only one child, B-S cannot estimate ``a``.
                # We treat them as a == 0 (Z_child = 0) so the child inherits
                # the parent's value entirely. This is conservative.
                for parent_key, sub in grouped:
                    sub_idx = sub.index.to_numpy()
                    means_sub = sub["mean"].to_numpy(dtype=float)
                    weights_sub = sub["weight"].to_numpy(dtype=float)
                    if means_sub.size < 2:
                        # Cannot estimate between-child variance at this parent.
                        z_sub = np.zeros_like(weights_sub)
                        # parent mean = the only child's mean
                        parent_mean = float(means_sub[0])
                        a_hat = 0.0
                        a_raw = 0.0
                    else:
                        # B-S step at this level: each "observation" is a child
                        # mean with its total weight and *fixed* known process
                        # variance v / weight (because the leaf-level v is the
                        # within-leaf variance per unit of weight).
                        # Bühlmann-Gisler unbiased estimator of a between
                        # children of a single parent:
                        w_sum = float(weights_sub.sum())
                        x_bar = float((weights_sub * means_sub).sum() / w_sum)
                        between_ss = float(
                            (weights_sub * (means_sub - x_bar) ** 2).sum()
                        )
                        m = means_sub.size
                        denom = w_sum - float((weights_sub**2).sum()) / w_sum
                        if denom <= 0:  # pragma: no cover - Cauchy-Schwarz guarantees denom > 0
                            a_raw = 0.0
                        else:
                            a_raw = (between_ss - (m - 1) * v_hat) / denom
                        a_hat = max(a_raw, 0.0)
                        if a_hat == 0.0:
                            z_sub = np.zeros_like(weights_sub)
                            parent_mean = x_bar
                        else:
                            k = v_hat / a_hat
                            z_sub = weights_sub / (weights_sub + k)
                            z_total = float(z_sub.sum())
                            parent_mean = float(
                                (z_sub * means_sub).sum() / z_total
                            )
                    z_vals[sub_idx] = z_sub
                    parent_key_tuple = (
                        parent_key if isinstance(parent_key, tuple) else (parent_key,)
                    )
                    parent_means[parent_key_tuple] = parent_mean
                    # Weight propagation: aggregate weight of a parent is the
                    # sum of its children's weights. This is the standard
                    # Jewell formulation used as the input weight for the
                    # next-coarser B-S step.
                    parent_weights_next[parent_key_tuple] = float(weights_sub.sum())
                    a_vals.append(a_hat)
                    a_raw_vals.append(a_raw)

                levels_info.append(
                    {
                        "level_cols": list(current_levels),
                        "parent_cols": list(parent_levels),
                        "Z": pd.Series(z_vals, index=current_index, name="Z"),
                        "X": pd.Series(current_means, index=current_index, name="X"),
                        "a_per_parent": a_vals,
                    }
                )

                # Build next iteration: parent-level means and weights.
                next_index = pd.MultiIndex.from_tuples(
                    list(parent_means.keys()), names=parent_levels
                )
                current_means = np.array(
                    [parent_means[k] for k in parent_means.keys()], dtype=float
                )
                current_weights = np.array(
                    [parent_weights_next[k] for k in parent_means.keys()],
                    dtype=float,
                )
                current_index = next_index
            else:
                # Top level: no parent, just compute the grand mean.
                z_vals = np.zeros(len(level_frame))
                # For consistency, run a single B-S at the very top with
                # one synthetic super-group → Z = w/(w + v/a). But there is no
                # parent, so the "credibility premium" at this level *is* the
                # estimate.
                grand_w = float(level_frame["weight"].sum())
                if grand_w > 0:
                    grand_mean = float(
                        (level_frame["weight"] * level_frame["mean"]).sum() / grand_w
                    )
                else:  # pragma: no cover - positive weights are enforced upstream
                    grand_mean = float(level_frame["mean"].mean())
                self._mu = grand_mean
                levels_info.append(
                    {
                        "level_cols": list(current_levels),
                        "parent_cols": [],
                        "Z": pd.Series(z_vals, index=current_index, name="Z"),
                        "X": pd.Series(current_means, index=current_index, name="X"),
                        "a_per_parent": [],
                    }
                )
                break

        # Top-down recombination to get credibility premium at each level.
        # levels_info is ordered finest -> coarsest; reverse it for top-down.
        levels_top_down = list(reversed(levels_info))
        # Start with mu at the very top.
        prev_premium_by_key: dict[tuple, float] = {}
        # For the top level (no parent), each group's premium is the grand mean
        # blended with its own (Z=0 since we set z_vals=0); effectively mu.
        top = levels_top_down[0]
        top_index = top["X"].index
        top_z = top["Z"].to_numpy()
        top_x = top["X"].to_numpy()
        top_premium = top_z * top_x + (1.0 - top_z) * cast(float, self._mu)
        for key, p in zip(top_index, top_premium, strict=True):
            key_t = key if isinstance(key, tuple) else (key,)
            prev_premium_by_key[key_t] = float(p)

        all_premiums: list[pd.Series] = [
            pd.Series(top_premium, index=top_index, name="P_cred")
        ]
        for lvl in levels_top_down[1:]:
            parent_cols = lvl["parent_cols"]
            x = lvl["X"]
            z = lvl["Z"]
            # For each entry, parent key is the prefix of its multi-index.
            premiums = np.empty(len(x))
            for j, key in enumerate(x.index):
                key_t = key if isinstance(key, tuple) else (key,)
                parent_key = key_t[: len(parent_cols)]
                parent_p = prev_premium_by_key.get(parent_key, cast(float, self._mu))
                premiums[j] = z.iloc[j] * x.iloc[j] + (1.0 - z.iloc[j]) * parent_p
            series = pd.Series(premiums, index=x.index, name="P_cred")
            all_premiums.append(series)
            # Update prev_premium_by_key for next deeper level.
            prev_premium_by_key = {
                (k if isinstance(k, tuple) else (k,)): float(v)
                for k, v in zip(x.index, premiums, strict=True)
            }

        # The leaf premium is the last one.
        self._leaf_premiums = all_premiums[-1]
        self._levels = levels_info  # finest -> coarsest
        self._hierarchy_cols = list(hierarchy_cols)
        self._observation_col = observation_col
        self._weight_col = weight_col
        self._fitted = True
        return self

    def credibility_factor(self, level: str | None = None) -> pd.Series:
        """Credibility factors at the requested level.

        Parameters
        ----------
        level : str, optional
            Hierarchy level. If ``None``, the finest level is returned.
        """
        require_fitted(self._fitted, type(self).__name__)
        cols = cast(list[str], self._hierarchy_cols)
        target = level if level is not None else cols[-1]
        if target not in cols:
            raise KeyError(
                f"Level {target!r} is not part of the hierarchy {cols!r}."
            )
        # Find the level entry whose level_cols ends with `target`.
        for entry in self._levels:
            if entry["level_cols"] and entry["level_cols"][-1] == target:
                return entry["Z"].copy()
        raise KeyError(f"Level {target!r} not found in fitted model.")  # pragma: no cover

    def credibility_premium(self) -> pd.Series:
        """Credibility-weighted premium at the finest level."""
        require_fitted(self._fitted, type(self).__name__)
        return cast(pd.Series, self._leaf_premiums).copy()

    @property
    def grand_mean(self) -> float:
        require_fitted(self._fitted, type(self).__name__)
        return cast(float, self._mu)
