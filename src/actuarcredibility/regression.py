"""Hachemeister regression credibility.

Implements Hachemeister's (1975) extension of Bühlmann-Straub credibility to
regression models. Each risk group is fit with its own weighted regression
and credibility-weighted toward a collective regression coefficient vector
using a credibility *matrix* rather than a scalar factor.

References
----------
- Hachemeister, C.A. (1975). "Credibility for Regression Models with
  Application to Trend." In *Credibility: Theory and Applications*,
  P.M. Kahn (ed.), Academic Press, 129-163.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory*. Springer,
  Chapter 8.
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


def _project_to_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the cone of positive-semidefinite matrices."""
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    return (eigvecs * eigvals_clipped) @ eigvecs.T


class HachemeisterRegression:
    """Hachemeister regression credibility model.

    The model fits, for each risk group ``i``,

        Y_i = X_i * beta_i + epsilon_i,    Var(epsilon_i) = sigma^2 * W_i^{-1}

    by weighted least squares, then credibility-weights the coefficient
    vectors:

        beta_cred_i = Z_i * beta_i_hat + (I - Z_i) * beta_collective,

    where the credibility matrix is

        Z_i = A * (A + sigma^2 * (X_i^T W_i X_i)^{-1})^{-1}.

    The structural parameters ``sigma^2`` and the between-coefficient
    covariance ``A`` are estimated from the data using moment estimators.
    If the moment estimate of ``A`` is not positive-semidefinite it is
    projected onto the PSD cone (eigenvalue clipping at 0).

    Parameters
    ----------
    None at construction. Call :meth:`fit` to estimate the model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for g in range(5):
    ...     intercept = rng.normal(0.6, 0.05)
    ...     slope = rng.normal(0.02, 0.005)
    ...     for t in range(6):
    ...         rows.append({"g": g, "t": t,
    ...                      "y": intercept + slope * t + rng.normal(0, 0.01),
    ...                      "w": 1.0})
    >>> df = pd.DataFrame(rows)
    >>> m = HachemeisterRegression().fit(
    ...     df, group_col="g", observation_col="y",
    ...     time_col="t", weight_col="w",
    ... )
    >>> bool(m.coefficients().shape == (5, 2))
    True
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._group_col: str | None = None
        self._observation_col: str | None = None
        self._time_col: str | None = None
        self._weight_col: str | None = None
        self._covariates: list[str] | None = None
        self._design_cols: list[str] | None = None
        self._groups: np.ndarray | None = None
        self._beta_hat: np.ndarray | None = None       # (r, p)
        self._beta_collective: np.ndarray | None = None  # (p,)
        self._beta_cred: np.ndarray | None = None      # (r, p)
        self._Z: np.ndarray | None = None              # (r, p, p)
        self._S: np.ndarray | None = None              # (r, p, p): (X'WX)^{-1}
        self._sigma2: float | None = None
        self._A: np.ndarray | None = None              # (p, p)
        self._A_raw: np.ndarray | None = None
        self._design_per_group: dict | None = None

    @staticmethod
    def _build_design(
        df: pd.DataFrame,
        time_col: str | None,
        covariates: list[str] | None,
    ) -> tuple[np.ndarray, list[str]]:
        if covariates:
            cols = ["_intercept", *covariates]
            mat = np.column_stack(
                [np.ones(len(df)), *(df[c].to_numpy(dtype=float) for c in covariates)]
            )
        elif time_col is not None:
            cols = ["_intercept", time_col]
            mat = np.column_stack(
                [np.ones(len(df)), df[time_col].to_numpy(dtype=float)]
            )
        else:
            raise ValueError(
                "HachemeisterRegression.fit requires either time_col or "
                "covariates to define the regression design."
            )
        return mat, cols

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str,
        observation_col: str,
        time_col: str | None = None,
        weight_col: str | None = None,
        covariates: list[str] | None = None,
    ) -> HachemeisterRegression:
        required = [group_col, observation_col]
        if time_col is not None:
            required.append(time_col)
        if weight_col is not None:
            required.append(weight_col)
        if covariates:
            required.extend(covariates)
        require_columns(data, required)

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

        design, design_cols = self._build_design(df, time_col, covariates)
        require_finite(design, "design matrix")
        p = design.shape[1]

        groups = df[group_col].to_numpy()
        unique_groups = pd.unique(groups)
        r = unique_groups.size
        if r < 2:
            raise ValueError(
                f"At least 2 risk groups are required; found {r}."
            )

        # Per-group fits.
        beta_hat = np.empty((r, p))
        S_inv = np.empty((r, p, p))  # X'WX (information)
        S = np.empty((r, p, p))      # (X'WX)^{-1}
        residual_ss = np.zeros(r)
        n_obs = np.zeros(r, dtype=int)
        design_per_group: dict = {}

        for i, g in enumerate(unique_groups):
            mask = groups == g
            X_i = design[mask]
            W_i = weights[mask]
            Y_i = obs[mask]
            if X_i.shape[0] < p:
                raise ValueError(
                    f"Group {g!r} has only {X_i.shape[0]} observation(s); "
                    f"need at least {p} to fit a {p}-parameter regression."
                )
            XtW = X_i.T * W_i  # (p, n_i)
            XtWX = XtW @ X_i
            XtWY = XtW @ Y_i
            try:
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"Singular design matrix for group {g!r}: regression cannot "
                    f"be fit. Check for collinearity or insufficient variation."
                ) from exc
            beta_i = XtWX_inv @ XtWY
            beta_hat[i] = beta_i
            S_inv[i] = XtWX
            S[i] = XtWX_inv
            resid = Y_i - X_i @ beta_i
            residual_ss[i] = float(resid @ (W_i * resid))
            n_obs[i] = X_i.shape[0]
            design_per_group[g] = (X_i, W_i, Y_i)

        # Pooled estimate of sigma^2.
        dof = int((n_obs - p).sum())
        if dof <= 0:
            raise ValueError(
                "Insufficient degrees of freedom to estimate residual variance "
                "(sum of n_i - p across groups must be positive)."
            )
        sigma2 = float(residual_ss.sum() / dof)

        # Estimate of A: sample covariance of beta_hat across groups minus
        # the average within-group sampling covariance sigma^2 * S.
        # (Hachemeister-style moment estimator.)
        beta_mean = beta_hat.mean(axis=0)
        centered = beta_hat - beta_mean
        sample_cov = (centered.T @ centered) / max(r - 1, 1)
        avg_S = S.mean(axis=0)
        A_raw = sample_cov - sigma2 * avg_S
        A = _project_to_psd(A_raw)

        # Credibility matrices: Z_i = A * (A + sigma^2 * S_i)^{-1}
        Z = np.empty_like(S)
        for i in range(r):
            inner = A + sigma2 * S[i]
            try:
                inner_inv = np.linalg.inv(inner)
            # pragma: no cover - sigma^2 * S_i is PD whenever the per-group fit succeeds
            except np.linalg.LinAlgError as exc:  # pragma: no cover
                raise ValueError(
                    f"Could not invert (A + sigma^2 * S_i) for group "
                    f"{unique_groups[i]!r}; numerical instability."
                ) from exc
            Z[i] = A @ inner_inv

        # Collective beta: weighted by Z_i (Bühlmann-Gisler 8.27).
        Z_sum = Z.sum(axis=0)
        try:
            Z_sum_inv = np.linalg.inv(Z_sum)
        except np.linalg.LinAlgError:
            # All Z_i are zero (A == 0): fall back to information-weighted mean.
            info_sum = S_inv.sum(axis=0)
            info_weighted = sum(S_inv[i] @ beta_hat[i] for i in range(r))
            beta_collective = np.linalg.solve(info_sum, info_weighted)
        else:
            zb = sum(Z[i] @ beta_hat[i] for i in range(r))
            beta_collective = Z_sum_inv @ zb

        beta_cred = np.empty_like(beta_hat)
        Ip = np.eye(p)
        for i in range(r):
            beta_cred[i] = Z[i] @ beta_hat[i] + (Ip - Z[i]) @ beta_collective

        self._fitted = True
        self._group_col = group_col
        self._observation_col = observation_col
        self._time_col = time_col
        self._weight_col = weight_col
        self._covariates = list(covariates) if covariates else None
        self._design_cols = design_cols
        self._groups = unique_groups
        self._beta_hat = beta_hat
        self._beta_collective = beta_collective
        self._beta_cred = beta_cred
        self._Z = Z
        self._S = S
        self._sigma2 = sigma2
        self._A = A
        self._A_raw = A_raw
        self._design_per_group = design_per_group
        return self

    def coefficients(self) -> pd.DataFrame:
        """Per-group credibility-weighted regression coefficients."""
        require_fitted(self._fitted, type(self).__name__)
        return pd.DataFrame(
            cast(np.ndarray, self._beta_cred),
            index=pd.Index(self._groups, name=self._group_col),
            columns=cast(list[str], self._design_cols),
        )

    def coefficients_individual(self) -> pd.DataFrame:
        """Per-group raw (un-credibilitised) WLS coefficients."""
        require_fitted(self._fitted, type(self).__name__)
        return pd.DataFrame(
            cast(np.ndarray, self._beta_hat),
            index=pd.Index(self._groups, name=self._group_col),
            columns=cast(list[str], self._design_cols),
        )

    def coefficients_collective(self) -> pd.Series:
        require_fitted(self._fitted, type(self).__name__)
        return pd.Series(
            cast(np.ndarray, self._beta_collective),
            index=cast(list[str], self._design_cols),
            name="beta_collective",
        )

    def credibility_factor(self) -> pd.DataFrame:
        """Per-group credibility matrices, stacked into a long DataFrame.

        For a ``p``-parameter regression and ``r`` groups, returns an
        ``r * p``-row DataFrame with the credibility matrix rows for every
        group. To recover the matrix for a specific group, use
        :meth:`credibility_matrix`.
        """
        require_fitted(self._fitted, type(self).__name__)
        Z = cast(np.ndarray, self._Z)
        r, p, _ = Z.shape
        rows = []
        for i, g in enumerate(self._groups):
            for k in range(p):
                row = {self._group_col: g, "row": cast(list[str], self._design_cols)[k]}
                for j in range(p):
                    row[cast(list[str], self._design_cols)[j]] = Z[i, k, j]
                rows.append(row)
        return pd.DataFrame(rows)

    def credibility_matrix(self, group: object) -> pd.DataFrame:
        """Credibility matrix ``Z_i`` for a single group."""
        require_fitted(self._fitted, type(self).__name__)
        groups = cast(np.ndarray, self._groups)
        idx = np.where(groups == group)[0]
        if idx.size == 0:
            raise KeyError(f"Group {group!r} was not present in the fitted data.")
        i = int(idx[0])
        cols = cast(list[str], self._design_cols)
        return pd.DataFrame(
            cast(np.ndarray, self._Z)[i], index=cols, columns=cols
        )

    def credibility_premium(self) -> pd.Series:
        """Credibility-weighted predicted value at each group's mean covariates.

        For trend models (intercept + time), this evaluates the credibility
        regression at the group's mean time, giving the credibility-weighted
        loss-ratio (or other target) over the observation window.
        """
        require_fitted(self._fitted, type(self).__name__)
        beta_cred = cast(np.ndarray, self._beta_cred)
        groups = cast(np.ndarray, self._groups)
        design_per_group = cast(dict, self._design_per_group)
        prem = np.empty(len(groups))
        for i, g in enumerate(groups):
            X_i, W_i, _ = design_per_group[g]
            mean_x = (W_i[:, None] * X_i).sum(axis=0) / W_i.sum()
            prem[i] = float(mean_x @ beta_cred[i])
        return pd.Series(prem, index=pd.Index(groups, name=self._group_col), name="P_cred")

    def predict(self, group: object, **covariate_values: float) -> float:
        """Predict the credibility-weighted target for a group at given covariates.

        Parameters
        ----------
        group : object
            Risk group label.
        **covariate_values : float
            Values for each design column except the implicit intercept.
            For trend models, pass the time value (keyed by ``time_col``).
        """
        require_fitted(self._fitted, type(self).__name__)
        cols = cast(list[str], self._design_cols)
        x = np.ones(len(cols))
        for j, c in enumerate(cols):
            if c == "_intercept":
                continue
            if c not in covariate_values:
                raise KeyError(
                    f"Missing value for covariate {c!r}. "
                    f"Required covariates: {[c for c in cols if c != '_intercept']}."
                )
            x[j] = float(covariate_values[c])
        groups = cast(np.ndarray, self._groups)
        idx = np.where(groups == group)[0]
        if idx.size == 0:
            raise KeyError(f"Group {group!r} was not present in the fitted data.")
        return float(x @ cast(np.ndarray, self._beta_cred)[int(idx[0])])

    @property
    def structural_parameters(self) -> dict[str, object]:
        """Estimated structural parameters: ``sigma2`` and matrix ``A``."""
        require_fitted(self._fitted, type(self).__name__)
        return {
            "sigma2": cast(float, self._sigma2),
            "A": cast(np.ndarray, self._A).copy(),
            "A_raw": cast(np.ndarray, self._A_raw).copy(),
        }
