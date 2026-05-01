"""Classical limited-fluctuation credibility.

Implements the oldest credibility framework: the **square-root rule** for
partial credibility once full-credibility volume requirements are not met.

References
----------
- Longley-Cook, L.H. (1962). "An Introduction to Credibility Theory."
  Proceedings of the Casualty Actuarial Society.
- Mahler, H.C. & Dean, C.G. (2001). "Credibility." In *Foundations of
  Casualty Actuarial Science*, Casualty Actuarial Society.
- Klugman, Panjer & Willmot. *Loss Models: From Data to Decisions*. Wiley.
"""

from __future__ import annotations

import math

from scipy.stats import norm


class LimitedFluctuationCredibility:
    """Classical limited-fluctuation (full and partial) credibility.

    Full credibility is granted when the volume of experience is large enough
    that the observed statistic is unlikely to deviate from the true value by
    more than a fraction ``k`` with probability ``p``.

    For a Poisson claim count, the standard for full credibility of frequency
    is:

        n_F = (z / k)**2,    z = Phi^{-1}((1 + p) / 2)

    With ``k = 0.05`` and ``p = 0.90``, ``z = 1.6449`` and ``n_F = 1082``.

    Partial credibility uses the square-root rule:

        Z = min(1, sqrt(n / n_F))

    Parameters
    ----------
    k : float, default 0.05
        Maximum tolerable relative deviation from the true value.
    p : float, default 0.90
        Required probability that the observation lies within ``k`` of truth.

    Examples
    --------
    >>> cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
    >>> cred.full_credibility_standard()
    1082
    >>> round(cred.credibility_factor(n_claims=500), 4)
    0.6797
    >>> round(cred.credibility_premium(observed=0.72, prior=0.65, n_claims=500), 4)
    0.6976
    """

    def __init__(self, k: float = 0.05, p: float = 0.90) -> None:
        if not 0 < k < 1:
            raise ValueError(f"k must be in (0, 1); got {k}.")
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1); got {p}.")
        self.k = float(k)
        self.p = float(p)

    @property
    def z_value(self) -> float:
        """Two-sided normal quantile at probability ``p``."""
        return float(norm.ppf((1.0 + self.p) / 2.0))

    def full_credibility_standard(self, cv: float = 1.0) -> int:
        """Number of claims required for full credibility.

        For a generic frequency distribution with coefficient of variation
        ``cv``, the standard is:

            n_F = round((z / k)**2 * cv**2)

        Rounding (rather than ceiling) is used to match the historical
        actuarial convention: ``k = 0.05``, ``p = 0.90`` gives the classical
        ``1082`` claims standard.

        For Poisson, ``cv = 1`` (variance equals mean). For a mixed Poisson
        with contagion parameter ``c``, ``cv**2 = 1 + c * E[N]`` and the
        number grows accordingly.

        Parameters
        ----------
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.

        Returns
        -------
        int
            Number of claims required for full credibility.
        """
        if cv <= 0:
            raise ValueError(f"cv must be positive; got {cv}.")
        n = (self.z_value / self.k) ** 2 * cv**2
        return int(round(n))

    def credibility_factor(self, n_claims: float, cv: float = 1.0) -> float:
        """Partial credibility factor by the square-root rule.

        Parameters
        ----------
        n_claims : float
            Observed (or expected) number of claims.
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.

        Returns
        -------
        float
            Credibility factor in ``[0, 1]``.
        """
        if n_claims < 0:
            raise ValueError(f"n_claims must be non-negative; got {n_claims}.")
        n_full = self.full_credibility_standard(cv=cv)
        return min(1.0, math.sqrt(n_claims / n_full))

    def credibility_premium(
        self,
        observed: float,
        prior: float,
        n_claims: float,
        cv: float = 1.0,
    ) -> float:
        """Credibility-weighted premium ``Z * observed + (1 - Z) * prior``.

        Parameters
        ----------
        observed : float
            Observed loss ratio, pure premium, or other target statistic.
        prior : float
            Prior estimate (manual rate, industry average, collective mean).
        n_claims : float
            Observed (or expected) number of claims supporting ``observed``.
        cv : float, default 1.0
            Coefficient of variation of the claim count distribution.

        Returns
        -------
        float
            Credibility-weighted premium estimate.
        """
        z = self.credibility_factor(n_claims=n_claims, cv=cv)
        return z * observed + (1.0 - z) * prior
