[![PyPI version](https://img.shields.io/pypi/v/actuarcredibility?color=blue)](https://pypi.org/project/actuarcredibility/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)]()

# actuarcredibility

Python library for actuarial credibility models. Covers Bühlmann, Bühlmann-Straub, Jewell hierarchical, Hachemeister regression, classical limited-fluctuation, and an optional Bayesian model via PyMC. Outputs integrate with pandas DataFrames. References Bühlmann & Gisler (2005) and Klugman ch. 20.

## Installation

From PyPI:

```bash
pip install actuarcredibility
```

From source:

```bash
git clone https://github.com/CosmikArt/actuarcredibility.git
cd actuarcredibility
pip install -e .
```

With Bayesian extensions:

```bash
pip install actuarcredibility[bayesian]
```

## Quickstart

Buhlmann-Straub on multi-year loss data:

```python
import pandas as pd
from actuarcredibility import BuhlmannStraubModel

# Multi-year loss experience by risk class
data = pd.DataFrame({
    "risk_class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year": [2021, 2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023],
    "loss_ratio": [0.62, 0.58, 0.65, 0.81, 0.77, 0.84, 0.45, 0.52, 0.48],
    "earned_premium": [5_000_000, 5_200_000, 5_500_000,
                       2_000_000, 2_100_000, 2_300_000,
                       8_000_000, 8_500_000, 9_000_000],
})

model = BuhlmannStraubModel()
model.fit(
    data,
    group_col="risk_class",
    observation_col="loss_ratio",
    weight_col="earned_premium",
)

# Credibility factor per risk class
print(model.credibility_factor())

# Credibility-weighted premium estimate
print(model.credibility_premium())
```

## Features

| Module | Description |
|---|---|
| `buhlmann` | `BuhlmannModel`, `BuhlmannStraubModel`. Non-parametric credibility with Bühlmann-Gisler unbiased structural estimators. |
| `hierarchical` | `JewellHierarchical`: multi-level hierarchical credibility for nested portfolios (region → territory → risk_class, etc.). |
| `regression` | `HachemeisterRegression` (regression credibility with covariates or a time trend; weighted least squares per group). |
| `classical` | `LimitedFluctuationCredibility`. Square-root rule for partial credibility, configurable tolerance and probability. |
| `bayesian` | `BayesianCredibility` is a PyMC hierarchical normal model with posterior credibility factors. Optional dependency. |
| `diagnostics` | `variance_decomposition`, `credibility_curve`, `shrinkage_summary`, `compare_models` |

## More examples

### Classical limited fluctuation

```python
from actuarcredibility import LimitedFluctuationCredibility

cred = LimitedFluctuationCredibility(k=0.05, p=0.90)
cred.full_credibility_standard()           # 1082 claims
cred.credibility_factor(n_claims=500)      # ~0.6797
cred.credibility_premium(observed=0.72, prior=0.65, n_claims=500)
```

### Hachemeister trend credibility

```python
import pandas as pd
from actuarcredibility import HachemeisterRegression

data = pd.DataFrame({
    "state": ["CA", "CA", "CA", "TX", "TX", "TX"],
    "year": [2021, 2022, 2023, 2021, 2022, 2023],
    "avg_claim_cost": [4500, 4700, 4900, 5200, 5400, 5600],
    "claim_count": [120, 130, 140, 80, 85, 95],
})

model = HachemeisterRegression().fit(
    data,
    group_col="state",
    observation_col="avg_claim_cost",
    time_col="year",
    weight_col="claim_count",
)
model.coefficients()              # credibility-weighted (intercept, slope) per state
model.predict("CA", year=2026)    # forecast next-year average cost
```

### Hierarchical (Jewell)

```python
import pandas as pd
from actuarcredibility import JewellHierarchical

data = pd.DataFrame({
    "region": (["W"]*8 + ["E"]*8),
    "territory": (["W1","W1","W2","W2"]*2 + ["E1","E1","E2","E2"]*2),
    "risk_class": (["A","B"]*8),
    "year": [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023] * 2,
    "loss_ratio": [0.62, 0.71, 0.58, 0.69, 0.65, 0.74, 0.60, 0.71,
                   0.81, 0.75, 0.66, 0.72, 0.83, 0.77, 0.68, 0.74],
    "earned_premium": [2_000_000]*16,
})

model = JewellHierarchical().fit(
    data,
    hierarchy_cols=["region", "territory", "risk_class"],
    observation_col="loss_ratio",
    weight_col="earned_premium",
)
model.credibility_factor(level="territory")
model.credibility_premium()       # finest-level credibility premium
```

### Diagnostics

```python
import pandas as pd
from actuarcredibility import BuhlmannStraubModel
from actuarcredibility.diagnostics import (
    variance_decomposition, credibility_curve, shrinkage_summary,
)

data = pd.DataFrame({
    "risk": ["A","A","A","B","B","B","C","C","C"],
    "year": [2021,2022,2023]*3,
    "loss_ratio": [0.62,0.58,0.65,0.81,0.77,0.84,0.45,0.52,0.48],
    "earned_premium": [5_000_000,5_200_000,5_500_000,
                       2_000_000,2_100_000,2_300_000,
                       8_000_000,8_500_000,9_000_000],
})

model = BuhlmannStraubModel().fit(data, "risk", "loss_ratio", "earned_premium")
variance_decomposition(model)     # v, a, between-share, k = v/a
credibility_curve(model)          # Z(w) tabulated for plotting
shrinkage_summary(model)          # raw vs. credibility distance to grand mean
```

## References

- Buhlmann, H. (1967). "Experience Rating and Credibility." *ASTIN Bulletin*, 4(3), 199-207.
- Buhlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and its Applications*. Springer.
- Klugman, S.A., Panjer, H.H. & Willmot, G.E. *Loss Models: From Data to Decisions*. Wiley.
- Jewell, W.S. (1975). "The Use of Collateral Data in Credibility Theory: A Hierarchical Model." *Giornale dell'Istituto Italiano degli Attuari*, 38, 1-16.
- Hachemeister, C.A. (1975). "Credibility for Regression Models with Application to Trend." In *Credibility: Theory and Applications*, P.M. Kahn (ed.), Academic Press.
- Casualty Actuarial Society. Exam 5 Study Notes: Credibility.

## Contributing

Run `pytest` before sending a PR.

## Author

Isaac López

MIT License. See [LICENSE](LICENSE).
