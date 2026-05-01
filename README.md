[![PyPI version](https://img.shields.io/pypi/v/actuarcredibility?color=blue)](https://pypi.org/project/actuarcredibility/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

# actuarcredibility

**Credibility models for actuarial pricing — from Buhlmann to Bayesian.**

---

## What is actuarcredibility?

Credibility theory is the backbone of actuarial pricing. Every P&C actuary learns it, most use it daily, and yet there is **zero dedicated Python implementation**. The state of the art remains Excel spreadsheets, SAS macros, and hand-rolled R scripts that never leave their author's laptop.

`actuarcredibility` fills that gap. It provides production-grade implementations of every major credibility framework — classical limited-fluctuation standards, the Buhlmann family, Jewell's hierarchical extension, Hachemeister's regression credibility, and modern Bayesian approaches via optional PyMC integration. The API is consistent, the math is documented, and the outputs integrate cleanly with pandas workflows.

If you price insurance, this library replaces your spreadsheet.

---

## Installation

**From PyPI (once published):**

```bash
pip install actuarcredibility
```

**From source:**

```bash
git clone https://github.com/CosmikArt/actuarcredibility.git
cd actuarcredibility
pip install -e .
```

**With Bayesian extensions:**

```bash
pip install actuarcredibility[bayesian]
```

---

## Quickstart

Fit a Buhlmann-Straub model to multi-year loss experience by risk class:

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

---

## Features

| Module | Description |
|---|---|
| `buhlmann` | `BuhlmannModel`, `BuhlmannStraubModel` — non-parametric credibility with Bühlmann-Gisler unbiased structural estimators |
| `hierarchical` | `JewellHierarchical` — multi-level hierarchical credibility for nested portfolios (region → territory → risk_class, etc.) |
| `regression` | `HachemeisterRegression` — regression credibility with covariates or time trend, weighted least squares per group |
| `classical` | `LimitedFluctuationCredibility` — square-root rule for partial credibility, configurable tolerance and probability |
| `bayesian` | `BayesianCredibility` — PyMC hierarchical normal model with posterior credibility factors (optional dependency) |
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
from actuarcredibility import HachemeisterRegression

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
from actuarcredibility import JewellHierarchical

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
from actuarcredibility import BuhlmannStraubModel
from actuarcredibility.diagnostics import (
    variance_decomposition, credibility_curve, shrinkage_summary,
)

model = BuhlmannStraubModel().fit(data, "risk", "loss_ratio", "earned_premium")
variance_decomposition(model)     # v, a, between-share, k = v/a
credibility_curve(model)          # Z(w) tabulated for plotting
shrinkage_summary(model)          # raw vs. credibility distance to grand mean
```

---

## References

- Buhlmann, H. (1967). "Experience Rating and Credibility." *ASTIN Bulletin*, 4(3), 199-207.
- Buhlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and its Applications*. Springer.
- Klugman, S.A., Panjer, H.H. & Willmot, G.E. *Loss Models: From Data to Decisions*. Wiley.
- Jewell, W.S. (1975). "The Use of Collateral Data in Credibility Theory: A Hierarchical Model." *Giornale dell'Istituto Italiano degli Attuari*, 38, 1-16.
- Hachemeister, C.A. (1975). "Credibility for Regression Models with Application to Trend." In *Credibility: Theory and Applications*, P.M. Kahn (ed.), Academic Press.
- Casualty Actuarial Society. Exam 5 Study Notes — Credibility.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request so we can discuss scope and approach.

```bash
git clone https://github.com/CosmikArt/actuarcredibility.git
cd actuarcredibility
pip install -e ".[dev]"
pytest
```

---

## Author

**Isaac López**

---

*MIT License. See [LICENSE](LICENSE) for details.*
