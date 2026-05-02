# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-01

### Added
- Real implementations (no longer scaffolds) for every credibility model:
  - `BuhlmannModel`: non-parametric Bühlmann credibility (equal weights).
  - `BuhlmannStraubModel`. Bühlmann-Straub with exposure weights, using
    Bühlmann-Gisler unbiased moment estimators of `v` and `a`.
  - `JewellHierarchical` (multi-level hierarchical credibility with
    arbitrary nesting depth, iterative bottom-up B-S estimation, and
    top-down credibility-premium recombination).
  - `HachemeisterRegression`: regression credibility with covariates or a
    time trend, weighted least squares per group, and PSD projection of the
    between-coefficient covariance estimate.
  - `LimitedFluctuationCredibility`. Full and partial credibility under the
    classical square-root rule.
  - `BayesianCredibility` (PyMC-based hierarchical normal model with
    posterior credibility factors and premiums).
- New `diagnostics` module: `variance_decomposition`, `credibility_curve`,
  `shrinkage_summary`, `compare_models`.
- `predict()` helpers on `BuhlmannStraubModel` and `HachemeisterRegression`.
- `summary()` and `structural_parameters` accessors on the B-S model.
- `py.typed` marker; the package now ships type information.
- Test suite of 109 tests covering numerical correctness, textbook formulas,
  input validation, and edge cases.

### Changed
- Source split from a single `core.py` into per-model modules
  (`classical`, `buhlmann`, `hierarchical`, `regression`, `bayesian`,
  `diagnostics`).
- Public API stays import-compatible from the package root.
- Version bumped to `0.1.0`.

## [0.0.1] - 2026-04-26

### Added
- Initial project scaffold (interface only; no implementations).
