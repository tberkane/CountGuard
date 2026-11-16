# Changelog

All notable changes to CountGuard are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may change the API.

## [0.1.0] - 2026-11-16

First public, early release. All results in the repository are on synthetic cohorts.

### Added

- **Count cubes** (`Cube`, `Schema`): atomic cells, every GROUP BY CUBE row, and the row-by-cell matrix.
- **Mechanisms**: `Suppress` (below *k*, rows omitted or shown as "1-(k-1)"), `Round`
  (including round-to-*k*), `Generalize` (merge levels, drop variables), `DiscreteGaussian`
  (discrete Gaussian noise via OpenDP on all marginals up to a chosen order, zCDP accounting,
  consistent non-negative least-squares cube), and `Pipeline` to compose them as
  generalize → DP → suppress/round.
- **Reconstruction attack** (`run_attack`): interval propagation, then LP and MIP bounds with HiGHS,
  for two threat models (`known_support`, `table_only`); generalized releases supported.
- **Bayesian reconstruction** (`countguard.bayes`, optional `[bayes]` extra): port of the original
  research code's Stan model and integer-constrained sampler.
- **Metrics**: re-identification risk above a blind-guessing floor, certain recoveries, median and mean
  relative error over fixed feasibility queries.
- **Privacy-utility frontier** (`countguard.frontier.sweep`): default grid, seeded DP repetitions,
  Pareto set, recommendation against the UH3 targets (risk < 1%, loss < 5%).
- **Cumulus I/O** (`countguard.cumulus`): the cumulus-library count-table SQL (run in DuckDB),
  reading and writing count tables as CSV, Parquet or pandas DataFrames.
- **External mode prototype** (`countguard.external`): per-user zCDP ledger, cached answers,
  differencing-attack planner.
- **CLI** `countguard synth | audit | protect | frontier`, with JSON or YAML reports that carry a
  provenance block (version, command, timestamp, input SHA-256, parameters).
- **Reproduction** of the original research code on a synthetic cohort (`reproduction/`), with a
  regression test: identical bounds for every cell.
- Tests (unit, Hypothesis property tests, CLI, regression), ruff, mypy, GitHub Actions CI,
  documentation site, Apache-2.0 license, NOTICE, CITATION.cff, codemeta.json, security policy.

[0.1.0]: https://github.com/tberkane/CountGuard/releases/tag/v0.1.0
