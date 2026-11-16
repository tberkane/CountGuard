# Roadmap

This is a plan, not a list of features: items here are **not yet implemented** unless the
[CHANGELOG](CHANGELOG.md) says so.

## Near term (to mid-2027)

- First risk report on synthetic cohorts shaped like the GIC-Recruit network
- Security-response process exercised end to end

## 1.0 (2027-2028)

- Packages on PyPI and conda-forge, a versioned Docker image, Zenodo DOI for each release
- Multi-query attacks (combining several queries to expose a patient)
- Utility metrics beyond relative error: R², Jensen-Shannon divergence, Cramér's V,
  log-odds-ratio error, reversed findings, classifier accuracy
- Monte Carlo audits against adversaries with varying background knowledge
- External-query mode for GIC-Recruit: external-tier policy, per-user budgets, repeated
  queries answered identically
- User guide and Jupyter notebook tutorials

## Later

- Splitting large cubes into subcubes over disjoint variables, with runtime benchmarks
  (target: under one hour on a standard site server)
- Attacks that difference monthly releases; genomic membership inference
- Cross-site audits after Cumulus export, with cumulative budgets per study
