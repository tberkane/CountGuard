# API reference

The most used names are re-exported from `countguard`:
`Cube`, `Schema`, `Suppress`, `Round`, `Generalize`, `DiscreteGaussian`, `Pipeline`,
`Release`, `run_attack`, `assess`, `accuracy_loss`, `make_queries`, `read_table`,
`release_to_table`, `write_table`, `kidney_cohort`.

## Cubes

::: countguard.cube
    options:
      members: [Schema, Cube, build_A, row_counts_from_cells]

## Mechanisms

::: countguard.mechanisms
    options:
      members: [Release, Suppress, Round, Generalize, DiscreteGaussian, Pipeline, PrivacyBudgetError, sample_discrete_gaussian, workload, nnls_fit]

## Attack

::: countguard.attack
    options:
      members: [run_attack, attack, AttackResult, propagate, InfeasibleRelease]

## Bayesian reconstruction

Requires the `[bayes]` extra.

::: countguard.bayes
    options:
      members: [reconstruct, BayesResult]

## Metrics

::: countguard.metrics
    options:
      members: [Privacy, assess, privacy, best_privacy, luck_floor, privacy_from_attack, privacy_from_dp, dp_guesses, Queries, make_queries, answer, accuracy_loss, in_box]

## Privacy-utility frontier

::: countguard.frontier
    options:
      members: [sweep, default_grid, generalization_ladder, mark_pareto, recommend, policy_text]

## Cumulus tables

::: countguard.cumulus
    options:
      members: [read_table, release_to_table, write_table, count_sql, cumulus_table_from_cohort]

## External mode (prototype)

::: countguard.external
    options:
      members: [SponsorSession, Ledger, BudgetExceeded, parse_query, differencing_plan]

## Synthetic data

::: countguard.synth
    options:
      members: [kidney_cohort]
