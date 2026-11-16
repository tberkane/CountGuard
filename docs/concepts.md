# Concepts

## Cubes, rows and cells

A **count cube** over variables $v_1, \dots, v_m$ is the output of
`GROUP BY CUBE(v1, ..., vm)`: one row for every combination in which each variable is either
fixed to a level or rolled up ("all"). An **atomic cell** fixes every variable. Every cube row is
the sum of the atomic cells under it, so a release is a set of linear constraints on the atomic
counts. That is what the attack exploits.

For the synthetic 10-variable kidney cohort there are 3,072 atomic cells and 137,781 cube rows.

## Threat models

| | `known_support` | `table_only` |
|---|---|---|
| Attacker knows | which combinations have at least one patient | only the released table |
| Unknowns | at most *n* (87 cells at n = 285) | every atomic cell (3,072) |
| Notes | the original research code's model: its cells are the observed combinations | weaker; no patient-level knowledge needed |

The auditor holds the true counts, so CountGuard can score either attacker. Results default to
`known_support`, the stronger attacker of the two. Its bounds are integer-programming bounds:
interval propagation first ("subtraction"), then LP and MIP with HiGHS for the cells still open.
Every step is a sound deduction, so the true count always lies within the bounds.
`tests/test_properties.py` checks this on random cubes.

## How risk is scored

**Small group**: an atomic cell with **1–9** true patients. This is fixed at 1–9 whatever the
rule's own *k*, so rules with different thresholds are scored on the same patients.

**Raw risk**: the share of *all* patients who are in a small group whose exact size the attacker
gets right.

**Attacker's guess** for each cell: the exact value when the bounds meet (min = max). Otherwise the
attacker uses whichever of two simple rules scores better on that release: the midpoint of
[min, max], or the smallest value ≥ 1 in it. For DP releases the attacker reads the released,
consistent estimate, rounded, either as is or raised to at least 1.

**Luck floor**: the share a blind attacker gets right by guessing one fixed value (the best in
hindsight, usually 1) for every small group, with no table at all.

$$\text{risk} = \max(0,\ \text{raw} - \text{luck floor})$$

**Certain recoveries**: small groups whose size is mathematically forced (min = max). Under DP
there are none.

For generalized releases, risk is judged on the **released** groups: a patient alone in a group
of released attributes is exposed even if other variables were dropped.

## How accuracy loss is scored

A fixed set of 500 random feasibility queries, each with 1–3 conditions (e.g. *sex = female and
mTOR inhibitor = yes*) and a non-zero true count, is answered from the released table. A
withheld answer is read as the midpoint of 1..k−1. For a generalized release, a condition on a
merged level or dropped variable is split uniformly.

**Loss** is the **median relative error** $|\hat c - c| / c$ over the queries. The report also gives
the mean relative error and the share of withheld answers, because the median can hide damage
that affects a minority of queries.

## Targets

The default targets are the GIC-Recruit UH3 milestones: re-identification risk **< 1%** and
accuracy loss **< 5%**. `frontier.recommend` picks the lowest-loss setting that meets both.
If none does, it picks the setting closest to meeting them and says so.

## Caveats

- The audit covers the attacks described here. It is not a proof that no other attack exists.
- Scores depend on the cohort: the same rule can be safe for 20,000 patients and unsafe for 285.
- v0.1.0 results are on synthetic cohorts only.
