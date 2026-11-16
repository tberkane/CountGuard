# Mechanisms

Every mechanism turns a true `Cube` into a `Release`. For each released row, the release records
the value a reader sees (`shown`, NaN if withheld) and the range `[lo, hi]` an attacker can deduce
from it. One attack engine therefore covers every exact mechanism.

| Mechanism | Reader sees | Attacker's range for a row with true count *c* |
|---|---|---|
| `Suppress(k)` | *c* if *c* ≥ *k*; otherwise the row is absent | [*c*, *c*] or [0, *k*−1] |
| `Suppress(k, display="range")` | *c* if *c* ≥ *k*; "1-(k−1)" if 1 ≤ *c* < *k*; 0 rows absent | [*c*, *c*], [1, *k*−1] or [0, 0] |
| `Round(base, k)` | suppress below *k*, then round to a multiple of `base` | [max(*k*, *r*−⌊b/2⌋), *r*+⌈b/2⌉−1] |
| `Generalize(merge, drop, k)` | a coarser table (levels merged, variables dropped), suppressed below *k* | as `Suppress`, on the coarse rows |
| `DiscreteGaussian(rho)` | a noisy, consistent, non-negative cube | no hard range (scored by the DP attacker) |

## Suppression

`Suppress(k)` is the Cumulus default (*k* = 10): rows below *k* are left out. With
`display="range"`, suppressed rows are printed as "1-9". This is common in dashboards, but it
also tells the reader which rows are truly zero. The attack takes that into account, and on the
same cube a "1-9" table never reveals less than an omitted one.

## Rounding

`Round(base, k)` suppresses below *k* and rounds the rest half-up to a multiple of `base`.
`Round(k, k=k)` rounds every returned count to a multiple of the threshold. A rounded value can
fall below *k* (e.g. 1 rounded to base 3 gives 0), but the underlying true count never does.

## Generalization

`Generalize(merge={"race": {"Asian": "Other race", ...}}, drop=("sex",), k=10)` merges levels
and drops variables before suppressing. `countguard.frontier.generalization_ladder()` gives the
four-step ladder used in the sweep: race in 4 groups, race in 2 groups, no race, then no race and no sex.

## Differential privacy

`DiscreteGaussian(rho, max_order=3)`:

1. Adds discrete Gaussian noise, drawn by [OpenDP](https://opendp.org), to every marginal
   with up to `max_order` grouping variables (all one-, two- and three-way tables, plus the
   grand total). One patient changes each marginal by 1 in one cell. With $|W|$ marginals at
   scale $\sigma$, the release satisfies $\rho$-zCDP with $\rho = |W| / (2\sigma^2)$. The
   accounting uses OpenDP's measurement map, and a release that would spend more than `rho`
   raises `PrivacyBudgetError`.
2. Fits the non-negative atomic counts that best match the noisy marginals (least squares,
   L-BFGS-B), rounded to integers. Every published row is a sum of those cells, so all subtotals
   add up and no count is negative. This step is post-processing and costs no budget.

!!! warning "Seeds are for benchmarks only"
    OpenDP's sampler is cryptographically secure and cannot be seeded. `DiscreteGaussian(seed=…)`
    (CLI: `--seed`) draws from NumPy instead (Canonne, Kamath & Steinke's rejection sampler) so
    that sweeps and tests are reproducible. Don't use a seed for a real release.

## Composition

`Pipeline(generalize, noise, threshold)` applies the steps in that order, and each step is optional:

```python
Pipeline(Generalize(merge=RACE_COARSE), DiscreteGaussian(1.0), Round(10, k=10))
```

When noise is present, suppression and rounding act on the noisy counts. They are post-processing,
so the zCDP guarantee is unchanged. The attacker is credited with a fresh consistent estimate
fitted to the published rows only. Without noise the release is exact, and the reconstruction
attack applies.

## What the sweep shows on synthetic cohorts

From the frontier shipped with the demo (`demo/data/`, synthetic seed 2, `known_support`
attacker; DP figures are medians over 20 noise draws):

| | n = 285 | n = 20,000 |
|---|---|---|
| Suppression, k = 10 | risk 42.1%, loss 0%; 72 of 82 small groups forced | risk 4.9%, loss 0%; all 436 forced |
| Suppression, k = 5 … 50 | risk 45% → 22%, loss 0% → 108% | risk 4.9% for every k (subtraction recovers every small group) |
| Generalization, no race and no sex | risk 19.3%, loss 19.6% | risk 0.95%, loss 8.7% |
| DP, ρ = 1 | risk 2.1%, loss 9.9% (closest to the targets) | risk 0.14%, loss 0.5% (meets both) |
| Settings meeting both targets | none | every DP budget from ρ = 0.05 to 5 |
