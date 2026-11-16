# Quickstart

## Install

CountGuard needs Python 3.12 or later.

```bash
git clone https://github.com/tberkane/CountGuard && cd CountGuard
pip install -e .
```

Optional extras: `[bayes]` for the Bayesian reconstruction (cmdstanpy; run
`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` once), `[demo]` for the local demo
page, `[dev]` for tests and linters, `[docs]` for this site.

## 1. Make a synthetic cohort

```bash
countguard synth -n 285 --seed 2 -o cohort.csv
```

`cohort.csv` has one row per (synthetic) patient: race, sex, seven immunosuppressant classes and
graft rejection, the shape of the preprint's 10-variable cube (3,072 combinations).

## 2. Release a count cube

```bash
countguard protect cohort.csv -o cube.csv
```

The default is the Cumulus convention: every GROUP BY CUBE row with at least 10 patients is
published, and the rest are left out.

```text
cnt,race,sex,cni,antimetabolite,steroid,mtor,belatacept,atg,il2ra,rejection
285,,,,,,,,,,
246,,,,,,,,,,no
39,,,,,,,,,,yes
```

An empty value means "all" (rolled up), as in Cumulus.

## 3. Audit it

With the patient-level source, the auditor scores the release against the strongest attacker
(see [Concepts](concepts.md)):

```bash
countguard audit cube.csv --k 10 --cohort cohort.csv
```

```json
"attack":   {"threat": "known_support", "seconds": 0.12, "atomic_cells": 3072, "unknowns": 87,
             "exactly_determined": 77, ...},
"exposure": {"small_groups_forced": 72, "patients_in_forced_groups": 154, "share_of_patients": 0.54},
"risk":     {"risk": 0.421, "raw": 0.579, "floor": 0.158, "certain": 72, "small_groups": 82, ...,
             "target": 0.01, "meets_target": false}
```

The attack proves the exact size of 72 of the 82 groups with 1–9 patients. Without `--cohort`,
the audit runs the `table_only` attacker on the table alone. It can't score guesses against the
truth, but it still reports which small groups are forced.

## 4. Protect it

Compose generalization, differential privacy and suppression, and audit the result:

```bash
countguard protect cohort.csv -o dp.csv \
  --merge "race:Asian=Other race" --merge "race:American Indian or Alaska Native=Other race" \
  --merge "race:Native Hawaiian or Other Pacific Islander=Other race" \
  --dp-rho 1 --k 10 --audit --format yaml
```

## 5. Explore the tradeoff

```bash
countguard frontier cohort.csv --report frontier.json
```

This sweeps suppression, round-to-k, generalization and DP settings, and reports risk and loss
for each one, the Pareto set, and the setting closest to the targets.

## In Python

```python
from countguard import Cube, DiscreteGaussian, Pipeline, Suppress, assess, kidney_cohort
from countguard.synth import KIDNEY_VARS, LEVELS

cube = Cube.from_records(kidney_cohort(285, seed=2), KIDNEY_VARS, LEVELS)
privacy, attack = assess(cube, Suppress(10).apply(cube), threat="known_support")
print(privacy.risk, privacy.certain)  # 0.42..., 72

release = Pipeline(noise=DiscreteGaussian(rho=1.0), threshold=Suppress(10)).apply(cube)
```
