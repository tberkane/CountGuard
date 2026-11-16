# Command-line interface

```text
countguard {synth, audit, protect, frontier} [options]
```

Every command prints a report to stdout as JSON (default) or YAML (`--format yaml`), or writes it
to `--report FILE`. Each report starts with a **provenance** block:

```json
"provenance": {
  "tool": "countguard", "version": "0.1.0", "command": "audit",
  "created": "2026-11-16T21:16:25+00:00",
  "inputs": [{"path": "cube.csv", "sha256": "633f5ab4…"}],
  "parameters": {"k": 10, "threat": "known_support", "vars": ["race", "sex", …]}
}
```

**Inputs.** A *count table* is a Cumulus-style table (`.csv` or `.parquet`): a `cnt` column plus
one column per variable, with empty values meaning "all". A *cohort* is a patient-level table:
one row per patient, one column per variable, and an optional `patient_ref` column. Empty values
become `cumulus__none` (missing).

## `countguard synth`

Write a synthetic kidney-transplant-like cohort. No real data is involved.

| Option | |
|---|---|
| `-n N` | patients (default 285) |
| `--seed S` | generator seed (default 2) |
| `--vars {all,8,6}` | all 10 variables, or the 8- or 6-variable subcube |
| `-o FILE` | output `.csv` or `.parquet` |

## `countguard audit TABLE`

Attack a released count table.

| Option | |
|---|---|
| `--k K` | threshold the table was released with (default 10) |
| `--cohort FILE` | patient-level source: enables the risk score and the `known_support` attacker |
| `--vars a,b,c` | variables to read from the cohort (default: the table's columns) |
| `--threat` | `known_support` (default with `--cohort`) or `table_only` |
| `--bounds` | include the proven [min, max] of every atomic cell |
| `--workers N` | solver processes |

The report contains `table`, `attack` (cells exactly determined, and how each cell was settled),
and `exposure`: small groups (1–9) whose size is forced, and how many patients they hold.
Exposure needs only the table. With `--cohort` it also contains `risk` (see
[Concepts](concepts.md)).

Without `--cohort`, levels are read from the table. A level whose rows are all suppressed never
appears. When a variable's published one-way counts can't add up to the total, CountGuard adds
one placeholder level for the hidden ones (`countguard__unseen`, listed under
`table.unseen_levels`). Pass `--cohort` for an exact audit.

## `countguard protect COHORT -o TABLE`

Release a protected count table. Steps run in order and each is optional:

| Step | Options |
|---|---|
| 1. generalize | `--merge VAR:OLD=NEW` (repeatable), `--drop VAR` (repeatable) |
| 2. differential privacy | `--dp-rho R`, `--dp-order 3`, `--seed S` (benchmarks only) |
| 3. suppress / round | `--k K` (default 10 unless `--dp-rho`), `--round BASE`, `--show-range` ("1-9") |

`--audit` adds risk and accuracy loss for the release (`--threat` picks the attacker for exact
releases).

## `countguard frontier COHORT`

Sweep suppression (k = 5…50), round-to-k, DP (ρ = 0.05…5) and, for the synthetic kidney
schema, the generalization ladder. Reports every point's risk and loss, the Pareto set, and the
recommended point.

| Option | |
|---|---|
| `--reps N` | DP repetitions per budget (default 5) |
| `--seed S` | DP seed (default 0; `-1` uses OpenDP's unseeded sampler) |
| `--threat` | attacker for exact rules (default `known_support`) |
| `--quiet` | no progress lines on stderr |
