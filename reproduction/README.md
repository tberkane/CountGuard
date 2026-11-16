# Reproduction

`original/` is an unmodified copy (commit `6bbdb33`) of
[smart-on-fhir/cube-inference-fidelity-evaluation](https://github.com/smart-on-fhir/cube-inference-fidelity-evaluation)
(Apache-2.0, see `original/LICENSE`), the research code behind the medRxiv preprint
*Sharing Aggregated Patient Counts in Place of Line-Level EHR Data* (doi:10.64898/2026.08.18.26359984).

All inputs here are **synthetic cohorts** from `countguard.synth`; no patient data is used.
`tests/test_reproduction.py` checks that CountGuard's bounds match the original code's, cell for cell,
on the committed synthetic cohort.

```bash
python reproduction/run_original.py   reproduction/data/synthetic_kidney_n285_seed2.csv runs/original_seed2
python reproduction/run_countguard.py reproduction/data/synthetic_kidney_n285_seed2.csv runs/countguard_seed2
python reproduction/build_table.py    # -> results/reproduction.json
```

| File | |
|---|---|
| `data/synthetic_kidney_n285_seed2.csv` | the synthetic cohort both pipelines run on |
| `run_original.py` | runs the original pipeline unchanged (own environment, the original's pinned dependencies); adds only timers, a fixed seed and an arm64 CBC path (`cbc_shim.py`) |
| `run_countguard.py` | CountGuard's version of the same pipeline |
| `evaluate_mae.py` | scores both with the original evaluation code |
| `build_table.py` | merges the runs into `results/reproduction.json` |
| `results/` | small committed outputs used by `tests/test_reproduction.py` |
