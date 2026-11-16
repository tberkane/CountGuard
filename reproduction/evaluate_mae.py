"""Score both pipelines with the ORIGINAL evaluation code (env `cg-original`).

MAE over atomic cells with true count <= 9, patient bootstrap (2,000 resamples,
seed 0), exactly as `evaluate.compare_suppressed_mae` builds it, minus the CTGAN
row (not part of this comparison).

    python reproduction/evaluate_mae.py COHORT.csv runs/original_seed2 runs/countguard_seed2/estimate_original_order.npy OUT.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original"))
import src.evaluate as ev  # noqa: E402


def main(csv, orig_dir, cg_est, out):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    rec = pickle.load(open(Path(orig_dir) / "rec.pkl", "rb"))
    cells_df = rec["cells_df"]
    true_counts = ev.atomic_counts_from_line_level(df, cells_df, cols)
    mask = true_counts <= 9
    rng = np.random.default_rng(ev.DEFAULT_SEED)
    estimates = {
        "original": ev.posterior_mean_counts(rec),
        "countguard": np.load(cg_est),
        "zero": np.zeros_like(true_counts),
        "five": np.full_like(true_counts, 5.0),
        "random": rng.integers(0, 10, size=len(true_counts)).astype(float),
        # reference: the exact true counts, scored by the same code
        "perfect": true_counts.copy(),
    }
    res = ev.cell_metric_patient_bootstrap(
        df, cells_df, estimates, ev.mean_absolute_error, demographic_cols=cols, mask=mask
    )
    point = {k: float(np.abs(v[mask] - true_counts[mask]).mean()) for k, v in estimates.items()}
    outj = {
        k: {"mae": res[k]["boot_mean"], "ci": [res[k]["ci_lower"], res[k]["ci_upper"]], "plugin": point[k]}
        for k in estimates
    }
    outj["_n_cells_scored"] = int(mask.sum())
    json.dump(outj, open(out, "w"), indent=1)
    print(json.dumps(outj, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
