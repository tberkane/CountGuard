"""Run the ORIGINAL research pipeline, unmodified, on a synthetic cohort.

Environment: conda env `cg-original` (Python 3.12, the original's pinned deps).
Only additions: (1) arm64 CBC for PuLP (cbc_shim), (2) timers wrapped around the
original's stage functions, (3) a fixed numpy seed. The pipeline itself is
`run_complete_bayesian_reconstruction` with the preprint's settings.

    python reproduction/run_original.py reproduction/data/synthetic_kidney_n285_seed2.csv runs/original_seed2
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "original"))

import cbc_shim  # noqa: E402

cbc_shim.install()
import src.infer_counts as ifc  # noqa: E402

TIMES = {}


def timed(name, fn):
    def wrap(*a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t
        print(f"[timer] {name}: {TIMES[name]:.1f}s", flush=True)
        return out

    return wrap


for name in (
    "line_level_to_powerset",
    "build_matrices",
    "ilp_bounds",
    "fit_stan_model",
    "integer_constrained_sampling",
):
    setattr(ifc, name, timed(name, getattr(ifc, name)))


def main(csv, outdir):
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    np.random.seed(0)
    os.chdir(out)  # the original writes ./reconstruction_results
    t = time.perf_counter()
    rec = ifc.run_complete_bayesian_reconstruction(
        df_line_level=df,
        demographic_cols=list(df.columns),
        suppress_threshold=9,
        iter_sampling=200,
        iter_warmup=100,
        chains=2,
        mh_iters=2000,
    )
    TIMES["total"] = time.perf_counter() - t
    summary = ifc.create_powerset_summary_table(rec)
    summary.to_csv(out / "powerset_summary.csv", index=False)
    rec["cells_df"].to_csv(out / "cells.csv", index=False)
    np.savez_compressed(
        out / "atomic.npz",
        L=rec["atomic_bounds"]["L"],
        U=rec["atomic_bounds"]["U"],
        posterior_mean=rec["atomic_posterior_samples"].mean(0),
        n_samples=rec["atomic_posterior_samples"].shape[0],
    )
    status = summary["Status"].value_counts().to_dict()
    json.dump(
        {
            "timings_s": TIMES,
            "status_counts": status,
            "n_cells": len(rec["cells_df"]),
            "total_suppressed": rec["total_suppressed"],
        },
        open(out / "result.json", "w"),
        indent=1,
    )
    with open(out / "rec.pkl", "wb") as f:
        pickle.dump({k: v for k, v in rec.items() if k not in ("results",)}, f)
    print("DONE", json.dumps({"timings_s": TIMES, "status": status}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
