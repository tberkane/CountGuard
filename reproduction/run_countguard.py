"""CountGuard's version of the preprint pipeline on the same synthetic cohort.

python reproduction/run_countguard.py reproduction/data/synthetic_kidney_n285_seed2.csv runs/countguard_seed2
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from countguard import KIDNEY_VARS, Cube, Suppress
from countguard.attack import run_attack
from countguard.bayes import reconstruct
from countguard.synth import LEVELS

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def first_appearance_cells(df, schema):
    """Atomic cells in the original code's order (drop_duplicates = first appearance)."""
    codes = np.stack(
        [df[v].map({l: i for i, l in enumerate(schema.levels[k])}).to_numpy() for k, v in enumerate(schema.names)], 1
    )
    idx = (codes * schema.cell_strides()).sum(1)
    _, first = np.unique(idx, return_index=True)
    return idx[np.sort(first)]


def row_determinism(cube, res, k=10):
    """Preprint/original definition over suppressed (non-empty) cube rows:
    deterministic iff sum(L) == min(k-1, sum(U)) over the row's cells."""
    m = res.in_model
    A = cube.A
    sL = A @ np.where(m, res.L, 0)
    sU = A @ np.where(m, res.U, 0)
    sup = (cube.rows > 0) & (cube.rows < k)
    det = (sL == np.minimum(k - 1, sU)) & sup
    return int(det.sum()), int(sup.sum())


def main(csv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    T = {}
    t = time.perf_counter()
    cube = Cube.from_records(df, KIDNEY_VARS, LEVELS)
    rel = Suppress(10).apply(cube)
    T["cube"] = time.perf_counter() - t
    t = time.perf_counter()
    res = run_attack(rel, "known_support", truth=cube.cells, workers=1)
    T["bounds"] = time.perf_counter() - t
    order_cells = first_appearance_cells(df, cube.schema)
    cols = np.flatnonzero(res.in_model)
    pos = {j: i for i, j in enumerate(cols)}
    b = reconstruct(rel, res, basis="original", seed=0, col_order=np.array([pos[j] for j in order_cells]))
    T["stan"], T["mh"] = b.timings["stan"], b.timings["mh"]
    T["total"] = sum(T.values())

    x = cube.cells
    det, sup = row_determinism(cube, res)
    by_size = {int(s): [int((res.exact & (x == s)).sum()), int((x == s).sum())] for s in (1, 2, 3)}
    np.save(out / "estimate_original_order.npy", b.estimate[order_cells])
    np.savez_compressed(out / "atomic.npz", L=res.L[order_cells], U=res.U[order_cells], cells=order_cells)
    result = {
        "timings_s": T,
        "rows_published": int(rel.published.sum()),
        "rows_suppressed_nonempty": sup,
        "rows_deterministic": det,
        "small_cells_exact_by_size": by_size,
        "cells_in_model": int(res.in_model.sum()),
        "cells_exact": int((res.exact & res.in_model).sum()),
        "null_dim": b.null_dim,
        "mh_acceptance_mean": float(b.acceptance.mean()),
    }
    json.dump(result, open(out / "result.json", "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
