"""Merge the original-code and CountGuard runs into the step-2 table (reproduction/results/reproduction.json).

python reproduction/build_table.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from countguard import KIDNEY_VARS, Cube, Suppress, kidney_cohort
from countguard.attack import run_attack
from countguard.synth import LEVELS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"  # local run outputs (see README)
CSV = ROOT / "reproduction" / "data" / "synthetic_kidney_n285_seed2.csv"
OUT = ROOT / "reproduction" / "results" / "reproduction.json"

PREPRINT = {  # medRxiv 10.64898/2026.08.18.26359984, 10-variable cube (real BCH cohort)
    "rows_published": 6494,
    "rows_suppressed": 18994,
    "rows_deterministic": 14554,
    "pct_deterministic": 76.6,
    "cells_exact_by_size": {"1": [47, 55], "2": [13, 18], "3": [5, 7]},
    "mae": {"pipeline": [1.06, 0.89, 1.25]},
}


def original_by_size(df):
    cells = pd.read_csv(RUNS / "original_seed2" / "cells.csv", dtype=str, keep_default_na=False)
    a = np.load(RUNS / "original_seed2" / "atomic.npz")
    cols = list(df.columns)
    truth = df.groupby(cols).size()
    x = np.array([truth.get(tuple(r[c] for c in cols), 0) for _, r in cells.iterrows()])
    exact = a["L"] == a["U"]
    return {str(s): [int((exact & (x == s)).sum()), int((x == s).sum())] for s in (1, 2, 3)}, a


def main():
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    o = json.load(open(RUNS / "original_seed2" / "result.json"))
    c = json.load(open(RUNS / "countguard_seed2" / "result.json"))
    mae = json.load(open(RUNS / "mae_seed2.json"))
    summ = pd.read_csv(RUNS / "original_seed2" / "powerset_summary.csv")
    o_pub = int((summ["Status"] == "Published").sum())
    o_det = int((summ["Status"] == "Deterministic").sum())
    o_sup = int((summ["Status"] != "Published").sum())
    o_size, o_atomic = original_by_size(df)
    cg_atomic = np.load(RUNS / "countguard_seed2" / "atomic.npz")
    bounds_identical = bool(
        np.array_equal(o_atomic["L"], cg_atomic["L"]) and np.array_equal(o_atomic["U"], cg_atomic["U"])
    )

    # spread of deterministic recovery over other synthetic seeds (same generator)
    spread = []
    for seed in range(6):
        cube = Cube.from_records(kidney_cohort(285, seed), KIDNEY_VARS, LEVELS)
        res = run_attack(Suppress(10).apply(cube), "known_support", truth=cube.cells, workers=1)
        A, m = cube.A, res.in_model
        sL, sU = A @ np.where(m, res.L, 0), A @ np.where(m, res.U, 0)
        sup = (cube.rows > 0) & (cube.rows < 10)
        spread.append({"seed": seed, "pct": float(100 * ((sL == np.minimum(9, sU)) & sup).sum() / sup.sum())})

    ot, ct = o["timings_s"], c["timings_s"]
    table = {
        "cohort": "Synthetic kidney-transplant-like cohort, n = 285, seed 2 (not patient data)",
        "cube": "race × sex × 7 immunosuppressant classes × rejection (10 variables), k = 10",
        "seed_note": (
            "Seed 2 of 0–5 was chosen because its single-patient recovery rate is closest to the "
            "preprint's; across seeds 0–5 deterministic recovery ranges "
            f"{min(s['pct'] for s in spread):.1f}–{max(s['pct'] for s in spread):.1f}%."
        ),
        "seed_spread": spread,
        "bounds_identical": bounds_identical,
        "rows": {
            "published": {"original": o_pub, "countguard": c["rows_published"], "preprint": PREPRINT["rows_published"]},
            "suppressed": {
                "original": o_sup,
                "countguard": c["rows_suppressed_nonempty"],
                "preprint": PREPRINT["rows_suppressed"],
            },
            "deterministic": {
                "original": o_det,
                "countguard": c["rows_deterministic"],
                "preprint": PREPRINT["rows_deterministic"],
            },
            "pct_deterministic": {
                "original": 100 * o_det / o_sup,
                "countguard": 100 * c["rows_deterministic"] / c["rows_suppressed_nonempty"],
                "preprint": PREPRINT["pct_deterministic"],
            },
        },
        "cells_exact_by_size": {
            "original": o_size,
            "countguard": {k: v for k, v in c["small_cells_exact_by_size"].items()},
            "preprint": PREPRINT["cells_exact_by_size"],
        },
        "mae": mae,
        "mae_preprint": PREPRINT["mae"],
        "runtime_s": {
            "original": {
                "cube": ot.get("line_level_to_powerset", 0) + ot.get("build_matrices", 0),
                "bounds": ot.get("ilp_bounds"),
                "stan": ot.get("fit_stan_model"),
                "mh": ot.get("integer_constrained_sampling"),
                "total": ot.get("total"),
            },
            "countguard": ct,
        },
    }
    table["speedup"] = {
        k: (table["runtime_s"]["original"][k] / ct[k]) if table["runtime_s"]["original"].get(k) and ct.get(k) else None
        for k in ("cube", "bounds", "stan", "mh", "total")
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(OUT, "w"), indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
