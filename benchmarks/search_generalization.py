"""Search every coarsening the generalization ladder draws from (k = 10): race at 6 / 4 / 3 / 2 levels or
dropped, times keep-or-drop for sex and each of the 7 drugs = 1,280 rules per cohort. Writes
runs/generalization_lattice_n{N}.json and prints the Pareto set.

    python benchmarks/search_generalization.py
"""

import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from countguard import KIDNEY_VARS, Cube, Generalize, kidney_cohort
from countguard.metrics import accuracy_loss, assess, make_queries
from countguard.synth import LEVELS

O = "Other race"
RACE = {
    "6": {},
    "4": {"race": {"Asian": O, "American Indian or Alaska Native": O, "Native Hawaiian or Other Pacific Islander": O}},
    "3": {
        "race": {
            "Asian": O,
            "American Indian or Alaska Native": O,
            "Native Hawaiian or Other Pacific Islander": O,
            "Black or African American": O,
        }
    },
    "2": {"race": {k: "Not White / unknown" for k in LEVELS["race"] if k != "White"}},
    "0": None,
}
DROPPABLE = ["sex", "cni", "antimetabolite", "steroid", "mtor", "belatacept", "atg", "il2ra"]


def ev(args):
    cube, q, r, drops = args
    merge = RACE[r] or {}
    drop = tuple(drops) + (("race",) if RACE[r] is None else ())
    rel = Generalize(merge=merge, drop=drop, k=10).apply(cube)
    p, _ = assess(cube, rel, "known_support", workers=1)
    a = accuracy_loss(cube, rel, q)
    return {
        "race": r,
        "drop": list(drops),
        "cells": int(rel.schema.n_cells),
        "risk": p.risk,
        "loss": a["median_rel_error"],
        "mean": a["mean_rel_error"],
        "certain": p.certain,
        "small": p.small_groups,
    }


if __name__ == "__main__":
    S = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "runs")
    for n in (285, 20000):
        t = time.time()
        cube = Cube.from_records(kidney_cohort(n, 2), KIDNEY_VARS, LEVELS)
        q = make_queries(cube)
        jobs = [
            (cube, q, r, d)
            for r in RACE
            for m in range(len(DROPPABLE) + 1)
            for d in itertools.combinations(DROPPABLE, m)
        ]
        with mp.get_context("spawn").Pool(10) as pool:
            res = pool.map(ev, jobs, chunksize=8)
        par = [
            a
            for a in res
            if not any(
                b["risk"] <= a["risk"] and b["loss"] <= a["loss"] and (b["risk"] < a["risk"] or b["loss"] < a["loss"])
                for b in res
            )
        ]
        par.sort(key=lambda a: a["loss"])
        json.dump(res, open(f"{S}/generalization_lattice_n{n}.json", "w"))
        print(
            f"n={n}: {len(res)} coarsenings in {time.time() - t:.0f}s; in box: {sum(a['risk'] < 0.01 and a['loss'] < 0.05 for a in res)}"
        )
        for a in par:
            print(
                f"  race {a['race']} drop {','.join(a['drop']) or '-':45s} cells {a['cells']:5d} risk {100 * a['risk']:5.2f}% loss {100 * a['loss']:5.1f}% (mean {100 * a['mean']:6.1f}%) certain {a['certain']}/{a['small']}"
            )
