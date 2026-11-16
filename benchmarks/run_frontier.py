"""Frontier sweep for both synthetic cohort sizes -> demo/data/frontier_n{N}.json

    python benchmarks/run_frontier.py [285 20000]
    python benchmarks/run_frontier.py --only generalization [285 20000]

--only GROUP recomputes just that group's rules and splices them into the existing JSON;
every other point is kept as is. Ids are renumbered in grid order; the Pareto set and the
recommendation are recomputed. DP draws are seeded (DP_SEED), so reruns are reproducible.

Note: the JSON shipped with v0.1.0 predates seeding; its DP points (and its four
round-to-3/5/10/20 rules) come from the 2026-11-14/16 runs.
"""

import json
import sys
import time
from pathlib import Path

from countguard import KIDNEY_VARS, Cube, kidney_cohort
from countguard.frontier import default_grid, mark_pareto, recommend, sweep
from countguard.synth import LEVELS

SEED = 2  # same synthetic cohort as the reproduction
DP_SEED = 0
OUT = Path(__file__).resolve().parents[1] / "demo" / "data"


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        del args[i : i + 2]
    OUT.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(g for g, _ in default_grid()))
    for n in map(int, args or ["285", "20000"]):
        print(f"n = {n}", flush=True)
        cube = Cube.from_records(kidney_cohort(n, SEED), KIDNEY_VARS, LEVELS)
        grid = [(g, m) for g, m in default_grid() if only is None or g == only]
        res = sweep(cube, grid, seed=DP_SEED, log=lambda s: print(s, flush=True))
        path = OUT / f"frontier_n{n}.json"
        if only:
            old = json.load(open(path))
            keep = [p for p in old["points"] if p["group"] != only]
            pts = sorted(keep + res["points"], key=lambda p: order.index(p["group"]))  # stable within a group
            for i, p in enumerate(pts):
                p["id"] = f"p{i}"
            mark_pareto(pts)
            old |= {
                "points": pts,
                "recommended": recommend(pts),
                "generated": f"{old.get('generated', '')}; {only} {time.strftime('%Y-%m-%d %H:%M')}",
            }
            res = old
        else:
            res |= {
                "n": n,
                "seed": SEED,
                "vars": KIDNEY_VARS,
                "synthetic": True,
                "generated": time.strftime("%Y-%m-%d %H:%M"),
            }
        json.dump(res, open(path, "w"))
        print(f"wrote {path.name} ({len(res['points'])} rules); recommended {res['recommended']}", flush=True)


if __name__ == "__main__":
    main()
