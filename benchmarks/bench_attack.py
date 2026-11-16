"""Benchmark the attack on the 10-variable kidney-like cube, both threat models."""

import json
import sys
import time

from countguard import KIDNEY_VARS, SUBCUBE_6, SUBCUBE_8, Cube, Suppress, kidney_cohort
from countguard.attack import attack
from countguard.synth import LEVELS


def main():
    out = []
    for n in map(int, sys.argv[1:] or ["285", "20000"]):
        df = kidney_cohort(n, seed=0)
        for vars_ in (SUBCUBE_6, SUBCUBE_8, KIDNEY_VARS):
            cube = Cube.from_records(df, vars_, LEVELS)
            t = time.perf_counter()
            rel = Suppress(10).apply(cube)
            tb = time.perf_counter() - t
            for threat in ("known_support", "table_only"):
                model = {}
                for ev in attack(rel, threat, truth=cube.cells):
                    if ev["type"] == "propagation":
                        prop = ev
                    elif ev["type"] == "model":
                        model = ev
                    elif ev["type"] == "done":
                        r = ev["result"]
                x = cube.cells
                small = (x > 0) & (x < 10)
                row = dict(
                    n=n,
                    vars=len(vars_),
                    threat=threat,
                    build_s=round(tb, 2),
                    unknowns=int(r.in_model.sum()),
                    settled_by_propagation=prop["settled"],
                    free=prop["free"],
                    lp_rows=model.get("rows"),
                    small=int(small.sum()),
                    small_exact=int((r.exact & small).sum()),
                    methods=r.counts,
                    seconds={k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.timings.items()},
                )
                print(json.dumps(row), flush=True)
                out.append(row)
    json.dump(out, open("benchmarks/bench_attack.json", "w"), indent=1)


if __name__ == "__main__":
    main()
