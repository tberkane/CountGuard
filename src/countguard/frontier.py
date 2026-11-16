"""Privacy-utility frontier: sweep mechanisms, find the Pareto set, recommend a policy."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import zlib
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .cube import Cube, Schema
from .mechanisms import RACE_COARSE, RACE_WHITE_OTHER, DiscreteGaussian, Generalize, Mechanism, Round, Suppress
from .metrics import TARGET_LOSS, TARGET_RISK, Queries, accuracy_loss, assess, in_box, make_queries, privacy_from_dp

DP_REPS = 20


def default_grid(schema: Schema | None = None) -> list[tuple[str, Mechanism]]:
    """The default sweep as (group, mechanism) pairs: suppression and round-to-k for
    k in 5..50, DP at seven budgets, and (for the synthetic kidney schema, which has the
    race levels it merges) the generalization ladder."""
    g: list[tuple[str, Mechanism]] = []
    ks = (5, 10, 15, 20, 30, 50)
    for k in ks:
        g.append(("suppression", Suppress(k)))
    for k in ks:  # round every count to a multiple of the threshold
        g.append(("rounding", Round(k, k=k)))
    if schema is None or _has_kidney_race(schema):
        for m in generalization_ladder():
            g.append(("generalization", m))
    for rho in (0.05, 0.1, 0.25, 0.5, 1, 2, 5):
        g.append(("dp", DiscreteGaussian(rho, 3)))
    return g


def _has_kidney_race(schema: Schema) -> bool:
    if "race" not in schema.names:
        return False
    return set(RACE_WHITE_OTHER["race"]) <= set(schema.levels[schema.names.index("race")])


def generalization_ladder(k: int = 10) -> list[Generalize]:
    """Coarser at every step, all with suppression below k: race 6 → 4 → 2 levels → removed, then
    sex removed too. benchmarks/search_generalization.py checks every combination of race level and keep-or-drop
    for sex and each drug; none reaches the UH3 targets."""
    return [
        Generalize(merge=RACE_COARSE, k=k, name="race → 4", short="race → 4"),
        Generalize(merge=RACE_WHITE_OTHER, k=k, name="race → 2", short="race → 2"),
        Generalize(drop=("race",), k=k, name="no race", short="no race"),
        Generalize(drop=("race", "sex"), k=k, name="no race, sex", short="+ no sex"),
    ]


def policy_text(m: Mechanism) -> str:
    """One-sentence plain-language description of a mechanism."""
    if isinstance(m, Suppress):
        return f"Publish every cube row; withhold any count below {m.k}."
    if isinstance(m, Round):
        return f"Withhold counts below {m.k}; round every published count to the nearest multiple of {m.base}."
    if isinstance(m, Generalize):
        bits = []
        if m.merge:
            levels = (
                "White / Black / Other race / missing" if m.merge == RACE_COARSE else "White / not White or unknown"
            )
            bits.append(f"merge race into {levels}")
        if m.drop:
            bits.append("leave out " + " and ".join(m.drop))
        return f"{'; '.join(bits).capitalize()}; withhold counts below {m.k}."
    if isinstance(m, DiscreteGaussian):
        return (
            f"Add discrete Gaussian noise (OpenDP, ρ = {m.rho:g} zCDP, σ ≈ {{sigma:.1f}}) to every table of up to "
            f"{m.max_order} variables, then publish the consistent non-negative cube fitted to them. "
            "No suppression needed."
        )
    return m.label


def _eval_exact(cube: Cube, m: Mechanism, q: Queries, threat: str, workers: int | None) -> dict:
    rel = m.apply(cube)
    p, res = assess(cube, rel, threat, workers=workers)
    a = accuracy_loss(cube, rel, q)
    return p.as_dict() | {
        "seconds": res.timings["total"] if res is not None else 0.0,
        "loss": a,
        "judged_on": "released groups" if rel.f2c is not None else "atomic cells",
    }


def _eval_dp_rep(args: tuple) -> tuple[dict, dict, dict]:
    cube, m, q = args
    rel = m.apply(cube)
    p = privacy_from_dp(cube, rel)
    a = accuracy_loss(cube, rel, q)
    return p.as_dict(), a, {"sigma": rel.params["sigma"], "marginals": rel.params["marginals"]}


def _params(m: Mechanism) -> dict:
    return {k: v for k, v in getattr(m, "__dict__", {}).items() if k in ("k", "base", "drop", "short", "display")}


def sweep(
    cube: Cube,
    grid: list[tuple[str, Mechanism]] | None = None,
    reps: int = DP_REPS,
    seed: int | None = 0,
    threat: str = "known_support",
    workers: int | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Audit every mechanism in `grid` on `cube` and return the privacy-utility frontier.

    Exact mechanisms are attacked once (`threat` model). Each DP mechanism is run `reps`
    times and summarised by the median risk and loss; with an integer `seed` every draw
    is seeded (from `seed`, the mechanism's label and the repetition), so a rerun, or a
    rerun of one family only, gives the same numbers. seed=None uses OpenDP's unseedable secure sampler.

    Returns a JSON-ready dict: `points` (one per mechanism, with risk, loss, Pareto flag
    and whether both UH3 targets are met), `recommended`, and run metadata.
    """
    grid = grid or default_grid(cube.schema)
    q = make_queries(cube)
    nw = workers or max(1, min(os.cpu_count() or 1, 12))
    points = []
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    for gi, (group, m) in enumerate(grid):
        t = time.perf_counter()
        pt: dict = {"id": f"p{gi}", "label": m.label, "family": m.family, "group": group}
        if isinstance(m, DiscreteGaussian):
            seeds = [
                None
                if seed is None
                else int(np.random.SeedSequence([seed, zlib.crc32(m.label.encode()), r]).generate_state(1)[0])
                for r in range(reps)
            ]
            jobs = [(cube, replace(m, seed=sd), q) for sd in seeds]
            with ctx.Pool(min(nw, reps)) as pool:
                rs = pool.map(_eval_dp_rep, jobs)
            risks = np.array([r[0]["risk"] for r in rs])
            losses = np.array([r[1]["median_rel_error"] for r in rs])
            p0, _, acct = rs[int(np.argsort(losses)[len(losses) // 2])]
            pt |= {
                "risk": float(np.median(risks)),
                "loss": float(np.median(losses)),
                "raw": float(np.median([r[0]["raw"] for r in rs])),
                "floor": p0["floor"],
                "floor_value": p0["floor_value"],
                "strategy": p0["strategy"],
                "certain": 0,
                "reps": [[float(a), float(b)] for a, b in zip(risks, losses, strict=True)],
                "params": {
                    "rho": m.rho,
                    "max_order": m.max_order,
                    "sigma": acct["sigma"],
                    "marginals": acct["marginals"],
                }
                | ({} if seed is None else {"seed": seed}),
                "policy": policy_text(m).format(sigma=acct["sigma"]),
            }
        else:
            ks = _eval_exact(cube, m, q, threat, max(1, nw // 3))
            pt |= {
                "risk": ks["risk"],
                "loss": ks["loss"]["median_rel_error"],
                "raw": ks["raw"],
                "floor": ks["floor"],
                "floor_value": ks["floor_value"],
                "certain": ks["certain"],
                "strategy": ks["strategy"],
                "params": _params(m),
                "policy": policy_text(m),
                "attack_seconds": ks["seconds"],
                "judged_on": ks["judged_on"],
                "small_groups": ks["small_groups"],  # denominator for "certain", at the judged granularity
            }
        pt["in_box"] = in_box(pt["risk"], pt["loss"])
        points.append(pt)
        if log:
            log(
                f"  {m.label:42s} risk {100 * pt['risk']:6.2f}%  loss {100 * pt['loss']:6.2f}%  "
                f"{'IN BOX' if pt['in_box'] else ''}  ({time.perf_counter() - t:.1f}s)"
            )
    mark_pareto(points)
    rec = recommend(points)
    return {
        "cohort": cube.summary(),
        "points": points,
        "recommended": rec,
        "n_queries": len(q.rows),
        "targets": {"risk": TARGET_RISK, "loss": TARGET_LOSS},
        "seconds": time.perf_counter() - t0,
        "threat": threat,
        "dp_reps": reps,
        "dp_seed": seed,
    }


def mark_pareto(points: list[dict]) -> None:
    """Pareto-optimal = no other point is at least as good on both axes and better on one."""
    for p in points:
        p["pareto"] = not any(
            (o["risk"] <= p["risk"] and o["loss"] <= p["loss"]) and (o["risk"] < p["risk"] or o["loss"] < p["loss"])
            for o in points
            if o is not p
        )


def recommend(points: list[dict], risk_target: float = TARGET_RISK, loss_target: float = TARGET_LOSS) -> dict:
    """Lowest accuracy loss inside the target box (ties: lower risk). If nothing is inside,
    the point closest to the box, flagged as missing it."""
    inside = [p for p in points if p["risk"] < risk_target and p["loss"] < loss_target]
    if inside:
        best = min(inside, key=lambda p: (p["loss"], p["risk"]))
        return {"id": best["id"], "in_box": True}
    gap = lambda p: max(0.0, p["risk"] / risk_target - 1) + max(0.0, p["loss"] / loss_target - 1)
    best = min(points, key=gap)
    return {
        "id": best["id"],
        "in_box": False,
        "note": "No setting meets both targets on this cohort; this is the closest one.",
    }
