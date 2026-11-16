"""Privacy and utility metrics.

Re-identification risk
    Share of ALL cohort patients who sit in a small group (true atomic count
    1-9) whose exact size the attacker gets right, minus the share the luck
    floor gets right. The luck floor guesses one fixed value for every small
    group, the value that would score best in hindsight (usually 1).
    Attacker's guess: exact value when min = max; otherwise the better (in
    hindsight, per release) of two simple rules: the midpoint of [min, max],
    or the smallest value >= 1 in it. Under DP: the released (noisy,
    consistent) count, rounded, as is or raised to at least 1, whichever
    scores better. (Changed 2026-11-14: the midpoint / plain-rounding rules
    alone lost to the luck floor and showed DP as exactly 0.)
Certain recoveries
    Small groups whose exact size is mathematically forced (min = max).
    Always 0 under DP.
Small group
    Fixed at true count 1-9 (k=10 in these functions) whatever the release's
    own threshold, so rules with different k are scored on the same patients.
Accuracy loss
    Median relative error over ~500 fixed feasibility queries (1-3
    conditions). A withheld answer is read as the midpoint of 1..k-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attack import AttackResult
from .cube import Cube, Schema
from .mechanisms import Release

TARGET_RISK = 0.01  # UH3 milestone: re-identification risk < 1%
TARGET_LOSS = 0.05  # UH3 milestone: accuracy loss < 5%


@dataclass
class Privacy:
    """Re-identification scores of one release (see the module docstring for definitions)."""

    risk: float  # raw - floor, clipped at 0
    raw: float  # share of patients in correctly-guessed small groups
    floor: float  # luck floor share
    floor_value: int  # the fixed guess the luck floor uses
    certain: int  # small groups forced exactly
    small_groups: int
    patients_in_small: int
    correct_groups: int
    strategy: str = ""  # which guessing rule scored best (see best_privacy)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def luck_floor(x: np.ndarray, k: int = 10) -> tuple[float, int]:
    """Share of all patients a blind attacker sizes correctly by guessing one fixed value
    (the best in hindsight) for every small group, and that value."""
    small = (x >= 1) & (x < k)
    n = x.sum()
    best, val = 0.0, 1
    for v in range(1, k):
        s = x[small & (x == v)].sum() / n
        if s > best:
            best, val = float(s), v
    return best, val


def privacy(x: np.ndarray, guess: np.ndarray, exact: np.ndarray | None = None, k: int = 10) -> Privacy:
    """Score one point guess per atomic cell against the true counts x."""
    small = (x >= 1) & (x < k)
    n = x.sum()
    hit = small & (guess == x)
    raw = float(x[hit].sum() / n)
    floor, fv = luck_floor(x, k)
    certain = int((small & exact).sum()) if exact is not None else 0
    return Privacy(
        max(0.0, raw - floor), raw, floor, fv, certain, int(small.sum()), int(x[small].sum()), int(hit.sum())
    )


def best_privacy(x: np.ndarray, guesses: dict, exact: np.ndarray | None = None, k: int = 10) -> Privacy:
    """Credit the attacker with whichever simple guessing rule scores better on this release
    (an auditor's view; the luck floor is a hindsight-best fixed guess in the same spirit)."""
    best: Privacy | None = None
    for name, g in guesses.items():
        p = privacy(x, g, exact, k)
        p.strategy = name
        if best is None or p.raw > best.raw:
            best = p
    if best is None:
        raise ValueError("no guesses to score")
    return best


def privacy_from_attack(cube: Cube, res: AttackResult, k: int = 10) -> Privacy:
    """Privacy of an exact release from its attack result."""
    return best_privacy(cube.cells, res.guesses(), res.exact, k)


def coarse_truth(cube: Cube, rel: Release) -> np.ndarray:
    """True counts of the released atomic cells (the cube's own cells unless generalized)."""
    if rel.f2c is None:
        return cube.cells
    return np.bincount(rel.f2c, weights=cube.cells, minlength=rel.schema.n_cells).astype(np.int64)


def coarse_view(cube: Cube, rel: Release) -> tuple[Release, np.ndarray]:
    """For a generalised release: the released (coarse) cube as its own release + truth.

    Re-identification is judged at the granularity that is released: a patient
    alone in a group of released attributes is identifiable even if variables
    were dropped or merged."""
    from dataclasses import replace

    from .cube import build_A

    return replace(rel, A=build_A(rel.schema), f2c=None), coarse_truth(cube, rel)


def assess(
    cube: Cube, rel: Release, threat: str = "table_only", workers: int | None = None, k: int = 10
) -> tuple[Privacy, AttackResult | None]:
    """Privacy of any release, judged at the granularity that is released.

    Exact releases are attacked (`threat`: "known_support" or "table_only"); the
    AttackResult is returned too (over the released cells). Noisy releases have no
    hard bounds; they are scored by the DP attacker and the result is None.
    """
    from .attack import run_attack

    if not rel.exact:
        return privacy_from_dp(cube, rel, k), None
    if rel.f2c is None:
        res = run_attack(rel, threat, truth=cube.cells, workers=workers)
        return privacy_from_attack(cube, res, k), res
    crel, ctruth = coarse_view(cube, rel)
    cres = run_attack(crel, threat, truth=ctruth, workers=workers)
    return best_privacy(ctruth, cres.guesses(), cres.exact, k), cres


def dp_guesses(cells_est: np.ndarray) -> dict[str, np.ndarray]:
    """DP attacker: the released (noisy, consistent) count, rounded, as is or raised to at least 1."""
    r = np.rint(cells_est).astype(np.int64)
    return {"lean to 1": np.maximum(1, r), "released count": r}


def privacy_from_dp(cube: Cube, rel: Release, k: int = 10) -> Privacy:
    """Privacy of a noisy release: the attacker reads the consistent estimate."""
    if rel.cells_est is None:
        raise ValueError("not a noisy release (no cells_est)")
    return best_privacy(coarse_truth(cube, rel), dp_guesses(rel.cells_est), None, k)


# ---------------------------------------------------------------- utility


@dataclass
class Queries:
    schema: Schema
    conds: list[list[tuple[int, int]]]  # [(var index, level index), ...] per query
    rows: np.ndarray  # cube-row id of each query (fine schema)

    def labels(self) -> list[str]:
        """Human-readable condition of each query."""
        return [" ∧ ".join(f"{self.schema.names[v]}={self.schema.levels[v][l]}" for v, l in q) for q in self.conds]


def make_queries(cube: Cube, n_queries: int = 500, seed: int = 20261114, max_conds: int = 3) -> Queries:
    """Fixed random feasibility queries with 1-3 conditions and a non-zero true count."""
    s = cube.schema
    rng = np.random.default_rng(seed)
    conds: list[list[tuple[int, int]]] = []
    rows: list[int] = []
    seen: set[int] = set()
    tries = 0
    while len(conds) < n_queries and tries < 200 * n_queries:
        tries += 1
        m = int(rng.integers(1, max_conds + 1))
        vs = sorted(rng.choice(s.n_vars, size=m, replace=False).tolist())
        q = [(v, int(rng.integers(s.sizes[v]))) for v in vs]
        code = s.sizes.copy()
        for v, l in q:
            code[v] = l
        r = int(s.encode_rows([code])[0])
        if r in seen or cube.rows[r] < 1:
            continue
        seen.add(r)
        conds.append(q)
        rows.append(r)
    return Queries(s, conds, np.array(rows, dtype=np.int64))


def answer(rel: Release, q: Queries, fine: Schema) -> np.ndarray:
    """What a sponsor reads off the released table for each query."""
    k = rel.k or 10
    mid = k / 2  # midpoint of 1..k-1
    if rel.f2c is None:
        v = rel.shown[q.rows]
        return np.where(np.isnan(v), mid, v)
    # generalised release: map each condition to the coarse schema; split a merged
    # level / dropped variable uniformly (the sponsor has no finer information)
    cs = rel.schema
    out = np.empty(len(q.conds))
    for i, cond in enumerate(q.conds):
        code = cs.sizes.copy()
        factor = 1.0
        for v, l in cond:
            name = fine.names[v]
            if name not in cs.names:
                factor /= fine.sizes[v]
                continue
            ci = cs.names.index(name)
            lev = fine.levels[v][l]
            merge = rel.params.get("merge", {}).get(name, {})
            target = merge.get(lev, lev)
            code[ci] = cs.levels[ci].index(target)
            group = sum(1 for fl in fine.levels[v] if merge.get(fl, fl) == target)
            factor /= group
        r = int(cs.encode_rows([code])[0])
        val = rel.shown[r]
        out[i] = (mid if np.isnan(val) else val) * factor
    return out


def accuracy_loss(cube: Cube, rel: Release, q: Queries) -> dict:
    """Median and mean relative error of the released answers, and the share withheld."""
    truth = cube.rows[q.rows].astype(float)
    est = answer(rel, q, cube.schema)
    rel_err = np.abs(est - truth) / truth
    return {
        "median_rel_error": float(np.median(rel_err)),
        "mean_rel_error": float(rel_err.mean()),
        "share_withheld": float(np.mean(np.isnan(rel.shown[q.rows]))) if rel.f2c is None else None,
        "n_queries": len(q.rows),
    }


def in_box(risk: float, loss: float) -> bool:
    """Both UH3 targets met (risk < 1%, loss < 5%)."""
    return risk < TARGET_RISK and loss < TARGET_LOSS
