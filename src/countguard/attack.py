"""Reconstruction attack: tightest [min, max] for every atomic cell.

Pipeline (each stage is a sound deduction, so the final bounds equal the
integer-programming bounds of the preprint's attack):

1. Interval propagation ("subtraction pass"). Every cube row is the sum of its
   children along any rolled-up variable, so [lo, hi] ranges propagate up and
   down the lattice. Pure numpy; pins most cells with no solver at all.
2. One HiGHS LP per worker, built once; only the objective changes per cell
   (simplex hot-starts). LP bounds, rounded inward, are valid integer bounds.
3. Auditor shortcut: the data owner knows a feasible integer point (the
   truth), so if ceil(LP min) equals the truth the integer min is settled
   without branching; same for the max. Only the rest go to the MIP.
4. MIP (HiGHS) per remaining cell, with a time limit; on timeout the MIP's
   dual bound is still a valid outer bound and the cell is flagged.

Threat models
-------------
known_support  The attacker knows which atomic cells are non-empty (the
               preprint's code: its cells are the observed combinations).
               Unknowns: at most n.  Empty cells are known zeros.
table_only     The attacker has only the released table. Unknowns: every
               atomic cell (3,072 for the 10-variable kidney cube).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .mechanisms import Release

THREATS = ("known_support", "table_only")
_EPS = 1e-6


class InfeasibleRelease(ValueError):
    """No set of counts is consistent with the table (e.g. a published row was deleted or edited)."""

    def __init__(self, n_rows: int):
        self.n_rows = n_rows
        super().__init__(
            "the table is internally inconsistent: no set of patient counts matches all of its "
            "rows (was a published row deleted or edited?)"
        )


# ---------------------------------------------------------------- propagation


def propagate(schema, lo: np.ndarray, hi: np.ndarray, max_sweeps: int = 200):
    """Tighten row ranges to a fixed point using parent = sum(children)."""
    shape = tuple(int(s) + 1 for s in schema.sizes)
    L = lo.astype(np.int64).reshape(shape).copy()
    U = hi.astype(np.int64).reshape(shape).copy()
    np.maximum(L, 0, out=L)
    for sweep in range(max_sweeps):  # noqa: B007 (sweep count is returned)
        changed = False
        for ax in range(L.ndim):
            l = np.moveaxis(L, ax, -1)  # views: writes go through to L/U
            u = np.moveaxis(U, ax, -1)
            cl, cu = l[..., :-1], u[..., :-1]
            pl, pu = l[..., -1], u[..., -1]
            scl, scu = cl.sum(-1), cu.sum(-1)
            npl, npu = np.maximum(pl, scl), np.minimum(pu, scu)
            ncl = np.maximum(cl, npl[..., None] - (scu[..., None] - cu))
            ncu = np.minimum(cu, npu[..., None] - (scl[..., None] - cl))
            if (npl != pl).any() or (npu != pu).any() or (ncl != cl).any() or (ncu != cu).any():
                changed = True
                pl[...] = npl
                pu[...] = npu
                cl[...] = ncl
                cu[...] = ncu
        if not changed:
            break
    L, U = L.reshape(-1), U.reshape(-1)
    if (L > U).any():
        raise InfeasibleRelease(int((L > U).sum()))
    return L, U, sweep + 1


# ---------------------------------------------------------------- results


@dataclass
class AttackResult:
    """Proven bounds on every atomic cell (one full combination of variable levels).

    `L[j] == U[j]` means the attacker knows cell j's count exactly. `method` records how each
    cell was settled (see `METHODS`); `in_model` marks the cells the attacker treats as unknown.
    """

    threat: str
    L: np.ndarray  # (J,) lower bound per atomic cell
    U: np.ndarray  # (J,) upper bound per atomic cell
    method: np.ndarray  # (J,) how each cell was settled
    in_model: np.ndarray  # (J,) bool: cell is an unknown for this attacker
    timings: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    @property
    def exact(self) -> np.ndarray:
        return self.L == self.U

    def guess(self) -> np.ndarray:
        """Attacker's point guess: the value if forced, else the smallest value >= 1 it can't rule
        out (small groups are mostly singletons, so leaning to 1 beats the midpoint)."""
        return np.where(self.exact, self.L, np.minimum(np.maximum(self.L, 1), self.U)).astype(np.int64)

    def guesses(self) -> dict:
        """The two simple guessing rules; the audit credits the attacker with the better one."""
        mid = np.where(self.exact, self.L, np.floor((self.L + self.U) / 2 + 0.5)).astype(np.int64)
        return {"lean to 1": self.guess(), "midpoint": mid}


METHODS = ("known-zero", "propagation", "lp", "lp+truth", "mip", "mip-timeout", "split")


# ---------------------------------------------------------------- LP / MIP worker

_W: dict = {}


def _worker_init(payload):
    import highspy

    n, Ac, Lc, Uc, rlo, rhi, xhat, tl = payload
    _W.update(n=n, Lc=Lc, Uc=Uc, xhat=xhat, tl=tl, payload=payload, mip=None)
    _W["lp"] = _build(highspy, n, Ac, Lc, Uc, rlo, rhi, integer=False)


def _new_highs(highspy):
    """A silent HiGHS instance. Its constructor prints a banner on the C-level stdout before
    any option can be set; send that to stderr so stdout (e.g. a JSON report) stays clean."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        h = highspy.Highs()
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    h.setOptionValue("output_flag", False)
    return h


def _build(highspy, n, Ac, Lc, Uc, rlo, rhi, integer, time_limit=None):
    h = _new_highs(highspy)
    h.setOptionValue("threads", 1)
    if time_limit:
        h.setOptionValue("time_limit", float(time_limit))
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = Ac.shape[0]
    lp.col_cost_ = np.zeros(n)
    lp.col_lower_ = Lc.astype(float)
    lp.col_upper_ = Uc.astype(float)
    lp.row_lower_ = rlo.astype(float)
    lp.row_upper_ = rhi.astype(float)
    csc = Ac.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data.astype(float)
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h.passModel(lp)
    return h


def _solve(h, j, sense, highspy):
    h.changeColCost(j, 1.0)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize if sense < 0 else highspy.ObjSense.kMaximize)
    h.run()
    st = h.getModelStatus()
    info = h.getInfo()
    val = info.objective_function_value
    h.changeColCost(j, 0.0)
    return st, val, info


def _bound_cell(j):
    """LP (+truth shortcut) then MIP for one free cell. Returns (j, lo, hi, method)."""
    import highspy

    h = _W["lp"]
    st1, vmin, _ = _solve(h, j, -1, highspy)
    st2, vmax, _ = _solve(h, j, +1, highspy)
    ok = highspy.HighsModelStatus.kOptimal
    lo = math.ceil(vmin - _EPS) if st1 == ok else int(_W["Lc"][j])
    hi = math.floor(vmax + _EPS) if st2 == ok else int(_W["Uc"][j])
    if lo == hi:
        return j, lo, hi, "lp"
    xt = _W["xhat"][j]
    need_min, need_max = lo < xt, hi > xt
    if not need_min and not need_max:
        return j, lo, hi, "lp+truth"
    if _W["mip"] is None:
        n, Ac, Lc, Uc, rlo, rhi, _, tl = _W["payload"]
        _W["mip"] = _build(highspy, n, Ac, Lc, Uc, rlo, rhi, integer=True, time_limit=tl)
    m = _W["mip"]
    timeout = False
    v: int | None
    for sense, need in ((-1, need_min), (+1, need_max)):
        if not need:
            continue
        st, val, info = _solve(m, j, sense, highspy)
        if st == ok:
            v = int(round(val))
        else:  # time limit: dual bound is still a valid outer bound
            timeout = True
            db = info.mip_dual_bound
            v = (math.ceil(db - _EPS) if sense < 0 else math.floor(db + _EPS)) if np.isfinite(db) else None
            if v is None:
                v = lo if sense < 0 else hi
        if sense < 0:
            lo = max(lo, v)
        else:
            hi = min(hi, v)
    return j, lo, hi, ("mip-timeout" if timeout else "mip")


# ---------------------------------------------------------------- driver


def _dominated_absent(schema, rlo, rhi):
    """Rows whose range is implied by a parent with an equal-or-tighter upper bound."""
    codes = schema.row_codes()
    sizes, rs = schema.sizes, schema.row_strides()
    drop = np.zeros(len(rlo), dtype=bool)
    cand = rlo == 0
    for i in range(schema.n_vars):
        g = codes[:, i] < sizes[i]
        parent = np.where(g, np.arange(len(rlo)) + (sizes[i] - codes[:, i]) * rs[i], 0)
        drop |= cand & g & (rhi[parent] <= rhi)
    return drop


def attack(
    rel: Release,
    threat: str = "table_only",
    truth: np.ndarray | None = None,
    workers: int | None = None,
    mip_time_limit: float = 30.0,
) -> Iterator[dict]:
    """Run the attack, yielding progress events. The last event has type 'done'
    and carries an AttackResult under 'result'.

    truth: true atomic counts (the auditor's knowledge; required for
    known_support, optional for table_only where a feasible point is found
    by MIP if absent).
    """
    if threat not in THREATS:
        raise ValueError(f"threat must be one of {THREATS}")
    if rel.f2c is not None:
        yield from _attack_generalized(rel, threat, truth, workers, mip_time_limit)
        return
    if not rel.exact:
        raise ValueError("attack bounds apply to exact releases; noisy (DP) releases have no hard ranges")
    t0 = time.perf_counter()
    A: sp.csr_matrix = rel.A
    R, J = A.shape
    same_schema = rel.schema.n_cells == J
    rlo, rhi = rel.lo.astype(np.int64).copy(), rel.hi.astype(np.int64).copy()
    in_model = np.ones(J, dtype=bool)
    if threat == "known_support":
        if truth is None:
            raise ValueError("known_support needs the true support (truth)")
        in_model = truth > 0

    if not same_schema:
        raise ValueError("release schema does not match the atomic cells (missing f2c map)")

    # 1. propagation on the released lattice
    atom = rel.schema.cell_to_row()
    if threat == "known_support":
        rlo[atom[~in_model]] = 0
        rhi[atom[~in_model]] = 0
    rlo, rhi, sweeps = propagate(rel.schema, rlo, rhi)
    Lcell, Ucell = rlo[atom], rhi[atom]
    method = np.full(J, "", dtype=object)
    method[~in_model] = "known-zero"
    settled = Lcell == Ucell
    method[settled & in_model] = "propagation"
    free = np.flatnonzero(~settled)
    t1 = time.perf_counter()
    yield {
        "type": "propagation",
        "threat": threat,
        "n_cells": J,
        "n_unknowns": int(in_model.sum()),
        "settled": int((settled & in_model).sum()),
        "free": int(len(free)),
        "sweeps": sweeps,
        "seconds": t1 - t0,
        "L": Lcell.copy(),
        "U": Ucell.copy(),
    }

    timings = {"propagation": t1 - t0}
    if len(free):
        # 2. LP over the free cells; fixed cells moved to the row bounds
        fixed = np.flatnonzero(settled)
        contrib = A[:, fixed] @ Lcell[fixed] if len(fixed) else np.zeros(R)
        Af = A[:, free].tocsr()
        keep = np.diff(Af.indptr) > 0
        keep &= ~_dominated_absent(rel.schema, rlo, rhi)
        lo2, hi2 = rlo - contrib, rhi - contrib
        Lf, Uf = Lcell[free], Ucell[free]
        imp = (Af @ Lf >= lo2) & (Af @ Uf <= hi2)  # row implied by its column bounds
        keep &= ~imp
        Af, lo2, hi2 = Af[keep], lo2[keep], hi2[keep]
        Uf = np.minimum(Uf, 10**9)
        if truth is not None:
            xhat = truth[free].astype(np.int64)
        else:
            xhat = _feasible_point(Af, Lf, Uf, lo2, hi2)
        payload = (len(free), Af, Lf, Uf, lo2, hi2, xhat, mip_time_limit)
        yield {"type": "model", "rows": int(Af.shape[0]), "cols": int(len(free)), "nnz": int(Af.nnz)}

        nw = workers or max(1, min(os.cpu_count() or 1, 16))
        nw = min(nw, max(1, len(free) // 8))
        tl = time.perf_counter()
        if nw == 1:
            _worker_init(payload)
            it: Iterator[tuple] = map(_bound_cell, range(len(free)))
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(nw, initializer=_worker_init, initargs=(payload,))
            it = pool.imap_unordered(_bound_cell, range(len(free)), chunksize=max(1, len(free) // (nw * 16)))
        try:
            done = 0
            for jj, lo, hi, how in it:
                j = free[jj]
                Lcell[j], Ucell[j], method[j] = lo, hi, how
                done += 1
                yield {
                    "type": "cell",
                    "cell": int(j),
                    "L": int(lo),
                    "U": int(hi),
                    "method": how,
                    "done": done,
                    "total": len(free),
                }
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        timings["solve"] = time.perf_counter() - tl
        timings["workers"] = nw
    timings["total"] = time.perf_counter() - t0
    counts = {m: int((method == m).sum()) for m in METHODS}
    res = AttackResult(threat, Lcell, Ucell, method, in_model, timings, counts)
    yield {"type": "done", "result": res, "seconds": timings["total"]}


def _attack_generalized(rel, threat, truth, workers, mip_time_limit):
    """Generalised release: every released row is a union of coarse cells, so attack
    the coarse cube, then split each coarse cell's [L, U] exactly over its fine cells
    (a fine cell alone in its coarse cell inherits [L, U]; otherwise [0, U])."""
    from dataclasses import replace

    from .cube import build_A

    f2c = rel.f2c
    J, Jc = len(f2c), rel.schema.n_cells
    crel = replace(rel, A=build_A(rel.schema), f2c=None)
    ctruth = None if truth is None else np.bincount(f2c, weights=truth, minlength=Jc).astype(np.int64)
    cres = None
    for ev in attack(crel, threat, ctruth, workers=workers, mip_time_limit=mip_time_limit):
        if ev["type"] == "done":
            cres = ev["result"]
        elif ev["type"] != "cell":  # coarse-cell progress is not meaningful per fine cell
            yield ev
    if cres is None:
        raise RuntimeError("attack ended without a result")
    in_model = (truth > 0) if threat == "known_support" and truth is not None else np.ones(J, dtype=bool)
    Ui = np.where(in_model, cres.U[f2c], 0)
    m = np.bincount(f2c, weights=in_model, minlength=Jc)[f2c]
    L = np.where(in_model & (m == 1), cres.L[f2c], 0).astype(np.int64)
    method = np.where(~in_model, "known-zero", np.where(m == 1, cres.method[f2c], "split")).astype(object)
    counts = {k: int((method == k).sum()) for k in METHODS}
    res = AttackResult(threat, L, Ui.astype(np.int64), method, in_model, dict(cres.timings), counts)
    yield {"type": "done", "result": res, "seconds": cres.timings["total"]}


def run_attack(rel: Release, threat: str = "table_only", truth: np.ndarray | None = None, **kw: Any) -> AttackResult:
    """Run the reconstruction attack on an exact release and return the bounds.

    Args:
        rel: the release (from a mechanism, or `read_table`).
        threat: "known_support" (attacker knows which combinations exist; needs `truth`)
            or "table_only" (attacker has only the table).
        truth: true atomic counts, known to the auditor. Speeds up the solver and is
            required for known_support; never used to tighten the bounds themselves.
        **kw: `workers` (solver processes) and `mip_time_limit` (seconds per cell).
    """
    for ev in attack(rel, threat, truth, **kw):
        if ev["type"] == "done":
            return ev["result"]
    raise RuntimeError("attack ended without a result")


def _feasible_point(Af, Lf, Uf, lo, hi) -> np.ndarray:
    import highspy

    h = _build(highspy, Af.shape[1], Af, Lf, Uf, lo, hi, integer=True, time_limit=120)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError("could not find a feasible integer point for the release")
    return np.rint(np.array(h.getSolution().col_value)).astype(np.int64)
