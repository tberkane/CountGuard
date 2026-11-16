"""Disclosure-control mechanisms. Each turns a true `Cube` into a `Release`.

A `Release` is what leaves the building, expressed the way an attacker reads
it: for every cube row of the released schema, a range [lo, hi] that the true
count is known to lie in, plus the value a consumer would see (`shown`, NaN
when the row is withheld). One attack engine therefore covers suppression,
rounding and generalization.

    published count c            -> [c, c]
    suppressed (c < k), omitted  -> [0, k-1]      (row absent from the table)
    suppressed, shown as "1-9"   -> [1, k-1]      (zero rows are then known to be 0)
    rounded to base b (r)        -> [r - floor(b/2), r + ceil(b/2) - 1]

Mechanisms compose with `Pipeline`: generalize, then add DP noise, then
suppress or round the noisy counts (post-processing, so the DP guarantee is
unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Literal, Protocol

import numpy as np
import scipy.sparse as sp

from .cube import Cube, Schema, _all_codes, build_A

Display = Literal["omit", "range"]


class Mechanism(Protocol):
    """Anything that turns a true cube into a release."""

    family: str

    @property
    def label(self) -> str: ...

    def apply(self, cube: Cube) -> Release: ...


@dataclass
class Release:
    """A protected table as an attacker sees it.

    Attributes:
        schema: schema of the released table (coarser than the cube's after generalization).
        A: released rows x *original* atomic cells, 0/1.
        lo, hi: per-row range the attacker knows the true count lies in. For noisy (DP)
            releases these are not hard bounds and `exact` is False.
        shown: per-row value printed in the table; NaN = withheld.
        label, family, params: description of the mechanism that produced it.
        k: suppression threshold, if any.
        exact: False for noisy (DP) releases.
        f2c: generalized releases only: original atomic cell -> released atomic cell.
        cells_est: noisy releases only: the attacker's consistent, non-negative estimate of
            the released atomic cells, computed from the published rows only.
    """

    schema: Schema
    A: sp.csr_matrix
    lo: np.ndarray
    hi: np.ndarray
    shown: np.ndarray
    label: str
    family: str
    params: dict = field(default_factory=dict)
    k: int | None = None
    exact: bool = True
    f2c: np.ndarray | None = None
    cells_est: np.ndarray | None = None

    @property
    def published(self) -> np.ndarray:
        """Boolean mask of rows whose value is printed."""
        return ~np.isnan(self.shown)

    @property
    def n_rows(self) -> int:
        return len(self.lo)

    def contains(self, truth_rows: np.ndarray) -> bool:
        """True if every true row count lies in its released range."""
        return bool(np.all((self.lo <= truth_rows) & (truth_rows <= self.hi)))

    def summary(self) -> dict:
        pub = self.published
        return {
            "mechanism": self.label,
            "rows_total": self.n_rows,
            "rows_published": int(pub.sum()),
            "rows_withheld": int((~pub).sum()),
        }


# ---------------------------------------------------------------- suppression and rounding


def _suppress_ranges(c: np.ndarray, k: int, display: Display = "omit"):
    pub = c >= k
    lo = np.where(pub, c, 0)
    hi = np.where(pub, c, k - 1)
    shown = np.where(pub, c.astype(float), np.nan)
    if display == "range":  # small non-zero rows printed as "1-(k-1)", so zero rows are visible too
        lo = np.where(~pub & (c > 0), 1, lo)
        hi = np.where(c == 0, 0, hi)
        shown = np.where(c == 0, 0.0, shown)
    return lo.astype(np.int64), hi.astype(np.int64), shown


@dataclass
class Suppress:
    """Withhold every row with count below k (the Cumulus default, k = 10).

    display="omit" drops withheld rows from the table (Cumulus behaviour).
    display="range" prints them as "1-(k-1)" (e.g. "1-9"), which also tells the reader
    which rows are truly zero; the attack engine accounts for that.
    """

    k: int = 10
    display: Display = "omit"
    family: str = "suppression"

    @property
    def label(self) -> str:
        return f"k={self.k}" + (f", 1-{self.k - 1} shown" if self.display == "range" else "")

    @property
    def params(self) -> dict:
        return {"k": self.k} | ({"display": "range"} if self.display == "range" else {})

    def ranges(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lo, hi, shown) for true row counts c."""
        return _suppress_ranges(c, self.k, self.display)

    def postprocess(self, shown: np.ndarray) -> np.ndarray:
        """Suppress already-released (e.g. noisy) counts. Withheld rows become NaN."""
        return np.where(shown >= self.k, shown, np.nan)

    def apply(self, cube: Cube) -> Release:
        lo, hi, shown = self.ranges(cube.rows)
        return Release(cube.schema, cube.A, lo, hi, shown, self.label, self.family, self.params, k=self.k)


@dataclass
class Round:
    """Suppress rows below k, then round the published counts to a multiple of `base`.

    Rounding is round-half-up: r = base * floor((c + floor(base/2)) / base).
    A published r means c is in [max(k, r - floor(b/2)), r + ceil(b/2) - 1].
    `Round(base=k, k=k)` rounds every returned count to a multiple of the threshold.
    """

    base: int = 5
    k: int = 10
    family: str = "rounding"

    @property
    def label(self) -> str:
        return f"k={self.k}, round to {self.base}"

    @property
    def params(self) -> dict:
        return {"k": self.k, "base": self.base}

    def _round(self, c: np.ndarray) -> np.ndarray:
        b = self.base
        return b * ((c + b // 2) // b)

    def ranges(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lo, hi, shown) for true row counts c."""
        b = self.base
        pub = c >= self.k
        r = self._round(c)
        lo = np.where(pub, np.maximum(self.k, r - b // 2), 0)
        hi = np.where(pub, r + (b - b // 2) - 1, self.k - 1)
        shown = np.where(pub, r.astype(float), np.nan)
        return lo.astype(np.int64), hi.astype(np.int64), shown

    def postprocess(self, shown: np.ndarray) -> np.ndarray:
        """Suppress, then round, already-released (e.g. noisy, integer) counts."""
        s = np.where(np.isnan(shown), 0, shown).astype(np.int64)
        return np.where(shown >= self.k, self._round(s).astype(float), np.nan)

    def apply(self, cube: Cube) -> Release:
        lo, hi, shown = self.ranges(cube.rows)
        return Release(cube.schema, cube.A, lo, hi, shown, self.label, self.family, self.params, k=self.k)


# ---------------------------------------------------------------- generalization


@dataclass
class Generalize:
    """Coarsen the schema (merge levels and/or drop variables), then suppress below k.

    merge: {var: {old_level: new_level}}; unmapped levels keep their name.
    drop:  variables removed from the released table entirely.
    Inside a `Pipeline` only `merge` and `drop` are used; the threshold comes from the pipeline.
    """

    merge: dict = field(default_factory=dict)
    drop: tuple = ()
    k: int = 10
    name: str | None = None
    short: str | None = None  # short label for charts
    family: str = "generalization"

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        bits = [f"merge {v}" for v in self.merge] + [f"drop {v}" for v in self.drop]
        return f"k={self.k}, " + ", ".join(bits)

    @property
    def params(self) -> dict:
        return {"k": self.k, "merge": self.merge, "drop": list(self.drop)}

    def coarse_schema(self, schema: Schema) -> tuple[Schema, np.ndarray]:
        """Coarse schema and the fine-cell -> coarse-cell map."""
        names, levels, maps = [], [], []
        for i, n in enumerate(schema.names):
            if n in self.drop:
                continue
            m = self.merge.get(n, {})
            new: list[str] = []
            for lv in schema.levels[i]:
                t = m.get(lv, lv)
                if t not in new:
                    new.append(t)
            names.append(n)
            levels.append(new)
            maps.append((i, np.array([new.index(m.get(lv, lv)) for lv in schema.levels[i]])))
        cs = Schema.make(names, levels)
        fine = schema.cell_codes()
        if maps:
            coarse_codes = np.stack([mp[fine[:, i]] for i, mp in maps], axis=1)
            f2c = (coarse_codes * cs.cell_strides()).sum(1)
        else:
            f2c = np.zeros(schema.n_cells, dtype=np.int64)
        return cs, f2c

    def coarse_cube(self, cube: Cube) -> tuple[Cube, np.ndarray]:
        """The released (coarse) cube and the fine-cell -> coarse-cell map."""
        cs, f2c = self.coarse_schema(cube.schema)
        cells = np.bincount(f2c, weights=cube.cells, minlength=cs.n_cells).astype(np.int64)
        return Cube(cs, cells), f2c

    def apply(self, cube: Cube) -> Release:
        coarse, f2c = self.coarse_cube(cube)
        lo, hi, shown = _suppress_ranges(coarse.rows, self.k)
        A = _lift(coarse.schema, f2c)
        return Release(coarse.schema, A, lo, hi, shown, self.label, self.family, self.params, k=self.k, f2c=f2c)


def _lift(cs: Schema, f2c: np.ndarray) -> sp.csr_matrix:
    """Released (coarse) rows x original atomic cells."""
    J = len(f2c)
    M = sp.csr_matrix((np.ones(J, dtype=np.int8), (np.arange(J), f2c)), shape=(J, cs.n_cells))
    A = (build_A(cs) @ M.T).tocsr()
    A.data[:] = 1
    return A.astype(np.int8)


# Generalizations of the synthetic kidney cohort's race variable used in the sweep.
# race collapsed to White, Black, Other, missing:
RACE_COARSE = {
    "race": {
        "Asian": "Other race",
        "American Indian or Alaska Native": "Other race",
        "Native Hawaiian or Other Pacific Islander": "Other race",
    }
}
# Coarser: White vs everyone else (missing race included).
RACE_WHITE_OTHER = {
    "race": {
        "Black or African American": "Not White / unknown",
        "Asian": "Not White / unknown",
        "American Indian or Alaska Native": "Not White / unknown",
        "Native Hawaiian or Other Pacific Islander": "Not White / unknown",
        "cumulus__none": "Not White / unknown",
    }
}


# ---------------------------------------------------------------- differential privacy


class PrivacyBudgetError(ValueError):
    """The privacy accounting exceeds the requested budget."""


def nnls_fit(M: sp.csr_matrix, y: np.ndarray, J: int, total: float | None = None) -> np.ndarray:
    """argmin_{x >= 0} ||M x - y||^2 (the Gaussian-noise MLE), by L-BFGS-B.

    The start point spreads `total` (default y[0], the noisy grand total) evenly over the J cells.
    """
    from scipy.optimize import minimize

    M = M.astype(float).tocsr()
    MT = M.T.tocsr()

    def f(x: np.ndarray) -> tuple[float, np.ndarray]:
        r = M @ x - y
        return 0.5 * float(r @ r), MT @ r

    x0 = np.full(J, max(float(y[0] if total is None else total), 1.0) / J)
    res = minimize(
        f,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] * J,
        options={"maxiter": 5000, "maxfun": 10000, "ftol": 1e-14, "gtol": 1e-8},
    )
    return np.asarray(res.x)


def marginal_rows(schema: Schema, subset: tuple[str, ...]) -> np.ndarray:
    """Cube-row ids of the marginal table grouped by exactly `subset` (variable names)."""
    idx = [schema.names.index(v) for v in subset]
    sub_sizes = schema.sizes[idx]
    codes = np.tile(schema.sizes, (int(np.prod(sub_sizes)) if idx else 1, 1))
    if idx:
        codes[:, idx] = _all_codes(sub_sizes)
    return schema.encode_rows(codes)


def workload(schema: Schema, max_order: int) -> list[tuple[str, ...]]:
    """All marginals with up to `max_order` grouping variables (incl. the grand total)."""
    return [c for r in range(min(max_order, schema.n_vars) + 1) for c in combinations(schema.names, r)]


def sample_discrete_gaussian(sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Discrete Gaussian N_Z(0, sigma^2) from a seeded numpy generator.

    Rejection sampling from a discrete Laplace (Canonne, Kamath & Steinke 2020, Alg. 3).
    For reproducible benchmarks only: numpy's generator is not a cryptographic source
    and floating-point acceptance is not exact, so real releases use OpenDP (seed=None).
    """
    t = math.floor(sigma) + 1
    p = 1 - math.exp(-1 / t)
    out = np.empty(size, dtype=np.int64)
    filled = 0
    while filled < size:
        m = max(16, 2 * (size - filled))
        y = rng.geometric(p, m) - rng.geometric(p, m)
        accept = rng.random(m) < np.exp(-((np.abs(y) - sigma**2 / t) ** 2) / (2 * sigma**2))
        y = y[accept][: size - filled]
        out[filled : filled + len(y)] = y
        filled += len(y)
    return out


@dataclass
class DiscreteGaussian:
    """zCDP release of a workload of marginals, then a consistent non-negative cube.

    1. Every marginal in the workload (all tables with <= `max_order` grouping
       variables) gets discrete Gaussian noise. Adding or removing one patient
       changes each marginal by 1 in one cell, so with |W| marginals at scale
       sigma the release satisfies rho-zCDP with rho = |W| / (2 sigma^2). The
       accounting is done by OpenDP's measurement map.
    2. Post-processing (no privacy cost): non-negative least squares (L-BFGS-B)
       for the atomic counts that best fit the noisy marginals, rounded to integers.
       Every published row is then a sum of those cells, so the table is
       internally consistent and never negative.

    seed=None (default) draws noise from OpenDP's cryptographically secure sampler,
    which cannot be seeded; use it for real releases. An integer seed uses
    `sample_discrete_gaussian` instead, so benchmark sweeps are reproducible.
    """

    rho: float = 0.5
    max_order: int = 3
    seed: int | None = None
    family: str = "dp"

    @property
    def label(self) -> str:
        return f"DP ρ={self.rho:g}" + ("" if self.max_order == 3 else f" (≤{self.max_order}-way)")

    @property
    def params(self) -> dict:
        return {"rho": self.rho, "max_order": self.max_order} | ({} if self.seed is None else {"seed": self.seed})

    def sigma(self, schema: Schema) -> float:
        """Noise scale that spends exactly `rho` over the workload."""
        return float(np.sqrt(len(workload(schema, self.max_order)) / (2 * self.rho)))

    def noisy_marginals(self, cube: Cube) -> tuple[np.ndarray, np.ndarray, dict]:
        """(cube-row ids, noisy counts, accounting) for the workload."""
        import opendp.prelude as dp

        dp.enable_features("contrib")
        s = cube.schema
        W = workload(s, self.max_order)
        sigma = self.sigma(s)
        meas = dp.m.make_gaussian(dp.vector_domain(dp.atom_domain(T=int)), dp.l2_distance(T=int), scale=sigma)
        rho_spent = float(meas.map(1)) * len(W)
        if rho_spent > self.rho * (1 + 1e-9):
            raise PrivacyBudgetError(f"accounting spends ρ={rho_spent:g}, budget is ρ={self.rho:g}")
        rows = np.concatenate([marginal_rows(s, w) for w in W])
        true = cube.rows[rows].astype(np.int64)
        if self.seed is None:
            noisy = np.array(meas(true.tolist()), dtype=float)
        else:
            noisy = (true + sample_discrete_gaussian(sigma, len(true), np.random.default_rng(self.seed))).astype(float)
        acct = {"sigma": sigma, "marginals": len(W), "rho_spent": rho_spent}
        return rows, noisy, acct

    def apply(self, cube: Cube) -> Release:
        s = cube.schema
        rows, noisy, acct = self.noisy_marginals(cube)
        A = cube.A
        x = np.rint(nnls_fit(A[rows], noisy, s.n_cells)).astype(np.int64)
        shown = (A @ x).astype(float)
        R = len(shown)
        return Release(
            s,
            A,
            np.zeros(R, dtype=np.int64),
            np.full(R, np.iinfo(np.int32).max, dtype=np.int64),
            shown,
            self.label,
            self.family,
            self.params | acct,
            k=None,
            exact=False,
            cells_est=x,
        )


# ---------------------------------------------------------------- composition


@dataclass
class Pipeline:
    """Compose mechanisms: generalize, then DP noise, then suppress or round.

    Every step is optional. With `noise`, the threshold step is applied to the noisy
    counts, so it is post-processing and the zCDP guarantee is unchanged; the attacker
    is then credited with a fresh consistent estimate fitted to the published rows only.
    Without `noise`, the release is exact and the reconstruction attack applies.

    Example: ``Pipeline(Generalize(merge=RACE_COARSE), DiscreteGaussian(1.0), Suppress(10))``.
    """

    generalize: Generalize | None = None
    noise: DiscreteGaussian | None = None
    threshold: Suppress | Round | None = None
    family: str = "pipeline"

    @property
    def steps(self) -> list:
        return [m for m in (self.generalize, self.noise, self.threshold) if m is not None]

    @property
    def label(self) -> str:
        parts = []
        if self.generalize is not None:
            g = self.generalize
            parts.append(g.name or ", ".join([f"merge {v}" for v in g.merge] + [f"drop {v}" for v in g.drop]))
        parts += [m.label for m in (self.noise, self.threshold) if m is not None]
        return " → ".join(parts) or "publish exactly"

    @property
    def k(self) -> int | None:
        return self.threshold.k if self.threshold is not None else None

    def apply(self, cube: Cube) -> Release:
        base, f2c = self.generalize.coarse_cube(cube) if self.generalize is not None else (cube, None)
        s = base.schema
        params: dict = {"steps": [m.family for m in self.steps]}
        if self.generalize is not None:
            params |= {"merge": self.generalize.merge, "drop": list(self.generalize.drop)}
        if self.noise is None:
            c = base.rows
            if self.threshold is None:
                lo, hi, shown = c.astype(np.int64), c.astype(np.int64), c.astype(float)
            else:
                lo, hi, shown = self.threshold.ranges(c)
                params |= self.threshold.params
            rel = Release(s, base.A, lo, hi, shown, self.label, self.family, params, k=self.k)
        else:
            rel = self.noise.apply(base)
            params |= rel.params
            if self.threshold is not None:
                shown = self.threshold.postprocess(rel.shown)
                params |= self.threshold.params
                pub = ~np.isnan(shown)
                if pub.any():
                    fit = nnls_fit(base.A[pub], shown[pub], s.n_cells, total=float(np.nanmax(shown)))
                    est = np.rint(fit).astype(np.int64)
                else:  # everything withheld: the attacker learns nothing
                    est = np.zeros(s.n_cells, dtype=np.int64)
                rel = replace(rel, shown=shown, cells_est=est, k=self.k)
            rel = replace(rel, label=self.label, family=self.family, params=params)
        if f2c is not None:
            rel = replace(rel, A=_lift(s, f2c), f2c=f2c)
        return rel
