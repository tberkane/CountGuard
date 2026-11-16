"""External mode: a sponsor asks feasibility questions; the service answers under a policy.

Policies
  suppress  exact count if >= k, otherwise withheld ("< k"). No budget.
  dp        each NEW question costs rho_q from the sponsor's zCDP budget and is
            answered with OpenDP discrete Gaussian noise (one patient changes a
            count by 1), clamped at 0. Re-asking any equivalent question returns
            the cached answer at no cost, so noise cannot be averaged away.
            A question that would overspend the budget is refused.

Differencing attack (`differencing_plan`): find a group G and a rare attribute
X such that exactly one patient in G has X (the target). Then
  count(G) - count(G with X=common)                     = 1   -> target is singled out
  count(G, rejection=yes) - count(G, X=common, rejection=yes) = target's rejection
Under suppression all four counts are >= k, so they are published exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .cube import Cube


class BudgetExceeded(Exception):
    pass


@dataclass
class Ledger:
    """Per-user zCDP budget. Spending is all-or-nothing and never exceeds the total."""

    total: float
    spent: float = 0.0
    history: list = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    def charge(self, rho: float, what: str):
        if rho < 0:
            raise ValueError("negative charge")
        if self.spent + rho > self.total + 1e-12:
            raise BudgetExceeded(f"needs ρ={rho:g}, only {self.remaining:g} left")
        self.spent += rho
        self.history.append((what, rho))


def parse_query(text: str, cube: Cube) -> tuple[tuple[int, int], ...]:
    """'sex=female and cni=yes' -> canonical ((var, level), ...), sorted, de-duplicated."""
    s = cube.schema
    names = {n.lower(): i for i, n in enumerate(s.names)}
    parts = [p.strip() for p in re.split(r"\s*(?:\band\b|&&?|∧|,)\s*", text.strip(), flags=re.I) if p.strip()]
    if not parts or parts == ["*"] or text.strip().lower() in ("all", "all patients"):
        return ()
    conds: dict[int, int] = {}
    for p in parts:
        m = re.fullmatch(r"\s*([\w ]+?)\s*=\s*(.+?)\s*", p)
        if not m:
            raise ValueError(f"cannot read condition {p!r}; use variable=value")
        var, val = m.group(1).lower(), m.group(2).strip().strip("'\"")
        if var not in names:
            raise ValueError(f"unknown variable {m.group(1)!r}; known: {', '.join(s.names)}")
        vi = names[var]
        lv = {l.lower(): j for j, l in enumerate(s.levels[vi])}
        if val.lower() not in lv:
            raise ValueError(f"unknown value {val!r} for {s.names[vi]}; known: {', '.join(s.levels[vi])}")
        li = lv[val.lower()]
        if vi in conds and conds[vi] != li:
            return ((vi, -1),)  # contradictory: empty set
        conds[vi] = li
    return tuple(sorted(conds.items()))


def query_label(key, cube: Cube) -> str:
    s = cube.schema
    if not key:
        return "all patients"
    if key[0][1] == -1:
        return "(contradictory conditions)"
    return " ∧ ".join(f"{s.names[v]}={s.levels[v][l]}" for v, l in key)


def true_count(key, cube: Cube) -> int:
    if key and key[0][1] == -1:
        return 0
    code = cube.schema.sizes.copy()
    for v, l in key:
        code[v] = l
    return int(cube.rows[cube.schema.encode_rows([code])[0]])


@dataclass
class SponsorSession:
    """One external user's session: answers count queries under a policy ("suppress" or "dp"),
    charges each new DP question to a per-user zCDP budget, and answers repeats from a cache."""

    cube: Cube
    policy: str = "suppress"  # "suppress" | "dp"
    k: int = 10
    budget: float = 1.0  # per-sponsor zCDP budget (dp)
    rho_per_query: float = 0.05
    user: str = "sponsor"
    cache: dict = field(default_factory=dict)
    ledger: Ledger = field(init=False)
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.policy not in ("suppress", "dp"):
            raise ValueError("policy must be 'suppress' or 'dp'")
        self.ledger = Ledger(self.budget)
        if self.policy == "dp":
            import opendp.prelude as dp

            dp.enable_features("contrib")
            self._meas = dp.m.make_gaussian(dp.atom_domain(T=int), dp.absolute_distance(T=int), scale=self.sigma)
            assert self._meas.map(1) <= self.rho_per_query * (1 + 1e-9)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(1 / (2 * self.rho_per_query)))

    def ask(self, text: str) -> dict:
        key = parse_query(text, self.cube)
        label = query_label(key, self.cube)
        out: dict[str, Any] = {"query": text, "canonical": label, "policy": self.policy}
        if key in self.cache:
            out |= self.cache[key] | {"cached": True, "cost": 0.0}
        else:
            c = true_count(key, self.cube)
            if self.policy == "suppress":
                ans: dict[str, Any] = {
                    "answer": c if c >= self.k else None,
                    "withheld": c < self.k,
                    "display": str(c) if c >= self.k else f"< {self.k}",
                }
                cost = 0.0
            else:
                try:
                    self.ledger.charge(self.rho_per_query, label)
                except BudgetExceeded as e:
                    out |= {
                        "refused": True,
                        "reason": str(e),
                        "cost": 0.0,
                        "cached": False,
                        "budget_left": self.ledger.remaining,
                        "budget_total": self.ledger.total,
                    }
                    self.log.append(out)
                    return out
                v = max(0, int(self._meas(c)))
                ans = {"answer": v, "withheld": False, "display": str(v), "noise_sd": self.sigma}
                cost = self.rho_per_query
            self.cache[key] = ans
            out |= ans | {"cached": False, "cost": cost}
        out |= {"budget_left": self.ledger.remaining, "budget_total": self.ledger.total, "refused": False}
        self.log.append(out)
        return out


# ---------------------------------------------------------------- differencing


def _fmt(cube, key):
    return " and ".join(f"{cube.schema.names[v]}={cube.schema.levels[v][l]}" for v, l in key)


def differencing_plan(
    cube: Cube, k: int = 10, outcome: str = "rejection", max_group_vars: int = 3, prefer_outcome_level: str = "yes"
) -> dict | None:
    """Find a group G and attribute X singling out one patient, with all four counts >= k."""
    s = cube.schema
    oi = s.names.index(outcome)
    oyes = s.levels[oi].index(prefer_outcome_level)
    binaries = [i for i in range(s.n_vars) if s.sizes[i] == 2 and i != oi]
    others = [i for i in range(s.n_vars) if i != oi]
    best = None
    for r in range(1, max_group_vars + 1):
        for gvars in combinations(others, r):
            for xi in binaries:
                if xi in gvars:
                    continue
                for glev in np.ndindex(*[int(s.sizes[v]) for v in gvars]):
                    G = tuple(zip(gvars, glev, strict=True))
                    for xrare in (0, 1):
                        xcommon = 1 - xrare
                        cG = true_count(tuple(sorted(G)), cube)
                        cGc = true_count(tuple(sorted(G + ((xi, xcommon),))), cube)
                        if cG - cGc != 1 or cGc < k:
                            continue
                        cGy = true_count(tuple(sorted(G + ((oi, oyes),))), cube)
                        cGcy = true_count(tuple(sorted(G + ((xi, xcommon), (oi, oyes)))), cube)
                        if cGy < k or cGcy < k:
                            continue
                        reveal = cGy - cGcy
                        score = (reveal == 1, -r, -cG)
                        if best is None or score > best[0]:
                            best = (score, G, xi, xrare, xcommon, (cG, cGc, cGy, cGcy))
        if best is not None and best[0][0]:
            break
    if best is None:
        return None
    _, G, xi, xrare, xcommon, counts = best
    gkey = tuple(sorted(G))
    qs = [
        _fmt(cube, gkey),
        _fmt(cube, tuple(sorted(G + ((xi, xcommon),)))),
        _fmt(cube, tuple(sorted(G + ((oi, oyes),)))),
        _fmt(cube, tuple(sorted(G + ((xi, xcommon), (oi, oyes))))),
    ]
    xname, xr = s.names[xi], s.levels[xi][xrare]
    return {
        "group": _fmt(cube, gkey),
        "rare": f"{xname}={xr}",
        "outcome": f"{outcome}={prefer_outcome_level}",
        "queries": qs,
        "steps": [
            f"A = count({qs[0]})",
            f"B = count({qs[1]})",
            f"A − B = 1  →  exactly one patient in this group has {xname}={xr}: the target",
            f"C = count({qs[2]})",
            f"D = count({qs[3]})",
            f"C − D = the target's {outcome}: 1 means {outcome}={prefer_outcome_level}",
        ],
        "true_counts": list(map(int, counts)),
        "target_outcome": int(counts[2] - counts[3]),
    }
