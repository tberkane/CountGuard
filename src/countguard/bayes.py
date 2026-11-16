"""Bayesian stage of the preprint's attack (port of the original research code).

Model (identical to the original's Stan program, vectorised with sparse rows):
    log lambda_j ~ Normal(0, 2)                      (X = identity, beta ~ N(0, 2))
    published row i:   y_i ~ Poisson(sum_{j in S_i} lambda_j)
    suppressed row i:  target += log P(Poisson(sum_{j in S_i} lambda_j) <= k-1)
NUTS: 2 chains, 100 warmup + 200 samples (400 rate draws).

Then, conditional on each rate draw, integer count vectors that satisfy every
published equality and the attack's [L, U] bounds are sampled by
Metropolis-Hastings on the integer null space of the equality system
(2,000 iterations, thinned by 10), starting from the min-sum feasible point.
The posterior mean is the point estimate.

basis="original": null-space basis of the FULL equality matrix over all model
    cells, in the original's column order (RREF is unique, so this reproduces
    sympy's basis exactly). Moves touching already-determined cells are always
    rejected, so chains can stall; kept for faithful reproduction.
basis="free" (default for CountGuard): basis of the equalities restricted to
    the cells the attack left uncertain; every proposal can move.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .attack import AttackResult, _new_highs
from .mechanisms import Release

STAN = r"""
// Rows with identical cell sets are merged with multiplicities m: the log
// posterior is unchanged up to a constant (same rate sum, same count).
data {
  int<lower=1> J;
  int<lower=0> nP;
  int<lower=0> nnzP;
  vector[nnzP] wP;
  array[nnzP] int vP;
  array[nP + 1] int uP;
  vector[nP] yP;            // count of the published row
  vector[nP] mP;            // multiplicity
  int<lower=0> nQ;
  int<lower=0> nnzQ;
  vector[nnzQ] wQ;
  array[nnzQ] int vQ;
  array[nQ + 1] int uQ;
  vector[nQ] mQ;
  int<lower=0> thr;         // k - 1
}
parameters {
  vector[J] beta;
}
transformed parameters {
  vector[J] lambda = exp(beta);
}
model {
  beta ~ normal(0, 2);
  if (nP > 0) {
    vector[nP] s = csr_matrix_times_vector(nP, J, wP, vP, uP, lambda);
    target += dot_product(mP .* yP, log(s)) - dot_product(mP, s);   // Poisson(y | s), constants dropped
  }
  if (nQ > 0) {
    vector[nQ] s = csr_matrix_times_vector(nQ, J, wQ, vQ, uQ, lambda);
    for (i in 1:nQ) target += mQ[i] * poisson_lcdf(thr | s[i]);
  }
}
"""

CACHE = Path(os.environ.get("COUNTGUARD_CACHE", Path.home() / ".cache" / "countguard"))


def _model():
    from cmdstanpy import CmdStanModel

    h = hashlib.sha1(STAN.encode()).hexdigest()[:10]
    d = CACHE / f"stan_{h}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "poisson_lognormal.stan"
    if not f.exists():
        f.write_text(STAN)
    return CmdStanModel(stan_file=str(f))


@dataclass
class BayesResult:
    """Output of the Bayesian reconstruction stage."""

    estimate: np.ndarray  # (J,) posterior mean count per atomic cell
    lambda_mean: np.ndarray  # (J,) posterior mean rate
    acceptance: np.ndarray  # (n_draws,) MH acceptance per rate draw
    null_dim: int
    timings: dict


def _system(rel: Release, res: AttackResult):
    """Rows of the attacker's model restricted to its unknown cells."""
    cols = np.flatnonzero(res.in_model)
    A = rel.A[:, cols].tocsr()
    nz = np.diff(A.indptr) > 0
    pub = rel.published & nz
    sup = (~rel.published) & nz
    return cols, A, pub, sup


def _merge(A: sp.csr_matrix, y=None):
    """Unique rows of a 0/1 matrix with multiplicities (and their common value y)."""
    keys: dict = {}
    order: list[list[int]] = []
    for i in range(A.shape[0]):
        key = A.indices[A.indptr[i] : A.indptr[i + 1]].tobytes()
        if key not in keys:
            keys[key] = len(order)
            order.append([i, 0])
        order[keys[key]][1] += 1
    idx = np.array([o[0] for o in order], dtype=np.int64)
    mult = np.array([o[1] for o in order], dtype=np.float64)
    return A[idx], mult, (None if y is None else y[idx])


def fit_rates(rel: Release, res: AttackResult, chains=2, warmup=100, samples=200, seed=0):
    cols, A, pub, sup = _system(rel, res)
    Ap, mP, yP = _merge(A[pub], rel.shown[pub])
    Aq, mQ, _ = _merge(A[sup])
    k = rel.k or 10
    data = {
        "J": len(cols),
        "nP": Ap.shape[0],
        "nnzP": Ap.nnz,
        "wP": Ap.data.astype(float).tolist(),
        "vP": (Ap.indices + 1).tolist(),
        "uP": (Ap.indptr + 1).tolist(),
        "yP": yP.tolist(),
        "mP": mP.tolist(),
        "nQ": Aq.shape[0],
        "nnzQ": Aq.nnz,
        "wQ": Aq.data.astype(float).tolist(),
        "vQ": (Aq.indices + 1).tolist(),
        "uQ": (Aq.indptr + 1).tolist(),
        "mQ": mQ.tolist(),
        "thr": k - 1,
    }
    m = _model()
    fit = m.sample(
        data=data,
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=samples,
        seed=seed,
        show_progress=False,
        show_console=False,
    )
    lam = fit.stan_variable("lambda")  # (draws, J_model)
    return cols, lam


def _rref_nullspace(M: np.ndarray) -> np.ndarray:
    """Integer null-space basis from the exact RREF (same as sympy's Matrix.nullspace)."""
    import sympy

    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=np.int64)
    # RREF depends only on the row space: keep an independent subset of rows first.
    Mf = M.astype(float)
    _, R, piv = __import__("scipy.linalg", fromlist=["qr"]).qr(Mf.T, pivoting=True, mode="economic")
    rank = int((np.abs(np.diag(R)) > 1e-9 * max(1.0, np.abs(R).max())).sum()) if R.size else 0
    rows = np.sort(piv[:rank])
    null = sympy.Matrix(M[rows].tolist()).nullspace()
    cols = []
    for v in null:
        den = 1
        for x in v:
            den = int(np.lcm(den, int(sympy.Rational(x).q)))
        w = np.array([int(sympy.Rational(x) * den) for x in v], dtype=np.int64)
        g = int(np.gcd.reduce(np.abs(w[w != 0]))) if (w != 0).any() else 1
        cols.append(w // max(g, 1))
    return np.column_stack(cols) if cols else np.zeros((M.shape[1], 0), dtype=np.int64)


def _min_sum_point(Aeq, beq, Aq, qhi, L, U):
    import highspy

    n = Aeq.shape[1]
    A = sp.vstack([Aeq, Aq]).tocsc()
    h = _new_highs(highspy)
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, A.shape[0]
    lp.col_cost_ = np.ones(n)
    lp.col_lower_, lp.col_upper_ = L.astype(float), U.astype(float)
    lp.row_lower_ = np.concatenate([beq, np.zeros(Aq.shape[0])]).astype(float)
    lp.row_upper_ = np.concatenate([beq, qhi]).astype(float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h.passModel(lp)
    h.run()
    return np.rint(np.array(h.getSolution().col_value)).astype(np.int64)


def sample_counts(
    rel: Release, res: AttackResult, cols, lam, basis="free", mh_iters=2000, thin=10, seed=0, col_order=None
):
    """Vectorised constrained MH, all rate draws in parallel. Returns (mean, acceptance, r)."""
    rng = np.random.default_rng(seed)
    _, A, pub, sup = _system(rel, res)
    k = rel.k or 10
    L, U = res.L[cols], res.U[cols]
    Aeq, beq = A[pub], np.rint(rel.shown[pub]).astype(np.int64)
    Aq = A[sup]
    qhi = np.full(Aq.shape[0], k - 1)
    n0 = _min_sum_point(Aeq, beq, Aq, qhi, L, U)

    if basis == "original":
        order = np.arange(len(cols)) if col_order is None else np.asarray(col_order)
        dense = Aeq.toarray()[:, order]
        Bo = _rref_nullspace(dense)
        B = np.zeros_like(Bo)
        B[order] = Bo
    else:
        move = np.flatnonzero(L < U)
        Bf = _rref_nullspace(Aeq[:, move].toarray()) if len(move) else np.zeros((0, 0), dtype=np.int64)
        B = np.zeros((len(cols), Bf.shape[1]), dtype=np.int64)
        B[move] = Bf
    r = B.shape[1]
    T = lam.shape[0]
    if r == 0:
        return n0.astype(float), np.zeros(T), 0

    # only cells some basis vector touches can change; the rest stay at n0
    act = np.flatnonzero((B != 0).any(1))
    Ba = B[act]  # (a, r)
    lam_a = lam[:, act]  # (T, a)
    Aq_a = Aq[:, act].tocsc()
    qrow = np.flatnonzero(np.diff(Aq_a.tocsr().indptr) > 0)
    Aq_a = Aq_a.tocsr()[qrow]
    base_q = (Aq @ n0)[qrow] - Aq_a @ n0[act]  # contribution of the other cells
    La, Ua = L[act], U[act]
    cur = np.tile(n0[act], (T, 1))  # (T, a)
    logl = np.log(lam_a)
    lp = lambda n: (n * logl - lam_a - gammaln(n + 1)).sum(1)
    lp_cur = lp(cur)
    acc = np.zeros(T)
    total = np.zeros_like(cur, dtype=float)
    ns = 0
    for it in range(mh_iters):
        kk = rng.integers(r, size=T)
        step = rng.choice([-1, 1], size=T)
        prop = cur + step[:, None] * Ba[:, kk].T
        ok = np.all((prop >= La) & (prop <= Ua), axis=1)
        if len(qrow):
            q = (Aq_a @ prop.T).T + base_q
            ok &= np.all((q >= 0) & (q <= k - 1), axis=1)
        lp_prop = np.where(ok, lp(np.maximum(prop, 0)), -np.inf)
        accept = ok & (np.log(rng.random(T)) < lp_prop - lp_cur)
        cur = np.where(accept[:, None], prop, cur)
        lp_cur = np.where(accept, lp_prop, lp_cur)
        acc += accept
        if it % thin == 0:
            total += cur
            ns += 1
    est = n0.astype(float)
    est[act] = total.sum(0) / (ns * T)
    return est, acc / mh_iters, r


def reconstruct(rel: Release, res: AttackResult, basis="free", seed=0, col_order=None, **kw) -> BayesResult:
    """Full Bayesian stage; returns the posterior-mean estimate for every atomic cell."""
    t = time.perf_counter()
    cols, lam = fit_rates(rel, res, seed=seed, **kw)
    t1 = time.perf_counter()
    est_m, acc, r = sample_counts(rel, res, cols, lam, basis=basis, seed=seed, col_order=col_order)
    t2 = time.perf_counter()
    J = len(res.L)
    est = np.zeros(J)
    est[cols] = est_m
    lm = np.zeros(J)
    lm[cols] = lam.mean(0)
    return BayesResult(est, lm, acc, r, {"stan": t1 - t, "mh": t2 - t1, "total": t2 - t})
