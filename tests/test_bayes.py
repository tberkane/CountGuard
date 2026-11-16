import numpy as np
import pytest

from countguard import Cube, Suppress, kidney_cohort
from countguard.attack import run_attack
from countguard.synth import LEVELS

pytest.importorskip("cmdstanpy")


@pytest.mark.parametrize("basis", ["original", "free"])
def test_posterior_mean_is_feasible(basis):
    from countguard.bayes import reconstruct

    vars_ = ["race", "sex", "cni", "antimetabolite", "steroid", "atg", "rejection"]
    cube = Cube.from_records(kidney_cohort(150, seed=5), vars_, LEVELS)
    rel = Suppress(10).apply(cube)
    res = run_attack(rel, "known_support", truth=cube.cells, workers=1)
    b = reconstruct(rel, res, basis=basis, seed=1, warmup=50, samples=50)
    e = b.estimate
    assert np.all(e >= res.L - 1e-9) and np.all(e <= res.U + 1e-9)
    pub = rel.published
    assert np.allclose(rel.A[pub] @ e, rel.shown[pub])  # mean of feasible points is feasible
    assert b.null_dim > 0 and b.acceptance.mean() > 0
