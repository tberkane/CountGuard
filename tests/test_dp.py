import numpy as np
import pytest

from countguard import SUBCUBE_8, Cube, kidney_cohort
from countguard.mechanisms import DiscreteGaussian, workload
from countguard.metrics import luck_floor, privacy
from countguard.synth import LEVELS


@pytest.fixture(scope="module")
def cube8():
    return Cube.from_records(kidney_cohort(2000, seed=7), SUBCUBE_8, LEVELS)


@pytest.mark.parametrize("rho,order", [(0.1, 3), (1.0, 2), (10.0, 3)])
def test_dp_nonnegative_consistent_and_within_budget(cube8, rho, order):
    rel = DiscreteGaussian(rho, order).apply(cube8)
    assert np.all(rel.shown >= 0) and np.all(rel.cells_est >= 0)
    assert np.array_equal(cube8.A @ rel.cells_est, rel.shown)  # every row = sum of its cells
    assert np.all(np.equal(np.mod(rel.shown, 1), 0))  # integer counts
    assert rel.params["rho_spent"] <= rho * (1 + 1e-9)
    assert rel.params["marginals"] == len(workload(cube8.schema, order))
    assert not rel.exact


def test_luck_floor_and_risk_definitions():
    x = np.array([1, 1, 1, 2, 2, 5, 40, 0, 0])  # n = 52
    floor, v = luck_floor(x)
    assert v == 5 and floor == pytest.approx(5 / 52)  # one 5-patient group (5) beats 2+2 (4) and 1+1+1 (3)
    p = privacy(x, guess=x.copy(), exact=np.ones_like(x, dtype=bool))
    assert p.raw == pytest.approx(12 / 52) and p.risk == pytest.approx(7 / 52) and p.certain == 6
    p0 = privacy(x, guess=np.full_like(x, 5))
    assert p0.risk == 0.0
