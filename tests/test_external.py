import numpy as np
import pytest

from countguard import KIDNEY_VARS, Cube, kidney_cohort
from countguard.external import BudgetExceeded, Ledger, SponsorSession, differencing_plan, parse_query, true_count
from countguard.synth import LEVELS


@pytest.fixture(scope="module")
def cube():
    return Cube.from_records(kidney_cohort(285, seed=2), KIDNEY_VARS, LEVELS)


def test_ledger_never_overspends():
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = Ledger(total=float(rng.uniform(0.1, 2)))
        for _ in range(50):
            rho = float(rng.choice([0.01, 0.05, 0.1, 0.3]))
            before = L.spent
            try:
                L.charge(rho, "q")
            except BudgetExceeded:
                assert L.spent == before  # refused charges cost nothing
            assert L.spent <= L.total + 1e-12


def test_dp_session_budget_and_cache(cube):
    s = SponsorSession(cube, "dp", budget=0.3, rho_per_query=0.1)
    qs = ["sex=female", "cni=yes", "rejection=yes", "race=Asian", "sex=male"]
    out = [s.ask(q) for q in qs]
    assert [o["refused"] for o in out] == [False, False, False, True, True]
    assert s.ledger.spent <= s.ledger.total + 1e-12
    again = s.ask("CNI = yes")  # equivalent wording -> cache, no cost
    assert again["cached"] and again["cost"] == 0 and again["answer"] == out[1]["answer"]
    assert all(o["answer"] >= 0 for o in out if not o["refused"])


def test_suppress_session_and_parse(cube):
    s = SponsorSession(cube, "suppress", k=10)
    a = s.ask("sex=female and rejection=yes")
    assert a["answer"] == true_count(parse_query("rejection=yes, sex=female", cube), cube)
    assert parse_query("sex=female and sex=male", cube) == ((cube.schema.names.index("sex"), -1),)
    with pytest.raises(ValueError):
        parse_query("blood=type", cube)


def test_differencing_plan_reveals_outcome(cube):
    p = differencing_plan(cube)
    assert p is not None
    A, B, C, D = p["true_counts"]
    assert A - B == 1 and min(B, C, D) >= 10 and C - D == p["target_outcome"]
