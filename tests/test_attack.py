import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from countguard import SUBCUBE_6, Cube, Generalize, Round, Suppress, kidney_cohort
from countguard.attack import run_attack
from countguard.mechanisms import RACE_COARSE
from countguard.synth import LEVELS

ORIGINAL = Path(__file__).resolve().parents[1] / "reproduction" / "original" / "src" / "infer_counts.py"


def _cbc_runs() -> bool:
    import pulp

    prob = pulp.LpProblem("probe", pulp.LpMinimize)
    x = pulp.LpVariable("x", 0, 1)
    prob += x
    try:
        return prob.solve(pulp.PULP_CBC_CMD(msg=False)) == pulp.LpStatusOptimal
    except Exception:  # noqa: BLE001
        return False


def _load_original():
    """Import the unmodified research code; stub heavy imports it does not need here."""
    for mod in ("matplotlib", "matplotlib.pyplot", "cmdstanpy", "sympy"):
        try:
            __import__(mod)
        except ImportError:
            m = types.ModuleType(mod)
            m.CmdStanModel = object
            sys.modules[mod] = m
            if mod == "matplotlib.pyplot":
                sys.modules["matplotlib"].pyplot = m
    sys.path.insert(0, str(ORIGINAL.parents[2]))
    import cbc_shim  # arm64 CBC for PuLP; see reproduction/cbc_shim.py

    cbc_shim.install()
    if not _cbc_runs():
        pytest.skip("PuLP's CBC cannot run here (e.g. x86-only binary on arm64 without a native cbc)")
    spec = importlib.util.spec_from_file_location("orig_infer_counts", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "seed,n,vars_",
    [
        (3, 80, ["race", "sex", "cni", "steroid", "rejection"]),
        (5, 150, ["race", "sex", "cni", "antimetabolite", "steroid", "atg", "rejection"]),
    ],
)
def test_bounds_equal_original(seed, n, vars_):
    pytest.importorskip("pulp")
    orig = _load_original()
    df = kidney_cohort(n, seed=seed)[vars_]
    cells_df, powerset_df = orig.line_level_to_powerset(df, vars_, min_count_threshold=10)
    J = len(cells_df)
    A_eq, y_eq, A_ineq, _ = orig.build_matrices(powerset_df, J)
    L0, U0 = orig.ilp_bounds(A_eq, y_eq, A_ineq, J, suppress_threshold=9)

    cube = Cube.from_records(df, vars_, LEVELS)
    res = run_attack(Suppress(10).apply(cube), "known_support", truth=cube.cells, workers=1)
    s = cube.schema
    codes = np.array([[s.levels[i].index(str(r[v])) for i, v in enumerate(vars_)] for _, r in cells_df.iterrows()])
    ours = (codes * s.cell_strides()).sum(1)
    assert np.array_equal(res.L[ours], L0)
    assert np.array_equal(res.U[ours], U0)


@pytest.fixture(scope="module")
def cube6():
    df = kidney_cohort(285, seed=2)
    return Cube.from_records(df, SUBCUBE_6, LEVELS)


@pytest.mark.parametrize("threat", ["known_support", "table_only"])
@pytest.mark.parametrize("mech", [Suppress(10), Round(5), Generalize(merge=RACE_COARSE)])
def test_bounds_contain_truth(cube6, mech, threat):
    res = run_attack(mech.apply(cube6), threat, truth=cube6.cells, workers=2)
    x = cube6.cells
    assert np.all((res.L <= x) & (x <= res.U))


def test_table_only_without_truth(cube6):
    """Without the auditor's truth the engine finds its own feasible point; bounds must agree."""
    rel = Suppress(10).apply(cube6)
    a = run_attack(rel, "table_only", truth=cube6.cells, workers=2)
    b = run_attack(rel, "table_only", truth=None, workers=2)
    assert np.array_equal(a.L, b.L) and np.array_equal(a.U, b.U)
