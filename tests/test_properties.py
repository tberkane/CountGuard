"""Property-based tests (Hypothesis) of mechanism invariants on random small cubes.

Exact mechanisms: every true count lies in its released range, and nothing below k
is printed. Attack: the proven [min, max] always contains the truth. DP: the release
is consistent (every row is the sum of its cells), non-negative, and never spends more
than its budget, alone or inside a Pipeline. Table I/O round-trips.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from countguard import Cube, DiscreteGaussian, Generalize, Pipeline, Round, Schema, Suppress, run_attack
from countguard.cumulus import read_table, release_to_table
from countguard.external import BudgetExceeded, Ledger

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def cubes(draw, max_vars=4):
    v = draw(st.integers(2, max_vars))
    sizes = draw(st.lists(st.integers(2, 3), min_size=v, max_size=v))
    names = [f"v{i}" for i in range(v)]
    levels = [[f"l{j}" for j in range(s)] for s in sizes]
    J = int(np.prod(sizes))
    # mostly small counts with many zeros: the regime where suppression leaks
    cells = draw(st.lists(st.one_of(st.just(0), st.integers(1, 9), st.integers(10, 60)), min_size=J, max_size=J))
    return Cube(Schema.make(names, levels), np.array(cells, dtype=np.int64))


thresholds = st.one_of(
    st.builds(Suppress, k=st.integers(2, 15), display=st.sampled_from(["omit", "range"])),
    st.builds(lambda b, k: Round(b, k=k), st.integers(2, 10), st.integers(1, 15)),
)


def _merge_first(cube: Cube) -> Generalize:
    lv = cube.schema.levels[0]
    return Generalize(merge={cube.schema.names[0]: {lv[1]: lv[0]}})


@SETTINGS
@given(cubes(), thresholds, st.booleans())
def test_exact_release_contains_truth(cube, threshold, generalize):
    gen = _merge_first(cube) if generalize else None
    rel = Pipeline(gen, None, threshold).apply(cube)
    truth = gen.coarse_cube(cube)[0].rows if gen else cube.rows
    assert rel.contains(truth)
    assert np.all(truth[rel.published & (truth > 0)] >= threshold.k)  # nothing below k is printed


@SETTINGS
@given(cubes(max_vars=3), st.integers(2, 12), st.sampled_from(["omit", "range"]), st.booleans())
def test_attack_bounds_contain_truth(cube, k, display, known_support):
    rel = Suppress(k, display=display).apply(cube)
    threat = "known_support" if known_support else "table_only"
    res = run_attack(rel, threat, truth=cube.cells, workers=1)
    assert np.all((res.L <= cube.cells) & (cube.cells <= res.U))


@SETTINGS
@given(cubes(), st.floats(0.05, 20), st.integers(1, 3), st.integers(0, 2**31))
def test_dp_consistent_nonnegative_within_budget(cube, rho, order, seed):
    rel = DiscreteGaussian(rho, order, seed=seed).apply(cube)
    assert rel.params["rho_spent"] <= rho * (1 + 1e-9)
    assert np.all(rel.cells_est >= 0)
    assert np.array_equal(rel.shown, (cube.A @ rel.cells_est).astype(float))  # every subtotal adds up


@SETTINGS
@given(cubes(), st.floats(0.1, 10), thresholds, st.integers(0, 2**31))
def test_pipeline_dp_then_threshold(cube, rho, threshold, seed):
    rel = Pipeline(None, DiscreteGaussian(rho, 2, seed=seed), threshold).apply(cube)
    assert rel.params["rho_spent"] <= rho * (1 + 1e-9)
    shown = rel.shown[rel.published]
    assert np.all(shown >= 0)
    if isinstance(threshold, Round):  # rounding may print a value below k (e.g. 1 -> 0 with base 3)
        assert np.all(shown % threshold.base == 0)
    else:
        assert np.all(shown >= threshold.k)
    assert np.all(rel.cells_est >= 0)


@SETTINGS
@given(cubes(), st.integers(2, 12), st.sampled_from(["omit", "range"]))
def test_table_roundtrip(cube, k, display):
    rel = Suppress(k, display=display).apply(cube)
    back = read_table(release_to_table(rel), schema=cube.schema, k=k)
    if display == "omit" or (rel.lo == rel.hi).all() or not (~rel.published & (rel.lo >= 1)).any():
        # without a range row, the reader cannot know zero rows are shown: it assumes [0, k-1]
        assert np.array_equal(back.lo[rel.published], rel.lo[rel.published])
        assert back.contains(cube.rows)
    else:
        assert np.array_equal(back.lo, rel.lo) and np.array_equal(back.hi, rel.hi)


@settings(max_examples=100, deadline=None)
@given(st.floats(0.1, 5), st.lists(st.floats(0, 2), max_size=40))
def test_ledger_never_overspends(total, charges):
    led = Ledger(total)
    for c in charges:
        try:
            led.charge(c, "q")
        except BudgetExceeded:
            pass
        assert led.spent <= total + 1e-12
