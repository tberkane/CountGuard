"""Composition (Pipeline), "1-k" display, seeded DP and privacy accounting."""

import numpy as np
import pytest

from countguard import (
    SUBCUBE_6,
    Cube,
    DiscreteGaussian,
    Generalize,
    Pipeline,
    PrivacyBudgetError,
    Round,
    Suppress,
    accuracy_loss,
    assess,
    kidney_cohort,
    make_queries,
    run_attack,
)
from countguard.cumulus import read_table, release_to_table
from countguard.mechanisms import RACE_COARSE, sample_discrete_gaussian
from countguard.synth import LEVELS


@pytest.fixture(scope="module")
def cube6():
    return Cube.from_records(kidney_cohort(285, seed=1), SUBCUBE_6, LEVELS)


def _same(a, b):
    return (
        np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi) and np.array_equal(a.shown, b.shown, equal_nan=True)
    )


def test_pipeline_reduces_to_single_mechanisms(cube6):
    assert _same(Pipeline(threshold=Suppress(10)).apply(cube6), Suppress(10).apply(cube6))
    assert _same(Pipeline(threshold=Round(10, k=10)).apply(cube6), Round(10, k=10).apply(cube6))
    g = Generalize(merge=RACE_COARSE, k=10)
    a, b = Pipeline(Generalize(merge=RACE_COARSE), threshold=Suppress(10)).apply(cube6), g.apply(cube6)
    assert _same(a, b) and np.array_equal(a.f2c, b.f2c) and (a.A != b.A).nnz == 0


def test_generalize_dp_round(cube6):
    pipe = Pipeline(Generalize(merge=RACE_COARSE), DiscreteGaussian(1.0, seed=3), Round(10, k=10))
    rel = pipe.apply(cube6)
    assert pipe.label == "merge race → DP ρ=1 → k=10, round to 10"
    assert not rel.exact and rel.f2c is not None and rel.schema.sizes[0] == 4
    shown = rel.shown[rel.published]
    assert np.all(shown % 10 == 0) and np.all(shown >= 10)
    p, res = assess(cube6, rel)
    assert res is None and 0 <= p.risk <= 1 and p.certain == 0
    loss = accuracy_loss(cube6, rel, make_queries(cube6, n_queries=100))
    assert loss["median_rel_error"] >= 0


def test_range_display_tells_reader_more(cube6):
    omit, rng = Suppress(10).apply(cube6), Suppress(10, display="range").apply(cube6)
    small = (cube6.rows > 0) & (cube6.rows < 10)
    assert np.all(rng.lo[small] == 1) and np.all(rng.hi[cube6.rows == 0] == 0)
    a = run_attack(omit, "table_only", truth=cube6.cells, workers=1)
    b = run_attack(rng, "table_only", truth=cube6.cells, workers=1)
    assert np.all((b.L <= cube6.cells) & (cube6.cells <= b.U))
    assert b.exact.sum() >= a.exact.sum()  # a reader of "1-9" knows at least as much


def test_range_display_table_roundtrip(cube6):
    rel = Suppress(10, display="range").apply(cube6)
    tab = release_to_table(rel)
    assert (tab["cnt"] == "1-9").sum() == int(((cube6.rows > 0) & (cube6.rows < 10)).sum())
    back = read_table(tab, schema=cube6.schema, k=10)
    assert np.array_equal(back.lo, rel.lo) and np.array_equal(back.hi, rel.hi)


def test_seeded_dp_is_reproducible(cube6):
    a = DiscreteGaussian(0.5, seed=7).apply(cube6)
    b = DiscreteGaussian(0.5, seed=7).apply(cube6)
    c = DiscreteGaussian(0.5, seed=8).apply(cube6)
    assert np.array_equal(a.shown, b.shown) and not np.array_equal(a.shown, c.shown)


def test_unseeded_dp_uses_opendp(cube6):
    rel = DiscreteGaussian(0.5).apply(cube6)
    assert "seed" not in rel.params and rel.params["rho_spent"] <= 0.5 * (1 + 1e-9)


@pytest.mark.parametrize("sigma", [0.7, 3.0, 12.5])
def test_discrete_gaussian_sampler_moments(sigma):
    x = sample_discrete_gaussian(sigma, 200_000, np.random.default_rng(0))
    assert x.dtype == np.int64
    assert abs(x.mean()) < 5 * sigma / np.sqrt(len(x))
    # the discrete Gaussian's variance is within a few percent of sigma^2 (below it for small sigma)
    assert 0.85 * sigma**2 < x.var() < 1.03 * sigma**2


def test_budget_overspend_is_refused(cube6, monkeypatch):
    m = DiscreteGaussian(1.0, seed=0)
    monkeypatch.setattr(DiscreteGaussian, "sigma", lambda self, schema: 0.5)  # too little noise for rho=1
    with pytest.raises(PrivacyBudgetError):
        m.apply(cube6)
