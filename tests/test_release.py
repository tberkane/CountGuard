import numpy as np
import pytest

from countguard import SUBCUBE_6, Cube, Generalize, Round, Suppress, kidney_cohort
from countguard.cumulus import cumulus_table_from_cohort, read_table, release_to_table
from countguard.mechanisms import RACE_COARSE
from countguard.synth import LEVELS


@pytest.fixture(scope="module")
def cube6():
    df = kidney_cohort(285, seed=1)
    return df, Cube.from_records(df, SUBCUBE_6, LEVELS)


def test_cube_rows_match_groupby(cube6):
    df, cube = cube6
    s = cube.schema
    assert cube.rows[s.encode_rows([s.sizes])[0]] == len(df)  # grand total
    # every fully-specified row equals its cell
    assert np.array_equal(cube.rows[s.cell_to_row()], cube.cells)
    # A @ cells reproduces every row count
    assert np.array_equal(cube.A @ cube.cells, cube.rows)
    # a 2-way margin against pandas
    g = df.groupby(["sex", "rejection"]).size()
    for (sx, rj), c in g.items():
        code = np.array(s.sizes)
        code[s.names.index("sex")] = s.levels[s.names.index("sex")].index(sx)
        code[s.names.index("rejection")] = s.levels[s.names.index("rejection")].index(rj)
        assert cube.rows[s.encode_rows([code])[0]] == c


@pytest.mark.parametrize("mech", [Suppress(10), Suppress(5), Round(5), Round(3, k=10), Round(10, k=1)])
def test_ranges_contain_truth(cube6, mech):
    _, cube = cube6
    rel = mech.apply(cube)
    assert rel.contains(cube.rows)
    assert np.all(rel.lo <= rel.hi)


def test_generalize_contains_truth(cube6):
    _, cube = cube6
    rel = Generalize(merge=RACE_COARSE, k=10).apply(cube)
    truth = rel.A @ cube.cells
    assert rel.contains(truth)
    assert rel.schema.sizes[0] == 4  # White, Black, Other race, cumulus__none


def test_duckdb_sql_matches_cube(cube6):
    df, cube = cube6
    t = cumulus_table_from_cohort(df, SUBCUBE_6, k=10)
    ours = release_to_table(Suppress(10).apply(cube))
    assert len(t) == len(ours)
    key = lambda d: sorted(map(tuple, d.astype(object).where(d.notna(), None).values.tolist()), key=str)
    assert key(t[ours.columns]) == key(ours)


def test_cumulus_roundtrip(cube6, tmp_path):
    df, cube = cube6
    t = cumulus_table_from_cohort(df, SUBCUBE_6, k=10)
    p = tmp_path / "count.csv"
    t.to_csv(p, index=False)
    rel = read_table(p, schema=cube.schema, k=10)
    ref = Suppress(10).apply(cube)
    assert np.array_equal(rel.lo, ref.lo) and np.array_equal(rel.hi, ref.hi)
