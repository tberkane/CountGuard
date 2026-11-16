"""Regression test: CountGuard reproduces the original research code's results on the
committed synthetic cohort (reproduction/data/synthetic_kidney_n285_seed2.csv).

The reference numbers in reproduction/results/original_seed2/ were produced by the
unmodified original code (reproduction/run_original.py). Synthetic data only: these are
not the preprint's numbers, which come from a real cohort.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from countguard import KIDNEY_VARS, Cube, Suppress, run_attack
from countguard.synth import LEVELS

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduction"
sys.path.insert(0, str(REPRO))
from run_countguard import first_appearance_cells, row_determinism  # noqa: E402


@pytest.fixture(scope="module")
def run():
    df = pd.read_csv(REPRO / "data" / "synthetic_kidney_n285_seed2.csv", dtype=str, keep_default_na=False)
    cube = Cube.from_records(df, KIDNEY_VARS, LEVELS)
    rel = Suppress(10).apply(cube)
    res = run_attack(rel, "known_support", truth=cube.cells, workers=1)
    return df, cube, rel, res


def test_bounds_identical_to_original_code(run):
    df, cube, _, res = run
    ref = np.load(REPRO / "results" / "original_seed2" / "atomic.npz")
    order = first_appearance_cells(df, cube.schema)
    assert np.array_equal(res.L[order], ref["L"]) and np.array_equal(res.U[order], ref["U"])


def test_row_counts_match_original_code(run):
    _, cube, rel, res = run
    ref = json.loads((REPRO / "results" / "original_seed2" / "result.json").read_text())
    det, sup = row_determinism(cube, res)
    assert int(rel.published.sum()) == ref["status_counts"]["Published"]  # 5,584
    assert sup == ref["total_suppressed"]  # 23,200
    assert det == ref["status_counts"]["Deterministic"]  # 19,750 (85.1% on this synthetic cohort)


def test_committed_countguard_snapshot_is_current(run):
    df, cube, _, res = run
    snap = np.load(REPRO / "results" / "countguard_seed2" / "atomic.npz")
    order = first_appearance_cells(df, cube.schema)
    assert np.array_equal(snap["cells"], order)
    assert np.array_equal(res.L[order], snap["L"]) and np.array_equal(res.U[order], snap["U"])
