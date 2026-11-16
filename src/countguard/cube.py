"""Count cubes: atomic cells, cube rows, and the sparse row-by-cell matrix A.

Atomic cell  = one full combination of variable levels (mixed radix over the
               level counts, last variable fastest).
Cube row     = one GROUP BY CUBE output row: each variable is either fixed to a
               level or rolled up ("all", NULL in Cumulus). Encoded in mixed
               radix over (levels + 1); code == n_levels means rolled up.
A[r, j] = 1  iff atomic cell j contributes to cube row r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class Schema:
    """Variables of a cube and the levels of each, in order."""

    names: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]

    @classmethod
    def make(cls, names, levels) -> Schema:
        return cls(tuple(names), tuple(tuple(map(str, lv)) for lv in levels))

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(lv) for lv in self.levels], dtype=np.int64)

    @property
    def n_vars(self) -> int:
        return len(self.names)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.sizes))

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.sizes + 1))

    def cell_strides(self) -> np.ndarray:
        return _strides(self.sizes)

    def row_strides(self) -> np.ndarray:
        return _strides(self.sizes + 1)

    def cell_codes(self) -> np.ndarray:
        """(J, v) level codes of every atomic cell."""
        return _all_codes(self.sizes)

    def row_codes(self, rows=None) -> np.ndarray:
        """(R, v) codes of cube rows; code == size means rolled up."""
        rows = np.arange(self.n_rows) if rows is None else np.asarray(rows)
        return _decode(rows, self.sizes + 1)

    def row_order(self, rows=None) -> np.ndarray:
        """Number of grouping (non-rolled-up) variables of each row."""
        return (self.row_codes(rows) < self.sizes).sum(1)

    def encode_rows(self, codes: npt.ArrayLike) -> np.ndarray:
        """Cube-row ids of (R, v) level codes (code == size means rolled up)."""
        return (np.asarray(codes, dtype=np.int64) * self.row_strides()).sum(1)

    def cell_to_row(self, cells=None) -> np.ndarray:
        """Cube-row id of the fully specified row for each atomic cell."""
        codes = self.cell_codes() if cells is None else _decode(np.asarray(cells), self.sizes)
        return self.encode_rows(codes)

    def row_label(self, row: int, sep: str = " ∧ ") -> str:
        c = self.row_codes([row])[0]
        parts = [f"{n}={self.levels[i][c[i]]}" for i, n in enumerate(self.names) if c[i] < len(self.levels[i])]
        return sep.join(parts) if parts else "all patients"

    def cell_label(self, cell: int, sep: str = " ∧ ") -> str:
        c = _decode(np.array([cell]), self.sizes)[0]
        return sep.join(f"{n}={self.levels[i][c[i]]}" for i, n in enumerate(self.names))

    def subset(self, names) -> Schema:
        idx = [self.names.index(n) for n in names]
        return Schema(tuple(self.names[i] for i in idx), tuple(self.levels[i] for i in idx))


def _strides(sizes: np.ndarray) -> np.ndarray:
    s = np.ones(len(sizes), dtype=np.int64)
    for i in range(len(sizes) - 2, -1, -1):
        s[i] = s[i + 1] * sizes[i + 1]
    return s


def _decode(idx: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty((len(idx), len(sizes)), dtype=np.int64)
    for i in range(len(sizes) - 1, -1, -1):
        out[:, i] = idx % sizes[i]
        idx = idx // sizes[i]
    return out


def _all_codes(sizes: np.ndarray) -> np.ndarray:
    return _decode(np.arange(int(np.prod(sizes))), sizes)


def build_A(schema: Schema) -> sp.csr_matrix:
    """Sparse (n_rows x n_cells) 0/1 matrix of every cube row over atomic cells."""
    sizes, rs = schema.sizes, schema.row_strides()
    codes = schema.cell_codes()
    J, v = codes.shape
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    cells = np.arange(J)
    for mask in range(1 << v):
        r = np.zeros(J, dtype=np.int64)
        for i in range(v):
            r += (codes[:, i] if (mask >> i) & 1 else sizes[i]) * rs[i]
        rows_list.append(r)
        cols_list.append(cells)
    rows_all = np.concatenate(rows_list)
    cols_all = np.concatenate(cols_list)
    data = np.ones(len(rows_all), dtype=np.int8)
    return sp.csr_matrix((data, (rows_all, cols_all)), shape=(schema.n_rows, J))


def row_counts_from_cells(schema: Schema, x: np.ndarray) -> np.ndarray:
    """All cube-row counts from atomic counts, via one cumulative tensor pass."""
    t = np.asarray(x).reshape(tuple(schema.sizes))
    for ax in range(t.ndim):
        t = np.concatenate([t, t.sum(axis=ax, keepdims=True)], axis=ax)
    return t.reshape(-1)


@dataclass
class Cube:
    """A full (unsuppressed) count cube: ground truth held by the data owner."""

    schema: Schema
    cells: np.ndarray  # (J,) true atomic counts
    _A: sp.csr_matrix | None = field(default=None, repr=False)

    @classmethod
    def from_records(cls, df: pd.DataFrame, names=None, levels: dict | None = None) -> Cube:
        """Build from one-row-per-patient records (categorical columns)."""
        names = list(names) if names is not None else [c for c in df.columns if c != "patient_ref"]
        lv = []
        for n in names:
            if levels and n in levels:
                lv.append(list(levels[n]))
            else:
                lv.append(sorted(df[n].astype(str).unique()))
        schema = Schema.make(names, lv)
        codes = np.empty((len(df), len(names)), dtype=np.int64)
        for i, n in enumerate(names):
            m = {l: k for k, l in enumerate(schema.levels[i])}
            vals = df[n].astype(str)
            bad = ~vals.isin(m.keys())
            if bad.any():
                raise ValueError(f"column {n!r} has values outside its levels: {sorted(vals[bad].unique())[:5]}")
            codes[:, i] = vals.map(m).to_numpy()
        idx = (codes * schema.cell_strides()).sum(1)
        x = np.bincount(idx, minlength=schema.n_cells).astype(np.int64)
        return cls(schema, x)

    @property
    def n(self) -> int:
        return int(self.cells.sum())

    @cached_property
    def rows(self) -> np.ndarray:
        """True count of every cube row (including zeros)."""
        return row_counts_from_cells(self.schema, self.cells)

    @property
    def A(self) -> sp.csr_matrix:
        if self._A is None:
            self._A = build_A(self.schema)
        return self._A

    def marginalize(self, names) -> Cube:
        """Sub-cube over a subset of the variables (sums out the others)."""
        keep = [self.schema.names.index(n) for n in names]
        if keep != sorted(keep):
            raise ValueError("marginalize: pass names in schema order")
        t = self.cells.reshape(tuple(self.schema.sizes))
        drop = tuple(i for i in range(t.ndim) if i not in keep)
        t = t.sum(axis=drop) if drop else t
        return Cube(self.schema.subset(names), t.reshape(-1).astype(np.int64))

    def summary(self) -> dict:
        x = self.cells
        return {
            "n_patients": self.n,
            "n_vars": self.schema.n_vars,
            "n_atomic_cells": self.schema.n_cells,
            "n_cube_rows": self.schema.n_rows,
            "nonempty_cells": int((x > 0).sum()),
            "small_cells_1_9": int(((x > 0) & (x < 10)).sum()),
            "patients_in_small_cells": int(x[(x > 0) & (x < 10)].sum()),
            "nonempty_rows": int((self.rows > 0).sum()),
        }
