"""Cumulus-library count tables: generate, read, write.

Format (cumulus-library `count.sql.jinja`, `studies/core/reference_sql/count_core.sql`):
  * one column `cnt` plus one column per variable;
  * NULL in a variable column = rolled up ("all");
  * the string 'cumulus__none' = the variable's value is missing;
  * produced by GROUP BY cube(...) over count(DISTINCT patient_ref),
    then filtered WHERE cnt >= 10 -- so suppressed rows and empty rows are
    both simply absent from the table.

CountGuard also writes and reads suppressed rows shown as a range ("1-9",
`Suppress(display="range")`); a table with such rows tells the reader that
every absent row is zero.

Tables can come from a file (.csv, .parquet) or from any query engine as a
pandas DataFrame, e.g. ``pd.read_sql("SELECT * FROM study__count_cube", trino_conn)``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .cube import Schema, build_A
from .mechanisms import Release
from .synth import MISSING


def count_sql(
    names: Sequence[str], source: str = "cohort", table: str = "countguard__count_cohort", k: int = 10
) -> str:
    """The SQL cumulus-library emits for a count table, specialised to `names`."""
    q = lambda n: f'"{n}"'
    cols = ", ".join(f"s.{q(n)}" for n in names)
    coalesced = ",\n            ".join(f"coalesce(cast({q(n)} AS varchar), '{MISSING}') AS {q(n)}" for n in names)
    group = ", ".join(q(n) for n in names)
    ids = ", ".join(f"coalesce({q(n)}, '')" for n in names)
    return f"""CREATE TABLE {table} AS (
    WITH
    filtered_table AS (
        SELECT s.patient_ref, {cols}
        FROM {source} AS s
    ),
    null_replacement AS (
        SELECT patient_ref,
            {coalesced}
        FROM filtered_table
    ),
    powerset AS (
        SELECT count(DISTINCT patient_ref) AS cnt_patient_ref,
            {group},
            concat_ws('-', {ids}) AS id
        FROM null_replacement
        GROUP BY cube({group})
    )
    SELECT cnt_patient_ref AS cnt, {group}
    FROM powerset
    WHERE cnt_patient_ref >= {k}
)"""


def cumulus_table_from_cohort(df: pd.DataFrame, names: Sequence[str], k: int = 10) -> pd.DataFrame:
    """Run the cumulus-library SQL pattern in DuckDB over a patient-level frame."""
    src = df[["patient_ref", *names]].copy()
    for n in names:  # missing values arrive as NULL, as in the source FHIR tables
        src[n] = src[n].where(src[n] != MISSING, None)
    con = duckdb.connect()
    con.register("cohort_df", src)
    con.execute("CREATE TABLE cohort AS SELECT * FROM cohort_df")
    con.execute(count_sql(names, k=k))
    out = con.execute("SELECT * FROM countguard__count_cohort").df()
    con.close()
    return _sort(out, names)


def _sort(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    order = df[list(names)].isna().sum(axis=1)
    return (
        df.assign(_o=-order).sort_values(["_o", *names], na_position="first").drop(columns="_o").reset_index(drop=True)
    )


def release_to_table(rel: Release, min_cnt: int | None = None) -> pd.DataFrame:
    """Cumulus-format table of the published rows of a release.

    With `Suppress(display="range")`, suppressed non-zero rows are included with
    cnt "1-(k-1)" (the column is then text) and zero rows are left out.
    """
    k = rel.k if rel.k is not None else 0
    band = rel.params.get("display") == "range" and k > 0
    pub = rel.published & ~(band & (np.nan_to_num(rel.shown, nan=-1) == 0))
    if min_cnt is not None:
        pub &= ~(np.nan_to_num(rel.shown, nan=np.inf) < min_cnt)
    if band:
        pub |= ~rel.published & (rel.lo >= 1)
    idx = np.flatnonzero(pub)
    codes = rel.schema.row_codes(idx)
    vals = rel.shown[idx]
    cnt: np.ndarray = np.rint(np.nan_to_num(vals)).astype(np.int64)
    if band:
        cnt = np.where(np.isnan(vals), f"1-{k - 1}", cnt.astype(str)).astype(object)
    data: dict[str, np.ndarray] = {"cnt": cnt}
    for i, n in enumerate(rel.schema.names):
        lv = np.array(list(rel.schema.levels[i]) + [None], dtype=object)
        data[n] = lv[codes[:, i]]
    return _sort(pd.DataFrame(data), rel.schema.names)


def read_table(src: pd.DataFrame | str | Path, schema: Schema | None = None, k: int = 10) -> Release:
    """Read a Cumulus count table (DataFrame, .csv or .parquet) as an attacker sees it.

    Rows present -> exact [cnt, cnt]; a range cnt "a-b" -> [a, b]; every absent
    cube row -> [0, k-1], or [0, 0] if the table shows suppressed rows as ranges.

    Levels come from `schema` if given, else from the values present. A level whose
    rows are all suppressed never appears in the table; when a variable's one-way
    counts prove such levels exist (they cannot add up to the grand total), the
    variable gets one extra level, `UNSEEN`, standing for all of them, with no
    upper bound of its own. Pass `schema` for an exact audit.
    """
    df = _load(src)
    if "cnt" not in df.columns:
        raise ValueError("not a Cumulus count table: no `cnt` column")
    names = [c for c in df.columns if c != "cnt"]
    inferred = schema is None
    if schema is None:
        levels = [sorted(df[n].dropna().astype(str).unique(), key=_level_key) for n in names]
        levels = _add_unseen(df, names, levels, k)
        schema = Schema.make(names, levels)
    else:
        missing = set(schema.names) - set(names)
        if missing:
            raise ValueError(f"table lacks columns {sorted(missing)}")
        df = df[["cnt", *schema.names]]
        names = list(schema.names)
    codes = np.empty((len(df), len(names)), dtype=np.int64)
    for i, n in enumerate(names):
        m = {l: j for j, l in enumerate(schema.levels[i])}
        col = df[n]
        codes[:, i] = [len(m) if pd.isna(v) else m[str(v)] for v in col]
    rows = schema.encode_rows(codes)
    clo, chi = _parse_cnt(df["cnt"])
    band = clo != chi
    R = schema.n_rows
    lo = np.zeros(R, dtype=np.int64)
    hi = np.full(R, 0 if band.any() else k - 1, dtype=np.int64)
    shown = np.full(R, np.nan)
    if inferred and not band.any():
        unseen = schema.row_codes() == np.array([len(lv) - 1 if lv[-1] == UNSEEN else -1 for lv in schema.levels])
        hi[unseen.any(axis=1)] = np.iinfo(np.int32).max
    lo[rows], hi[rows] = clo, chi
    shown[rows] = np.where(band, np.nan, clo)
    params: dict = {"k": k} | ({"display": "range"} if band.any() else {})
    return Release(schema, build_A(schema), lo, hi, shown, f"Cumulus table (k={k})", "suppression", params, k=k)


UNSEEN = "countguard__unseen"


def _add_unseen(df: pd.DataFrame, names: list[str], levels: list[list[str]], k: int) -> list[list[str]]:
    """Add UNSEEN to every variable whose published one-way counts cannot reach the grand total."""
    rolled = df[names].isna()
    total_rows = df[rolled.all(axis=1)]
    if len(total_rows) != 1 or isinstance(total_rows["cnt"].iloc[0], str) and "-" in total_rows["cnt"].iloc[0]:
        return levels
    total = int(total_rows["cnt"].iloc[0])
    out = []
    for i, n in enumerate(names):
        one_way = df[~rolled[n] & rolled.drop(columns=n).all(axis=1)]
        published = sum(int(c) for c in one_way["cnt"].astype(str) if c.isdigit())
        absent = len(levels[i]) - len(one_way)
        out.append(levels[i] + [UNSEEN] if published + absent * (k - 1) < total else levels[i])
    return out


_RANGE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def _parse_cnt(col: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if pd.api.types.is_integer_dtype(col):
        v = col.to_numpy(dtype=np.int64)
        return v, v
    lo, hi = np.empty(len(col), dtype=np.int64), np.empty(len(col), dtype=np.int64)
    for i, c in enumerate(col.astype(str)):
        m = _RANGE.match(c)
        if m:
            lo[i], hi[i] = int(m.group(1)), int(m.group(2))
        else:
            lo[i] = hi[i] = int(c)
    return lo, hi


def _level_key(lv: str) -> tuple[bool, str]:
    return (lv == MISSING, lv)


def _load(src: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    p = Path(src)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a count table as .parquet, or as CSV for any other suffix."""
    p = Path(path)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)
    return p
