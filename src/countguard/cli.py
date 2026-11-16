"""Command-line interface: ``countguard synth | audit | protect | frontier``.

Every command prints a JSON (or YAML) report with a provenance block: the CountGuard
version, the command, a UTC timestamp, the SHA-256 of each input file and the parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .attack import THREATS, run_attack
from .cube import Cube
from .cumulus import UNSEEN, _load, read_table, release_to_table, write_table
from .mechanisms import DiscreteGaussian, Generalize, Pipeline, Round, Suppress
from .metrics import TARGET_LOSS, TARGET_RISK, accuracy_loss, assess, in_box, make_queries, privacy_from_attack
from .synth import KIDNEY_VARS, MISSING, SUBCUBE_6, SUBCUBE_8, kidney_cohort

SMALL = 10  # risk is always scored on true counts 1-9


# ---------------------------------------------------------------- helpers


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance(args: argparse.Namespace, inputs: Sequence[str], params: dict) -> dict:
    return {
        "tool": "countguard",
        "version": __version__,
        "command": args.command,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": [{"path": p, "sha256": _sha256(p)} for p in inputs],
        "parameters": params,
    }


def _emit(report: dict, args: argparse.Namespace) -> None:
    report = _jsonable(report)
    if args.format == "yaml":
        import yaml

        text = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _level_key(lv: str) -> tuple[bool, str]:
    return (lv == MISSING, lv)


def read_cohort(path: str | Path, names: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """Patient-level table (.csv or .parquet, one row per patient, categorical columns).

    Empty values become 'cumulus__none'. Returns (frame, variables, levels); levels are
    sorted with the missing marker last, as `read_table` sorts them.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p).astype(str)
    else:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df = df.replace({"": MISSING, "nan": MISSING, "None": MISSING})
    names = list(names) if names else [c for c in df.columns if c != "patient_ref"]
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"error: cohort has no column(s) {missing}")
    levels = {n: sorted(df[n].unique(), key=_level_key) for n in names}
    return df, names, levels


def _vars(s: str | None) -> list[str] | None:
    return [v.strip() for v in s.split(",") if v.strip()] if s else None


# ---------------------------------------------------------------- commands

PRESETS = {"all": KIDNEY_VARS, "8": SUBCUBE_8, "6": SUBCUBE_6}


def cmd_synth(args: argparse.Namespace) -> None:
    df = kidney_cohort(args.n, args.seed)
    cols = ["patient_ref", *PRESETS[args.vars]]
    write_table(df[cols], args.out)
    _emit(
        {
            "provenance": _provenance(args, [], {"n": args.n, "seed": args.seed, "vars": PRESETS[args.vars]}),
            "written": args.out,
            "patients": len(df),
            "synthetic": True,
        },
        args,
    )


def cmd_audit(args: argparse.Namespace) -> None:
    cube: Cube | None = None
    inputs = [args.table]
    if args.cohort:
        table_vars = [c for c in _load(args.table).columns if c != "cnt"]
        df, names, levels = read_cohort(args.cohort, _vars(args.vars) or table_vars)
        inputs.append(args.cohort)
        cube = Cube.from_records(df, names, levels)
        rel = read_table(args.table, schema=cube.schema, k=args.k)
    else:
        rel = read_table(args.table, k=args.k)
    threat = args.threat or ("known_support" if cube is not None else "table_only")
    if threat == "known_support" and cube is None:
        raise SystemExit(
            "error: --threat known_support needs --cohort (the attacker's knowledge of which "
            "combinations exist comes from the patient-level data)"
        )
    truth = cube.cells if cube is not None else None
    res = run_attack(rel, threat, truth=truth, workers=args.workers)

    # Exposure an auditor can see from the bounds alone, without the true counts.
    unseen_idx = np.array([len(lv) - 1 if lv[-1] == UNSEEN else -1 for lv in rel.schema.levels])
    real = ~(rel.schema.cell_codes() == unseen_idx).any(axis=1)  # UNSEEN cells are not single groups
    forced = res.exact & (res.L >= 1) & (res.L < SMALL) & real
    total_row = rel.schema.n_rows - 1  # the all-rolled-up row: the grand total
    n = int(rel.lo[total_row]) if rel.lo[total_row] == rel.hi[total_row] else None
    exposure = {
        "small_groups_forced": int(forced.sum()),
        "patients_in_forced_groups": int(res.L[forced].sum()),
        "share_of_patients": (float(res.L[forced].sum() / n) if n else None),
    }
    report: dict = {
        "provenance": _provenance(args, inputs, {"k": args.k, "threat": threat, "vars": list(rel.schema.names)}),
        "table": {
            "variables": list(rel.schema.names),
            "unseen_levels": [n for n, lv in zip(rel.schema.names, rel.schema.levels, strict=True) if lv[-1] == UNSEEN],
            "rows_possible": rel.n_rows,
            "rows_published": int(rel.published.sum()),
            "patients": n,
        },
        "attack": {
            "threat": threat,
            "seconds": round(res.timings["total"], 3),
            "atomic_cells": len(res.L),
            "unknowns": int(res.in_model.sum()),
            "exactly_determined": int((res.exact & res.in_model).sum()),
            "settled_by": {k: v for k, v in res.counts.items() if v},
        },
        "exposure": exposure,
    }
    if cube is not None:
        p = privacy_from_attack(cube, res)
        report["risk"] = p.as_dict() | {"target": TARGET_RISK, "meets_target": p.risk < TARGET_RISK}
    if args.bounds:
        report["bounds"] = [
            {"cell": rel.schema.cell_label(int(j)), "min": int(res.L[j]), "max": int(res.U[j])}
            for j in np.flatnonzero(res.in_model)
        ]
    _emit(report, args)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Pipeline from the protect flags (generalize -> DP -> suppress/round)."""
    merge: dict[str, dict[str, str]] = {}
    for spec in args.merge or []:
        try:
            var, rule = spec.split(":", 1)
            old, new = rule.split("=", 1)
        except ValueError:
            raise SystemExit(f"error: --merge expects VAR:OLD=NEW, got {spec!r}") from None
        merge.setdefault(var, {})[old] = new
    gen = Generalize(merge=merge, drop=tuple(args.drop or ())) if (merge or args.drop) else None
    noise = DiscreteGaussian(args.dp_rho, args.dp_order, seed=args.seed) if args.dp_rho else None
    threshold: Suppress | Round | None
    if args.round:
        threshold = Round(args.round, k=args.k if args.k is not None else args.round)
    elif args.k is not None or noise is None:
        threshold = Suppress(args.k if args.k is not None else 10, display="range" if args.show_range else "omit")
    else:
        threshold = None
    return Pipeline(gen, noise, threshold)


def cmd_protect(args: argparse.Namespace) -> None:
    df, names, levels = read_cohort(args.cohort, _vars(args.vars))
    cube = Cube.from_records(df, names, levels)
    pipe = build_pipeline(args)
    rel = pipe.apply(cube)
    tab = release_to_table(rel, min_cnt=1 if not rel.exact else None)
    write_table(tab, args.out)
    params = {"mechanism": rel.label, "steps": [m.family for m in pipe.steps], "vars": names} | {
        k: v for k, v in rel.params.items() if k not in ("steps",)
    }
    report: dict = {
        "provenance": _provenance(args, [args.cohort], params),
        "written": args.out,
        "release": rel.summary() | {"rows_written": len(tab)},
    }
    if args.audit:
        p, _ = assess(cube, rel, args.threat, workers=args.workers)
        a = accuracy_loss(cube, rel, make_queries(cube))
        report["audit"] = {
            "threat": args.threat if rel.exact else "dp",
            "risk": p.as_dict(),
            "loss": a,
            "targets": {"risk": TARGET_RISK, "loss": TARGET_LOSS},
            "meets_targets": in_box(p.risk, a["median_rel_error"]),
        }
    _emit(report, args)


def cmd_frontier(args: argparse.Namespace) -> None:
    from .frontier import sweep

    df, names, levels = read_cohort(args.cohort, _vars(args.vars))
    cube = Cube.from_records(df, names, levels)
    seed = None if args.seed < 0 else args.seed
    res = sweep(
        cube,
        reps=args.reps,
        seed=seed,
        threat=args.threat,
        workers=args.workers,
        log=(lambda s: print(s, file=sys.stderr, flush=True)) if not args.quiet else None,
    )
    res["provenance"] = _provenance(
        args, [args.cohort], {"reps": args.reps, "seed": seed, "threat": args.threat, "vars": names}
    )
    _emit(res, args)


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="countguard", description="Audit and protect aggregate count-cube releases.")
    ap.add_argument("--version", action="version", version=f"countguard {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--format", choices=("json", "yaml"), default="json", help="report format (default json)")
        p.add_argument("--report", metavar="FILE", help="write the report here instead of stdout")

    def cohort_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("cohort", help="patient-level .csv/.parquet, one row per patient")
        p.add_argument("--vars", help="comma-separated variables (default: every column except patient_ref)")
        p.add_argument("--workers", type=int, default=None, help="solver processes (default: CPU count)")

    p = sub.add_parser("synth", help="write a synthetic kidney-transplant cohort (no real data)")
    p.add_argument("-n", type=int, default=285, help="patients (default 285)")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument(
        "--vars", choices=tuple(PRESETS), default="all", help="all 10 variables, or the 8/6-variable subcube"
    )
    p.add_argument("-o", "--out", required=True, help="output .csv or .parquet")
    common(p)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("audit", help="attack a released count table and report what it exposes")
    p.add_argument("table", help="Cumulus count table (.csv/.parquet): a cnt column plus one column per variable")
    p.add_argument("--k", type=int, default=10, help="suppression threshold the table was released with (default 10)")
    p.add_argument("--cohort", help="patient-level source: enables the risk score and the known_support attacker")
    p.add_argument("--vars", help="variables to read from --cohort (default: the table's columns)")
    p.add_argument(
        "--threat", choices=THREATS, help="attacker model (default: known_support with --cohort, else table_only)"
    )
    p.add_argument("--bounds", action="store_true", help="include the proven [min, max] of every atomic cell")
    p.add_argument("--workers", type=int, default=None, help="solver processes (default: CPU count)")
    common(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("protect", help="release a protected count table from patient-level data")
    cohort_args(p)
    p.add_argument("-o", "--out", required=True, help="protected table (.csv or .parquet)")
    g = p.add_argument_group("1. generalize")
    g.add_argument("--merge", action="append", metavar="VAR:OLD=NEW", help="merge a level (repeatable)")
    g.add_argument("--drop", action="append", metavar="VAR", help="leave a variable out (repeatable)")
    g = p.add_argument_group("2. differential privacy")
    g.add_argument("--dp-rho", type=float, help="zCDP budget; adds discrete Gaussian noise via OpenDP")
    g.add_argument("--dp-order", type=int, default=3, help="noise all marginals up to this many variables (3)")
    g.add_argument("--seed", type=int, default=None, help="seeded noise for reproducible tests (not for real releases)")
    g = p.add_argument_group("3. suppress / round")
    g.add_argument("--k", type=int, default=None, help="suppress counts below k (default 10 unless --dp-rho)")
    g.add_argument("--round", type=int, metavar="BASE", help="round published counts to a multiple of BASE")
    g.add_argument("--show-range", action="store_true", help='print suppressed counts as "1-(k-1)"')
    p.add_argument("--audit", action="store_true", help="also report risk and accuracy loss for this release")
    p.add_argument("--threat", choices=THREATS, default="known_support", help="attacker model for --audit")
    common(p)
    p.set_defaults(func=cmd_protect)

    p = sub.add_parser("frontier", help="sweep mechanisms and report the privacy-utility frontier")
    cohort_args(p)
    p.add_argument("--reps", type=int, default=5, help="DP repetitions per budget (default 5)")
    p.add_argument("--seed", type=int, default=0, help="DP seed; -1 = OpenDP's unseeded sampler")
    p.add_argument("--threat", choices=THREATS, default="known_support")
    p.add_argument("--quiet", action="store_true", help="no progress on stderr")
    common(p)
    p.set_defaults(func=cmd_frontier)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
