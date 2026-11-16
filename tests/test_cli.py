"""End-to-end CLI: synth -> protect -> audit -> frontier, JSON and YAML reports."""

import json

import pandas as pd
import pytest
import yaml

from countguard.cli import main


def run(capsys, *argv):
    main(list(map(str, argv)))
    return capsys.readouterr().out


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    p = tmp_path_factory.mktemp("cli") / "cohort.csv"
    main(["synth", "-n", "285", "--seed", "2", "--vars", "6", "-o", str(p), "--report", str(p.with_suffix(".json"))])
    return p


def test_synth(cohort):
    df = pd.read_csv(cohort)
    assert len(df) == 285 and list(df.columns)[0] == "patient_ref"
    assert json.loads(cohort.with_suffix(".json").read_text())["synthetic"] is True


def test_protect_and_audit(cohort, tmp_path, capsys):
    table = tmp_path / "cube.csv"
    rep = json.loads(run(capsys, "protect", cohort, "-o", table))
    assert rep["provenance"]["parameters"]["mechanism"] == "k=10"
    assert len(rep["provenance"]["inputs"][0]["sha256"]) == 64
    assert pd.read_csv(table)["cnt"].min() >= 10

    blind = json.loads(run(capsys, "audit", table, "--k", "10"))
    assert blind["attack"]["threat"] == "table_only" and "risk" not in blind
    assert blind["table"]["patients"] == 285
    assert blind["exposure"]["small_groups_forced"] > 0

    full = yaml.safe_load(run(capsys, "audit", table, "--cohort", cohort, "--format", "yaml"))
    assert full["attack"]["threat"] == "known_support"
    assert full["risk"]["certain"] == full["exposure"]["small_groups_forced"]
    assert 0 < full["risk"]["risk"] < 1


def test_protect_composed_with_audit(cohort, tmp_path, capsys):
    table = tmp_path / "dp.parquet"
    rep = json.loads(
        run(
            capsys,
            "protect", cohort, "-o", table,
            "--merge", "race:Asian=Other", "--merge", "race:American Indian or Alaska Native=Other",
            "--dp-rho", "1", "--seed", "0", "--round", "10", "--audit",
        )
    )  # fmt: skip
    assert rep["provenance"]["parameters"]["steps"] == ["generalization", "dp", "rounding"]
    assert rep["audit"]["threat"] == "dp" and rep["audit"]["risk"]["certain"] == 0
    cnt = pd.read_parquet(table)["cnt"]
    assert (cnt % 10 == 0).all() and (cnt >= 10).all()


def test_protect_range_display(cohort, tmp_path, capsys):
    table = tmp_path / "range.csv"
    run(capsys, "protect", cohort, "-o", table, "--k", "10", "--show-range")
    cnt = pd.read_csv(table, dtype=str)["cnt"]
    assert (cnt == "1-9").any()
    rep = json.loads(run(capsys, "audit", table, "--k", "10"))
    assert rep["attack"]["exactly_determined"] > 0


def test_frontier(cohort, capsys):
    rep = json.loads(run(capsys, "frontier", cohort, "--reps", "2", "--quiet", "--workers", "2"))
    fams = {p["family"] for p in rep["points"]}
    assert fams == {"suppression", "rounding", "generalization", "dp"}
    assert rep["dp_seed"] == 0 and rep["recommended"]["id"] in {p["id"] for p in rep["points"]}
    assert any(p["pareto"] for p in rep["points"])
