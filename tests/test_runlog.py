import csv
import json
import random

import numpy as np

from dx300sim.runlog import set_global_seed, start_run


def test_seed_is_reproducible():
    set_global_seed(7)
    a = (random.random(), np.random.rand())
    set_global_seed(7)
    assert (random.random(), np.random.rand()) == a


def test_run_writes_files_and_csv(tmp_path):
    cfg = {"seed": 3, "env": {"w_energy": 0.05}}
    run = start_run("unit", cfg, seed=3, runs_dir=tmp_path)
    run.finish({"error_deg": 0.5})
    assert json.loads((run.dir / "config.json").read_text()) == cfg
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["seed"] == 3 and "numpy" in meta["versions"]
    assert json.loads((run.dir / "metrics.json").read_text()) == {"error_deg": 0.5}
    rows = list(csv.DictReader((tmp_path / "runs.csv").open()))
    assert (
        len(rows) == 1
        and rows[0]["name"] == "unit"
        and json.loads(rows[0]["metrics"])["error_deg"] == 0.5
    )


def test_two_runs_same_second_get_separate_folders(tmp_path):
    a = start_run("x", {}, seed=0, runs_dir=tmp_path)
    b = start_run("x", {}, seed=0, runs_dir=tmp_path)
    assert a.dir != b.dir
    a.finish({})
    b.finish({})
    assert len(list(csv.DictReader((tmp_path / "runs.csv").open()))) == 2


def test_hydra_config_is_accepted(tmp_path):
    from omegaconf import OmegaConf

    run = start_run("h", OmegaConf.create({"a": {"b": 1}}), seed=0, runs_dir=tmp_path)
    assert json.loads((run.dir / "config.json").read_text()) == {"a": {"b": 1}}
    run.finish({})
