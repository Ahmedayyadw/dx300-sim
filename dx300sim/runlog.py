"""Reproducible run logging: seed everything, record what produced each result.

Every run gets its own folder runs/<YYYYmmdd-HHMMSS>_<name>/ (gitignored) with:
  config.json   the exact parameters of the run
  meta.json     git commit, uncommitted changes y/n, command line, python + package versions, seed
  metrics.json  the final numbers (written by finish())
and one row is appended to runs/runs.csv (committed), so any figure or table can be traced back to
code + config + seed.

Optional Weights & Biases mirror: set DX300_WANDB=1 (default mode offline, so nothing leaves the
laptop; `wandb sync runs/wandb/...` uploads later). Without the variable, or without wandb
installed, nothing W&B-related happens.

Usage:
    from dx300sim.runlog import set_global_seed, start_run
    set_global_seed(cfg["seed"])
    run = start_run("14_sac", cfg, seed=cfg["seed"])
    ...train / evaluate...
    run.finish({"error_deg": 0.42, "energy_kJ": 12.3})
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
CSV_FIELDS = [
    "run_id",
    "name",
    "seed",
    "git_commit",
    "git_dirty",
    "started",
    "duration_s",
    "config",
    "metrics",
]
PACKAGES = ["numpy", "scipy", "torch", "gymnasium", "stable_baselines3", "d3rlpy", "hydra-core"]


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed). Env seeds still go to env.reset(seed=...)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _versions() -> dict:
    v = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return v


def _jsonable(cfg) -> dict:
    """Accept a plain dict or an OmegaConf DictConfig (Hydra)."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


@dataclass
class Run:
    run_id: str
    name: str
    seed: int
    dir: Path
    config: dict
    meta: dict
    t0: float = field(default_factory=time.time)
    _wandb: object = None
    finished: bool = False

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Intermediate metrics; only mirrored to W&B (the CSV keeps the final ones)."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self, metrics: dict) -> None:
        if self.finished:
            raise RuntimeError(f"run {self.run_id} already finished")
        duration = round(time.time() - self.t0, 1)
        (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        csv_path = self.dir.parent / "runs.csv"
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "run_id": self.run_id,
                    "name": self.name,
                    "seed": self.seed,
                    "git_commit": self.meta["git_commit"],
                    "git_dirty": self.meta["git_dirty"],
                    "started": self.meta["started"],
                    "duration_s": duration,
                    "config": json.dumps(self.config, sort_keys=True),
                    "metrics": json.dumps(metrics, sort_keys=True),
                }
            )
        if self._wandb is not None:
            self._wandb.summary.update(metrics)
            self._wandb.finish()
        self.finished = True


def start_run(name: str, config, seed: int, runs_dir: Path | str | None = None) -> Run:
    """Create the run folder, write config.json + meta.json, optionally start W&B."""
    runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{seed}"
    run_dir = runs_dir / run_id
    i = 1
    while run_dir.exists():  # two runs in the same second
        run_dir = runs_dir / f"{run_id}_{i}"
        i += 1
    run_id = run_dir.name
    run_dir.mkdir(parents=True)
    cfg = _jsonable(config)
    meta = {
        "run_id": run_id,
        "name": name,
        "seed": seed,
        "started": started,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": " ".join([sys.executable, *sys.argv]),
        "host": platform.node(),
        "versions": _versions(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if meta["git_dirty"]:
        print(
            f"[runlog] warning: uncommitted changes, {run_id} is not exactly"
            f" reproducible from {meta['git_commit'][:8]}"
        )

    wb = None
    if os.environ.get("DX300_WANDB") == "1":
        try:
            import wandb

            wb = wandb.init(
                project=os.environ.get("DX300_WANDB_PROJECT", "dx300-sim"),
                name=run_id,
                config={**cfg, "seed": seed, "git_commit": meta["git_commit"]},
                mode=os.environ.get("WANDB_MODE", "offline"),
                dir=str(runs_dir),
            )
        except ImportError:
            print("[runlog] DX300_WANDB=1 but wandb is not installed, skipping W&B")
    return Run(run_id, name, seed, run_dir, cfg, meta, _wandb=wb)
