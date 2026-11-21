"""Template for new labs: Hydra config + fixed seeds + run logging. Copy it to rl/NN_name.py.

What you get for free:
  - all parameters in configs/<name>.yaml, overridable from the command line (no hardcoded values)
  - python / numpy / torch / env seeded from cfg.seed
  - runs/<run_id>/{config,meta,metrics}.json + one row in runs/runs.csv (git commit, seed, results)
  - optional W&B mirror with DX300_WANDB=1 (offline by default)

Usage (repo root, .venv active):
  python rl/00_lab_template.py                         smoke test, 2,000 steps
  python rl/00_lab_template.py train.steps=30000 seed=1
  python rl/00_lab_template.py -m seed=0,1,2           one run per seed
"""

import os
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from boom_env import BoomEnv  # noqa: E402

from dx300sim.runlog import set_global_seed, start_run  # noqa: E402


def evaluate(model, env, n, seed):
    """Mean final tracking error [deg] and energy [kJ] over n scenarios from one RNG seed."""
    errs, energy = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        errs.append(abs(np.degrees(env.target - env.th)))
        energy.append(env.energy_J / 1e3)
    return float(np.mean(errs)), float(np.mean(energy))


@hydra.main(version_base=None, config_path="../configs", config_name="lab_template")
def main(cfg: DictConfig):
    from stable_baselines3 import SAC

    set_global_seed(cfg.seed)
    run = start_run(cfg.name, cfg, seed=cfg.seed)
    env = BoomEnv(w_energy=cfg.env.w_energy)
    env.reset(seed=cfg.seed)
    model = SAC("MlpPolicy", env, device=cfg.train.device, seed=cfg.seed, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=cfg.train.steps)
    train_s = time.time() - t0
    err, kj = evaluate(
        model, BoomEnv(w_energy=cfg.env.w_energy), cfg.eval.n_scenarios, cfg.eval.scenario_seed
    )
    model.save(str(run.dir / "model"))
    run.finish({"error_deg": err, "energy_kJ": kj, "train_s": round(train_s, 1)})
    print(f"{run.run_id}: error {err:.2f} deg, energy {kj:.1f} kJ, {train_s:.0f} s -> {run.dir}")


if __name__ == "__main__":
    main()
