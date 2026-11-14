"""Lab 12h: longer SAC training, to test whether the Lab 12 models were undertrained.

Same env and SAC settings as rl/12_robust_train.py (BoomEnvRealRobust, residual on the balanced
PID point, SAC defaults, device cpu), one config: w = 0.03 (best accuracy in the Lab 12d sweep),
seeds 0, 1, 2, 300k steps instead of 100k (ASSUMED budget, 3x Lab 12).

  checkpoints  every 50k steps: rl/models/12h_residual_w0.03_s{s}_{n}_steps.zip, with the replay
               buffer (..._replay_buffer_{n}_steps.pkl) so a crash costs at most 50k steps.
  resume       re-running the script continues each seed from its last checkpoint (model, replay
               buffer and step count); a seed with its 300k checkpoint is skipped.
  curve        every finished training episode (200 steps) is appended to
               rl/12h_train_curve.csv: seed, timesteps, episode reward.
  log          wall-clock per seed run to rl/12h_train_log.csv (seed, from_step, to_step, train_s).

Checkpoints are evaluated on the Lab 12 protocol by rl/12h_long_eval.py.

Usage (repo root, .venv active):
  python rl/12h_long_train.py              all 3 seeds in turn (about 4-5 h)
  python rl/12h_long_train.py --seed 1     one seed
  python rl/12h_long_train.py --dry-run    2,000 steps, checkpoint every 1,000, *_dry files
"""

import csv
import glob
import os
import re
import sys
import time

from boom_env_real_robust import BoomEnvRealRobust
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

DRY_RUN = "--dry-run" in sys.argv
W = 0.03
SEEDS = [int(sys.argv[sys.argv.index("--seed") + 1])] if "--seed" in sys.argv else [0, 1, 2]
STEPS = 2_000 if DRY_RUN else 300_000
EVERY = 1_000 if DRY_RUN else 50_000
SUFFIX = "_dry" if DRY_RUN else ""
MODEL_DIR = "rl/models"
CURVE = f"rl/12h_train_curve{SUFFIX}.csv"
LOG = f"rl/12h_train_log{SUFFIX}.csv"


def prefix(seed):
    return f"12h_residual_w{W}_s{seed}{SUFFIX}"


def ckpt(seed, n):
    return f"{MODEL_DIR}/{prefix(seed)}_{n}_steps.zip"


def buffer(seed, n):
    return f"{MODEL_DIR}/{prefix(seed)}_replay_buffer_{n}_steps.pkl"


def last_checkpoint(seed):
    steps = [
        int(m.group(1))
        for p in glob.glob(f"{MODEL_DIR}/{prefix(seed)}_*_steps.zip")
        if (m := re.search(r"_(\d+)_steps\.zip$", p))
    ]
    steps = [n for n in steps if os.path.exists(buffer(seed, n))]
    return max(steps, default=0)


def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


class CurveCallback(BaseCallback):
    """Appends (seed, timesteps, episode reward) for every finished episode (Monitor info)."""

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                append(
                    CURVE,
                    ["seed", "timesteps", "ep_reward"],
                    [self.seed, self.num_timesteps, f"{ep['r']:.4f}"],
                )
        return True


def train(seed):
    done = last_checkpoint(seed)
    if done >= STEPS:
        print(f"{prefix(seed)}: {done} steps done, skipped")
        return
    env = BoomEnvRealRobust(w_energy=W, residual=True)
    if done:
        model = SAC.load(ckpt(seed, done), env=env, device="cpu", tensorboard_log="rl/tb")
        model.load_replay_buffer(buffer(seed, done))
        print(f"{prefix(seed)}: resuming at {done} steps")
    else:
        model = SAC("MlpPolicy", env, device="cpu", seed=seed, verbose=0, tensorboard_log="rl/tb")
    callbacks = CallbackList(
        [
            CheckpointCallback(EVERY, MODEL_DIR, name_prefix=prefix(seed), save_replay_buffer=True),
            CurveCallback(seed),
        ]
    )
    t0 = time.time()
    model.learn(
        total_timesteps=STEPS - done,
        callback=callbacks,
        tb_log_name=prefix(seed),
        reset_num_timesteps=(done == 0),
    )
    train_s = time.time() - t0
    append(
        LOG,
        ["seed", "from_step", "to_step", "train_s"],
        [seed, done, model.num_timesteps, f"{train_s:.1f}"],
    )
    print(f"{prefix(seed)}: {done} -> {model.num_timesteps} steps in {train_s:.0f} s")
    for n in range(EVERY, STEPS, EVERY):  # keep the models, drop all but the last buffer
        if os.path.exists(buffer(seed, n)):
            os.remove(buffer(seed, n))


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    for s in SEEDS:
        train(s)
