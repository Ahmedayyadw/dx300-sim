"""Lab 12 (issue #7): train SAC on BoomEnvRealRobust (real plant, randomized), plain and residual.

Same layout and SAC settings as rl/08b_robust_train.py. Configs ["plain", "residual"] x w
[0.0, 0.1, 0.3] x seeds [0, 1, 2], 100k steps each (ASSUMED, as 08b). Saves
rl/models/12_{config}_w{w}_s{s}.zip and appends (config, w, seed, train_s) to
rl/12_robust_train_log.csv. Resumable: a model already saved is skipped.

Usage: python rl/12_robust_train.py [--dry-run] [--train-noise] [--config residual] [--w 0.1] [--seed 0]
  --config / --w / --seed  train only that subset (default: all)
  --dry-run                2,000 steps, residual, w=0.1, seed 0, saved as *_dry.zip, not logged
  --train-noise            Lab 12 step 3: BoomEnvRealRobust(train_noise=True), sensor noise per
                           episode from {0, 0.2, 0.5, 1.0} deg. Saved as rl/models/12n_{config}_w{w}_s{s}.zip
                           (the 12_* models are not touched), logged with config "{config}_noise".
  --base economical        Lab 12e: residual SAC on the economical PID point instead of balanced.
                           Saved as rl/models/12eco_residual_w{w}_s{s}.zip, logged as "residual_eco".
"""

import csv
import os
import sys
import time

from stable_baselines3 import SAC

from boom_env_real_robust import BoomEnvRealRobust


def arg(flag, cast, default):
    return [cast(sys.argv[sys.argv.index(flag) + 1])] if flag in sys.argv else default


DRY_RUN = "--dry-run" in sys.argv
TRAIN_NOISE = "--train-noise" in sys.argv
BASE = arg("--base", str, ["balanced"])[0]
assert BASE in {"balanced", "economical"}
assert not (TRAIN_NOISE and BASE != "balanced"), "--train-noise with --base is not supported"
PREFIX = "12n" if TRAIN_NOISE else ("12eco" if BASE == "economical" else "12")
CONFIGS = arg("--config", str, ["residual"] if DRY_RUN else ["plain", "residual"])
WEIGHTS = arg("--w", float, [0.1] if DRY_RUN else [0.0, 0.1, 0.3])
SEEDS = arg("--seed", int, [0] if DRY_RUN else [0, 1, 2])
STEPS = 2_000 if DRY_RUN else 100_000
SUFFIX = "_dry" if DRY_RUN else ""
MODEL_DIR = "rl/models"
LOG_PATH = "rl/12_robust_train_log.csv"
assert set(CONFIGS) <= {"plain", "residual"}
assert BASE == "balanced" or CONFIGS == ["residual"], "--base needs --config residual"

os.makedirs(MODEL_DIR, exist_ok=True)
for config in CONFIGS:
    for w in WEIGHTS:
        for s in SEEDS:
            name = f"{PREFIX}_{config}_w{w}_s{s}{SUFFIX}"
            path = f"{MODEL_DIR}/{name}.zip"
            if os.path.exists(path) and not DRY_RUN:
                print(f"{name}: exists, skipped")
                continue
            env = BoomEnvRealRobust(
                w_energy=w, residual=(config == "residual"), train_noise=TRAIN_NOISE, base=BASE
            )
            model = SAC("MlpPolicy", env, device="cpu", seed=s, verbose=0, tensorboard_log="rl/tb")
            t0 = time.time()
            model.learn(total_timesteps=STEPS, tb_log_name=name)
            train_s = time.time() - t0
            model.save(path)
            print(f"{name}: {STEPS} steps in {train_s:.0f} s -> {path}")
            if not DRY_RUN:
                new_log = not os.path.exists(LOG_PATH)
                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    if new_log:
                        writer.writerow(["config", "w", "seed", "train_s"])
                    tag = "_noise" if TRAIN_NOISE else ("_eco" if BASE == "economical" else "")
                    writer.writerow([config + tag, w, s, f"{train_s:.1f}"])
