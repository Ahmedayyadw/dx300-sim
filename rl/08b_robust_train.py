"""Lab 08b: train SAC on RobustBoomEnv (full range, randomized physics), plain and PID residual.

Configs ["plain", "residual"] x w [0.0, 0.1, 0.3] x seeds [0, 1, 2], 100k steps each, SAC settings
as in 05. Saves rl/models/08b_{config}_w{w}_s{s}.zip and appends (config, w, seed, train_s) to
rl/08b_train_log.csv. Resumable: a model already saved is skipped.

Usage: python rl/08b_robust_train.py [--dry-run]
  --dry-run: 2,000 steps, plain, w=0.1, seed 0, saved as 08b_plain_w0.1_s0_dry.zip, not logged
"""
import csv
import os
import sys
import time
from stable_baselines3 import SAC
from boom_env_robust import RobustBoomEnv

DRY_RUN = "--dry-run" in sys.argv
CONFIGS = ["plain"] if DRY_RUN else ["plain", "residual"]
WEIGHTS = [0.1] if DRY_RUN else [0.0, 0.1, 0.3]
SEEDS = [0] if DRY_RUN else [0, 1, 2]
STEPS = 2_000 if DRY_RUN else 100_000
SUFFIX = "_dry" if DRY_RUN else ""
MODEL_DIR = "rl/models"
LOG_PATH = "rl/08b_train_log.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
for config in CONFIGS:
    for w in WEIGHTS:
        for s in SEEDS:
            name = f"08b_{config}_w{w}_s{s}{SUFFIX}"
            path = f"{MODEL_DIR}/{name}.zip"
            if os.path.exists(path) and not DRY_RUN:
                print(f"{name}: exists, skipped")
                continue
            env = RobustBoomEnv(w_energy=w, residual=(config == "residual"))
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
                    writer.writerow([config, w, s, f"{train_s:.1f}"])
