"""Step 9b: sensor noise and payload step. Saved 05 models re-evaluated, same layout as 06.

Eval only, never trains: loads rl/models/sac_w{w}_s{s}.zip from 05 and evaluates each on the same
100 unseen scenarios (RNG seed 123) under five variants, each on a fresh BoomEnv:
  baseline      default boom_env (should match the baseline rows of 06)
  noise_0.2     Gaussian noise on observed theta and omega, sigma_th = 0.2 deg
  noise_0.5     same, sigma_th = 0.5 deg
  noise_1.0     same, sigma_th = 1.0 deg
  payload_x1.2  bucket filling: env.m *= 1.2 from step 100 on (mass reset each episode)

Noise: sigma_om = sigma_th / dt x 0.1 (dt = 0.05 s), applied to the policy's observation only.
The observed error target - theta uses the noisy theta. Accuracy and energy use the true state.
Each noisy env has its own noise RNG (seed 0), so runs are reproducible.

Usage: python rl/06b_noise.py [--dry-run]   (dry run: seed 0, 5 scenarios, no CSV written)
"""
import csv
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from boom_env import BoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P

DRY_RUN = "--dry-run" in sys.argv
WEIGHTS = [0.0, 0.1, 0.3]
SEEDS = [0] if DRY_RUN else [0, 1, 2]
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
CSV_PATH = "rl/06b_noise.csv"

rng = np.random.default_rng(123)    # same 100 unseen scenarios as 05 and 06
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
SCEN = NEW[:5] if DRY_RUN else NEW


class NoisyBoomEnv(BoomEnv):
    """BoomEnv whose observation has Gaussian noise on theta and omega."""

    def __init__(self, sigma_th_deg, **kwargs):
        super().__init__(**kwargs)
        self.sigma_th = np.radians(sigma_th_deg)
        self.sigma_om = self.sigma_th / self.dt * 0.1
        self.noise_rng = np.random.default_rng(0)

    def _obs(self):
        th = self.th + self.noise_rng.normal(0.0, self.sigma_th)
        om = self.om + self.noise_rng.normal(0.0, self.sigma_om)
        return np.array([th, om, self.target - th], dtype=np.float32)


class PayloadBoomEnv(BoomEnv):
    """BoomEnv whose mass steps to 1.2 x m at step 100 (bucket filling)."""

    def reset(self, seed=None, options=None):
        if hasattr(self, "m0"):
            self.m = self.m0                       # undo the previous episode's step
        self.m0 = self.m
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.k == 100:
            self.m *= 1.2
        return super().step(action)


VARIANTS = {
    "baseline": lambda w: BoomEnv(w_energy=w),
    "noise_0.2": lambda w: NoisyBoomEnv(0.2, w_energy=w),
    "noise_0.5": lambda w: NoisyBoomEnv(0.5, w_energy=w),
    "noise_1.0": lambda w: NoisyBoomEnv(1.0, w_energy=w),
    "payload_x1.2": lambda w: PayloadBoomEnv(w_energy=w),
}


def run(model, env, th0, target):
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs = env._obs()
    for _ in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, _ = env.step(a)
        if trunc:
            break
    return abs(np.degrees(env.target - env.th)), env.energy_J / 1000


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


rows = []
for w in WEIGHTS:
    for s in SEEDS:
        model = SAC.load(f"{MODEL_DIR}/sac_w{w}_s{s}.zip", device="cpu")
        for name, make in VARIANTS.items():
            env = make(w)
            errs, kj = map(np.array, zip(*[run(model, env, *sc) for sc in SCEN]))
            rows.append({"variant": name, "w": w, "seed": s,
                         "mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": kj.mean(),
                         "share_over_20cm": float((cm(errs) > 20).mean())})
        print(f"w={w:<4} seed={s} done")

if DRY_RUN:
    print(f"\nDRY RUN: seeds {SEEDS}, {len(SCEN)} scenarios, {CSV_PATH} not written")
    print(",".join(rows[0]))
    for r in rows:
        print(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()))
else:
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

# mean over seeds, except max = worst move over all seeds (same as 05)
print("\n| weight | variant      | mean deg (cm) | max deg (cm)  | mean kJ | % >20 cm |")
print("|--------|--------------|---------------|---------------|--------:|---------:|")
for w in WEIGHTS:
    for name in VARIANTS:
        rs = [r for r in rows if r["w"] == w and r["variant"] == name]
        m = np.mean([r["mean_deg"] for r in rs]); x = max(r["max_deg"] for r in rs)
        k = np.mean([r["mean_kJ"] for r in rs]); o = np.mean([r["share_over_20cm"] for r in rs]) * 100
        print(f"| {w:<6} | {name:<12} | {m:5.2f} ({cm(m):5.1f}) | {x:5.2f} ({cm(x):5.1f}) |"
              f" {k:7.1f} | {o:7.1f}% |")
if not DRY_RUN:
    print(f"\nCSV: {CSV_PATH}")
