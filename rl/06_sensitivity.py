"""Step 9: parameter sensitivity. Saved 05 models re-evaluated under DX300-based physics changes.

Eval only, never trains: loads rl/models/sac_w{w}_s{s}.zip from 05 and evaluates each on the same
100 unseen scenarios (RNG seed 123) under four variants, each applied to a fresh BoomEnv
(see docs/param_audit.md for the factors):
  baseline   default boom_env (should match the new100 rows of 05)
  m_x0.42    gravity torque at boom-only mass, env.m *= 0.42
  J_x0.38    uniform-rod inertia estimate, env.J *= 0.38
  tau_d      tau_max = 5e5 x |d(theta)| / |d(0)|, d = cylinder moment arm from boom_geometry

Angle convention: boom_geometry theta is the angle of line A->D from horizontal, the same as
boom_env theta (gravity uses cos(theta)). THETA_REF_RAD is only the drawing pose used to derive
DELTA_RAD, not an offset, so theta is passed through unchanged.

Usage: python rl/06_sensitivity.py [--dry-run]   (dry run: seed 0, 5 scenarios, no CSV written)
"""
import csv
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from boom_env import BoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P
from dx300sim.boom_geometry import moment_arm

DRY_RUN = "--dry-run" in sys.argv
WEIGHTS = [0.0, 0.1, 0.3]
SEEDS = [0] if DRY_RUN else [0, 1, 2]
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
CSV_PATH = "rl/06_sensitivity.csv"

rng = np.random.default_rng(123)    # same 100 unseen scenarios as 05
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
SCEN = NEW[:5] if DRY_RUN else NEW


def d(theta):
    return moment_arm(theta, P.B_XY_M, P.C_M, P.DELTA_RAD)


D0_M = abs(d(0.0))   # about 0.898 m


class TauDBoomEnv(BoomEnv):
    """BoomEnv with angle-dependent torque limit tau_max = 5e5 x |d(theta)| / |d(0)|."""

    def step(self, action):
        self.tau_max = 5e5 * abs(d(self.th)) / D0_M   # limit at the pre-step angle
        return super().step(action)


def make_m(w):
    env = BoomEnv(w_energy=w); env.m *= 0.42; return env


def make_J(w):
    env = BoomEnv(w_energy=w); env.J *= 0.38; return env


VARIANTS = {
    "baseline": lambda w: BoomEnv(w_energy=w),
    "m_x0.42": make_m,
    "J_x0.38": make_J,
    "tau_d": lambda w: TauDBoomEnv(w_energy=w),
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
print("\n| weight | variant  | mean deg (cm) | max deg (cm)  | mean kJ | % >20 cm |")
print("|--------|----------|---------------|---------------|--------:|---------:|")
for w in WEIGHTS:
    for name in VARIANTS:
        rs = [r for r in rows if r["w"] == w and r["variant"] == name]
        m = np.mean([r["mean_deg"] for r in rs]); x = max(r["max_deg"] for r in rs)
        k = np.mean([r["mean_kJ"] for r in rs]); o = np.mean([r["share_over_20cm"] for r in rs]) * 100
        print(f"| {w:<6} | {name:<8} | {m:5.2f} ({cm(m):5.1f}) | {x:5.2f} ({cm(x):5.1f}) |"
              f" {k:7.1f} | {o:7.1f}% |")
if not DRY_RUN:
    print(f"\nCSV: {CSV_PATH}")
