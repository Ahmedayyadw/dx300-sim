"""Step 9: parameter sensitivity. Saved 05 models re-evaluated under DX300-based physics changes.

No training: loads rl/models/sac_w{w}_s{s}.zip from 05 and evaluates each on the same 100 unseen
scenarios under four env variants (see docs/param_audit.md for the factors):
  baseline   default boom_env (should match the new100 rows of 05)
  m_x0.42    gravity torque at boom-only mass, 0.42 x m
  J_x0.38    uniform-rod inertia estimate, 0.38 x J
  tau_d      tau_max x d(theta) / 0.90, d = cylinder moment arm from boom_geometry
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

WEIGHTS = [0.0, 0.1, 0.3]
SEEDS = [0, 1, 2]
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
D_REF_M = 0.90   # max moment arm over the stroke (0.8995 m), so tau_d <= default tau_max
MODEL_DIR = "rl/models"
CSV_PATH = "rl/06_sensitivity.csv"

rng = np.random.default_rng(123)    # same 100 unseen scenarios as 05
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]

_default = BoomEnv()


def tau_d(theta):
    return _default.tau_max * moment_arm(theta, P.B_XY_M, P.C_M, P.DELTA_RAD) / D_REF_M


VARIANTS = {
    "baseline": {},
    "m_x0.42": {"m": _default.m * 0.42},
    "J_x0.38": {"J": _default.J * 0.38},
    "tau_d": {"tau_max_fn": tau_d},
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
        for name, kwargs in VARIANTS.items():
            env = BoomEnv(w_energy=w, **kwargs)
            errs, kj = map(np.array, zip(*[run(model, env, *sc) for sc in NEW]))
            rows.append({"w_energy": w, "seed": s, "variant": name, "n": len(NEW),
                         "mean_deg": errs.mean(), "p95_deg": np.percentile(errs, 95),
                         "max_deg": errs.max(), "mean_kJ": kj.mean(),
                         "share_over_20cm": float((cm(errs) > 20).mean())})
        print(f"w={w:<4} seed={s} done")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

# mean over seeds, except max = worst move over all seeds (same as 05)
print("\n| weight | variant  | mean kJ | p95 cm | max cm | % >20 cm |")
print("|--------|----------|--------:|-------:|-------:|---------:|")
for w in WEIGHTS:
    for name in VARIANTS:
        rs = [r for r in rows if r["w_energy"] == w and r["variant"] == name]
        k = np.mean([r["mean_kJ"] for r in rs]); p = np.mean([r["p95_deg"] for r in rs])
        x = max(r["max_deg"] for r in rs); o = np.mean([r["share_over_20cm"] for r in rs]) * 100
        print(f"| {w:<6} | {name:<8} | {k:7.1f} | {cm(p):6.1f} | {cm(x):6.1f} | {o:7.1f}% |")
print(f"\nCSV: {CSV_PATH}")
