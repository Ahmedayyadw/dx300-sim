"""Lab 08c: numerical check of the J_x0.38 chattering in 08b. Eval only, never trains.

Re-runs the 08b models (plain and residual, w = 0, 0.1, 0.3, seeds 0..2) and both 07 PIDs on new100
for the variants baseline and J_x0.38, with RobustBoomEnv(substeps=1) and (substeps=10). substeps = 10
holds each command for the 0.05 s control step but integrates it with ten 0.005 s Euler steps.
Reset and variant handling are the same as rl/08b_eval.py.

First check: substeps = 1 must reproduce rl/08b_eval.csv exactly (same rows, same floats).

Usage: python rl/08c_substep_check.py
"""
import csv
import os
import sys
import time
import numpy as np
from stable_baselines3 import SAC
from boom_env_robust import RobustBoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P

CONFIGS = ["plain", "residual"]
WEIGHTS = [0.0, 0.1, 0.3]
SEEDS = [0, 1, 2]
PIDS = {"pid": (0.5, 0.0, 0.25), "pid_ki": (2.0, 0.1, 1.0)}   # Kp, Ki, Kd tuned in 07
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
REF_CSV = "rl/08b_eval.csv"
VARIANTS = ["baseline", "J_x0.38"]
SUBSTEPS = [1, 10]

rng = np.random.default_rng(123)    # same as 05/06/07/08b
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
J_NOM = RobustBoomEnv().J


def start(env, variant, th0, target):
    """Reset, apply the variant, set the scenario state; returns the first observation (as 08b)."""
    env.reset(seed=0)
    env.J = J_NOM
    if variant == "J_x0.38":
        env.J = J_NOM * 0.38
    env.th, env.target, env.om = th0, target, 0.0
    env.integ, env.pid_integ, env.k, env.energy_J = 0.0, 0.0, 0, 0.0
    env._measure()
    return env._obs()


def sac_policy(model):
    def act(obs, state):
        return model.predict(obs, deterministic=True)[0]
    return act


def pid_policy(gains, u_ff_scale, dt):
    kp, ki, kd = gains

    def act(obs, state):   # same law as 07/08b
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])
        u_raw = kp * e + ki * state["integ"] + kd * (-om) + u_ff_scale * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup
            state["integ"] += e * dt
        return np.array([np.clip(u_raw, -1.0, 1.0)])
    return act


def run(act, env, variant, th0, target):
    obs, state = start(env, variant, th0, target), {"integ": 0.0}
    for _ in range(env.max_steps):
        obs, _, _, trunc, _ = env.step(act(obs, state))
        if trunc:
            break
    return abs(np.degrees(env.target - env.th)), env.energy_J / 1000


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


jobs = [(c, w, s) for c in CONFIGS for w in WEIGHTS for s in SEEDS] + [(p, "", 0) for p in PIDS]
rows = []
for controller, w, s in jobs:
    t0 = time.time()
    model = SAC.load(f"{MODEL_DIR}/08b_{controller}_w{w}_s{s}.zip", device="cpu") \
        if controller in CONFIGS else None
    for variant in VARIANTS:
        for n in SUBSTEPS:
            env = RobustBoomEnv(w_energy=0.0 if w == "" else w, residual=(controller == "residual"),
                                randomize=False, substeps=n)
            act = sac_policy(model) if model else pid_policy(PIDS[controller], env.u_ff_scale, env.dt)
            errs, kj = map(np.array, zip(*[run(act, env, variant, *sc) for sc in NEW]))
            rows.append({"controller": controller, "w": w, "seed": s, "variant": variant,
                         "substeps": n, "mean_deg": errs.mean(), "max_deg": errs.max(),
                         "mean_kJ": kj.mean(), "share_over_20cm": float((cm(errs) > 20).mean())})
    print(f"{controller:<8} w={w!s:<4} seed={s} done in {time.time() - t0:5.1f} s")

# check 1: substeps = 1 vs 08b (new100, same controller / w / seed / variant)
with open(REF_CSV) as f:
    ref = {(r["controller"], r["w"], r["seed"], r["variant"]): r for r in csv.DictReader(f)
           if r["set"] == "new100"}
keys = ["mean_deg", "max_deg", "mean_kJ", "share_over_20cm"]
worst, exact = 0.0, True
for r in rows:
    if r["substeps"] != 1:
        continue
    rr = ref[(r["controller"], str(r["w"]), str(r["seed"]), r["variant"])]
    for k in keys:
        exact &= float(rr[k]) == r[k]
        worst = max(worst, abs(float(rr[k]) - r[k]))
n1 = sum(r["substeps"] == 1 for r in rows)
print(f"\nCheck substeps=1 vs {REF_CSV}: {n1} rows, "
      f"{'EXACT match' if exact else 'MISMATCH'} (max abs diff {worst:.3g})")

# table: aggregate over seeds as 08b (mean of means, worst of maxima)
LABELS = [(c, w, f"{c} w={w}") for c in CONFIGS for w in WEIGHTS] + [(p, "", p) for p in PIDS]


def agg(c, w, variant, n):
    rs = [r for r in rows if r["controller"] == c and r["w"] == w and r["variant"] == variant
          and r["substeps"] == n]
    return (np.mean([r["mean_deg"] for r in rs]), max(r["max_deg"] for r in rs),
            np.mean([r["mean_kJ"] for r in rs]), np.mean([r["share_over_20cm"] for r in rs]) * 100)


print("\nnew100, seeds aggregated (mean of means, worst of maxima)")
print("| controller     | variant  | substeps | mean cm | worst cm | mean kJ | % >20 cm |")
print("|----------------|----------|---------:|--------:|---------:|--------:|---------:|")
for c, w, label in LABELS:
    for variant in VARIANTS:
        for n in SUBSTEPS:
            m, x, k, o = agg(c, w, variant, n)
            print(f"| {label:<14} | {variant:<8} | {n:8d} | {cm(m):7.1f} | {cm(x):8.1f} |"
                  f" {k:7.1f} | {o:7.1f}% |")
