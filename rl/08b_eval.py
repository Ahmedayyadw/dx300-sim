"""Lab 08b eval: robust-trained SAC (plain and PID residual) vs the 07 PIDs. Eval only, never trains.

Controllers: rl/models/08b_{plain,residual}_w{0.0,0.1,0.3}_s{0,1,2}.zip on RobustBoomEnv
(randomize=False, residual matching the config), plus "pid" and "pid_ki" from 07 (same gains,
nominal gravity feed-forward, anti-windup) on RobustBoomEnv(residual=False).
Sets: new100 (RNG seed 123, as 05/06/07) and out100 (as 05b, RNG seed 321).
Variants (as 06/06b), applied AFTER reset because RobustBoomEnv.reset() restores m and tau_max:
  baseline, m_x0.42, J_x0.38, tau_d, noise_0.2/0.5/1.0, payload_x1.2 (m *= 1.2 at step 100)
Noise lives inside the env (sigma_th_deg), so the policy observation and the residual PID see the
same noisy theta and omega. Error and energy use the true state.

Usage: python rl/08b_eval.py [--dry-run]   (dry run: seed 0, 5 scenarios per set, no CSV written)
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

DRY_RUN = "--dry-run" in sys.argv
CONFIGS = ["plain", "residual"]
WEIGHTS = [0.0, 0.1, 0.3]
SEEDS = [0] if DRY_RUN else [0, 1, 2]
PIDS = {"pid": (0.5, 0.0, 0.25), "pid_ki": (2.0, 0.1, 1.0)}   # Kp, Ki, Kd tuned in 07
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
CSV_PATH = "rl/08b_eval.csv"

rng = np.random.default_rng(123)    # same as 05/06/07
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
OUT_TGT = [(-0.6, -0.3), (0.9, 1.0)]  # as 05b: targets outside the 05 training range
rng = np.random.default_rng(321)    # same as 05b
OUT = [(rng.uniform(-0.6, 1.0), rng.uniform(*OUT_TGT[i % 2])) for i in range(100)]
SETS = {"new100": NEW[:5] if DRY_RUN else NEW, "out100": OUT[:5] if DRY_RUN else OUT}

VARIANTS = ["baseline", "m_x0.42", "J_x0.38", "tau_d", "noise_0.2", "noise_0.5", "noise_1.0",
            "payload_x1.2"]
J_NOM = RobustBoomEnv().J


def make_env(variant, w, residual):
    sigma = float(variant.split("_")[1]) if variant.startswith("noise") else 0.0
    return RobustBoomEnv(w_energy=w, residual=residual, randomize=False, sigma_th_deg=sigma)


def start(env, variant, th0, target):
    """Reset, apply the variant, set the scenario state; returns the first observation."""
    env.reset(seed=0)
    env.J = J_NOM
    if variant == "m_x0.42":
        env.m = env.m_nom * 0.42
    elif variant == "J_x0.38":
        env.J = J_NOM * 0.38
    elif variant == "tau_d":
        env.tau_d = True
    elif variant == "payload_x1.2":
        env.payload_k, env.payload_factor = 100, 1.2
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

    def act(obs, state):   # same law as 07, on the (noisy) observation
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
rows, sac_t, sac_n = [], 0.0, 0
for controller, w, s in jobs:
    t0 = time.time()
    model = SAC.load(f"{MODEL_DIR}/08b_{controller}_w{w}_s{s}.zip", device="cpu") \
        if controller in CONFIGS else None
    for set_name, scen in SETS.items():
        for variant in VARIANTS:
            env = make_env(variant, 0.0 if w == "" else w, residual=(controller == "residual"))
            act = sac_policy(model) if model else pid_policy(PIDS[controller], env.u_ff_scale, env.dt)
            errs, kj = map(np.array, zip(*[run(act, env, variant, *sc) for sc in scen]))
            rows.append({"controller": controller, "w": w, "seed": s, "set": set_name,
                         "variant": variant, "mean_deg": errs.mean(), "max_deg": errs.max(),
                         "mean_kJ": kj.mean(), "share_over_20cm": float((cm(errs) > 20).mean())})
    if model:
        sac_t += time.time() - t0; sac_n += 1
    print(f"{controller:<8} w={w!s:<4} seed={s} done in {time.time() - t0:5.1f} s")

if DRY_RUN:
    episodes = len(SETS) * len(VARIANTS) * len(next(iter(SETS.values())))
    per_ep = sac_t / (sac_n * episodes)
    full_s = per_ep * len(CONFIGS) * len(WEIGHTS) * 3 * len(SETS) * len(VARIANTS) * 100
    print(f"\nDRY RUN: seeds {SEEDS}, {episodes // (len(SETS) * len(VARIANTS))} scenarios per set,"
          f" {CSV_PATH} not written")
    print(f"SAC: {per_ep * 1000:.0f} ms per episode -> full run about {full_s / 60:.0f} min")
else:
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

# controller labels; aggregate over seeds: mean of means, worst of maxima
LABELS = [(c, w, f"{c} w={w}") for c in CONFIGS for w in WEIGHTS] + [(p, "", p) for p in PIDS]


def agg(c, w, set_name, variant):
    rs = [r for r in rows if r["controller"] == c and r["w"] == w and r["set"] == set_name
          and r["variant"] == variant]
    return (np.mean([r["mean_deg"] for r in rs]), max(r["max_deg"] for r in rs),
            np.mean([r["mean_kJ"] for r in rs]), np.mean([r["share_over_20cm"] for r in rs]) * 100)


print("\nHeadline (new100 baseline, out100 baseline, worst variant on new100)")
print("| controller     | mean cm | worst cm | mean kJ | % >20 cm | out100 % >20 | worst variant % >20 |")
print("|----------------|--------:|---------:|--------:|---------:|-------------:|--------------------:|")
for c, w, label in LABELS:
    m, x, k, o = agg(c, w, "new100", "baseline")
    o_out = agg(c, w, "out100", "baseline")[3]
    worst = max(VARIANTS, key=lambda v: agg(c, w, "new100", v)[3])
    print(f"| {label:<14} | {cm(m):7.1f} | {cm(x):8.1f} | {k:7.1f} | {o:7.1f}% | {o_out:11.1f}% |"
          f" {agg(c, w, 'new100', worst)[3]:6.1f}% {worst:<12} |")

print("\nFull")
print("| controller     | set    | variant      | mean deg (cm) | max deg (cm)  | mean kJ | % >20 cm |")
print("|----------------|--------|--------------|---------------|---------------|--------:|---------:|")
for c, w, label in LABELS:
    for set_name in SETS:
        for variant in VARIANTS:
            m, x, k, o = agg(c, w, set_name, variant)
            print(f"| {label:<14} | {set_name} | {variant:<12} | {m:5.2f} ({cm(m):5.1f}) |"
                  f" {x:5.2f} ({cm(x):5.1f}) | {k:7.1f} | {o:7.1f}% |")
if not DRY_RUN:
    print(f"\nCSV: {CSV_PATH}")
