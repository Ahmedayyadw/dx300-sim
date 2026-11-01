"""Lab 08d: eval at the Simscape inertia with front attachment and the real tau_up(theta). Eval only, never trains.

Models: rl/models/08b_{plain,residual}_w0.1_s{0,1,2}.zip, plus "pid" and "pid_ki" from 07 (as 08b_eval.py).
Changes vs the 08b baseline: J = J_with_front (about 262k kg m^2) and the torque limit
tau_max(theta) = interp of tau_up(theta), used for both directions (tau_down is not used).
Unchanged: gravity term (boom mass only, Lc), damping b, nominal feed-forward scale (5e5 N m),
10 substeps per 0.05 s command, scenario sets new100 (seed 123) and out100 (seed 321).
Both come from rl/boom_env_robust.py through RobustBoomEnv(randomize=False).

Usage: python rl/08d_eval_simscape_J.py [--substeps N]   (default 10)
Writes rl/08d_eval_simscape_J.csv (one row per controller, seed, set).
"""
import csv
import json
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from boom_env_robust import RobustBoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P

SUBSTEPS = int(sys.argv[sys.argv.index("--substeps") + 1]) if "--substeps" in sys.argv else 10
JSON_PATH = "matlab/lab08a/export/boom_params_lab08a.json"
CSV_PATH = "rl/08d_eval_simscape_J.csv"
MODEL_DIR = "rl/models"
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
W = 0.1
SEEDS = [0, 1, 2]
PIDS = {"pid": (0.5, 0.0, 0.25), "pid_ki": (2.0, 0.1, 1.0)}   # Kp, Ki, Kd tuned in 07

with open(JSON_PATH) as f:
    prm = json.load(f)
J_SIM = prm["J_with_front"]["value"]
TH_TAB = np.array(prm["tau_up"]["value"]["theta_rad"])
TAU_TAB = np.array(prm["tau_up"]["value"]["tau_Nm"])

rng = np.random.default_rng(123)    # same as 05/06/07/08b
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
OUT_TGT = [(-0.6, -0.3), (0.9, 1.0)]
rng = np.random.default_rng(321)    # same as 05b/08b
OUT = [(rng.uniform(-0.6, 1.0), rng.uniform(*OUT_TGT[i % 2])) for i in range(100)]
SETS = {"new100": NEW, "out100": OUT}


class SimscapeEnv(RobustBoomEnv):
    """RobustBoomEnv with J = J_with_front and tau_max(theta) = tau_up(theta) at the pre-step angle."""
    def step(self, action):
        self.tau_max = float(np.interp(self.th, TH_TAB, TAU_TAB))
        return super().step(action)


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


def sac_policy(model):
    return lambda obs, state: model.predict(obs, deterministic=True)[0]


def pid_policy(gains, u_ff_scale, dt):
    kp, ki, kd = gains

    def act(obs, state):   # same law as 07 / 08b
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])
        u_raw = kp * e + ki * state["integ"] + kd * (-om) + u_ff_scale * np.cos(th)
        if abs(u_raw) < 1.0:
            state["integ"] += e * dt
        return np.array([np.clip(u_raw, -1.0, 1.0)])
    return act


def run(act, env, th0, target):
    env.reset(seed=0)
    env.J = J_SIM
    env.th, env.target, env.om = th0, target, 0.0
    env.integ, env.pid_integ, env.k, env.energy_J = 0.0, 0.0, 0, 0.0
    env._measure()
    obs, state = env._obs(), {"integ": 0.0}
    for _ in range(env.max_steps):
        obs, _, _, trunc, _ = env.step(act(obs, state))
        if trunc:
            break
    return abs(np.degrees(env.target - env.th)), env.energy_J / 1000


jobs = [(c, W, s) for c in ("plain", "residual") for s in SEEDS] + [(p, "", 0) for p in PIDS]
rows = []
for controller, w, s in jobs:
    model = SAC.load(f"{MODEL_DIR}/08b_{controller}_w{w}_s{s}.zip", device="cpu") \
        if controller in ("plain", "residual") else None
    env = SimscapeEnv(w_energy=0.0 if w == "" else w, residual=(controller == "residual"),
                      randomize=False, substeps=SUBSTEPS)
    act = sac_policy(model) if model else pid_policy(PIDS[controller], env.u_ff_scale, env.dt)
    for set_name, scen in SETS.items():
        errs, kj = map(np.array, zip(*[run(act, env, *sc) for sc in scen]))
        rows.append({"controller": controller, "w": w, "seed": s, "set": set_name,
                     "mean_deg": errs.mean(), "max_deg": errs.max(),
                     "mean_cm": cm(errs).mean(), "max_cm": cm(errs).max(),
                     "mean_kJ": kj.mean(), "share_over_20cm": float((cm(errs) > 20).mean())})
    print(f"{controller:<8} w={w!s:<4} seed={s} done", flush=True)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

print(f"\nJ = {J_SIM:.0f} kg m^2, tau_up {TAU_TAB.min():.3g}..{TAU_TAB.max():.3g} N m, substeps {SUBSTEPS}")
print("| controller   | set    | mean cm | worst cm | mean kJ | % >20 cm |")
print("|--------------|--------|--------:|---------:|--------:|---------:|")
for c, w in [("plain", W), ("residual", W)] + [(p, "") for p in PIDS]:
    for set_name in SETS:
        rs = [r for r in rows if r["controller"] == c and r["set"] == set_name]
        print(f"| {c + (' w=' + str(w) if w != '' else ''):<12} | {set_name} | {np.mean([r['mean_cm'] for r in rs]):7.1f} |"
              f" {max(r['max_cm'] for r in rs):8.1f} | {np.mean([r['mean_kJ'] for r in rs]):7.1f} |"
              f" {np.mean([r['share_over_20cm'] for r in rs]) * 100:7.1f}% |")
print(f"\nCSV: {CSV_PATH}")
