"""Lab 05b: extrapolation test + generalization plot. Eval only, no training.

05 tested 100 unseen scenarios drawn from the SAME ranges as training (interpolation).
This adds 100 scenarios whose targets lie OUTSIDE the training range (still inside the
mechanical end stops), evaluates the saved 05 models on all three sets, and plots
error (cm) and energy (kJ) vs target angle for the operating point w=0.1.

Needs the models saved by 05 in rl/models/ (sac_w{w}_s{s}.zip). Missing models are skipped.
"""
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from boom_env import BoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim.dx300_params import BOOM_LENGTH_OFFICIAL_M as BOOM_LENGTH_M

CANDIDATES = [0.0, 0.1, 0.3, 0.5]
SEEDS = [0, 1, 2]
W_OP = 0.1                                  # operating point
MODEL_DIR = "rl/models"
CSV_PATH = "rl/05b_extrapolation.csv"
PNG_PATH = "rl/05b_generalization.png"

TRAIN_TH0 = (-0.4, 0.9)                     # rad, start range used in training (BoomEnv.reset)
TRAIN_TGT = (-0.3, 0.9)                     # rad, target range used in training
OUT_TGT = [(-0.6, -0.3), (0.9, 1.0)]        # rad, outside training, inside end stops (-0.824, 1.054)

rng = np.random.default_rng(7)              # same as 03/04/05
ORIG = [(rng.uniform(*TRAIN_TH0), rng.uniform(*TRAIN_TGT)) for _ in range(20)]
rng = np.random.default_rng(123)            # same as 05
NEW = [(rng.uniform(*TRAIN_TH0), rng.uniform(*TRAIN_TGT)) for _ in range(100)]
rng = np.random.default_rng(321)            # new: out-of-range targets, half low / half high
OUT = [(rng.uniform(-0.6, 1.0), rng.uniform(*OUT_TGT[i % 2])) for i in range(100)]
SETS = [("orig20", ORIG), ("new100", NEW), ("out100", OUT)]


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


rows, per_move = [], []                     # per_move: w=W_OP only, for the plot
for w in CANDIDATES:
    for s in SEEDS:
        path = f"{MODEL_DIR}/sac_w{w}_s{s}.zip"
        if not os.path.exists(path):
            print(f"skip w={w} seed={s}: {path} not found (run 05 first)")
            continue
        env = BoomEnv(w_energy=w)
        model = SAC.load(path, env=env, device="cpu")
        for name, scen in SETS:
            res = [run(model, env, *sc) for sc in scen]
            errs, kj = map(np.array, zip(*res))
            rows.append({"w_energy": w, "seed": s, "set": name, "n": len(scen),
                         "mean_deg": errs.mean(), "p95_deg": np.percentile(errs, 95),
                         "max_deg": errs.max(), "mean_kJ": kj.mean(),
                         "share_over_20cm": float((cm(errs) > 20).mean())})
            if w == W_OP:
                per_move += [(name, s, np.degrees(tg), e, k) for (_, tg), e, k in zip(scen, errs, kj)]
        print(f"w={w:<4} seed={s} done")

if not rows:
    sys.exit("No models found in rl/models/. Run rl/05_generalization.py first.")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

# ---- summary table (mean over seeds, max over seeds) ----
print("\n  w   | set    | mean deg (cm) | p95 deg (cm)  | max deg (cm)   | energy kJ | moves > 20 cm")
for w in CANDIDATES:
    for name, _ in SETS:
        rs = [r for r in rows if r["w_energy"] == w and r["set"] == name]
        if not rs:
            continue
        m = np.mean([r["mean_deg"] for r in rs]); p = np.mean([r["p95_deg"] for r in rs])
        x = max(r["max_deg"] for r in rs); k = np.mean([r["mean_kJ"] for r in rs])
        o = np.mean([r["share_over_20cm"] for r in rs]) * 100
        print(f" {w:<4} | {name:<6} | {m:5.2f} ({cm(m):4.1f}) | {p:5.2f} ({cm(p):4.1f}) |"
              f" {x:5.2f} ({cm(x):5.1f}) | {k:6.0f}    | {o:4.1f}%")

# ---- gap statement for the operating point ----
op = {name: [r for r in rows if r["w_energy"] == W_OP and r["set"] == name] for name, _ in SETS}
if all(op.values()):
    base = np.mean([r["mean_deg"] for r in op["orig20"]])
    for name in ["new100", "out100"]:
        m = np.mean([r["mean_deg"] for r in op[name]])
        x = max(r["max_deg"] for r in op[name])
        print(f"\nw={W_OP} {name}: mean error {100 * (m / base - 1):+.0f}% vs orig20,"
              f" worst case {cm(x):.1f} cm")

# ---- plot: error and energy vs target angle, w = W_OP ----
if per_move:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    style = {"orig20": ("o", "train scenarios (20)"), "new100": ("s", "unseen, in range (100)"),
             "out100": ("^", "unseen, out of range (100)")}
    for name, (marker, label) in style.items():
        pts = [p for p in per_move if p[0] == name]
        if not pts:
            continue
        tg = np.array([p[2] for p in pts]); e = np.array([p[3] for p in pts]); k = np.array([p[4] for p in pts])
        ax1.scatter(tg, cm(e), marker=marker, s=18, alpha=0.6, label=label)
        ax2.scatter(tg, k, marker=marker, s=18, alpha=0.6)
    for ax in (ax1, ax2):
        ax.axvspan(*np.degrees(TRAIN_TGT), color="0.85", zorder=0)
        ax.grid(alpha=0.3)
    ax1.axhline(20, color="k", lw=0.8, ls="--")
    ax1.text(np.degrees(TRAIN_TGT[0]) + 1, 20.5, "20 cm limit", fontsize=8)
    ax1.set_ylabel("final error at boom tip (cm)")
    ax1.set_title(f"SAC boom, w={W_OP}, {len(SEEDS)} seeds (grey = training target range)")
    ax1.legend(fontsize=8)
    ax2.set_ylabel("energy per move (kJ)")
    ax2.set_xlabel("target angle (deg)")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"\nPlot: {PNG_PATH}")
print(f"CSV:  {CSV_PATH}")
