"""Step 7: knee sweep. Finer energy weights x 3 seeds, results saved to CSV.

Re-runs w=0 and w=2.0 on the current code (new end stops) so the whole curve is comparable.
Resumable: (w, seed) pairs already in the CSV are skipped, so Ctrl+C loses at most one run.
"""
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from boom_env import BoomEnv

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
SEEDS = [0, 1, 2]
STEPS = 20_000
CSV_PATH = "rl/04_knee_sweep.csv"
PNG_PATH = "rl/04_knee_sweep.png"
FIELDS = ["w_energy", "seed", "error_deg", "energy_kJ", "valve_effort", "train_s"]

rng = np.random.default_rng(7)  # same 20 scenarios as 03_energy_sweep.py
SCENARIOS = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(20)]


def run(model, env, th0, target):
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs, effort = env._obs(), 0.0
    for _ in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, _ = env.step(a)
        effort += float(a[0]) ** 2 * env.dt
        if trunc:
            break
    err = abs(np.degrees(env.target - env.th))
    return err, env.energy_J / 1000, effort


def load_done():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


rows = load_done()
done = {(r["w_energy"], int(r["seed"])) for r in rows}
total = len(WEIGHTS) * len(SEEDS)

new_file = not os.path.exists(CSV_PATH)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    for w in WEIGHTS:
        for s in SEEDS:
            if (w, s) in done:
                print(f"skip w={w} seed={s} (already in CSV)")
                continue
            env = BoomEnv(w_energy=w)
            model = SAC("MlpPolicy", env, device="cpu", seed=s, verbose=0, tensorboard_log="rl/tb")
            t0 = time.time()
            model.learn(total_timesteps=STEPS, tb_log_name=f"knee_w{w}_s{s}")
            errs, energies, efforts = zip(*[run(model, env, *sc) for sc in SCENARIOS])
            row = {"w_energy": w, "seed": s, "error_deg": float(np.mean(errs)),
                   "energy_kJ": float(np.mean(energies)), "valve_effort": float(np.mean(efforts)),
                   "train_s": round(time.time() - t0, 1)}
            writer.writerow(row); f.flush()
            rows.append(row)
            print(f"[{len(rows)}/{total}] w={w:<5} seed={s} | error {row['error_deg']:6.2f} deg"
                  f" | energy {row['energy_kJ']:5.0f} kJ | effort {row['valve_effort']:5.2f}"
                  f" | {row['train_s']:.0f} s")

# ---- summary: mean +/- std over seeds per weight ----
print("\n w_energy | error deg (mean +/- std) | energy kJ (mean +/- std) | seeds")
summary = []
for w in WEIGHTS:
    rs = [r for r in rows if r["w_energy"] == w]
    if not rs:
        continue
    e = np.array([r["error_deg"] for r in rs]); k = np.array([r["energy_kJ"] for r in rs])
    summary.append((w, e.mean(), e.std(), k.mean(), k.std()))
    print(f" {w:<8} | {e.mean():7.2f} +/- {e.std():5.2f}        | {k.mean():6.0f} +/- {k.std():4.0f}"
          f"           | {len(rs)}")

# ---- plot: Pareto front, energy vs error ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([r["energy_kJ"] for r in rows], [r["error_deg"] for r in rows],
           s=15, c="gray", alpha=0.5, label="single seed")
w_, em, es, km, ks = zip(*summary)
ax.errorbar(km, em, xerr=ks, yerr=es, fmt="o-", capsize=3, label="mean +/- std over seeds")
for w, e, k in zip(w_, em, km):
    ax.annotate(f"w={w}", (k, e), xytext=(5, 5), textcoords="offset points")
ax.axhline(2.0, ls="--", c="k", lw=1, label="2 deg target")
ax.set_xlabel("mean mechanical energy per move [kJ]")
ax.set_ylabel("mean final error [deg]")
ax.set_title(f"DX300 boom SAC: accuracy vs energy ({len(SCENARIOS)} scenarios, {len(SEEDS)} seeds)")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(PNG_PATH, dpi=150)
print(f"\nCSV: {CSV_PATH}\nplot: {PNG_PATH}")
