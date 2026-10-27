"""Step 8: generalization test. Candidate weights on 100 unseen scenarios, worst case in cm.

Trains w in CANDIDATES x SEEDS (same settings as 04), SAVES the models so they can be reused,
then evaluates each on the original 20 scenarios and on 100 new ones (different RNG seed).
Resumable: a model already saved in rl/models/ is loaded instead of retrained.
"""
import csv
import os
import time
import numpy as np
from stable_baselines3 import SAC
from boom_env import BoomEnv

CANDIDATES = [0.1, 0.3, 0.5]
SEEDS = [0, 1, 2]
STEPS = 20_000
BOOM_LENGTH_M = 6.245  # dx300_params.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
CSV_PATH = "rl/05_generalization.csv"

rng = np.random.default_rng(7)      # the original 20 (same as 03/04)
ORIG = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(20)]
rng = np.random.default_rng(123)    # 100 unseen scenarios, same ranges
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]


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


os.makedirs(MODEL_DIR, exist_ok=True)
rows = []
for w in CANDIDATES:
    for s in SEEDS:
        env = BoomEnv(w_energy=w)
        path = f"{MODEL_DIR}/sac_w{w}_s{s}.zip"
        t0 = time.time()
        if os.path.exists(path):
            model = SAC.load(path, env=env, device="cpu")
            how = "loaded"
        else:
            model = SAC("MlpPolicy", env, device="cpu", seed=s, verbose=0, tensorboard_log="rl/tb")
            model.learn(total_timesteps=STEPS, tb_log_name=f"gen_w{w}_s{s}")
            model.save(path)
            how = "trained"
        for name, scen in [("orig20", ORIG), ("new100", NEW)]:
            errs, kj = map(np.array, zip(*[run(model, env, *sc) for sc in scen]))
            rows.append({"w_energy": w, "seed": s, "set": name, "n": len(scen),
                         "mean_deg": errs.mean(), "p95_deg": np.percentile(errs, 95),
                         "max_deg": errs.max(), "mean_kJ": kj.mean(),
                         "share_over_20cm": float((cm(errs) > 20).mean())})
        r = rows[-1]
        print(f"w={w:<4} seed={s} {how} in {time.time() - t0:4.0f} s | new100: mean {r['mean_deg']:.2f} deg,"
              f" p95 {r['p95_deg']:.2f}, max {r['max_deg']:.2f} ({cm(r['max_deg']):.0f} cm), {r['mean_kJ']:.0f} kJ")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

print("\n  w   | set    | mean deg (cm) | p95 deg (cm)  | max deg (cm)   | energy kJ | moves > 20 cm")
for w in CANDIDATES:
    for name in ["orig20", "new100"]:
        rs = [r for r in rows if r["w_energy"] == w and r["set"] == name]
        m = np.mean([r["mean_deg"] for r in rs]); p = np.mean([r["p95_deg"] for r in rs])
        x = max(r["max_deg"] for r in rs); k = np.mean([r["mean_kJ"] for r in rs])
        o = np.mean([r["share_over_20cm"] for r in rs]) * 100
        print(f" {w:<4} | {name:<6} | {m:5.2f} ({cm(m):4.1f}) | {p:5.2f} ({cm(p):4.1f}) |"
              f" {x:5.2f} ({cm(x):5.1f}) | {k:6.0f}    | {o:4.1f}%")
print(f"\nCSV: {CSV_PATH}   models: {MODEL_DIR}/")
