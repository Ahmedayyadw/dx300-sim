"""Step 10: PID baseline. Classical controller on the same boom, scenarios and variants as 06/06b.

No RL training. Controller, on what the env observes (theta, omega, e = target - theta):
  u = clip(Kp e + Ki int(e) dt + Kd (-omega) + u_ff, -1, 1)
  u_ff = m g Lc cos(theta) / tau_max with the NOMINAL BoomEnv values, never the perturbed env's
  anti-windup: the integral is frozen while u is saturated
Gains: grid search on the 20 original scenarios (RNG seed 7), lowest mean energy among gains with
worst-case error < 20 cm ("pid"), and the same with Ki > 0 required ("pid_ki"). Both evaluated on the 100 unseen scenarios (RNG seed 123) under every
variant of 06 and 06b (env classes copied from there; noise only on what the PID observes).
CSV has the 06 columns with w = "pid" or "pid_ki", seed = 0. Summary table puts SAC w=0 and w=0.1 next to it.

Usage: python rl/07_pid_baseline.py [--dry-run]   (dry run: 5 scenarios each, no CSV written)
"""
import csv
import itertools
import os
import sys
import time
import numpy as np
from boom_env import BoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P
from dx300sim.boom_geometry import moment_arm

DRY_RUN = "--dry-run" in sys.argv
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
CSV_PATH = "rl/07_pid_baseline.csv"
SAC_CSVS = ["rl/06_sensitivity.csv", "rl/06b_noise.csv"]
SAC_WEIGHTS = ["0.0", "0.1"]

KP = [0.25, 0.5, 1, 2, 5, 10, 20, 30, 50]
KI = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10]
KD = [0, 0.25, 0.5, 1, 2]

rng = np.random.default_rng(7)      # the original 20 (same as 03/04/05)
ORIG = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(20)]
rng = np.random.default_rng(123)    # 100 unseen scenarios, same as 05/06/06b
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
if DRY_RUN:
    ORIG, NEW = ORIG[:5], NEW[:5]

_nominal = BoomEnv()
U_FF_SCALE = _nominal.m * 9.81 * _nominal.Lc / _nominal.tau_max   # nominal, fixed for all variants
DT = _nominal.dt


# --- env variants, copied from 06_sensitivity.py and 06b_noise.py ---
def d(theta):
    return moment_arm(theta, P.B_XY_M, P.C_M, P.DELTA_RAD)


D0_M = abs(d(0.0))   # about 0.898 m


class TauDBoomEnv(BoomEnv):
    """BoomEnv with angle-dependent torque limit tau_max = 5e5 x |d(theta)| / |d(0)|."""

    def step(self, action):
        self.tau_max = 5e5 * abs(d(self.th)) / D0_M   # limit at the pre-step angle
        return super().step(action)


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


def make_m():
    env = BoomEnv(); env.m *= 0.42; return env


def make_J():
    env = BoomEnv(); env.J *= 0.38; return env


VARIANTS = {
    "baseline": BoomEnv,
    "m_x0.42": make_m,
    "J_x0.38": make_J,
    "tau_d": TauDBoomEnv,
    "noise_0.2": lambda: NoisyBoomEnv(0.2),
    "noise_0.5": lambda: NoisyBoomEnv(0.5),
    "noise_1.0": lambda: NoisyBoomEnv(1.0),
    "payload_x1.2": PayloadBoomEnv,
}


# --- PID ---
def run(gains, env, th0, target):
    kp, ki, kd = gains
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs, integ = env._obs(), 0.0
    for _ in range(env.max_steps):
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])   # what the PID observes
        u_raw = kp * e + ki * integ + kd * (-om) + U_FF_SCALE * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup
            integ += e * DT
        obs, _, _, trunc, _ = env.step(np.array([np.clip(u_raw, -1.0, 1.0)]))
        if trunc:
            break
    return abs(np.degrees(env.target - env.th)), env.energy_J / 1000


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


def evaluate(gains, env, scen):
    errs, kj = map(np.array, zip(*[run(gains, env, *sc) for sc in scen]))
    return errs, kj


# --- grid search on the original 20 ---
t0 = time.time()
grid = []
env = BoomEnv()
for gains in itertools.product(KP, KI, KD):
    errs, kj = evaluate(gains, env, ORIG)
    grid.append((gains, errs.max(), kj.mean()))
print(f"grid: {len(grid)} gain sets on {len(ORIG)} scenarios in {time.time() - t0:.1f} s")

# "pid": spec'd criterion. "pid_ki": same criterion, Ki > 0 required (lowest energy picks Ki = 0,
# which leaves no correction when the real mass differs from the nominal feed-forward).
TUNED = {}
for label, allowed in [("pid", lambda g: True), ("pid_ki", lambda g: g[0][1] > 0)]:
    ok = [g for g in grid if cm(g[1]) < 20 and allowed(g)]
    if not ok:
        sys.exit(f"{label}: no gains meet worst case < 20 cm; widen the grid")
    best, best_max, best_kj = min(ok, key=lambda g: g[2])
    TUNED[label] = best
    print(f"{label}: {len(ok)} gain sets with worst case < 20 cm | tuned Kp={best[0]} Ki={best[1]}"
          f" Kd={best[2]} | orig: max {best_max:.2f} deg ({cm(best_max):.1f} cm), {best_kj:.1f} kJ")
    print("  next lowest energy:", ", ".join(f"{g[0]} {cm(g[1]):.1f} cm {g[2]:.1f} kJ"
                                             for g in sorted(ok, key=lambda g: g[2])[1:4]))

# --- evaluate tuned PIDs on the 100 unseen scenarios, every variant ---
rows = []
for name, make in VARIANTS.items():
    for label, gains in TUNED.items():
        errs, kj = evaluate(gains, make(), NEW)
        rows.append({"variant": name, "w": label, "seed": 0,
                     "mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": kj.mean(),
                     "share_over_20cm": float((cm(errs) > 20).mean())})

if DRY_RUN:
    print(f"\nDRY RUN: {len(ORIG)} orig, {len(NEW)} unseen scenarios, {CSV_PATH} not written")
else:
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

# SAC rows from 06/06b (full 100 scenarios), aggregated as in 06: mean over seeds, max over seeds
sac = {}
for path in SAC_CSVS:
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["w"] in SAC_WEIGHTS:
                sac.setdefault((r["variant"], r["w"]), []).append(r)


def agg(rs):
    return (np.mean([float(r["mean_deg"]) for r in rs]), max(float(r["max_deg"]) for r in rs),
            np.mean([float(r["mean_kJ"]) for r in rs]),
            np.mean([float(r["share_over_20cm"]) for r in rs]) * 100)


print("\n" + "   ".join(f"{label} Kp={g[0]} Ki={g[1]} Kd={g[2]}" for label, g in TUNED.items())
      + ("  (SAC rows are full 100 scenarios, PID is dry run)" if DRY_RUN else ""))
print("| variant      | controller | mean deg (cm) | max deg (cm)  | mean kJ | % >20 cm |")
print("|--------------|------------|---------------|---------------|--------:|---------:|")
for name in VARIANTS:
    entries = [(r["w"].upper().replace("_KI", " Ki>0"),
                (r["mean_deg"], r["max_deg"], r["mean_kJ"], r["share_over_20cm"] * 100))
               for r in rows if r["variant"] == name]
    for w in SAC_WEIGHTS:
        rs = sac.get((name, w), [])
        if rs:
            entries.append((f"SAC w={w}", agg(rs[:3])))   # baseline is in both CSVs; take 06's
    for ctrl, (m, x, k, o) in entries:
        print(f"| {name:<12} | {ctrl:<10} | {m:5.2f} ({cm(m):5.1f}) | {x:5.2f} ({cm(x):5.1f}) |"
              f" {k:7.1f} | {o:7.1f}% |")
if not DRY_RUN:
    print(f"\nCSV: {CSV_PATH}")
