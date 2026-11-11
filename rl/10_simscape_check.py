"""Lab 10b: check rl/boom_env_real.py against the Lab 08a Simscape model. No training.

Nothing is tuned here. The script only reports differences.

1. Static holding torque over the angle range.
   Python: gravity_torque(theta), the torque the plant needs to hold still.
   Simscape: (a) the logged boom-up of Lab 09b2 (matlab/lab09b2/export/simscape_up.csv, lab08a
   model without friction): tau - J dw/dt, averaged in 5 deg bins; (b) the logged torque at
   t = 0 of the boom-up (theta_min) and the boom-down (theta_max) runs. (b) is not an
   independent check: build_stage4.m sets the initial cap pressure from the same analytic
   gravity formula.
   Plant test: the env is held with u = tau_g / tau_up for 2 s; the angle must not move.
2. Boom-up time, u = 1 from theta_min to theta_max - 0.2 deg (window of stage4_run.m).
   Simscape: boom_up_time_s (no friction) and boom_up_time_friction_s in the Lab 08a JSON.
3. Boom-up mechanical work in the same window. Simscape: rl/09b2_energy_check.csv (Lab 09b2).

The env runs with dt = 0.005 s and one substep, the same integration step as the normal
0.05 s / 10 substeps, so the arrival time resolves to 5 ms (then linear interpolation).

Usage: python rl/10_simscape_check.py        writes rl/10_simscape_check.csv
"""
import csv
import json
import math
import numpy as np
import boom_env_real as R

CSV_PATH = "rl/10_simscape_check.csv"
SIM_UP = "matlab/lab09b2/export/simscape_up.csv"        # read only
SIM_DOWN = "matlab/lab09b2/export/simscape_down.csv"    # read only
REF_09B2 = "rl/09b2_energy_check.csv"                   # read only
TOL_RAD = math.radians(0.2)                             # "reached theta_max", as stage4_run.m
MAX_TIME_DIFF_PCT = 15.0                                # stop rule of Lab 10
BIN_DEG = list(range(-40, 51, 10))
HALF_BIN_DEG = 2.5

with open(R.JSON_PATH) as f:
    J8 = json.load(f)
with open(REF_09B2) as f:
    REF = {(r["variant"], r["move"]): r for r in csv.DictReader(f)}

rows = []


def add(check, quantity, python, simscape, unit, note=""):
    diff = python - simscape
    rows.append({"check": check, "quantity": quantity, "python": python, "simscape": simscape,
                 "unit": unit, "diff": diff, "diff_pct": 100.0 * diff / simscape, "note": note})


# --- 1. static holding torque ---
up = np.genfromtxt(SIM_UP, delimiter=",", names=True)
win = up["in_window"] == 1
alpha = np.gradient(up["omega"], up["t"])
tau_static = up["tau"] - R.J_NOM * alpha                 # Simscape torque minus the inertia term
for deg in BIN_DEG:
    sel = win & (np.abs(np.degrees(up["theta"]) - deg) <= HALF_BIN_DEG)
    sim = float(tau_static[sel].mean()) / 1e3
    py = float(np.mean([R.gravity_torque(t) for t in up["theta"][sel]])) / 1e3
    add("static", f"holding torque at {deg:+d} deg", py, sim, "kN m",
        f"Simscape boom-up log, tau - J dw/dt, mean of {int(sel.sum())} samples in +-{HALF_BIN_DEG} deg")
down = np.genfromtxt(SIM_DOWN, delimiter=",", names=True)
for name, log in (("theta_min", up), ("theta_max", down)):
    th0 = float(log["theta"][0])
    add("static", f"holding torque at {name} ({math.degrees(th0):+.1f} deg), t = 0",
        R.gravity_torque(th0) / 1e3, float(log["tau"][0]) / 1e3, "kN m",
        "Simscape initial state; not independent (initial pressure set from the analytic formula)")

hold = []
for deg in [-45] + BIN_DEG + [58]:
    env = R.BoomEnvReal()
    env.reset(seed=0)
    env.th, env.om, env.target = math.radians(deg), 0.0, math.radians(deg)
    u = R.gravity_torque(env.th) / R.tau_up(env.th)
    for _ in range(40):                                    # 2 s
        env.step(np.array([u]))
    hold.append((deg, u, math.degrees(env.th) - deg, env.om))
max_drift = max(abs(h[2]) for h in hold)


# --- 2. and 3. boom-up at full command ---
def boom_up(fric_scale):
    env = R.BoomEnvReal(fric_scale=fric_scale)
    env.dt, env.SUBSTEPS, env.max_steps = 0.005, 1, 4000   # same integration step, finer log
    env.reset(seed=0)
    env.th, env.om, env.target = R.TH_MIN, 0.0, R.TH_MAX
    goal = R.TH_MAX - TOL_RAD
    t, peak_v, n_flow = 0.0, 0.0, 0
    while True:
        prev = (env.th, env.W_pos_J, env.W_neg_J, env.energy_J, env.fric_J)
        _, _, _, trunc, info = env.step(np.array([1.0]))
        n_flow += info["flow_lim"]
        peak_v = max(peak_v, R.moment_arm(prev[0]) * env.om)   # r at the angle the limit was applied
        if env.th >= goal:
            f = (goal - prev[0]) / (env.th - prev[0])      # fraction of the last step
            lerp = lambda a, b: a + f * (b - a)
            w_pos, w_neg = lerp(prev[1], env.W_pos_J), lerp(prev[2], env.W_neg_J)
            return {"t_s": t + f * env.dt, "signed_kJ": (w_pos - w_neg) / 1e3,
                    "abs_kJ": lerp(prev[3], env.energy_J) / 1e3, "fric_kJ": lerp(prev[4], env.fric_J) / 1e3,
                    "ekin_kJ": 0.5 * env.J * env.om ** 2 / 1e3, "peak_v": peak_v,
                    "flow_share": n_flow / env.n_sub, "violations": env.n_violation,
                    "dEp_kJ": (R.potential_energy(goal) - R.potential_energy(R.TH_MIN)) / 1e3}
        t += env.dt
        assert not trunc, "boom-up did not reach theta_max"


runs = {"no friction": (boom_up(0.0), float(J8["boom_up_time_s"]["value"]), REF[("_first", "up")]),
        "friction": (boom_up(1.0), float(J8["boom_up_time_friction_s"]["value"]), REF[("_damped", "up")])}
for name, (py, t_sim, ref) in runs.items():
    add("boom_up_time", f"time to theta_max - 0.2 deg, {name}", py["t_s"], t_sim, "s",
        "Simscape: Lab 08a JSON " + ("boom_up_time_s" if name == "no friction" else "boom_up_time_friction_s"))
    add("boom_up_work", f"signed work, {name}", py["signed_kJ"], float(ref["sim_E_mech_signed_kJ"]), "kJ",
        f"Simscape: {REF_09B2} {ref['variant']} sim_E_mech_signed_kJ")
    add("boom_up_work", f"absolute work, {name}", py["abs_kJ"], float(ref["sim_E_mech_abs_kJ"]), "kJ",
        f"Simscape: {REF_09B2} {ref['variant']} sim_E_mech_abs_kJ")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

# --- print ---
print("| check | quantity | Python | Simscape | diff | diff % |")
print("|---|---|---:|---:|---:|---:|")
for r in rows:
    digits = 3 if r["unit"] == "s" else 1
    print(f"| {r['check']} | {r['quantity']} [{r['unit']}] | {r['python']:.{digits}f} |"
          f" {r['simscape']:.{digits}f} | {r['diff']:+.{digits}f} | {r['diff_pct']:+.2f} |")
static = [r for r in rows if r["check"] == "static" and "t = 0" not in r["quantity"]]
print(f"\nstatic, 5 deg bins: max |diff| {max(abs(r['diff_pct']) for r in static):.2f} %"
      f" of the Simscape value")
print(f"plant hold test, u = tau_g / tau_up for 2 s at {len(hold)} angles"
      f" ({hold[0][0]} .. {hold[-1][0]} deg): max |drift| {max_drift:.2e} deg,"
      f" hold command {min(h[1] for h in hold):.3f} .. {max(h[1] for h in hold):.3f}")
for name, (py, t_sim, ref) in runs.items():
    print(f"boom-up, {name}: {py['t_s']:.3f} s, peak rod speed {py['peak_v']:.4f} m/s"
          f" (limit {R.V_EXT_MAX:.4f}), on the flow limit {100 * py['flow_share']:.1f} % of the substeps,"
          f" 330 bar violations {py['violations']}")
    print(f"  work split [kJ]: dEp {py['dEp_kJ']:.1f} + E_kin at arrival {py['ekin_kJ']:.1f}"
          f" + friction {py['fric_kJ']:.1f} = {py['dEp_kJ'] + py['ekin_kJ'] + py['fric_kJ']:.1f};"
          f" signed work {py['signed_kJ']:.1f}")
worst = max(abs(r["diff_pct"]) for r in rows if r["check"] == "boom_up_time")
print(f"\nboom-up time: largest difference {worst:.2f} % (stop rule: > {MAX_TIME_DIFF_PCT:.0f} %)"
      f" -> {'STOP' if worst > MAX_TIME_DIFF_PCT else 'within the limit'}")
print(f"CSV: {CSV_PATH}")
