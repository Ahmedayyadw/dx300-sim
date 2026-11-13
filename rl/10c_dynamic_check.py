"""Lab 10c: dynamic check of rl/boom_env_real.py against the Lab 09b2 Simscape runs. No training.

Lab 10b checked static holding torque, boom-up time and boom-up work. This lab compares the whole
trajectories theta(t) and omega(t) of the full-command moves, including boom-down, which 10b did
not cover. Nothing is tuned; the script only reports differences. Reads only, writes three files.

Simscape (read only, matlab/lab09b2/export/): the spool steps to its full opening at t = 1 s; the
move window runs from that step until the boom is within 0.2 deg of the end stop (column
in_window). Time is shifted so that t = 0 is the spool step.
  up            simscape_up.csv          spool +100 %, no friction      Python u = +1, fric_scale 0
  down          simscape_down.csv        spool -100 %, no friction      Python u = -1, fric_scale 0
  up, friction  simscape_up_damped.csv   spool +100 %, friction model   Python u = +1, fric_scale 1
simscape_down_damped.csv is not used: it was run at spool -80 %, which has no direct counterpart in
the torque command u of the Python plant.

Python: BoomEnvReal with dt = 0.005 s and one substep (the integration step of the normal
0.05 s / 10 substeps, as in Lab 10b), started at rest on the end stop.

Metrics, inside the Simscape window (Python interpolated onto the Simscape time grid):
arrival time, RMS and max angle difference, RMS speed difference, peak speed of both.
Status values are ESTIMATED (plant geometry and friction from a drawing, Lab 08a).

Usage: python rl/10c_dynamic_check.py
Writes rl/10c_dynamic_check.csv, rl/10c_dynamic_check.png, rl/10c_dynamic_check.md
"""

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boom_env_real as R  # noqa: E402

EXPORT = "matlab/lab09b2/export"
OUT = "rl/10c_dynamic_check"
TOL_RAD = math.radians(0.2)  # arrival tolerance, as stage4_run.m / lab09b2_energy_split.m
T_SPOOL = 1.0  # spool step time of the Simscape runs
CASES = [  # name, Simscape file, command u, friction scale
    ("up", "simscape_up.csv", 1.0, 0.0),
    ("down", "simscape_down.csv", -1.0, 0.0),
    ("up, friction", "simscape_up_damped.csv", 1.0, 1.0),
]


def simscape(path):
    d = np.genfromtxt(os.path.join(EXPORT, path), delimiter=",", names=True)
    w = np.where(d["in_window"] == 1)[0]
    i0, i1 = w[0], w[-1]
    return d["t"][i0 : i1 + 1] - T_SPOOL, d["theta"][i0 : i1 + 1], d["omega"][i0 : i1 + 1]


def python_move(u, fric_scale, up):
    env = R.BoomEnvReal(fric_scale=fric_scale)
    env.dt, env.SUBSTEPS, env.max_steps = 0.005, 1, 4000
    env.reset(seed=0)
    start, goal = (R.TH_MIN, R.TH_MAX - TOL_RAD) if up else (R.TH_MAX, R.TH_MIN + TOL_RAD)
    env.th, env.om, env.target = start, 0.0, (R.TH_MAX if up else R.TH_MIN)
    t, th, om = [0.0], [env.th], [env.om]
    while True:
        prev = env.th
        _, _, _, trunc, _ = env.step(np.array([u]))
        t.append(t[-1] + env.dt)
        th.append(env.th)
        om.append(env.om)
        if (up and env.th >= goal) or (not up and env.th <= goal):
            f = (goal - prev) / (env.th - prev)
            return np.array(t), np.array(th), np.array(om), t[-2] + f * env.dt
        assert not trunc, "move did not reach the end stop"


rows, curves = [], []
for name, path, u, fric in CASES:
    ts, ths, oms = simscape(path)
    tp, thp, omp, t_arr = python_move(u, fric, up=u > 0)
    sel = ts <= min(ts[-1], tp[-1])  # common part of both windows
    th_i = np.interp(ts[sel], tp, thp)
    om_i = np.interp(ts[sel], tp, omp)
    d_th = np.degrees(th_i - ths[sel])
    rows.append(
        {
            "move": name,
            "arrival_python_s": t_arr,
            "arrival_simscape_s": float(ts[-1]),
            "arrival_diff_pct": 100.0 * (t_arr - ts[-1]) / ts[-1],
            "rms_angle_diff_deg": float(np.sqrt(np.mean(d_th**2))),
            "max_angle_diff_deg": float(np.max(np.abs(d_th))),
            "rms_speed_diff_rad_s": float(np.sqrt(np.mean((om_i - oms[sel]) ** 2))),
            "peak_speed_python_rad_s": float(np.max(np.abs(omp))),
            "peak_speed_simscape_rad_s": float(np.max(np.abs(oms))),
        }
    )
    curves.append((name, ts, ths, oms, tp, thp, omp))

with open(OUT + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fig, ax = plt.subplots(2, len(curves), figsize=(4.2 * len(curves), 5.6), sharex="col")
for k, (name, ts, ths, oms, tp, thp, omp) in enumerate(curves):
    ax[0, k].plot(ts, np.degrees(ths), color="#1f5fa8", lw=2.2, label="Simscape (Lab 09b2)")
    ax[0, k].plot(
        tp, np.degrees(thp), color="#d9822b", lw=1.4, ls="--", label="Python (boom_env_real)"
    )
    ax[1, k].plot(ts, oms, color="#1f5fa8", lw=2.2)
    ax[1, k].plot(tp, omp, color="#d9822b", lw=1.4, ls="--")
    ax[0, k].set_title(f"boom {name}")
    ax[1, k].set_xlabel("time after spool step [s]")
    for a in ax[:, k]:
        a.grid(alpha=0.3)
ax[0, 0].set_ylabel("boom angle [deg]")
ax[1, 0].set_ylabel("boom speed [rad/s]")
ax[0, 0].legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=150)

down = np.genfromtxt(os.path.join(EXPORT, "simscape_down.csv"), delimiter=",", names=True)
dw = down["in_window"] == 1
v_down = float(-down["v"][dw].min())
p_rod_min_bar = float(down["p_rod"][dw].min()) / 1e5
q_rod_max = float(down["Q_rod"][dw].max()) * 6e4

COLS = [
    ("move", "move", "{}"),
    ("arrival Python s", "arrival_python_s", "{:.3f}"),
    ("arrival Simscape s", "arrival_simscape_s", "{:.3f}"),
    ("diff %", "arrival_diff_pct", "{:+.2f}"),
    ("RMS angle diff deg", "rms_angle_diff_deg", "{:.2f}"),
    ("max angle diff deg", "max_angle_diff_deg", "{:.2f}"),
    ("RMS speed diff rad/s", "rms_speed_diff_rad_s", "{:.4f}"),
    ("peak speed Python rad/s", "peak_speed_python_rad_s", "{:.3f}"),
    ("peak speed Simscape rad/s", "peak_speed_simscape_rad_s", "{:.3f}"),
]
table = [
    "| " + " | ".join(c[0] for c in COLS) + " |",
    "|---|" + "---:|" * (len(COLS) - 1),
]
for r in rows:
    table.append("| " + " | ".join(fmt.format(r[key]) for _, key, fmt in COLS) + " |")

lines = [
    "# Lab 10c: dynamic check of the real plant against Simscape",
    "",
    "Generated by `rl/10c_dynamic_check.py`. No training, nothing tuned. Full-command moves",
    "of the Lab 09b2 Simscape model (spool step at t = 0) against `rl/boom_env_real.py`",
    "(u = +-1, dt = 0.005 s, one substep). Window: spool step until within 0.2 deg of the end",
    "stop. All values ESTIMATED.",
    "",
    *table,
    "",
    "![Python plant vs Simscape, full-command moves](10c_dynamic_check.png)",
    "",
    "## Reading",
    "",
    "- **Boom-up** (with and without friction): the mean motion matches. Simscape oscillates",
    "  around the Python speed at about 2 Hz, most likely the oil column (compressibility),",
    "  which the rigid, flow-limited Python plant does not model.",
    f"- **Boom-down**: Python is slower. Simscape retracts at up to {v_down:.3f} m/s rod speed",
    f"  against the Python retraction limit of {R.V_RET_MAX:.3f} m/s (pump flow / annulus area).",
    f"  The Simscape rod side drops to {p_rod_min_bar:.2f} bar (gauge, about vacuum) and fills at",
    f"  up to {q_rod_max:.0f} L/min against 494 L/min pump flow, so in Simscape gravity pulls the",
    "  boom down faster than the pump refills the rod side. Python caps the speed at the pump",
    "  flow. A real DX300 boom-down regeneration or anti-cavitation circuit would decide which",
    "  is closer; that is not modelled in either. Lowering moves of Labs 10-12 are therefore",
    "  conservative (slower) whenever they hit the retraction limit.",
    "",
    "Not covered: `simscape_down_damped.csv` (spool -80 %, no counterpart in the torque",
    "command u); partial commands; valve dynamics.",
]
with open(OUT + ".md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(table))
print(f"wrote {OUT}.csv, {OUT}.png, {OUT}.md")
