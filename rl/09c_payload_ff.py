"""Lab 09c: payload-aware gravity feed-forward for the Lab 07 PID.

No RL, no training. Two controllers with the same Lab 07 "pid" gains (same grid search):
  pid             u_ff = m_nom g Lc cos(theta) / tau_max        (Lab 07, unchanged)
  pid_payload_ff  u_ff = m_hat g Lc cos(theta) / tau_max
m_hat is estimated online from the moment balance about pin A with a first-order low-pass
filter (T_f = 0.2 s), derivation in docs/09c_payload_estimator.md, code in rl/lab09_common.py.

Evaluated on the 100 unseen scenarios (RNG seed 123) of 05/06/07 under baseline,
payload_x1.2 and m_x0.42. Metrics as in 06/07 (mean_kJ is E_mech = env.energy_J) plus the
pump energy E_pump of Lab 09b (dp_valve and eta_pump ASSUMED - not from manual).
m_hat_over_m is the mean of m_hat / true m at the end of the episode.

Usage: python rl/09c_payload_ff.py
"""
import csv
import numpy as np
from lab09_common import HYD, NEW, PID, VARIANTS, cm, col, e_pump_J, evaluate, tune_pid

CSV_PATH = "rl/09c_payload_ff.csv"
TABLE_PATH = "rl/09c_comparison.md"
PID_CSV = "rl/07_pid_baseline.csv"          # read only, to check the rerun matches Lab 07
T_F_S = 0.2

gains = tune_pid()
CONTROLLERS = {"pid": PID(gains), "pid_payload_ff": PID(gains, payload_aware=True, t_f=T_F_S)}
with open(PID_CSV) as f:
    lab07 = {r["variant"]: r for r in csv.DictReader(f) if r["w"] == "pid"}

rows = []
for name, make in VARIANTS.items():
    for label, ctrl in CONTROLLERS.items():
        rs = evaluate(ctrl, make(), NEW)
        errs = col(rs, "err_deg")
        rows.append({"variant": name, "w": label, "seed": 0,
                     "mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": col(rs, "kJ").mean(),
                     "mean_pump_kJ": np.mean([e_pump_J(r) for r in rs]) / 1000,
                     "share_over_20cm": float((cm(errs) > 20).mean()),
                     "m_hat_over_m": col(rs, "m_hat_ratio").mean() if ctrl.payload_aware else ""})
    old = next(r for r in rows if r["variant"] == name and r["w"] == "pid")
    for key in ("mean_deg", "max_deg", "mean_kJ", "share_over_20cm"):
        ref = float(lab07[name][key])
        assert abs(old[key] - ref) <= 1e-9 * max(1.0, abs(ref)), f"{name} {key}: differs from {PID_CSV}"
    print(f"{name}: old pid rerun matches {PID_CSV} (mean_deg, max_deg, mean_kJ, share_over_20cm)")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

NAMES = {"pid": "old PID (Lab 07)", "pid_payload_ff": "payload-aware PID"}
lines = [
    "# Lab 09c: old PID vs payload-aware PID",
    "",
    f"Both controllers: Kp={gains[0]} Ki={gains[1]} Kd={gains[2]} (Lab 07 `pid`). "
    f"Estimator filter T_f = {T_F_S} s. {len(NEW)} unseen scenarios (RNG seed 123), 10 s each.",
    f"E_pump uses dp_valve = {HYD.DP_VALVE_PA / 1e5:.0f} bar and eta_pump = {HYD.ETA_PUMP}, "
    "both ASSUMED - not from manual (`rl/09b_hyd_params.py`).",
    "Miss rate = share of moves with final tip error over 20 cm. Source: `rl/09c_payload_ff.csv`.",
    "",
    "| variant | controller | mean deg (cm) | max deg (cm) | E_mech kJ | E_pump kJ | miss rate >20 cm | m_hat / m |",
    "|---|---|---|---|---:|---:|---:|---:|",
]
for r in rows:
    ratio = f"{r['m_hat_over_m']:.3f}" if r["m_hat_over_m"] != "" else "n/a"
    lines.append(f"| {r['variant']} | {NAMES[r['w']]} | {r['mean_deg']:.2f} ({cm(r['mean_deg']):.1f}) |"
                 f" {r['max_deg']:.2f} ({cm(r['max_deg']):.1f}) | {r['mean_kJ']:.1f} |"
                 f" {r['mean_pump_kJ']:.1f} | {r['share_over_20cm'] * 100:.0f}% | {ratio} |")
with open(TABLE_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n" + "\n".join(lines[2:]))
print(f"\nCSV: {CSV_PATH}\nTable: {TABLE_PATH}")
