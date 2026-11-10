"""Lab 09b: hydraulic energy model and energy balance for the Lab 07 PID.

No RL, no training. Reruns the Lab 07 "pid" controller (same grid search, same 100 unseen
scenarios, RNG seed 123) on baseline, payload_x1.2 and m_x0.42 and logs, per episode:
  E_mech = sum |F_cyl v_cyl| dt                      (equals env.energy_J, asserted)
  E_pump = sum (max(p_load, 0) + dp_valve) Q dt / eta_pump
with the two-sided cylinder model of rl/lab09_common.py. Lowering under gravity (p_load < 0)
costs the standby loss dp_valve * Q; no energy is recovered.

Balance, mean over the episodes:
  E_pump = dE_pot + dE_kin + losses + residual
  dE_pot      sum m g Lc (sin th[k+1] - sin th[k])   work against gravity (m of that step)
  dE_kin      J (om_end^2 - om_0^2) / 2
  pump        E_pump (1 - eta_pump)
  valve       dp_valve * swept volume
  overrun     sum max(-tau om, 0) dt                 boom drives the cylinder, throttled away
  damping     sum b om^2 dt
  end stop    energy removed in steps that end on a stop
  residual    what is left: the integration error of the env's Euler step (dt = 0.05 s)
A second baseline run with dt = 0.005 s (2000 steps, control at 200 Hz) shows how the
residual scales with the step.

Sensitivity: E_pump for dp_valve in 10, 20, 30 bar x eta_pump in 0.80, 0.85, 0.90.
dp_valve and eta_pump are ASSUMED - not from manual (rl/09b_hyd_params.py).

Usage: python rl/09b_hydraulic_energy.py
"""
import csv
import numpy as np
from boom_env import BoomEnv
from lab09_common import HYD, NEW, PID, VARIANTS, col, e_pump_J, evaluate, tune_pid

CSV_BALANCE = "rl/09b_energy_balance.csv"
CSV_SENS = "rl/09b_sensitivity.csv"
PID_CSV = "rl/07_pid_baseline.csv"          # read only, to check the rerun matches Lab 07
DP_SWEEP_PA = [10e5, 20e5, 30e5]
ETA_SWEEP = [0.80, 0.85, 0.90]
DP, ETA = HYD.DP_VALVE_PA, HYD.ETA_PUMP


def balance(rs, variant, dt):
    """Mean energy terms [kJ] over the episodes rs."""
    e_pump = np.array([e_pump_J(r) for r in rs])
    terms = {
        "dE_pot_kJ": col(rs, "dEp_J"), "dE_kin_kJ": col(rs, "dEk_J"),
        "loss_pump_kJ": e_pump * (1 - ETA), "loss_valve_kJ": DP * col(rs, "vol_m3"),
        "loss_overrun_kJ": col(rs, "W_neg_J"), "loss_damping_kJ": col(rs, "damp_J"),
        "loss_endstop_kJ": col(rs, "stop_J"),
    }
    resid = e_pump - sum(terms.values())
    row = {"variant": variant, "controller": "pid", "dt_s": dt, "episodes": len(rs),
           "E_mech_kJ": col(rs, "E_mech_J").mean() / 1000, "E_pump_kJ": e_pump.mean() / 1000}
    row.update({k: v.mean() / 1000 for k, v in terms.items()})
    row["residual_kJ"] = resid.mean() / 1000
    row["residual_pct_of_E_pump"] = 100 * resid.mean() / e_pump.mean()
    row["max_abs_residual_kJ"] = np.abs(resid).max() / 1000
    row["p_pump_max_bar"] = (col(rs, "p_load_max_Pa").max() + DP) / 1e5
    row["episodes_on_stop"] = int((col(rs, "stop_steps") > 0).sum())
    return row


gains = tune_pid()
with open(PID_CSV) as f:
    lab07 = {r["variant"]: r for r in csv.DictReader(f) if r["w"] == "pid"}

rows, sens, runs = [], [], {}
for name, make in VARIANTS.items():
    rs = runs[name] = evaluate(PID(gains), make(), NEW)
    ref = float(lab07[name]["mean_kJ"])
    assert abs(col(rs, "kJ").mean() - ref) < 1e-9 * ref, f"{name}: rerun differs from {PID_CSV}"
    print(f"{name}: E_mech {col(rs, 'kJ').mean():.6f} kJ = {PID_CSV} mean_kJ {ref:.6f}")
    rows.append(balance(rs, name, BoomEnv().dt))
    e_ref = np.mean([e_pump_J(r) for r in rs])
    for dp in DP_SWEEP_PA:
        for eta in ETA_SWEEP:
            e = np.mean([e_pump_J(r, dp, eta) for r in rs])
            sens.append({"variant": name, "controller": "pid", "dp_valve_bar": dp / 1e5,
                         "eta_pump": eta, "E_pump_kJ": e / 1000,
                         "change_pct_vs_20bar_0.85": 100 * (e / e_ref - 1)})

# residual check: same baseline scenarios with a 10x smaller step
fine = BoomEnv()
fine.dt, fine.max_steps = fine.dt / 10, fine.max_steps * 10
rows.append(balance(evaluate(PID(gains, dt=fine.dt), fine, NEW), "baseline_dt_check", fine.dt))

for path, data in [(CSV_BALANCE, rows), (CSV_SENS, sens)]:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)

print(f"\nLab 07 pid Kp={gains[0]} Ki={gains[1]} Kd={gains[2]} | dp_valve {DP / 1e5:.0f} bar,"
      f" eta_pump {ETA} (both ASSUMED - not from manual) | mean kJ per episode, {len(NEW)} episodes")
print("| variant           | dt s  | E_mech | E_pump | dE_pot | dE_kin | pump | valve | overrun |"
      " damping | end stop | residual (% E_pump) | max abs resid | p_pump max bar |")
print("|-------------------|------:|-------:|-------:|-------:|-------:|-----:|------:|--------:|"
      "--------:|---------:|--------------------:|--------------:|---------------:|")
for r in rows:
    print(f"| {r['variant']:<17} | {r['dt_s']:5.3f} | {r['E_mech_kJ']:6.1f} | {r['E_pump_kJ']:6.1f} |"
          f" {r['dE_pot_kJ']:6.1f} | {r['dE_kin_kJ']:6.2f} | {r['loss_pump_kJ']:4.1f} |"
          f" {r['loss_valve_kJ']:5.1f} | {r['loss_overrun_kJ']:7.1f} | {r['loss_damping_kJ']:7.1f} |"
          f" {r['loss_endstop_kJ']:8.2f} | {r['residual_kJ']:8.2f} ({r['residual_pct_of_E_pump']:6.2f}%) |"
          f" {r['max_abs_residual_kJ']:13.2f} | {r['p_pump_max_bar']:14.1f} |")
print(f"episodes touching an end stop: "
      + ", ".join(f"{r['variant']} {r['episodes_on_stop']}" for r in rows))

print("\nSensitivity: mean E_pump kJ (change vs 20 bar, 0.85)")
print("| variant      | dp_valve bar | eta 0.80        | eta 0.85        | eta 0.90        |")
print("|--------------|-------------:|-----------------|-----------------|-----------------|")
for name in VARIANTS:
    for dp in DP_SWEEP_PA:
        cells = [next(s for s in sens if s["variant"] == name and s["dp_valve_bar"] == dp / 1e5
                      and s["eta_pump"] == eta) for eta in ETA_SWEEP]
        print(f"| {name:<12} | {dp / 1e5:12.0f} | "
              + " | ".join(f"{c['E_pump_kJ']:6.1f} ({c['change_pct_vs_20bar_0.85']:+5.1f}%)" for c in cells)
              + " |")
print(f"\nCSV: {CSV_BALANCE}, {CSV_SENS}")
