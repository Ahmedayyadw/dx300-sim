"""Lab 09b2: check the Lab 09b Python hydraulic energy model against the Simscape boom.

No RL, no training, no Simscape run here. Reads the logs that matlab/lab09b2/lab09b2_run.m
exported (boom-up and boom-down, full spool, lab08a hydraulic model) and

  1. feeds the logged cylinder force F and rod velocity v into the Lab 09b formula
     (rl/lab09_common.py, two-cylinder total):
        v >= 0: p_load =  F / A_cap, Q = A_cap v        v < 0: p_load = -F / A_ann, Q = A_ann |v|
        E_mech = sum |F v| dt
        E_pump = sum (max(p_load, 0) + dp_valve) Q dt / eta_pump
  2. compares E_mech and E_pump with the Simscape energies per move,
  3. splits the E_pump difference into relief flow, supply-valve drop, cylinder-side terms
     and the Python pump-loss term,
  4. solves for the dp_valve / eta_pump that would reproduce the Simscape pump energy.

Window: spool opening (t = 1 s) to first arrival within 0.2 deg of the end stop, as in
lab08a stage4_run.m. The 1 s before the spool opens (pump dead-headed) is listed separately.

Status of the values:
  OFFICIAL   bore, rod, stroke, pump flow 494 L/min, relief 330 bar
  ESTIMATED  Simscape valve area, spool stroke, oil data, dead volumes, pin geometry
  ASSUMED    dp_valve = 20 bar, eta_pump = 0.85 (rl/09b_hyd_params.py, not from manual)
The Simscape pump is an ideal flow source: it has no efficiency, so its energy is hydraulic
output. The Python E_pump is shaft input; E_hyd = E_pump * eta_pump is the like-for-like number.

Variants: the script repeats the comparison for every Simscape export it finds (VARIANTS below):
the first run and the damped model with a metered boom-down (matlab/lab09b2/lab09b2_run.m header).

Usage: python rl/09b2_energy_check.py       (after matlab -batch "run('matlab/lab09b2/lab09b2_run.m')")
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lab09_common import A_ANN_M2, A_CAP_M2, HYD, e_pump_J

EXPORT = "matlab/lab09b2/export"
CSV_OUT = "rl/09b2_energy_check.csv"
FIG_OUT = "rl/09b2_power{tag}.png"
# Simscape variants: (export tag, description). Each is one lab09b2_run.m call; missing exports are skipped.
VARIANTS = [
    ("", "undamped lab08a model, spool step, full spool both ways (first run)"),
    ("_damped", "damped lab08a model (cylinder friction 1.9 kN + 2.5 kN s/m, ESTIMATED), spool step,"
                " boom-down metered at 80 % spool"),
]
DP, ETA = HYD.DP_VALVE_PA, HYD.ETA_PUMP        # ASSUMED - not from manual
MOVES = ["up", "down"]
T_SPOOL = 1.0                                  # s, spool step time of the lab08a model


def python_model(d, dp=DP, eta=ETA):
    """Lab 09b formula on the logged F, v. Returns per-sample powers [W] and Q [m^3/s]."""
    f, v = d["F"], d["v"]
    ext = v >= 0
    p_load = np.where(ext, f / A_CAP_M2, -f / A_ANN_M2)
    q = np.where(ext, A_CAP_M2, A_ANN_M2) * np.abs(v)
    return {"P_mech": f * v, "Q": q, "p_pump": np.maximum(p_load, 0.0) + dp,
            "P_pump": (np.maximum(p_load, 0.0) + dp) * q / eta}



def run_variant(tag, label):
    with open(f"{EXPORT}/simscape_energy{tag}.csv") as fh:
        SIM = {r["move"]: {k: float(v) for k, v in r.items() if k != "move"} for r in csv.DictReader(fh)}

    rows, series = [], {}
    for mv in MOVES:
        d = np.genfromtxt(f"{EXPORT}/simscape_{mv}{tag}.csv", delimiter=",", names=True)
        dt = float(d["t"][1] - d["t"][0])
        w = d["in_window"] > 0
        py = python_model(d)
        a = {"W_pos_J": np.sum(np.maximum(py["P_mech"][w], 0.0)) * dt, "vol_m3": np.sum(py["Q"][w]) * dt}
        e_pump_py = e_pump_J(a)                                    # the Lab 09b function itself
        assert abs(e_pump_py - np.sum(py["P_pump"][w]) * dt) < 1e-6 * e_pump_py, "sample sum != e_pump_J"
        s = SIM[mv]
        r = {"variant": tag or "_first", "move": mv, "spool_frac": s["spool_frac"], "dt_s": dt, "t_move_s": s["t_move_s"],
             # Simscape, from lab09b2_run.m (trapz on the solver steps)
             "sim_E_pump_kJ": s["E_pump_kJ"], "sim_E_relief_kJ": s["E_relief_kJ"],
             "sim_E_cyl_in_kJ": s["E_cyl_in_kJ"], "sim_E_cyl_out_kJ": s["E_cyl_out_kJ"],
             "sim_E_cyl_net_kJ": s["E_cyl_net_kJ"], "sim_E_mech_signed_kJ": s["E_mech_kJ"],
             "sim_E_mech_abs_kJ": s["E_mech_abs_kJ"], "sim_loss_supply_valve_kJ": s["loss_supply_valve_kJ"],
             "sim_loss_cyl_to_mech_kJ": s["loss_cyl_to_mech_kJ"], "sim_E_pump_standby_kJ": s["E_pump_standby_kJ"],
             "sim_relief_open_frac": s["relief_open_frac"], "sim_p_cap_min_bar": s["p_cap_min_bar"],
             "sim_p_rod_min_bar": s["p_rod_min_bar"], "sim_p_pump_mean_bar": s["p_pump_mean_bar"], "sim_v_peak_mps": s["v_peak_mps"],
             "py_Q_over_pump_frac": float(np.mean(py["Q"][w] > 494 / 6e4)), "py_p_over_relief_frac": float(np.mean(py["p_pump"][w] > 330e5)),
             # Python Lab 09b formula on the logged F, v (rectangle sum on the export grid)
             "py_E_mech_kJ": np.sum(np.abs(py["P_mech"][w])) * dt / 1e3,
             "py_W_pos_kJ": a["W_pos_J"] / 1e3, "py_vol_m3": a["vol_m3"],
             "py_E_pump_kJ": e_pump_py / 1e3, "py_E_hyd_kJ": e_pump_py * ETA / 1e3,
             # cross-check of the export: Simscape pump energy recomputed from the 1 ms CSV
             "sim_E_pump_from_csv_kJ": np.sum((d["p_pump"] * d["Q_pump"])[w]) * dt / 1e3}
        for key, ref in [("E_mech", "sim_E_mech_abs_kJ"), ("E_pump", "sim_E_pump_kJ"), ("E_hyd", "sim_E_pump_kJ")]:
            r[f"diff_{key}_kJ"] = r[f"py_{key}_kJ"] - r[ref]
            r[f"diff_{key}_pct"] = 100 * r[f"diff_{key}_kJ"] / r[ref]
        # where the E_pump difference comes from (Simscape minus Python), kJ. Sums to -diff_E_pump_kJ.
        r["term_relief_kJ"] = r["sim_E_relief_kJ"]
        r["term_supply_valve_kJ"] = r["sim_loss_supply_valve_kJ"]
        r["term_cyl_in_minus_W_pos_kJ"] = r["sim_E_cyl_in_kJ"] - r["py_W_pos_kJ"]
        r["term_py_valve_kJ"] = -DP * r["py_vol_m3"] / 1e3
        r["term_py_pump_loss_kJ"] = -(r["py_E_pump_kJ"] - r["py_E_hyd_kJ"])
        r["term_sum_kJ"] = sum(r[k] for k in r if k.startswith("term_"))
        # dp_valve that reproduces the Simscape pump energy on this move alone
        for eta in (ETA, 1.0):
            r[f"dp_fit_bar_eta{eta:.2f}"] = (eta * r["sim_E_pump_kJ"] - r["py_W_pos_kJ"]) * 1e3 / r["py_vol_m3"] / 1e5
        # the same loss expressed as a pressure on the swept volume [bar]
        r["sim_supply_valve_bar"] = r["sim_loss_supply_valve_kJ"] * 1e3 / r["py_vol_m3"] / 1e5
        r["sim_relief_bar"] = r["sim_E_relief_kJ"] * 1e3 / r["py_vol_m3"] / 1e5
        rows.append(r)
        series[mv] = (d, py, w)


    # --- report ---
    print(f"\n########## variant {tag or '_first'}: {label}\nPython model: dp_valve {DP / 1e5:.0f} bar, eta_pump {ETA} (both ASSUMED - not from manual);"
          f" A_cap {A_CAP_M2:.5f} m^2, A_ann {A_ANN_M2:.5f} m^2 (OFFICIAL bore/rod)")
    print("\n| move | quantity | Simscape kJ | Python kJ | diff kJ | diff % |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['move']} | E_mech = sum abs(F v) dt | {r['sim_E_mech_abs_kJ']:.1f} | {r['py_E_mech_kJ']:.1f} |"
              f" {r['diff_E_mech_kJ']:+.1f} | {r['diff_E_mech_pct']:+.2f} |")
        print(f"| {r['move']} | E_pump (Python shaft input vs Simscape hydraulic output) | {r['sim_E_pump_kJ']:.1f} |"
              f" {r['py_E_pump_kJ']:.1f} | {r['diff_E_pump_kJ']:+.1f} | {r['diff_E_pump_pct']:+.1f} |")
        print(f"| {r['move']} | E_hyd = E_pump x eta_pump (like for like) | {r['sim_E_pump_kJ']:.1f} |"
              f" {r['py_E_hyd_kJ']:.1f} | {r['diff_E_hyd_kJ']:+.1f} | {r['diff_E_hyd_pct']:+.1f} |")

    print("\nSimscape energy split, motion window [kJ]")
    print("| move | t_move s | E_pump | relief | supply valve | into cylinder | out of cylinder | net cylinder |"
          " joint work (signed) | cyl - joint | standby 0-1 s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['move']} | {r['t_move_s']:.2f} | {r['sim_E_pump_kJ']:.1f} | {r['sim_E_relief_kJ']:.1f} |"
              f" {r['sim_loss_supply_valve_kJ']:.1f} | {r['sim_E_cyl_in_kJ']:.1f} | {r['sim_E_cyl_out_kJ']:.1f} |"
              f" {r['sim_E_cyl_net_kJ']:.1f} | {r['sim_E_mech_signed_kJ']:.1f} | {r['sim_loss_cyl_to_mech_kJ']:.2f} |"
              f" {r['sim_E_pump_standby_kJ']:.1f} |")

    print("\nWhere the E_pump difference comes from: Simscape minus Python [kJ]")
    print("| move | relief flow | supply-valve drop | cylinder inlet energy above W_pos | Python valve term |"
          " Python pump-loss term | sum | Simscape - Python |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['move']} | {r['term_relief_kJ']:+.1f} | {r['term_supply_valve_kJ']:+.1f} |"
              f" {r['term_cyl_in_minus_W_pos_kJ']:+.1f} | {r['term_py_valve_kJ']:+.1f} | {r['term_py_pump_loss_kJ']:+.1f} |"
              f" {r['term_sum_kJ']:+.1f} | {-r['diff_E_pump_kJ']:+.1f} |")

    print("\ndp_valve that reproduces the Simscape pump energy, one move at a time")
    print("| move | W_pos kJ | swept volume L | dp at eta 0.85 bar | dp at eta 1.00 bar |"
          " Simscape supply-valve drop bar | Simscape relief loss as bar |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['move']} | {r['py_W_pos_kJ']:.1f} | {r['py_vol_m3'] * 1e3:.2f} | {r['dp_fit_bar_eta0.85']:.1f} |"
              f" {r['dp_fit_bar_eta1.00']:.1f} | {r['sim_supply_valve_bar']:.1f} | {r['sim_relief_bar']:.1f} |")

    # one pair for both moves: eta * E_sim - dp * V = W_pos, two equations, two unknowns
    E = np.array([r["sim_E_pump_kJ"] for r in rows]) * 1e3
    V = np.array([r["py_vol_m3"] for r in rows])
    W = np.array([r["py_W_pos_kJ"] for r in rows]) * 1e3
    eta_both, dp_both = np.linalg.solve(np.column_stack([E, -V]), W)
    print(f"exact pair for both moves: eta_pump = {eta_both:.3f}, dp_valve = {dp_both / 1e5:.1f} bar")
    print("| fit | dp_valve bar | eta_pump | up: Python kJ (error) | down: Python kJ (error) |")
    print("|---|---:|---:|---|---|")
    fits = [("Lab 09b values (ASSUMED)", DP, ETA), ("exact, both moves", dp_both, eta_both)]
    for eta in (1.0, ETA):                                    # least squares on dp with eta fixed
        dp_ls = float(np.sum(V * (eta * E - W)) / np.sum(V * V))
        fits.append((f"least squares, eta fixed {eta:.2f}", dp_ls, eta))
    for label, dp, eta in fits:
        e = (W + dp * V) / eta
        print(f"| {label} | {dp / 1e5:.1f} | {eta:.3f} | "
              + " | ".join(f"{e[i] / 1e3:.1f} ({100 * (e[i] / E[i] - 1):+.1f} %)" for i in range(2)) + " |")
    print(f"\nexport cross-check (Simscape pump energy from the 1 ms CSV vs solver steps): "
          + ", ".join(f"{r['move']} {r['sim_E_pump_from_csv_kJ']:.2f} vs {r['sim_E_pump_kJ']:.2f} kJ" for r in rows))

    # --- figure: power vs time, both models, one panel per move (same axis, kW) ---
    C_SIM, C_PY, C_MECH, INK, MUTED = "#2a78d6", "#eb6834", "#6b6b66", "#1f1f1e", "#6b6b66"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b8b7b0", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharey=True)
    for ax, r in zip(axes, rows):
        d, py, w = series[r["move"]]
        t = d["t"]
        t_end = T_SPOOL + r["t_move_s"]
        k = t <= t_end                                        # the end-stop impact after arrival is not plotted
        ax.axvspan(0, T_SPOOL, color="#f1f0ec", lw=0)
        ax.axhline(0, color="#b8b7b0", lw=0.8)
        ax.axvline(t_end, color="#b8b7b0", lw=0.8, ls=":")
        ax.plot(t[k], (d["F"] * d["v"])[k] / 1e3, color=C_MECH, lw=1.0, ls=(0, (4, 2)), label="Cylinder mechanical power F v (input to both)")
        ax.plot(t[k], (d["p_pump"] * d["Q_pump"])[k] / 1e3, color=C_SIM, lw=1.6, label="Simscape pump power p_pump Q_pump")
        ax.plot(t[w], py["P_pump"][w] / 1e3, color=C_PY, lw=1.6, label=f"Python Lab 09b pump power ({DP / 1e5:.0f} bar, eta {ETA})")
        ax.set_xlim(0, t_end + 0.25); ax.set_ylim(-640, 380); ax.grid(axis="y"); ax.set_ylabel("Power [kW]")
        ax.set_title(f"Boom-{r['move']}: Simscape {r['sim_E_pump_kJ']:.0f} kJ, Python {r['py_E_pump_kJ']:.0f} kJ"
                     f" ({r['diff_E_pump_pct']:+.0f} %) in the motion window, spool {100 * r['spool_frac']:+.0f} %",
                     loc="left", fontsize=9.5, color=INK)
        ax.text(T_SPOOL / 2, -620, "spool closed,\npump over relief", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(t_end - 0.04, -620, "end stop", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=1, frameon=False, fontsize=8.5)
    axes[1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(FIG_OUT.format(tag=tag), dpi=200); plt.close(fig)
    print(f"wrote {FIG_OUT.format(tag=tag)}")
    return rows


ALL = []
for tag, label in VARIANTS:
    if os.path.exists(f"{EXPORT}/simscape_energy{tag}.csv"):
        ALL += run_variant(tag, label)
with open(CSV_OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(ALL[0]))
    writer.writeheader(); writer.writerows(ALL)

# --- one (dp_valve, eta_pump) for all moves of all variants: least squares on eta * E_sim - dp * V = W_pos ---
E = np.array([r["sim_E_pump_kJ"] for r in ALL]) * 1e3
V = np.array([r["py_vol_m3"] for r in ALL])
W = np.array([r["py_W_pos_kJ"] for r in ALL]) * 1e3
names = [f"{r['variant']}/{r['move']}" for r in ALL]
print(f"\n########## all {len(ALL)} moves together")
print("| fit | dp_valve bar | eta_pump | " + " | ".join(names) + " |")
print("|---|---:|---:|" + "---|" * len(ALL))
fits = [("Lab 09b values (ASSUMED)", DP, ETA)]
for eta in (1.0, ETA):
    fits.append((f"least squares, eta fixed {eta:.2f}", float(np.sum(V * (eta * E - W)) / np.sum(V * V)), eta))
for label, dp, eta in fits:
    e = (W + dp * V) / eta
    print(f"| {label} | {dp / 1e5:.1f} | {eta:.3f} | " + " | ".join(f"{100 * (e[i] / E[i] - 1):+.1f} %" for i in range(len(ALL))) + " |")
print(f"\nwrote {CSV_OUT}")
