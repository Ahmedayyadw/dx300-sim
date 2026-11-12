"""Lab 11 report: builds rl/11_real_plant_tuning.md from the Lab 11 result files. No simulation.

Reads rl/11_floor_moves.csv, rl/11_gain_grid.csv, rl/11_pareto_front.csv, rl/11_points.json,
rl/11_dump_diagnosis.csv, rl/11_dump_fix_tuning.csv, rl/11_dump_fix_choice.json,
rl/11_dump_fix_checks.csv and rl/11_real_plant_tuning.csv, all written by rl/11_real_plant_tuning.py.

Usage (repo root, .venv active): python rl/11_report.py
"""
import importlib.util
import json
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab11", os.path.join(HERE, "11_real_plant_tuning.py"))
M = importlib.util.module_from_spec(_spec)
sys.modules["lab11"] = M
_spec.loader.exec_module(M)
L10, R = M.L10, M.R
status, span = L10.status, L10.span
VARIANTS = list(L10.VARIANTS)
CASES = M.ALL_CASES
ROLES = list(M.ROLES)
E_PUMP = "E_pump kJ (lower bound, ASSUMED parameters)"
HOT_PCT = 100.0          # "energy blow-up": mean E_mech more than this percent above the floor (ASSUMED)


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def table(head, lines, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    return ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"] + ["| " + " | ".join(c) + " |" for c in lines]


def est_text(e):
    """Plain description of one estimator setting."""
    parts = []
    if e.get("clamp"):
        parts.append(f"m_hat limited to {e['clamp'][0]:.0f}-{e['clamp'][1]:.0f} kg")
    if e.get("adapt"):
        parts.append(f"adaptive filter, slow T_f = {e['t_f']:g} s, detector c = {e['adapt'][0]:g}, n_hold = {e['adapt'][1]}")
    elif e.get("t_f", L10.T_F_S) != L10.T_F_S:
        parts.append(f"filter T_f = {e['t_f']:g} s")
    if e.get("acc_gate") is not None:
        parts.append(f"update only if |J_nom dw/dt| <= {e['acc_gate']:g} x g x_tip (3406 - 1120 kg)")
    return "; ".join(parts) or f"Lab 10 estimator: T_f = {L10.T_F_S:g} s, m_hat limited to 0-{2 * M.M_FULL:.0f} kg"


def build():
    moves = M.read_csv(M.FLOOR_CSV)
    grid, front = M.read_csv(M.GRID_CSV), M.read_csv(M.FRONT_CSV)
    diag, tune = M.read_csv(M.DIAG_CSV), {r["estimator"]: r for r in M.read_csv(M.FIXTUNE_CSV)}
    rows = M.read_csv(M.OUT_CSV)
    chk = {(r["estimator"], r["condition"], r["variant"]): r for r in M.read_csv(M.CHECK_CSV)}
    with open(M.POINTS_JSON) as f:
        pj = json.load(f)
    with open(M.FIX_JSON) as f:
        fj = json.load(f)
    gains = {role: (pj["points"][role]["kp"], pj["points"][role]["kd"]) for role in ROLES}
    gains.update({"L10_tuned": (M.L10_TUNED[0], M.L10_TUNED[2]), "L10_old": (M.L10_OLD[0], M.L10_OLD[2])})
    ctrl5 = ["L10_tuned", "L10_old"] + ROLES
    gtxt = {c: f"Kp {gains[c][0]:g}, Kd {gains[c][1]:g}" for c in ctrl5}
    name = {c: f"{c} ({gtxt[c]})" for c in ctrl5}
    ests = fj["estimators"]                               # base, clamp, slow, adapt, gate, together
    est_names = [e["name"] for e in ests]
    fix_label = dict(zip(est_names, ["base (Lab 10)", "fix 1: clamp", "fix 2a: slower filter",
                                     "fix 2b: adaptive filter", "fix 3: acceleration gate", "all three together"]))
    rec = min(est_names, key=lambda n: M.rank_key(tune[n]))       # best of the six on the tuning set
    idx = {(r["controller"], r["estimator"], r["condition"], r["variant"]): r for r in rows}
    keys72 = [(n, v) for n, _ in CASES for v in VARIANTS]
    n_new = int(rows[0]["n_scen"])

    def rows_of(ctrl, est="base", variants=VARIANTS, cases=CASES):
        return [idx[(ctrl, est, n, v)] for n, _ in cases for v in variants]

    def breaks(ctrl, est):
        return [f"{n} {v} ({idx[(ctrl, est, n, v)]['share_over_20cm'] * 100:.0f} %)" for n, v in keys72
                if idx[(ctrl, est, n, v)]["share_over_20cm"] > 0.05]

    def hot(ctrl, est):
        """Rows with mean E_mech more than HOT_PCT above the floor."""
        return [f"{n} {v} ({idx[(ctrl, est, n, v)]['pct_above_floor']:+.0f} %)" for n, v in keys72
                if idx[(ctrl, est, n, v)]["pct_above_floor"] > HOT_PCT]

    def tally(rs):
        c = {"BREAKS": 0, "degraded": 0, "ok": 0}
        for r in rs:
            c[status(r["share_over_20cm"] * 100)] += 1
        return f"{c['BREAKS']} break, {c['degraded']} degraded, {c['ok']} ok"

    out = []
    w = out.append

    # ------------------------------------------------------------------ header
    hyd = L10.HYD
    out += [
        "# Lab 11: energy floor, time-limited tuning and the dump failure on the real plant",
        "",
        "No RL and no training. Plant: `rl/boom_env_real.py` (Lab 10a, unchanged). Controller: `sched` of "
        "`rl/10_sched_real.py` (unchanged); `Sched11` in `rl/11_real_plant_tuning.py` adds estimator options "
        "and equals `sched` bit for bit with the options off (`--step check`: largest difference 0.0 on the "
        "45 gain sets of `rl/10_tuning_grid.csv` and on 72 noisy rows; the two Lab 10 controllers rerun here "
        f"reproduce all 144 rows of `rl/10_real_plant.csv` exactly). All numbers below come from `{M.OUT_CSV}`, "
        f"`{M.FLOOR_CSV}`, `{M.GRID_CSV}`, `{M.FRONT_CSV}`, `{M.DIAG_CSV}`, `{M.FIXTUNE_CSV}`, `{M.CHECK_CSV}`, "
        f"`{M.POINTS_JSON}` and `{M.FIX_JSON}`; this file is built from them by `rl/11_report.py`.",
        "",
        "## Status of the values",
        "",
        "- **OFFICIAL**: pump flow 494 L/min, relief 330 bar, cylinder bore and rod, boom mass and length "
        f"({R.L_BOOM_M} m, `dx300sim/dx300_params.py`), bucket mass {M.M_EMPTY:.0f} kg (Lab 10: model unverified).",
        "- **ESTIMATED**: pins, moment arm, torque limits, inertia, front point masses, friction "
        f"({R.F_COULOMB_N / 1e3:.1f} kN, {R.C_VISC_NSM / 1e3:.1f} kN s/m), full tip mass {M.M_FULL:.0f} kg "
        "(Lab 08a). **Every result in this report (E_mech, floor, time, miss rate, m_hat) is a model output "
        "and therefore ESTIMATED.** Nothing here is measured on a machine.",
        "- **ASSUMED**: all gains and grids, the 20 cm bound, the 10 s move, the payload variants and their "
        "step at 5 s, the noise levels, the selection rule of 11b and its two thresholds, every estimator "
        "parameter of 11c, the number of noise seeds.",
        f"- **{E_PUMP.split(' kJ ')[0]}**: lower bound, ASSUMED parameters (dp_valve = {hyd.DP_VALVE_PA / 1e5:.0f} bar, "
        f"eta_pump = {hyd.ETA_PUMP}, `rl/09b_hyd_params.py`, not from manual).",
        "",
        "## Use of the unseen scenarios",
        "",
        "- Tuning set: the 20 original scenarios (RNG seed 7). Every choice was made on it: the Pareto front, "
        "the three operating points, the diagnosis of 11c and every estimator parameter of 11c.",
        f"- The 100 unseen scenarios (RNG seed 123) were run once, after `{M.POINTS_JSON}` was approved and "
        f"`{M.FIX_JSON}` was written. The script refuses to load them before the approval.",
        "- **I looked at unseen results before choosing, in one respect:** `rl/10_real_plant.md` was read at "
        "the start. It holds the Lab 10 unseen results of Kp 0.25 / Kd 0.5 and Kp 0.55 / Kd 0.25, including "
        "which dump rows break. That knowledge motivated 11c and is why the dump variant with 1.0 deg noise "
        "was the diagnosis case. No Lab 11 choice uses an unseen number: the selection rule and the 11c "
        "ranking read tuning-set files only.",
        "- The 11c sweep ran in three passes, all on the tuning set and all before the unseen run. Pass 1 put "
        "the best detector factor (c = 2) and the best gate (0.5) on the edge of the grid; pass 2 added "
        "c = 0.5, 1, 1.5 and gates 1, 2; that put the best adaptive T_f (1 s) on the edge, so pass 3 added "
        "adaptive T_f = 0.5 s. The final choices are inside the grid.",
        "",
    ]

    # ------------------------------------------------------------------ 11a
    tg0 = R.gravity_torque(0.0)
    out += [
        "## 1. Energy floor (11a)",
        "",
        "### 1.1 Definition",
        "",
        "Floor = least mechanical work of a move from the start angle to the target with no overshoot "
        "(monotone, rest to rest). E_mech of the plant is the sum of |tau_cyl omega| h.",
        "",
        "```",
        "raise:  floor = dPE(m_tip) + F_c ds",
        "lower:  floor = |dPE(m_tip)| - (F_c + c_v v_ret_max) ds",
        "dPE = integral of |tau_g(theta, m_tip)| dtheta      ds = integral of r(theta) dtheta (rod travel)",
        "```",
        "",
        f"- tau_g > 0 over the whole range (checked at import; {tg0 / 1e3:.1f} kN m at theta = 0 with "
        f"{M.M_FULL:.0f} kg, ESTIMATED), so the integral of |tau_g| is the change of potential energy.",
        "- m_tip is the tip mass of the variant at the start of the move. The mass step comes at 5 s; on the "
        "tuning set the three operating points are within 20 cm of the target by then on every move "
        f"(slowest move {max(g['max_t_reach_s'] for g in front if (g['kp'], g['kd']) in [gains[r] for r in ROLES]):.2f} s).",
        "- **Friction sign.** The task said \"plus friction work\". That holds for a raise. On a lowering move "
        "the cylinder absorbs |dPE| minus what friction dissipates, so friction lowers E_mech. Adding it "
        "would put the floor above a perfect move. The floor uses minus, with viscous friction at the "
        f"retraction speed limit ({R.V_RET_MAX:.3f} m/s), the largest help friction can give. The friction "
        "term is under 1 % of the floor on each of the 20 moves of 1.2.",
        "- The floor does not depend on the move time as long as gravity alone can start the lowering move and "
        f"stop the raise ({tg0 / R.J_NOM:.2f} rad/s^2 at theta = 0 with {M.M_FULL:.0f} kg, ESTIMATED). Below that time "
        "the cylinder has to push or brake and the true minimum is above this floor. **The minimum-energy move "
        "for a given move time was not computed**, so at the fast end the gap to the floor is an upper bound "
        "of what a better controller can gain.",
        "",
        "Split of E_mech per 0.05 s control step, by the angle the step covers:",
        "",
        "- **direct**: first pass over the angle between start and target.",
        "- **overshoot**: first outward and first return pass over the angle beyond the target.",
        "- **oscillation**: every repeat pass, and travel behind the start.",
        "",
        "`E_mech - floor = overshoot + oscillation + direct excess - shortfall` (asserted on every move). "
        "**Direct excess** = direct work minus the floor of the angle really covered: reverse-torque braking, "
        "powered lowering, viscous friction, kinetic energy carried past the target. **Shortfall** = floor of "
        "the angle between the furthest point reached and the target: work saved by not finishing the move. "
        "The task asked for two terms; the other two are needed to close the balance. A reversal inside one "
        "control step is attributed by net travel.",
        "",
        "### 1.2 Floor of every move, 20 original scenarios, baseline (3406 kg)",
        "",
        "All values ESTIMATED. Cells: E_mech kJ (percent above the floor).",
        "",
    ]
    mv = {(m["controller"], m["variant"], int(m["move"])): m for m in moves}
    n_orig = max(int(m["move"]) for m in moves)
    lines = []
    for i in range(1, n_orig + 1):
        m0 = mv[(ctrl5[0], "baseline", i)]
        lines.append([str(i), f"{m0['th0_deg']:.1f}", f"{m0['target_deg']:.1f}", f"{m0['floor_grav_kJ']:.1f}",
                      f"{m0['floor_fric_kJ']:+.2f}", f"{m0['floor_kJ']:.1f}"]
                     + [f"{mv[(c, 'baseline', i)]['E_mech_kJ']:.1f} ({100 * (mv[(c, 'baseline', i)]['E_mech_kJ'] / m0['floor_kJ'] - 1):+.0f} %)"
                        for c in ctrl5])

    def agg(c, v):
        ms = [mv[(c, v, i)] for i in range(1, n_orig + 1)]
        a = {k: mean(ms, k) for k in ("floor_kJ", "E_mech_kJ", "overshoot_kJ", "oscillation_kJ", "direct_excess_kJ",
                                      "shortfall_kJ", "t_reach_s", "floor_grav_kJ", "floor_fric_kJ")}
        a["pct"] = 100 * (a["E_mech_kJ"] / a["floor_kJ"] - 1)
        return a

    a0 = agg(ctrl5[0], "baseline")
    lines.append(["mean", "", "", f"{a0['floor_grav_kJ']:.1f}", f"{a0['floor_fric_kJ']:+.2f}", f"{a0['floor_kJ']:.1f}"]
                 + [f"{agg(c, 'baseline')['E_mech_kJ']:.1f} ({agg(c, 'baseline')['pct']:+.1f} %)" for c in ctrl5])
    out += table(["move", "start deg", "target deg", "gravity kJ", "friction kJ", "floor kJ"] + [name[c] for c in ctrl5],
                 lines, ["---:"] * (6 + len(ctrl5)))
    out += ["", "### 1.3 Split of the excess, 20 original scenarios, nominal plant", "",
            "Mean over the 20 moves, kJ, all ESTIMATED. Payload variants (step at 5 s): `baseline` 3406 kg; "
            "`fill_half` 2263 -> 3406 kg; `fill` 1120 -> 3406 kg; `dump` 3406 -> 1120 kg.", ""]

    def split_line(label, v, a, extra):
        return [label, v, f"{a['floor_kJ']:.1f}", f"{a['E_mech_kJ']:.1f}", f"{a['pct']:+.1f}",
                f"{a['E_mech_kJ'] - a['floor_kJ']:+.1f}", f"{a['overshoot_kJ']:.1f}", f"{a['oscillation_kJ']:.1f}",
                f"{a['direct_excess_kJ']:+.1f}", f"{-a['shortfall_kJ']:+.1f}"] + extra

    split_head = ["controller", "variant", "floor", "E_mech", "above floor %", "excess", "overshoot", "oscillation",
                  "direct excess", "shortfall", "reach 20 cm s"]
    split_align = ["---", "---"] + ["---:"] * 9
    out += table(split_head, [split_line(name[c], v, agg(c, v), [f"{agg(c, v)['t_reach_s']:.2f}"])
                              for c in ctrl5 for v in VARIANTS], split_align)

    def agg_row(r):
        return {"floor_kJ": r["mean_floor_kJ"], "E_mech_kJ": r["mean_E_mech_kJ"], "pct": r["pct_above_floor"],
                "overshoot_kJ": r["mean_overshoot_kJ"], "oscillation_kJ": r["mean_oscillation_kJ"],
                "direct_excess_kJ": r["mean_direct_excess_kJ"], "shortfall_kJ": r["mean_shortfall_kJ"]}

    out += ["", f"### 1.4 Split of the excess, {n_new} unseen scenarios, no stress (`none`)", "",
            f"Mean over the {n_new} moves, kJ, all ESTIMATED. Lab 10 estimator.", ""]
    out += table(split_head, [split_line(name[c], v, agg_row(idx[(c, "base", "none", v)]),
                                         [f"{idx[(c, 'base', 'none', v)]['mean_t_reach_s']:.2f}"])
                              for c in ctrl5 for v in VARIANTS], split_align)
    out += ["", "Percent above the floor over all 72 rows of Lab 10 (18 conditions x 4 variants), unseen, ESTIMATED:", ""]
    out += table(["controller", "above floor %: lowest row", "highest row", "mean", "overshoot kJ (mean)",
                  "oscillation kJ (mean)", "direct excess kJ (mean)", "shortfall kJ (mean)"],
                 [[name[c], f"{min(r['pct_above_floor'] for r in rows_of(c)):+.1f}",
                   f"{max(r['pct_above_floor'] for r in rows_of(c)):+.1f}", f"{mean(rows_of(c), 'pct_above_floor'):+.1f}",
                   f"{mean(rows_of(c), 'mean_overshoot_kJ'):.1f}", f"{mean(rows_of(c), 'mean_oscillation_kJ'):.1f}",
                   f"{mean(rows_of(c), 'mean_direct_excess_kJ'):+.1f}", f"{-mean(rows_of(c), 'mean_shortfall_kJ'):+.1f}"]
                  for c in ctrl5])
    t_, o_, f_, b_, e_ = (agg(c, "baseline") for c in ctrl5)
    low = [i for i in range(1, n_orig + 1) if mv[("fast", "baseline", i)]["target_deg"] < mv[("fast", "baseline", i)]["th0_deg"]]
    f_low = [100 * (mv[("fast", "baseline", i)]["E_mech_kJ"] / mv[("fast", "baseline", i)]["floor_kJ"] - 1) for i in low]
    f_up = [100 * (mv[("fast", "baseline", i)]["E_mech_kJ"] / mv[("fast", "baseline", i)]["floor_kJ"] - 1)
            for i in range(1, n_orig + 1) if i not in low]
    out += [
        "",
        "Findings (20 original scenarios, baseline, ESTIMATED):",
        "",
        f"- **The Lab 10 tuned gains are below the floor** ({t_['pct']:+.1f} %). Overshoot and oscillation are "
        f"{t_['overshoot_kJ']:.1f} and {t_['oscillation_kJ']:.1f} kJ; the saving is shortfall "
        f"({-t_['shortfall_kJ']:+.1f} kJ): the move stops short of the target. The Lab 07 rule (lowest energy "
        "with no miss) rewarded an unfinished move.",
        f"- The Lab 10 old gains are {o_['pct']:+.1f} % above the floor: {o_['overshoot_kJ']:.1f} kJ overshoot and "
        f"{o_['oscillation_kJ']:.1f} kJ oscillation of {o_['E_mech_kJ'] - o_['floor_kJ']:.1f} kJ excess.",
        f"- `economical` sits on the floor ({e_['pct']:+.1f} %) at {e_['t_reach_s']:.2f} s. `balanced` is "
        f"{b_['pct']:+.1f} % above at {b_['t_reach_s']:.2f} s, `fast` {f_['pct']:+.1f} % at {f_['t_reach_s']:.2f} s.",
        f"- The excess of `fast` is direct excess ({f_['direct_excess_kJ']:+.1f} of {f_['E_mech_kJ'] - f_['floor_kJ']:+.1f} kJ), "
        f"not overshoot ({f_['overshoot_kJ']:.1f} kJ). It sits on the lowering moves: {min(f_low):+.0f} to "
        f"{max(f_low):+.0f} % above the floor on the {len(low)} lowering moves, {min(f_up):+.0f} to {max(f_up):+.0f} % "
        f"on the {n_orig - len(low)} raising moves. The cylinder pushes the boom down: its positive work on the "
        f"{len(low)} lowering moves is {np.mean([mv[('fast', 'baseline', i)]['W_pos_kJ'] for i in low]):.1f} kJ per move "
        f"for `fast`, {np.mean([mv[('balanced', 'baseline', i)]['W_pos_kJ'] for i in low]):.1f} kJ for `balanced` and "
        f"{np.mean([mv[('economical', 'baseline', i)]['W_pos_kJ'] for i in low]):.1f} kJ for `economical`; a lowering "
        "move driven by gravity needs none. The push is paid twice, once to give it and once to absorb it.",
        "- On the `fill` variants the excess is overshoot and oscillation after the mass step at 5 s: the boom "
        "sags until the estimate has followed, then comes back.",
        "",
    ]

    # ------------------------------------------------------------------ 11b
    ok = [g for g in grid if g["max_cm"] < 20]
    role_of = {(gains[r][0], gains[r][1]): r for r in ROLES}
    adm = [g for g in front if g["pct_above_floor"] >= 0 and g["max_peak_overshoot_cm"] < pj["overshoot_max_cm"]]
    kps, kds = sorted({g["kp"] for g in grid}), sorted({g["kd"] for g in grid})
    edge = [g for g in front if g["kp"] in (kps[0], kps[-1]) or g["kd"] in (kds[0], kds[-1])]
    fr = {role_of.get((g["kp"], g["kd"])): g for g in front}
    out += [
        "## 2. Time-limited tuning (11b)",
        "",
        f"Grid: {len(kps)} Kp values from {kps[0]:g} to {kps[-1]:g} times {len(kds)} Kd values from {kds[0]:g} to "
        f"{kds[-1]:g} (ASSUMED; the Lab 07 values plus 1-1.2-1.5-2-2.5-3-4-5-6-8 steps per decade), Ki = 0, "
        f"nominal plant, `baseline`, the {int(grid[0]['n_scen'])} original scenarios. {len(ok)} of {len(grid)} gain "
        f"sets have a worst-case error under 20 cm; {len(front)} of them are on the Pareto front of mean E_mech "
        "against mean time to come within 20 cm.",
        "",
        "![Pareto front of mean E_mech against mean time to reach 20 cm, with the three operating points](11_pareto.png)",
        "",
        "Pareto front, all values ESTIMATED. Noise columns: the same 20 scenarios with 1.0 deg sensor noise "
        "(tuning set, used for information only).",
        "",
    ]
    out += table(["#", "Kp", "Kd", "reach 20 cm s (mean)", "(max)", "E_mech kJ", "above floor %", E_PUMP,
                  "worst error cm", "peak overshoot cm", "flow limit %", "noise: E_mech kJ",
                  "noise: worst miss % (4 variants)", "point"],
                 [[str(i + 1), f"{g['kp']:g}", f"{g['kd']:g}", f"{g['mean_t_reach_s']:.2f}", f"{g['max_t_reach_s']:.2f}",
                   f"{g['mean_E_mech_kJ']:.1f}", f"{g['pct_above_floor']:+.1f}", f"{g['mean_E_pump_kJ']:.1f}",
                   f"{g['max_cm']:.1f}", f"{g['max_peak_overshoot_cm']:.1f}", f"{g['share_flow_limit'] * 100:.1f}",
                   f"{g['n1_baseline_E_mech_kJ']:.1f}", f"{g['n1_worst_miss_share'] * 100:.0f}",
                   f"**{role_of[(g['kp'], g['kd'])]}**" if (g["kp"], g["kd"]) in role_of else
                   ("Lab 10 tuned" if (g["kp"], g["kd"]) == (M.L10_TUNED[0], M.L10_TUNED[2]) else "")]
                  for i, g in enumerate(front)], ["---:"] * 13 + ["---"])
    out += [
        "",
        "Selection rule (tuning set only; thresholds ASSUMED):",
        "",
        f"1. Admissible = front points with mean E_mech at or above the mean floor and peak overshoot under "
        f"{pj['overshoot_max_cm']:.0f} cm. {len(adm)} points are left. The front points below the floor save energy "
        "only by stopping short (section 1); the fastest front points overshoot by up to "
        f"{max(g['max_peak_overshoot_cm'] for g in front):.0f} cm before they settle.",
        "2. `economical` = lowest E_mech admissible.",
        f"3. `fast` = lowest E_mech within {pj['fast_band_s']:.2f} s (two control steps) of the fastest admissible.",
        "4. `balanced` = admissible point between them that is closest to (time of `fast`, energy of "
        "`economical`), axes scaled by the span between the two.",
        "",
        "**Approved operating points** (user approval in chat, 2026-11-12, after the front was shown):",
        "",
    ]
    out += table(["point", "Kp", "Ki", "Kd", "reach 20 cm s", "E_mech kJ", "above floor %", E_PUMP, "worst error cm"],
                 [[r, f"{gains[r][0]:g}", "0", f"{gains[r][1]:g}", f"{fr[r]['mean_t_reach_s']:.2f}",
                   f"{fr[r]['mean_E_mech_kJ']:.1f}", f"{fr[r]['pct_above_floor']:+.1f}", f"{fr[r]['mean_E_pump_kJ']:.1f}",
                   f"{fr[r]['max_cm']:.1f}"] for r in ROLES])
    out += ["", "Front points on the edge of the grid: "
            + (", ".join(f"Kp {g['kp']:g} / Kd {g['kd']:g}" for g in edge) or "none")
            + ". They are at the slow end, below the floor, and not admissible; the three points are inside the grid.", ""]

    # ------------------------------------------------------------------ evaluation
    out += [
        f"## 3. The three points on the {n_new} unseen scenarios, all 72 rows of Lab 10",
        "",
        f"Lab 10 estimator (T_f = {L10.T_F_S} s, m_hat limited to 0-{2 * M.M_FULL:.0f} kg), 10 s per move, all values "
        "ESTIMATED. Status: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok. The two Lab 10 controllers are "
        "rerun for reference.",
        "",
    ]
    out += table(["controller", "miss %", "E_mech kJ", "above floor %", E_PUMP, "reach 20 cm s", "flow limit %", "status over 72 rows"],
                 [[name[c], span([r["share_over_20cm"] * 100 for r in rows_of(c)], 0),
                   span([r["mean_E_mech_kJ"] for r in rows_of(c)]), span([r["pct_above_floor"] for r in rows_of(c)]),
                   span([r["mean_E_pump_kJ"] for r in rows_of(c)]), span([r["mean_t_reach_s"] for r in rows_of(c)], 2),
                   span([r["share_flow_limit"] * 100 for r in rows_of(c)]), tally(rows_of(c))] for c in ROLES + ["L10_tuned", "L10_old"]],
                 ["---"] + ["---:"] * 6 + ["---"])
    out += [""] + [f"- Rows over 5 % miss, {name[c]}: " + ("; ".join(breaks(c, "base")) or "none") + "."
                   for c in ROLES + ["L10_tuned", "L10_old"]]
    viol = int(sum(r["flow_violations"] for r in rows))
    hf = idx[("fast", "base", "J_x0.38", "dump")]
    out += [f"- **The miss rate hides an instability of `fast`.** Rows with E_mech more than {HOT_PCT:.0f} % above the "
            f"floor (threshold ASSUMED): " + "; ".join(f"`{c}`: " + (", ".join(hot(c, "base")) or "none") for c in ROLES)
            + f". (The Lab 10 old gains have {len(hot('L10_old', 'base'))} such rows, from overshoot; the Lab 10 tuned "
            f"gains {len(hot('L10_tuned', 'base'))}.) On `J_x0.38` with a light bucket (`fill` before 5 s, `dump` after) "
            f"`fast` oscillates without any sensor noise: {hf['mean_E_mech_kJ']:.0f} kJ on `J_x0.38` `dump`, of which "
            f"{hf['mean_oscillation_kJ']:.0f} kJ oscillation, m_hat / m_tip spread {hf['m_hat_over_m_std']:.2f}, "
            f"{hf['share_over_20cm'] * 100:.0f} % misses. `J_x0.38` `baseline` is quiet "
            f"({idx[('fast', 'base', 'J_x0.38', 'baseline')]['mean_E_mech_kJ']:.0f} kJ), and with the adaptive filter of 11c "
            f"the `dump` row costs {idx[('fast', rec, 'J_x0.38', 'dump')]['mean_E_mech_kJ']:.0f} kJ. So it takes low inertia, a "
            "light bucket and the 0.2 s estimate together (the estimator uses J_nom). The tuning grid ran on the "
            "nominal plant and could not see it.",
            f"- 330 bar could not hold the flow limit in {viol} substeps over all rows of `{M.OUT_CSV}`.", "",
            "Columns per point: f = fast, b = balanced, e = economical. Miss % = share of moves with final tip "
            "error over 20 cm. m_hat / m_tip = estimate over true tip mass in the last 2 s.", ""]
    h3 = (["condition", "variant", "floor kJ"] + [f"miss %: {x}" for x in "fbe"] + [f"E_mech kJ: {x}" for x in "fbe"]
          + [f"above floor %: {x}" for x in "fbe"] + [f"{E_PUMP}: {x}" for x in "fbe"] + [f"reach s: {x}" for x in "fbe"]
          + ["m_hat / m_tip (e)", "status f / b / e"])
    l3 = []
    for n, v in keys72:
        rs = [idx[(c, "base", n, v)] for c in ROLES]
        l3.append([n, v, f"{rs[0]['mean_floor_kJ']:.1f}"] + [f"{r['share_over_20cm'] * 100:.0f}" for r in rs]
                  + [f"{r['mean_E_mech_kJ']:.1f}" for r in rs] + [f"{r['pct_above_floor']:+.1f}" for r in rs]
                  + [f"{r['mean_E_pump_kJ']:.1f}" for r in rs] + [f"{r['mean_t_reach_s']:.2f}" for r in rs]
                  + [f"{rs[2]['m_hat_over_m_mean']:.3f} ± {rs[2]['m_hat_over_m_std']:.3f}",
                     " / ".join(status(r["share_over_20cm"] * 100) for r in rs)])
    out += table(h3, l3, ["---", "---"] + ["---:"] * 16 + ["---", "---"])
    out.append("")

    # ------------------------------------------------------------------ 11c
    sig = math.radians(1.0)
    sig_om = sig / 0.05 * 0.1
    sig_acc = math.sqrt(2.0) * sig_om / 0.05
    gx = R.G * R.x_tip(0.0)
    m_noise = R.J_NOM * sig_acc / gx
    didx = {(r["controller"], r["condition"], r["estimator"]): r for r in diag}
    dctrl = ["L10_tuned"] + ROLES
    n_seeds = int(diag[0]["n_seeds"])
    out += [
        "## 4. Dump failure (11c)",
        "",
        "### 4.1 Diagnosis",
        "",
        f"Tuning set: 20 original scenarios x {n_seeds} noise seeds (ASSUMED) = {int(diag[0]['n_moves'])} moves per cell, "
        "`dump` variant (3406 -> 1120 kg at 5 s), the four conditions with 1.0 deg sensor noise. Three "
        "counterfactual estimators isolate the cause: **true m_tip** (feed-forward uses the true mass), "
        "**true state** (the estimator reads the noise-free angle and speed; the PD loop still sees the noise), "
        f"**no limit** (m_hat not limited to 0-{2 * M.M_FULL:.0f} kg). All values ESTIMATED.",
        "",
    ]
    out += table(["controller", "m_hat error for 20 cm, kg", "condition", "miss %: Lab 10 estimator", "true m_tip",
                  "true state", "no limit", "m_hat / m_tip", "m_hat std kg", "updates at the 0 kg limit %",
                  "mean signed error deg", "no limit: m_hat / m_tip"],
                 [[name[c], f"{M.kg_per_20cm(gains[c][0]):.0f}", n] + [f"{didx[(c, n, e)]['share_over_20cm'] * 100:.1f}"
                                                                      for e in ("base", "oracle", "clean_state", "no_limit")]
                  + [f"{didx[(c, n, 'base')]['m_hat_over_m_mean']:.3f} ± {didx[(c, n, 'base')]['m_hat_over_m_std']:.3f}",
                     f"{didx[(c, n, 'base')]['m_hat_tail_std_kg']:.0f}", f"{didx[(c, n, 'base')]['share_at_lower_limit'] * 100:.1f}",
                     f"{didx[(c, n, 'base')]['mean_err_signed_deg']:+.2f}",
                     f"{didx[(c, n, 'no_limit')]['m_hat_over_m_mean']:.3f} ± {didx[(c, n, 'no_limit')]['m_hat_over_m_std']:.3f}"]
                  for c in dctrl for n, _ in M.N1_CASES], ["---", "---:", "---"] + ["---:"] * 4 + ["---", "---:", "---:", "---:", "---"])
    worst = max((didx[(c, n, "base")] for c in dctrl for n, _ in M.N1_CASES), key=lambda r: r["share_over_20cm"])
    base_std = [didx[(c, n, "base")]["m_hat_tail_std_kg"] for c in dctrl for n, _ in M.N1_CASES if "J0.38" not in n]
    out += [
        "",
        "Cause, in four steps:",
        "",
        f"1. **The estimate is noisy in kilograms, not in percent.** The estimator differentiates the measured "
        f"speed. With 1.0 deg angle noise the speed noise is {sig_om:.4f} rad/s and the acceleration noise "
        f"{sig_acc:.2f} rad/s^2 per step. Times J_nom / (g x_tip) that is about {m_noise:.0f} kg per raw sample; "
        f"after the 0.2 s filter {min(base_std):.0f}-{max(base_std):.0f} kg remain. Lab 10 shows the same spread at "
        "3406 kg (0.177 x 3406 = 603 kg). After the dump the same kilograms sit on 1120 kg, which is the "
        "+-0.5 to +-0.75 of Lab 10. The spread of the ratio is a division by a smaller mass, not a worse estimator. "
        "This noise is differenced sensor noise: consecutive errors cancel, so it is fast and the boom hardly "
        f"follows it. That is why {chk[('base', 'noise_1.0', 'baseline')]['m_hat_tail_std_kg']:.0f} kg of spread gives "
        f"{chk[('base', 'noise_1.0', 'baseline')]['share_over_20cm'] * 100:.0f} % misses on `baseline` (`economical`, "
        f"`{M.CHECK_CSV}`), although a constant error of {M.kg_per_20cm(gains['economical'][0]):.0f} kg would give 20 cm.",
        "2. **The lower limit at 0 kg turns the noise into a bias.** With a spread of that size around 1120 kg the "
        "estimate reaches the 0 kg limit (column \"updates at the 0 kg limit\"). The limit cuts the low side only, so "
        "the mean of m_hat rises above the true mass (m_hat / m_tip > 1). Without the limit the mean is back at 1 "
        "and the misses are gone, with the same spread.",
        "3. **The bias lifts the boom.** The feed-forward holds more mass than is there. The mean signed error is "
        "negative on every row: the boom ends above the target.",
        "4. **A small Kp cannot push it back, and Ki = 0.** With P action alone a constant m_hat error of "
        f"{M.kg_per_20cm(M.L10_TUNED[0]):.0f} kg gives 20 cm at Kp = {M.L10_TUNED[0]:g}; at Kp = {gains['fast'][0]:g} it takes "
        f"{M.kg_per_20cm(gains['fast'][0]):.0f} kg. That is why the Lab 10 tuned gains break and the old gains mostly do not.",
        "",
        f"Evidence: with the true mass, with a noise-free estimator input, or with no limit the miss rate is "
        f"{max(didx[(c, n, e)]['share_over_20cm'] for c in dctrl for n, _ in M.N1_CASES for e in ('oracle', 'no_limit')) * 100:.1f} % "
        f"at most (true m_tip and no limit; {max(didx[(c, n, 'clean_state')]['share_over_20cm'] for c in dctrl for n, _ in M.N1_CASES) * 100:.1f} % "
        f"at most with the true state), against up to {worst['share_over_20cm'] * 100:.0f} % with the Lab 10 estimator "
        f"({worst['controller']}, {worst['condition']}). `J_x0.38` is the worst case because the estimator "
        "uses J_nom: the real boom is lighter, reacts more to the noisy command, and the wrong inertia term adds "
        "to the estimate noise.",
        "",
        "### 4.2 Fixes: parameters chosen on the tuning set",
        "",
        f"Sweep on the 20 original scenarios: three approved points x 72 rows = {int(tune['base']['cells'])} cells per "
        f"setting, {int(fj['noise_seeds'])} noise seeds on the noisy rows, {len(tune)} settings in `{M.FIXTUNE_CSV}`. "
        "Rule (ASSUMED): lowest worst-cell miss rate, then lowest mean miss rate, then lowest mean E_mech. "
        "All values ESTIMATED.",
        "",
    ]
    for e in ests:
        w(f"- **{fix_label[e['name']]}** (`{e['name']}`): {est_text(e)}.")
    out.append("")
    show = list(dict.fromkeys(est_names + [n for n in tune if n.startswith("slow_") or n.startswith("gate_")]
                              + [fj["roles"]["clamp_filter"]]))
    out += table(["setting", "worst miss %", "cells > 5 %", "mean miss %", "dump, 1.0 deg: worst miss %", "mean miss %",
                  "m_hat / m_tip", "other rows: worst miss %", "mean miss %", "mean E_mech kJ", "steps with update %"],
                 [[f"`{n}`" + (f" ({fix_label[n]})" if n in fix_label and n != "base" else ""), f"{tune[n]['worst_miss_pct']:.1f}",
                   f"{tune[n]['cells_over_5pct']:.0f}", f"{tune[n]['mean_miss_pct']:.2f}", f"{tune[n]['dump_n1_worst_miss_pct']:.1f}",
                   f"{tune[n]['dump_n1_mean_miss_pct']:.2f}",
                   f"{tune[n]['dump_n1_m_hat_ratio_mean']:.3f} ± {tune[n]['dump_n1_m_hat_ratio_std']:.3f}",
                   f"{tune[n]['other_worst_miss_pct']:.1f}", f"{tune[n]['other_mean_miss_pct']:.2f}",
                   f"{tune[n]['mean_E_mech_kJ']:.1f}", f"{tune[n]['mean_est_update_share'] * 100:.0f}"] for n in show],
                 ["---"] + ["---:"] * 5 + ["---"] + ["---:"] * 4)
    out += ["", f"Best of the six by the rule, used as **the recommended fix** below: `{rec}` ({fix_label[rec]}).", ""]

    # 4.3 unseen
    dump_rows = [(n, "dump") for n, _ in CASES]
    other_rows = [(n, v) for n, v in keys72 if v != "dump"]
    out += [f"### 4.3 Fixes on the {n_new} unseen scenarios", "",
            "Dump rows with 1.0 deg sensor noise, all values ESTIMATED. Cells: miss % | m_hat / m_tip in the last 2 s.", ""]
    l43 = []
    for c in ROLES:
        for n, _ in M.N1_CASES:
            l43.append([c, n] + [f"{idx[(c, e, n, 'dump')]['share_over_20cm'] * 100:.0f} \\| "
                                 f"{idx[(c, e, n, 'dump')]['m_hat_over_m_mean']:.3f} ± {idx[(c, e, n, 'dump')]['m_hat_over_m_std']:.3f}"
                                 for e in est_names])
    out += table(["point", "condition"] + [fix_label[e] for e in est_names], l43, ["---", "---"] + ["---:"] * len(est_names))
    out += ["", f"All 18 dump rows and the 54 other rows, per point, all values ESTIMATED. \"Rows worse\" = rows with "
            "a higher miss rate than the Lab 10 estimator. dE_mech = change of the mean E_mech against the Lab 10 "
            "estimator.", ""]
    l44 = []
    for c in ROLES:
        for e in est_names:
            d = [idx[(c, e, n, v)] for n, v in dump_rows]
            o = [idx[(c, e, n, v)] for n, v in other_rows]
            db = [idx[(c, "base", n, v)] for n, v in dump_rows]
            ob = [idx[(c, "base", n, v)] for n, v in other_rows]
            l44.append([c, fix_label[e], f"{max(r['share_over_20cm'] for r in d) * 100:.0f}", f"{mean(d, 'share_over_20cm') * 100:.2f}",
                        f"{sum(r['share_over_20cm'] > 0.05 for r in d)}",
                        span([r["m_hat_over_m_std"] for r in d if r["sigma_th_deg"] == 1.0], 3),
                        f"{max(r['share_over_20cm'] for r in o) * 100:.0f}", f"{mean(o, 'share_over_20cm') * 100:.2f}",
                        f"{sum(r['share_over_20cm'] > 0.05 for r in o)}",
                        f"{sum(a['share_over_20cm'] > b['share_over_20cm'] for a, b in zip(o, ob))}",
                        f"{mean(o, 'mean_E_mech_kJ') - mean(ob, 'mean_E_mech_kJ'):+.1f}",
                        f"{mean(d, 'mean_E_mech_kJ') - mean(db, 'mean_E_mech_kJ'):+.1f}",
                        f"{mean(o, 'mean_t_reach_s') - mean(ob, 'mean_t_reach_s'):+.2f}"])
    out += table(["point", "estimator", "dump rows: worst miss %", "mean miss %", "rows > 5 %", "m_hat / m_tip std at 1.0 deg",
                  "other rows: worst miss %", "mean miss %", "rows > 5 %", "rows worse", "dE_mech kJ, other rows",
                  "dE_mech kJ, dump rows", "d reach s, other rows"], l44, ["---", "---"] + ["---:"] * 11)

    def loss(e):
        o = [idx[(c, e, n, v)] for c in ROLES for n, v in other_rows]
        ob = [idx[(c, "base", n, v)] for c in ROLES for n, v in other_rows]
        worse = sum(a["share_over_20cm"] > b["share_over_20cm"] for a, b in zip(o, ob))
        d = [idx[(c, e, n, "dump")] for c in ROLES for n, _ in CASES]
        return (worse, max(a["share_over_20cm"] - b["share_over_20cm"] for a, b in zip(o, ob)) * 100,
                mean(o, "mean_E_mech_kJ") - mean(ob, "mean_E_mech_kJ"), max(r["share_over_20cm"] for r in d) * 100,
                sum(r["share_over_20cm"] > 0.05 for r in d), max(r["share_over_20cm"] for r in o) * 100)

    out += ["", "Verdict per fix on the unseen set (three points, 54 dump rows and 162 other rows in total). "
            "dE_mech = change of the mean E_mech of the 54 other rows of a point against the Lab 10 estimator.", ""]
    for e in est_names[1:]:
        worse, dmax, _, dworst, dbreak, oworst = loss(e)
        de = {c: mean([idx[(c, e, n, v)] for n, v in other_rows], "mean_E_mech_kJ")
              - mean([idx[(c, "base", n, v)] for n, v in other_rows], "mean_E_mech_kJ") for c in ROLES}
        rise = max(idx[(c, e, n, v)]["mean_E_mech_kJ"] - idx[(c, "base", n, v)]["mean_E_mech_kJ"]
                   for c in ROLES for n, v in other_rows)
        verdict = (f"**loss on the other rows**: {worse} of {3 * len(other_rows)} rows have a higher miss rate "
                   f"(largest rise {dmax:.0f} points, worst row {oworst:.0f} %)" if worse else
                   f"no other row has a higher miss rate (worst row {oworst:.0f} %)")
        w(f"- **{fix_label[e]}**: dump rows worst miss {dworst:.0f} %, {dbreak} of {3 * len(dump_rows)} over 5 %; {verdict}; "
          "dE_mech " + ", ".join(f"`{c}` {de[c]:+.1f} kJ" for c in ROLES) + f"; largest rise on one row {rise:+.1f} kJ.")
    c2, c4 = chk[("slow_Tf2", "none", "dump")], chk[("slow_Tf4", "none", "fill")]
    g5b, g5d, g2b = chk[("gate_0.5", "noise_1.0", "baseline")], chk[("gate_0.5", "noise_1.0", "dump")], chk[("gate_2", "noise_1.0", "baseline")]
    b1b = chk[("base", "noise_1.0", "baseline")]
    rec_de = {c: mean([idx[(c, rec, n, v)] for n, v in other_rows], "mean_E_mech_kJ")
              - mean([idx[(c, "base", n, v)] for n, v in other_rows], "mean_E_mech_kJ") for c in ROLES}
    out += [
        "",
        f"**No loss on the other rows, for the recommended fix (`{rec}`):** no row of the 162 has a higher miss "
        "rate. Energy is not free on two points: mean E_mech of the other rows changes by "
        + ", ".join(f"{rec_de[c]:+.1f} kJ on `{c}`" for c in ROLES)
        + ". On `balanced` and `economical` the cost sits on the rows with a mass step (the slower estimate "
        "follows the step later); `baseline` rows get cheaper. On `fast` the saving comes from the `J_x0.38` "
        "`fill` and `dump` rows, where the oscillation of section 3 disappears "
        f"({idx[('fast', 'base', 'J_x0.38', 'dump')]['mean_E_mech_kJ']:.0f} -> "
        f"{idx[('fast', rec, 'J_x0.38', 'dump')]['mean_E_mech_kJ']:.0f} kJ on `J_x0.38` `dump`).",
        "",
        f"Why the fixes behave this way (spot checks on the tuning set in `{M.CHECK_CSV}`, `economical`):",
        "",
        f"- **Clamp to {M.M_EMPTY:.0f}-{M.M_FULL:.0f} kg.** After the dump the true mass is exactly the lower limit, and "
        "on `baseline` exactly the upper limit. The noise can then move m_hat to one side only. That is the "
        "mechanism of 4.1 made stronger: bias up after a dump, bias down at full load. Unseen set, `economical`, "
        f"`noise_1.0`: m_hat / m_tip {idx[('economical', 'clamp', 'noise_1.0', 'dump')]['m_hat_over_m_mean']:.2f} on `dump` "
        f"(mean signed error {idx[('economical', 'clamp', 'noise_1.0', 'dump')]['mean_err_signed_deg']:+.1f} deg, boom high) "
        f"and {idx[('economical', 'clamp', 'noise_1.0', 'baseline')]['m_hat_over_m_mean']:.2f} on `baseline` "
        f"({idx[('economical', 'clamp', 'noise_1.0', 'baseline')]['mean_err_signed_deg']:+.1f} deg, boom low). With the "
        "clamp the `baseline`, `fill_half` and `fill` rows at 1.0 deg break too on `balanced` and `economical`. "
        "On `fast` the large Kp hides the bias. The clamp is safe only once the "
        "spread is small, and even with the adaptive filter it keeps a bias (tuning set, "
        f"`{fj['roles']['clamp_filter']}`: m_hat / m_tip {tune[fj['roles']['clamp_filter']]['dump_n1_m_hat_ratio_mean']:.3f}, "
        f"worst cell {tune[fj['roles']['clamp_filter']]['worst_miss_pct']:.0f} % miss).",
        "- **Slower filter.** It cuts the spread, so the estimate stops reaching the limit and the bias goes. "
        f"Too slow and the estimate has not followed the 5 s step by the end of the move: without any noise, "
        f"T_f = 2 s gives {c2['share_over_20cm'] * 100:.0f} % misses on `dump` (m_hat / m_tip {c2['m_hat_over_m_mean']:.2f} at "
        f"the end) and T_f = 4 s {c4['share_over_20cm'] * 100:.0f} % on `fill` ({c4['m_hat_over_m_mean']:.2f}).",
        "- **Adaptive filter.** Slow gain in steady state, restart on a detected change. The threshold scales "
        "with the estimated noise. With c >= 2 it is above the 2286 kg dump step at 1.0 deg noise and the filter "
        "equals the slow filter there (identical m_hat statistics in the sweep); c = 1 was chosen in pass 2.",
        "- **Acceleration gate.** Without noise it does nothing harmful "
        f"({chk[('gate_0.5', 'none', 'dump')]['share_est_updates'] * 100:.0f} % of the steps still update, "
        f"{chk[('gate_0.5', 'none', 'dump')]['share_over_20cm'] * 100:.0f} % misses on `dump`). With 1.0 deg noise the measured "
        f"acceleration term is rarely small: gate 0.5 updates on {g5b['share_est_updates'] * 100:.0f} % of the steps. "
        f"The estimate stays unbiased (m_hat / m_tip {g5b['m_hat_over_m_mean']:.3f} on `baseline`) and its spread even "
        f"falls ({g5b['m_hat_tail_std_kg']:.0f} kg against {b1b['m_hat_tail_std_kg']:.0f} kg), yet the misses rise "
        f"({g5b['share_over_20cm'] * 100:.0f} % on `baseline`, {g5d['share_over_20cm'] * 100:.0f} % on `dump`, against "
        f"{b1b['share_over_20cm'] * 100:.0f} % and {chk[('base', 'noise_1.0', 'dump')]['share_over_20cm'] * 100:.0f} %). A wide gate "
        f"is worse: gate 2 updates on {g2b['share_est_updates'] * 100:.0f} % of the steps, spread {g2b['m_hat_tail_std_kg']:.0f} kg, "
        f"{g2b['share_over_20cm'] * 100:.0f} % misses on `baseline`. Interpretation (not tested separately): dropping "
        "samples breaks the cancellation of the differenced noise (4.1, step 1). What is left is slower noise, and "
        "the boom follows slow errors.",
        "",
    ]

    # ------------------------------------------------------------------ limits
    out += [
        "## 5. Limits",
        "",
        "- The floor is the floor of the E_mech metric of this plant: no oil compressibility, no valve "
        "dynamics, no relief flow, and absorbed work counts like delivered work. It is not a pump-energy "
        f"floor. {E_PUMP.split(' kJ ')[0]} stays a lower bound with ASSUMED parameters.",
        "- The floor uses the tip mass at the start of the move. On `fill` and `dump` the work after the 5 s step "
        "is compared with that floor; a controller that knew the step in advance could go below it on `dump`.",
        "- The split has 0.05 s resolution. A reversal inside one control step is attributed by net travel.",
        "- Tuning used the nominal plant and Ki = 0 only. The points were not retuned per variant or per noise level.",
        f"- 11c parameters were chosen with {int(fj['noise_seeds'])} noise seeds on 20 scenarios: 1 % resolution on a "
        "noisy cell. The unseen evaluation uses the single noise stream of Lab 10.",
        "- The detector and the gate were tested in the forms given in `rl/11_real_plant_tuning.py` only. Other "
        "designs of the same ideas may behave differently.",
        "- All geometry is ESTIMATED from a drawing.",
        "",
    ]

    # ------------------------------------------------------------------ what is left
    out += [
        "## What is left for a learned controller",
        "",
        f"### Gap between the best PID and the energy floor at the three move times",
        "",
        f"{n_new} unseen scenarios, kJ per move, all values ESTIMATED. The best PID at each move time is the "
        "Pareto point of section 2. First block: nominal plant, `baseline`. Second block: mean over all 72 rows, "
        f"with the Lab 10 estimator and with the recommended fix (`{rec}`).",
        "",
    ]
    lg = []
    for c in ROLES:
        r = idx[(c, "base", "none", "baseline")]
        lg.append([c, gtxt[c], f"{r['mean_t_reach_s']:.2f}", f"{r['mean_floor_kJ']:.1f}", f"{r['mean_E_mech_kJ']:.1f}",
                   f"{r['mean_E_mech_kJ'] - r['mean_floor_kJ']:+.1f}", f"{r['pct_above_floor']:+.1f}",
                   f"{r['mean_overshoot_kJ']:.1f}", f"{r['mean_oscillation_kJ']:.1f}", f"{r['mean_direct_excess_kJ']:+.1f}",
                   f"{-r['mean_shortfall_kJ']:+.1f}"])
    out += table(["point", "gains", "move time s (reach 20 cm)", "floor", "E_mech", "gap kJ", "gap %", "overshoot",
                  "oscillation", "direct excess", "shortfall"], lg, ["---", "---"] + ["---:"] * 9)
    out.append("")
    lg = []
    for c in ROLES:
        for e in dict.fromkeys(["base", rec]):
            rs = rows_of(c, e)
            lg.append([c, fix_label[e], span([r["mean_t_reach_s"] for r in rs], 2), f"{mean(rs, 'mean_floor_kJ'):.1f}",
                       f"{mean(rs, 'mean_E_mech_kJ'):.1f}", f"{mean(rs, 'mean_E_mech_kJ') - mean(rs, 'mean_floor_kJ'):+.1f}",
                       f"{100 * (mean(rs, 'mean_E_mech_kJ') / mean(rs, 'mean_floor_kJ') - 1):+.1f}",
                       span([r["pct_above_floor"] for r in rs]), f"{mean(rs, 'mean_overshoot_kJ'):.1f}",
                       f"{mean(rs, 'mean_oscillation_kJ'):.1f}", f"{mean(rs, 'mean_direct_excess_kJ'):+.1f}",
                       f"{-mean(rs, 'mean_shortfall_kJ'):+.1f}"])
    out += table(["point", "estimator", "move time s (72 rows)", "floor", "E_mech", "gap kJ", "gap %", "gap % per row",
                  "overshoot", "oscillation", "direct excess", "shortfall"], lg, ["---", "---"] + ["---:"] * 10)
    nom = {c: idx[(c, "base", "none", "baseline")] for c in ROLES}
    n1 = {c: [r for r in rows_of(c, rec) if r["sigma_th_deg"] == 1.0] for c in ROLES}
    n0 = {c: [r for r in rows_of(c, rec) if r["sigma_th_deg"] == 0.0] for c in ROLES}
    out += [
        "",
        "Reading:",
        "",
        f"- **Nominal plant.** At {nom['economical']['mean_t_reach_s']:.1f} s the PID is on the floor "
        f"({nom['economical']['pct_above_floor']:+.1f} %): nothing is left. At {nom['balanced']['mean_t_reach_s']:.1f} s "
        f"{nom['balanced']['mean_E_mech_kJ'] - nom['balanced']['mean_floor_kJ']:.1f} kJ "
        f"({nom['balanced']['pct_above_floor']:+.1f} %) are left, at {nom['fast']['mean_t_reach_s']:.1f} s "
        f"{nom['fast']['mean_E_mech_kJ'] - nom['fast']['mean_floor_kJ']:.1f} kJ ({nom['fast']['pct_above_floor']:+.1f} %). "
        "The fast gap is direct excess: powered lowering and reverse-torque braking. The floor does not depend on "
        "the move time while gravity alone can start and stop the boom (section 1.1), so this gap is open to a "
        "controller that lets gravity start the lowering move and stop the raise. A linear PD with gravity "
        "feed-forward cannot, because its command follows the error. The gap is an upper bound of the gain: "
        "the minimum-energy move at 1.2 s was not computed.",
        "- **Under sensor noise the gap is larger at every move time.** With the recommended fix, mean gap over "
        "the rows without sensor noise against the rows with 1.0 deg: "
        + "; ".join(f"`{c}` {100 * (mean(n0[c], 'mean_E_mech_kJ') / mean(n0[c], 'mean_floor_kJ') - 1):+.1f} % against "
                    f"{100 * (mean(n1[c], 'mean_E_mech_kJ') / mean(n1[c], 'mean_floor_kJ') - 1):+.1f} %" for c in ROLES)
        + ". That part is oscillation: the derivative term and the mass estimate pass the noise to the cylinder.",
        "- **Payload steps.** On `fill` and `fill_half` the excess after the 5 s step is overshoot and oscillation "
        "(section 1.3). A controller cannot know the step in advance; what is left is a faster and quieter "
        "estimate, not a better move.",
        "",
        "### Rows that still break after the 11c fixes",
        "",
        f"{n_new} unseen scenarios, 72 rows per point, break = miss rate over 5 %, all values ESTIMATED.",
        "",
    ]
    lb = []
    for c in ROLES:
        for e in est_names:
            b = breaks(c, e)
            lb.append([c, fix_label[e] + (" **(recommended)**" if e == rec else ""), str(len(b)),
                       tally(rows_of(c, e)), "; ".join(b) or "none", "; ".join(hot(c, e)) or "none"])
    out += table(["point", "estimator", "rows > 5 %", "status over 72 rows", "rows that break (miss %)",
                  f"rows with E_mech more than {HOT_PCT:.0f} % above the floor"], lb,
                 ["---", "---", "---:", "---", "---", "---"])
    left = {c: breaks(c, rec) for c in ROLES}
    n_left = sum(len(v) for v in left.values())
    deg = {c: [f"{n} {v} ({idx[(c, rec, n, v)]['share_over_20cm'] * 100:.0f} %)" for n, v in keys72
               if 0.01 < idx[(c, rec, n, v)]["share_over_20cm"] <= 0.05] for c in ROLES}
    out += [
        "",
        f"- **With the recommended fix (`{rec}`)**: "
        + (f"{n_left} rows still break: " + "; ".join(f"`{c}`: {', '.join(v)}" for c, v in left.items() if v) + "."
           if n_left else "no row breaks on any of the three points.")
        + " Degraded rows (1-5 %): "
        + ("; ".join(f"`{c}`: {', '.join(v)}" for c, v in deg.items() if v) or "none") + "."
        + f" Rows with E_mech more than {HOT_PCT:.0f} % above the floor, which the miss rate does not count: "
        + ("; ".join(f"`{c}`: {', '.join(hot(c, rec))}" for c in ROLES if hot(c, rec)) or "none") + ".",
        "- The clamp, the gate and the combination of all three break rows that the Lab 10 estimator does not "
        "(table above, `balanced` and `economical`). They are not part of the recommended fix. On `fast` no "
        "estimator breaks a row.",
        "- Not covered by any row here, so open for any controller: oil compressibility and valve dynamics "
        "(section 5), a mass step during the move instead of at rest, and noise levels above 1.0 deg.",
    ]
    with open(M.MD_PATH, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {M.MD_PATH} ({len(out)} lines)")


if __name__ == "__main__":
    build()
