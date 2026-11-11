"""Lab 09f: retune Kp of the sched controller (pid_gain_sched of Lab 09c3). No RL, no training.

Lab 09e: sched (Kp = 0.5) misses 1-10 % of the moves on the combined cases because long moves
are still converging after 10 s, and the residual SAC of 09d does not repair that. This lab
raises Kp instead.

Tuning (20 original scenarios, RNG seed 7; the 100 unseen scenarios are not used):
  Kp grid 0.50, 0.55, ... 1.00. Ki and Kd stay at the Lab 07 values (rl/09c3_ff_torque_limit.csv).
  Every row of the two 09c3 tables (ten single conditions, nine combined cases, variants
  baseline and payload_x1.2), one Euler step per 0.05 s.
  Rule: the Lab 07 criterion, applied to these 38 rows instead of the nominal plant: among the
  Kp with no move over 20 cm, the lowest mean E_mech. If every Kp has a miss: fewest misses,
  then lowest mean E_mech.
Evaluation (100 unseen scenarios, RNG seed 123, as 09c2/09c3/09e): every Kp of the grid on the
same 38 rows, plus the four J x0.38 rows with 10 substeps (group substep_check). Kp = 0.5 must
equal the pid_gain_sched rows of rl/09c3_ff_torque_limit.csv (asserted). Only the tuned Kp is
checked against the goal; the other Kp are reported as a sweep.
E_pump as in 09b (dp_valve and eta_pump ASSUMED - not from manual, rl/09b_hyd_params.py).

Goal, as Lab 09e, against sched with Kp = 0.5, per row:
  1. the 18 combined rows: miss rate <= 3 %
  2. the 18 combined rows: E_mech <= 1.10 x sched and E_mech < 100 kJ
  3. the 20 single rows: miss rate <= sched and E_mech <= 1.10 x sched

Usage: python rl/09f_kp_retune.py [--dry-run] [--report-only]
  --dry-run       Kp 0.5 and 0.6, 3 original and 5 unseen scenarios; prints the report, writes
                  no file
  --report-only   rebuild rl/09f_kp_retune.md from the existing CSV, no simulation
"""
import csv
import multiprocessing as mp
import os
import sys
import time
import numpy as np
from lab09_common import HYD, NEW, ORIG, cm
from lab09d_common import COMBINED, GAINS, SCHED_CSV, SINGLES, T_F_S, VARIANT_NAMES, Sched, run_case

CSV_PATH = "rl/09f_kp_retune.csv"
MD_PATH = "rl/09f_kp_retune.md"
REF_09E = "rl/09e_comparison.csv"           # read only: residual SAC and 08b SAC columns
DRY_RUN = "--dry-run" in sys.argv
REPORT_ONLY = "--report-only" in sys.argv
KP_GRID = [0.5, 0.6] if DRY_RUN else [round(0.5 + 0.05 * i, 2) for i in range(11)]
SUBSTEP = [(name + "_sub10", {**cond, "substeps": 10}) for name, cond in SINGLES + COMBINED
           if cond.get("j") == 0.38]
GROUPS = [("single", SINGLES), ("combined", COMBINED), ("substep_check", SUBSTEP)]
SETS = [("orig20", ORIG[:3] if DRY_RUN else ORIG, GROUPS[:2]),
        ("new100", NEW[:5] if DRY_RUN else NEW, GROUPS)]
MAX_MISS, MAX_E_RATIO, MAX_E_KJ = 3.0, 1.10, 100.0
LABELS_09E = {"sched_residual": "sched + residual (09e)", "sac08b_w0.3": "SAC 08b w=0.3 (09e)"}


def run_job(kp):
    """All rows of one Kp on both scenario sets. Runs in a worker process."""
    t0 = time.time()
    gains = (kp, GAINS[1], GAINS[2])

    def make(tau_noise):
        return Sched(gains=gains, tau_noise=tau_noise)

    rows = []
    for set_name, scen, groups in SETS:
        for group, cases in groups:
            for name, cond in cases:
                for variant in VARIANT_NAMES:
                    row = {"set": set_name, "n_scen": len(scen), "group": group, "condition": name,
                           "sigma_th_deg": cond.get("sigma", 0.0), "tau_noise": cond.get("tau_noise", 0.0),
                           "J_factor": cond.get("j", 1.0), "b_factor": cond.get("b", 1.0),
                           "tau_d": int(cond.get("tau_d", False)), "substeps": cond.get("substeps", 1),
                           "variant": variant, "controller": "sched", "kp": kp, "ki": gains[1],
                           "kd": gains[2], "t_f_s": T_F_S}
                    row.update(run_case(make, cond, variant, scen))
                    rows.append(row)
    return rows, time.time() - t0


def parse(r):
    out = {}
    for k, v in r.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def simulate():
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(len(KP_GRID), os.cpu_count() or 1)) as pool:
        results = pool.map(run_job, KP_GRID)
    rows = [r for rs, _ in results for r in rs]
    per_kp = sum(len(scen) * sum(len(cases) for _, cases in groups) * len(VARIANT_NAMES)
                 for _, scen, groups in SETS)
    print(f"{len(KP_GRID)} Kp x {per_kp} episodes in {time.time() - t0:.0f} s"
          f" (slowest worker {max(t for _, t in results):.0f} s)")

    # Kp = 0.5 on the unseen scenarios must equal the pid_gain_sched rows of 09c3
    if not DRY_RUN:
        with open(SCHED_CSV) as f:
            ref = {(r["condition"], r["variant"]): r for r in csv.DictReader(f) if r["w"] == "pid_gain_sched"}
        keys = ("mean_deg", "max_deg", "mean_kJ", "mean_pump_kJ", "share_over_20cm",
                "m_hat_over_m_mean", "m_hat_over_m_std")
        n = 0
        for r in rows:
            if r["set"] == "new100" and r["kp"] == GAINS[0] and r["group"] != "substep_check":
                for key in keys:
                    want = float(ref[(r["condition"], r["variant"])][key])
                    assert abs(r[key] - want) <= 1e-9 * max(1.0, abs(want)), f"{r['condition']} {key}"
                n += 1
        assert n == 2 * len(SINGLES + COMBINED), n
        print(f"Kp={GAINS[0]:g}: {n} rows match {SCHED_CSV} (pid_gain_sched)")
    return rows


def report(rows, write=True):
    idx = {(r["set"], round(r["kp"], 2), r["condition"], r["variant"]): r for r in rows}
    grid = sorted({round(r["kp"], 2) for r in rows})
    kp0 = round(GAINS[0], 2)
    main_cases = SINGLES + COMBINED
    n_orig = int(idx[("orig20", kp0, "none", "baseline")]["n_scen"])
    n_new = int(idx[("new100", kp0, "none", "baseline")]["n_scen"])

    def val(set_name, kp, name, variant, key, scale=1.0):
        return idx[(set_name, kp, name, variant)][key] * scale

    # --- tuning on the original scenarios ---
    tune = {}
    for kp in grid:
        rs = [idx[("orig20", kp, name, v)] for name, _ in main_cases for v in VARIANT_NAMES]
        tune[kp] = {"misses": sum(int(round(r["share_over_20cm"] * r["n_scen"])) for r in rs),
                    "moves": sum(int(r["n_scen"]) for r in rs),
                    "worst_cm": cm(max(r["max_deg"] for r in rs)),
                    "kJ": float(np.mean([r["mean_kJ"] for r in rs]))}
    best = min(grid, key=lambda k: (tune[k]["misses"], tune[k]["kJ"]))
    clean = [k for k in grid if tune[k]["misses"] == 0]

    # --- goal check on the unseen scenarios, against Kp = 0.5 ---
    def check(cases, kp, miss_limit):
        out = []
        for name, _ in cases:
            for variant in VARIANT_NAMES:
                miss = val("new100", kp, name, variant, "share_over_20cm", 100)
                miss_s = val("new100", kp0, name, variant, "share_over_20cm", 100)
                e, e_s = val("new100", kp, name, variant, "mean_kJ"), val("new100", kp0, name, variant, "mean_kJ")
                limit = miss_s if miss_limit is None else miss_limit
                out.append({"row": f"{name} {variant}", "miss": miss, "miss_sched": miss_s, "e": e,
                            "e_sched": e_s, "ratio": e / e_s, "miss_ok": miss <= limit + 1e-9,
                            "ratio_ok": e / e_s <= MAX_E_RATIO, "abs_ok": e < MAX_E_KJ})
        return out

    def span(vals, digits=1, unit=""):
        return f"{min(vals):.{digits}f}-{max(vals):.{digits}f}{unit}"

    def goal(kp):
        comb, sing = check(COMBINED, kp, MAX_MISS), check(SINGLES, kp, None)
        return (all(r["miss_ok"] for r in comb), all(r["ratio_ok"] and r["abs_ok"] for r in comb),
                all(r["miss_ok"] and r["ratio_ok"] for r in sing), comb, sing)

    g1, g2, g3, comb, sing = goal(best)
    w_miss = max(comb, key=lambda r: r["miss"])
    w_ratio, w_e = max(comb, key=lambda r: r["ratio"]), max(comb, key=lambda r: r["e"])
    s_miss, s_e = [r for r in sing if not r["miss_ok"]], [r for r in sing if not r["ratio_ok"]]
    goal_lines = [
        f"1. **Combined rows, miss rate <= {MAX_MISS:.0f} %: {'met' if g1 else 'NOT met'}.** "
        f"{sum(r['miss_ok'] for r in comb)} of {len(comb)} rows pass. Range "
        f"{span([r['miss'] for r in comb], 0, '%')} (Kp={kp0:g}: {span([r['miss_sched'] for r in comb], 0, '%')}). "
        f"Worst row: {w_miss['row']}, {w_miss['miss']:.0f}% (Kp={kp0:g}: {w_miss['miss_sched']:.0f}%).",
        f"2. **Combined rows, E_mech <= {MAX_E_RATIO:.2f} x sched and < {MAX_E_KJ:.0f} kJ: "
        f"{'met' if g2 else 'NOT met'}.** {sum(r['ratio_ok'] for r in comb)} of {len(comb)} rows within "
        f"the ratio, {sum(r['abs_ok'] for r in comb)} of {len(comb)} under {MAX_E_KJ:.0f} kJ. "
        f"E_mech {span([r['e'] for r in comb])} kJ (Kp={kp0:g}: {span([r['e_sched'] for r in comb])} kJ), "
        f"ratio {span([r['ratio'] for r in comb], 3)}. Highest ratio: {w_ratio['row']}, "
        f"{w_ratio['ratio']:.3f}. Highest E_mech: {w_e['row']}, {w_e['e']:.1f} kJ.",
        f"3. **Single rows, no loss (miss <= sched, E_mech <= {MAX_E_RATIO:.2f} x sched): "
        f"{'met' if g3 else 'NOT met'}.** Miss above Kp={kp0:g} on {len(s_miss)} of {len(sing)} rows"
        + (": " + "; ".join(f"{r['row']} {r['miss']:.0f}% vs {r['miss_sched']:.0f}%" for r in s_miss)
           if s_miss else "")
        + f". Miss range {span([r['miss'] for r in sing], 0, '%')} "
        f"(Kp={kp0:g}: {span([r['miss_sched'] for r in sing], 0, '%')}). "
        f"E_mech above the ratio on {len(s_e)} of {len(sing)} rows"
        + (": " + "; ".join(f"{r['row']} {r['e']:.1f} vs {r['e_sched']:.1f} kJ ({r['ratio']:.3f})" for r in s_e)
           if s_e else "")
        + f". Ratio {span([r['ratio'] for r in sing], 3)}.",
    ]

    tune_table = [f"| Kp | moves over 20 cm (of {tune[kp0]['moves']}) | worst error cm | mean E_mech kJ | |",
                  "|---:|---:|---:|---:|---|"]
    for kp in grid:
        t = tune[kp]
        tune_table.append(f"| {kp:.2f} | {t['misses']} | {t['worst_cm']:.1f} | {t['kJ']:.1f} |"
                          f" {'**tuned**' if kp == best else 'Lab 07 / 09c3' if kp == kp0 else ''} |")

    sweep = ["| Kp | combined: miss % | rows <= 3 % | E_mech kJ | ratio to Kp=0.5 | rows within ratio and"
             " < 100 kJ | single: miss % | rows with miss above Kp=0.5 | ratio to Kp=0.5 | rows within"
             " ratio | goal |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for kp in grid:
        a, b, c, cb, sg = goal(kp)
        sweep.append(
            f"| {kp:.2f}{' (tuned)' if kp == best else ''} | {span([r['miss'] for r in cb], 0)} |"
            f" {sum(r['miss_ok'] for r in cb)}/{len(cb)} | {span([r['e'] for r in cb])} |"
            f" {span([r['ratio'] for r in cb], 3)} | {sum(r['ratio_ok'] and r['abs_ok'] for r in cb)}/{len(cb)} |"
            f" {span([r['miss'] for r in sg], 0)} | {sum(not r['miss_ok'] for r in sg)}/{len(sg)} |"
            f" {span([r['ratio'] for r in sg], 3)} | {sum(r['ratio_ok'] for r in sg)}/{len(sg)} |"
            f" {'reference' if kp == kp0 else 'met' if a and b and c else 'NOT met'} |")

    # --- comparison table, with the 09e controllers when rl/09e_comparison.csv is there ---
    ref09e = {}
    if os.path.exists(REF_09E):
        with open(REF_09E) as f:
            for r in map(parse, csv.DictReader(f)):
                ref09e.setdefault((r["controller"], r["condition"], r["variant"]), []).append(r)
    extra = [c for c in LABELS_09E if any(k[0] == c for k in ref09e)]
    names = [f"sched Kp={kp0:g}", f"sched Kp={best:g}"] + [LABELS_09E[c] for c in extra]

    def table(cases):
        heads = " | ".join(names)
        out = [f"| condition | variant | miss %: {heads} | E_mech kJ: {heads} | E_pump kJ: {heads} |",
               "|---|---|" + "---:|" * (3 * len(names))]
        for name, _ in cases:
            for variant in VARIANT_NAMES:
                cols = []
                for key, scale, digits in (("share_over_20cm", 100, 0), ("mean_kJ", 1, 1), ("mean_pump_kJ", 1, 1)):
                    cells = [f"{val('new100', kp, name, variant, key, scale):.{digits}f}" for kp in (kp0, best)]
                    for c in extra:
                        v = np.array([r[key] for r in ref09e[(c, name, variant)]]) * scale
                        cells.append(f"{v.mean():.1f} ± {v.std():.1f}")
                    cols.append(" | ".join(cells))
                out.append(f"| {name} | {variant} | " + " | ".join(cols) + " |")
        return out

    met = g1 and g2 and g3
    lines = [
        "# Lab 09f: Kp retune of the sched controller",
        "",
        f"No RL and no training. Controller: `pid_gain_sched` of Lab 09c3 (payload-aware PID with "
        f"gain scheduling, T_f = {T_F_S} s), Ki = {GAINS[1]:g} and Kd = {GAINS[2]:g} as before, only Kp "
        f"changed. All numbers are from `{CSV_PATH}`. One Euler step per 0.05 s, 10 s per move. Miss % = "
        "share of moves with final tip error over 20 cm.",
        f"E_pump uses dp_valve = {HYD.DP_VALVE_PA / 1e5:.0f} bar and eta_pump = {HYD.ETA_PUMP}, both "
        "ASSUMED - not from manual.",
        "",
        "## Goal",
        "",
        f"**The goal is {'met' if met else 'NOT met'}** with Kp = {best:g} (tuned on the {n_orig} original "
        f"scenarios, checked on the {n_new} unseen scenarios, against sched with Kp = {kp0:g}, per row).",
        "",
        *goal_lines,
        "",
        f"## Tuning on the {n_orig} original scenarios",
        "",
        f"Scenarios of RNG seed 7 (Labs 03-07), the {len(main_cases)} conditions of the 09c3 tables x "
        f"{len(VARIANT_NAMES)} variants = {2 * len(main_cases)} rows. Rule (Lab 07 criterion on these rows): "
        "lowest mean E_mech among the Kp with no move over 20 cm; if every Kp has a miss, fewest misses, "
        "then lowest mean E_mech. "
        + (f"Kp without a miss: {', '.join(f'{k:g}' for k in clean)}. " if clean else "No Kp is free of misses. ")
        + f"**Tuned Kp = {best:g}.**",
        "",
        *tune_table,
        "",
        "Limits of this tuning:",
        f"- Before this lab, Kp = 0.6, 0.7 and 0.9 were tried on the {n_new} unseen scenarios in a scratch "
        "check (Lab 09e diagnosis). The rule above does not read those scenarios, but the choice of grid "
        "and the decision to retune Kp came after seeing them.",
        "- The tuning conditions are the evaluation conditions (same noise levels, J and b factors, tau_d, "
        "payload step); only the moves differ. This is not a test on unseen plant parameters.",
        "- Kd and the filter time were not retuned.",
        "",
        f"## Kp sweep on the {n_new} unseen scenarios",
        "",
        f"Not used for the tuning. Goal columns as above, each Kp against Kp = {kp0:g}.",
        "",
        *sweep,
        "",
        "## Comparison on every 09c3 row",
        "",
        f"{n_new} unseen scenarios (RNG seed 123)."
        + (f" The 09e columns are read from `{REF_09E}` (mean ± std over three seeds)." if extra else ""),
        "",
        *table(main_cases),
        "",
        "## J x0.38 rows with 10 substeps",
        "",
        "Same rows with the dynamics integrated in 10 steps of 0.005 s per command (integrator check as "
        "README Lab 08c). Not part of the goal.",
        "",
        "| condition | variant | miss %: " + f"Kp={kp0:g} | Kp={best:g} | E_mech kJ: Kp={kp0:g} | Kp={best:g} |",
        "|---|---|---:|---:|---:|---:|",
        *[f"| {name} | {v} | {val('new100', kp0, name, v, 'share_over_20cm', 100):.0f} |"
          f" {val('new100', best, name, v, 'share_over_20cm', 100):.0f} |"
          f" {val('new100', kp0, name, v, 'mean_kJ'):.1f} | {val('new100', best, name, v, 'mean_kJ'):.1f} |"
          for name, _ in SUBSTEP for v in VARIANT_NAMES],
    ]
    print("\n" + "\n".join(lines[2:]))
    if write:
        with open(MD_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nCSV: {CSV_PATH}\nReport: {MD_PATH}")


if __name__ == "__main__":
    if DRY_RUN:
        report(simulate(), write=False)
        print(f"\nDRY RUN: Kp {KP_GRID}, {len(SETS[0][1])} original and {len(SETS[1][1])} unseen scenarios."
              " No file written.")
    else:
        if not REPORT_ONLY:
            rows = simulate()
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        with open(CSV_PATH) as f:
            report([parse(r) for r in csv.DictReader(f)])
