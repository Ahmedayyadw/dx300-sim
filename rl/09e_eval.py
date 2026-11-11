"""Lab 09e: sched alone vs sched + residual SAC (Lab 09d) vs the robust SAC of Lab 08b.

Eval only, never trains. Same 100 unseen scenarios (RNG seed 123), variants (baseline,
payload_x1.2) and metrics as 09c2/09c3, on every row of the two 09c3 tables: the ten single
conditions and the nine combined cases (J x0.38 with b x2 left out, Euler stability limit).
Controllers (rl/lab09d_common.py):
  sched            pid_gain_sched of 09c3, no seed. Its rows must equal
                   rl/09c3_ff_torque_limit.csv (asserted).
  sched_residual   sched + 0.2 x SAC residual, rl/models/09d_residual_w0.3_s{0,1,2}.zip
  sac08b_w0.3      plain robust SAC of 08b, rl/models/08b_plain_w0.3_s{0,1,2}.zip
  sac08b_w0.1      plain robust SAC of 08b, rl/models/08b_plain_w0.1_s{0,1,2}.zip
The 08b SAC does not read the torque signal, so the 5 % torque noise does not touch it.
All rows use one Euler step per 0.05 s, as 09c2/09c3. At J x0.38 the 08b SAC chatters with that
step (README, Lab 08c), so the four J x0.38 rows are run again with 10 substeps (group
substep_check) for every controller.
E_pump as in 09b (dp_valve and eta_pump ASSUMED - not from manual, rl/09b_hyd_params.py).

Goal (checked on the mean over the three seeds, per row):
  1. the 18 combined rows: miss rate <= 3 %
  2. the 18 combined rows: E_mech <= 1.10 x sched and E_mech < 100 kJ
  3. the 20 single rows: miss rate <= sched and E_mech <= 1.10 x sched

Usage: python rl/09e_eval.py [--dry-run] [--report-only]
  --dry-run       5 scenarios, seed 0, the residual model is the 09d dry-run model; prints the
                  report, writes no file
  --report-only   rebuild rl/09e_comparison.md from the existing CSV, no simulation
"""
import csv
import multiprocessing as mp
import os
import sys
import time
import numpy as np
from lab09_common import HYD, NEW
from lab09d_common import (A_MAX, COMBINED, GAINS, SCHED_CSV, SINGLES, T_F_S, VARIANT_NAMES, W_PUMP,
                           ResidualSched, SacPolicy08b, Sched, run_case)

CSV_PATH = "rl/09e_comparison.csv"
MD_PATH = "rl/09e_comparison.md"
REF_08B = "rl/08b_eval.csv"                 # read only, to check the 08b SAC rerun
MODEL_DIR = "rl/models"
DRY_RUN = "--dry-run" in sys.argv
REPORT_ONLY = "--report-only" in sys.argv
SEEDS = [0] if DRY_RUN else [0, 1, 2]
SCEN = NEW[:5] if DRY_RUN else NEW
CONTROLLERS = ["sched", "sched_residual", "sac08b_w0.3", "sac08b_w0.1"]
NAMES = {"sched": "sched", "sched_residual": "sched + residual", "sac08b_w0.3": "SAC 08b w=0.3",
         "sac08b_w0.1": "SAC 08b w=0.1"}
SUBSTEP = [(name + "_sub10", {**cond, "substeps": 10}) for name, cond in SINGLES + COMBINED
           if cond.get("j") == 0.38]
GROUPS = [("single", SINGLES), ("combined", COMBINED), ("substep_check", SUBSTEP)]
MAX_MISS, MAX_E_RATIO, MAX_E_KJ = 3.0, 1.10, 100.0


def model_path(label, seed):
    if label == "sched_residual":
        return f"{MODEL_DIR}/09d_residual_w{W_PUMP}_s{seed}{'_dry' if DRY_RUN else ''}.zip"
    return f"{MODEL_DIR}/08b_plain_w{label.split('_w')[1]}_s{seed}.zip"


def run_job(job):
    """All rows of one controller and seed. Runs in a worker process."""
    label, seed = job
    t0 = time.time()
    if label == "sched":
        def make(tau_noise):
            return Sched(tau_noise=tau_noise)
    else:
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        model = SAC.load(model_path(label, seed), device="cpu")
        if label == "sched_residual":
            def make(tau_noise):
                return ResidualSched(model, tau_noise=tau_noise)
        else:
            def make(tau_noise):
                return SacPolicy08b(model)
    rows = []
    for group, cases in GROUPS:
        for name, cond in cases:
            for variant in VARIANT_NAMES:
                row = {"group": group, "condition": name, "sigma_th_deg": cond.get("sigma", 0.0),
                       "tau_noise": cond.get("tau_noise", 0.0), "J_factor": cond.get("j", 1.0),
                       "b_factor": cond.get("b", 1.0), "tau_d": int(cond.get("tau_d", False)),
                       "substeps": cond.get("substeps", 1), "variant": variant,
                       "controller": label, "seed": seed}
                row.update(run_case(make, cond, variant, SCEN))
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
    jobs = [("sched", 0)] + [(c, s) for c in CONTROLLERS[1:] for s in SEEDS]
    missing = [model_path(c, s) for c, s in jobs if c != "sched" and not os.path.exists(model_path(c, s))]
    if missing:
        sys.exit("missing model files, nothing written:\n  " + "\n  ".join(missing))
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        results = pool.map(run_job, jobs)
    rows = [r for rs, _ in results for r in rs]
    n_rows = sum(len(cases) for _, cases in GROUPS) * len(VARIANT_NAMES)
    print(f"{len(jobs)} controller runs x {n_rows} rows x {len(SCEN)} episodes in {time.time() - t0:.0f} s"
          f" (slowest worker {max(t for _, t in results):.0f} s)")

    # sched must equal the pid_gain_sched rows of 09c3
    with open(SCHED_CSV) as f:
        ref = {(r["condition"], r["variant"]): r for r in csv.DictReader(f) if r["w"] == "pid_gain_sched"}
    keys = ("mean_deg", "max_deg", "mean_kJ", "mean_pump_kJ", "share_over_20cm",
            "m_hat_over_m_mean", "m_hat_over_m_std")
    n = 0
    for r in rows:
        if r["controller"] == "sched" and r["group"] != "substep_check" and not DRY_RUN:
            for key in keys:
                want = float(ref[(r["condition"], r["variant"])][key])
                assert abs(r[key] - want) <= 1e-9 * max(1.0, abs(want)), f"{r['condition']} {key}"
            n += 1
    if not DRY_RUN:
        print(f"sched: {n} rows match {SCHED_CSV} (pid_gain_sched)")

    # 08b SAC rerun against rl/08b_eval.csv (new100, one Euler step) on the noise-free rows
    with open(REF_08B) as f:
        ref08 = {(r["w"], int(r["seed"]), r["variant"]): r for r in csv.DictReader(f)
                 if r["controller"] == "plain" and r["set"] == "new100"}
    pairs = [("none", "baseline", "baseline"), ("none", "payload_x1.2", "payload_x1.2"),
             ("tau_d", "baseline", "tau_d"), ("J_x0.38", "baseline", "J_x0.38")]
    worst = 0.0
    for r in rows:
        for cond_name, variant, name08 in pairs:
            if (r["controller"].startswith("sac08b") and r["condition"] == cond_name
                    and r["variant"] == variant and not DRY_RUN):
                want = ref08[(r["controller"].split("_w")[1], r["seed"], name08)]
                worst = max(worst, abs(r["mean_kJ"] / float(want["mean_kJ"]) - 1),
                            abs(r["share_over_20cm"] - float(want["share_over_20cm"])))
    if not DRY_RUN:
        print(f"08b SAC rerun vs {REF_08B} (none, payload_x1.2, tau_d, J_x0.38; 6 models): largest"
              f" difference {worst:.2e} (relative E_mech or miss share)")
    assert worst < 0.02, f"08b SAC rerun differs from {REF_08B}"
    return rows


def report(rows, write=True):
    def cell(condition, variant, label, key, scale=1.0):
        vals = np.array([r[key] for r in rows if r["condition"] == condition
                         and r["variant"] == variant and r["controller"] == label]) * scale
        return vals.mean(), vals.std(), vals.max()

    def fmt(condition, variant, label, key, scale=1.0, digits=1):
        m, s, _ = cell(condition, variant, label, key, scale)
        return f"{m:.{digits}f}" if label == "sched" else f"{m:.{digits}f} ± {s:.{digits}f}"

    def table(cases):
        heads = " | ".join(NAMES[c] for c in CONTROLLERS)
        out = [f"| condition | variant | miss %: {heads} | E_mech kJ: {heads} | E_pump kJ: {heads} |",
               "|---|---|" + "---:|" * (3 * len(CONTROLLERS))]
        for name, _ in cases:
            for variant in VARIANT_NAMES:
                cols = [" | ".join(fmt(name, variant, c, key, scale) for c in CONTROLLERS)
                        for key, scale in (("share_over_20cm", 100), ("mean_kJ", 1), ("mean_pump_kJ", 1))]
                out.append(f"| {name} | {variant} | " + " | ".join(cols) + " |")
        return out

    def check(cases, label, miss_limit):
        """Per row: miss (mean over seeds) against the limit, E_mech against sched."""
        out = []
        for name, _ in cases:
            for variant in VARIANT_NAMES:
                miss, _, miss_seed = cell(name, variant, label, "share_over_20cm", 100)
                e = cell(name, variant, label, "mean_kJ")[0]
                miss_s = cell(name, variant, "sched", "share_over_20cm", 100)[0]
                e_s = cell(name, variant, "sched", "mean_kJ")[0]
                limit = miss_s if miss_limit is None else miss_limit
                out.append({"row": f"{name} {variant}", "miss": miss, "miss_seed": miss_seed,
                            "miss_sched": miss_s, "e": e, "e_sched": e_s, "ratio": e / e_s,
                            "miss_ok": miss <= limit + 1e-9, "ratio_ok": e / e_s <= MAX_E_RATIO,
                            "abs_ok": e < MAX_E_KJ})
        return out

    def span(vals, digits=1, unit=""):
        return f"{min(vals):.{digits}f}-{max(vals):.{digits}f}{unit}"

    def worst(rs, key):
        r = max(rs, key=lambda x: x[key])
        return r

    def goal_lines(label):
        comb, sing = check(COMBINED, label, MAX_MISS), check(SINGLES, label, None)
        w_miss, w_ratio, w_e = worst(comb, "miss"), worst(comb, "ratio"), worst(comb, "e")
        g1 = all(r["miss_ok"] for r in comb)
        g2 = all(r["ratio_ok"] and r["abs_ok"] for r in comb)
        g3 = all(r["miss_ok"] and r["ratio_ok"] for r in sing)
        s_miss = [r for r in sing if not r["miss_ok"]]
        s_e = [r for r in sing if not r["ratio_ok"]]
        lines = [
            f"1. **Combined rows, miss rate <= {MAX_MISS:.0f} %: {'met' if g1 else 'NOT met'}.** "
            f"{sum(r['miss_ok'] for r in comb)} of {len(comb)} rows pass. Range "
            f"{span([r['miss'] for r in comb], 1, '%')} (sched {span([r['miss_sched'] for r in comb], 0, '%')}). "
            f"Worst row: {w_miss['row']}, {w_miss['miss']:.1f}% (sched {w_miss['miss_sched']:.0f}%). "
            f"Worst single seed on any combined row: {max(r['miss_seed'] for r in comb):.0f}%.",
            f"2. **Combined rows, E_mech <= {MAX_E_RATIO:.2f} x sched and < {MAX_E_KJ:.0f} kJ: "
            f"{'met' if g2 else 'NOT met'}.** {sum(r['ratio_ok'] for r in comb)} of {len(comb)} rows within "
            f"the ratio, {sum(r['abs_ok'] for r in comb)} of {len(comb)} under {MAX_E_KJ:.0f} kJ. "
            f"E_mech {span([r['e'] for r in comb])} kJ (sched {span([r['e_sched'] for r in comb])} kJ), "
            f"ratio to sched {span([r['ratio'] for r in comb], 2)}. Highest ratio: {w_ratio['row']}, "
            f"{w_ratio['ratio']:.2f}. Highest E_mech: {w_e['row']}, {w_e['e']:.1f} kJ.",
            f"3. **Single rows, no loss (miss <= sched, E_mech <= {MAX_E_RATIO:.2f} x sched): "
            f"{'met' if g3 else 'NOT met'}.** Miss above sched on {len(s_miss)} of {len(sing)} rows"
            + (": " + "; ".join(f"{r['row']} {r['miss']:.1f}% vs {r['miss_sched']:.0f}%" for r in s_miss)
               if s_miss else "")
            + f". E_mech above the ratio on {len(s_e)} of {len(sing)} rows"
            + (": " + "; ".join(f"{r['row']} {r['e']:.1f} vs {r['e_sched']:.1f} kJ" for r in s_e)
               if s_e else "")
            + f". Ratio to sched {span([r['ratio'] for r in sing], 2)}.",
        ]
        return g1 and g2 and g3, lines

    met, goal = goal_lines("sched_residual")
    seeds = sorted({int(r["seed"]) for r in rows if r["controller"] == "sched_residual"})
    context = []
    for label in CONTROLLERS[2:]:
        comb = check(COMBINED, label, MAX_MISS)
        context.append(f"- {NAMES[label]} on the combined rows: miss {span([r['miss'] for r in comb], 1, '%')}, "
                       f"E_mech {span([r['e'] for r in comb])} kJ.")
    lines = [
        "# Lab 09e: sched vs sched + residual SAC vs robust SAC (08b)",
        "",
        f"{len(NEW)} unseen scenarios (RNG seed 123), 10 s each, one Euler step per 0.05 s. All numbers "
        f"are from `{CSV_PATH}`. SAC controllers: mean ± std over seeds {seeds}; sched has no seed.",
        f"**sched**: `pid_gain_sched` of Lab 09c3 (Kp={GAINS[0]:g} Ki={GAINS[1]:g} Kd={GAINS[2]:g}, "
        f"T_f = {T_F_S} s). **sched + residual**: u = clip(u_sched + {A_MAX} a), a from the Lab 09d SAC "
        f"(w = {W_PUMP}). **SAC 08b**: plain robust SAC of Lab 08b, w = 0.3 and w = 0.1.",
        "The residual SAC was trained on the stress conditions of this table (other scenarios); sched "
        "and the 08b SAC were not.",
        f"E_pump uses dp_valve = {HYD.DP_VALVE_PA / 1e5:.0f} bar and eta_pump = {HYD.ETA_PUMP}, both "
        "ASSUMED - not from manual. Miss % = share of moves with final tip error over 20 cm.",
        "",
        "## Goal",
        "",
        f"**The goal is {'met' if met else 'NOT met'}** (sched + residual, mean over seeds, per row).",
        "",
        *goal,
        "",
        *context,
        "",
        "## Comparison on every 09c3 row",
        "",
        *table(SINGLES + COMBINED),
        "",
        "## J x0.38 rows with 10 substeps",
        "",
        "Same rows with the dynamics integrated in 10 steps of 0.005 s per command (integrator "
        "check as README Lab 08c). Not part of the goal.",
        "",
        *table(SUBSTEP),
    ]
    print("\n" + "\n".join(lines[2:]))
    if write:
        with open(MD_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nCSV: {CSV_PATH}\nReport: {MD_PATH}")


if __name__ == "__main__":
    if DRY_RUN:
        report(simulate(), write=False)
        print(f"\nDRY RUN: {len(SCEN)} scenarios, seeds {SEEDS}, residual model"
              f" {model_path('sched_residual', 0)} (3,000 steps, not a result). No file written.")
    else:
        if not REPORT_ONLY:
            rows = simulate()
            with open(CSV_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        with open(CSV_PATH) as f:
            report([parse(r) for r in csv.DictReader(f)])
