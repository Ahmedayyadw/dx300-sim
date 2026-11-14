"""Lab 12g: does the PID-vs-SAC ranking survive errors in the estimated plant? No training.

The real plant's geometry and friction are ESTIMATED from a drawing. This lab perturbs the plant
(not the controllers, which keep the nominal model) and checks, for every perturbation, whether
the economical PID still dominates every residual SAC model of Lab 12e, i.e. has no more misses
(tip error over 20 cm) AND no more mechanical energy than that SAC model.

Perturbed plant factors (BoomEnvReal knobs):
  r  moment arm r(theta): torque limits, friction torque and the flow speed limit
  g  gravity torque (CoM position and mass lumping)
  j  inertia J
  b  cylinder friction (Coulomb and viscous)
Grid (33 plant settings):
  nominal                          1
  one at a time, x 0.8/0.9/1.1/1.2  16  (each factor alone, the others at 1)
  joint corners, x 0.8 or 1.2      16  (full factorial over r, g, j, b)
Rows per setting: the 4 payload variants (baseline, fill_half, fill, dump) x 2 sensor cases
(clean; theta noise 0.5 deg + 5 % torque noise), 100 unseen scenarios (new100), 10 substeps,
same rollout code as rl/12_eval.py and rl/12e_stats_eval.py.
Controllers: economical PID; residual SAC on the balanced and economical points, w 0.03 and 0.1,
3 seeds each (the Lab 12e models).

The report also re-reads rl/12e_moves.csv.gz (if present) for the 18 conditions of the Lab 12
protocol, which already include J x 0.38/1.5 and friction x 0.5/2, alone and combined with noise.

Usage (repo root, .venv active):
  python rl/12g_sensitivity_eval.py           evaluate (about 1-1.5 h on 16 cores) + report
  python rl/12g_sensitivity_eval.py --report  report only, from rl/12g_moves.csv.gz
  python rl/12g_sensitivity_eval.py --smoke   2 settings, SAC residual 0, no models needed
Writes rl/12g_moves.csv.gz (per move, not committed), rl/12g_sensitivity.csv (per setting and
controller) and rl/12g_sensitivity.md.
"""

import collections
import csv
import gzip
import importlib.util
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("lab12_eval", os.path.join(HERE, "12_eval.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
E, L10, L11 = M.E, M.L10, M.L11

REF = "economical"
SAC = {  # label: (model prefix, base point, w), as rl/12e_stats_eval.py
    "res_bal_w0.03": ("12", "balanced", 0.03),
    "res_bal_w0.1": ("12", "balanced", 0.1),
    "res_eco_w0.03": ("12eco", "economical", 0.03),
    "res_eco_w0.1": ("12eco", "economical", 0.1),
}
FACTORS = ("r", "g", "j", "b")
OAT_LEVELS = (0.8, 0.9, 1.1, 1.2)
CORNER_LEVELS = (0.8, 1.2)
SENSORS = {"clean": {}, "noisy": {"sigma": 0.5, "tau_noise": 0.05}}
MISS_CM = 20.0
MOVES = "rl/12g_moves.csv.gz"
CSV_PATH = "rl/12g_sensitivity.csv"
MD_PATH = "rl/12g_sensitivity.md"
MOVES_12E = "rl/12e_moves.csv.gz"  # read only
FIELDS = (
    "controller",
    "seed",
    "setting",
    "sensor",
    "variant",
    "scen",
    "err_cm",
    "E_mech_kJ",
    "E_pump_kJ",
)


def settings():
    """[(name, kind, {r, g, j, b})], nominal first."""
    one = dict.fromkeys(FACTORS, 1.0)
    out = [("nominal", "nominal", one)]
    for f in FACTORS:
        for k in OAT_LEVELS:
            out.append((f"{f}x{k}", "one at a time", {**one, f: k}))
    for ks in itertools.product(CORNER_LEVELS, repeat=len(FACTORS)):
        name = "_".join(f"{f}{k}" for f, k in zip(FACTORS, ks))
        out.append((name, "corner", dict(zip(FACTORS, ks))))
    return out


SETTINGS = settings()
SETTING = {name: p for name, _, p in SETTINGS}
KIND = {name: kind for name, kind, _ in SETTINGS}


def model_path(label, seed):
    prefix, _, w = SAC[label]
    return f"{M.MODEL_DIR}/{prefix}_residual_w{w}_s{seed}.zip"


def make_env(variant, p, sensor):
    cond = SENSORS[sensor]
    env = L10.make_env(variant, sigma=cond.get("sigma", 0.0), j=p["j"], b=p["b"])
    env.r_scale, env.g_scale = p["r"], p["g"]
    return env


def make_ctrl(label, seed, tau_noise, smoke=False):
    if label == REF:
        return L11.Sched11(E.point_gains(REF), E.EST, tau_noise=tau_noise)
    model = None if smoke else M.load_model(model_path(label, seed))
    return M.SacCtrl(model, True, gains=E.point_gains(SAC[label][1]), tau_noise=tau_noise)


def run_row(job):
    """One controller (and seed) on one setting, sensor case and variant. Worker."""
    label, seed, set_name, setting, sensor, variant, smoke = job
    env = make_env(variant, SETTING[setting], sensor)
    ctrl = make_ctrl(label, seed, SENSORS[sensor].get("tau_noise", 0.0), smoke)
    out = []
    for i, sc in enumerate(L11.scenarios(set_name)):
        r = L11.rollout(ctrl, env, *sc)
        out.append(
            {
                "controller": label,
                "seed": "" if seed is None else seed,
                "setting": setting,
                "sensor": sensor,
                "variant": variant,
                "scen": i,
                "err_cm": float(M.cm(r["err_deg"])),
                "E_mech_kJ": r["E_mech_kJ"],
                "E_pump_kJ": r["E_pump_kJ"],
            }
        )
    return out


def jobs(names=None, set_name=M.SET, smoke=False):
    names = names or [n for n, _, _ in SETTINGS]
    rows = [(n, s, v) for n in names for s in SENSORS for v in L10.VARIANTS]
    out = [(REF, None, set_name, n, s, v, smoke) for n, s, v in rows]
    out += [
        (lab, sd, set_name, n, s, v, smoke) for lab in SAC for sd in M.SEEDS for n, s, v in rows
    ]
    return out


# ----------------------------------------------------------------------------- report
def summarise(moves, key="setting"):
    """(group, controller) -> per-seed list of (miss %, mean cm, E_mech, E_pump), and the
    seed-mean per-move arrays (err, E_mech) sorted by move for the paired test."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in moves:
        by[(m[key], m["controller"])][m["seed"]].append(m)
    per_seed, paired = {}, {}
    for gc, seeds in by.items():
        stats, errs, ens = [], [], []
        for _, ms in sorted(seeds.items()):
            ms.sort(
                key=lambda m: tuple(m[k] for k in ("sensor", "condition", "variant") if k in m)
                + (int(m["scen"]),)
            )
            err = np.array([float(m["err_cm"]) for m in ms])
            en = np.array([float(m["E_mech_kJ"]) for m in ms])
            ep = np.array([float(m["E_pump_kJ"]) for m in ms])
            stats.append((100 * (err > MISS_CM).mean(), err.mean(), en.mean(), ep.mean()))
            errs.append(err)
            ens.append(en)
        per_seed[gc] = np.array(stats)
        paired[gc] = (np.mean(errs, axis=0), np.mean(ens, axis=0))
    return per_seed, paired


def dominance(per_seed, paired, groups, labels):
    """Per group: reference stats, per-SAC stats, dominated flags, worst Wilcoxon p."""
    from scipy.stats import wilcoxon

    out = []
    for g in groups:
        ref = per_seed[(g, REF)].mean(axis=0)
        e_ref, en_ref = paired[(g, REF)]
        row = {"group": g, "ref": ref, "sac": {}, "dominated": {}, "p": {}}
        for lab in labels:
            if (g, lab) not in per_seed:
                continue
            s = per_seed[(g, lab)].mean(axis=0)
            row["sac"][lab] = s
            row["dominated"][lab] = bool(ref[0] <= s[0] and ref[2] <= s[2])
            e, en = paired[(g, lab)]
            pe = wilcoxon(e, e_ref).pvalue if np.any(e - e_ref) else 1.0
            pn = wilcoxon(en, en_ref).pvalue if np.any(en - en_ref) else 1.0
            row["p"][lab] = (pe, pn)
        out.append(row)
    return out


def report(moves):
    seen = {m["controller"] for m in moves}
    labels = [lab for lab in SAC if lab in seen]
    per_seed, paired = summarise(moves)
    present = {m["setting"] for m in moves}
    groups = [n for n, _, _ in SETTINGS if n in present]
    rows = dominance(per_seed, paired, groups, labels)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "setting",
                "kind",
                *FACTORS,
                "controller",
                "miss_pct",
                "mean_cm",
                "E_mech_kJ",
                "E_pump_kJ",
                "dominated_by_pid",
                "p_error",
                "p_energy",
            ]
        )
        for r in rows:
            p = SETTING[r["group"]]
            base = [r["group"], KIND[r["group"]], *(p[f] for f in FACTORS)]
            w.writerow(base + [REF, *(f"{x:.4f}" for x in r["ref"]), "", "", ""])
            for lab, s in r["sac"].items():
                pe, pn = r["p"][lab]
                w.writerow(
                    base
                    + [
                        lab,
                        *(f"{x:.4f}" for x in s),
                        int(r["dominated"][lab]),
                        f"{pe:.3e}",
                        f"{pn:.3e}",
                    ]
                )

    n_dom = sum(all(r["dominated"].values()) for r in rows)
    fails = [(r["group"], lab) for r in rows for lab, d in r["dominated"].items() if not d]
    p_max = max((max(pe, pn) for r in rows for pe, pn in r["p"].values()), default=float("nan"))
    lines = [
        "# Lab 12g: sensitivity of the PID-vs-SAC ranking to the estimated plant",
        "",
        "Generated by `python rl/12g_sensitivity_eval.py`. No training. The plant is perturbed;",
        "the controllers keep the nominal model. Factors: r = moment arm r(theta) (torque",
        "limits, friction torque, flow speed limit), g = gravity torque (CoM and mass lumping),",
        "j = inertia, b = cylinder friction. Grid: nominal, each factor alone at x 0.8/0.9/1.1/1.2",
        "(one at a time), and all 16 joint corners at x 0.8 or 1.2 (full factorial). Per setting:",
        "4 payload",
        "variants x 2 sensor cases (clean; 0.5 deg + 5 % torque noise) x 100 unseen scenarios =",
        "800 moves per controller and seed. SAC: mean over 3 seeds. 'Dominated' = the economical",
        "PID has no more misses (> 20 cm) and no more E_mech than that SAC model. p = paired",
        "Wilcoxon against the economical PID per move (SAC = seed mean), tip error and E_mech.",
        "All values ESTIMATED.",
        "",
        "## Summary",
        "",
        f"- Economical PID dominates all {len(labels)} SAC models in {n_dom} of {len(rows)} plant"
        " settings.",
        f"- Largest Wilcoxon p over all settings, models and both metrics: {p_max:.1e}.",
    ]
    if fails:
        lines.append("- Not dominated: " + ", ".join(f"{g} ({lab})" for g, lab in fails) + ".")
    lines += [
        "",
        "## Per setting",
        "",
        "Misses % / E_mech kJ. A mark * after a SAC entry: not dominated by the PID.",
        "",
        "| setting | kind | PID economical | " + " | ".join(labels) + " |",
        "|---|---|---:|" + "---:|" * len(labels),
    ]
    for r in rows:
        cells = []
        for lab in labels:
            s = r["sac"].get(lab)
            cells.append(
                ""
                if s is None
                else f"{s[0]:.2f} / {s[2]:.1f}" + ("" if r["dominated"][lab] else " *")
            )
        lines.append(
            f"| {r['group']} | {KIND[r['group']]} | {r['ref'][0]:.2f} / "
            f"{r['ref'][2]:.1f} | " + " | ".join(cells) + " |"
        )

    if os.path.exists(MOVES_12E):
        with gzip.open(MOVES_12E, "rt") as f:
            m12 = [m for m in csv.DictReader(f) if m["controller"] in (REF, *labels)]
        ps, pa = summarise(m12, key="condition")
        conds = [n for _, cases in L10.GROUPS for n, _ in cases]
        r12 = dominance(ps, pa, conds, labels)
        n12 = sum(all(r["dominated"].values()) for r in r12)
        lines += [
            "",
            "## Lab 12 conditions (from rl/12e_moves.csv.gz)",
            "",
            f"The 18 conditions of the Lab 12 protocol, 4 payload variants each. They include"
            f" J x 0.38/1.5 and friction x 0.5/2, alone and combined with noise. The economical"
            f" PID dominates all SAC models in {n12} of {len(r12)} conditions.",
            "",
            "| condition | PID economical | " + " | ".join(labels) + " |",
            "|---|---:|" + "---:|" * len(labels),
        ]
        for r in r12:
            cells = [
                f"{s[0]:.2f} / {s[2]:.1f}" + ("" if r["dominated"][lab] else " *")
                for lab, s in r["sac"].items()
            ]
            lines.append(
                f"| {r['group']} | {r['ref'][0]:.2f} / {r['ref'][2]:.1f} | "
                + " | ".join(cells)
                + " |"
            )
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_moves(rows):
    with gzip.open(MOVES, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_moves():
    with gzip.open(MOVES, "rt") as f:
        return list(csv.DictReader(f))


def smoke():
    """Nominal setting = plain L10 env (bit for bit); one perturbed setting runs."""
    global CSV_PATH, MD_PATH
    CSV_PATH, MD_PATH = "rl/12g_smoke.csv", "rl/12g_smoke.md"
    v = "dump"
    sc = L11.scenarios("orig20")[:3]
    worst = 0.0
    for sensor, cond in SENSORS.items():
        for label in (REF, "res_eco_w0.1"):
            a = [
                L11.rollout(
                    make_ctrl(label, None, cond.get("tau_noise", 0.0), True),
                    make_env(v, SETTING["nominal"], sensor),
                    *s,
                )
                for s in sc
            ]
            b = [
                L11.rollout(
                    make_ctrl(label, None, cond.get("tau_noise", 0.0), True),
                    L10.make_env(v, **cond),
                    *s,
                )
                for s in sc
            ]
            for x, y in zip(a, b):
                worst = max(
                    worst, abs(x["err_deg"] - y["err_deg"]), abs(x["E_mech_kJ"] - y["E_mech_kJ"])
                )
    print(f"smoke: nominal setting vs Lab 12 env, largest difference {worst}")
    js = [
        j
        for j in jobs(["nominal", "r0.8_g1.2_j0.8_b1.2"], "orig20", smoke=True)
        if j[1] in (None, 0)
    ]
    res = M.pool_map(run_row, js)
    print(f"smoke: {len(js)} rows, {sum(len(r) for r in res)} moves")
    report([m for chunk in res for m in chunk])
    print("settings:", len(SETTINGS), "| jobs in a full run:", len(jobs()))
    print("model files:", model_path("res_bal_w0.03", 0), model_path("res_eco_w0.1", 2))


def main():
    if "--smoke" in sys.argv:
        smoke()
        return
    if "--report" in sys.argv:
        report(read_moves())
        return
    missing = [
        model_path(lab, s) for lab in SAC for s in M.SEEDS if not os.path.exists(model_path(lab, s))
    ]
    if missing:
        sys.exit("missing models (train them first):\n  " + "\n  ".join(missing))
    rows = [m for chunk in M.pool_map(run_row, jobs()) for m in chunk]
    write_moves(rows)
    report(read_moves())


if __name__ == "__main__":
    main()
