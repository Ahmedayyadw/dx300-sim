"""Lab 12e: statistics of the real-plant comparison, and residual SAC on the economical PID.

Answers three review points with the unchanged Lab 12 protocol (rl/12_eval.py: 100 unseen
scenarios x 72 rows, 10 substeps, same rollout code):
  1. Residual SAC on the economical PID point (rl/models/12eco_residual_w{w}_s{s}.zip, trained
     with `python rl/12_robust_train.py --config residual --base economical --w W`), next to
     residual SAC on the balanced point (rl/models/12_residual_w{w}_s{s}.zip).
  2. Per-move results (rl/12e_moves.csv.gz) and a paired Wilcoxon signed-rank test of every SAC
     controller against the economical PID, on tip error and energy (same row and scenario).
  3. Miss rates at 10, 20 and 30 cm, mean/p95/worst tip error, mean +- std over seeds.
rl/12_eval.py is imported and used unchanged.

Usage (repo root, .venv active):
  python rl/12e_stats_eval.py            evaluate (about 20-25 min on 16 cores) + report
  python rl/12e_stats_eval.py --report   report only, from rl/12e_moves.csv.gz
  python rl/12e_stats_eval.py --smoke    4 rows with residual 0, no models needed
Writes rl/12e_moves.csv.gz and rl/12e_stats.md.
"""

import collections
import csv
import gzip
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location("lab12_eval", os.path.join(HERE, "12_eval.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
E, L10, L11 = M.E, M.L10, M.L11

W_LIST = (0.03, 0.1)
PIDS = ("economical", "balanced")
SAC = {  # label: (model prefix, base point, w)
    **{f"res_bal_w{w}": ("12", "balanced", w) for w in W_LIST},
    **{f"res_eco_w{w}": ("12eco", "economical", w) for w in W_LIST},
}
REF = "economical"
THRESHOLDS_CM = (10, 20, 30)
MOVES = "rl/12e_moves.csv.gz"
MD_PATH = "rl/12e_stats.md"
TRAIN_LOG = "rl/12_robust_train_log.csv"
FIELDS = (
    "controller",
    "seed",
    "condition",
    "variant",
    "scen",
    "err_cm",
    "E_mech_kJ",
    "E_pump_kJ",
    "floor_kJ",
    "t_reach_s",
)


def model_path(label, seed):
    prefix, _, w = SAC[label]
    return f"{M.MODEL_DIR}/{prefix}_residual_w{w}_s{seed}.zip"


def make_ctrl(label, seed, tau_noise):
    if label in PIDS:
        return L11.Sched11(E.point_gains(label), E.EST, tau_noise=tau_noise)
    model = None if seed is None else M.load_model(model_path(label, seed))
    return M.SacCtrl(model, True, gains=E.point_gains(SAC[label][1]), tau_noise=tau_noise)


def run_row(job):
    """One controller (and seed) on one row: per-move results. Worker."""
    label, seed, set_name, name, cond, variant = job
    env = L10.make_env(variant, **cond)
    ctrl = make_ctrl(label, seed, cond.get("tau_noise", 0.0))
    out = []
    for i, sc in enumerate(L11.scenarios(set_name)):
        r = L11.rollout(ctrl, env, *sc)
        out.append(
            {
                "controller": label,
                "seed": "" if seed is None else seed,
                "condition": name,
                "variant": variant,
                "scen": i,
                "err_cm": float(M.cm(r["err_deg"])),
                "E_mech_kJ": r["E_mech_kJ"],
                "E_pump_kJ": r["E_pump_kJ"],
                "floor_kJ": r["floor_kJ"],
                "t_reach_s": r["t_reach_s"],
            }
        )
    return out


def jobs(set_name=M.SET):
    rows = [(n, c, v) for _, n, c, v in M.ROWS72]
    out = [(p, None, set_name, n, c, v) for p in PIDS for n, c, v in rows]
    out += [(lab, s, set_name, n, c, v) for lab in SAC for s in M.SEEDS for n, c, v in rows]
    return out


def write_moves(rows):
    with gzip.open(MOVES, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_moves():
    with gzip.open(MOVES, "rt") as f:
        return list(csv.DictReader(f))


def arrays(moves):
    """controller -> seed -> (keys, err_cm, E_mech, E_pump, floor), sorted by row and scenario."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in moves:
        by[m["controller"]][m["seed"]].append(m)
    out = {}
    for ctrl, seeds in by.items():
        out[ctrl] = {}
        for seed, ms in seeds.items():
            ms.sort(key=lambda m: (m["condition"], m["variant"], int(m["scen"])))
            out[ctrl][seed] = (
                [(m["condition"], m["variant"], m["scen"]) for m in ms],
                np.array([float(m["err_cm"]) for m in ms]),
                np.array([float(m["E_mech_kJ"]) for m in ms]),
                np.array([float(m["E_pump_kJ"]) for m in ms]),
                np.array([float(m["floor_kJ"]) for m in ms]),
            )
    return out


def train_cost():
    if not os.path.exists(TRAIN_LOG):
        return {}
    t = collections.defaultdict(list)
    for r in csv.DictReader(open(TRAIN_LOG)):
        t[(r["config"], float(r["w"]))].append(float(r["train_s"]))
    return t


def report(moves):
    from scipy.stats import wilcoxon

    A = arrays(moves)
    keys_ref, err_ref, e_ref, _, _ = A[REF][""]
    order = list(PIDS) + [lab for lab in SAC if lab in A]
    lines = [
        "# Lab 12e: statistics of the real-plant comparison",
        "",
        "Generated by `python rl/12e_stats_eval.py`. No training. Lab 12 protocol: 100 unseen",
        "scenarios x 72 rows = 7200 moves per controller and seed, 10 substeps. SAC: residual on",
        "the balanced (bal) or economical (eco) PID point, 3 seeds; values are mean +- std over",
        "seeds of the per-seed results. All values ESTIMATED.",
        "",
        "## Accuracy and energy",
        "",
        "| controller | miss > 10 cm % | miss > 20 cm % | miss > 30 cm % | mean cm | p95 cm"
        " | worst cm | E_mech kJ | E_pump kJ | gap to floor % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def ms(x, f="{:.2f}"):
        x = np.asarray(x, float)
        return f.format(x.mean()) + (" +- " + f.format(x.std(ddof=1)) if len(x) > 1 else "")

    for lab in order:
        per = list(A[lab].values())
        miss = {t: [100 * (p[1] > t).mean() for p in per] for t in THRESHOLDS_CM}
        lines.append(
            f"| {lab} | "
            + " | ".join(ms(miss[t]) for t in THRESHOLDS_CM)
            + f" | {ms([p[1].mean() for p in per], '{:.1f}')} | "
            f"{ms([np.percentile(p[1], 95) for p in per], '{:.1f}')} | "
            f"{max(p[1].max() for p in per):.1f} | {ms([p[2].mean() for p in per], '{:.1f}')} | "
            f"{ms([p[3].mean() for p in per], '{:.1f}')} | "
            f"{ms([100 * (p[2].mean() / p[4].mean() - 1) for p in per], '{:.1f}')} |"
        )

    lines += [
        "",
        f"## Paired Wilcoxon signed-rank test against the {REF} PID",
        "",
        "Per move (same row and scenario, 7200 pairs); SAC = mean of its 3 seeds per move.",
        "Positive difference = SAC is worse (larger error, more energy).",
        "",
        "| controller | median diff tip error cm | p (error) | median diff E_mech kJ"
        " | p (energy) |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in order:
        if lab == REF:
            continue
        per = list(A[lab].values())
        assert all(p[0] == keys_ref for p in per), f"{lab}: moves not aligned with {REF}"
        err = np.mean([p[1] for p in per], axis=0)
        en = np.mean([p[2] for p in per], axis=0)
        de, dE = err - err_ref, en - e_ref
        pe = wilcoxon(err, err_ref).pvalue if np.any(de) else 1.0
        pE = wilcoxon(en, e_ref).pvalue if np.any(dE) else 1.0
        lines.append(
            f"| {lab} | {np.median(de):+.2f} | {pe:.1e} | {np.median(dE):+.2f} | {pE:.1e} |"
        )

    t = train_cost()
    if t:
        lines += [
            "",
            "## Training cost",
            "",
            "SAC: 100k environment steps per model; wall-clock from rl/12_robust_train_log.csv"
            " (depends on parallel load). PID points: no training (grid search on 20 tuning"
            " scenarios, Lab 11).",
            "",
            "| config | w | models | mean train time h |",
            "|---|---:|---:|---:|",
        ]
        for (cfg, w), v in sorted(t.items()):
            lines.append(f"| {cfg} | {w:g} | {len(v)} | {np.mean(v) / 3600:.2f} |")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def smoke():
    """4 rows: residual 0 on each base must equal that PID point, move by move."""
    rows = [(n, c, v) for _, n, c, v in M.ROWS72[:2]]
    js = [(p, None, "orig20", n, c, v) for p in PIDS for n, c, v in rows]
    js += [
        (lab, None, "orig20", n, c, v)
        for lab in ("res_bal_w0.1", "res_eco_w0.1")
        for n, c, v in rows
    ]
    res = M.pool_map(run_row, js)
    k = len(rows)
    pid = {"economical": res[0:k], "balanced": res[k : 2 * k]}
    worst = 0.0
    for lab, base, chunk in (
        ("res_bal_w0.1", "balanced", res[2 * k : 3 * k]),
        ("res_eco_w0.1", "economical", res[3 * k : 4 * k]),
    ):
        for a, b in zip(chunk, pid[base]):
            for x, y in zip(a, b):
                worst = max(
                    worst, abs(x["err_cm"] - y["err_cm"]), abs(x["E_mech_kJ"] - y["E_mech_kJ"])
                )
    print(
        f"smoke: residual 0 on balanced/economical base vs PID, {4 * k} rows,"
        f" largest difference {worst}"
    )
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
