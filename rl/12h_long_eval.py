"""Lab 12h: evaluate the longer-trained SAC checkpoints on the Lab 12 protocol. No training.

Models from rl/12h_long_train.py: residual SAC on the balanced PID point, w = 0.03, seeds 0-2,
checkpoints every 50k steps up to 300k (rl/models/12h_residual_w0.03_s{s}_{n}_steps.zip).
References on the same rows: the economical PID, and the Lab 12d 100k models
(rl/models/12_residual_w0.03_s{s}.zip) trained in one 100k run.
Protocol as rl/12e_stats_eval.py: 100 unseen scenarios x 72 rows, 10 substeps, same rollout.

The question: does SAC keep improving after 100k steps, and does any checkpoint beat the
economical PID on both misses and energy?

Usage (repo root, .venv active):
  python rl/12h_long_eval.py           evaluate (about 35-45 min on 16 cores) + report
  python rl/12h_long_eval.py --report  report only, from rl/12h_moves.csv.gz
  python rl/12h_long_eval.py --dry-run the *_dry checkpoints of `12h_long_train.py --dry-run`,
                                       2 rows, tuning scenarios, outputs *_dry
Writes rl/12h_moves.csv.gz (per move, not committed), rl/12h_long.csv, rl/12h_long.md and
rl/12h_long.png (training reward, misses and energy against training steps).
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

DRY = "--dry-run" in sys.argv
SUFFIX = "_dry" if DRY else ""
W = 0.03
REF = "economical"
CKPTS = (1_000, 2_000) if DRY else tuple(range(50_000, 300_001, 50_000))
ORIG = "orig_100k"  # Lab 12d model, one 100k run
SET = "orig20" if DRY else M.SET
ROWS = M.ROWS72[:2] if DRY else M.ROWS72
MISS_CM = 20.0
MOVES = f"rl/12h_moves{SUFFIX}.csv.gz"
CSV_PATH = f"rl/12h_long{SUFFIX}.csv"
MD_PATH = f"rl/12h_long{SUFFIX}.md"
PNG_PATH = f"rl/12h_long{SUFFIX}.png"
CURVE = f"rl/12h_train_curve{SUFFIX}.csv"
TRAIN_LOG = f"rl/12h_train_log{SUFFIX}.csv"
FIELDS = (
    "controller",
    "steps",
    "seed",
    "condition",
    "variant",
    "scen",
    "err_cm",
    "E_mech_kJ",
    "E_pump_kJ",
)


def model_path(steps, seed):
    if steps == ORIG:
        return f"{M.MODEL_DIR}/12_residual_w{W}_s{seed}.zip"
    return f"{M.MODEL_DIR}/12h_residual_w{W}_s{seed}{SUFFIX}_{steps}_steps.zip"


def run_row(job):
    """One controller (PID, or a SAC checkpoint and seed) on one row. Worker."""
    steps, seed, name, cond, variant = job
    env = L10.make_env(variant, **cond)
    tn = cond.get("tau_noise", 0.0)
    if steps is None:
        ctrl, label = L11.Sched11(E.point_gains(REF), E.EST, tau_noise=tn), REF
    else:
        model = M.load_model(model_path(steps, seed))
        ctrl, label = M.SacCtrl(model, True, gains=E.BAL_GAINS, tau_noise=tn), "sac"
    out = []
    for i, sc in enumerate(L11.scenarios(SET)):
        r = L11.rollout(ctrl, env, *sc)
        out.append(
            {
                "controller": label,
                "steps": "" if steps is None else steps,
                "seed": "" if seed is None else seed,
                "condition": name,
                "variant": variant,
                "scen": i,
                "err_cm": float(M.cm(r["err_deg"])),
                "E_mech_kJ": r["E_mech_kJ"],
                "E_pump_kJ": r["E_pump_kJ"],
            }
        )
    return out


def jobs():
    rows = [(n, c, v) for _, n, c, v in ROWS]
    steps = list(CKPTS) + ([] if DRY else [ORIG])
    out = [(None, None, n, c, v) for n, c, v in rows]
    out += [(st, s, n, c, v) for st in steps for s in M.SEEDS for n, c, v in rows]
    return out


def available_seeds(steps):
    return [s for s in M.SEEDS if os.path.exists(model_path(steps, s))]


# ----------------------------------------------------------------------------- report
def stats(moves):
    """(controller, steps) -> array of per-seed (miss %, mean cm, p95 cm, E_mech, E_pump)."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in moves:
        by[(m["controller"], m["steps"])][m["seed"]].append(m)
    out = {}
    for key, seeds in by.items():
        rows = []
        for ms in seeds.values():
            err = np.array([float(m["err_cm"]) for m in ms])
            rows.append(
                (
                    100 * (err > MISS_CM).mean(),
                    err.mean(),
                    np.percentile(err, 95),
                    np.mean([float(m["E_mech_kJ"]) for m in ms]),
                    np.mean([float(m["E_pump_kJ"]) for m in ms]),
                )
            )
        out[key] = np.array(rows)
    return out


def curve():
    """seed -> (timesteps, episode reward) from the training curve CSV."""
    if not os.path.exists(CURVE):
        return {}
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(CURVE)):
        by[int(r["seed"])].append((int(r["timesteps"]), float(r["ep_reward"])))
    return {s: np.array(sorted(v)) for s, v in by.items()}


def plot(S, order, ref):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for s, c in curve().items():
        k = max(1, len(c) // 60)
        smooth = np.convolve(c[:, 1], np.ones(k) / k, mode="valid")
        ax[0].plot(c[k - 1 :, 0] / 1e3, smooth, lw=1, label=f"seed {s}")
    ax[0].set(
        xlabel="training steps (k)", ylabel="episode reward (rolling mean)", title="Training reward"
    )
    ax[0].legend(fontsize=8)
    x = np.array([int(st) for st in order if st != ORIG]) / 1e3
    for j, (col, ylab) in enumerate(((0, "misses > 20 cm (%)"), (3, "E_mech per move (kJ)"))):
        a = ax[j + 1]
        y = [S[("sac", str(st))][:, col] for st in order if st != ORIG]
        a.errorbar(
            x,
            [v.mean() for v in y],
            yerr=[v.std(ddof=1) if len(v) > 1 else 0 for v in y],
            marker="o",
            capsize=3,
            label="SAC residual, w = 0.03 (300k run)",
        )
        if ("sac", ORIG) in S:
            o = S[("sac", ORIG)][:, col]
            a.errorbar(
                [100],
                [o.mean()],
                yerr=[o.std(ddof=1)],
                marker="s",
                ls="none",
                capsize=3,
                label="Lab 12d model (100k run)",
            )
        a.axhline(ref[col], color="k", ls="--", lw=1, label="economical PID")
        a.set(xlabel="training steps (k)", ylabel=ylab)
        a.legend(fontsize=8)
    ax[1].set_title("Misses")
    ax[2].set_title("Energy")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)


def report(moves):
    S = stats(moves)
    ref = S[(REF, "")].mean(axis=0)
    order = [str(st) for st in CKPTS if ("sac", str(st)) in S]
    if ("sac", ORIG) in S:
        order.append(ORIG)

    def ms(x, f):
        return f.format(x.mean()) + (" +- " + f.format(x.std(ddof=1)) if len(x) > 1 else "")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "controller",
                "steps",
                "n_seeds",
                "miss_pct",
                "miss_pct_std",
                "mean_cm",
                "p95_cm",
                "E_mech_kJ",
                "E_mech_std",
                "E_pump_kJ",
            ]
        )
        w.writerow(
            [REF, "", 0, *(f"{v:.4f}" for v in (ref[0], 0, ref[1], ref[2], ref[3], 0, ref[4]))]
        )
        for st in order:
            a = S[("sac", st)]
            sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(5)
            mu = a.mean(axis=0)
            w.writerow(
                [
                    "sac_residual_w0.03",
                    st,
                    len(a),
                    *(f"{v:.4f}" for v in (mu[0], sd[0], mu[1], mu[2], mu[3], sd[3], mu[4])),
                ]
            )

    lines = [
        "# Lab 12h: longer SAC training (residual on balanced, w = 0.03)",
        "",
        "Generated by `python rl/12h_long_eval.py` from the checkpoints of",
        "`python rl/12h_long_train.py` (3 seeds, 300k steps, checkpoint every 50k). Lab 12",
        f"protocol: {len(ROWS)} rows x {len(L11.scenarios(SET))} scenarios ({SET}), 10 substeps.",
        "SAC: mean +- std over seeds. The 100k row of the 300k run and the Lab 12d model (one",
        "100k run) differ only by run-to-run randomness. All values ESTIMATED.",
        "",
        "| controller | steps | miss > 20 cm % | mean cm | p95 cm | E_mech kJ | E_pump kJ"
        " | beats PID on both |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| economical PID | - | {ref[0]:.2f} | {ref[1]:.1f} | {ref[2]:.1f} | {ref[3]:.1f}"
        f" | {ref[4]:.1f} | - |",
    ]
    for st in order:
        a = S[("sac", st)]
        mu = a.mean(axis=0)
        beats = mu[0] <= ref[0] and mu[3] <= ref[3]
        name = "Lab 12d model" if st == ORIG else "SAC 300k run"
        steps = "100k" if st == ORIG else f"{int(st) // 1000}k"
        lines.append(
            f"| {name} | {steps} | {ms(a[:, 0], '{:.2f}')} | {ms(a[:, 1], '{:.1f}')}"
            f" | {ms(a[:, 2], '{:.1f}')} | {ms(a[:, 3], '{:.1f}')}"
            f" | {ms(a[:, 4], '{:.1f}')} | {'yes' if beats else 'no'} |"
        )
    ck = [st for st in order if st != ORIG]
    if len(ck) >= 2:
        first = "100000" if "100000" in ck else ck[0]
        a, b = S[("sac", first)].mean(axis=0), S[("sac", ck[-1])].mean(axis=0)
        lines += [
            "",
            f"From {int(first) // 1000}k to {int(ck[-1]) // 1000}k steps: misses"
            f" {a[0]:.2f} -> {b[0]:.2f} %, mean error {a[1]:.1f} -> {b[1]:.1f} cm, E_mech"
            f" {a[3]:.1f} -> {b[3]:.1f} kJ.",
        ]
    if os.path.exists(TRAIN_LOG):
        t = collections.defaultdict(float)
        for r in csv.DictReader(open(TRAIN_LOG)):
            t[int(r["seed"])] += float(r["train_s"])
        lines += [
            "",
            "Training time per seed (h): "
            + ", ".join(f"seed {s}: {v / 3600:.2f}" for s, v in sorted(t.items()))
            + ".",
        ]
    lines += [
        "",
        f"![training curve, misses and energy against steps]({os.path.basename(PNG_PATH)})",
    ]
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(S, order, ref)


def write_moves(rows):
    with gzip.open(MOVES, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_moves():
    with gzip.open(MOVES, "rt") as f:
        return list(csv.DictReader(f))


def main():
    if "--report" in sys.argv:
        report(read_moves())
        return
    need = [model_path(st, s) for st in CKPTS for s in M.SEEDS]
    if DRY:
        need = [model_path(st, 0) for st in CKPTS]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit("missing checkpoints (run rl/12h_long_train.py first):\n  " + "\n  ".join(missing))
    js = jobs()
    js = [j for j in js if j[0] is None or os.path.exists(model_path(j[0], j[1]))]
    rows = [m for chunk in M.pool_map(run_row, js) for m in chunk]
    write_moves(rows)
    report(read_moves())


if __name__ == "__main__":
    main()
