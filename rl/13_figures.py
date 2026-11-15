"""Lab 13 figures for the Design of AI Systems paper. Existing result files only: no training, no
evaluation, no rollout.

Sources (every number; the list per figure is in paper/figs/lab13/README.md):
  rl/13_benchmark.csv         all controllers except the fast PID: error, E_mech, gap, du/dt, energy split
  rl/12_eval.csv              fast PID (72 rows of the same protocol) and the mean energy floor
  docs/13_benchmark.md        reward return of the balanced and economical PID without noise (section 3)
  rl/tb/12_residual_w0.03_*   training return of SAC, TD3 and PPO (TensorBoard rollout/ep_rew_mean).
  rl/tb/13_*                  rl/tb is gitignored, so the values read are saved to
                              paper/figs/lab13/train_returns.csv and read from there when rl/tb is missing.

Writes paper/figs/lab13/fig1..fig5 as .pdf and .png, 3.5 in wide (one IEEE column).
Usage (repo root, .venv active):  python rl/13_figures.py
"""

import csv
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dx300sim import dx300_params as P

BOOM_M = P.BOOM_LENGTH_OFFICIAL_M  # 6.245 m, read from the parameter file

OUT = "paper/figs/lab13"
RET_CSV = f"{OUT}/train_returns.csv"
COL_W = 3.5
# Okabe-Ito colours, one per group; the marker shape repeats the group, so colour is never alone.
GROUPS = {"classical": ("#000000", "s"), "online on-policy": ("#D55E00", "^"),
          "online off-policy": ("#0072B2", "o"), "offline": ("#009E73", "D")}
NAME = {"fast": "fast PID", "balanced": "balanced PID", "economical": "economical PID",
        "ppo_100k": "PPO 100k", "ppo_1M": "PPO 1M", "res_bal_w0.03": "SAC", "td3_100k": "TD3",
        "bc_1M": "BC", "cql_1M": "CQL", "iql_1M": "IQL"}
MAIN = ("fast", "balanced", "economical", "ppo_100k", "ppo_1M", "res_bal_w0.03", "td3_100k",
        "bc_1M", "cql_1M", "iql_1M")
PID_RETURN = {"balanced": -9.78, "economical": -10.30}  # docs/13_benchmark.md section 3, sigma = 0 rows
TB_RUNS = {"res_bal_w0.03": ("12_residual_w0.03_s{}_*", 100_000), "td3_100k": ("13_td3_100k_s{}_*", 100_000),
           "ppo_100k": ("13_ppo_100k_s{}_*", 100_352), "ppo_1M": ("13_ppo_1M_s{}_*", 1_001_472)}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                     "grid.linewidth": 0.4, "grid.color": "#d0d0d0", "lines.linewidth": 1.2})


def f(x):
    return float(x) if x not in ("", "nan") else float("nan")


def load():
    """label -> dict(group, err, err_sd, E, E_sd, gap, gap_sd, du, osc_kJ)."""
    d = {}
    for r in csv.DictReader(open("rl/13_benchmark.csv")):
        d[r["controller"]] = dict(group=r["group"], err=f(r["mean_cm_mean"]), err_sd=f(r["mean_cm_std"]),
                                  E=f(r["E_mech_mean"]), E_sd=f(r["E_mech_std"]), gap=f(r["gap_mean"]),
                                  gap_sd=f(r["gap_std"]), du=f(r["du_dt"]),
                                  osc_kJ=f(r["excess_kJ"]) * f(r["oscillation_share"]) / 100)
    rows = [r for r in csv.DictReader(open("rl/12_eval.csv")) if r["set"] == "new100"]
    fast = [r for r in rows if r["controller"] == "fast"]
    eco = [r for r in rows if r["controller"] == "economical"]
    assert len(fast) == 72 and len(eco) == 72
    floor = float(np.mean([f(r["mean_floor_kJ"]) for r in fast]))
    E = float(np.mean([f(r["mean_E_mech_kJ"]) for r in fast]))
    err = float(np.mean([BOOM_M * np.radians(f(r["mean_deg"])) * 100 for r in fast]))
    nan = float("nan")
    d["fast"] = dict(group="classical", err=err, err_sd=nan, E=E, E_sd=nan, gap=100 * (E / floor - 1),
                     gap_sd=nan, du=nan, osc_kJ=nan)
    # the two files must describe the same protocol: economical PID agrees
    assert abs(np.mean([f(r["mean_E_mech_kJ"]) for r in eco]) - d["economical"]["E"]) < 1e-6
    return d, floor


def train_returns():
    """label -> list of per-seed training returns (mean of rollout/ep_rew_mean over the last 10 % of steps)."""
    out = {}
    if os.path.isdir("rl/tb") and glob.glob("rl/tb/13_td3_100k_s0_*"):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        rows = []
        for lab, (pat, steps) in TB_RUNS.items():
            for seed in (0, 1, 2):
                best = None
                for run in sorted(glob.glob("rl/tb/" + pat.format(seed))):
                    if re.search(r"smoke|dry", run):
                        continue
                    ea = EventAccumulator(run)
                    ea.Reload()
                    s = ea.Scalars("rollout/ep_rew_mean")
                    if s[-1].step == steps:  # the run that reached the full budget
                        best = (run, s)
                assert best, (lab, seed)
                vals = [x.value for x in best[1] if x.step > 0.9 * steps]
                rows.append({"controller": lab, "seed": seed, "tb_run": best[0], "steps": steps,
                             "points": len(vals), "train_return": float(np.mean(vals))})
        with open(RET_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in csv.DictReader(open(RET_CSV)):
        out.setdefault(r["controller"], []).append(float(r["train_return"]))
    return out


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{OUT}/{name}.pdf/.png")


def point(ax, d, lab, x, y, xe=None, ye=None, hollow=False):
    c, m = GROUPS[d[lab]["group"]]
    ax.errorbar(x, y, xerr=None if xe is None or np.isnan(xe) else xe,
                yerr=None if ye is None or np.isnan(ye) else ye, fmt=m, ms=4.5, color=c,
                mfc="white" if hollow else c, mec=c, mew=0.9, elinewidth=0.6, capsize=1.5, zorder=3)


def label(ax, text, x, y, dx, dy, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=6.5, color="#222222")


def group_legend(ax, **kw):
    h = [plt.Line2D([], [], marker=m, ls="", ms=4.5, color=c, label=g) for g, (c, m) in GROUPS.items()]
    return ax.legend(handles=h, frameon=False, handletextpad=0.2, borderaxespad=0.2, **kw)


def fig1(d, floor):
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 4.4), gridspec_kw={"hspace": 0.42})
    off = {"fast": (5, 0, "left"), "ppo_100k": (0, -13, "center"), "ppo_1M": (0, 9, "center"),
           "res_bal_w0.03": (0, -10, "center"), "td3_100k": (0, -9, "center"), "cql_1M": (5, -13, "left")}
    for lab in MAIN:
        r = d[lab]
        point(a, d, lab, r["err"], r["E"], r["err_sd"], r["E_sd"])
        if lab in off:
            label(a, NAME[lab], r["err"], r["E"], off[lab][0], off[lab][1], off[lab][2])
    a.axhline(floor, color="#555555", ls="--", lw=0.8)
    a.text(20.5, floor + 4, f"energy floor {floor:.1f} kJ", ha="right", va="bottom", fontsize=6.5, color="#555555")
    a.add_patch(plt.Rectangle((1.0, 156), 2.0, 14, fill=False, ec="#888888", lw=0.6, ls=":"))
    a.text(3.15, 151, "(b)", fontsize=6.5, color="#555555", va="top")
    a.set(xlim=(0, 21), ylim=(120, 345), ylabel="mean energy E_mech (kJ)")
    a.set_title("(a) all controllers", loc="left")
    group_legend(a, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    off = {"balanced": (6, 0, "left"), "economical": (-6, -4, "right"), "bc_1M": (6, -5, "left"),
           "iql_1M": (6, 4, "left")}
    for lab in off:
        r = d[lab]
        point(b, d, lab, r["err"], r["E"], r["err_sd"], r["E_sd"])
        label(b, NAME[lab], r["err"], r["E"], *off[lab])
    b.set(xlim=(1.0, 3.2), ylim=(157.5, 169), xlabel="mean tip error (cm)", ylabel="mean energy E_mech (kJ)")
    b.set_title("(b) zoom: PID points, BC and IQL", loc="left")
    for ax in (a, b):
        ax.grid(True)
        ax.set_axisbelow(True)
    save(fig, "fig1_energy_vs_error")


def fig2(d):
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    labs = MAIN[::-1]
    for i, lab in enumerate(labs):
        r = d[lab]
        c = GROUPS[r["group"]][0]
        sd = 0 if np.isnan(r["gap_sd"]) else r["gap_sd"]
        ax.barh(i, r["gap"], height=0.62, color=c, xerr=sd or None,
                error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#444444"))
        ax.text(r["gap"] + sd + 2.5, i, f"{r['gap']:.1f}", va="center", fontsize=6.5, color="#222222")
    ax.set_yticks(range(len(labs)), [NAME[k] for k in labs])
    ax.set(xlim=(0, 172), xlabel="energy above the floor (%)")
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, label=g) for g, (c, _) in GROUPS.items()]
    ax.legend(handles=h, frameon=False, loc="lower right", handlelength=1.0, handleheight=0.8)
    save(fig, "fig2_gap_to_floor")


def fig3(d):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw={"wspace": 0.5})
    sizes = ("100k", "300k", "1M")
    style = {"bc": ("#009E73", "D", "-", "BC"), "iql": ("#CC79A7", "v", "--", "IQL")}
    for ax, key, sd, ylab, ref in ((axes[0], "err", "err_sd", "mean tip error (cm)", d["economical"]["err"]),
                                   (axes[1], "E", "E_sd", "mean E_mech (kJ)", d["economical"]["E"])):
        for j, (al, (c, m, ls, nm)) in enumerate(style.items()):
            x = np.arange(3) + (j - 0.5) * 0.12
            ax.errorbar(x, [d[f"{al}_{s}"][key] for s in sizes], yerr=[d[f"{al}_{s}"][sd] for s in sizes],
                        color=c, marker=m, ms=4, ls=ls, lw=1.0, elinewidth=0.6, capsize=1.5, label=nm)
        ax.axhline(ref, color="#000000", ls=":", lw=0.8, label="economical PID")
        ax.set_xticks(range(3), sizes)
        ax.set(xlim=(-0.4, 2.4), xlabel="dataset transitions", ylabel=ylab)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 5.5)
    axes[1].set_ylim(158, 166)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=3, handlelength=1.8,
               columnspacing=1.2)
    save(fig, "fig3_dataset_size")


def fig4(d):
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    off = {"balanced": (7, -5, "left"), "economical": (0, -9, "center"), "ppo_100k": (-6, 0, "right"),
           "ppo_1M": (-6, 0, "right"), "res_bal_w0.03": (6, 2, "left"), "td3_100k": (6, -4, "left"),
           "cql_1M": (5, 8, "left"), "bc_1M": (-7, -1, "right"), "iql_1M": (-5, 10, "right")}
    for lab in off:
        r = d[lab]
        point(ax, d, lab, r["du"], r["osc_kJ"])
        label(ax, NAME[lab].replace(" PID", ""), r["du"], r["osc_kJ"], *off[lab])
    ax.set(xscale="log", yscale="log", xlim=(0.3, 30), ylim=(10, 200),
           xlabel="command activity, mean |du/dt| (1/s)", ylabel="oscillation energy (kJ per move)")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.grid(True)
    ax.set_axisbelow(True)
    group_legend(ax, loc="upper left")
    save(fig, "fig4_smoothness_vs_oscillation")


def fig5(d, ret):
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    for lab, (dx, dy, ha) in {"balanced": (6, 3, "left"), "economical": (0, -10, "center")}.items():
        point(ax, d, lab, PID_RETURN[lab], d[lab]["gap"])
        label(ax, NAME[lab], PID_RETURN[lab], d[lab]["gap"], dx, dy, ha)
    b, e = PID_RETURN["balanced"], PID_RETURN["economical"]
    ax.annotate("", (b, d["balanced"]["gap"]), (e, d["economical"]["gap"]),
                arrowprops=dict(arrowstyle="->", color="#000000", lw=0.8, shrinkA=4, shrinkB=4))
    ax.text(-8.5, 118, "reward prefers the balanced PID:\nhigher return, more energy", fontsize=6.5, ha="right",
            va="center", color="#222222")
    for lab, (dx, dy, ha) in {"res_bal_w0.03": (0, 10, "center"), "td3_100k": (0, -9, "center"),
                              "ppo_100k": (0, 12, "center"), "ppo_1M": (0, 10, "center")}.items():
        v = ret[lab]
        point(ax, d, lab, np.mean(v), d[lab]["gap"], np.std(v, ddof=1), d[lab]["gap_sd"], hollow=True)
        label(ax, NAME[lab], np.mean(v), d[lab]["gap"], dx, dy, ha)
    ax.set(xlim=(-12.4, -8.4), ylim=(0, 175), xlabel="reward return per episode (higher is better)",
           ylabel="energy above the floor (%)")
    ax.grid(True)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="s", ls="", ms=4.5, color="#000000", label="PID, return without noise"),
         plt.Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", color="#555555",
                    label="online learner, training return")]
    ax.legend(handles=h, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.2,
              borderaxespad=0.1)
    save(fig, "fig5_reward_mismatch")


def main():
    os.makedirs(OUT, exist_ok=True)
    d, floor = load()
    ret = train_returns()
    fig1(d, floor)
    fig2(d)
    fig3(d)
    fig4(d)
    fig5(d, ret)
    print(f"energy floor {floor:.2f} kJ; fast PID: {d['fast']['err']:.2f} cm, {d['fast']['E']:.1f} kJ,"
          f" gap {d['fast']['gap']:.1f} %")
    for k, v in ret.items():
        print(f"training return {k}: seeds {', '.join(f'{x:.2f}' for x in v)}; mean {np.mean(v):.2f}"
              f" +- {np.std(v, ddof=1):.2f}")


if __name__ == "__main__":
    main()
