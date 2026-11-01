"""Lab 09: paper figures and Table 1 from a 08b-style results CSV.

Reads --csv (default rl/08b_eval_n10.csv, columns as written by rl/08b_eval.py) and writes to paper/:
  figs/fig1_energy_vs_robustness.{png,pdf}  x = mean kJ on new100 baseline, y = worst variant
                                            % > 20 cm on new100 (all variants), one point per controller
  figs/fig2_robustness_bars.{png,pdf}       % > 20 cm per variant on new100, selected controllers
  figs/fig3_payload_trajectory.{png,pdf}    theta(t) and target (top), applied command u(t) (bottom),
                                            one new100 scenario chosen by rule (FIG3_TARGET_DEG),
                                            payload x1.2 at step 100, run live with substeps = 10
  table1.md, table1.tex                     new100 baseline (mean cm, worst cm, kJ, % > 20 cm),
                                            out100 baseline % > 20 cm, worst variant % > 20 cm
Aggregation over seeds as 08b: mean of means, worst of maxima, mean of shares. The worst variant is
the variant with the highest seed-mean share. Errors in cm use BOOM_LENGTH_OFFICIAL_M.
Fig 3 does not read the CSV: it always runs RobustBoomEnv(substeps=10), seed-0 models.

Usage: python rl/09_paper_figures.py [--csv PATH]
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from boom_env_robust import RobustBoomEnv, PID_GAINS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P

CSV_PATH = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else "rl/08b_eval_n10.csv"
BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
MODEL_DIR = "rl/models"
FIG_DIR, PAPER_DIR = "paper/figs", "paper"
CONFIGS, WEIGHTS = ["plain", "residual"], ["0.0", "0.1", "0.3"]
PIDS = {"pid": (0.5, 0.0, 0.25), "pid_ki": (2.0, 0.1, 1.0)}   # Kp, Ki, Kd tuned in 07
FIG2_VARIANTS = ["baseline", "m_x0.42", "tau_d", "payload_x1.2", "noise_1.0"]
FIG2_CONTROLLERS = ["pid", "pid_ki", "plain w=0.1", "residual w=0.1", "plain w=0.3"]
FIG3_CONTROLLERS = ["pid_ki", "plain w=0.1", "residual w=0.1"]
FIG3_SUBSTEPS, PAYLOAD_K, PAYLOAD_FACTOR = 10, 100, 1.2
# Fig 3 caption rule: among the new100 scenarios with target in [-10, +15] deg (boom near horizontal,
# gravity torque large, so the payload step matters), the one with the largest move |target - theta0|.
FIG3_TARGET_DEG = (-10.0, 15.0)

# one color per controller in every figure: blues = plain, oranges = residual, greys = PID
COLORS = {"plain w=0.0": "#9ecae1", "plain w=0.1": "#3182bd", "plain w=0.3": "#08519c",
          "residual w=0.0": "#fdae6b", "residual w=0.1": "#e6550d", "residual w=0.3": "#a63603",
          "pid": "#969696", "pid_ki": "#252525"}
LABELS = [f"{c} w={w}" for c in CONFIGS for w in WEIGHTS] + list(PIDS)
LABEL_OFFSETS = {"plain w=0.1": (4, -9), "residual w=0.3": (4, -9)}   # Fig 1, points; default (4, 3)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL_W, PAGE_W = 3.5, 7.16   # in, one column and full width of a 2-column paper


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
VARIANTS = list(dict.fromkeys(r["variant"] for r in rows))


def label_of(r):
    return r["controller"] if r["controller"] in PIDS else f"{r['controller']} w={r['w']}"


def agg(label, set_name, variant):
    """(mean cm, worst cm, mean kJ, % > 20 cm) over seeds."""
    rs = [r for r in rows if label_of(r) == label and r["set"] == set_name and r["variant"] == variant]
    assert rs, (label, set_name, variant)
    return (cm(np.mean([float(r["mean_deg"]) for r in rs])), cm(max(float(r["max_deg"]) for r in rs)),
            np.mean([float(r["mean_kJ"]) for r in rs]),
            np.mean([float(r["share_over_20cm"]) for r in rs]) * 100)


def worst_variant(label):
    v = max(VARIANTS, key=lambda v: agg(label, "new100", v)[3])
    return v, agg(label, "new100", v)[3]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"{FIG_DIR}/{name}.png/.pdf")


os.makedirs(FIG_DIR, exist_ok=True)

# Fig 1: energy vs robustness
fig, ax = plt.subplots(figsize=(COL_W, 2.6))
for label in LABELS:
    x, y = agg(label, "new100", "baseline")[2], worst_variant(label)[1]
    ax.scatter(x, y, s=28, color=COLORS[label], edgecolor="k", linewidth=0.4, zorder=3)
    ax.annotate(label, (x, y), xytext=LABEL_OFFSETS.get(label, (4, 3)), textcoords="offset points",
                fontsize=6.5)
ax.set_xlabel("Mean energy, new100 baseline [kJ]")
ax.set_ylabel("Worst variant: share > 20 cm [%]")
ax.grid(alpha=0.3, linewidth=0.5)
ax.set_ylim(bottom=0)
save(fig, "fig1_energy_vs_robustness")

# Fig 2: robustness bars
fig, ax = plt.subplots(figsize=(PAGE_W, 2.4))
n, width = len(FIG2_CONTROLLERS), 0.8 / len(FIG2_CONTROLLERS)
xs = np.arange(len(FIG2_VARIANTS))
for i, label in enumerate(FIG2_CONTROLLERS):
    ys = [agg(label, "new100", v)[3] for v in FIG2_VARIANTS]
    ax.bar(xs + (i - (n - 1) / 2) * width, ys, width, color=COLORS[label], edgecolor="k",
           linewidth=0.4, label=label)
ax.set_xticks(xs, FIG2_VARIANTS)
ax.set_ylabel("Share > 20 cm [%]")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.legend(ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
save(fig, "fig2_robustness_bars")

# Fig 3: payload trajectory, live, same reset/variant logic as rl/08b_eval.py
rng = np.random.default_rng(123)    # new100, same as 05/06/07/08b
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]
lo, hi = np.radians(FIG3_TARGET_DEG)
i_sc = max((i for i, (_, tg) in enumerate(NEW) if lo <= tg <= hi),
           key=lambda i: abs(NEW[i][1] - NEW[i][0]))
th0, target = NEW[i_sc]
print(f"fig3 scenario: new100 #{i_sc}, theta0 {np.degrees(th0):.1f} deg -> target"
      f" {np.degrees(target):.1f} deg (rule: target in {FIG3_TARGET_DEG} deg, largest move)")
J_NOM = RobustBoomEnv().J


def start(env, th0, target):
    """08b start() for the variant payload_x1.2."""
    env.reset(seed=0)
    env.J = J_NOM
    env.payload_k, env.payload_factor = PAYLOAD_K, PAYLOAD_FACTOR
    env.th, env.target, env.om = th0, target, 0.0
    env.integ, env.pid_integ, env.k, env.energy_J = 0.0, 0.0, 0, 0.0
    env._measure()
    return env._obs()


def pid_policy(gains, u_ff_scale, dt):
    kp, ki, kd = gains

    def act(obs, state):   # same law as 07/08b
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])
        u_raw = kp * e + ki * state["integ"] + kd * (-om) + u_ff_scale * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup
            state["integ"] += e * dt
        return np.array([np.clip(u_raw, -1.0, 1.0)])
    return act


def applied_u(env, a):
    """Command u that env.step(a) applies: a clipped, or for the residual env the PID sum, as in
    RobustBoomEnv.step(). Call before env.step(a)."""
    u = float(np.clip(a[0], -1, 1))
    if env.residual:
        kp, ki, kd = PID_GAINS
        e = env.target - env.th_meas
        u = float(np.clip(kp * e + ki * env.pid_integ - kd * env.om_meas
                          + env.u_ff_scale * np.cos(env.th_meas) + u, -1, 1))
    return u


fig, (ax, ax_u) = plt.subplots(2, 1, figsize=(COL_W, 3.6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
for label in FIG3_CONTROLLERS:
    if label in PIDS:
        env = RobustBoomEnv(w_energy=0.0, residual=False, randomize=False, substeps=FIG3_SUBSTEPS)
        act = pid_policy(PIDS[label], env.u_ff_scale, env.dt)
    else:
        config, w = label.split(" w=")
        env = RobustBoomEnv(w_energy=float(w), residual=(config == "residual"), randomize=False,
                            substeps=FIG3_SUBSTEPS)
        model = SAC.load(f"{MODEL_DIR}/08b_{config}_w{w}_s0.zip", device="cpu")
        act = lambda obs, state, model=model: model.predict(obs, deterministic=True)[0]
    obs, state, th, us = start(env, th0, target), {"integ": 0.0}, [th0], []
    for _ in range(env.max_steps):
        a = act(obs, state)
        us.append(applied_u(env, a))
        obs, _, _, trunc, _ = env.step(a)
        th.append(env.th)
        if trunc:
            break
    t = np.arange(len(th)) * env.dt
    ax.plot(t, np.degrees(th), color=COLORS[label], linewidth=1.1, label=label)
    ax_u.step(t[:-1], us, where="post", color=COLORS[label], linewidth=0.9)   # u held over each step
    print(f"fig3 {label:<14} final error {cm(abs(np.degrees(target - env.th))):5.1f} cm,"
          f" {env.energy_J / 1000:6.1f} kJ")
ax.axhline(np.degrees(target), color="k", linestyle="--", linewidth=0.8, label="target")
for a in (ax, ax_u):
    a.axvline(PAYLOAD_K * env.dt, color="#d62728", linestyle=":", linewidth=0.9)
    a.grid(alpha=0.3, linewidth=0.5)
ax.annotate(f"payload x{PAYLOAD_FACTOR}", (PAYLOAD_K * env.dt, ax.get_ylim()[1]),
            xytext=(-3, -9), textcoords="offset points", ha="right", color="#d62728", fontsize=6.5)
ax.set_ylabel(r"Boom angle $\theta$ [deg]")
ax.legend(frameon=False, loc="best")
ax_u.set_ylim(-1.05, 1.05)
ax_u.set_xlabel("Time [s]")
ax_u.set_ylabel("Command $u$ [-]")
fig.align_ylabels((ax, ax_u))
save(fig, "fig3_payload_trajectory")

# Table 1
table = []
for label in LABELS:
    m, x, k, o = agg(label, "new100", "baseline")
    v, ov = worst_variant(label)
    table.append((label, m, x, k, o, agg(label, "out100", "baseline")[3], ov, v))
src = f"Source: {CSV_PATH}. Seeds: mean of means, worst of maxima, mean of shares."
with open(f"{PAPER_DIR}/table1.md", "w") as f:
    f.write("| Controller | Mean [cm] | Worst [cm] | Energy [kJ] | > 20 cm [%] | out100 > 20 cm [%] |"
            " Worst variant > 20 cm [%] |\n")
    f.write("|---|---:|---:|---:|---:|---:|---|\n")
    for label, m, x, k, o, oo, ov, v in table:
        f.write(f"| {label} | {m:.1f} | {x:.1f} | {k:.1f} | {o:.1f} | {oo:.1f} | {ov:.1f} ({v}) |\n")
    f.write(f"\nnew100 baseline unless noted. {src}\n")
with open(f"{PAPER_DIR}/table1.tex", "w") as f:
    f.write("% " + src + "\n\\begin{tabular}{lrrrrrl}\n\\hline\n"
            "Controller & Mean & Worst & Energy & $>$20\\,cm & out100 $>$20\\,cm & Worst variant $>$20\\,cm \\\\\n"
            " & [cm] & [cm] & [kJ] & [\\%] & [\\%] & [\\%] \\\\\n\\hline\n")
    for label, m, x, k, o, oo, ov, v in table:
        tex_label, tex_v = label.replace("_", "\\_"), v.replace("_", "\\_")
        f.write(f"{tex_label} & {m:.1f} & {x:.1f} & {k:.1f} & {o:.1f} & {oo:.1f} & {ov:.1f} ({tex_v}) \\\\\n")
    f.write("\\hline\n\\end{tabular}\n")
print(f"{PAPER_DIR}/table1.md, {PAPER_DIR}/table1.tex")
