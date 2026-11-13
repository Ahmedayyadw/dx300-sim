"""Lab 12 step 2 (issue #7): evaluate the trained SAC models on the real plant. No training.

Models  rl/models/12_{residual,plain}_w0.1_s{0,1,2}.zip, trained by rl/12_robust_train.py on
        BoomEnvRealRobust (rl/boom_env_real_robust.py), 100k steps each, no sensor noise;
        rl/models/12n_residual_w0.1_s{0,1,2}.zip (step 3), same with --train-noise (sensor noise
        per episode from {0, 0.2, 0.5, 1.0} deg).
Against the three Lab 11 points (rl/11_points.json: fast, balanced, economical) with fix 2b
        (adapt_Tf1_c1_n2 of rl/11_dump_fix_choice.json).
Protocol exactly as `--step eval` of rl/11_real_plant_tuning.py: the 100 unseen scenarios
        (new100), all 72 rows (18 conditions x 4 variants of rl/10_sched_real.py), 10 s per move,
        10 substeps, noise seed 0 (the Lab 10 noise streams), Lab 11 rollout and energy floor.

SAC in the Lab 11 rollout: SacCtrl below has the controller interface of Sched11 (act / update).
  observation  [theta, omega, e, int(e) dt] as BoomEnvRealRobust, built from the measured signals
               the PID also sees (noisy on the noise rows); the integral uses the measured e and is
               clipped to +-1, reset each move. Without sensor noise this is the training
               observation exactly.
  action       model.predict(obs, deterministic=True), the SB3 policy as trained.
  residual     u = clip(u_base + a, -1, 1), u_base = Sched11(balanced, fix 2b); the estimator is
               told the applied u, as in BoomEnvRealRobust.step.
  plain        u = a. The Sched11 inside still runs as an observer only (it gives the m_hat columns
               of the row); its command is not used.
  torque limit tau_scale = 1 (nominal plant): the 72 rows do not scale the torque limit.

Checks (run before every evaluation, tuning scenarios orig20 only):
  1  SacCtrl residual with a = 0 equals Sched11 balanced + fix 2b bit for bit (4 noisy/stress rows).
  2  SacCtrl with each seed-0 model equals stepping BoomEnvRealRobust (randomize=False) with
     model.predict, bit for bit (no sensor noise, baseline and dump variants).
  3  after the evaluation: the three PID points equal the fix 2b rows of rl/11_real_plant_tuning.csv
     (same 72 rows, unseen set) on every compared metric.

Seeds: per row, mean metrics are the mean over the 3 seeds (mean of means), max metrics the largest
over the 3 seeds (worst of maxima). Status of a row as Lab 10/11: BREAKS if miss > 5 %, degraded if
> 1 %, else ok; counted on the seed-mean miss, plus the rows over 5 % in any single seed.

Writes rl/12_eval.csv (one line per controller x seed x row) and rl/12_eval.md.
Usage (repo root, .venv active):
  python rl/12_eval.py            checks, evaluation, report
  python rl/12_eval.py --report   report only, from rl/12_eval.csv
  python rl/12_eval.py --check    checks only
"""
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import boom_env_real_robust as E  # noqa: E402

L11, L10, R = E.L11, E.L10, E.L10.R
cm = L10.cm

OUT_CSV = "rl/12_eval.csv"
MD_PATH = "rl/12_eval.md"
L11_CSV = "rl/11_real_plant_tuning.csv"                 # read only
MODEL_DIR = "rl/models"
TRAIN_LOG = "rl/12_robust_train_log.csv"                 # read only
W = 0.1
SEEDS = (0, 1, 2)
SAC_MODELS = {                                           # label: (file prefix, config, train log config)
    "sac_residual": ("12", "residual", "residual"),
    "sac_plain": ("12", "plain", "plain"),
    "sac_noise_residual": ("12n", "residual", "residual_noise"),
}
SAC_NAME = {"sac_residual": "SAC residual", "sac_plain": "SAC plain",
            "sac_noise_residual": "SAC residual, noise-trained (12n)"}
ROLES = ("fast", "balanced", "economical")
SET = "new100"
ROWS72 = [(group, name, cond, variant) for group, cases in L10.GROUPS for name, cond in cases
          for variant in L10.VARIANTS]


def model_name(label, seed):
    prefix, config, _ = SAC_MODELS[label]
    return f"{prefix}_{config}_w{W:g}_s{seed}"


def model_path(label, seed):
    prefix, config, _ = SAC_MODELS[label]
    return f"{MODEL_DIR}/{prefix}_{config}_w{W}_s{seed}.zip"


_MODELS = {}


def load_model(path):
    if path not in _MODELS:
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        _MODELS[path] = SAC.load(path, device="cpu")
    return _MODELS[path]


class SacCtrl(L11.Sched11):
    """SAC policy (plain or residual on Sched11) with the Sched11 controller interface."""

    def __init__(self, model, residual, gains=E.BAL_GAINS, est=E.EST, tau_noise=0.0):
        super().__init__(gains, est, tau_noise=tau_noise)
        self.model, self.residual = model, residual

    def reset(self):
        super().reset()
        self.integ_obs = 0.0

    def act(self, obs):
        u_base = super().act(obs)                       # sets prev; Sched11 state as in the env
        x = np.array([obs[0], obs[1], obs[2], self.integ_obs], dtype=np.float32)
        a = 0.0 if self.model is None else float(np.clip(self.model.predict(x, deterministic=True)[0][0], -1, 1))
        u = float(np.clip(u_base + a, -1.0, 1.0)) if self.residual else a
        self.u_prev = u                                  # estimator: the command really applied
        return u

    def update(self, obs, tau, th_min, th_max):
        super().update(obs, tau, th_min, th_max)
        self.integ_obs = float(np.clip(self.integ_obs + float(obs[2]) * self.dt, -E.INTEG_CLIP, E.INTEG_CLIP))


def make_ctrl(kind, arg, tau_noise):
    if kind == "pid":
        return L11.Sched11(arg, E.EST, tau_noise=tau_noise)
    label, seed = arg
    model = None if seed is None else load_model(model_path(label, seed))
    return SacCtrl(model, SAC_MODELS[label][1] == "residual", tau_noise=tau_noise)


def run_row(job):
    """One row: a controller on one condition and variant, all scenarios of a set. Worker."""
    label, kind, arg, seed, set_name, group, name, cond, variant = job
    env = L10.make_env(variant, **cond)
    ctrl = make_ctrl(kind, arg, cond.get("tau_noise", 0.0))
    rs = [L11.rollout(ctrl, env, *sc) for sc in L11.scenarios(set_name)]
    errs = np.array([r["err_deg"] for r in rs])
    reach = np.array([r["t_reach_s"] for r in rs])
    tail = np.concatenate([r["tail"] for r in rs])
    e_mean = float(np.mean([r["E_mech_kJ"] for r in rs]))
    row = {"set": set_name, "n_scen": len(rs), "controller": label, "sac_seed": "" if seed is None else seed,
           "group": group, "condition": name, "sigma_th_deg": cond.get("sigma", 0.0),
           "tau_noise": cond.get("tau_noise", 0.0), "J_factor": cond.get("j", 1.0),
           "b_factor": cond.get("b", 1.0), "variant": variant,
           "mean_deg": float(errs.mean()), "max_deg": float(errs.max()), "max_cm": float(cm(errs.max())),
           "share_over_20cm": float((cm(errs) > 20).mean()),
           "mean_E_mech_kJ": e_mean, "mean_E_pump_kJ": float(np.mean([r["E_pump_kJ"] for r in rs])),
           "mean_t_reach_s": float(np.nanmean(reach)) if not np.all(np.isnan(reach)) else float("nan"),
           "max_t_reach_s": float(np.nanmax(reach)) if not np.all(np.isnan(reach)) else float("nan"),
           "n_never_reached": int(np.isnan(reach).sum()),
           "share_flow_limit": float(np.mean([r["flow"] for r in rs])),
           "m_hat_over_m_mean": float(tail.mean()), "m_hat_over_m_std": float(tail.std())}
    for key in L11.SUM_KEYS:
        row["mean_" + key] = float(np.mean([r[key] for r in rs]))
    row["pct_above_floor"] = 100.0 * (e_mean / row["mean_floor_kJ"] - 1.0)
    return row


def jobs(set_name=SET):
    pts = dict(L11.load_points())
    out = [(role, "pid", pts[role], None, set_name, g, n, c, v) for role in ROLES for g, n, c, v in ROWS72]
    out += [(lab, "sac", (lab, s), s, set_name, g, n, c, v)
            for lab in SAC_MODELS for s in SEEDS for g, n, c, v in ROWS72]
    return out


def pool_map(fn, items):
    with mp.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
        return pool.map(fn, items, chunksize=1)


# ----------------------------------------------------------------------------- checks
CHECK_ROWS = [("noise_1.0", "dump"), ("all_n0.5_J0.38_b0.5", "fill"), ("tau_noise_5pct", "baseline"),
              ("b_x2", "fill_half")]


def check_zero_residual():
    conds = {n: c for _, cases in L10.GROUPS for n, c in cases}
    keys = ("err_deg", "E_mech_kJ", "t_reach_s", "overshoot_kJ", "oscillation_kJ")
    worst, n = 0.0, 0
    for name, variant in CHECK_ROWS:
        cond = conds[name]
        ref, out = ([[L11.rollout(make_ctrl(kind, arg, cond.get("tau_noise", 0.0)), env, *sc)
                      for sc in L10.SCEN["orig20"]]
                     for kind, arg in (("pid", E.BAL_GAINS), ("sac", ("sac_residual", None)))
                     for env in [L10.make_env(variant, **cond)]])
        for a, b in zip(ref, out):
            n += 1
            worst = max(worst, max(0.0 if (math.isnan(a[k]) and math.isnan(b[k])) else abs(a[k] - b[k]) for k in keys))
    assert worst == 0.0, worst
    log(f"check 1: residual SAC with a = 0 equals Sched11 balanced + fix 2b on {n} moves"
          f" ({len(CHECK_ROWS)} rows x 20 tuning scenarios), largest difference {worst}")


class _TrueInteg(SacCtrl):
    """Check 2 only: the error integral from the true float64 error, as BoomEnvRealRobust."""

    def update(self, obs, tau, th_min, th_max):
        L11.Sched11.update(self, obs, tau, th_min, th_max)
        e = self.env.target - self.env.th
        self.integ_obs = float(np.clip(self.integ_obs + e * self.dt, -E.INTEG_CLIP, E.INTEG_CLIP))


def check_env_path():
    """The plant gives the controller a float32 observation; BoomEnvRealRobust integrates the float64
    error. (a) with the env's integral the paths agree bit for bit; (b) the eval integral (measured
    float32 e) moves the result by the reported amount."""
    worst, n = {False: 0.0, True: 0.0}, 0
    for label, (_, cfg, _) in SAC_MODELS.items():
        model = load_model(model_path(label, 0))
        for variant in ("baseline", "dump"):
            m_tip, payload = L10.VARIANTS[variant]
            for th0, target in L10.SCEN["orig20"][:5]:
                ref = E.BoomEnvRealRobust(w_energy=W, residual=(cfg == "residual"), randomize=False,
                                          m_tip=m_tip, payload=payload)
                obs, _ = ref.reset(seed=0, options={"th0": th0, "target": target})
                trunc = False
                while not trunc:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, _, _, trunc, _ = ref.step(a)
                for true_integ in (True, False):
                    env = L10.make_env(variant)
                    ctrl = (_TrueInteg if true_integ else SacCtrl)(model, cfg == "residual")
                    ctrl.env = env
                    r = L11.rollout(ctrl, env, th0, target)
                    d = max(abs(ref.th - env.th), abs(ref.om - env.om), abs(ref.energy_J / 1e3 - r["E_mech_kJ"]))
                    worst[true_integ] = max(worst[true_integ], d)
                n += 1
    assert worst[True] == 0.0, worst
    assert worst[False] < 1e-3, worst
    log(f"check 2: eval path equals BoomEnvRealRobust stepped with model.predict on {n} moves"
        f" ({len(SAC_MODELS)} models, seed 0, baseline and dump, 5 tuning scenarios, train_noise off):"
        f" largest difference {worst[True]}"
        f" with the env's float64 error integral; {worst[False]:.1e} (rad, rad/s or kJ) with the integral"
        f" of the measured float32 e used in the evaluation")


def check_l11(rows):
    ref = {(r["controller"], r["condition"], r["variant"]): r for r in L11.read_csv(L11_CSV)
           if r["estimator"] == E.EST_NAME and r["set"] == SET}
    keys = ("mean_E_mech_kJ", "share_over_20cm", "mean_t_reach_s", "max_deg", "mean_floor_kJ",
            "pct_above_floor", "mean_overshoot_kJ", "mean_oscillation_kJ", "m_hat_over_m_std")
    worst, n = 0.0, 0
    for r in rows:
        if r["controller"] in ROLES:
            o = ref[(r["controller"], r["condition"], r["variant"])]
            worst, n = max([worst] + [abs(r[k] - o[k]) for k in keys]), n + 1
    assert n == 3 * 72 and worst < 1e-9, (n, worst)
    log(f"check 3: {n} PID rows equal the fix 2b rows of {L11_CSV}, largest difference {worst:.1e}")


# ----------------------------------------------------------------------------- report
MEAN_KEYS = ("share_over_20cm", "mean_E_mech_kJ", "mean_floor_kJ", "mean_t_reach_s", "mean_overshoot_kJ",
             "mean_oscillation_kJ", "mean_direct_excess_kJ", "mean_shortfall_kJ", "share_flow_limit")
MAX_KEYS = ("max_cm", "max_t_reach_s", "n_never_reached")


def per_row(rows, label):
    """{(condition, variant): row}; SAC seeds merged: mean of means, worst of maxima."""
    by = {}
    for r in rows:
        if r["controller"] == label:
            by.setdefault((r["condition"], r["variant"]), []).append(r)
    out = {}
    for key, rs in by.items():
        m = {k: float(np.mean([r[k] for r in rs])) for k in MEAN_KEYS}
        m.update({k: float(np.max([r[k] for r in rs])) for k in MAX_KEYS})
        m["pct_above_floor"] = 100.0 * (m["mean_E_mech_kJ"] / m["mean_floor_kJ"] - 1.0)
        m["worst_seed_miss"] = float(np.max([r["share_over_20cm"] for r in rs]))
        m["sigma"] = float(rs[0]["sigma_th_deg"])
        m["n_seeds"] = len(rs)
        out[key] = m
    assert len(out) == 72, (label, len(out))
    return out


def summarise(cells):
    c = list(cells.values())
    miss = np.array([x["share_over_20cm"] for x in c]) * 100
    worst_seed = np.array([x["worst_seed_miss"] for x in c]) * 100
    e, fl = np.mean([x["mean_E_mech_kJ"] for x in c]), np.mean([x["mean_floor_kJ"] for x in c])
    none_base = cells[("none", "baseline")]
    return {"miss_mean": miss.mean(), "miss_max": miss.max(),
            "n_break": int((miss > 5).sum()), "n_degr": int(((miss > 1) & (miss <= 5)).sum()),
            "n_break_any_seed": int((worst_seed > 5).sum()),
            "E": e, "floor": fl, "gap": 100.0 * (e / fl - 1.0),
            "gap_lo": min(x["pct_above_floor"] for x in c), "gap_hi": max(x["pct_above_floor"] for x in c),
            "over": np.mean([x["mean_overshoot_kJ"] for x in c]),
            "osc": np.mean([x["mean_oscillation_kJ"] for x in c]),
            "dex": np.mean([x["mean_direct_excess_kJ"] for x in c]),
            "reach": np.nanmean([x["mean_t_reach_s"] for x in c]),
            "reach_max": np.nanmax([x["max_t_reach_s"] for x in c]),
            "never": int(sum(x["n_never_reached"] for x in c)),
            "max_cm": max(x["max_cm"] for x in c),
            "none_base_E": none_base["mean_E_mech_kJ"], "none_base_miss": none_base["share_over_20cm"] * 100,
            "breaks": [f"{k[0]} {k[1]} ({v['share_over_20cm'] * 100:.0f} %)" for k, v in cells.items()
                       if v["share_over_20cm"] > 0.05]}


def fnum(x, d=1):
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.{d}f}"


def read_rows():
    out = []
    for r in L11.read_csv(OUT_CSV):
        r["sac_seed"] = "" if r["sac_seed"] in ("", None) or (isinstance(r["sac_seed"], float) and math.isnan(r["sac_seed"])) else int(r["sac_seed"])
        out.append(r)
    return out


def report(rows):
    pts = dict(L11.load_points())
    present = {r["controller"] for r in rows}
    sac = [lab for lab in SAC_MODELS if lab in present]
    labels = list(ROLES) + sac
    name = {role: f"{role} PID (Kp {pts[role][0]:g}, Kd {pts[role][2]:g}) + fix 2b" for role in ROLES}
    name.update({lab: f"{SAC_NAME[lab]}, w {W:g}, mean of 3 seeds" for lab in sac})
    cells = {lab: per_row(rows, lab) for lab in labels}
    summ = {lab: summarise(cells[lab]) for lab in labels}
    seed_summ = {(lab, s): summarise(per_row([r for r in rows if r["sac_seed"] == s], lab))
                 for lab in sac for s in SEEDS}
    with open(TRAIN_LOG) as f:
        tlog = {(r["config"], int(r["seed"])): float(r["train_s"]) for r in csv.DictReader(f)
                if float(r["w"]) == W}
    n_new = int(rows[0]["n_scen"])
    out = []
    w = out.append
    w("# Lab 12 steps 2-3: SAC on the real plant against the Lab 11 PID points")
    w("")
    w("Issue #7. Generated by `python rl/12_eval.py` from `rl/12_eval.csv`. No training in this step.")
    w("")
    w("## Status of the values")
    w("")
    w("- All energies, times and miss rates are simulation results on `rl/boom_env_real.py`: **ESTIMATED**.")
    w("- Energy floor and its split: the Lab 11 definitions (`rl/11_real_plant_tuning.md`, section 1): **ESTIMATED**.")
    w("- SAC settings (100k steps, SB3 defaults, w = 0.1, randomization ranges of `rl/boom_env_real_robust.py`): **ASSUMED**.")
    w("- 20 cm miss bound, BREAKS > 5 % and degraded 1-5 %: Lab 10/11 convention, **ASSUMED**.")
    w("")
    w("## Protocol")
    w("")
    w(f"- Exactly the Lab 11 evaluation (`--step eval`): the {n_new} unseen scenarios, all 72 rows"
      " (9 single conditions + 9 combined conditions x 4 payload variants), 10 s per move, 10 substeps,"
      " noise seed 0 (the Lab 10 noise streams), Lab 11 rollout and floor.")
    w("- PID: the three approved Lab 11 points with fix 2b (adaptive m_hat filter, slow T_f = 1 s, c = 1, n_hold = 2).")
    w("- SAC: " + ", ".join(f"`rl/models/{model_name(lab, '{0,1,2}')}.zip`" for lab in sac)
      + ", deterministic action (`model.predict`). Residual: u = clip(u_balanced + a, -1, 1) on balanced"
      " + fix 2b. Plain: u = a.")
    w("- SAC observation [theta, omega, e, integral of e] from the measured signals: on the noise rows the"
      " policy sees the same noisy theta / omega as the PID. The `12_*` models were trained without sensor"
      " noise; the `12n_*` models (step 3) with sensor noise drawn per episode from {0, 0.2, 0.5, 1.0} deg"
      " (Lab 10 noise model). No model was trained with torque-reading noise.")
    w("- Seeds: per row, mean metrics (miss %, E_mech, floor, reach time, split) are the mean over the 3 seeds;"
      " max metrics (worst error, slowest reach, moves never reached) are the worst seed.")
    w("- Gap to floor = 100 x (mean E_mech over the 72 rows / mean floor over the 72 rows - 1).")
    w("")
    w("Checks (output of this run):")
    w("")
    for line in CHECK_LOG:
        w(f"- `{line}`")
    w("")
    w(f"## 1. All 72 rows, {n_new} unseen scenarios")
    w("")
    head = ["controller", "miss % mean", "worst row miss %", "rows > 5 % (break)", "rows 1-5 % (degraded)",
            "E_mech kJ", "gap to floor %", "row gap % range", "reach 20 cm s", "slowest reach s",
            "never reached", "worst error cm"]
    w("| " + " | ".join(head) + " |")
    w("|---|" + "---:|" * (len(head) - 1))
    for lab in labels:
        s = summ[lab]
        brk = f"{s['n_break']}" + (f" ({s['n_break_any_seed']} in any seed)" if lab.startswith("sac") else "")
        w(f"| {name[lab]} | {s['miss_mean']:.2f} | {s['miss_max']:.0f} | {brk} | {s['n_degr']} | {s['E']:.1f} |"
          f" {s['gap']:.1f} | {s['gap_lo']:.1f} to {s['gap_hi']:.1f} | {fnum(s['reach'], 2)} | {fnum(s['reach_max'], 2)} |"
          f" {s['never']} | {s['max_cm']:.1f} |")
    w("")
    w(f"Mean floor over the 72 rows: {summ['balanced']['floor']:.1f} kJ (same moves for every controller).")
    w("")
    w("Split of the excess above the floor, mean over the 72 rows [kJ]:")
    w("")
    w("| controller | overshoot | oscillation | direct excess | none / baseline E_mech kJ | none / baseline miss % |")
    w("|---|---:|---:|---:|---:|---:|")
    for lab in labels:
        s = summ[lab]
        w(f"| {name[lab]} | {s['over']:.1f} | {s['osc']:.1f} | {s['dex']:+.1f} | {s['none_base_E']:.1f} | {s['none_base_miss']:.0f} |")
    w("")
    for lab in labels:
        b = sorted(summ[lab]["breaks"], key=lambda t: -float(t.rsplit("(", 1)[1].split()[0]))
        txt = "; ".join(b) if len(b) <= 8 else f"{len(b)} rows, worst five: " + "; ".join(b[:5])
        w(f"- Rows over 5 % miss, {name[lab]}: {txt or 'none'}.")
    w("")
    w("## 2. With and without sensor noise")
    w("")
    w("Rows grouped by the angle-sensor noise level (0 deg: 9 conditions x 4 variants = 36 rows, the others"
      " 3 conditions x 4 variants = 12 rows each). Cells: gap to floor % | miss % mean | reach 20 cm s.")
    w("")
    levels = sorted({v["sigma"] for v in cells["balanced"].values()})
    w("| controller | " + " | ".join(f"noise {lv:g} deg" for lv in levels) + " |")
    w("|---|" + "---:|" * len(levels))
    for lab in labels:
        parts = []
        for lv in levels:
            c = [v for v in cells[lab].values() if v["sigma"] == lv]
            gap = 100.0 * (np.mean([x["mean_E_mech_kJ"] for x in c]) / np.mean([x["mean_floor_kJ"] for x in c]) - 1)
            parts.append(f"{gap:.1f} \\| {np.mean([x['share_over_20cm'] for x in c]) * 100:.2f} \\|"
                         f" {np.nanmean([x['mean_t_reach_s'] for x in c]):.2f}")
        w(f"| {name[lab]} | " + " | ".join(parts) + " |")
    w("")
    w("## 3. Each SAC seed")
    w("")
    w("| model | train s (log) | miss % mean | worst row miss % | break | degraded | E_mech kJ | gap % |"
      " reach 20 cm s | never reached | worst error cm |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lab in sac:
        for s in SEEDS:
            x = seed_summ[(lab, s)]
            t = tlog.get((SAC_MODELS[lab][2], s), float("nan"))
            w(f"| {model_name(lab, s)} | {t:.0f} | {x['miss_mean']:.2f} | {x['miss_max']:.0f} |"
              f" {x['n_break']} | {x['n_degr']} | {x['E']:.1f} | {x['gap']:.1f} | {fnum(x['reach'], 2)} |"
              f" {x['never']} | {x['max_cm']:.1f} |")
    w("")
    w("## 4. Reading")
    w("")
    b = summ["balanced"]
    for lab in sac:
        s = summ[lab]
        w(f"- {SAC_NAME[lab]} against balanced PID + fix 2b: gap {s['gap']:.1f} % vs {b['gap']:.1f} %"
          f" ({s['gap'] - b['gap']:+.1f} points), mean E_mech {s['E'] - b['E']:+.1f} kJ, reach"
          f" {fnum(s['reach'], 2)} s vs {fnum(b['reach'], 2)} s, miss {s['miss_mean']:.2f} % vs {b['miss_mean']:.2f} %,"
          f" {s['n_break']} break / {s['n_degr']} degraded rows vs {b['n_break']} / {b['n_degr']}.")
    w("")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {MD_PATH}")
    return summ


CHECK_LOG = []


def log(msg):
    print(msg)
    CHECK_LOG.append(msg)


def main():
    if "--report" in sys.argv:                     # keep the check lines of the evaluation run
        with open(MD_PATH) as f:
            CHECK_LOG.extend(ln[3:-1] for ln in f.read().splitlines()
                             if ln.startswith(("- `check", "- `evaluation")))
        report(read_rows())
        return
    for lab in SAC_MODELS:
        for s in SEEDS:
            assert os.path.exists(model_path(lab, s)), f"missing {model_path(lab, s)}: train it first"
    check_zero_residual()
    check_env_path()
    if "--check" in sys.argv:
        return
    t0 = time.time()
    rows = pool_map(run_row, jobs())
    log(f"evaluation: {len(rows)} rows x {rows[0]['n_scen']} unseen moves in {time.time() - t0:.0f} s")
    check_l11(rows)
    L11.write_csv(OUT_CSV, rows)
    report(read_rows())


if __name__ == "__main__":
    main()
