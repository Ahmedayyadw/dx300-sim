"""Lab 13c and 13d: benchmark of the learning methods with the Lab 12e protocol. No training.

Protocol: rl/12_eval.py and rl/12e_stats_eval.py, imported unchanged. 100 unseen scenarios x 72
rows = 7,200 moves per controller and seed, 10 substeps, Lab 11 rollout, deterministic action.

Controllers
  from existing files, not re-run (13c): economical and balanced PID and SAC res_bal_w0.03 from
      rl/12e_moves.csv.gz; SAC training time from rl/12_robust_train_log.csv.
  new: PPO 100k and 1M, TD3 100k (rl/13_online_train.py); BC and IQL at 100k / 300k / 1M
      transitions and CQL at 1M (rl/13_offline_train.py). 3 seeds each, all residual on the
      balanced PID: u = clip(u_balanced + a, -1, 1), observation [theta, omega, e, int(e) dt].
LogCtrl below is SacCtrl of rl/12_eval.py plus a record of the applied command, so every policy
runs through the Lab 12 controller code.

13d needs mean |du/dt|, which Lab 12 did not log. For that table only, the two PID points and the
three SAC models are rolled out again with LogCtrl; their error and energy must equal
rl/12e_moves.csv.gz (asserted, reported).

Checks (20 tuning scenarios only, before the evaluation):
  1  LogCtrl with residual 0 equals Sched11 balanced bit for bit.
  2  every new policy type (seed 0): the evaluation path equals stepping BoomEnvRealRobust
     (randomize=False) with the same policy, up to the float32 error integral (< 1e-3, as check 2
     of rl/12_eval.py).

Usage (repo root, .venv active):
  python rl/13_eval.py            checks, evaluation (about 1 h on 16 cores), report
  python rl/13_eval.py --report   report only, from rl/13_moves.csv.gz
  python rl/13_eval.py --check    checks only
  python rl/13_eval.py --smoke    *_smoke models, 2 rows of the tuning scenarios, files under
                                  rl/models/13_smoke_* (gitignored). No unseen scenario is used.
  python rl/13_eval.py --dir      checks, evaluation with the direction split; writes only
                                  rl/13_moves_dir.csv.gz (no report, other files untouched)
Writes rl/13_moves.csv.gz, rl/13_benchmark.csv, rl/13_benchmark.md.
"""

import collections
import csv
import gzip
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab13_common", os.path.join(HERE, "13_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab13_common"] = C
_spec.loader.exec_module(C)
S = C.load("lab12e_stats_eval", "12e_stats_eval.py")  # imports only; main guarded
M = S.M
E, L10, L11 = M.E, M.L10, M.L11

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEEDS = (0,) if SMOKE else C.SEEDS
MOVES = f"{C.MODEL_DIR}/13_smoke_moves.csv.gz" if SMOKE else "rl/13_moves.csv.gz"
CSV_PATH = f"{C.MODEL_DIR}/13_smoke_benchmark.csv" if SMOKE else "rl/13_benchmark.csv"
MD_PATH = f"{C.MODEL_DIR}/13_smoke_benchmark.md" if SMOKE else "rl/13_benchmark.md"
DIR = "--dir" in sys.argv  # direction split of E_mech, see S.dir_kJ
MOVES_DIR = f"{C.MODEL_DIR}/13_smoke_moves_dir.csv.gz" if SMOKE else "rl/13_moves_dir.csv.gz"
DATA_CSV = "rl/13_dataset.csv"
REF, SAC12 = "economical", "res_bal_w0.03"
REROLL = ("economical", "balanced", SAC12)  # 13d only
OFFLINE_SIZES = {"bc": ("100k", "300k", "1M"), "iql": ("100k", "300k", "1M"), "cql": ("1M",)}
NEW = {  # label: (group, policy kind, algo, tag)
    "ppo_100k": ("online on-policy", "ppo", "ppo", "100k"),
    "ppo_1M": ("online on-policy", "ppo", "ppo", "1M"),
    "td3_100k": ("online off-policy", "td3", "td3", "100k"),
    **{f"{a}_{s}": ("offline", "script", a, s) for a, sizes in OFFLINE_SIZES.items() for s in sizes},
}
if SMOKE:
    NEW = {k: v for k, v in NEW.items() if k in ("ppo_100k", "td3_100k", "bc_1M", "iql_1M", "cql_1M")}
GROUP = {"economical": "classical", "balanced": "classical", SAC12: "online off-policy",
         **{k: v[0] for k, v in NEW.items()}}
NAME = {"economical": "economical PID", "balanced": "balanced PID", SAC12: "SAC 100k (Lab 12)",
        "ppo_100k": "PPO 100k", "ppo_1M": "PPO 1M (not equal budget)", "td3_100k": "TD3 100k",
        **{f"{a}_{s}": f"{a.upper()}, {s} transitions" for a, sizes in OFFLINE_SIZES.items() for s in sizes}}
MAIN = ("economical", "balanced", "ppo_100k", "ppo_1M", SAC12, "td3_100k", "bc_1M", "cql_1M", "iql_1M")
THRESHOLDS_CM = (10, 20, 30)
EXTRA = ("sigma_deg", "du_dt", "overshoot_kJ", "oscillation_kJ", "direct_excess_kJ")
FIELDS = S.FIELDS + EXTRA
NUM = ("err_cm", "E_mech_kJ", "E_pump_kJ", "floor_kJ") + EXTRA


class LogCtrl(M.SacCtrl):
    """SacCtrl of Lab 12 that also sums |u_k - u_(k-1)| of the applied command over the move."""

    def reset(self):
        super().reset()
        self.du, self.n_du, self.last = 0.0, 0, None

    def act(self, obs):
        u = super().act(obs)
        if self.last is not None:
            self.du += abs(u - self.last)
            self.n_du += 1
        self.last = u
        return u


def policy_name(label, seed):
    _, _, algo, tag = NEW[label]
    return C.model_name(algo, tag, seed, SUFFIX)


def make_ctrl(label, seed, tau_noise):
    if label in ("economical", "balanced"):
        return LogCtrl(None, True, gains=E.point_gains(label), tau_noise=tau_noise)
    if label == SAC12:
        return LogCtrl(M.load_model(S.model_path(label, seed)), True, tau_noise=tau_noise)
    return LogCtrl(C.load_policy(NEW[label][1], policy_name(label, seed)), True, tau_noise=tau_noise)


def run_row(job):
    """One controller (and seed) on one row: per-move results. Worker."""
    label, seed, set_name, name, cond, variant = job
    env = L10.make_env(variant, **cond)
    ctrl = make_ctrl(label, seed, cond.get("tau_noise", 0.0))
    out = []
    for i, sc in enumerate(L11.scenarios(set_name)):
        r = L11.rollout(ctrl, env, *sc)
        out.append({"controller": label, "seed": "" if seed is None else seed, "condition": name,
                    "variant": variant, "scen": i, "err_cm": float(M.cm(r["err_deg"])),
                    "E_mech_kJ": r["E_mech_kJ"], "E_pump_kJ": r["E_pump_kJ"],
                    "floor_kJ": r["floor_kJ"], "t_reach_s": r["t_reach_s"],
                    "sigma_deg": cond.get("sigma", 0.0),
                    "du_dt": ctrl.du / max(1, ctrl.n_du) / ctrl.dt,
                    "overshoot_kJ": r["overshoot_kJ"], "oscillation_kJ": r["oscillation_kJ"],
                    "direct_excess_kJ": r["direct_excess_kJ"]})
        if DIR:
            out[-1].update(S.dir_kJ(env, r))
    return out


def jobs():
    rows = [(n, c, v) for _, n, c, v in (M.ROWS72[:2] if SMOKE else M.ROWS72)]
    set_name = "orig20" if SMOKE else M.SET
    out = [(p, None, set_name, n, c, v) for p in REROLL[:2] for n, c, v in rows]
    if not SMOKE:
        out += [(SAC12, s, set_name, n, c, v) for s in SEEDS for n, c, v in rows]
    out += [(lab, s, set_name, n, c, v) for lab in NEW for s in SEEDS for n, c, v in rows]
    return out


def missing_models():
    out = [S.model_path(SAC12, s) for s in SEEDS if not SMOKE and not os.path.exists(S.model_path(SAC12, s))]
    for lab in NEW:
        for s in SEEDS:
            p = C.policy_file(NEW[lab][1], policy_name(lab, s))
            if not os.path.exists(p):
                out.append(p)
    return out


# ----------------------------------------------------------------------------- checks
CHECK_LOG = []


def log(msg):
    print(msg)
    CHECK_LOG.append(msg)


def check_zero_residual():
    conds = {n: c for _, cases in L10.GROUPS for n, c in cases}
    worst, n = 0.0, 0
    for name, variant in M.CHECK_ROWS:
        cond = conds[name]
        tn = cond.get("tau_noise", 0.0)
        res = []
        for ctrl in (L11.Sched11(E.BAL_GAINS, E.EST, tau_noise=tn), make_ctrl("balanced", None, tn)):
            env = L10.make_env(variant, **cond)
            res.append([L11.rollout(ctrl, env, *sc) for sc in L10.SCEN["orig20"]])
        for a, b in zip(*res):
            n += 1
            worst = max(worst, abs(a["err_deg"] - b["err_deg"]), abs(a["E_mech_kJ"] - b["E_mech_kJ"]))
    assert worst == 0.0, worst
    log(f"check 1: LogCtrl with residual 0 equals Sched11 balanced + fix 2b on {n} moves"
        f" ({len(M.CHECK_ROWS)} rows x 20 tuning scenarios), largest difference {worst}")


def check_env_path():
    worst, n, kinds = 0.0, 0, []
    for lab in NEW:
        kind = NEW[lab][1]
        if kind in kinds:
            continue
        kinds.append(kind)
        pol = C.load_policy(kind, policy_name(lab, 0))
        for variant in ("baseline", "dump"):
            m_tip, payload = L10.VARIANTS[variant]
            for th0, target in L10.SCEN["orig20"][:3]:
                ref = E.BoomEnvRealRobust(w_energy=C.W, residual=True, randomize=False,
                                          m_tip=m_tip, payload=payload)
                obs, _ = ref.reset(seed=0, options={"th0": th0, "target": target})
                trunc = False
                while not trunc:
                    obs, _, _, trunc, _ = ref.step(pol.predict(obs, deterministic=True)[0])
                env = L10.make_env(variant)
                r = L11.rollout(make_ctrl(lab, 0, 0.0), env, th0, target)
                worst = max(worst, abs(ref.th - env.th), abs(ref.om - env.om),
                            abs(ref.energy_J / 1e3 - r["E_mech_kJ"]))
                n += 1
    assert worst < 1e-3, worst
    log(f"check 2: evaluation path equals BoomEnvRealRobust stepped with the same policy on {n}"
        f" moves (policy types {', '.join(kinds)}; seed 0, baseline and dump, 3 tuning scenarios):"
        f" largest difference {worst:.1e} (rad, rad/s or kJ)")


# ----------------------------------------------------------------------------- data
def read_moves(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def arrays(moves):
    """controller -> seed -> {"keys": [...], field: array}, sorted by row and scenario."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in moves:
        by[m["controller"]][str(m["seed"])].append(m)
    out = {}
    for ctrl, seeds in by.items():
        out[ctrl] = {}
        for seed, ms in seeds.items():
            ms.sort(key=lambda m: (m["condition"], m["variant"], int(m["scen"])))
            d = {"keys": [(m["condition"], m["variant"], str(m["scen"])) for m in ms]}
            for k in NUM:
                if k in ms[0]:
                    d[k] = np.array([float(m[k]) for m in ms])
            out[ctrl][seed] = d
    return out


def mstd(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else float("nan"))


def fmt(ms, f="{:.2f}"):
    return f.format(ms[0]) + ("" if np.isnan(ms[1]) else " +- " + f.format(ms[1]))


def train_hours():
    """label -> mean wall-clock per model [h]."""
    t = collections.defaultdict(list)
    if os.path.exists(C.TRAIN_LOG):
        for r in csv.DictReader(open(C.TRAIN_LOG)):
            t[f"{r['algo']}_{r['tag']}"].append(float(r["train_s"]))
    for r in csv.DictReader(open(S.TRAIN_LOG)):
        if r["config"] == "residual" and float(r["w"]) == C.W:
            t[SAC12].append(float(r["train_s"]))
    return {k: float(np.mean(v)) / 3600 for k, v in t.items()}


def summarise(A12, A13):
    """One dict per controller: the 12e metrics, the Wilcoxon test and the 13d columns."""
    from scipy.stats import wilcoxon

    ref = A12[REF][""] if not SMOKE else A13[REF][""]
    hours = train_hours()
    out = {}
    for lab in [k for k in NAME if k in A13 or k in A12]:
        src = A12 if (lab in REROLL and not SMOKE) else A13
        per = list(src[lab].values())
        row = {"controller": lab, "group": GROUP[lab], "seeds": len(per),
               "source": "rl/12e_moves.csv.gz" if src is A12 else MOVES}
        for t in THRESHOLDS_CM:
            row[f"miss{t}"] = mstd([100 * (p["err_cm"] > t).mean() for p in per])
        row["mean_cm"] = mstd([p["err_cm"].mean() for p in per])
        row["p95_cm"] = mstd([np.percentile(p["err_cm"], 95) for p in per])
        row["E_mech"] = mstd([p["E_mech_kJ"].mean() for p in per])
        row["E_pump"] = mstd([p["E_pump_kJ"].mean() for p in per])
        row["gap"] = mstd([100 * (p["E_mech_kJ"].mean() / p["floor_kJ"].mean() - 1) for p in per])
        row["train_h"] = hours.get(lab, float("nan"))
        if lab != REF:
            assert all(p["keys"] == ref["keys"] for p in per), f"{lab}: moves not aligned with {REF}"
            err = np.mean([p["err_cm"] for p in per], axis=0)
            en = np.mean([p["E_mech_kJ"] for p in per], axis=0)
            de, dE = err - ref["err_cm"], en - ref["E_mech_kJ"]
            row["d_err"], row["d_E"] = float(np.median(de)), float(np.median(dE))
            row["p_err"] = float(wilcoxon(err, ref["err_cm"]).pvalue) if np.any(de) else 1.0
            row["p_E"] = float(wilcoxon(en, ref["E_mech_kJ"]).pvalue) if np.any(dE) else 1.0
        if lab in A13:  # 13d
            d = list(A13[lab].values())
            cat = {k: np.concatenate([p[k] for p in d]) for k in NUM}
            excess = cat["E_mech_kJ"].mean() - cat["floor_kJ"].mean()
            row["du_dt"] = float(cat["du_dt"].mean())
            row["excess_kJ"] = float(excess)
            for k in ("oscillation", "overshoot", "direct_excess"):
                row[f"{k}_share"] = 100 * float(cat[f"{k}_kJ"].mean()) / excess
            for lv in sorted(set(cat["sigma_deg"])):
                row[f"err_n{lv:g}"] = float(cat["err_cm"][cat["sigma_deg"] == lv].mean())
        out[lab] = row
    eco = out[REF]
    for lab, row in out.items():
        if lab != REF:
            row["beats"] = bool(row["miss20"][0] <= eco["miss20"][0] and row["E_mech"][0] < eco["E_mech"][0]
                                and row["d_E"] < 0 and row["p_E"] < 0.05
                                and not (row["d_err"] > 0 and row["p_err"] < 0.05))
    return out


def check_reroll(A12, A13):
    worst = 0.0
    for lab in REROLL:
        for seed, a in A12[lab].items():
            b = A13[lab][seed]
            assert a["keys"] == b["keys"], lab
            worst = max(worst, float(np.abs(a["err_cm"] - b["err_cm"]).max()),
                        float(np.abs(a["E_mech_kJ"] - b["E_mech_kJ"]).max()))
    assert worst < 1e-9, worst
    return (f"re-rolled PID points and SAC {SAC12} (13d only) equal rl/12e_moves.csv.gz on error"
            f" and E_mech, 5 x 7200 moves, largest difference {worst:.1e}")


# ----------------------------------------------------------------------------- report
def write_csv(summ):
    keys = []
    flat = []
    for row in summ.values():
        f = {}
        for k, v in row.items():
            if isinstance(v, tuple):
                f[k + "_mean"], f[k + "_std"] = v
            else:
                f[k] = v
        flat.append(f)
        keys += [k for k in f if k not in keys]
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def pval(p):
    return "<1e-300" if p == 0 else f"{p:.1e}"  # 0 = below the float range of scipy


def main_table(summ, labels):
    head = ["group", "controller", "miss > 10 cm %", "miss > 20 cm %", "miss > 30 cm %", "mean cm",
            "p95 cm", "E_mech kJ", "E_pump kJ (lower bound, ASSUMED)", "gap to floor %",
            "median diff error cm (p)", "median diff E_mech kJ (p)",
            "train h per model (wall-clock, shared CPU)", "beats economical PID"]
    out = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 3) + "---|"]
    for lab in labels:
        if lab not in summ:
            continue
        r = summ[lab]
        wil = ("-", "-") if lab == REF else (f"{r['d_err']:+.2f} ({pval(r['p_err'])})",
                                             f"{r['d_E']:+.2f} ({pval(r['p_E'])})")
        th = "-" if np.isnan(r["train_h"]) else f"{r['train_h']:.2f}"
        out.append(f"| {r['group']} | {NAME[lab]} | " + " | ".join(fmt(r[f'miss{t}']) for t in THRESHOLDS_CM)
                   + f" | {fmt(r['mean_cm'], '{:.1f}')} | {fmt(r['p95_cm'], '{:.1f}')} |"
                   f" {fmt(r['E_mech'], '{:.1f}')} | {fmt(r['E_pump'], '{:.1f}')} | {fmt(r['gap'], '{:.1f}')} |"
                   f" {wil[0]} | {wil[1]} | {th} | {'-' if lab == REF else ('yes' if r['beats'] else 'no')} |")
    return out


def report(summ, reroll_line):
    w = []
    a = w.append
    a("# Lab 13: benchmark of learning methods against the economical PID")
    a("")
    a("Generated by `python rl/13_eval.py`. Lab 12e protocol: 100 unseen scenarios x 72 rows = 7200")
    a("moves per controller and seed, 10 substeps. Learned controllers: residual on the balanced PID,")
    a("w = 0.03, 3 seeds, mean +- std over seeds of the per-seed results.")
    a("")
    a("## Status of the values")
    a("")
    a("- Every error, energy, miss rate and time below is a simulation or wall-clock result: **ESTIMATED**.")
    a("- E_pump is a lower bound from the valve pressure drop and pump efficiency of Lab 09b: **ASSUMED** model.")
    a("- Budgets, dataset noise 0.1, 50/50 mix, the rule for 'beats' and all hyperparameters: **ASSUMED**.")
    a("- Boom length, pump flow, relief pressure: **OFFICIAL** (read from the parameter files).")
    a("")
    a("## Setup and changes from the library defaults")
    a("")
    a("- Env of every learner: `BoomEnvRealRobust(w_energy=0.03, residual=True)`, the env of the Lab 12 SAC.")
    a("- TD3 (SB3): `action_noise = NormalActionNoise(0, 0.1)`. Nothing else changed.")
    a("- PPO (SB3): `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`, statistics frozen at"
      " evaluation. Nothing else changed. A 100k run stops at 100,352 steps (2,048 per update).")
    a("- BC, CQL, IQL (d3rlpy, run in `.venv-d3rlpy`): library defaults, 100,000 gradient steps, final model.")
    a("  CQL was trained at 1M transitions only (run time).")
    for algo in OFFLINE_SIZES:
        p = f"{C.MODEL_DIR}/{C.model_name(algo, '1M', 0, SUFFIX)}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            cfg = {k: v for k, v in j["config"].items() if not isinstance(v, (dict, list)) and v is not None}
            a(f"  - {algo.upper()} config (d3rlpy {j['d3rlpy']}): `{cfg}`")
    a("- Rows from existing files, not re-run: economical PID, balanced PID and SAC from"
      " `rl/12e_moves.csv.gz`; SAC training time from `rl/12_robust_train_log.csv`.")
    a("")
    a("Checks (output of this run):")
    a("")
    for line in CHECK_LOG + ([reroll_line] if reroll_line else []):
        a(f"- `{line}`")
    a("")
    if os.path.exists(DATA_CSV):
        a("## Offline dataset")
        a("")
        a("PID with Gaussian noise (sigma 0.1) on the command, training distribution only (`rl/13_dataset.py`).")
        a("")
        rows = list(csv.DictReader(open(DATA_CSV)))
        a("| " + " | ".join(rows[0]) + " |")
        a("|---|---|" + "---:|" * (len(rows[0]) - 2))
        for r in rows:
            a("| " + " | ".join(v if k in ("subset", "behaviour") else f"{float(v):.3g}" for k, v in r.items()) + " |")
        a("")
    a("## 13c. All controllers")
    a("")
    a("Wilcoxon signed-rank test paired per move against the economical PID (7200 pairs, learned")
    a("controllers = mean of their seeds per move). Positive difference = worse than the PID.")
    a("'Beats' = no more misses over 20 cm, lower mean E_mech with p < 0.05, and no significantly larger error.")
    a("")
    w += main_table(summ, MAIN)
    a("")
    p1, p2 = summ.get("ppo_100k"), summ.get("ppo_1M")
    a("Training time is the mean wall-clock per model while other training runs shared the machine. It is"
      " not a compute cost and not comparable between rows"
      + (f": PPO 100k took {p1['train_h']:.2f} h for 100,352 steps, PPO 1M {p2['train_h']:.2f} h for"
         " 1,001,472 steps (`rl/13_train_log.csv`)." if p1 and p2 else "."))
    a("")
    a("## Dataset size (offline)")
    a("")
    sized = [f"{al}_{s}" for al, sizes in OFFLINE_SIZES.items() if len(sizes) > 1 for s in sizes]
    w += main_table(summ, sized)
    a("")
    off = [f"{al}_1M" for al in OFFLINE_SIZES if f"{al}_1M" in summ]
    best_off = min(off, key=lambda k: (summ[k]["miss20"][0], summ[k]["E_mech"][0]))
    al = best_off.split("_")[0]
    steps = [f"{al}_{s}" for s in OFFLINE_SIZES[al] if f"{al}_{s}" in summ]
    a(f"Best offline method at 1M transitions (fewest misses over 20 cm, then least energy): {al.upper()}."
      " With the dataset size: "
      + "; ".join(f"{k.split('_')[1]}: {summ[k]['miss10'][0]:.2f} % misses over 10 cm,"
                  f" {summ[k]['miss20'][0]:.2f} % over 20 cm, mean error {summ[k]['mean_cm'][0]:.1f} cm,"
                  f" {summ[k]['E_mech'][0]:.1f} kJ" for k in steps) + ".")
    a("")
    a("## 13d. Why the learned controllers lose")
    a("")
    a("mean abs(du/dt): mean of abs(u_k - u_(k-1)) / dt of the applied command, per move. Shares are of the")
    a("energy above the floor (E_mech - floor), Lab 11 split; they do not add to 100 % (shortfall term).")
    a("")
    levels = sorted(k for k in summ["balanced"] if k.startswith("err_n"))
    a("| controller | mean abs(du/dt) 1/s | excess kJ | oscillation % | overshoot % | direct excess % | "
      + " | ".join(f"error cm, noise {k[5:]} deg" for k in levels) + " |")
    a("|---|" + "---:|" * (5 + len(levels)))
    for lab in MAIN:
        if lab in summ and "du_dt" in summ[lab]:
            r = summ[lab]
            a(f"| {NAME[lab]} | {r['du_dt']:.2f} | {r['excess_kJ']:.1f} | {r['oscillation_share']:.0f} |"
              f" {r['overshoot_share']:.0f} | {r['direct_excess_share']:.0f} | "
              + " | ".join(f"{r[k]:.1f}" for k in levels) + " |")
    a("")
    a("Reading of the table (numbers of this run):")
    a("")
    eco, bal = summ[REF], summ["balanced"]
    n0, n1 = levels[0], levels[-1]
    online = [summ[k] for k in MAIN if k in summ and GROUP[k].startswith("online") and "du_dt" in summ[k]]
    if online:
        du = [r["du_dt"] for r in online]
        ex = [r["excess_kJ"] for r in online]
        osc = [r["oscillation_share"] for r in online]
        e0 = [r[n0] for r in online]
        drift = max(abs(r[n1] - r[n0]) for r in online)
        a(f"- Online learners (PPO, SAC, TD3) move the valve command {min(du) / eco['du_dt']:.0f} to"
          f" {max(du) / eco['du_dt']:.0f} times more than the economical PID (mean abs(du/dt) {min(du):.2f} to"
          f" {max(du):.2f} against {eco['du_dt']:.2f}). Their energy above the floor is {min(ex):.1f} to"
          f" {max(ex):.1f} kJ against {eco['excess_kJ']:.1f} kJ; {min(osc):.0f} to {max(osc):.0f} % of it is"
          " oscillation.")
        a(f"- Their error does not come from the sensor noise: without noise it is already {min(e0):.1f} to"
          f" {max(e0):.1f} cm (economical PID {eco[n0]:.1f} cm), and from {n0[5:]} to {n1[5:]} deg it changes by"
          f" {drift:.1f} cm at most (economical PID {eco[n1] - eco[n0]:+.1f} cm). The residual itself moves the"
          " boom off the target.")
    for k in ("bc_1M", "iql_1M"):
        if k in summ and "du_dt" in summ[k]:
            r = summ[k]
            between = eco["E_mech"][0] < r["E_mech"][0] < bal["E_mech"][0]
            a(f"- {NAME[k]}: mean abs(du/dt) {r['du_dt']:.2f}, the smoothness of the PID points"
              f" ({eco['du_dt']:.2f} and {bal['du_dt']:.2f}). E_mech {r['E_mech'][0]:.1f} kJ"
              + (f" lies between the economical ({eco['E_mech'][0]:.1f} kJ) and the balanced PID"
                 f" ({bal['E_mech'][0]:.1f} kJ), the two behaviour policies of the dataset."
                 if between else f" (economical PID {eco['E_mech'][0]:.1f} kJ, balanced PID {bal['E_mech'][0]:.1f} kJ)."))
    if "cql_1M" in summ and "du_dt" in summ["cql_1M"]:
        r = summ["cql_1M"]
        a(f"- {NAME['cql_1M']}: smooth command (mean abs(du/dt) {r['du_dt']:.2f}), but {r[n0]:.1f} cm error"
          f" without noise and {r['excess_kJ']:.1f} kJ above the floor, {r['overshoot_share']:.0f} % of it overshoot.")
    if os.path.exists(DATA_CSV):
        ret = {r["behaviour"]: float(r["mean_return"]) for r in csv.DictReader(open(DATA_CSV)) if r["subset"] == "1M"}
        a("- Reward: the training reward at w = 0.03 ranks the balanced PID above the economical PID. Return"
          " per episode -9.78 against -10.30 without noise (probe on 300 training-distribution episodes,"
          f" `docs/13_benchmark.md` section 3), {ret['balanced']:.2f} against {ret['economical']:.2f} in the 1M"
          " dataset with noise (table above; the two behaviours ran on different episodes). A learner that"
          " maximises this reward is not pushed toward the economical PID. The return of the learned"
          " controllers was not measured.")
    a("")
    a("## Statement")
    a("")
    learned = [k for k in summ if k not in ("economical", "balanced")]
    winners = [NAME[k] for k in learned if summ[k]["beats"]]
    eco = summ[REF]
    if winners:
        a(f"These learning methods beat the economical PID on the rule above: {', '.join(winners)}.")
    else:
        a(f"No learning method, online or offline, beats the economical PID ({eco['miss20'][0]:.2f} % misses"
          f" over 20 cm, {eco['E_mech'][0]:.1f} kJ, {eco['gap'][0]:.1f} % above the floor).")
    best = min(learned, key=lambda k: (summ[k]["miss20"][0], summ[k]["E_mech"][0]))
    b = summ[best]
    a(f"Closest learned controller (fewest misses, then least energy): {NAME[best]},"
      f" {b['miss20'][0]:.2f} % misses, {b['E_mech'][0]:.1f} kJ.")
    a("")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(w) + "\n")
    print("\n".join(w))
    print(f"wrote {MD_PATH}, {CSV_PATH}")


def make_report():
    A13 = arrays(read_moves(MOVES))
    A12 = {} if SMOKE else arrays(S.read_moves())
    line = None if SMOKE else check_reroll(A12, A13)
    summ = summarise(A12, A13)
    write_csv(summ)
    report(summ, line)


def main():
    if "--report" in sys.argv:
        if os.path.exists(MD_PATH):
            with open(MD_PATH) as f:
                CHECK_LOG.extend(ln[3:-1] for ln in f.read().splitlines()
                                 if ln.startswith(("- `check", "- `evaluation")))
        make_report()
        return
    miss = missing_models()
    if miss:
        sys.exit("missing models (train them first):\n  " + "\n  ".join(miss))
    check_zero_residual()
    check_env_path()
    if "--check" in sys.argv:
        return
    t0 = time.time()
    js = jobs()
    rows = [m for chunk in M.pool_map(run_row, js) for m in chunk]
    log(f"evaluation: {len(js)} rows, {len(rows)} moves in {time.time() - t0:.0f} s")
    if DIR:
        with gzip.open(MOVES_DIR, "wt", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS + S.DIR_FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {MOVES_DIR} ({len(rows)} moves)")
        return
    with gzip.open(MOVES, "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    make_report()


if __name__ == "__main__":
    main()
