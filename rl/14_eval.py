"""Lab 14e: evaluate the Lab 14 controllers with the Lab 13 protocol and compare with Lab 13. No training.

Protocol: rl/13_eval.py, imported unchanged (LogCtrl, Lab 11 rollout, Lab 12e rows, Wilcoxon test against
the economical PID, the 'beats' rule, table format). 100 unseen scenarios x 72 rows = 7,200 moves per
controller and seed, deterministic action.

Controllers
  Lab 13 rows, not re-run: economical / balanced PID and SAC (Lab 12) from rl/12e_moves.csv.gz, TD3, BC,
      IQL, CQL from rl/13_moves.csv.gz.
  Lab 14 (rl/14_online_train.py, rl/14_offline_train.py), all residual on the balanced PID:
      14b  SAC 100k, TD3 100k (corrected reward); IQL, CQL on the relabelled 1M dataset; 3 seeds
      14c  BC on the economical-PID episodes of the 1M dataset (500k transitions); 3 seeds
      14d  SAC and TD3 with learning rate x0.3 and x3 (corrected reward); seed 0 only

Return probe (20 tuning scenarios x 4 variants = 80 episodes, randomize off, no noise; the unseen scenarios
are not used): return under the corrected and the Lab 13 reward of the three PID points, the Lab 13 learners
and the Lab 14 learners. It tells whether a learner reached a higher training return than the economical PID.

Checks (tuning scenarios only, before the evaluation):
  1  LogCtrl with residual 0 equals Sched11 balanced (check 1 of rl/13_eval.py).
  2  per Lab 14 policy type (seed 0): evaluation path equals stepping BoomEnvRealRobust with the policy.
  3  Lab 13 rows recomputed from rl/13_moves.csv.gz equal rl/13_benchmark.csv.

Usage (repo root, .venv active, OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):
  python rl/14_eval.py              checks, evaluation (about 15 min on 14 processes), probe, report
  python rl/14_eval.py --report     report only, from rl/14_moves.csv.gz and rl/14_returns.csv
  python rl/14_eval.py --quick      checks, 2 rows on the tuning scenarios, seed 0, nothing written
Writes rl/14_moves.csv.gz, rl/14_returns.csv, rl/14_benchmark.csv, rl/14_reward_fix.md. The section on the\nreward loophole is added when rl/14_hack_check.csv exists (python rl/14_hack_check.py, then --report).
"""

import collections
import csv
import gzip
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab14_common", os.path.join(HERE, "14_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab14_common"] = C
_spec.loader.exec_module(C)
C13 = C.C13
E13 = C.load("lab13_eval", "13_eval.py")  # main guarded; reads --smoke / --dir only (not used here)
assert not E13.SMOKE and not E13.DIR
S, M = E13.S, E13.M
E, L10, L11 = M.E, M.L10, M.L11

QUICK = "--quick" in sys.argv
PROCS = 14
MOVES = "rl/14_moves.csv.gz"
RETURNS = "rl/14_returns.csv"
CSV_PATH = "rl/14_benchmark.csv"
MD_PATH = "rl/14_reward_fix.md"
REF, SAC12 = E13.REF, E13.SAC12

NEW14 = {  # label: (group, policy kind, algo, tag, seeds)
    "r14_sac_100k": ("online off-policy", "sac", "sac", "100k", C.SEEDS),
    "r14_td3_100k": ("online off-policy", "td3", "td3", "100k", C.SEEDS),
    "r14_iql_1M": ("offline", "script", "iql", "1M", C.SEEDS),
    "r14_cql_1M": ("offline", "script", "cql", "1M", C.SEEDS),
    "r14_bc_eco500k": ("offline", "script", "bc", "eco500k", C.SEEDS),
    "r14_sac_100k_lr0.3": ("online off-policy", "sac", "sac", "100k_lr0.3", (0,)),
    "r14_sac_100k_lr3": ("online off-policy", "sac", "sac", "100k_lr3", (0,)),
    "r14_td3_100k_lr0.3": ("online off-policy", "td3", "td3", "100k_lr0.3", (0,)),
    "r14_td3_100k_lr3": ("online off-policy", "td3", "td3", "100k_lr3", (0,)),
}
NAME14 = {
    "r14_sac_100k": "SAC 100k, corrected reward",
    "r14_td3_100k": "TD3 100k, corrected reward",
    "r14_iql_1M": "IQL 1M, corrected reward",
    "r14_cql_1M": "CQL 1M, corrected reward",
    "r14_bc_eco500k": "BC, economical-PID data only (500k)",
    "r14_sac_100k_lr0.3": "SAC 100k, corrected reward, lr x0.3 (1 seed)",
    "r14_sac_100k_lr3": "SAC 100k, corrected reward, lr x3 (1 seed)",
    "r14_td3_100k_lr0.3": "TD3 100k, corrected reward, lr x0.3 (1 seed)",
    "r14_td3_100k_lr3": "TD3 100k, corrected reward, lr x3 (1 seed)",
}
NAME13 = {SAC12: "SAC 100k (Lab 12), Lab 13 reward", "td3_100k": "TD3 100k, Lab 13 reward",
          "iql_1M": "IQL 1M, Lab 13 reward", "cql_1M": "CQL 1M, Lab 13 reward",
          "bc_1M": "BC 1M (both PIDs), Lab 13"}
PAIRS = [(SAC12, "r14_sac_100k"), ("td3_100k", "r14_td3_100k"), ("iql_1M", "r14_iql_1M"),
         ("cql_1M", "r14_cql_1M"), ("bc_1M", "r14_bc_eco500k")]
LR_ROWS = ["r14_sac_100k_lr0.3", "r14_sac_100k_lr3", "r14_td3_100k_lr0.3", "r14_td3_100k_lr3"]
TABLE = ["economical", "balanced"] + [x for p in PAIRS for x in p] + LR_ROWS
EXCESS_MIN_KJ = 5.0  # 13d shares are shown only above this excess energy
HACK = "rl/14_hack_check.csv"  # rl/14_hack_check.py
PROBE = {  # label: seeds (None = PID point, no model)
    "fast": (None,), "balanced": (None,), "economical": (None,),
    SAC12: C.SEEDS, "td3_100k": C.SEEDS, "iql_1M": C.SEEDS, "cql_1M": C.SEEDS, "bc_1M": C.SEEDS,
    **{k: v[4] for k, v in NEW14.items()},
}


# ----------------------------------------------------------------------------- policies
def model_name(label, seed):
    _, _, algo, tag, _ = NEW14[label]
    return C.model_name(algo, tag, seed)


def get_policy(label, seed):
    if label == SAC12:
        return M.load_model(S.model_path(SAC12, seed))
    if label in E13.NEW:
        return C13.load_policy(E13.NEW[label][1], E13.policy_name(label, seed))
    kind = NEW14[label][1]
    if kind == "sac":
        return M.load_model(f"{C.MODEL_DIR}/{model_name(label, seed)}.zip")
    return C13.load_policy(kind, model_name(label, seed))


def model_file(label, seed):
    kind = NEW14[label][1]
    return f"{C.MODEL_DIR}/{model_name(label, seed)}" + (".pt" if kind == "script" else ".zip")


def make_ctrl(label, seed, tau_noise):
    return E13.LogCtrl(get_policy(label, seed), True, tau_noise=tau_noise)


# ----------------------------------------------------------------------------- evaluation rows
def run_row(job):
    """One Lab 14 controller and seed on one row: per-move results (same fields as rl/13_eval.py)."""
    label, seed, set_name, name, cond, variant = job
    env = L10.make_env(variant, **cond)
    ctrl = make_ctrl(label, seed, cond.get("tau_noise", 0.0))
    out = []
    for i, sc in enumerate(L11.scenarios(set_name)):
        r = L11.rollout(ctrl, env, *sc)
        out.append({"controller": label, "seed": seed, "condition": name, "variant": variant, "scen": i,
                    "err_cm": float(M.cm(r["err_deg"])), "E_mech_kJ": r["E_mech_kJ"],
                    "E_pump_kJ": r["E_pump_kJ"], "floor_kJ": r["floor_kJ"], "t_reach_s": r["t_reach_s"],
                    "sigma_deg": cond.get("sigma", 0.0), "du_dt": ctrl.du / max(1, ctrl.n_du) / ctrl.dt,
                    "overshoot_kJ": r["overshoot_kJ"], "oscillation_kJ": r["oscillation_kJ"],
                    "direct_excess_kJ": r["direct_excess_kJ"]})
    return out


def jobs():
    rows = [(n, c, v) for _, n, c, v in (M.ROWS72[:2] if QUICK else M.ROWS72)]
    set_name = "orig20" if QUICK else M.SET
    out = []
    for lab, spec in NEW14.items():
        for s in ((0,) if QUICK else spec[4]):
            out += [(lab, s, set_name, n, c, v) for n, c, v in rows]
    return out


def pool_map(fn, items):
    with mp.get_context("spawn").Pool(PROCS) as pool:
        return pool.map(fn, items, chunksize=1)


# ----------------------------------------------------------------------------- return probe
def probe_episode(job):
    """Return of one controller on one tuning episode under the corrected and the Lab 13 reward."""
    label, seed, variant, k = job
    th0, target = L10.SCEN["orig20"][k][:2]
    m_tip, payload = L10.VARIANTS[variant]
    pid = seed is None
    env = C.make_env(randomize=False, m_tip=m_tip, payload=payload, base=label if pid else "balanced")
    obs, _ = env.reset(seed=0, options={"th0": th0, "target": target})
    pol = None if pid else get_policy(label, seed)
    g14 = g13 = 0.0
    trunc = False
    while not trunc:
        a = np.zeros(1, np.float32) if pid else pol.predict(obs, deterministic=True)[0]
        obs, r, _, trunc, info = env.step(a)
        g14 += r
        g13 += info["r_lab13"]
    return {"controller": label, "seed": "" if pid else seed, "variant": variant, "scen": k,
            "G_corrected": g14, "G_lab13": g13, "E_mech_kJ": env.energy_J / 1e3,
            "final_err_cm": abs(env.target - env.th) * 100 * E.R.L_BOOM_M}


def probe_jobs():
    n = 2 if QUICK else len(L10.SCEN["orig20"])
    return [(lab, s, v, k) for lab, seeds in PROBE.items() for s in ((seeds[0],) if QUICK else seeds)
            for v in L10.VARIANTS for k in range(n)]


# ----------------------------------------------------------------------------- checks
def check_env_path():
    worst, n, kinds = 0.0, 0, []
    for lab, spec in NEW14.items():
        if spec[1] in kinds:
            continue
        kinds.append(spec[1])
        pol = get_policy(lab, 0)
        for variant in ("baseline", "dump"):
            m_tip, payload = L10.VARIANTS[variant]
            for th0, target in L10.SCEN["orig20"][:3]:
                ref = C.make_env(randomize=False, m_tip=m_tip, payload=payload)
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
    E13.log(f"check 2: evaluation path equals the Lab 14 training env stepped with the same policy on {n}"
            f" moves (policy types {', '.join(kinds)}; seed 0, baseline and dump, 3 tuning scenarios):"
            f" largest difference {worst:.1e} (rad, rad/s or kJ)")


def check_lab13(summ):
    ref = {r["controller"]: r for r in csv.DictReader(open("rl/13_benchmark.csv"))}
    worst, n = 0.0, 0
    for lab in ["economical", "balanced"] + [p[0] for p in PAIRS]:
        for key in ("miss20", "mean_cm", "E_mech", "E_pump"):
            worst = max(worst, abs(summ[lab][key][0] - float(ref[lab][key + "_mean"])))
            n += 1
    assert worst < 1e-9, worst
    E13.log(f"check 3: Lab 13 rows recomputed from rl/13_moves.csv.gz and rl/12e_moves.csv.gz equal"
            f" rl/13_benchmark.csv ({n} values: miss > 20 cm, mean error, E_mech, E_pump), largest"
            f" difference {worst:.1e}")


# ----------------------------------------------------------------------------- report
def train_hours():
    t = collections.defaultdict(list)
    for r in csv.DictReader(open(C.TRAIN_LOG)):
        t[f"r14_{r['algo']}_{r['tag']}"].append(float(r["train_s"]))
    return {k: float(np.mean(v)) / 3600 for k, v in t.items()}


def probe_table(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["controller"]][r["seed"]].append(r)
    eco = sorted(by["economical"][""], key=lambda r: (r["variant"], int(r["scen"])))
    g_eco = np.array([float(r["G_corrected"]) for r in eco])
    out = {}
    for lab, seeds in by.items():
        per = []
        for s, rs in seeds.items():
            rs.sort(key=lambda r: (r["variant"], int(r["scen"])))
            assert [(r["variant"], r["scen"]) for r in rs] == [(r["variant"], r["scen"]) for r in eco]
            g = np.array([float(r["G_corrected"]) for r in rs])
            per.append({"g14": g.mean(), "g13": np.mean([float(r["G_lab13"]) for r in rs]),
                        "E": np.mean([float(r["E_mech_kJ"]) for r in rs]),
                        "err": np.mean([float(r["final_err_cm"]) for r in rs]),
                        "share": 100 * float(np.mean(g > g_eco))})
        out[lab] = {k: E13.mstd([p[k] for p in per]) for k in per[0]}
        out[lab]["seeds"] = len(per)
    return out


def hack_table():
    """Mean per controller of rl/14_hack_check.csv (all seeds and episodes), or {} if not run."""
    if not os.path.exists(HACK):
        return {}
    rows = list(csv.DictReader(open(HACK)))
    out = {}
    for r in rows:
        out.setdefault(r["controller"], []).append(r)
    keys = [k for k in rows[0] if k not in ("controller", "seed", "variant", "scen")]
    return {lab: {k: float(np.mean([float(r[k]) for r in rs])) for k in keys} for lab, rs in out.items()}


def report(summ, probe):
    NAME = E13.NAME
    w = []
    a = w.append
    a("# Lab 14: does a corrected reward close the gap to the economical PID?")
    a("")
    a("Generated by `python rl/14_eval.py`. Lab 13 protocol (`rl/13_eval.py`, imported unchanged): 100 unseen")
    a("scenarios x 72 rows = 7200 moves per controller and seed, 10 substeps, deterministic action. Every")
    a("learned controller is a residual on the balanced PID, as in Lab 13. Mean +- std over seeds.")
    a("")
    a("## Status of the values")
    a("")
    a("- Every error, energy, miss rate, return and time below is a simulation or wall-clock result: **ESTIMATED**.")
    a("- E_mech (delivered mechanical energy) is the main energy metric. E_pump is a lower bound from the valve")
    a("  pressure drop and pump efficiency of Lab 09b: **ASSUMED** model, shown only as a labelled lower bound.")
    a(f"- Reward constants (w = {C.W:g}, k = {C.K:g}, tolerance {C.TOL_CM:g} cm, stall speed {C.OM_S:g} rad/s,"
      f" scale 1/{C.SCALE:g}), budgets, hyperparameters and the 'beats' rule: **ASSUMED**.")
    a("- Boom length, pump flow, relief pressure: **OFFICIAL** (read from the parameter files).")
    a("")
    a("## 14a. Corrected reward")
    a("")
    a("Lab 13 reward per control step: `r = -(e^2 + 0.01 omega^2 + 0.03 P / P_REF)`. Corrected reward:")
    a("")
    a("```")
    a(f"r = -(e^2 + 0.01 omega^2 + {C.W:g} P / P_REF + {C.K:g} h) / {C.SCALE:g}")
    a(f"h = max(0, |e| - {C.TOL_CM:g} cm / L_boom)  if |omega| < {C.OM_S:g} rad/s,  else 0")
    a("```")
    a("")
    a("Chosen on the 20 tuning scenarios x 4 variants only (`rl/14_reward_probe.py`, `rl/14_reward_probe.csv`):")
    a("")
    a("- A larger energy weight alone cannot rank economical > balanced > fast: economical > balanced needs")
    a("  w > 2.42, but above w = 0.61 the economical PID told to stop 20 cm short of the target scores better")
    a("  than the one stopping 10 cm short, and above w = 0.95 better than the fast PID. The e^2 term charges")
    a("  mainly the approach, where the fast PID wins; a small error at the end costs almost nothing.")
    a("- The stall term h charges an error over the tolerance only while the boom is (nearly) stopped, so it")
    a("  hits stopping short and not a slow approach. With it the order holds for w >= 4.2 (stall speed")
    a("  0.01 rad/s); w = 20 is where the smallest margin stops growing. The scale 1/50 puts the economical")
    a("  return on the tuning set at the Lab 13 value.")
    a("- The margin is thin: on the tuning set the mean return of the economical PID is 0.7 % above the")
    a("  balanced one, and the order holds in 42 % of the single episodes.")
    a("")
    a("## Setup")
    a("")
    a("- Online (14b, 14d): `rl/14_online_train.py`, env of Lab 13 with the corrected reward, 100,000 steps.")
    a("  SAC: SB3 defaults. TD3: `NormalActionNoise(0, 0.1)` as Lab 13. 14d: learning rate x0.3 and x3 of the")
    a("  SB3 default (SAC 3e-4, TD3 1e-3), seed 0.")
    a("- Offline (14b): `rl/14_relabel.py` replays every episode of the Lab 13 1M dataset with its stored actions")
    a("  and writes the corrected reward; observations and actions unchanged, no new data. IQL and CQL:")
    a("  d3rlpy defaults, 100,000 gradient steps, final model, as Lab 13.")
    a("- 14c: BC (d3rlpy defaults) on the 2,500 economical-PID + noise episodes of the same dataset (500k")
    a("  transitions). BC ignores the reward: this tests the data, not the reward.")
    a("- Difference from Lab 13: the offline runs used 1 torch thread (Lab 13: 4); every run had")
    a("  OMP_NUM_THREADS=1, 14 runs at a time. The reward scale changes the balance between reward and the")
    a("  entropy bonus of SAC (automatic entropy tuning kept).")
    a("")
    a("Checks (output of this run):")
    a("")
    for line in E13.CHECK_LOG:
        a(f"- `{line}`")
    a("")
    a("## 14e. All controllers against the economical PID")
    a("")
    a("Wilcoxon signed-rank test paired per move against the economical PID (7200 pairs, learned controllers")
    a("= mean of their seeds per move). Positive difference = worse than the PID. 'Beats' = no more misses over")
    a("20 cm, lower mean E_mech with p < 0.05, and no significantly larger error. Lab 13 rows are the Lab 13")
    a("results, not re-run. Training time: wall-clock per model on a shared CPU, not comparable between labs.")
    a("")
    w += E13.main_table(summ, TABLE)
    a("")
    a("## Does the gap close?")
    a("")
    eco = summ[REF]
    a(f"Gap = learned controller minus economical PID ({eco['miss20'][0]:.2f} % misses over 20 cm,"
      f" {eco['mean_cm'][0]:.1f} cm mean error, {eco['E_mech'][0]:.1f} kJ E_mech).")
    a("")
    a("| method | miss > 20 cm, Lab 13 | miss > 20 cm, Lab 14 | mean error gap cm, Lab 13 | Lab 14 |"
      " E_mech gap kJ, Lab 13 | Lab 14 | gap closed |")
    a("|---|---:|---:|---:|---:|---:|---:|---|")
    verdict = {}
    for l13, l14 in PAIRS:
        r13, r14 = summ[l13], summ[l14]
        dE13, dE14 = r13["E_mech"][0] - eco["E_mech"][0], r14["E_mech"][0] - eco["E_mech"][0]
        de13, de14 = r13["mean_cm"][0] - eco["mean_cm"][0], r14["mean_cm"][0] - eco["mean_cm"][0]
        if r14["beats"]:
            v = "yes (beats the PID)"
        elif dE14 < dE13 and r14["miss20"][0] <= r13["miss20"][0]:
            v = "no, narrower"
        else:
            v = "no"
        verdict[l14] = v
        a(f"| {NAME14[l14]} vs {NAME13[l13]} | {r13['miss20'][0]:.2f} | {r14['miss20'][0]:.2f} | {de13:+.1f} |"
          f" {de14:+.1f} | {dE13:+.1f} | {dE14:+.1f} | {v} |")
    a("")
    a("## Return on the training objective (20 tuning scenarios x 4 variants, no noise)")
    a("")
    a("Mean return per episode of each controller under both rewards, and the share of the 80 episodes in")
    a("which it beats the economical PID under the corrected reward. A learner above the economical PID here")
    a("found a better policy for the reward it was trained on.")
    a("")
    a("| controller | seeds | return, corrected reward | return, Lab 13 reward | E_mech kJ | final error cm |"
      " episodes above economical PID, corrected % |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    pnames = {"fast": "fast PID", "balanced": "balanced PID", "economical": "economical PID",
              **NAME13, **NAME14}
    for lab in PROBE:
        if lab in probe:
            p = probe[lab]
            a(f"| {pnames[lab]} | {p['seeds']} | {E13.fmt(p['g14'], '{:.3f}')} | {E13.fmt(p['g13'], '{:.3f}')} |"
              f" {E13.fmt(p['E'], '{:.1f}')} | {E13.fmt(p['err'], '{:.1f}')} |"
              f" {'-' if lab == 'economical' else E13.fmt(p['share'], '{:.0f}')} |")
    a("")
    a("## Command smoothness and energy split (Lab 13d columns)")
    a("")
    levels = sorted(k for k in summ["balanced"] if k.startswith("err_n"))
    a("| controller | mean abs(du/dt) 1/s | excess kJ | oscillation % | overshoot % | direct excess % | "
      + " | ".join(f"error cm, noise {k[5:]} deg" for k in levels) + " |")
    a("|---|" + "---:|" * (5 + len(levels)))
    for lab in TABLE:
        if lab in summ and "du_dt" in summ[lab]:
            r = summ[lab]
            small = r["excess_kJ"] < EXCESS_MIN_KJ
            shares = ("-", "-", "-") if small else tuple(f"{r[k + '_share']:.0f}" for k in
                                                         ("oscillation", "overshoot", "direct_excess"))
            a(f"| {NAME[lab]} | {r['du_dt']:.2f} | {r['excess_kJ']:.1f} | {shares[0]} | {shares[1]} | {shares[2]} | "
              + " | ".join(f"{r[k]:.1f}" for k in levels) + " |")
    a("")
    a(f"'-': excess under {EXCESS_MIN_KJ:g} kJ, the shares of so small an excess are not meaningful. The floor is the")
    a("least energy of the full move; a controller that does not finish the move can come close to it or below it.")
    a("")
    hk = hack_table()
    if hk:
        a("## Why SAC and TD3 score higher on the corrected reward while missing")
        a("")
        a("`rl/14_hack_check.py`, same 80 tuning episodes, no noise. Return terms of the corrected reward (mean per")
        a("episode, already divided by the scale); steps outside the 5 cm tolerance split by whether the stall term")
        a(f"is active (|omega| < {C.OM_S:g} rad/s) or not; mean |omega| over the last 2 s.")
        a("")
        a("| controller | tracking term | energy term | stall term | steps outside tolerance, stopped % |"
          " outside tolerance, moving % | mean abs(omega) last 2 s rad/s | final error cm |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        for lab, h in hk.items():
            a(f"| {pnames[lab]} | {h['tracking']:.3f} | {h['energy']:.3f} | {h['stall']:.3f} |"
              f" {h['pct_out_stalled']:.1f} | {h['pct_out_moving']:.1f} | {h['tail_abs_omega']:.4f} |"
              f" {h['final_err_cm']:.1f} |")
        a("")
        he, hs, ht = hk["economical"], hk["r14_sac_100k"], hk["r14_td3_100k"]
        a(f"- Against the economical PID, SAC gains {hs['energy'] - he['energy']:+.3f} on the energy term and loses"
          f" {he['tracking'] - hs['tracking']:.3f} on tracking and {he['stall'] - hs['stall']:.3f} on the stall term"
          f" (TD3: gains {ht['energy'] - he['energy']:+.3f}, loses {he['tracking'] - ht['tracking']:.3f} and"
          f" {he['stall'] - ht['stall']:.3f}). Missing by {hs['final_err_cm']:.0f} cm saves more than it costs.")
        a(f"- They are outside the tolerance while still moving in {hs['pct_out_moving']:.0f} % (SAC) and"
          f" {ht['pct_out_moving']:.0f} % (TD3) of the steps, against {he['pct_out_moving']:.0f} % for the PID, and"
          f" end with mean |omega| {hs['tail_abs_omega']:.3f} and {ht['tail_abs_omega']:.3f} rad/s, above the"
          f" {C.OM_S:g} rad/s stall speed: the stall term mostly does not apply to them.")
        a("- The 14a check tested stopping 10 and 20 cm short with the boom at rest. It did not test a policy")
        a("  that misses by more and keeps the boom moving. The corrected reward ranks the reference controllers")
        a("  correctly but is not a faithful objective: it can be exploited.")
        a("")
    a("## Statement")
    a("")
    learned14 = [k for k in NEW14 if k in summ]
    winners = [NAME14[k] for k in learned14 if summ[k]["beats"]]
    if winners:
        a(f"With the corrected reward these controllers beat the economical PID: {', '.join(winners)}.")
    else:
        a(f"**No Lab 14 controller beats the economical PID** ({eco['miss20'][0]:.2f} % misses over 20 cm,"
          f" {eco['mean_cm'][0]:.1f} cm mean error, {eco['E_mech'][0]:.1f} kJ E_mech). **The gap does not close.**")
    best = min(learned14, key=lambda k: (summ[k]["miss20"][0], summ[k]["E_mech"][0]))
    b = summ[best]
    a(f"Closest Lab 14 controller (fewest misses over 20 cm, then least E_mech): {NAME14[best]},"
      f" {b['miss20'][0]:.2f} % misses, {b['mean_cm'][0]:.1f} cm mean error, {b['E_mech'][0]:.1f} kJ"
      f" ({b['E_mech'][0] - eco['E_mech'][0]:+.1f} kJ against the economical PID).")
    a("")
    above = [pnames[k] for k in NEW14 if k in probe and probe[k]["g14"][0] > probe["economical"]["g14"][0]]
    a("Training objective: " + (f"{'; '.join(above)} reach a higher corrected return than the economical PID"
      f" ({probe['economical']['g14'][0]:.3f}) on the tuning scenarios." if above else
      "no Lab 14 learner reaches the corrected return of the economical PID on the tuning scenarios."))
    a("")
    a("What this means for the reward-mismatch explanation:")
    a("")
    s14, t14 = summ["r14_sac_100k"], summ["r14_td3_100k"]
    a(f"- Online (SAC, TD3): the learners did optimise the corrected reward (higher return than the economical"
      f" PID), but the reward could be exploited. E_mech fell below the PID ({s14['E_mech'][0]:.1f} and"
      f" {t14['E_mech'][0]:.1f} kJ) because the boom does not finish the move: {s14['miss20'][0]:.1f} and"
      f" {t14['miss20'][0]:.1f} % misses over 20 cm. The learning-rate check (x0.3, x3, 1 seed) gives"
      f" {min(summ[k]['miss20'][0] for k in LR_ROWS):.1f} to {max(summ[k]['miss20'][0] for k in LR_ROWS):.1f} %"
      " misses; no setting changes the verdict.")
    i13, i14, c13, c14 = summ["iql_1M"], summ["r14_iql_1M"], summ["cql_1M"], summ["r14_cql_1M"]
    a(f"- Offline (IQL, CQL on the same data): the corrected reward moved them toward less energy (IQL"
      f" {i13['E_mech'][0]:.1f} to {i14['E_mech'][0]:.1f} kJ, CQL {c13['E_mech'][0]:.1f} to {c14['E_mech'][0]:.1f} kJ)"
      f" and lowered their accuracy (IQL {i13['mean_cm'][0]:.1f} to {i14['mean_cm'][0]:.1f} cm, misses over 20 cm"
      f" {i13['miss20'][0]:.2f} to {i14['miss20'][0]:.2f} %). IQL reaches the PID energy, not its accuracy; CQL"
      f" still misses {c14['miss20'][0]:.1f} %.")
    bc = summ["r14_bc_eco500k"]
    a(f"- Data (14c): BC on the economical-PID data alone gives {bc['E_mech'][0]:.1f} kJ"
      f" ({bc['E_mech'][0] - eco['E_mech'][0]:+.1f} kJ) at {bc['mean_cm'][0]:.1f} cm mean error"
      f" ({bc['mean_cm'][0] - eco['mean_cm'][0]:+.1f} cm, p {E13.pval(bc['p_err'])}) and {bc['miss20'][0]:.2f} %"
      " misses: a near copy of the PID, slightly cheaper and less accurate. The data supports the PID; a cloned"
      " PID does not beat it.")
    a("- A reward that ranks the PID points in the intended order was not enough. With a per-step trade of energy")
    a("  against error, the online learners traded accuracy for energy instead of finding the economical PID.")
    a("  The Lab 13 conclusion (no learner beats the economical PID) holds in this test. Not tested: a reward or")
    a("  constraint that cannot be paid for by missing (for example a hard accuracy constraint).")
    a("")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(w) + "\n")
    print("\n".join(w))
    print(f"wrote {MD_PATH}, {CSV_PATH}")


def make_report():
    A13 = E13.arrays(E13.read_moves("rl/13_moves.csv.gz"))
    A12 = E13.arrays(S.read_moves())
    A14 = E13.arrays(E13.read_moves(MOVES))
    reroll = E13.check_reroll(A12, A13)
    E13.NAME.update({**NAME13, **NAME14})
    E13.GROUP.update({k: v[0] for k, v in NEW14.items()})
    summ = E13.summarise(A12, {**A13, **A14})
    hours = train_hours()
    for lab in NEW14:
        summ[lab]["train_h"] = hours.get(lab, float("nan"))
    check_lab13(summ)
    E13.log(reroll)
    E13.CSV_PATH = CSV_PATH  # write_csv of rl/13_eval.py writes to its module global: Lab 14 path
    E13.write_csv({k: summ[k] for k in TABLE if k in summ})
    probe = probe_table(list(csv.DictReader(open(RETURNS))))
    report(summ, probe)


def main():
    if "--report" in sys.argv:
        if os.path.exists(MD_PATH):
            with open(MD_PATH) as f:
                E13.CHECK_LOG.extend(ln[3:-1] for ln in f.read().splitlines()
                                     if ln.startswith(("- `check 1", "- `check 2", "- `evaluation", "- `return")))
        make_report()
        return
    miss = [model_file(lab, s) for lab, spec in NEW14.items() for s in spec[4] if not os.path.exists(model_file(lab, s))]
    if miss:
        sys.exit("missing models:\n  " + "\n  ".join(miss))
    E13.check_zero_residual()
    check_env_path()
    t0 = time.time()
    js = jobs()
    rows = [m for chunk in pool_map(run_row, js) for m in chunk]
    E13.log(f"evaluation: {len(js)} rows, {len(rows)} moves in {time.time() - t0:.0f} s on {PROCS} processes")
    t1 = time.time()
    pj = probe_jobs()
    prows = pool_map(probe_episode, pj)
    E13.log(f"return probe: {len(prows)} tuning episodes ({len(PROBE)} controllers) in {time.time() - t1:.0f} s")
    if QUICK:
        e = np.array([r["err_cm"] for r in rows])
        print(f"quick: {len(rows)} moves, mean error {e.mean():.2f} cm; {len(prows)} probe episodes;"
              " nothing written")
        return
    with gzip.open(MOVES, "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=E13.FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    with open(RETURNS, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(prows[0]))
        wr.writeheader()
        wr.writerows(prows)
    make_report()


if __name__ == "__main__":
    main()
