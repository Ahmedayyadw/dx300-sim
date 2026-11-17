"""Lab 14e: how the online learners reach a higher corrected return than the economical PID while missing.

Tuning scenarios only (20 x 4 variants, randomize off, no noise; the unseen scenarios are not used).
Per controller, over all steps of the 80 episodes:
  return terms of the corrected reward (tracking e^2 + 0.01 omega^2, energy W P / P_REF, stall K h),
  each divided by SCALE, mean per episode;
  share of steps outside the tolerance (|e| > TOL) split by |omega| < OM_S (stall term active) and
  |omega| >= OM_S (stall term not active);
  mean |omega| over the last 2 s (40 steps) and the final error.

Usage: python rl/14_hack_check.py      (about 1 min on 14 processes). Writes rl/14_hack_check.csv.
"""

import csv
import importlib.util
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab14_eval", os.path.join(HERE, "14_eval.py"))
V = importlib.util.module_from_spec(_spec)
sys.modules["lab14_eval"] = V
_spec.loader.exec_module(V)
C, L10, E = V.C, V.L10, V.E
OUT = "rl/14_hack_check.csv"
LABELS = {"economical": (None,), "balanced": (None,), "r14_sac_100k": C.SEEDS, "r14_td3_100k": C.SEEDS,
          "r14_iql_1M": C.SEEDS, "r14_cql_1M": C.SEEDS, "r14_bc_eco500k": C.SEEDS}
TAIL = 40


def episode(job):
    label, seed, variant, k = job
    th0, target = L10.SCEN["orig20"][k][:2]
    m_tip, payload = L10.VARIANTS[variant]
    pid = seed is None
    env = C.make_env(randomize=False, m_tip=m_tip, payload=payload, base=label if pid else "balanced")
    obs, _ = env.reset(seed=0, options={"th0": th0, "target": target})
    pol = None if pid else V.get_policy(label, seed)
    tol = C.TOL_CM / 100.0 / E.R.L_BOOM_M
    trk = en = st = 0.0
    n = n_stall = n_move = 0
    om_tail = []
    trunc = False
    while not trunc:
        a = np.zeros(1, np.float32) if pid else pol.predict(obs, deterministic=True)[0]
        obs, _, _, trunc, info = env.step(a)
        e, om = env.target - env.th, env.om
        trk += e**2 + 0.01 * om**2
        en += C.W * info["P_W"] / E.P_REF_W
        out = abs(e) > tol
        if out and abs(om) < C.OM_S:
            n_stall += 1
            st += C.K * (abs(e) - tol)
        elif out:
            n_move += 1
        n += 1
        if n > env.max_steps - TAIL:
            om_tail.append(abs(om))
    return {"controller": label, "seed": "" if pid else seed, "variant": variant, "scen": k,
            "tracking": -trk / C.SCALE, "energy": -en / C.SCALE, "stall": -st / C.SCALE,
            "pct_out_stalled": 100 * n_stall / n, "pct_out_moving": 100 * n_move / n,
            "tail_abs_omega": float(np.mean(om_tail)), "final_err_cm": abs(e) * 100 * E.R.L_BOOM_M}


def main():
    jobs = [(lab, s, v, k) for lab, seeds in LABELS.items() for s in seeds for v in L10.VARIANTS
            for k in range(len(L10.SCEN["orig20"]))]
    with mp.get_context("spawn").Pool(14) as pool:
        rows = pool.map(episode, jobs, chunksize=4)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} tuning episodes, wrote {OUT}")
    keys = ("tracking", "energy", "stall", "pct_out_stalled", "pct_out_moving", "tail_abs_omega", "final_err_cm")
    print("controller | " + " | ".join(keys))
    for lab in LABELS:
        rs = [r for r in rows if r["controller"] == lab]
        print(f"{lab} | " + " | ".join(f"{np.mean([r[k] for r in rs]):.4f}" for k in keys))


if __name__ == "__main__":
    main()
