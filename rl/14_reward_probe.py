"""Lab 14a: corrected training reward. PID returns on the 20 tuning scenarios only. No training.

Lab 12/13 reward per control step (BoomEnvRealRobust):
    r = -(e^2 + 0.01 omega^2 + w P / P_REF),   w = 0.03
Candidate family (adds one stall term; k = 0 is the Lab 13 form):
    r = -(e^2 + 0.01 omega^2 + w P / P_REF + k h),
    h = max(0, |e| - TOL_RAD) if |omega| < OM_S else 0     TOL = 5 cm at the tip
h charges an error outside the tolerance only while the boom is (nearly) stopped, so it hits a policy that
stops short of the target and not one that approaches slowly. OM_S is scanned (OM_S = inf: plain hinge).
Every term is a per-step sum, so the return of one episode is linear in (w, k):
    G(w, k) = -(S_err + S_om) - w S_P - k S_h[OM_S]
This script measures S_err, S_om, S_P, S_h of six reference controllers in the training env
(residual a = 0, randomize off, no sensor noise):
    fast, balanced, economical   the Lab 11 PID points (BoomEnvRealRobust(residual=True, base=<point>))
    eco_short10, eco_short20     the economical PID told a target 10 / 20 cm short of the true one, towards
                                 the start: a policy that saves energy by missing (reward keeps the true target)
    no_move                      u = 0
on L10.SCEN["orig20"] (the 20 tuning (theta0, target) pairs) x L10.VARIANTS (4 tip-mass / payload cases)
= 80 episodes of 200 steps each. The 100 unseen scenarios are not used.

Required order of the mean return over the 80 episodes:
    economical > balanced > fast > eco_short10 > eco_short20,   fast > no_move
It prints the order at the Lab 13 reward, the w interval with k = 0, and a (w, k) grid search; the
proposed point maximises the smallest pairwise margin (G_x - G_y) / |G_economical|.

Run: python rl/14_reward_probe.py    (about 1 min on 4 processes)
Writes rl/14_reward_probe.csv (one row per controller, variant, scenario).
"""

import csv
import importlib.util
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = "rl/14_reward_probe.csv"
W_LAB13 = 0.03
TOL_CM = 5.0  # tolerance of the hinge term (ASSUMED): half the 10 cm miss threshold of Lab 13
W_NEW, K_NEW, OM_NEW = 20.0, 17.5, 0.01  # proposed point (see the w scan printed at the end)
SCALE = 50.0  # proposed reward = r / SCALE: economical return on the tuning set ~ the Lab 13 value
OM_S = (0.01, 0.02, 0.05, float("inf"))  # rad/s, stall threshold of h (scanned)
SHORT_CM = {"eco_short10": 10.0, "eco_short20": 20.0}
CTRLS = ("fast", "balanced", "economical", "eco_short10", "eco_short20", "no_move")
ORDER = [("economical", "balanced"), ("balanced", "fast"), ("fast", "eco_short10"),
         ("eco_short10", "eco_short20"), ("fast", "no_move")]


def _load(name, fname):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(job):
    ctrl, variant, k = job
    import boom_env_real_robust as RR

    L10 = _load("lab10_sched_real", "10_sched_real.py")
    th0, target = L10.SCEN["orig20"][k][:2]
    m_tip, payload = L10.VARIANTS[variant]
    if ctrl == "no_move":
        env = RR.BoomEnvRealRobust(w_energy=0.0, residual=False, randomize=False, m_tip=m_tip, payload=payload)
    elif ctrl in SHORT_CM:
        d = SHORT_CM[ctrl] / 100.0 / RR.R.L_BOOM_M * np.sign(target - th0)

        class Short(RR.BoomEnvRealRobust):
            def _plant_obs(self):  # what the base PID sees: error to the shortened target
                o = RR.BoomEnvRealRobust._plant_obs(self).copy()
                o[2] -= d
                return o

        env = Short(w_energy=0.0, residual=True, randomize=False, m_tip=m_tip, payload=payload,
                    base="economical")
    else:
        env = RR.BoomEnvRealRobust(w_energy=0.0, residual=True, randomize=False, m_tip=m_tip,
                                   payload=payload, base=ctrl)
    tol = TOL_CM / 100.0 / RR.R.L_BOOM_M
    env.reset(seed=0, options={"th0": th0, "target": target})
    s = dict(S_err=0.0, S_om=0.0, S_P=0.0, **{f"S_h_{o:g}": 0.0 for o in OM_S})
    a = np.zeros(1, np.float32)
    trunc, n = False, 0
    while not trunc:
        _, _, _, trunc, info = env.step(a)
        e = env.target - env.th
        s["S_err"] += e**2
        s["S_om"] += 0.01 * env.om**2
        s["S_P"] += info["P_W"] / RR.P_REF_W
        for o in OM_S:
            if abs(env.om) < o:
                s[f"S_h_{o:g}"] += max(0.0, abs(e) - tol)
        n += 1
    return {"controller": ctrl, "variant": variant, "scen": k, "steps": n, **s,
            "E_mech_kJ": env.energy_J / 1e3, "final_err_cm": abs(env.target - env.th) * 100 * RR.R.L_BOOM_M}


def returns(M, w, k):
    return {c: -(M[c]["S_err"] + M[c]["S_om"] + w * M[c]["S_P"] + k * M[c]["S_h"]) for c in CTRLS}


def margins(M, w, k):
    G = returns(M, w, k)
    ref = abs(G["economical"].mean())
    return {f"{x}>{y}": (G[x].mean() - G[y].mean()) / ref for x, y in ORDER}, G


def report(M, w, k, label):
    m, G = margins(M, w, k)
    print(f"\n{label}: w = {w:.4g}, k = {k:.4g}")
    print("  mean return per episode: " + ", ".join(f"{c} {G[c].mean():.3f}" for c in CTRLS))
    print("  order margin (G_x - G_y)/|G_eco|: " + ", ".join(f"{p} {v:+.4f}" for p, v in m.items()))
    print("  share of the 80 episodes where x > y: "
          + ", ".join(f"{x}>{y} {100 * np.mean(G[x] > G[y]):.0f} %" for x, y in ORDER))
    for c in ("fast", "balanced", "economical"):
        parts = (M[c]["S_err"].mean() + M[c]["S_om"].mean(), w * M[c]["S_P"].mean(), k * M[c]["S_h"].mean())
        print(f"  {c:>10} return terms: tracking {-parts[0]:.3f}, energy {-parts[1]:.3f}, hinge {-parts[2]:.3f}")


def main():
    L10 = _load("lab10_sched_real", "10_sched_real.py")
    n_scen = len(L10.SCEN["orig20"])
    jobs = [(c, v, k) for c in CTRLS for v in L10.VARIANTS for k in range(n_scen)]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} episodes ({n_scen} tuning scenarios x {len(L10.VARIANTS)} variants x {len(CTRLS)}"
          f" controllers), steps per episode {sorted({r['steps'] for r in rows})}, wrote {OUT}")
    M = {c: {key: np.array([r[key] for r in rows if r["controller"] == c])
             for key in rows[0] if key.startswith("S_") or key in ("E_mech_kJ", "final_err_cm")} for c in CTRLS}
    for c in CTRLS:
        M[c]["S_h"] = M[c]["S_h_inf"]

    print(f"\nmean per episode ({n_scen * len(L10.VARIANTS)} tuning episodes per controller):")
    hk = [f"S_h_{o:g}" for o in OM_S]
    print(f"{'controller':>11} | {'S_err':>7} | {'S_om':>6} | {'S_P':>7} | " + " | ".join(f"{h:>8}" for h in hk)
          + " | E_mech kJ | final err cm mean/max")
    for c in CTRLS:
        d = M[c]
        print(f"{c:>11} | {d['S_err'].mean():7.3f} | {d['S_om'].mean():6.3f} | {d['S_P'].mean():7.3f} |"
              + " " + " | ".join(f"{d[h].mean():8.3f}" for h in hk) + f" | {d['E_mech_kJ'].mean():9.1f} |"
              f" {d['final_err_cm'].mean():.1f} / {d['final_err_cm'].max():.1f}")

    report(M, W_LAB13, 0.0, "Lab 13 reward")

    # k = 0: closed-form w interval of every pair (G_x > G_y <=> dA > w dB)
    print("\nk = 0 (energy weight only), condition on w per pair:")
    lo, hi = 0.0, np.inf
    for x, y in ORDER:
        dA = (M[y]["S_err"] + M[y]["S_om"]).mean() - (M[x]["S_err"] + M[x]["S_om"]).mean()
        dB = M[x]["S_P"].mean() - M[y]["S_P"].mean()
        if dB < 0:
            lo = max(lo, dA / dB)
            print(f"  {x} > {y}: w > {dA / dB:.4f}")
        elif dB > 0:
            hi = min(hi, dA / dB)
            print(f"  {x} > {y}: w < {dA / dB:.4f}")
    print(f"  feasible w: {lo:.4f} < w < {hi:.4f} -> {'exists' if lo < hi else 'EMPTY'}")

    # (w, k) grid per stall threshold
    ws, ks = np.geomspace(0.01, 100, 161), np.concatenate([[0.0], np.geomspace(0.01, 1000, 201)])
    overall = None
    for o in OM_S:
        for c in CTRLS:
            M[c]["S_h"] = M[c][f"S_h_{o:g}"]
        best, n_ok, ok_w, ok_k = None, 0, [], []
        for w in ws:
            for k in ks:
                m, _ = margins(M, w, k)
                worst = min(m.values())
                if worst > 0:
                    n_ok += 1
                    ok_w.append(w)
                    ok_k.append(k)
                if best is None or worst > best[0]:
                    best = (worst, w, k)
        print(f"\nOM_S = {o:g} rad/s, (w, k) grid {len(ws)} x {len(ks)}: {n_ok} points give the required order"
              + (f"; w in [{min(ok_w):.3g}, {max(ok_w):.3g}], k in [{min(ok_k):.3g}, {max(ok_k):.3g}]" if n_ok else ""))
        print(f"  largest smallest margin {best[0]:+.4f} at w = {best[1]:.4g}, k = {best[2]:.4g}")
        if overall is None or best[0] > overall[0]:
            overall = (best[0], best[1], best[2], o)
    for c in CTRLS:
        M[c]["S_h"] = M[c][f"S_h_{overall[3]:g}"]
    print(f"  best over all OM_S: margin {overall[0]:+.4f} at OM_S = {overall[3]:g}, w = {overall[1]:.4g}, k = {overall[2]:.4g}")

    # w scan at OM_NEW: best k per w; the margin saturates, the e^2 share of the return shrinks
    for c in CTRLS:
        M[c]["S_h"] = M[c][f"S_h_{OM_NEW:g}"]
    print(f"\nw scan at OM_S = {OM_NEW:g} rad/s (k = best for each w):")
    for w in (5, 10, 20, 30, 50, 100, 1000):
        mk = max((min(margins(M, w, k)[0].values()), k) for k in np.geomspace(0.5, 2000, 400))
        _, G = margins(M, w, mk[1])
        trk = (M["economical"]["S_err"] + M["economical"]["S_om"]).mean() / abs(G["economical"].mean())
        print(f"  w = {w:6g}: k = {mk[1]:7.2f}, smallest margin {mk[0]:+.4f}, e^2 + omega^2 share of the"
              f" economical return {100 * trk:.1f} %")
    report(M, W_NEW, K_NEW, f"PROPOSED reward (OM_S = {OM_NEW:g} rad/s, TOL = {TOL_CM:g} cm)")
    G = returns(M, W_NEW, K_NEW)
    print(f"  divided by SCALE = {SCALE:g}: " + ", ".join(f"{c} {G[c].mean() / SCALE:.3f}" for c in CTRLS))

if __name__ == "__main__":
    main()
