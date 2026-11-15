"""Lab 13b: offline dataset from the PID points with exploration noise. No training here.

Env: BoomEnvRealRobust(w_energy=0.03, residual=True), the training env of the best Lab 12 SAC, so
the stored observation [theta, omega, e, int(e) dt], the residual action and the reward are the
ones every Lab 13 learner uses. 5,000 episodes x 200 steps = 1,000,000 transitions.

  scenarios  the training distribution: env.reset(seed=1_000_000 + i). The 100 unseen scenarios are
             not used; the script asserts that no episode starts from one of them.
  behaviour  episode i even: balanced PID + noise      a = eps
             episode i odd:  economical PID + noise    a = clip(u_eco + eps - u_bal, -1, 1)
             (50/50, interleaved, ASSUMED). u_eco comes from a second Sched11 with the economical
             gains that sees the same observations and is told the applied command, as the base
             controller inside the env. u_bal is read from a copy of the base controller, so the
             env's own controller is not touched.
  noise      eps ~ N(0, SIGMA^2), SIGMA = 0.1 (ASSUMED, docs/13_benchmark.md section 3), a new draw
             every control step, generator default_rng([13, i]) per episode.
  stored     obs before the step, the float32 residual passed to env.step, the env reward.
             Episodes end by the time limit only (no terminal state).

Subsets for the dataset-size study: the first 500 / 1,500 / 5,000 episodes (100k / 300k / 1M).

Writes rl/models/13_dataset_1M.npz (gitignored), rl/13_dataset.csv (summary per behaviour and
subset) and a "dataset" line in rl/13_train_log.csv.
Usage (repo root, .venv active):
  python rl/13_dataset.py              about 2 min on 16 cores
  python rl/13_dataset.py --smoke      20 episodes -> rl/models/13_dataset_smoke.npz, no summary file
                                       (not --dry-run: rl/10_sched_real.py reads that flag)
"""

import copy
import csv
import importlib.util
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab13_common", os.path.join(HERE, "13_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab13_common"] = C
_spec.loader.exec_module(C)

SIGMA = 0.1
SEED0 = 1_000_000
BEHAV = ("balanced", "economical")
SUMMARY = "rl/13_dataset.csv"


def episode(i):
    """Episode i: (obs (200, 4), act (200, 1), rew (200,), meta)."""
    sys.path.insert(0, HERE)
    import boom_env_real_robust as E

    L11, R = E.L11, E.L10.R
    env = C.make_env()
    assert env.max_steps == C.EP_STEPS and E.BAL_GAINS[1] == 0.0
    behav = BEHAV[i % 2]
    rng = np.random.default_rng([13, i])
    obs, _ = env.reset(seed=SEED0 + i)
    th0, target = env.th, env.target
    eco = L11.Sched11(E.point_gains("economical"), E.EST) if behav == "economical" else None
    O = np.zeros((C.EP_STEPS, 4), np.float32)
    A = np.zeros((C.EP_STEPS, 1), np.float32)
    Rw = np.zeros(C.EP_STEPS, np.float32)
    n_clip, ret = 0, 0.0
    for k in range(C.EP_STEPS):
        eps = rng.normal(0.0, SIGMA)
        if eco is None:
            a_raw = eps
        else:
            po = env._plant_obs()
            u_bal = copy.copy(env.ctrl).act(po)  # what the base will output; env.ctrl unchanged
            a_raw = eco.act(po) + eps - u_bal
        a = np.clip(np.array([a_raw]), -1.0, 1.0).astype(np.float32)
        n_clip += abs(a_raw) > 1.0
        O[k], A[k] = obs, a
        obs, r, _, trunc, info = env.step(a)
        if eco is not None:
            eco.u_prev = info["u"]  # estimator: the command really applied
            eco.update(env._plant_obs(), info["tau_cyl"], env.th_min, env.th_max)
        Rw[k] = r
        ret += r
    assert trunc
    meta = (i, i % 2, th0, target, abs(target - env.th) * R.L_BOOM_M * 100.0, env.energy_J / 1e3,
            ret, n_clip / C.EP_STEPS, float(np.abs(A).mean()))
    return O, A, Rw, meta


def check_unseen(meta):
    """No dataset episode starts from one of the 100 unseen (start, target) pairs."""
    sys.path.insert(0, HERE)
    import boom_env_real_robust as E

    new = np.array(E.L10.SCEN["new100"])
    assert len(new) == 100
    d = np.abs(meta[:, None, 2:4] - new[None, :, :]).max(axis=2).min()
    assert d > 1e-6, d
    return float(d)


def main():
    dry = "--smoke" in sys.argv
    n_ep = 20 if dry else C.SIZES["1M"]
    path = f"{C.MODEL_DIR}/13_dataset_smoke.npz" if dry else C.DATASET
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
        out = pool.map(episode, range(n_ep), chunksize=10)
    gen_s = time.time() - t0
    obs = np.concatenate([o[0] for o in out])
    act = np.concatenate([o[1] for o in out])
    rew = np.concatenate([o[2] for o in out])
    meta = np.array([o[3] for o in out], float)
    d_min = check_unseen(meta)
    np.savez(path, observations=obs, actions=act, rewards=rew, meta=meta, sigma=SIGMA,
             ep_steps=C.EP_STEPS)
    print(f"{path}: {len(obs)} transitions, {n_ep} episodes in {gen_s:.0f} s; closest unseen"
          f" scenario {d_min:.2e} rad away (largest of the start and target differences)")
    rows = []
    for size, n in ([("smoke", n_ep)] if dry else C.SIZES.items()):
        for b, name in enumerate(BEHAV):
            m = meta[:n][meta[:n, 1] == b]
            rows.append({"subset": size, "behaviour": name, "sigma": SIGMA, "episodes": len(m),
                         "transitions": len(m) * C.EP_STEPS,
                         "miss_over_20cm_pct": 100.0 * float((m[:, 4] > 20).mean()),
                         "mean_err_cm": float(m[:, 4].mean()), "mean_E_mech_kJ": float(m[:, 5].mean()),
                         "mean_return": float(m[:, 6].mean()),
                         "clipped_action_pct": 100.0 * float(m[:, 7].mean()),
                         "mean_abs_action": float(m[:, 8].mean())})
    for r in rows:
        print("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    if not dry:
        with open(SUMMARY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        C.append_log("dataset", "pid_noise", "1M", "", len(obs), gen_s, "cpu")


if __name__ == "__main__":
    main()
