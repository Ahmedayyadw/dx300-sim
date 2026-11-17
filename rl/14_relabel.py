"""Lab 14b/14c: relabel the Lab 13 offline dataset with the corrected reward. No new data.

Input rl/models/13_dataset_1M.npz (rl/13_dataset.py; not modified). Every episode i is replayed in
rl/14_common.make_env() from env.reset(seed=1_000_000 + i) with the STORED actions. The env is
deterministic, so the replay passes through the same states. The script asserts, per transition, that
the replayed observation and the replayed Lab 13 reward equal the stored ones (float32 tolerance); only
then is the new reward written. The replay is needed because the dataset does not store P, and the
last step of each episode has no next state to read e and omega from.

Output rl/models/14_dataset_1M.npz (gitignored):
  observations, actions        copied unchanged from the Lab 13 file
  rewards                      corrected reward (Lab 14a)
  rewards_lab13                the stored Lab 13 reward
  behaviour                    per episode: 0 balanced PID + noise, 1 economical PID + noise
                               (episode i -> i % 2, as rl/13_dataset.py; also checked against meta)
  ep_steps
It prints the mean return per behaviour under both rewards.

Usage (repo root, .venv active, OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):
  python rl/14_relabel.py [--procs 14]        about 1 min on 14 processes
  python rl/14_relabel.py --smoke             first 20 episodes, rl/models/14_dataset_smoke.npz
"""

import importlib.util
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

SRC = f"{C.MODEL_DIR}/13_dataset_1M.npz"
SEED0 = 1_000_000  # rl/13_dataset.py
TOL_OBS, TOL_R = 1e-5, 1e-6


def replay(args):
    i, obs, act, rew = args
    env = C.make_env()
    o, _ = env.reset(seed=SEED0 + i)
    new = np.zeros(len(act), np.float32)
    d_obs = d_r = 0.0
    for k in range(len(act)):
        d_obs = max(d_obs, float(np.max(np.abs(o - obs[k]))))
        o, r, _, trunc, info = env.step(act[k])
        d_r = max(d_r, abs(float(info["r_lab13"]) - float(rew[k])))
        new[k] = r
    assert trunc
    return i, new, d_obs, d_r


def main():
    dry = "--smoke" in sys.argv
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 14
    d = np.load(SRC)
    ep = int(d["ep_steps"])
    assert ep == C.EP_STEPS
    n_ep = 20 if dry else len(d["observations"]) // ep
    O, A, R = d["observations"][: n_ep * ep], d["actions"][: n_ep * ep], d["rewards"][: n_ep * ep]
    meta = d["meta"][:n_ep]
    assert np.array_equal(meta[:, 0].astype(int), np.arange(n_ep))
    behaviour = meta[:, 1].astype(np.int8)
    assert np.array_equal(behaviour, np.arange(n_ep) % 2)
    jobs = [(i, O[i * ep:(i + 1) * ep], A[i * ep:(i + 1) * ep], R[i * ep:(i + 1) * ep]) for i in range(n_ep)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        out = pool.map(replay, jobs, chunksize=10)
    new = np.concatenate([o[1] for o in out])
    d_obs, d_r = max(o[2] for o in out), max(o[3] for o in out)
    print(f"replayed {n_ep} episodes ({len(new)} transitions) in {time.time() - t0:.0f} s on {procs} processes;"
          f" largest difference to the stored data: observation {d_obs:.1e}, Lab 13 reward {d_r:.1e}")
    assert d_obs < TOL_OBS and d_r < TOL_R, "replay does not reproduce the stored transitions"
    path = f"{C.MODEL_DIR}/14_dataset_smoke.npz" if dry else C.DATASET
    np.savez(path, observations=O, actions=A, rewards=new, rewards_lab13=R, behaviour=behaviour, ep_steps=ep)
    ret_new, ret_old = new.reshape(n_ep, ep).sum(1), R.reshape(n_ep, ep).sum(1)
    for b, nm in enumerate(("balanced", "economical")):
        m = behaviour == b
        print(f"  {nm:>10} + noise: {m.sum()} episodes, mean return Lab 13 reward {ret_old[m].mean():.3f},"
              f" corrected reward {ret_new[m].mean():.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
