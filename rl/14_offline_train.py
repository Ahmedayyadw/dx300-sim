"""Lab 14b/14c: offline RL on the relabelled dataset (IQL, CQL) and BC on the economical-PID data.

Runs in .venv-d3rlpy, like rl/13_offline_train.py, with the same d3rlpy defaults, budget and export:
  data      rl/models/14_dataset_1M.npz of rl/14_relabel.py: the Lab 13 1M transitions (same
            observations and actions) with the corrected reward.
              iql, cql   all 5,000 episodes (1M transitions), corrected reward          (14b)
              bc         the 2,500 economical-PID + noise episodes only (500k), tag eco500k  (14c)
                         BC ignores the reward: this tests the data, not the reward.
            terminals = 0, timeouts = 1 at the last step of every episode.
  algos     d3rlpy BCConfig, CQLConfig, IQLConfig, library defaults, device cpu. Config saved (*.json).
  budget    100,000 gradient steps, final model (as Lab 13).
  threads   torch.set_num_threads(1) by default (Lab 13 used 4).
  export    save_policy -> rl/models/14_{algo}_{tag}_s{seed}.pt, checked against algo.predict.

Appends the wall-clock to rl/14_train_log.csv. Resumable: a policy already exported is skipped.
Usage (repo root, OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):
  .venv-d3rlpy/bin/python rl/14_offline_train.py --algo iql --seed 0
  .venv-d3rlpy/bin/python rl/14_offline_train.py --algo bc --seed 0
  .venv-d3rlpy/bin/python rl/14_offline_train.py --algo cql --smoke [--steps 500]
        on rl/models/14_dataset_smoke.npz, *_smoke, not logged, prints steps per second
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab14_common", os.path.join(HERE, "14_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab14_common"] = C
_spec.loader.exec_module(C)

STEPS = 100_000
LOG_DIR = f"{C.MODEL_DIR}/14_d3rlpy_logs"  # gitignored
TAGS = {"iql": "1M", "cql": "1M", "bc": "eco500k"}


def arg(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    algo_name = arg("--algo", str, None)
    assert algo_name in TAGS, "--algo bc | cql | iql"
    dry = "--smoke" in sys.argv
    seed = arg("--seed", int, 0)
    threads = arg("--threads", int, 1)
    steps = arg("--steps", int, 500) if dry else STEPS
    tag = TAGS[algo_name]
    name = C.model_name(algo_name, tag, seed, "_smoke" if dry else "")
    path = f"{C.MODEL_DIR}/{name}.pt"
    if os.path.exists(path) and not dry:
        print(f"{name}: exists, skipped")
        return

    import d3rlpy
    import torch
    from d3rlpy.logging import FileAdapterFactory

    torch.set_num_threads(threads)
    d3rlpy.seed(seed)
    data = np.load(f"{C.MODEL_DIR}/14_dataset_smoke.npz" if dry else C.DATASET)
    ep = int(data["ep_steps"])
    obs, act, rew = data["observations"], data["actions"], data["rewards"]
    behaviour = data["behaviour"]
    assert len(obs) == len(behaviour) * ep
    if algo_name == "bc":
        keep = np.repeat(behaviour == 1, ep)  # economical-PID episodes only
        obs, act, rew = obs[keep], act[keep], rew[keep]
    n = len(obs)
    assert n % ep == 0
    timeouts = np.zeros(n, np.float32)
    timeouts[ep - 1 :: ep] = 1.0
    dataset = d3rlpy.dataset.MDPDataset(observations=obs, actions=act, rewards=rew,
                                        terminals=np.zeros(n, np.float32), timeouts=timeouts)
    assert dataset.size() == n // ep, (dataset.size(), n // ep)
    config = {"bc": d3rlpy.algos.BCConfig, "cql": d3rlpy.algos.CQLConfig,
              "iql": d3rlpy.algos.IQLConfig}[algo_name]()
    algo = config.create(device="cpu")
    t0 = time.time()
    algo.fit(dataset, n_steps=steps, n_steps_per_epoch=min(steps, 10_000), experiment_name=name,
             with_timestamp=False, logger_adapter=FileAdapterFactory(root_dir=LOG_DIR),
             show_progress=False, save_interval=10**9)
    train_s = time.time() - t0
    algo.save_policy(path)
    x = obs[:1000]
    with torch.no_grad():
        a_script = torch.jit.load(path)(torch.as_tensor(x)).numpy()
    diff = float(np.max(np.abs(a_script - algo.predict(x))))
    assert diff < 1e-5, diff
    assert float(np.abs(a_script).max()) <= 1.0 + 1e-6
    with open(path[:-3] + ".json", "w") as f:
        json.dump({"algo": algo_name, "d3rlpy": d3rlpy.__version__, "torch": torch.__version__,
                   "data": tag, "transitions": n, "economical_only": algo_name == "bc", "seed": seed,
                   "steps": steps, "threads": threads, "reward": {"W": C.W, "K": C.K, "OM_S": C.OM_S,
                   "TOL_CM": C.TOL_CM, "SCALE": C.SCALE}, "config": json.loads(config.serialize())},
                  f, indent=1)
    print(f"{name}: {steps} gradient steps on {n} transitions in {train_s:.0f} s"
          f" = {steps / train_s:.0f} steps/s ({threads} thread) -> {path}; TorchScript vs algo.predict"
          f" largest difference {diff:.1e}")
    if not dry:
        C.append_log("offline", algo_name, tag, seed, steps, train_s, "cpu")


if __name__ == "__main__":
    main()
