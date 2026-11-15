"""Lab 13b: offline RL on the PID dataset (behaviour cloning, CQL, IQL; d3rlpy).

Runs in the separate environment .venv-d3rlpy (d3rlpy pins gymnasium 1.0.0; .venv keeps the
gymnasium of Lab 12). Needs numpy, torch and d3rlpy only: no env, no stable-baselines3.

  data      rl/models/13_dataset_1M.npz of rl/13_dataset.py; --size 100k / 300k / 1M = the first
            500 / 1,500 / 5,000 episodes. terminals = 0, timeouts = 1 at the last step of every
            episode (time limit, not a terminal state).
  algos     d3rlpy BCConfig, CQLConfig, IQLConfig with the library defaults, device cpu. The full
            config of the run is written next to the model (*.json) and printed in the report.
  budget    100,000 gradient steps (ASSUMED: the number of updates of the Lab 12 SAC in its 100k
            env steps), at every dataset size. The final model is used; nothing is selected with
            the unseen scenarios.
  export    save_policy -> rl/models/13_{algo}_{size}_s{seed}.pt (TorchScript, greedy action). The
            script asserts that the exported policy equals algo.predict on 1,000 dataset
            observations. rl/13_eval.py loads it with torch.jit.load in .venv.

Appends the wall-clock to rl/13_train_log.csv. Resumable: a policy already exported is skipped.
Usage (repo root):
  .venv-d3rlpy/bin/python rl/13_offline_train.py --algo cql --size 1M --seed 0
  .venv-d3rlpy/bin/python rl/13_offline_train.py --algo bc --smoke
        500 steps on rl/models/13_dataset_smoke.npz, *_smoke, not logged, prints steps per second
"""

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

STEPS = 100_000
LOG_DIR = f"{C.MODEL_DIR}/13_d3rlpy_logs"  # gitignored


def arg(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    algo_name = arg("--algo", str, None)
    assert algo_name in {"bc", "cql", "iql"}, "--algo bc | cql | iql"
    dry = "--smoke" in sys.argv
    size = arg("--size", str, "1M")
    seed = arg("--seed", int, 0)
    threads = arg("--threads", int, 4)
    steps = arg("--steps", int, 500) if dry else STEPS
    assert size in C.SIZES
    name = C.model_name(algo_name, size, seed, "_smoke" if dry else "")
    path = f"{C.MODEL_DIR}/{name}.pt"
    if os.path.exists(path) and not dry:
        print(f"{name}: exists, skipped")
        return

    import d3rlpy
    import torch
    from d3rlpy.logging import FileAdapterFactory

    torch.set_num_threads(threads)
    d3rlpy.seed(seed)
    data = np.load(f"{C.MODEL_DIR}/13_dataset_smoke.npz" if dry else C.DATASET)
    ep = int(data["ep_steps"])
    n = len(data["observations"]) if dry else C.SIZES[size] * ep
    assert n <= len(data["observations"]) and n % ep == 0
    timeouts = np.zeros(n, np.float32)
    timeouts[ep - 1 :: ep] = 1.0
    dataset = d3rlpy.dataset.MDPDataset(
        observations=data["observations"][:n],
        actions=data["actions"][:n],
        rewards=data["rewards"][:n],
        terminals=np.zeros(n, np.float32),
        timeouts=timeouts,
    )
    assert dataset.size() == n // ep, (dataset.size(), n // ep)
    config = {"bc": d3rlpy.algos.BCConfig, "cql": d3rlpy.algos.CQLConfig,
              "iql": d3rlpy.algos.IQLConfig}[algo_name]()
    algo = config.create(device="cpu")
    t0 = time.time()
    algo.fit(
        dataset,
        n_steps=steps,
        n_steps_per_epoch=min(steps, 10_000),
        experiment_name=name,
        with_timestamp=False,
        logger_adapter=FileAdapterFactory(root_dir=LOG_DIR),
        show_progress=False,
        save_interval=10**9,  # the policy is exported below; no intermediate models
    )
    train_s = time.time() - t0
    algo.save_policy(path)
    x = data["observations"][:1000]
    with torch.no_grad():
        a_script = torch.jit.load(path)(torch.as_tensor(x)).numpy()
    diff = float(np.max(np.abs(a_script - algo.predict(x))))
    assert diff < 1e-5, diff
    assert float(np.abs(a_script).max()) <= 1.0 + 1e-6
    with open(path[:-3] + ".json", "w") as f:
        json.dump({"algo": algo_name, "d3rlpy": d3rlpy.__version__, "torch": torch.__version__,
                   "size": size, "transitions": n, "seed": seed, "steps": steps,
                   "threads": threads, "config": json.loads(config.serialize())}, f, indent=1)
    print(f"{name}: {steps} gradient steps on {n} transitions in {train_s:.0f} s"
          f" = {steps / train_s:.0f} steps/s -> {path}; TorchScript vs algo.predict largest"
          f" difference {diff:.1e}")
    if not dry:
        C.append_log("offline", algo_name, size, seed, steps, train_s, "cpu")


if __name__ == "__main__":
    main()
