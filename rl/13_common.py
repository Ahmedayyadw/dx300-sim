"""Lab 13: shared names, training log and policy adapters. No training here.

Imported by every rl/13_*.py with importlib (the file name starts with a digit). Top-level imports
are the standard library only, so rl/13_offline_train.py can load it in .venv-d3rlpy, which has
neither stable-baselines3 nor the env's gymnasium version.

Setup of the best Lab 12 SAC (rl/12_robust_train.py --config residual --w 0.03), kept identical:
  env     BoomEnvRealRobust(w_energy=0.03, residual=True): residual on the balanced PID,
          randomize on, train_noise off
  budget  100,000 env steps (online) / 100,000 gradient steps (offline, ASSUMED), seeds 0, 1, 2

Model files (rl/models/, gitignored):
  13_td3_100k_s{s}.zip                       TD3
  13_ppo_{100k,1M}_s{s}.zip + _norm.npz      PPO and its frozen VecNormalize statistics
  13_{bc,cql,iql}_{100k,300k,1M}_s{s}.pt     offline policies, TorchScript (d3rlpy save_policy)
  13_dataset_1M.npz                          offline dataset (rl/13_dataset.py)

Adapters give every policy the call the Lab 12 controller uses:
  policy.predict(x, deterministic=True) -> (action array of shape (1,), None)
"""

import csv
import fcntl
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = 0.03
SEEDS = (0, 1, 2)
EP_STEPS = 200  # BoomEnvReal.max_steps, checked in rl/13_dataset.py
MODEL_DIR = "rl/models"
TRAIN_LOG = "rl/13_train_log.csv"
DATASET = f"{MODEL_DIR}/13_dataset_1M.npz"
SIZES = {"100k": 500, "300k": 1500, "1M": 5000}  # dataset size: episodes
STEP_TAGS = {100_000: "100k", 1_000_000: "1M"}
LOG_FIELDS = ("group", "algo", "tag", "seed", "steps", "train_s", "device")


def load(name, fname):
    """Import rl/<fname> as module <name> (once)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_name(algo, tag, seed, suffix=""):
    return f"13_{algo}_{tag}_s{seed}{suffix}"


def append_log(group, algo, tag, seed, steps, train_s, device, path=TRAIN_LOG):
    """One line per finished run. File lock: three runs in parallel write the same file."""
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(LOG_FIELDS)
        w.writerow([group, algo, tag, seed, steps, f"{train_s:.1f}", device])


def make_env():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from boom_env_real_robust import BoomEnvRealRobust

    return BoomEnvRealRobust(w_energy=W, residual=True)


# ----------------------------------------------------------------------------- policy adapters
class NormPolicy:
    """PPO trained under VecNormalize: the frozen observation statistics, then the SB3 policy.

    Same formula as VecNormalize.normalize_obs: clip((x - mean) / sqrt(var + eps), -clip, clip).
    rl/13_online_train.py asserts this equality when it saves the statistics.
    """

    def __init__(self, model, norm_path):
        import numpy as np

        self.np, self.model = np, model
        d = np.load(norm_path)
        self.mean, self.var = d["mean"], d["var"]
        self.clip, self.eps = float(d["clip"]), float(d["eps"])

    def normalize(self, x):
        np = self.np
        z = (x - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def predict(self, x, deterministic=True):
        return self.model.predict(self.normalize(x), deterministic=deterministic)


class ScriptPolicy:
    """TorchScript greedy policy exported by d3rlpy save_policy."""

    def __init__(self, path):
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.policy = torch.jit.load(path, map_location="cpu")
        self.policy.eval()

    def predict(self, x, deterministic=True):
        with self.torch.no_grad():
            a = self.policy(self.torch.as_tensor(x, dtype=self.torch.float32).reshape(1, -1))
        return a.numpy()[0], None


_POLICIES = {}


def load_policy(kind, name):
    """kind: td3, ppo (with _norm.npz) or script. Cached per process."""
    key = (kind, name)
    if key not in _POLICIES:
        import torch

        torch.set_num_threads(1)
        base = f"{MODEL_DIR}/{name}"
        if kind == "script":
            _POLICIES[key] = ScriptPolicy(base + ".pt")
        elif kind == "td3":
            from stable_baselines3 import TD3

            _POLICIES[key] = TD3.load(base + ".zip", device="cpu")
        elif kind == "ppo":
            from stable_baselines3 import PPO

            _POLICIES[key] = NormPolicy(PPO.load(base + ".zip", device="cpu"), base + "_norm.npz")
        else:
            raise ValueError(kind)
    return _POLICIES[key]


def policy_file(kind, name):
    return f"{MODEL_DIR}/{name}" + (".pt" if kind == "script" else ".zip")
