"""Lab 14: shared names and the corrected-reward env. No training here.

Imported by every rl/14_*.py with importlib (file name starts with a digit). Top-level imports are the
standard library and numpy only (via rl/14_reward_probe.py), so rl/14_offline_train.py can load it in
.venv-d3rlpy.

Corrected reward (Lab 14a, approved; constants read from rl/14_reward_probe.py, single source):
    r = -(e^2 + 0.01 omega^2 + W P / P_REF + K h) / SCALE
    h = max(0, |e| - TOL) if |omega| < OM_S else 0,   TOL = TOL_CM at the tip / L_BOOM
e, omega after the step (true values, as the Lab 13 reward), P = delivered mechanical power of the step
(info["P_W"]). Everything else of the Lab 13 setup is unchanged: BoomEnvRealRobust(residual=True),
residual on the balanced PID, randomize on, train_noise off.

Model files (rl/models/, gitignored):
  14_{sac,td3}_{tag}_s{s}.zip          online, tag 100k, 100k_lr0.3, 100k_lr3
  14_{iql,cql}_1M_s{s}.pt              offline on the relabelled 1M dataset (TorchScript)
  14_bc_eco500k_s{s}.pt                BC on the economical-PID episodes of the 1M dataset
  14_dataset_1M.npz                    relabelled dataset (rl/14_relabel.py)
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fname):
    """Import rl/<fname> as module <name> (once)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C13 = load("lab13_common", "13_common.py")
_A = load("lab14_reward_probe", "14_reward_probe.py")  # stdlib + numpy at top level, guarded main
W, K, OM_S, TOL_CM, SCALE = _A.W_NEW, _A.K_NEW, _A.OM_NEW, _A.TOL_CM, _A.SCALE

SEEDS = (0, 1, 2)
EP_STEPS = C13.EP_STEPS
MODEL_DIR = C13.MODEL_DIR
TRAIN_LOG = "rl/14_train_log.csv"
DATASET = f"{MODEL_DIR}/14_dataset_1M.npz"
LR_FACTORS = {"100k": 1.0, "100k_lr0.3": 0.3, "100k_lr3": 3.0}


def model_name(algo, tag, seed, suffix=""):
    return f"14_{algo}_{tag}_s{seed}{suffix}"


def append_log(group, algo, tag, seed, steps, train_s, device):
    C13.append_log(group, algo, tag, seed, steps, train_s, device, path=TRAIN_LOG)


def reward14(e, om, p_w, p_ref_w, l_boom_m):
    """Corrected reward of one step. Plain floats (the relabel check uses the same function)."""
    tol = TOL_CM / 100.0 / l_boom_m
    h = max(0.0, abs(e) - tol) if abs(om) < OM_S else 0.0
    return -(e**2 + 0.01 * om**2 + W * p_w / p_ref_w + K * h) / SCALE


def make_env(**kw):
    """Lab 13 env with the corrected reward (kw: extra BoomEnvRealRobust arguments, e.g. randomize=False). info["r_lab13"] keeps the Lab 13 reward of the step."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import boom_env_real_robust as RR

    class BoomEnv14(RR.BoomEnvRealRobust):
        def step(self, action):
            obs, r13, term, trunc, info = super().step(action)
            r = reward14(self.target - self.th, self.om, info["P_W"], RR.P_REF_W, RR.R.L_BOOM_M)
            info["r_lab13"] = r13
            return obs, r, term, trunc, info

    return BoomEnv14(w_energy=C13.W, residual=True, **kw)
