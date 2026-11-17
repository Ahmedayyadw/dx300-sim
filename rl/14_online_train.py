"""Lab 14b/14d: SAC and TD3 with the corrected reward, the Lab 13 setup otherwise.

Env: rl/14_common.make_env() = BoomEnvRealRobust(residual=True) with the Lab 14a reward. Device cpu,
MlpPolicy, 100,000 env steps.
  SAC  SB3 defaults (as rl/12_robust_train.py).
  TD3  action_noise = NormalActionNoise(0, 0.1), as rl/13_online_train.py. Nothing else changed.
  --lr x   learning_rate = x * the SB3 default of the algorithm (read from the class signature);
           x in {0.3, 3} is Lab 14d (1 seed). Default x = 1 (Lab 14b, seeds 0, 1, 2).

Saves rl/models/14_{algo}_{100k|100k_lr0.3|100k_lr3}_s{seed}.zip, appends the wall-clock to
rl/14_train_log.csv. Resumable: a model already saved is skipped.

Usage (repo root, .venv active, OMP_NUM_THREADS=1 MKL_NUM_THREADS=1):
  python rl/14_online_train.py --algo sac --seed 0
  python rl/14_online_train.py --algo td3 --seed 0 --lr 3
  python rl/14_online_train.py --algo sac --smoke      2,000 steps, *_smoke, not logged
"""

import importlib.util
import inspect
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab14_common", os.path.join(HERE, "14_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab14_common"] = C
_spec.loader.exec_module(C)


def arg(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    algo = arg("--algo", str, None)
    assert algo in {"sac", "td3"}, "--algo sac | td3"
    dry = "--smoke" in sys.argv
    seed = arg("--seed", int, 0)
    lr_x = arg("--lr", float, 1.0)
    tag = {v: k for k, v in C.LR_FACTORS.items()}.get(lr_x)
    assert tag is not None, f"--lr must be one of {sorted(C.LR_FACTORS.values())}"
    steps = 2_000 if dry else 100_000
    name = C.model_name(algo, tag, seed, "_smoke" if dry else "")
    path = f"{C.MODEL_DIR}/{name}.zip"
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    if os.path.exists(path) and not dry:
        print(f"{name}: exists, skipped")
        return
    cls = {"sac": SAC, "td3": TD3}[algo]
    lr0 = inspect.signature(cls.__init__).parameters["learning_rate"].default
    kw = dict(device="cpu", seed=seed, verbose=0, tensorboard_log="rl/tb", learning_rate=lr_x * lr0)
    if algo == "td3":
        kw["action_noise"] = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
    model = cls("MlpPolicy", C.make_env(), **kw)
    t0 = time.time()
    model.learn(total_timesteps=steps, tb_log_name=name)
    train_s = time.time() - t0
    model.save(path)
    print(f"{name}: {model.num_timesteps} steps in {train_s:.0f} s ({model.num_timesteps / train_s:.0f} steps/s),"
          f" learning_rate {kw['learning_rate']:.2e} (= {lr_x:g} x SB3 default {lr0:.0e}) -> {path}")
    if not dry:
        C.append_log("online", algo, tag, seed, model.num_timesteps, train_s, "cpu")


if __name__ == "__main__":
    main()
