"""Lab 13a: online RL other than SAC, in the setup of the best Lab 12 SAC.

TD3 (off-policy) and PPO (on-policy) from stable-baselines3 as residual on the balanced PID:
BoomEnvRealRobust(w_energy=0.03, residual=True), the env of rl/12_robust_train.py --config residual
--w 0.03, device cpu, MlpPolicy, 100,000 env steps, seeds 0, 1, 2.

Changes from the library defaults (every one; all else is the SB3 default):
  TD3  action_noise = NormalActionNoise(mean 0, sigma 0.1), the SB3 TD3 doc example. The default
       is None: a deterministic policy without exploration.
  PPO  VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0), the SB3 example "Normalizing
       input features". The observation statistics are saved next to the model
       (*_norm.npz) and applied frozen at evaluation; the reward function is unchanged, its
       scaling is used during training only.
PPO collects 2,048 steps per update, so a 100,000-step run stops at 100,352 steps (49 updates).
--steps 1000000 is the extra PPO row "not equal budget" (ASSUMED).

Saves rl/models/13_{algo}_{100k|1M}_s{seed}.zip and appends the wall-clock to rl/13_train_log.csv.
Resumable: a model already saved is skipped.

Usage (repo root, .venv active):
  python rl/13_online_train.py --algo td3 --seed 0
  python rl/13_online_train.py --algo ppo --seed 0 [--steps 1000000]
  python rl/13_online_train.py --algo td3 --smoke     2,000 steps (PPO 4,096), *_smoke, not logged
"""

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lab13_common", os.path.join(HERE, "13_common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["lab13_common"] = C
_spec.loader.exec_module(C)


def arg(flag, cast, default):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def build(algo, seed):
    from stable_baselines3 import PPO, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    kw = dict(device="cpu", seed=seed, verbose=0, tensorboard_log="rl/tb")
    if algo == "td3":
        noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
        return TD3("MlpPolicy", C.make_env(), action_noise=noise, **kw), None
    venv = DummyVecEnv([lambda: Monitor(C.make_env())])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return PPO("MlpPolicy", venv, **kw), venv


def save_norm(venv, path):
    """Frozen observation statistics of VecNormalize, and the proof that NormPolicy applies them."""
    np.savez(path, mean=venv.obs_rms.mean, var=venv.obs_rms.var, clip=venv.clip_obs, eps=venv.epsilon)
    x = np.random.default_rng(0).normal(0.0, 2.0, (1000, 4)).astype(np.float32)
    venv.training = False
    pol = C.NormPolicy(None, path)
    diff = float(np.max(np.abs(pol.normalize(x) - venv.normalize_obs(x))))
    assert diff < 1e-6, diff
    return diff


def main():
    algo = arg("--algo", str, None)
    assert algo in {"td3", "ppo"}, "--algo td3 | ppo"
    dry = "--smoke" in sys.argv
    seed = arg("--seed", int, 0)
    steps = arg("--steps", int, 100_000)
    assert steps in C.STEP_TAGS, f"--steps must be one of {sorted(C.STEP_TAGS)}"
    assert algo == "ppo" or steps == 100_000, "the 1M row is PPO only"
    tag = C.STEP_TAGS[steps]
    if dry:
        steps = 4_096 if algo == "ppo" else 2_000
    name = C.model_name(algo, tag, seed, "_smoke" if dry else "")
    path = f"{C.MODEL_DIR}/{name}.zip"
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    if os.path.exists(path) and not dry:
        print(f"{name}: exists, skipped")
        return
    model, venv = build(algo, seed)
    t0 = time.time()
    model.learn(total_timesteps=steps, tb_log_name=name)
    train_s = time.time() - t0
    model.save(path)
    msg = f"{name}: {model.num_timesteps} steps in {train_s:.0f} s -> {path}"
    if venv is not None:
        diff = save_norm(venv, path[:-4] + "_norm.npz")
        msg += f"; normalization saved, NormPolicy vs VecNormalize largest difference {diff:.1e}"
    print(msg)
    if not dry:
        C.append_log("online", algo, tag, seed, model.num_timesteps, train_s, "cpu")


if __name__ == "__main__":
    main()
