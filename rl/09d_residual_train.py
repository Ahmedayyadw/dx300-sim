"""Lab 09d: train a residual SAC on top of the sched controller of Lab 09c3.

One variant, seeds 0, 1, 2, 200k steps each. Env, observation, residual bound and reward:
rl/lab09d_common.py (ResidualSchedEnv). SAC settings as in 08b (SB3 defaults, device cpu).
  u = clip(u_sched + 0.2 a, -1, 1)
  r = -(e^2 + 0.1 |e| + 0.3 P_pump / 2.5e5 + 0.01 (a[k] - a[k-1])^2)
P_pump is the pump power of Lab 09b; dp_valve and eta_pump are ASSUMED - not from manual
(rl/09b_hyd_params.py).
Training conditions per episode: p = 0.7 one of the nine combined cases of 09c2/09c3 (sensor
noise, 5 % torque noise, J and b scaled, tau_d; J x0.38 with b x2 left out), p = 0.3 one of the
ten single rows of 09c3; p = 0.5 payload step.

Saves rl/models/09d_residual_w0.3_s{seed}.zip and appends (config, w, seed, steps, train_s) to
rl/09d_train_log.csv. Resumable: a model already saved is skipped.

Usage: python rl/09d_residual_train.py [--seed N] [--device cpu|cuda] [--dry-run]
  --seed N    train this seed only (default: 0, 1, 2 one after the other)
  --dry-run   3,000 steps, seed 0, saved as 09d_residual_w0.3_s0_dry.zip, not logged; then
              checks Sched against rl/09c3_ff_torque_limit.csv and the training env against
              the evaluation path of Lab 09e
"""
import csv
import os
import sys
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from lab09_common import HYD, NEW, rollout
from lab09d_common import (A_MAX, COMBINED, SCHED_CSV, SINGLES, W_PUMP, ResidualSched,
                           ResidualSchedEnv, Sched, make_env, run_case)


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


DRY_RUN = "--dry-run" in sys.argv
DEVICE = arg("--device", "cpu")
SEEDS = [0] if DRY_RUN else [int(arg("--seed", -1))] if "--seed" in sys.argv else [0, 1, 2]
STEPS = 3_000 if DRY_RUN else 200_000
SUFFIX = "_dry" if DRY_RUN else ""
MODEL_DIR = "rl/models"
LOG_PATH = "rl/09d_train_log.csv"


class Progress(BaseCallback):
    """Prints steps and elapsed time every `every` steps."""

    def __init__(self, name, every):
        super().__init__()
        self.name, self.every, self.t0 = name, every, time.time()

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            print(f"{self.name}: {self.num_timesteps} steps, {time.time() - self.t0:.0f} s", flush=True)
        return True


print(f"dp_valve {HYD.DP_VALVE_PA / 1e5:.0f} bar, eta_pump {HYD.ETA_PUMP}: ASSUMED - not from manual")
os.makedirs(MODEL_DIR, exist_ok=True)
for s in SEEDS:
    name = f"09d_residual_w{W_PUMP}_s{s}{SUFFIX}"
    path = f"{MODEL_DIR}/{name}.zip"
    if os.path.exists(path) and not DRY_RUN:
        print(f"{name}: exists, skipped")
        continue
    env = ResidualSchedEnv()
    model = SAC("MlpPolicy", env, device=DEVICE, seed=s, verbose=0, tensorboard_log="rl/tb")
    t0 = time.time()
    model.learn(total_timesteps=STEPS, tb_log_name=name, callback=Progress(name, STEPS // 10))
    train_s = time.time() - t0
    model.save(path)
    print(f"{name}: {STEPS} steps in {train_s:.0f} s ({STEPS / train_s:.0f} steps/s, {DEVICE}) -> {path}")
    if not DRY_RUN:
        try:
            with open(LOG_PATH, "x", newline="") as f:
                csv.writer(f).writerow(["config", "w", "seed", "steps", "train_s"])
        except FileExistsError:
            pass
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(["residual", W_PUMP, s, STEPS, f"{train_s:.1f}"])

if DRY_RUN:
    # 1. Sched must reproduce the pid_gain_sched rows of 09c3 (two rows here, all 38 in 09e)
    with open(SCHED_CSV) as f:
        ref = {(r["condition"], r["variant"]): r for r in csv.DictReader(f) if r["w"] == "pid_gain_sched"}
    cases = dict(SINGLES + COMBINED)
    for cond_name, variant in [("tau_d", "baseline"), ("all_n1.0_J1.5_b0.5", "payload_x1.2")]:
        got = run_case(lambda tau_noise: Sched(tau_noise=tau_noise), cases[cond_name], variant)
        for key, val in got.items():
            want = float(ref[(cond_name, variant)][key])
            assert abs(val - want) <= 1e-9 * max(1.0, abs(want)), f"{cond_name} {variant} {key}"
        print(f"check 1: Sched {cond_name} {variant} = {SCHED_CSV} pid_gain_sched: miss"
              f" {got['share_over_20cm'] * 100:.0f}%, E_mech {got['mean_kJ']:.1f} kJ,"
              f" E_pump {got['mean_pump_kJ']:.1f} kJ")

    # 2. training env and evaluation path (make_env + ResidualSched + rollout) must give the
    #    same episode. Noise-free condition: the two paths use different noise generators.
    cond, (th0, target) = {"j": 1.5, "b": 0.5, "tau_d": True}, NEW[68]
    r = rollout(ResidualSched(model), make_env("payload_x1.2", **cond), th0, target)
    env = ResidualSchedEnv()
    obs, _ = env.reset(seed=0, options={"cond": cond, "payload": (100, 1.2), "th0": th0, "target": target})
    ret, trunc = 0.0, False
    while not trunc:
        obs, reward, _, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
        ret += reward
    err_deg = abs(np.degrees(env.target - env.th))
    assert abs(err_deg - r["err_deg"]) < 1e-9 and abs(env.energy_J / 1000 - r["kJ"]) < 1e-9, \
        "training env and evaluation path differ"
    print(f"check 2: training env = evaluation path on scenario 68, J x1.5, b x0.5, tau_d, payload:"
          f" final error {err_deg:.6f} deg = {r['err_deg']:.6f} deg, E_mech {env.energy_J / 1000:.6f} kJ"
          f" = {r['kJ']:.6f} kJ, episode return {ret:.2f}")
    print(f"DRY RUN: {STEPS} steps, residual bound {A_MAX}, {LOG_PATH} not written")
