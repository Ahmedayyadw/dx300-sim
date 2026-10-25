"""Step 5: SAC learns to position the DX300 boom with low hydraulic effort."""
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from boom_env import BoomEnv

env = BoomEnv()
check_env(env)                                  # verifies the env follows the Gymnasium API
print("environment check passed")

model = SAC("MlpPolicy", env, device="cpu", seed=0, verbose=0, tensorboard_log="rl/tb")
t0 = time.time()
model.learn(total_timesteps=30_000, tb_log_name="sac_boom")
print(f"training time: {time.time() - t0:.0f} s")
model.save("rl/sac_boom")

# test scenario: lift the boom from -0.3 rad (-17 deg) to 0.6 rad (+34 deg)
env.reset(seed=1)
env.th, env.om, env.target = -0.3, 0.0, 0.6
obs = env._obs()
t, th, u = [], [], []
for _ in range(env.max_steps):
    a, _ = model.predict(obs, deterministic=True)
    obs, _, _, trunc, _ = env.step(a)
    t.append(env.k * env.dt); th.append(np.degrees(env.th)); u.append(float(a[0]))
    if trunc:
        break

print(f"final angle: {th[-1]:.1f} deg (target {np.degrees(0.6):.1f} deg)")
print(f"mechanical energy used: {env.energy_J / 1000:.0f} kJ")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax[0].plot(t, th, label="boom angle"); ax[0].axhline(np.degrees(0.6), ls="--", c="r", label="target")
ax[0].set_ylabel("angle [deg]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t, u); ax[1].set_ylabel("valve command u"); ax[1].set_xlabel("time [s]"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("rl/02_boom_test.png", dpi=150)
print("plot saved: rl/02_boom_test.png")
