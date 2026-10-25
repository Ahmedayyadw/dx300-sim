"""Step 4: first RL agent. SAC learns to swing up and hold a pendulum (~ one boom joint)."""
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, device="cpu", seed=0, verbose=0, tensorboard_log="rl/tb")

before, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"before training: mean episode reward = {before:.0f}")

t0 = time.time()
model.learn(total_timesteps=20_000, tb_log_name="sac_pendulum")
print(f"training time: {time.time() - t0:.0f} s")

after, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"after training:  mean episode reward = {after:.0f}")
model.save("rl/sac_pendulum")
print("model saved: rl/sac_pendulum.zip")
