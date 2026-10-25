"""1-DOF DX300-like boom environment. Parameters are rough placeholders."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BoomEnv(gym.Env):
    def __init__(self, w_energy=0.05):
        self.J, self.m, self.Lc, self.b = 8e4, 5000.0, 3.5, 6e5   # kg*m^2, kg, m, N*m*s/rad
        self.tau_max, self.dt, self.max_steps = 5e5, 0.05, 200    # N*m, s, steps (10 s)
        self.th_min, self.th_max = -0.5, 1.0                      # rad, mechanical limits
        self.w_energy = w_energy
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)

    def _obs(self):
        return np.array([self.th, self.om, self.target - self.th], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = self.np_random.uniform(-0.4, 0.9)
        self.target = self.np_random.uniform(-0.3, 0.9)
        self.om, self.k, self.energy_J = 0.0, 0, 0.0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(action[0], -1, 1))
        tau = u * self.tau_max
        tau_g = self.m * 9.81 * self.Lc * np.cos(self.th)
        self.om += (tau - tau_g - self.b * self.om) / self.J * self.dt
        self.th += self.om * self.dt
        if self.th <= self.th_min or self.th >= self.th_max:      # hit end stop
            self.th, self.om = float(np.clip(self.th, self.th_min, self.th_max)), 0.0
        self.energy_J += abs(tau * self.om) * self.dt              # mechanical work
        err = self.target - self.th
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * u ** 2)
        self.k += 1
        return self._obs(), reward, False, self.k >= self.max_steps, {}
