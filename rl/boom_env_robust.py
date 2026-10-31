"""Lab 08b: BoomEnv with full-range scenarios, physics randomization and optional PID residual.

RobustBoomEnv(w_energy, residual=False, randomize=True):
  reset      theta0 and target uniform in [th_min + 0.05, th_max - 0.05]
  randomize  each reset: m = nominal x U(0.4, 1.3); p = 0.5 angle-dependent torque limit
             (5e5 x |d(theta)| / |d(0)|, as tau_d in 06); p = 0.3 payload step at a random
             step in 50..150 that multiplies m by U(0.9, 1.3)
  obs        [theta, omega, e, int(e) dt], integral clipped to +-1, reset each episode
  reward     -(e^2 + 0.01 omega^2 + w |tau omega| / 2.5e5), mechanical power instead of u^2
  residual   u = clip(u_PID + a, -1, 1), PID = pid_ki from 07 with NOMINAL gravity feed-forward;
             PID integral frozen while u is saturated (anti-windup)
Dynamics, end stops, energy_J and the final-error measurement are the same as BoomEnv.
"""
import os
import sys
import numpy as np
from gymnasium import spaces
from boom_env import BoomEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for dx300sim
from dx300sim import dx300_params as P
from dx300sim.boom_geometry import moment_arm

PID_GAINS = (2.0, 0.1, 1.0)   # Kp, Ki, Kd: pid_ki tuned in rl/07_pid_baseline.py
POWER_SCALE_W = 2.5e5         # reward scale for |tau omega|


def d(theta):
    return moment_arm(theta, P.B_XY_M, P.C_M, P.DELTA_RAD)


D0_M = abs(d(0.0))   # about 0.898 m


class RobustBoomEnv(BoomEnv):
    def __init__(self, w_energy=0.05, residual=False, randomize=True):
        super().__init__(w_energy=w_energy)
        self.residual, self.randomize = residual, randomize
        self.m_nom, self.tau_max_nom = self.m, self.tau_max
        self.u_ff_scale = self.m_nom * 9.81 * self.Lc / self.tau_max_nom   # nominal feed-forward
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)

    def _obs(self):
        return np.array([self.th, self.om, self.target - self.th, self.integ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.integ, self.pid_integ = 0.0, 0.0
        super().reset(seed=seed, options=options)
        lo, hi = self.th_min + 0.05, self.th_max - 0.05
        self.th, self.target = self.np_random.uniform(lo, hi), self.np_random.uniform(lo, hi)
        self.m, self.tau_max = self.m_nom, self.tau_max_nom
        self.tau_d, self.payload_k, self.payload_factor = False, None, 1.0
        if self.randomize:
            self.m = self.m_nom * self.np_random.uniform(0.4, 1.3)
            self.tau_d = self.np_random.random() < 0.5
            if self.np_random.random() < 0.3:
                self.payload_k = int(self.np_random.integers(50, 151))
                self.payload_factor = self.np_random.uniform(0.9, 1.3)
        return self._obs(), {}

    def step(self, action):
        if self.k == self.payload_k:
            self.m *= self.payload_factor
        if self.tau_d:
            self.tau_max = self.tau_max_nom * abs(d(self.th)) / D0_M   # limit at the pre-step angle
        u = float(np.clip(action[0], -1, 1))
        if self.residual:
            kp, ki, kd = PID_GAINS
            e = self.target - self.th
            u_raw = kp * e + ki * self.pid_integ - kd * self.om + self.u_ff_scale * np.cos(self.th) + u
            if abs(u_raw) < 1.0:                                       # anti-windup
                self.pid_integ += e * self.dt
            u = float(np.clip(u_raw, -1, 1))
        tau = u * self.tau_max
        tau_g = self.m * 9.81 * self.Lc * np.cos(self.th)
        self.om += (tau - tau_g - self.b * self.om) / self.J * self.dt
        self.th += self.om * self.dt
        if self.th <= self.th_min or self.th >= self.th_max:      # hit end stop
            self.th, self.om = float(np.clip(self.th, self.th_min, self.th_max)), 0.0
        power_W = abs(tau * self.om)
        self.energy_J += power_W * self.dt                        # mechanical work, same as BoomEnv
        err = self.target - self.th
        self.integ = float(np.clip(self.integ + err * self.dt, -1.0, 1.0))
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * power_W / POWER_SCALE_W)
        self.k += 1
        return self._obs(), reward, False, self.k >= self.max_steps, {}
