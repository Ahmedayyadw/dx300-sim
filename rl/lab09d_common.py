"""Shared code for Labs 09d and 09e: the sched controller, the residual SAC wrapper around it,
the stress conditions and the training env.

Nothing here trains or writes files. 09c2_stress.py and 09c3_ff_torque_limit.py cannot be
imported (they run on import), so the env mixins, the conditions and the sched controller
(pid_gain_sched) are copied from them. Lab 09e asserts that Sched reproduces the pid_gain_sched
rows of rl/09c3_ff_torque_limit.csv.

Residual controller (Lab 09d):
  u[k] = clip(u_sched[k] + A_MAX a[k], -1, 1),  a in [-1, 1] from the SAC policy
  k_hat reads tau[k] / u[k] with the total command u[k], not the sched part alone.
Policy observation (7 values, built by policy_obs for training and evaluation alike):
  theta, omega, e = target - theta      measured (noisy when the condition has sensor noise)
  int(e) dt                             from the measured e, clipped to +-1, as in Lab 08b
  m_hat / m_nom, k_hat / 5e5            the two estimates of the sched controller
  a[k-1]                                previous residual (the reward has an action-rate term)
Reward of ResidualSchedEnv, true state:
  r = -(e^2 + 0.1 |e| + w P_pump / 2.5e5 + 0.01 (a[k] - a[k-1])^2)
  P_pump = (max(tau omega, 0) + dp_valve A |v_cyl|) / eta_pump     pump power of Lab 09b
dp_valve and eta_pump from rl/09b_hyd_params.py (ASSUMED - not from manual).
"""
import csv
import os
import numpy as np
from gymnasium import spaces
from boom_env import BoomEnv
from lab09_common import (A_ANN_M2, A_CAP_M2, B_NOM, G, HYD, J_NOM, LC, M_NOM, NEW, PID, TAU_MAX,
                          PayloadBoomEnv, cm, d, e_pump_J, rollout)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHED_CSV = "rl/09c3_ff_torque_limit.csv"   # read only: sched gains and reference rows
T_F_S = 0.2
TAIL_STEPS = 40                             # last 2 s, as in 09c2/09c3
U_MIN = 0.05                                # no gain update below this command
D0_M = abs(d(0.0))                          # about 0.898 m, as in 06
A_MAX = 0.2                                 # residual bound, in command units
W_PUMP = 0.3                                # weight of the pump power in the reward
POWER_SCALE_W = 2.5e5                       # power scale in the reward, as in Lab 08b
W_ABS = 0.1                                 # weight of |e|
W_RATE = 0.01                               # weight of the action-rate term
P_COMBINED = 0.7                            # training: share of episodes with a combined case
P_PAYLOAD = 0.5                             # training: share of episodes with a payload step
VARIANT_NAMES = ["baseline", "payload_x1.2"]

with open(os.path.join(HERE, os.path.basename(SCHED_CSV))) as _f:
    _row = next(csv.DictReader(_f))
GAINS = (float(_row["kp"]), float(_row["ki"]), float(_row["kd"]))   # Lab 07 "pid" gains


# --- env pieces, copied from 09c2_stress.py ---
class NoisyMixin:
    """Gaussian noise on observed theta and omega. Copied from NoisyBoomEnv in 06b."""

    def _obs(self):
        th = self.th + self.noise_rng.normal(0.0, self.sigma_th)
        om = self.om + self.noise_rng.normal(0.0, self.sigma_om)
        return np.array([th, om, self.target - th], dtype=np.float32)


class TauDMixin:
    """Angle-dependent torque limit. Copied from TauDBoomEnv in 06."""

    def step(self, action):
        self.tau_max = 5e5 * abs(d(self.th)) / D0_M   # limit at the pre-step angle
        return super().step(action)


class SubstepBoomEnv(BoomEnv):
    """BoomEnv.step with the dynamics integrated in SUBSTEPS steps of dt / SUBSTEPS, command held
    (loop as in boom_env_robust.py). energy_J uses the end-of-step omega, as rollout() does."""
    SUBSTEPS = 10

    def step(self, action):
        u = float(np.clip(action[0], -1, 1))
        tau = u * (self.tau_max if self.tau_max_fn is None else self.tau_max_fn(self.th))
        h = self.dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            tau_g = self.m * 9.81 * self.Lc * np.cos(self.th)
            self.om += (tau - tau_g - self.b * self.om) / self.J * h
            self.th += self.om * h
            if self.th <= self.th_min or self.th >= self.th_max:      # hit end stop
                self.th, self.om = float(np.clip(self.th, self.th_min, self.th_max)), 0.0
        self.energy_J += abs(tau * self.om) * self.dt
        err = self.target - self.th
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * u ** 2)
        self.k += 1
        return self._obs(), reward, False, self.k >= self.max_steps, {}


def make_env(variant, sigma=0.0, j=1.0, b=1.0, tau_d=False, substeps=1, **_):
    core = SubstepBoomEnv if substeps > 1 else BoomEnv
    bases = (([TauDMixin] if tau_d else []) + ([NoisyMixin] if sigma > 0 else [])
             + ([PayloadBoomEnv] if variant == "payload_x1.2" else []) + [core])
    env = type("StressEnv", tuple(bases), {})()
    env.J *= j
    env.b *= b
    if sigma > 0:
        env.sigma_th = np.radians(sigma)
        env.sigma_om = env.sigma_th / env.dt * 0.1
        env.noise_rng = np.random.default_rng(0)
    return env


# --- conditions, copied from 09c3_ff_torque_limit.py (the rows of its two tables) ---
SINGLES = [("none", {})] + [(f"noise_{s}", {"sigma": s}) for s in (0.2, 0.5, 1.0)] + [
    ("tau_noise_5pct", {"tau_noise": 0.05}),
    ("J_x0.38", {"j": 0.38}), ("J_x1.5", {"j": 1.5}),
    ("b_x0.5", {"b": 0.5}), ("b_x2", {"b": 2.0}),
    ("tau_d", {"tau_d": True}),
]
# J x0.38 with b x2 is left out: Euler step at its stability limit (09c2, section 5)
COMBINED = [(f"all_n{s}_J{j}_b{b}", {"sigma": s, "tau_noise": 0.05, "j": j, "b": b, "tau_d": True})
            for s in (0.2, 0.5, 1.0) for j in (0.38, 1.5) for b in (0.5, 2.0)
            if not (j == 0.38 and b == 2.0)]


# --- controllers ---
class Sched(PID):
    """pid_gain_sched of Lab 09c3: payload-aware PID, feed-forward and PID terms scaled by
    5e5 / k_hat. Copied from FFPID (mode "gain_all"); act() takes an optional residual."""

    def __init__(self, gains=GAINS, t_f=T_F_S, tau_noise=0.0):
        self.tau_noise, self.tau_rng = tau_noise, np.random.default_rng(1)
        super().__init__(gains, payload_aware=True, t_f=t_f)

    def reset(self):
        super().reset()
        self.k_hat, self.u_prev, self.hist = TAU_MAX, 0.0, []

    def act(self, obs, residual=0.0):
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])   # what the PID observes
        k = self.k_hat
        fb = (self.kp * e + self.ki * self.integ + self.kd * (-om)) * TAU_MAX / k
        u_raw = fb + self.m_hat * G * LC / k * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup (sched part)
            self.integ += e * self.dt
        self.prev = (th, om)
        self.u_prev = float(np.clip(u_raw + residual, -1.0, 1.0)) # total command
        return self.u_prev

    def update(self, obs, tau, th_min, th_max):
        if self.tau_noise > 0:                                     # one reading feeds both estimates
            tau = tau * (1.0 + self.tau_noise * self.tau_rng.normal())
        if abs(self.u_prev) >= U_MIN:
            k_raw = tau / self.u_prev
            self.k_hat = float(np.clip(self.k_hat + self.alpha * (k_raw - self.k_hat),
                                       0.2 * TAU_MAX, 2.0 * TAU_MAX))
        super().update(obs, tau, th_min, th_max)
        self.hist.append(self.m_hat)


def policy_obs(obs, obs_integ, sched, a_prev):
    """The 7-value observation of the residual policy."""
    return np.array([obs[0], obs[1], obs[2], obs_integ, sched.m_hat / M_NOM, sched.k_hat / TAU_MAX,
                     a_prev], dtype=np.float32)


class ResidualSched(Sched):
    """Sched plus the residual of a trained SAC model (deterministic). For rollout()."""

    def __init__(self, model, a_max=A_MAX, tau_noise=0.0):
        self.model, self.a_max = model, a_max
        super().__init__(tau_noise=tau_noise)

    def reset(self):
        super().reset()
        self.obs_integ, self.a_prev = 0.0, 0.0

    def act(self, obs):
        action = self.model.predict(policy_obs(obs, self.obs_integ, self, self.a_prev),
                                    deterministic=True)[0]
        self.a_prev = float(np.clip(action[0], -1.0, 1.0))
        return super().act(obs, self.a_max * self.a_prev)

    def update(self, obs, tau, th_min, th_max):
        super().update(obs, tau, th_min, th_max)
        self.obs_integ = float(np.clip(self.obs_integ + float(obs[2]) * self.dt, -1.0, 1.0))


class SacPolicy08b:
    """A plain SAC model of Lab 08b (deterministic) on its observation
    [theta, omega, e, int(e) dt], integral clipped to +-1 as in boom_env_robust.py."""
    payload_aware = False

    def __init__(self, model, dt=0.05):
        self.model, self.dt = model, dt
        self.reset()

    def reset(self):
        self.obs_integ, self.m_hat, self.hist = 0.0, float("nan"), []

    def act(self, obs):
        o = np.array([obs[0], obs[1], obs[2], self.obs_integ], dtype=np.float32)
        return float(np.clip(self.model.predict(o, deterministic=True)[0][0], -1.0, 1.0))

    def update(self, obs, tau, th_min, th_max):
        self.obs_integ = float(np.clip(self.obs_integ + float(obs[2]) * self.dt, -1.0, 1.0))


# --- evaluation of one table row, as run_case in 09c3 ---
def run_case(make_ctrl, cond, variant, scen=NEW):
    """Metrics of one condition and variant. make_ctrl(tau_noise) returns the controller."""
    env = make_env(variant, **cond)
    ctrl = make_ctrl(cond.get("tau_noise", 0.0))
    errs, kj, pump, tail = [], [], [], []
    for sc in scen:
        r = rollout(ctrl, env, *sc)
        errs.append(r["err_deg"]); kj.append(r["kJ"]); pump.append(e_pump_J(r) / 1000)
        tail.extend(np.array(ctrl.hist[-TAIL_STEPS:]) / env.m)
    errs = np.array(errs)
    return {"mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": np.mean(kj),
            "mean_pump_kJ": np.mean(pump), "share_over_20cm": float((cm(errs) > 20).mean()),
            "m_hat_over_m_mean": np.mean(tail) if tail else float("nan"),
            "m_hat_over_m_std": np.std(tail) if tail else float("nan")}


# --- training env ---
def sample_case(rng):
    """One training condition: a combined case (p = P_COMBINED) or a single row of 09c3."""
    cases = COMBINED if rng.random() < P_COMBINED else SINGLES
    return cases[int(rng.integers(len(cases)))][1]


class ResidualSchedEnv(BoomEnv):
    """BoomEnv driven by Sched; the action is the residual. Physics is BoomEnv.step, unchanged.

    reset()  theta0 and target uniform in [th_min + 0.05, th_max - 0.05] (as Lab 08b); a
             condition from sample_case; with p = P_PAYLOAD a payload step at a step in
             50..150 that multiplies m by U(0.9, 1.3) (as Lab 08b).
             options = {"cond": {...}, "payload": (step, factor) or None, "th0": .., "target": ..}
             fixes any of these (used by the consistency check of 09d).
    Sensor noise, torque-reading noise and the condition draw use the env's np_random.
    """

    def __init__(self, w_pump=W_PUMP, a_max=A_MAX):
        super().__init__()
        self.w_pump, self.a_max = w_pump, a_max
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.sched = Sched()
        self.sigma_th = self.sigma_om = 0.0
        self.tau_d, self.payload = False, None
        self.cond = {}

    def _obs(self):
        """Measured [theta, omega, e], one noise draw per call (order as NoisyMixin)."""
        th, om = self.th, self.om
        if self.sigma_th > 0:
            th = th + self.np_random.normal(0.0, self.sigma_th)
            om = om + self.np_random.normal(0.0, self.sigma_om)
        return np.array([th, om, self.target - th], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.sigma_th = self.sigma_om = 0.0
        super().reset(seed=seed)
        o, rng = options or {}, self.np_random
        lo, hi = self.th_min + 0.05, self.th_max - 0.05
        self.th, self.target = rng.uniform(lo, hi), rng.uniform(lo, hi)
        self.cond = sample_case(rng)
        self.payload = (int(rng.integers(50, 151)), rng.uniform(0.9, 1.3)) \
            if rng.random() < P_PAYLOAD else None
        self.th, self.target = o.get("th0", self.th), o.get("target", self.target)
        self.cond, self.payload = o.get("cond", self.cond), o.get("payload", self.payload)
        self.m, self.tau_max = M_NOM, TAU_MAX
        self.J, self.b = J_NOM * self.cond.get("j", 1.0), B_NOM * self.cond.get("b", 1.0)
        self.tau_d = bool(self.cond.get("tau_d", False))
        self.sigma_th = np.radians(self.cond.get("sigma", 0.0))
        self.sigma_om = self.sigma_th / self.dt * 0.1
        self.sched.reset()
        self.sched.tau_noise, self.sched.tau_rng = self.cond.get("tau_noise", 0.0), rng
        self.obs_integ, self.a_prev = 0.0, 0.0
        self.obs3 = self._obs()
        return policy_obs(self.obs3, self.obs_integ, self.sched, self.a_prev), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        if self.payload is not None and self.k == self.payload[0]:
            self.m *= self.payload[1]
        if self.tau_d:
            self.tau_max = 5e5 * abs(d(self.th)) / D0_M            # limit at the pre-step angle
        u = self.sched.act(self.obs3, self.a_max * a)
        th_a = self.th
        self.obs3, _, _, trunc, _ = super().step(np.array([u]))    # BoomEnv physics
        tau = u * self.tau_max
        self.sched.update(self.obs3, tau, self.th_min, self.th_max)
        self.obs_integ = float(np.clip(self.obs_integ + float(self.obs3[2]) * self.dt, -1.0, 1.0))
        # pump power of this step, same terms as rollout() in lab09_common.py
        v_cyl = d(th_a) * self.om
        area = A_CAP_M2 if v_cyl >= 0 else A_ANN_M2
        p_pump_W = (max(tau * self.om, 0.0) + HYD.DP_VALVE_PA * area * abs(v_cyl)) / HYD.ETA_PUMP
        err = self.target - self.th
        reward = -(err ** 2 + W_ABS * abs(err) + self.w_pump * p_pump_W / POWER_SCALE_W
                   + W_RATE * (a - self.a_prev) ** 2)
        self.a_prev = a
        return (policy_obs(self.obs3, self.obs_integ, self.sched, self.a_prev), reward, False,
                trunc, {"p_pump_W": p_pump_W})
