"""Lab 12 step 1 (issue #7): robust SAC env on the real plant. No training here.

Plant: BoomEnvReal of rl/boom_env_real.py (Lab 10a), unchanged and not modified; this class only
sets its inputs (tip mass, payload step, J factor, command) and reads its state. The Lab 08b
features of rl/boom_env_robust.py are added on top:

  reset      theta0 and target uniform in [th_min + 0.05, th_max - 0.05] (full end-stop range).
             options={"th0": .., "target": ..} sets a fixed scenario instead (evaluation, tests).
  randomize  each reset (all ranges ASSUMED, class attributes):
               m_tip    U(1120, 3406) kg, empty bucket (OFFICIAL) to full bucket (ESTIMATED)
               payload  p = 0.3: at a step U{50..150} the tip mass jumps to U(1120, 3406) kg
               torque   p = 0.5: torque limit x U(0.7, 1.0), applied as a gain on the command
                        (tau_cmd = s u tau_up(theta)). The 330 bar clamp of the plant's flow-limit
                        branch is the relief valve and is not scaled.
               inertia  J factor U(0.38, 1.5) on the plant's J (covers the Lab 10/11 rows J_x0.38
                        and J_x1.5)
  obs        [theta, omega, e, int(e) dt], integral clipped to +-1, reset each episode
  reward     -(e^2 + 0.01 omega^2 + w P / P_REF)
               P     = mean delivered mechanical power |tau_cyl omega| over the 10 substeps of the
                       step = (energy_J increment) / dt
               P_REF = Q_pump x p_relief = 271.7 kW (corner power of the pump, OFFICIAL values)
  residual   u = clip(u_base + a, -1, 1). u_base = Sched11 of rl/11_real_plant_tuning.py with the
             Lab 11 `balanced` gains (rl/11_points.json: Kp 1.5, Ki 0, Kd 1) and the adaptive mass
             filter adapt_Tf1_c1_n2 (rl/11_dump_fix_choice.json). The base controller sees the
             plant's [theta, omega, e] and the delivered torque, in the call order of the Lab 11
             rollout; its gain estimator is told the applied command u (= u_base when a = 0).
             residual=False: u = a.
  substeps   10, BoomEnvReal.SUBSTEPS (semi-implicit Euler, command held over the control step).

With a = 0, randomize=False and the m_tip / payload of a Lab 10 variant, the residual env gives the
Lab 11 `balanced` rollout bit for bit (tests/test_boom_env_real_robust.py).
info of step(): the plant's info plus u_base, u, P_W.
"""
import importlib.util
import json
import os
import sys
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                     # rl/, for boom_env_real and lab09_common
import boom_env_real as R  # noqa: E402


def _load(name, fname):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L11 = _load("lab11_real_plant_tuning", "11_real_plant_tuning.py")   # imports only; main guarded
L10 = L11.L10

with open(os.path.join(HERE, "11_points.json")) as _f:
    _pts = json.load(_f)
assert _pts["approved"], "rl/11_points.json is not approved"
BAL_GAINS = tuple(float(_pts["points"]["balanced"][k]) for k in ("kp", "ki", "kd"))
EST_NAME = "adapt_Tf1_c1_n2"
with open(os.path.join(HERE, "11_dump_fix_choice.json")) as _f:
    EST = next(e for e in json.load(_f)["estimators"] if e["name"] == EST_NAME)

M_EMPTY, M_FULL = L10.M_EMPTY, L10.M_FULL      # 1120 kg (OFFICIAL), 3406 kg (ESTIMATED)
P_REF_W = R.Q_PUMP_M3S * R.P_RELIEF_PA          # 271.7 kW
INTEG_CLIP = 1.0
EDGE_RAD = 0.05                                 # scenario margin from the end stops, as 08b


class BoomEnvRealRobust(R.BoomEnvReal):
    """Lab 08b features on the Lab 10a plant. See the module docstring."""

    M_TIP_RANGE = (M_EMPTY, M_FULL)
    P_PAYLOAD, PAYLOAD_STEPS = 0.3, (50, 150)
    P_TAU, TAU_RANGE = 0.5, (0.7, 1.0)
    J_RANGE = (0.38, 1.5)

    def __init__(self, w_energy=0.1, residual=False, randomize=True, m_tip=M_FULL, payload=None):
        super().__init__(w_energy=w_energy, m_tip=m_tip, payload=payload)
        self.residual, self.randomize = residual, randomize
        self.m_tip_fixed, self.payload_fixed = float(m_tip), payload
        self.tau_scale = 1.0
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.ctrl = L11.Sched11(BAL_GAINS, EST) if residual else None

    def _plant_obs(self):
        return R.BoomEnvReal._obs(self)          # [theta, omega, e], what the base controller sees

    def _obs(self):
        return np.array([self.th, self.om, self.target - self.th, self.integ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.integ = 0.0
        super().reset(seed=seed, options=options)
        rng = self.np_random
        lo, hi = self.th_min + EDGE_RAD, self.th_max - EDGE_RAD
        self.th, self.target = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if options and "th0" in options:
            self.th, self.target = float(options["th0"]), float(options["target"])
        m_tip, payload, j, s = self.m_tip_fixed, self.payload_fixed, 1.0, 1.0
        if self.randomize:
            m_tip = rng.uniform(*self.M_TIP_RANGE)
            payload = None
            if rng.random() < self.P_PAYLOAD:
                payload = (int(rng.integers(self.PAYLOAD_STEPS[0], self.PAYLOAD_STEPS[1] + 1)),
                           float(rng.uniform(*self.M_TIP_RANGE)))
            if rng.random() < self.P_TAU:
                s = rng.uniform(*self.TAU_RANGE)
            j = rng.uniform(*self.J_RANGE)
        self.m_tip0 = self.m_tip = float(m_tip)
        self.payload, self.j_scale, self.tau_scale = payload, j, s
        if self.ctrl is not None:
            self.ctrl.reset()
        return self._obs(), {"m_tip": self.m_tip0, "payload": payload, "j_scale": j, "tau_scale": s}

    def step(self, action):
        a = float(np.clip(action[0], -1, 1))
        u_base = 0.0
        if self.ctrl is not None:
            u_base = self.ctrl.act(self._plant_obs())
            u = float(np.clip(u_base + a, -1.0, 1.0))
            self.ctrl.u_prev = u                  # gain estimator: the command really applied
        else:
            u = a
        e0 = self.energy_J
        _, _, _, trunc, info = super().step(np.array([u * self.tau_scale]))
        if self.ctrl is not None:
            self.ctrl.update(self._plant_obs(), info["tau_cyl"], self.th_min, self.th_max)
        err = self.target - self.th
        self.integ = float(np.clip(self.integ + err * self.dt, -INTEG_CLIP, INTEG_CLIP))
        p_w = (self.energy_J - e0) / self.dt
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * p_w / P_REF_W)
        info.update(u_base=u_base, u=u, P_W=p_w)
        return self._obs(), reward, False, trunc, info
