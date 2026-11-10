"""Shared code for Labs 09b and 09c: scenarios, env variants, Lab 07 PID, energy logging.

Nothing here trains or writes files. Existing labs are not imported (importing
07_pid_baseline.py would rerun it and rewrite its CSV); the scenario generators, the
PayloadBoomEnv class, the gain grid and the tuning criterion are copied from it unchanged.

Cylinder model (two-cylinder total, see docs/09c_payload_estimator.md):
  F_cyl = tau / r(theta)        r = moment_arm from dx300sim/boom_geometry.py
  v_cyl = r(theta) * omega
  extending  (v > 0): p_load =  F_cyl / A_cap, Q = A_cap * v
  retracting (v < 0): p_load = -F_cyl / A_ann, Q = A_ann * |v|
  p_pump = max(p_load, 0) + dp_valve
  E_pump = sum(p_pump * Q) * dt / eta_pump
A_cap = N_CYL pi BORE^2 / 4 and A_ann = N_CYL pi (BORE^2 - ROD^2) / 4 from dx300_params.py.
dp_valve and eta_pump from rl/09b_hyd_params.py (ASSUMED - not from manual).

Since p_pump * Q = max(tau * omega, 0) + dp_valve * Q, one rollout stores the positive work
W_pos and the swept volume, and E_pump follows for any (dp_valve, eta_pump) without a rerun.
"""
import importlib.util
import itertools
import math
import os
import sys
import time
import numpy as np
from boom_env import BoomEnv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for dx300sim
from dx300sim import dx300_params as P
from dx300sim.boom_geometry import moment_arm

_spec = importlib.util.spec_from_file_location("hyd_params_09b", os.path.join(HERE, "09b_hyd_params.py"))
HYD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HYD)

BOOM_LENGTH_M = P.BOOM_LENGTH_OFFICIAL_M
A_CAP_M2 = P.N_CYL * math.pi * P.BORE_M ** 2 / 4.0                     # both cylinders, cap side
A_ANN_M2 = P.N_CYL * math.pi * (P.BORE_M ** 2 - P.ROD_M ** 2) / 4.0    # both cylinders, rod side
G = 9.81                                                              # same value as boom_env.py

_nominal = BoomEnv()
M_NOM, J_NOM, B_NOM, LC = _nominal.m, _nominal.J, _nominal.b, _nominal.Lc
TAU_MAX, DT = _nominal.tau_max, _nominal.dt
U_FF_SCALE = _nominal.m * 9.81 * _nominal.Lc / _nominal.tau_max   # nominal feed-forward, as in 07

# --- scenarios and gain grid, copied from 07_pid_baseline.py ---
KP = [0.25, 0.5, 1, 2, 5, 10, 20, 30, 50]
KI = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10]
KD = [0, 0.25, 0.5, 1, 2]

rng = np.random.default_rng(7)      # the original 20 (same as 03/04/05/07)
ORIG = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(20)]
rng = np.random.default_rng(123)    # 100 unseen scenarios, same as 05/06/06b/07
NEW = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(100)]


class PayloadBoomEnv(BoomEnv):
    """BoomEnv whose mass steps to 1.2 x m at step 100 (bucket filling). Copied from 07."""

    def reset(self, seed=None, options=None):
        if hasattr(self, "m0"):
            self.m = self.m0                       # undo the previous episode's step
        self.m0 = self.m
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.k == 100:
            self.m *= 1.2
        return super().step(action)


def make_m():
    env = BoomEnv(); env.m *= 0.42; return env


VARIANTS = {"baseline": BoomEnv, "payload_x1.2": PayloadBoomEnv, "m_x0.42": make_m}


def d(theta):
    return moment_arm(theta, P.B_XY_M, P.C_M, P.DELTA_RAD)


def cm(deg):
    return BOOM_LENGTH_M * np.radians(deg) * 100


# --- controller ---
class PID:
    """Lab 07 PID. payload_aware=True swaps the nominal mass in the feed-forward for m_hat.

    Estimator (docs/09c_payload_estimator.md), one control step late:
      m_raw = (tau[k] - J_nom (om[k+1] - om[k]) / dt - b_nom om[k]) / (g Lc cos th[k])
      m_hat += alpha (m_raw - m_hat), alpha = dt / (T_f + dt), m_hat clipped to [0, 2 m_nom]
      no update on a step that ends on an end stop
    """

    def __init__(self, gains, payload_aware=False, t_f=0.2, dt=DT):
        self.kp, self.ki, self.kd = gains
        self.payload_aware, self.dt = payload_aware, dt
        self.alpha = dt / (t_f + dt)
        self.reset()

    def reset(self):
        self.integ, self.m_hat, self.prev = 0.0, M_NOM, None

    def act(self, obs):
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])   # what the PID observes
        scale = self.m_hat * G * LC / TAU_MAX if self.payload_aware else U_FF_SCALE
        u_raw = self.kp * e + self.ki * self.integ + self.kd * (-om) + scale * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup
            self.integ += e * self.dt
        self.prev = (th, om)
        return float(np.clip(u_raw, -1.0, 1.0))

    def update(self, obs, tau, th_min, th_max):
        """Estimator step. obs is the observation after the step that applied torque tau."""
        if not self.payload_aware:
            return
        th1, om1 = float(obs[0]), float(obs[1])
        th0, om0 = self.prev
        if th1 <= th_min + 1e-6 or th1 >= th_max - 1e-6:           # end-stop gate
            return
        m_raw = (tau - J_NOM * (om1 - om0) / self.dt - B_NOM * om0) / (G * LC * np.cos(th0))
        self.m_hat = float(np.clip(self.m_hat + self.alpha * (m_raw - self.m_hat), 0.0, 2.0 * M_NOM))


# --- rollout with energy logging ---
def rollout(ctrl, env, th0, target):
    """One episode, set up as in 06/07. Returns the 06/07 metrics plus the energy terms [J].

    The logging reads the env state before and after env.step; the env itself is unchanged.
    """
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs = env._obs()
    ctrl.reset()
    dt = env.dt
    a = {k: 0.0 for k in ("E_mech_J", "W_pos_J", "W_neg_J", "vol_m3", "dEp_J", "dEk_J",
                           "damp_J", "stop_J", "stop_steps", "p_load_max_Pa")}
    for _ in range(env.max_steps):
        u = ctrl.act(obs)
        th_a, om_a = env.th, env.om
        obs, _, _, trunc, _ = env.step(np.array([u]))
        tau = u * env.tau_max
        ctrl.update(obs, tau, env.th_min, env.th_max)
        th_b, om_b = env.th, env.om
        power_W = tau * om_b                                       # same product as env.energy_J
        a["E_mech_J"] += abs(power_W) * dt
        a["W_pos_J"] += max(power_W, 0.0) * dt                     # cylinder drives the boom
        a["W_neg_J"] += max(-power_W, 0.0) * dt                    # boom drives the cylinder
        r = d(th_a)
        f_cyl, v_cyl = tau / r, r * om_b
        if v_cyl >= 0:
            p_load, area = f_cyl / A_CAP_M2, A_CAP_M2
        else:
            p_load, area = -f_cyl / A_ANN_M2, A_ANN_M2
        a["vol_m3"] += area * abs(v_cyl) * dt                      # oil through the valve
        if v_cyl != 0:
            a["p_load_max_Pa"] = max(a["p_load_max_Pa"], p_load)
        d_ep = env.m * G * env.Lc * (np.sin(th_b) - np.sin(th_a))  # work against gravity
        d_ek = 0.5 * env.J * (om_b ** 2 - om_a ** 2)
        a["dEp_J"] += d_ep
        a["dEk_J"] += d_ek
        if om_b == 0.0 and (th_b == env.th_min or th_b == env.th_max):
            a["stop_J"] += -(d_ep + d_ek)                          # energy taken by the end stop
            a["stop_steps"] += 1
        else:
            a["damp_J"] += env.b * om_b ** 2 * dt
        if trunc:
            break
    assert abs(a["E_mech_J"] - env.energy_J) <= 1e-9 * max(1.0, env.energy_J), "E_mech != env.energy_J"
    a["err_deg"] = abs(np.degrees(env.target - env.th))
    a["kJ"] = env.energy_J / 1000
    a["m_hat_ratio"] = ctrl.m_hat / env.m
    return a


def e_pump_J(a, dp_valve_Pa=HYD.DP_VALVE_PA, eta_pump=HYD.ETA_PUMP):
    """Pump input energy of one rollout: (W_pos + dp_valve * swept volume) / eta_pump."""
    return (a["W_pos_J"] + dp_valve_Pa * a["vol_m3"]) / eta_pump


def evaluate(ctrl, env, scen):
    return [rollout(ctrl, env, *sc) for sc in scen]


def col(rs, key):
    return np.array([r[key] for r in rs])


def tune_pid():
    """Lab 07 "pid" gains: grid search on the 20 original scenarios, lowest mean energy among
    gain sets with worst-case error < 20 cm. Same grid and criterion as 07_pid_baseline.py."""
    t0 = time.time()
    env, grid = BoomEnv(), []
    for gains in itertools.product(KP, KI, KD):
        rs = evaluate(PID(gains), env, ORIG)
        grid.append((gains, col(rs, "err_deg").max(), col(rs, "kJ").mean()))
    ok = [g for g in grid if cm(g[1]) < 20]
    best, best_max, best_kj = min(ok, key=lambda g: g[2])
    print(f"grid: {len(grid)} gain sets on {len(ORIG)} scenarios in {time.time() - t0:.1f} s")
    print(f"pid: {len(ok)} gain sets with worst case < 20 cm | tuned Kp={best[0]} Ki={best[1]}"
          f" Kd={best[2]} | orig: max {best_max:.2f} deg ({cm(best_max):.1f} cm), {best_kj:.1f} kJ")
    return best
