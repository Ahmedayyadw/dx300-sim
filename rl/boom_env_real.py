"""Lab 10a: 1-DOF DX300 boom plant with the Lab 08a Simscape parameters. No training here.

Same observation and action interface as rl/boom_env.py (BoomEnv). Design, equations and the
source of every value: docs/10_real_plant.md. Status tags: OFFICIAL (Doosan/Develon document),
ESTIMATED (drawing or literature, Lab 08a), ASSUMED (model choice of this lab).

Plant, per substep of h = dt / SUBSTEPS (semi-implicit Euler, command held):
  J(m_tip) om' = tau_cyl - tau_g(th, m_tip) - tau_f(th, om)
  tau_cmd = u tau_up(th) for u >= 0, u tau_down(th) for u < 0      330 bar at |u| = 1
  tau_g   = g [m_boom Lc cos th + sum_i m_i (x_i cos th - y_i sin th)]   boom + 3 point masses
  tau_f   = r(th) [F_c sign(v) + c_v v],  v = r(th) om              cylinder friction
  flow limit: -Q / A_ann <= v <= Q / A_cap. On the limit om = v_lim / r and
              tau_cyl = J (om_new - om_old) / h + tau_g + tau_f, else tau_cyl = tau_cmd.
  end stops: th clipped, om = 0 (as BoomEnv).
Energy uses the delivered torque tau_cyl: energy_J = sum |tau_cyl om| h.

Everything is read from matlab/lab08a/export/boom_params_lab08a.json except the cylinder bore,
rod and count (dx300sim/dx300_params.py, OFFICIAL; not in the JSON).
"""
import bisect
import json
import math
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # repo root, for dx300sim
from dx300sim import dx300_params as P

JSON_PATH = os.path.join(ROOT, "matlab", "lab08a", "export", "boom_params_lab08a.json")
with open(JSON_PATH) as _f:
    _D = json.load(_f)


def _v(key):
    return _D[key]["value"]


G = 9.81                                           # m/s^2, same value as boom_env.py
M_BOOM_KG = float(_v("m_boom"))                    # OFFICIAL
L_BOOM_M = float(_v("L_boom"))                     # OFFICIAL
assert L_BOOM_M == P.BOOM_LENGTH_OFFICIAL_M, "JSON boom length differs from dx300_params.py"
LC_M = float(_v("Lc"))                             # ESTIMATED (CoM at mid-boom)
J_BOOM = float(_v("J_boom_only"))                  # ESTIMATED (brick about A)
_FM = _v("front_masses")
M_D_KG = float(_FM["m_D_point_kg"])                # mass OFFICIAL, lumping ESTIMATED
M_ARM_KG = float(_FM["m_arm_point_kg"])            # mass OFFICIAL, lumping ESTIMATED
M_TIP_NOM_KG = float(_FM["m_tip_point_kg"])        # ESTIMATED: bucket 1120 kg + 1.27 m^3 x 1800 kg/m^3
LA_M = float(_FM["arm_length_m"])                  # OFFICIAL
PHI_REL_RAD = float(_FM["arm_angle_rel_boom_rad"])  # ESTIMATED pose (arm perpendicular to boom)
TH_MIN, TH_MAX = float(_v("th_min")), float(_v("th_max"))   # ESTIMATED
Q_PUMP_M3S = float(_v("Q_pump"))                   # OFFICIAL, 494 L/min
P_RELIEF_PA = float(_v("p_relief"))                # OFFICIAL, 330 bar
F_COULOMB_N = float(_v("friction_cylinder_eq")["coulomb_N"])        # ESTIMATED (Li 2020, scaled)
C_VISC_NSM = float(_v("friction_cylinder_eq")["viscous_Ns_per_m"])  # ESTIMATED (Li 2020, scaled)

A_CAP_M2 = P.N_CYL * math.pi * P.BORE_M ** 2 / 4.0                    # both cylinders, OFFICIAL-derived
A_ANN_M2 = P.N_CYL * math.pi * (P.BORE_M ** 2 - P.ROD_M ** 2) / 4.0   # both cylinders, OFFICIAL-derived
V_EXT_MAX = Q_PUMP_M3S / A_CAP_M2                  # m/s, rod speed limit extending (about 0.267)
V_RET_MAX = Q_PUMP_M3S / A_ANN_M2                  # m/s, rod speed limit retracting (about 0.496)
                                                   # hard retraction limit is ASSUMED (no cavitation)

# front point masses in the boom frame (x along A->D, y up), as stage3_validate.m
P_D = (L_BOOM_M, 0.0)
P_ARM = (L_BOOM_M + 0.5 * LA_M * math.cos(PHI_REL_RAD), 0.5 * LA_M * math.sin(PHI_REL_RAD))
P_TIP = (L_BOOM_M + LA_M * math.cos(PHI_REL_RAD), LA_M * math.sin(PHI_REL_RAD))
RHO2_TIP_M2 = P_TIP[0] ** 2 + P_TIP[1] ** 2        # |p_tip|^2

# torque-limit tables (ESTIMATED: 330 bar x area x moment arm, no losses)
_TH = [float(x) for x in _v("tau_up")["theta_rad"]]
_TAU_UP = [float(x) for x in _v("tau_up")["tau_Nm"]]
_TAU_DOWN = [float(x) for x in _v("tau_down")["tau_Nm"]]
_R = [t / (P_RELIEF_PA * A_CAP_M2) for t in _TAU_UP]   # moment arm r(theta) [m]
_N = len(_TH)                                      # 51 points, spacing not uniform
assert _v("tau_down")["theta_rad"] == _v("tau_up")["theta_rad"]
assert _v("b_friction_rot_table")["theta_rad"] == _v("tau_up")["theta_rad"]
assert all(_TH[i] < _TH[i + 1] for i in range(_N - 1)), "theta grid not increasing"
assert abs(_TH[0] - TH_MIN) < 1e-12 and abs(_TH[-1] - TH_MAX) < 1e-12
assert max(abs(d / u - A_ANN_M2 / A_CAP_M2) for d, u in zip(_TAU_DOWN, _TAU_UP)) < 1e-6
_B_TAB = _v("b_friction_rot_table")["b_Nms_per_rad"]   # must equal c_v r^2
assert max(abs(C_VISC_NSM * r * r / b - 1.0) for r, b in zip(_R, _B_TAB)) < 1e-9


def _interp(tab, th):
    """Linear interpolation in a JSON table over theta, clipped at the ends."""
    if th <= _TH[0]:
        return tab[0]
    if th >= _TH[-1]:
        return tab[-1]
    i = bisect.bisect_right(_TH, th) - 1
    return tab[i] + (th - _TH[i]) / (_TH[i + 1] - _TH[i]) * (tab[i + 1] - tab[i])


def moment_arm(th):
    """Cylinder moment arm r(theta) [m], from the JSON tau_up table."""
    return _interp(_R, th)


def tau_up(th):
    """Boom-up torque limit [N m] (cap side at 330 bar)."""
    return _interp(_TAU_UP, th)


def tau_down(th):
    """Boom-down torque limit [N m], positive number (rod side at 330 bar)."""
    return _interp(_TAU_DOWN, th)


def inertia(m_tip=M_TIP_NOM_KG):
    """J about A [kg m^2]: boom brick plus the three front point masses."""
    return (J_BOOM + M_D_KG * (P_D[0] ** 2 + P_D[1] ** 2)
            + M_ARM_KG * (P_ARM[0] ** 2 + P_ARM[1] ** 2) + m_tip * RHO2_TIP_M2)


def _moments(m_tip):
    """(Cc, Cs) [kg m] with tau_g = g (Cc cos th - Cs sin th)."""
    cc = M_BOOM_KG * LC_M + M_D_KG * P_D[0] + M_ARM_KG * P_ARM[0] + m_tip * P_TIP[0]
    cs = M_D_KG * P_D[1] + M_ARM_KG * P_ARM[1] + m_tip * P_TIP[1]
    return cc, cs


def gravity_torque(th, m_tip=M_TIP_NOM_KG):
    """Gravity torque about A [N m], positive = pulls the boom down."""
    cc, cs = _moments(m_tip)
    return G * (cc * math.cos(th) - cs * math.sin(th))


def potential_energy(th, m_tip=M_TIP_NOM_KG):
    """Potential energy [J], zero at theta = 0 with the tip mass at its y = P_TIP[1]."""
    cc, cs = _moments(m_tip)
    return G * (cc * math.sin(th) + cs * math.cos(th))


def x_tip(th):
    """Horizontal distance of the bucket tip mass from A [m]."""
    return P_TIP[0] * math.cos(th) - P_TIP[1] * math.sin(th)


J_NOM = inertia()
assert abs(J_NOM - float(_v("J_with_front"))) < 1e-6 * J_NOM, "J differs from JSON J_with_front"


class BoomEnvReal(gym.Env):
    """BoomEnv interface on the Lab 08a plant.

    m_tip       bucket tip mass at the start of every episode [kg]
    payload     None or (step, m_tip_after): the tip mass changes at that control step
    j_scale     factor on J (stress test; the plant's J, not the controller's)
    fric_scale  factor on the cylinder friction F_c and c_v (stress test)
    info of step(): tau_cmd, tau_cyl (delivered torque, mean over the step) [N m], flow_lim and
    torque_lim (substeps of this step on each limit), flow_violation (substeps where 330 bar
    could not hold the speed limit).
    """
    SUBSTEPS = 10

    def __init__(self, w_energy=0.05, m_tip=M_TIP_NOM_KG, payload=None, j_scale=1.0, fric_scale=1.0):
        self.dt, self.max_steps = 0.05, 200                       # s, steps (10 s), as BoomEnv
        self.th_min, self.th_max = TH_MIN, TH_MAX
        self.m_tip0, self.m_tip, self.payload = float(m_tip), float(m_tip), payload
        self.j_scale, self.fric_scale = j_scale, fric_scale
        self.w_energy = w_energy
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)

    @property
    def J(self):
        return self.j_scale * inertia(self.m_tip)

    def _obs(self):
        return np.array([self.th, self.om, self.target - self.th], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = self.np_random.uniform(-0.4, 0.9)                # ranges as BoomEnv
        self.target = self.np_random.uniform(-0.3, 0.9)
        self.om, self.k, self.energy_J = 0.0, 0, 0.0
        self.m_tip = self.m_tip0
        self.W_pos_J = self.W_neg_J = self.vol_m3 = self.fric_J = self.stop_J = 0.0
        self.n_sub = self.n_flow = self.n_torque = self.n_violation = 0
        return self._obs(), {}

    def step(self, action):
        u = float(np.clip(action[0], -1, 1))
        if self.payload is not None and self.k == self.payload[0]:
            self.m_tip = float(self.payload[1])
        h = self.dt / self.SUBSTEPS
        J = self.J
        cc, cs = _moments(self.m_tip)
        th, om = float(self.th), float(self.om)
        tau_sum, n_flow, n_torque, n_viol = 0.0, 0, 0, 0
        tau_cmd = 0.0
        for _ in range(self.SUBSTEPS):
            r = _interp(_R, th)
            t_up, t_dn = _interp(_TAU_UP, th), _interp(_TAU_DOWN, th)
            tau_cmd = u * t_up if u >= 0.0 else u * t_dn
            tau_g = G * (cc * math.cos(th) - cs * math.sin(th))
            f_c = self.fric_scale * F_COULOMB_N * r               # Coulomb torque
            b = self.fric_scale * C_VISC_NSM * r * r              # viscous coefficient
            drive = tau_cmd - tau_g - b * om
            if om == 0.0:
                om_new = 0.0 if abs(drive) <= f_c else (drive - math.copysign(f_c, drive)) / J * h
            else:
                om_new = om + (drive - math.copysign(f_c, om)) / J * h
                if om_new * om < 0.0:                             # would cross zero: stop here
                    om_new = 0.0
            tau_cyl, on_flow = tau_cmd, False
            om_hi, om_lo = V_EXT_MAX / r, -V_RET_MAX / r
            if om_new > om_hi or om_new < om_lo:                  # pump flow limit
                om_lim = om_hi if om_new > om_hi else om_lo
                fric = b * om + math.copysign(f_c, om_lim)
                need = J * (om_lim - om) / h + tau_g + fric       # torque that gives om_lim
                tau_cyl = min(max(need, -t_dn), t_up)             # 330 bar
                if tau_cyl != need:
                    om_lim = om + (tau_cyl - tau_g - fric) / J * h
                    n_viol += 1
                om_new, on_flow = om_lim, True
            th_new = th + om_new * h
            if th_new <= self.th_min or th_new >= self.th_max:    # hit end stop
                th_new = min(max(th_new, self.th_min), self.th_max)
                self.stop_J += 0.5 * J * om_new ** 2              # kinetic energy taken by the stop
                om_new = 0.0
            power = tau_cyl * om_new                              # delivered mechanical power
            self.energy_J += abs(power) * h
            if power >= 0.0:
                self.W_pos_J += power * h
            else:
                self.W_neg_J += -power * h
            v = r * om_new
            self.vol_m3 += (A_CAP_M2 if v >= 0.0 else A_ANN_M2) * abs(v) * h
            self.fric_J += (b * om_new + (f_c if om_new != 0.0 else 0.0) * math.copysign(1.0, om_new)) * om_new * h
            tau_sum += tau_cyl
            n_flow += on_flow
            n_torque += (abs(u) >= 1.0 and not on_flow)
            th, om = th_new, om_new
        self.th, self.om = th, om
        self.n_sub += self.SUBSTEPS
        self.n_flow += n_flow
        self.n_torque += n_torque
        self.n_violation += n_viol
        err = self.target - self.th
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * u ** 2)
        self.k += 1
        info = {"tau_cmd": tau_cmd, "tau_cyl": tau_sum / self.SUBSTEPS, "flow_lim": n_flow,
                "torque_lim": n_torque, "flow_violation": n_viol}
        return self._obs(), reward, False, self.k >= self.max_steps, info
