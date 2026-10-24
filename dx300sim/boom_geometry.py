"""Boom cylinder geometry: rod-eye position, length, moment arm, force, torque.

All functions are pure, SI units (m, rad, Pa, N, N*m). Planar side view,
x forward, y up, origin O at the boom foot pin, theta counter-clockwise
(boom up) positive. Source: own derivation, see [[Boom Moment Arm Derivation]].
No DX300 parameter values live here.
"""
from __future__ import annotations

import math
from typing import Tuple


def rod_eye_position(theta_rad: float, r_B_m: float, beta_rad: float) -> Tuple[float, float]:
    """Step 1: B(theta) = (rB cos(theta+beta), rB sin(theta+beta)) [m].

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    a = theta_rad + beta_rad
    return (r_B_m * math.cos(a), r_B_m * math.sin(a))


def cylinder_length(theta_rad: float, A_xy_m: Tuple[float, float],
                    r_B_m: float, beta_rad: float) -> float:
    """Step 2: L = |B - A| [m].

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    bx, by = rod_eye_position(theta_rad, r_B_m, beta_rad)
    return math.hypot(bx - A_xy_m[0], by - A_xy_m[1])


def moment_arm(theta_rad: float, A_xy_m: Tuple[float, float],
               r_B_m: float, beta_rad: float) -> float:
    """Step 6: signed d = (A_x B_y - A_y B_x) / L [m]; equals dL/dtheta.

    Positive: cylinder extension rotates the boom counter-clockwise (up).
    Zero in the singular pose where the cylinder line passes through O.
    Raises ValueError if L <= 0 (pins coincide, direction undefined).
    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    bx, by = rod_eye_position(theta_rad, r_B_m, beta_rad)
    L = math.hypot(bx - A_xy_m[0], by - A_xy_m[1])
    if L <= 0.0:
        raise ValueError(f"cylinder length L={L} m <= 0: pins A and B coincide")
    return (A_xy_m[0] * by - A_xy_m[1] * bx) / L


def cylinder_force(p_cap_Pa: float, p_rod_Pa: float, bore_m: float,
                   rod_m: float, f_fric_N: float = 0.0) -> float:
    """Step 9: F = p_cap*A_cap - p_rod*A_ann - F_fric [N], extension positive.

    A_cap = pi D^2/4, A_ann = pi (D^2 - d_rod^2)/4.
    Own derivation, see [[Boom Moment Arm Derivation]] (pressure terms:
    Mitrev 2017 Eq. 22; friction: Casoli 2015).
    """
    A_cap = math.pi * bore_m**2 / 4.0
    A_ann = math.pi * (bore_m**2 - rod_m**2) / 4.0
    return p_cap_Pa * A_cap - p_rod_Pa * A_ann - f_fric_N


def boom_torque(F_N: float, d_m: float, n_cyl: int) -> float:
    """Step 8: tau = n_cyl * F * d [N*m] about O, counter-clockwise positive.

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    return n_cyl * F_N * d_m
