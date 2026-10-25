"""Boom cylinder geometry: rod-eye position, length, moment arm, force, torque.

All functions are pure, SI units (m, rad, Pa, N, N*m). Planar side view,
x forward, y up, theta counter-clockwise (boom up) positive.
No DX300 parameter values live here.

Pin names (same as 03-Machine/DX300 Boom Parameters):
    A  boom foot pin, origin of the frame
    B  cylinder base pin on the upper frame (fixed)
    C  cylinder rod-eye pin on the boom (moves with theta)
    D  boom-arm pin; theta is the angle of line A->D from horizontal

Constants:
    c      = |AC| [m]
    delta  = angle from line AC to line AD [rad], so angle(AC) = theta - delta
    L      = |BC|, cylinder length pin to pin [m]

Source: own derivation, see [[Boom Moment Arm Derivation]].
"""
from __future__ import annotations

import math
from typing import Tuple


def rod_eye_position(theta_rad: float, c_m: float, delta_rad: float) -> Tuple[float, float]:
    """Step 1: C(theta) = (c cos(theta - delta), c sin(theta - delta)) [m].

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    a = theta_rad - delta_rad
    return (c_m * math.cos(a), c_m * math.sin(a))


def cylinder_length(theta_rad: float, B_xy_m: Tuple[float, float],
                    c_m: float, delta_rad: float) -> float:
    """Step 2: L = |C - B| [m].

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    cx, cy = rod_eye_position(theta_rad, c_m, delta_rad)
    return math.hypot(cx - B_xy_m[0], cy - B_xy_m[1])


def moment_arm(theta_rad: float, B_xy_m: Tuple[float, float],
               c_m: float, delta_rad: float) -> float:
    """Step 6: signed d = (B_x C_y - B_y C_x) / L [m]; equals dL/dtheta.

    Positive: cylinder extension rotates the boom counter-clockwise (up).
    Zero in the singular pose where the cylinder line passes through A.
    Raises ValueError if L <= 0 (pins coincide, direction undefined).
    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    cx, cy = rod_eye_position(theta_rad, c_m, delta_rad)
    L = math.hypot(cx - B_xy_m[0], cy - B_xy_m[1])
    if L <= 0.0:
        raise ValueError(f"cylinder length L={L} m <= 0: pins B and C coincide")
    return (B_xy_m[0] * cy - B_xy_m[1] * cx) / L


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
    """Step 8: tau = n_cyl * F * d [N*m] about A, counter-clockwise positive.

    Own derivation, see [[Boom Moment Arm Derivation]].
    """
    return n_cyl * F_N * d_m
