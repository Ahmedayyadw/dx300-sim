"""Tests for boom cylinder moment-arm geometry.

Hand cases 1-5 from the own derivation, see [[Boom Moment Arm Derivation]].
"""
import math

import numpy as np
import pytest

from dx300sim.boom_geometry import (
    boom_torque,
    cylinder_force,
    cylinder_length,
    moment_arm,
    rod_eye_position,
)

R = dict(rel=1e-6)


def test_case1_length_arm_torque():
    B, c, delta, th = (0.0, -1.0), 1.0, 0.0, 0.0
    x, y = rod_eye_position(th, c, delta)
    assert (x, y) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert cylinder_length(th, B, c, delta) == pytest.approx(1.414214, **R)
    d = moment_arm(th, B, c, delta)
    assert d == pytest.approx(0.707107, **R)
    assert boom_torque(1e5, d, 2) == pytest.approx(141421.4, **R)


def test_case2():
    B, c, delta, th = (1.0, 0.0), 2.0, 0.0, math.pi / 2
    assert cylinder_length(th, B, c, delta) == pytest.approx(2.236068, **R)
    assert moment_arm(th, B, c, delta) == pytest.approx(0.894427, **R)


def test_case3_singular_pose_zero_arm():
    B, c, delta, th = (-1.0, 0.0), 1.0, 0.0, 0.0
    assert cylinder_length(th, B, c, delta) == pytest.approx(2.0, **R)
    assert moment_arm(th, B, c, delta) == pytest.approx(0.0, abs=1e-12)


def test_case4_force_from_pressures():
    F = cylinder_force(10e6, 1e6, 0.10, 0.05, 0.0)
    assert F == pytest.approx(72649.3, **R)


def test_case4_friction_reduces_force():
    assert cylinder_force(10e6, 1e6, 0.10, 0.05, 1000.0) == pytest.approx(
        72649.33 - 1000.0, rel=1e-6
    )


def _fd_dLdtheta(th, B, c, delta, h=1e-6):
    return (
        cylinder_length(th + h, B, c, delta) - cylinder_length(th - h, B, c, delta)
    ) / (2 * h)


def test_case5_dL_dtheta_equals_d():
    B, c, delta, th = (0.0, -1.0), 1.0, 0.0, 0.0
    assert _fd_dLdtheta(th, B, c, delta) == pytest.approx(0.707107, rel=1e-6)
    assert _fd_dLdtheta(th, B, c, delta) == pytest.approx(
        moment_arm(th, B, c, delta), rel=1e-6
    )


def test_virtual_work_random_poses():
    rng = np.random.default_rng(12345)
    n = 0
    while n < 20:
        B = tuple(rng.uniform(-3, 3, 2))
        c = rng.uniform(0.5, 4.0)
        delta = rng.uniform(-math.pi, math.pi)
        th = rng.uniform(-math.pi, math.pi)
        if cylinder_length(th, B, c, delta) < 0.2:
            continue  # avoid near-coincident pins
        d = moment_arm(th, B, c, delta)
        assert _fd_dLdtheta(th, B, c, delta) == pytest.approx(d, rel=1e-5, abs=1e-8)
        n += 1


def test_force_sign_extension_cap_pressure_gives_positive_force_and_boom_up():
    F = cylinder_force(10e6, 0.0, 0.10, 0.05)
    assert F > 0
    d = moment_arm(0.0, (0.0, -1.0), 1.0, 0.0)  # case 1 geometry
    assert boom_torque(F, d, 2) > 0


def test_rod_pressure_only_gives_negative_force():
    assert cylinder_force(0.0, 5e6, 0.10, 0.05) < 0


def test_cosine_rule_equals_vector_length():
    rng = np.random.default_rng(7)
    for _ in range(20):
        B = rng.uniform(-3, 3, 2)
        rB, phiB = math.hypot(*B), math.atan2(B[1], B[0])
        c = rng.uniform(0.5, 4.0)
        delta = rng.uniform(-math.pi, math.pi)
        th = rng.uniform(-math.pi, math.pi)
        gamma = th - delta - phiB
        L_cos = math.sqrt(rB**2 + c**2 - 2 * rB * c * math.cos(gamma))
        assert cylinder_length(th, tuple(B), c, delta) == pytest.approx(
            L_cos, rel=1e-9, abs=1e-9
        )
        # polar form of the arm: d = rB c sin(gamma) / L, rB = |AB|
        if L_cos > 0.2:
            assert moment_arm(th, tuple(B), c, delta) == pytest.approx(
                rB * c * math.sin(gamma) / L_cos, rel=1e-9, abs=1e-9
            )


def test_zero_length_raises():
    # B coincides with C(theta=0) = (1, 0) -> L = 0, moment arm undefined
    with pytest.raises(ValueError):
        moment_arm(0.0, (1.0, 0.0), 1.0, 0.0)
