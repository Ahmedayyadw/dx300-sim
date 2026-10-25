"""Checks that the DX300 parameters and the geometry functions agree.

Reference values: hand calculation of 2026-10-25 (drawing at 4.4 mm/px).
"""
import math

import pytest

from dx300sim import dx300_params as P
from dx300sim.boom_geometry import cylinder_length, rod_eye_position


def test_boom_length_matches_official():
    assert P.BOOM_LENGTH_M == pytest.approx(P.BOOM_LENGTH_OFFICIAL_M, abs=0.005)


def test_derived_constants():
    assert P.C_M == pytest.approx(2.763, abs=0.001)
    assert math.degrees(P.DELTA_RAD) == pytest.approx(-24.2, abs=0.1)
    assert math.degrees(P.THETA_REF_RAD) == pytest.approx(-6.3, abs=0.1)


def test_rod_eye_reproduces_drawing():
    cx, cy = rod_eye_position(P.THETA_REF_RAD, P.C_M, P.DELTA_RAD)
    assert (cx, cy) == pytest.approx(P.C_REF_XY_M, abs=1e-9)


def test_cylinder_length_at_drawing_pose():
    L = cylinder_length(P.THETA_REF_RAD, P.B_XY_M, P.C_M, P.DELTA_RAD)
    assert L == pytest.approx(2.573, abs=0.001)
    assert P.CLOSED_LENGTH_M < L < P.EXTENDED_LENGTH_M


@pytest.mark.parametrize("theta_deg, L_expected", [
    (-47.2, P.CLOSED_LENGTH_M),     # boom fully down
    (60.4, P.EXTENDED_LENGTH_M),    # boom fully up
])
def test_end_stops_match_cylinder_stroke(theta_deg, L_expected):
    L = cylinder_length(math.radians(theta_deg), P.B_XY_M, P.C_M, P.DELTA_RAD)
    assert L == pytest.approx(L_expected, abs=0.005)
