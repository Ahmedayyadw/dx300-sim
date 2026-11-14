"""Lab 12g plant knobs of BoomEnvReal: moment-arm factor r_scale and gravity factor g_scale."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))
import boom_env_real as R  # noqa: E402


def _env(**kw):
    env = R.BoomEnvReal(**kw)
    env.reset(seed=0)
    env.th, env.om = 0.2, 0.0
    return env


def test_defaults_are_one():
    env = R.BoomEnvReal()
    assert env.r_scale == 1.0 and env.g_scale == 1.0


@pytest.mark.parametrize("k", (0.8, 1.2))
def test_r_scale_scales_torque_limit(k):
    # huge J: theta barely moves in one step, so the full-command torque is k x nominal
    nom, sc = _env(j_scale=1e9), _env(j_scale=1e9, r_scale=k)
    _, _, _, _, i_nom = nom.step(np.array([1.0]))
    _, _, _, _, i_sc = sc.step(np.array([1.0]))
    assert i_sc["tau_cmd"] == pytest.approx(k * i_nom["tau_cmd"], rel=1e-9)


@pytest.mark.parametrize("k", (0.8, 1.2))
def test_g_scale_scales_gravity(k):
    # no friction, no command: the boom falls with acceleration -k tau_g / J
    nom, sc = _env(fric_scale=0.0), _env(fric_scale=0.0, g_scale=k)
    nom.SUBSTEPS = sc.SUBSTEPS = 1
    nom.step(np.array([0.0]))
    sc.step(np.array([0.0]))
    assert nom.om < 0.0
    assert sc.om == pytest.approx(k * nom.om, rel=1e-12)
