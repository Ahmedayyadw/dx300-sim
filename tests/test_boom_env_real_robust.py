"""BoomEnvRealRobust (Lab 12 step 1, issue #7) against the Lab 11 `balanced` controller.

With residual action 0 and randomization off, the residual env must reproduce the Lab 11 rollout of
Sched11(balanced, adapt_Tf1_c1_n2) on BoomEnvReal: same final state, same E_mech, bit for bit.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))
import boom_env_real_robust as E  # noqa: E402

L10, L11 = E.L10, E.L11
SCEN = L10.SCEN["orig20"][:4]                    # tuning scenarios only, never the unseen set
VARIANTS = ("baseline", "fill", "dump")


def test_base_point_is_lab11_balanced():
    assert E.BAL_GAINS == (1.5, 0.0, 1.0)
    assert E.EST["name"] == "adapt_Tf1_c1_n2"
    assert E.EST["t_f"] == 1.0 and list(E.EST["adapt"]) == [1.0, 2]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("i", range(len(SCEN)))
def test_zero_residual_reproduces_lab11_balanced(variant, i):
    th0, target = SCEN[i]
    ref_env = L10.make_env(variant)
    ref = L11.rollout(L11.Sched11(E.BAL_GAINS, E.EST), ref_env, th0, target)

    m_tip, payload = L10.VARIANTS[variant]
    env = E.BoomEnvRealRobust(residual=True, randomize=False, m_tip=m_tip, payload=payload)
    obs, _ = env.reset(seed=0, options={"th0": th0, "target": target})
    assert obs.shape == (4,)
    trunc, n = False, 0
    while not trunc:
        obs, r, _, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        assert np.isfinite(r)
        n += 1
    assert n == env.max_steps == 200
    assert env.th == ref_env.th and env.om == ref_env.om
    assert env.energy_J / 1e3 == ref["E_mech_kJ"]
    assert abs(np.degrees(env.target - env.th)) == ref["err_deg"]
    assert env.n_flow == ref_env.n_flow and env.n_torque == ref_env.n_torque


def test_randomized_reset_ranges():
    env = E.BoomEnvRealRobust(residual=True)
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        assert env.th_min + 0.05 <= env.th <= env.th_max - 0.05
        assert E.M_EMPTY <= info["m_tip"] <= E.M_FULL
        assert 0.38 <= info["j_scale"] <= 1.5 and 0.7 <= info["tau_scale"] <= 1.0
        assert obs[3] == 0.0
