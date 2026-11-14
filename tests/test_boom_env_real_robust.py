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
SCEN = L10.SCEN["orig20"][:4]  # tuning scenarios only, never the unseen set
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


# ----------------------------------------------------------------------------- train_noise (step 3)
REF_COMMIT = "07cd264"  # the env the 12_* models were trained on


def _env_at_commit():
    """BoomEnvRealRobust as committed in REF_COMMIT, loaded from git under its own module name."""
    import importlib.util
    import subprocess
    import tempfile

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        src = subprocess.run(
            ["git", "show", f"{REF_COMMIT}:rl/boom_env_real_robust.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {REF_COMMIT} not available")
    rl_dir = os.path.join(root, "rl")
    head = "HERE = os.path.dirname(os.path.abspath(__file__))"
    assert src.count(head) == 1
    src = src.replace(head, f"HERE = {rl_dir!r}")  # its JSON files and imports: rl/
    path = os.path.join(tempfile.mkdtemp(), "boom_env_real_robust_ref.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("boom_env_real_robust_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("residual", (True, False))
def test_train_noise_off_is_unchanged(residual):
    """train_noise=False (default) equals the committed env bit for bit: obs, reward, info."""
    Ref = _env_at_commit().BoomEnvRealRobust
    for seed in range(6):
        new, ref = E.BoomEnvRealRobust(residual=residual), Ref(residual=residual)
        assert new.train_noise is False
        o1, i1 = new.reset(seed=seed)
        o2, i2 = ref.reset(seed=seed)
        assert np.array_equal(o1, o2) and i1 == i2
        acts = (
            np.random.default_rng(100 + seed).uniform(-1, 1, (new.max_steps, 1)).astype(np.float32)
        )
        for a in acts:
            o1, r1, _, t1, f1 = new.step(a)
            o2, r2, _, t2, f2 = ref.step(a)
            assert np.array_equal(o1, o2) and r1 == r2 and t1 == t2
            assert f1.keys() == f2.keys() and all(np.array_equal(f1[k], f2[k]) for k in f1)


@pytest.mark.parametrize("sigma", (0.2, 0.5, 1.0))
@pytest.mark.parametrize("variant", ("baseline", "dump"))
def test_train_noise_matches_lab11_noise_rows(sigma, variant):
    """train_noise=True, a = 0, level fixed, Lab 10 noise stream: equals the Lab 11 rollout of
    balanced + fix 2b on the noise_<sigma> row, so the noise model is the eval model.
    NoisyBoomEnvReal.reset draws one measurement before the rollout places the boom; the noise
    Generator given to the env is advanced past it."""
    m_tip, payload = L10.VARIANTS[variant]
    for th0, target in SCEN:
        ref_env = L10.make_env(variant, sigma=sigma)
        ref = L11.rollout(L11.Sched11(E.BAL_GAINS, E.EST), ref_env, th0, target)
        g = np.random.default_rng(0)  # stream of L10.make_env
        g.normal(0.0, 1.0), g.normal(0.0, 1.0)  # the draw in its reset
        env = E.BoomEnvRealRobust(
            residual=True, randomize=False, m_tip=m_tip, payload=payload, train_noise=True
        )
        obs, info = env.reset(
            seed=0, options={"th0": th0, "target": target, "sigma_deg": sigma, "noise_rng": g}
        )
        assert info["sigma_deg"] == sigma
        trunc = False
        while not trunc:
            obs, _, _, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        assert env.th == ref_env.th and env.energy_J / 1e3 == ref["E_mech_kJ"]
        assert np.array_equal(obs[:3], env.meas)  # policy sees the measurement


def test_train_noise_levels_and_other_draws():
    """Levels come from the eval set; the other randomization of a seed equals train_noise=False."""
    seen = set()
    for seed in range(200):
        on, off = E.BoomEnvRealRobust(residual=True, train_noise=True), E.BoomEnvRealRobust(
            residual=True
        )
        _, i_on = on.reset(seed=seed)
        _, i_off = off.reset(seed=seed)
        assert i_on.pop("sigma_deg") in E.NOISE_LEVELS_DEG
        assert i_on == i_off and on.th == off.th and on.target == off.target
        seen.add(on.sigma_deg)
    assert seen == set(E.NOISE_LEVELS_DEG)


def test_economical_point_gains():
    assert E.point_gains("economical") == (0.8, 0.0, 0.8)
    assert E.point_gains("balanced") == E.BAL_GAINS


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("i", range(2))
def test_zero_residual_reproduces_lab11_economical(variant, i):
    """Lab 12e: base="economical" with residual 0 equals the Lab 11 economical rollout."""
    th0, target = SCEN[i]
    ref_env = L10.make_env(variant)
    ref = L11.rollout(L11.Sched11(E.point_gains("economical"), E.EST), ref_env, th0, target)

    m_tip, payload = L10.VARIANTS[variant]
    env = E.BoomEnvRealRobust(
        residual=True, randomize=False, m_tip=m_tip, payload=payload, base="economical"
    )
    env.reset(seed=0, options={"th0": th0, "target": target})
    trunc = False
    while not trunc:
        _, _, _, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
    assert env.th == ref_env.th and env.om == ref_env.om
    assert env.energy_J / 1e3 == ref["E_mech_kJ"]
