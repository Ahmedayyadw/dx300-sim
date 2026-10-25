"""Step 6: energy-weight sweep. Accuracy vs hydraulic effort trade-off."""
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from boom_env import BoomEnv

WEIGHTS = [0.0, 0.05, 0.5, 2.0]
STEPS = 20_000
rng = np.random.default_rng(7)
SCENARIOS = [(rng.uniform(-0.4, 0.9), rng.uniform(-0.3, 0.9)) for _ in range(20)]

def run(model, env, th0, target):
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs, effort, trace = env._obs(), 0.0, []
    for _ in range(env.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, _, trunc, _ = env.step(a)
        effort += float(a[0]) ** 2 * env.dt
        trace.append(np.degrees(env.th))
        if trunc:
            break
    err = abs(np.degrees(env.target - env.th))
    return err, env.energy_J / 1000, effort, trace

results = {}
for w in WEIGHTS:
    env = BoomEnv(w_energy=w)
    model = SAC("MlpPolicy", env, device="cpu", seed=0, verbose=0, tensorboard_log="rl/tb")
    t0 = time.time()
    model.learn(total_timesteps=STEPS, tb_log_name=f"sweep_w{w}")
    errs, energies, efforts, _ = zip(*[run(model, env, *s) for s in SCENARIOS])
    trace = run(model, env, -0.3, 0.6)[3]
    results[w] = (np.mean(errs), np.mean(energies), np.mean(efforts), trace)
    print(f"w_energy={w:<5} | error {np.mean(errs):5.2f} deg | mech energy {np.mean(energies):5.0f} kJ"
          f" | valve effort {np.mean(efforts):5.2f} | trained in {time.time() - t0:.0f} s")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
t = np.arange(1, 201) * 0.05
for w, (_, _, _, tr) in results.items():
    ax[0].plot(t[:len(tr)], tr, label=f"w_energy={w}")
ax[0].axhline(np.degrees(0.6), ls="--", c="k", lw=1)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("boom angle [deg]")
ax[0].set_title("Lift -17 deg -> 34 deg"); ax[0].legend(); ax[0].grid(alpha=0.3)
for w, (e, _, eff, _) in results.items():
    ax[1].scatter(eff, e, s=60); ax[1].annotate(f"w={w}", (eff, e), xytext=(5, 5), textcoords="offset points")
ax[1].set_xlabel("valve effort (sum u^2 dt)"); ax[1].set_ylabel("mean final error [deg]")
ax[1].set_title("Accuracy vs effort (20 scenarios)"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("rl/03_energy_sweep.png", dpi=150)
print("plot saved: rl/03_energy_sweep.png")
