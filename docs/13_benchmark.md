# Lab 13: benchmark of other learning methods (plan)

Status: **done (2026-11-15). 30 models trained, evaluated with the Lab 12e protocol. Results: `rl/13_benchmark.md`
and `rl/13_benchmark.csv`.** Plan approved by the user in chat on 2026-11-14 and 2026-11-15.
Approved choices: d3rlpy in a separate `.venv-d3rlpy` (option A), sigma 0.1 with the 50/50 mix, 100k offline
gradient steps, the PPO 1M row. Changed after the smoke test: CQL at 1M transitions only (Section 8).

Question from the supervisor: why SAC, what other methods could work, and a benchmark. Lab 13 trains
TD3 and PPO (online) and BC, CQL and IQL (offline) in the setup of the best Lab 12 SAC and evaluates
them with the Lab 12e protocol, next to the PID points and the Lab 12 SAC from the existing result files.

## 0. Status of the values

- **OFFICIAL**: boom length (read from `dx300sim/dx300_params.py`), pump flow, relief pressure.
- **ESTIMATED**: every simulation result and every measured time quoted here.
- **ASSUMED**: every design value of this lab (budgets, noise level, mix, hyperparameters, E_pump model).
- Reference numbers (ESTIMATED, `rl/12e_stats.md`): economical PID 0.00 % misses over 20 cm, 159.1 kJ,
  20.0 % above the floor; best SAC `res_bal_w0.03` 1.01 +- 0.30 % misses, 191.5 +- 7.7 kJ.

## 1. The reference setup (what "identical" means)

Read from `rl/12_robust_train.py`, `rl/boom_env_real_robust.py` and `rl/12_eval.py`:

| item | value |
|---|---|
| env | `BoomEnvRealRobust(w_energy=0.03, residual=True)`: randomize on, `train_noise` off, base `balanced` |
| observation | `[theta, omega, e, integral of e]`, integral clipped to +-1 |
| action | residual `a` in [-1, 1], `u = clip(u_balanced + a, -1, 1)` |
| reward | `-(e^2 + 0.01 omega^2 + w P / P_REF)`, w = 0.03 |
| training distribution | start and target uniform over the end-stop range, tip mass, payload step, torque limit and J factor randomized (ranges in the env file) |
| budget | 100,000 env steps, seeds 0, 1, 2 |
| SAC settings | SB3 defaults, `device="cpu"` |
| evaluation | deterministic action, 100 unseen scenarios x 72 rows, Lab 11 rollout |

Every Lab 13 learner uses this env class unchanged (imported, not copied).

## 2. 13a: online RL (TD3, PPO)

Same env, 100k env steps, seeds 0, 1, 2, `device="cpu"`, `MlpPolicy`. Changes from the library defaults:

| algorithm | change | source |
|---|---|---|
| TD3 | `action_noise = NormalActionNoise(mean=0, sigma=0.1)` | SB3 TD3 doc example (`docs/modules/td3.md`). The default is `None`: a deterministic policy with no exploration. |
| PPO | `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)`; the saved statistics are applied at evaluation, frozen | SB3 example "Normalizing input features" (`docs/guide/examples.md`) |

Nothing else is changed. Defaults that differ between the algorithms on their own (installed SB3 2.9.0,
printed from the models; not my changes):

| | SAC (Lab 12) | TD3 | PPO |
|---|---|---|---|
| learning rate | 3e-4 | 1e-3 | 3e-4 |
| network | [256, 256] ReLU | [400, 300] ReLU | [64, 64] tanh |
| batch | 256 | 256 | 64, n_steps 2048, 10 epochs |
| exploration | entropy, `ent_coef="auto"` | Gaussian action noise 0.1 | Gaussian policy, `log_std_init=0` |

Two points to know before reading the PPO row:

1. PPO normalization rescales the same four observation signals and, during training only, the reward.
   The reward function is the same. This is listed as a change, not hidden.
2. On-policy PPO at 100k steps gets 48 policy updates (100k / 2048). The equal-step row is the fair
   comparison asked for, but it is a known handicap. **Proposed extra row**: PPO at 1M steps, labelled
   "not equal budget", so the paper can separate "on-policy needs more samples" from "PPO cannot do it".
   Cost about 20 min. Drop it if you do not want it.

## 3. 13b: offline RL (BC, CQL, IQL)

### Dataset

- Env: the same `BoomEnvRealRobust(w_energy=0.03, residual=True)`, so observation, action meaning and
  reward are the ones of Section 1. 5,000 episodes x 200 steps = 1,000,000 transitions.
- Episode seeds 1,000,000 + i (training distribution). The script asserts that no dataset episode
  starts from one of the 100 unseen (start, target) pairs.
- Behaviour, 50 % of the episodes each (ASSUMED mix):
  - balanced PID + noise: `a = eps`
  - economical PID + noise: `a = u_economical + eps - u_balanced`, clipped to [-1, 1]. A second
    `Sched11` with the economical gains runs next to the base controller and is told the applied command.
  - `eps ~ N(0, sigma^2)`, new draw every control step (20 Hz).
- Stored action = the residual passed to the env, so the offline policies are residuals on the balanced
  PID exactly like SAC, TD3 and PPO.
- Episodes end by time limit only: `terminals = 0`, `timeouts = 1` at step 200.
- File `rl/models/13_dataset_1M.npz` (about 25 MB, in the gitignored models folder).

### Proposed noise level: sigma = 0.1 (ASSUMED)

Measured on 300 training-distribution episodes per cell (scratch probe, seeds 900,000+, no file written,
ESTIMATED). Output of the probe:

```
behav sigma | miss>20cm % | mean err cm | p95 cm | E_mech kJ | return | mean|du/dt| 1/s | clipped a % | mean|a|
bal 0.00 |  0.00 |  0.62 |   0.8 |  183.9 |   -9.78 |  0.09 |  0.00 | 0.000
bal 0.05 |  0.00 |  2.32 |   5.6 |  203.8 |   -9.85 |  1.32 |  0.00 | 0.040
bal 0.10 |  0.33 |  4.72 |  11.3 |  227.5 |   -9.94 |  2.60 |  0.00 | 0.080
bal 0.20 | 10.33 |  9.64 |  22.8 |  278.9 |  -10.18 |  5.13 |  0.00 | 0.159
bal 0.30 | 25.00 | 14.62 |  34.5 |  338.0 |  -10.51 |  7.47 |  0.10 | 0.239
eco 0.00 |  0.00 |  1.09 |   1.8 |  176.2 |  -10.30 |  0.07 |  0.34 | 0.043
eco 0.05 |  0.00 |  3.67 |   8.8 |  196.4 |  -10.37 |  1.29 |  0.31 | 0.073
eco 0.10 |  3.33 |  7.47 |  18.1 |  220.9 |  -10.48 |  2.56 |  0.33 | 0.108
eco 0.20 | 27.67 | 15.09 |  36.2 |  274.9 |  -10.84 |  5.05 |  0.38 | 0.182
eco 0.30 | 48.00 | 22.83 |  56.7 |  337.2 |  -11.38 |  7.37 |  0.60 | 0.259
```

Reading:

- sigma = 0.05: the data stays almost on the PID trajectories (0 % misses). Too narrow for CQL and IQL to
  learn the value of any action the PID did not take.
- sigma = 0.2: 10 to 28 % of the episodes miss by over 20 cm and energy is 50 % up. The data is no longer
  near-expert.
- sigma = 0.1: 0.3 to 3.3 % misses, mean residual size 0.08 to 0.11. It is also the exploration noise of
  TD3 (Section 2), so online and offline explore at the same scale.

One thing the probe shows that matters for the conclusion: under the training reward at w = 0.03 the
balanced PID scores better than the economical PID (return -9.78 against -10.30, sigma = 0 rows). The
reward is kept as it is, because the setup must be identical to Lab 12. But a method that maximises this
reward is not being pushed towards the economical PID. This goes into the 13d discussion.

### Training

- Library: d3rlpy 2.8.1 (not installed yet, see Section 6). Library defaults for every hyperparameter;
  the script prints the full config of the installed version into `rl/13_benchmark.md`. I will not quote
  the default values from memory here.
- Budget: 100,000 gradient steps (ASSUMED), the number of updates the SAC of Lab 12 made in its 100k env
  steps. Seeds 0, 1, 2. The final model is used. No checkpoint is picked with the unseen scenarios.
- Each policy is exported with `save_policy` as TorchScript (`rl/models/13_{algo}_{size}_s{seed}.pt`), so
  the evaluation needs PyTorch only.
- Dataset size: 100k, 300k, 1M transitions = the first 500, 1,500, 5,000 episodes (behaviours interleaved,
  so every size keeps the 50/50 mix). Same 100k gradient steps at every size.
- **Proposal**: train all three methods at all three sizes (27 runs) instead of the best method only.
  Reason: picking the best method needs the evaluation, so "best only" means a second training stop.
  BC and IQL are cheap. The report shows the size table for the best method, as asked; the rest goes to
  the CSV.

Expectation, stated before the runs: BC of a 50/50 mix should land near a PID halfway between balanced
and economical. That is a legitimate outcome and will be reported as such.

## 4. 13c: evaluation

- `rl/12_eval.py` is imported unchanged. A new controller class wraps any policy (SB3 model, PPO with
  frozen normalization, TorchScript) behind the `SacCtrl` interface. Same rollout, same 7,200 moves.
- Checks before the evaluation, on the 20 tuning scenarios only: residual 0 equals the balanced PID bit
  for bit; each policy type gives the same trajectory as stepping the env with the library's own predict.
- Metrics per controller, mean +- std over 3 seeds: miss rate at 10 / 20 / 30 cm, mean and p95 tip error,
  E_mech, E_pump (lower bound, ASSUMED), gap to floor. Paired Wilcoxon against the economical PID per
  move (seed mean), on error and on E_mech. Training time from `rl/13_train_log.csv`.
- Rows taken from existing files, not re-run: economical and balanced PID and SAC `res_bal_w0.03` from
  `rl/12e_moves.csv.gz`; SAC training time from `rl/12_robust_train_log.csv`.
- Table groups: classical (economical, balanced) / online on-policy (PPO) / online off-policy (SAC, TD3) /
  offline (BC, CQL, IQL at 1M).
- "Beats the economical PID" = no more misses over 20 cm and lower mean E_mech, with Wilcoxon p < 0.05 on
  energy and no significantly larger error (ASSUMED rule, the Lab 12h "beats PID on both" plus the test).

## 5. 13d: diagnosis

One table, one line per controller:

| column | definition |
|---|---|
| mean abs(du/dt) [1/s] | mean of abs(u_k - u_(k-1)) / dt over the move, u = applied command |
| oscillation share % | oscillation kJ / (E_mech - floor), Lab 11 split |
| direct excess share % | direct excess kJ / (E_mech - floor) |
| error by noise level | mean tip error on the rows with sensor noise 0 / 0.2 / 0.5 / 1.0 deg |

abs(du/dt) was never logged in Lab 12. For this table only, the two PIDs and the three SAC models are
rolled out again with the logging controller (about 4 min). Their error and energy must equal
`rl/12e_moves.csv.gz`, which is asserted. The 13c table still uses the existing file.

## 6. Decision needed: how to install d3rlpy

`pip install --dry-run d3rlpy` in `.venv` reports:

```
Would install click-8.5.1 colorama-0.4.6 d3rlpy-2.8.1 dataclasses-json-0.6.7 gym-0.26.2 gym-notices-0.1.0
gymnasium-1.0.0 h5py-3.16.0 marshmallow-3.26.2 mypy_extensions-1.2.0 structlog-26.1.0 typing-inspect-0.9.0
```

`.venv` has gymnasium 1.4.0. A plain install would move it to 1.0.0 and add the old `gym` 0.26.2, under
the env that produced every Lab 12 result.

| option | effect | risk |
|---|---|---|
| **A (recommended)** separate `.venv-d3rlpy`, used by `rl/13_offline_train.py` only | `.venv` untouched. Dataset and evaluation run in `.venv`; policies cross over as TorchScript. | Second PyTorch install (CPU wheel, same version 2.14.1). One script runs outside `.venv`, against the CLAUDE.md rule. |
| B `pip install --no-deps d3rlpy` plus its small dependencies in `.venv` | gymnasium stays 1.4.0 | d3rlpy pins gymnasium 1.0.0 and may fail on import. Known only after installing. |
| C plain `pip install d3rlpy` in `.venv` | simplest | gymnasium changes under Lab 12; the bit-for-bit tests would have to pass again. |

## 7. Files

All new. No existing script, model or result file is modified.

| file | purpose |
|---|---|
| `rl/13_common.py` | env factory, paths, policy adapters, logging controller |
| `rl/13_online_train.py` | TD3 and PPO, `--algo --seed --steps --dry-run` |
| `rl/13_dataset.py` | 1M-transition dataset and its summary |
| `rl/13_offline_train.py` | BC, CQL, IQL, `--algo --size --seed --dry-run` |
| `rl/13_eval.py` | 13c and 13d, `--check --smoke --report` |
| `rl/13_train_log.csv` | wall-clock per run |
| `rl/13_moves.csv.gz`, `rl/13_benchmark.csv`, `rl/13_benchmark.md` | results |
| `rl/models/13_*` | models and dataset (gitignored) |

## 8. Expected run time (measured in the smoke tests, ESTIMATED)

```
SAC: 3000 steps in 26.0 s = 115 steps/s -> 100k steps in 14.4 min (1 run alone)
TD3: 3000 steps in 18.4 s = 163 steps/s -> 100k steps in 10.2 min (1 run alone)
PPO: 8192 steps in 5.9 s = 1398 steps/s -> 100k steps in 1.2 min (1 run alone)
13_bc_1M_s0_smoke: 500 gradient steps on 4000 transitions in 2 s = 291 steps/s
cql 3 parallel threads 2: in 30 s = 10 steps/s      (the other two: 9 and 8)
iql 3 parallel threads 2: in 14 s = 72 steps/s      (the other two: 71 and 65)
```

CQL at about 9 steps/s needs 3.1 h per run. All 9 CQL runs would take 9.3 h and the whole plan about
11.5 h, over the 10 h limit. Approved smaller plan: **CQL at 1M transitions only** (3 runs); BC and IQL keep
100k / 300k / 1M. The dataset-size table is therefore for BC and IQL.

| block | runs | time with 3 in parallel |
|---|---:|---:|
| TD3 100k | 3 | 17 min |
| PPO 100k | 3 | 2 min |
| PPO 1M | 3 | 20 min |
| dataset 1M | 1 | 2 min |
| BC | 9 | 20 min |
| IQL | 9 | 75 min |
| CQL 1M | 3 | 3.1 h |
| **total training** | 33 | **about 5.3 h** |
| evaluation (not training) | 35 controller-seeds | about 1 h |

The smoke flag of the Lab 13 scripts is `--smoke`, not `--dry-run`: `rl/10_sched_real.py` reads
`--dry-run` at import and then keeps only 5 of the 100 unseen scenarios.

## 9. Open points for approval

1. d3rlpy install: option A, B or C (Section 6).
2. Dataset noise sigma = 0.1 and the 50/50 mix.
3. Offline budget 100k gradient steps; all three methods at all three dataset sizes (27 runs).
4. PPO with VecNormalize; the extra PPO 1M row yes or no.
