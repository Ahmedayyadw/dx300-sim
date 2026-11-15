# Lab 12: learned controller on the real plant (design)

Status: **design, waiting for approval**. Nothing is implemented or trained yet.

Goal (from the task): against the balanced PID with fix 2b, on the 100 unseen scenarios and all 72 rows:
mean gap to the floor cut by at least 25 % relative; mean move time within +0.1 s; no row over 5 % miss and
no row more than 1 point worse; the no-stress `baseline` row not more than 1 kJ worse.

## 0. Status of the values and use of the unseen scenarios

- **OFFICIAL**: boom length 6.245 m (`dx300sim/dx300_params.py`), pump flow, relief pressure, cylinder bore and rod.
- **ESTIMATED**: every plant value of Lab 08a and every simulation result quoted here.
- **ASSUMED**: every design value of this lab (filter grid, gain range, history length, reward weights, SAC
  settings, training distribution, selection rules). Each is tagged below.
- Base controller: `Sched11((1.5, 0, 1), est={"t_f": 1.0, "adapt": [1.0, 2]})` of `rl/11_real_plant_tuning.py`
  (balanced point, fix 2b `adapt_Tf1_c1_n2`). Loaded, not modified.

**Unseen scenarios.** I read `rl/11_real_plant_tuning.md` as instructed. It contains unseen results (sections 1.4,
3, 4.3 and "What is left"), and the task text quotes three of them (26.4 %, 37 to 71 %, 10.4 %). No value in this
design comes from them. Every number used for a choice below is from the 20 original scenarios (RNG seed 7):
Lab 11 sections 1.2, 1.3, 2, 4.1, 4.2, and two read-only runs made for this design (`Sched11`, tuning set, no file
written). One thing I know from the unseen tables and did not use: the `fast` gains oscillate on `J_x0.38` with a
light bucket. The gain range of 12b was not narrowed because of it.

Tuning-set starting point, balanced PID with fix 2b, 20 original scenarios x 72 rows, 5 noise seeds on the noisy
rows (run for this design, all ESTIMATED):

| rows | floor kJ | E_mech kJ | gap % | overshoot kJ | oscillation kJ | direct excess kJ | reach 20 cm s | miss % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all 72 | 141.2 | 175.1 | 24.0 | 11.8 | 16.7 | 5.4 | 1.50 | 0.00 |
| no sensor noise (24) | 141.2 | 158.9 | 12.5 | 7.0 | 6.1 | 4.6 | 1.53 | 0.00 |
| 0.2 deg (16) | 141.2 | 167.2 | 18.4 | 8.9 | 11.3 | 5.8 | 1.51 | 0.00 |
| 0.5 deg (16) | 141.2 | 180.6 | 27.9 | 13.3 | 20.1 | 5.9 | 1.48 | 0.00 |
| 1.0 deg (16) | 141.2 | 202.0 | 43.0 | 20.3 | 34.7 | 5.7 | 1.48 | 0.00 |

Reading: of 33.9 kJ mean excess, about 15 kJ is added by sensor noise (overshoot and oscillation rise with the
noise level), about 13 kJ is overshoot and oscillation that remain without sensor noise (mostly after the payload
step), 5.4 kJ is direct excess. A 25 % cut is 8.5 kJ.

## 1. 12a: balanced PID with a low-pass filter on the measured speed

```
w_f[k] = w_f[k-1] + alpha (w_meas[k] - w_f[k-1]),  alpha = dt / (T_w + dt),  w_f = 0 at reset
u = (K_REF (Kp e - Kd w_f) + tau_g(theta, m_hat)) / k_hat          Kp 1.5, Kd 1, Ki 0
```

- The filter acts on the derivative term only. The mass estimator and the flow gate keep the raw measured speed,
  so fix 2b stays as tuned in Lab 11. T_w = 0 gives the base controller exactly (checked bit for bit).
- Grid (ASSUMED): T_w = 0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4 s. If the best value is the largest, the
  grid is extended upwards, as in Lab 11c.
- Tuning set: 20 original scenarios x 72 rows, 5 noise seeds on the noisy rows.
- Rule (ASSUMED, the Lab 12 goal applied to the tuning set): admissible = no row over 5 % miss, no row more than
  1 point above the base, mean reach time at most base + 0.1 s, `none` `baseline` E_mech at most base + 1 kJ.
  Choice = lowest mean gap over the 72 rows among the admissible; tie: smaller T_w.
- Expected limit (ESTIMATED): with 1.0 deg noise the command noise of the PD law is 0.026 from Kp (1.5 x 0.0175 rad)
  and 0.035 from Kd (1 x 0.0349 rad/s). The speed filter can only remove the second part.

## 2. 12b: SAC as a gain scheduler on the balanced point

### 2.1 Action form

```
a = (a_p, a_d) in [-1, 1]^2          g_p = 4^a_p,  g_d = 4^a_d      (0.25 to 4, ASSUMED)
u = (K_REF (Kp g_p e - Kd g_d w) + tau_g(theta, m_hat)) / k_hat     e, w = measured, as the PID sees them
```

- **Hard requirement 1.** At e = 0 and w = 0 the bracket is 0 for every action: u = tau_g(theta, m_hat) / k_hat.
  The action multiplies e and w; it adds nothing. A constant action cannot hold a constant command offset, which
  was the Lab 09d flaw.
- Steady state: e_ss = (model torque error) / (K_REF Kp g_p). The action can only scale an error the base already
  has, by at most 4 (g_p = 0.25). With a correct m_hat the error is 0 for every action.
- Both gains stay positive: the law is always a PD law towards the target.
- Effective range Kp 0.375 to 6, Kd 0.25 to 4. It contains the three Lab 11 points (5 / 2, 1.5 / 1, 0.8 / 0.8).
- Why this can cut each part of the gap (ESTIMATED reasoning, not a result):
  - noise: lower gains at rest pass less noise (down to 25 %); with the history the policy can also shrink single
    noisy samples (g = filtered / measured value, inside the range).
  - direct excess: the cylinder pushes down when Kp g_p |e| > tau_g / K_REF = 0.415 (theta = 0, 3406 kg). At
    Kp 1.5 that is |e| > 0.28 rad; at g_p = 0.25 it is |e| > 1.1 rad.
  - payload step: higher gains right after the step shorten the sag.
- Rejected: residual times a gate on measured e and w. With 1.0 deg noise the measured e and w are never 0 at
  rest, so the gate stays open and a constant residual can bias the position again.
- The estimator (fix 2b) and the gain schedule k_hat are unchanged and read the total command.

### 2.2 Zero initialisation and its proof

- Weights and bias of the last actor layer (`actor.mu`) set to 0. Deterministic action = tanh(0) = 0, so
  g_p = g_d = 4^0 = 1 exactly.
- Check before training (`rl/12_sac_train.py --check`): untrained model against `Sched11` on the 20 original
  scenarios x 72 rows: every command, E_mech and final error equal, largest difference 0.0 (asserted with `==`).
- Second check for requirement 1: 10 000 random states with e = w = 0 and random actions give the same u exactly.
- Exploration start (ASSUMED): `actor.log_std` bias -1.5, weights 0 (std 0.22, gain factor about x1.36), so early
  data stay near the base controller.

### 2.3 Observation (29 values, measured signals only)

| values | count | source |
|---|---:|---|
| theta[k] | 1 | measured angle (noisy) |
| e[k] ... e[k-7] | 8 | target minus measured angle |
| w[k] ... w[k-7] | 8 | measured speed (noisy) |
| u[k-1] ... u[k-8] | 8 | commands already sent |
| m_hat / 3406, m_fast / 3406 | 2 | fix 2b estimator (slow and 0.2 s fast estimate), built from the measured signals and the 5 % noisy torque reading |
| a[k-1] | 2 | previous action (for the action-rate term) |

- History 8 steps = 0.4 s (ASSUMED), the range of the 12a filter grid. At reset the history is filled with the
  first measurement, u with 0.
- No clock, no true state, no true mass, no J or friction factor. The policy cannot know when the payload steps.
- **For approval:** m_hat and m_fast are estimator outputs, not raw sensors. The PID uses m_hat. If you want raw
  sensors only, these two are replaced by the torque reading history.

### 2.4 Reward

Per control step k (0.05 s), computed from the true state (training only):

```
r[k] = -( e^2 + 0.1 |e|  +  0.05 [d > 1]  +  w_E X[k]  +  0.005 |a[k] - a[k-1]|^2 )
last step only:  r[K] -= 5 min(d, 3)^2

e     true angle error [rad]
d     = |e| x 6.245 m / 0.20 m          tip error in units of the 20 cm bound
X[k]  = (dE_mech[k] - dFloor[k]) / 1 kJ  mechanical work of the step above the floor rate
dFloor[k] = Lab 11 floor of the new angle covered towards the target in this step (0 on any repeat pass)
```

- **Tracking**: `e^2 + 0.1 |e|` as Labs 08b and 09d, plus 0.05 per step outside 20 cm (1.0 per second; this is the
  reach-time metric).
- **Power above the floor rate**: the sum of X over an episode equals overshoot + oscillation + direct excess of
  the Lab 11 split (asserted in the smoke test). A move on the floor costs 0. Stopping short earns nothing: the
  floor of the angle not covered is not credited, so the 09d incentive to save energy by not finishing is gone.
- **Action rate**: 0.005 |da|^2 (ASSUMED). A gain that changes by x2 every step costs 0.5 per episode.
- **Final error**: 5 min(d, 3)^2 at the last step (ASSUMED). A miss costs at least 5, about one episode of
  tracking cost. Limit: the policy has no clock, so this term acts as an expected penalty on error at rest.
- **Energy weights** (ASSUMED, calibrated on the tuning set): w_E = **0.06** and **0.12** per kJ.
  Calibration, fixed-gain PIDs with fix 2b, 20 original scenarios, `baseline`, mean per move (ESTIMATED):

  | Kp / Kd | sensor noise | sum of e^2 + 0.1 abs(e) | steps outside 20 cm | excess kJ |
  |---|---:|---:|---:|---:|
  | 1 / 0.8 | 0 | 4.97 | 32.5 | 4.2 |
  | **1.5 / 1** | 0 | 4.66 | 28.7 | 7.5 |
  | 2 / 1.2 | 0 | 4.56 | 27.4 | 10.2 |
  | 1 / 0.8 | 1.0 deg | 5.01 | 32.3 | 35.6 |
  | **1.5 / 1** | 1.0 deg | 4.71 | 27.9 | 44.5 |
  | 2 / 1.2 | 1.0 deg | 4.63 | 26.9 | 55.7 |

  The balanced point beats both neighbours in this reward for 0.060 < w_E < 0.152 without noise and for
  0.012 < w_E < 0.058 with 1.0 deg noise. The two intervals do not overlap. **0.06** lies on the edge between
  them: a fixed-gain policy has almost no reason to leave the balanced move time in either case. **0.12** leans
  to energy: more saving, more risk on the +0.1 s limit.
- Base controller, reward per episode at w_E = 0.06 (ESTIMATED): tracking 4.7 + time 1.4 + energy 0.45 (no noise)
  to 2.7 (1.0 deg) + final 0.00 to 0.07.

### 2.5 Training distribution (per episode, all ASSUMED)

| item | draw |
|---|---|
| plant | `rl/boom_env_real.py`, 10 s, 200 steps, unchanged |
| start, target | theta0 ~ U(-0.4, 0.9), target ~ U(-0.3, 0.9) rad, the ranges of the scenario generator |
| sensor noise | p = 0.2: 0 deg; else U(0, 1.0) deg. Speed noise = angle noise / dt x 0.1, as Lab 10 |
| torque reading noise | p = 0.5: 5 %; else 0 |
| J factor | p = 0.3: 1; else log-uniform 0.38 to 1.5 |
| friction factor | p = 0.3: 1; else log-uniform 0.5 to 2 |
| payload variant | `baseline`, `fill_half`, `fill`, `dump`, each p = 0.25 |
| payload step | uniform in step 80 to 140 (4 to 7 s); the evaluation uses step 100 |

- **Different from the 100 unseen scenarios**: a draw closer than 0.001 rad (in theta0 and target) to any of the
  100 unseen or the 20 original scenarios is redrawn. Only the start and target angles of the unseen set are read
  for this, no result. The smoke test asserts the smallest distance over 10 000 resets. The 20 original scenarios
  stay out of training too, so they can serve as the validation set of 2.7.
- Noise streams come from the SAC seed; the evaluation uses the Lab 10 streams.

### 2.6 SAC settings

| setting | value |
|---|---|
| library | stable-baselines3 2.9.0 `SAC("MlpPolicy")`, no new dependency |
| network | 2 x 256, ReLU (SB3 default) |
| learning rate, batch, gamma, tau | 3e-4, 256, 0.99, 0.005 (SB3 defaults) |
| buffer | 1 000 000 (holds every step) |
| entropy | `auto_0.01`, target -2 (ASSUMED start value; rewards per step are about 0.01) |
| learning starts | 2 000 steps |
| steps per run | 300 000 = 1 500 episodes (ASSUMED) |
| runs | w_E 0.06 and 0.12, seeds 0, 1, 2 = 6 runs |
| evaluation | deterministic action |

- Budget rule: the smoke test (at most 5 minutes) measures steps per second on cpu and on cuda. Six runs, 3 in
  parallel, need 42 steps/s per run to finish 300 000 steps in 4 hours (Lab 09d ran at 48 steps/s). If the
  measured speed is lower I do not start: I propose 200 000 steps (needs 28 steps/s) or fewer runs.

### 2.7 Which weight counts as "the SAC" (fixed before the unseen run)

After training, both weights run on the tuning set (20 original scenarios x 72 rows, 5 noise seeds, mean over the
3 SAC seeds). The weight that is admissible by the rule of section 1 and has the lower mean gap is written to
`rl/12_choice.json` together with the 12a filter time. The unseen evaluation refuses to start without that file.
If no weight is admissible on the tuning set, the one with the fewest violated conditions is used and that is
stated. Both weights are reported on the unseen set; the goal is judged on the chosen one.

## 3. 12c: evaluation

- 100 unseen scenarios, 72 rows (18 conditions x 4 variants), Lab 10 noise streams, run once.
- Controllers: balanced + fix 2b (base), 12a, SAC per weight (mean and std over 3 seeds), fast and economical
  with fix 2b for reference.
- Metrics per row, as Lab 11: miss %, E_mech, gap to the floor %, overshoot, oscillation, direct excess,
  shortfall, time to reach 20 cm.
- Cross-check: the base rows must equal the `balanced` `adapt_Tf1_c1_n2` rows of `rl/11_real_plant_tuning.csv`
  exactly.
- Goal test, on the mean over the 3 seeds per row:
  1. gap = 100 (mean E_mech / mean floor - 1) over the 72 rows, as Lab 11: SAC at most 0.75 x base.
  2. mean of the 72 reach times: SAC at most base + 0.1 s.
  3. every row: SAC miss at most 5 % and at most base + 1 point.
  4. `none` `baseline`: SAC E_mech at most base + 1 kJ.
  The worst single seed is reported next to the mean. Same four tests for SAC against 12a, as information.
- Output: `rl/12_comparison.csv`, `rl/12_comparison.md`.

## 4. Files (new only)

| file | content |
|---|---|
| `rl/12_common.py` | 12a controller, SAC controller, training env, floor-rate reward, shared rows |
| `rl/12_filtered_pid.py` | 12a sweep on the tuning set, writes `rl/12_filter_tuning.csv`, `rl/12_filter_choice.json` |
| `rl/12_sac_train.py` | `--check`, `--smoke`, training (`--w`, `--seed`, `--device`), skips saved models |
| `rl/12_eval.py` | `--step validate` (tuning set), `--step eval` (unseen), report |
| `rl/models/12_sac_w{w}_s{seed}.zip`, `rl/12_train_log.csv` | models (gitignored) and training log |

## 5. Risks

- The gain range reaches Kp 6 / Kd 4. If the policy sits there under noise the energy rises; the energy term and
  the training on `J x0.38` are the only guards. A narrower range (0.5 to 2) is the fallback.
- A gain scheduler cannot filter like a linear low-pass. 12a may beat the SAC on the noise rows. That is reported
  as it comes out.
- Reward weights are calibrated on fixed-gain PIDs. A learned policy may trade time and energy differently.
- 300 000 steps and 3 seeds: seed spread may be larger than the effect.
