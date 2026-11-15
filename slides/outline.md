# DX300 boom: SAC, alternatives and a benchmark — outline

12 slides, 16:9. Slides 6–8 and 11 hold the measured Lab 13 results (updated 2026-11-15); before that,
slides 6–8 were expectations from the literature. Slides 5, 9 and 10 were brought in line with what Lab 13 did.
Every number has its source in `slides/sources.md`.

## 1. Problem: hold the boom on target with less energy
- DX300 boom: 6.245 m, one joint, gravity dominates the load
- Goal: reach a boom angle target within 20 cm at the tip
- Trade-off: tighter tracking costs more actuator energy
- Must survive payload, mass and torque-limit changes
- Stage 1 of the thesis; paper target ICEEE 2027

## 2. Plant: placeholder in Stage 1, real plant since Lab 10
- Stage 1 SAC results ran on a placeholder plant
- Labs 10–11: plant built from the Lab 08a Simscape export
- Checked against Simscape: boom-up time, work, holding torque
- Front attachment included: gravity torque ×2.21 at θ = 0
- Geometry, inertia, friction ESTIMATED; nothing measured on a machine
- Table: boom-up time 5.552 vs 5.619 s (−1.19 %); boom-up work 665.3 vs 672.3 kJ (−1.04 %); holding torque within 1.90 %

## 3. Lab 11 on the real plant: PID versus the energy floor
| Operating point | Kp / Kd | Reach 20 cm [s] | E_mech [kJ] | Gap, nominal | Gap, rows no noise | Gap, rows 1.0° noise |
|---|---|---:|---:|---:|---:|---:|
| fast | 5 / 2 | 1.18 | 171.0 | +10.4 % | +15.4 % | +71.2 % |
| balanced | 1.5 / 1 | 1.45 | 162.8 | +5.1 % | +13.7 % | +46.7 % |
| economical | 0.8 / 0.8 | 1.97 | 155.5 | +0.3 % | +10.7 % | +37.0 % |

- Floor: least work for a monotone move, no overshoot
- At 2.0 s the PID is on the floor; gap grows with speed
- 1.0° sensor noise multiplies the gap: oscillation, not overshoot
- Dump: 0 kg limit biased mass estimate; up to 80 % misses
- Fix 2b, adaptive filter: 0 of 54 dump rows over 5 %

## 4. Stage 1 SAC results: motivation for the redesign — "Placeholder plant, superseded"
- Figure: paper/figs/fig1_energy_vs_robustness.png
- Table: SAC w=0.1, SAC w=0.3, PID + FF, PID + I + FF (kJ, > 20 cm %, Payload ×1.2 %, worst variant %)
- SAC bought payload robustness at about twice PID energy
- Residual SAC on scheduled PID (09e): goal not met
- Redesign: real plant, reward above floor, gains not offsets

## 5. Why RL here, and why SAC
- Left for learning: fast-move excess and noise oscillation
- Reward: tracking error, speed and actuator power (w = 0.03)
- Action: one continuous residual on the balanced PID command
- Entropy term: broad exploration, less seed sensitivity [1, 2]
- Replay buffer reuses every step: fewer simulator steps
- Measured at 100k steps: SAC 1.01 % misses, PPO 24.34 % (slide 7)

## 6. Lab 13: no learner beats the economical PID — "Measured in Lab 13 · 100 unseen scenarios × 72 rows"
- Figure: paper/figs/lab13/fig1_energy_vs_error.png

| Group | Controller | Miss > 20 cm % | Mean error cm | E_mech kJ | Above floor % |
|---|---|---:|---:|---:|---:|
| classical | Economical PID | 0.00 | 2.1 | 159.1 | 20.0 |
| classical | Balanced PID | 0.00 | 1.4 | 167.5 | 26.4 |
| online on-policy | PPO 100k | 24.34 | 14.5 | 318.6 | 140.4 |
| online on-policy | PPO 1M * | 7.42 | 7.8 | 216.2 | 63.1 |
| online off-policy | SAC 100k | 1.01 | 5.7 | 191.5 | 44.5 |
| online off-policy | TD3 100k | 25.86 | 13.5 | 184.3 | 39.0 |
| offline | BC | 0.00 | 2.2 | 162.6 | 22.6 |
| offline | CQL | 10.94 | 8.0 | 190.8 | 43.9 |
| offline | IQL | 0.02 | 2.4 | 164.1 | 23.8 |

- Closest: BC. Same misses, but +1.13 kJ median (p = 6.3e-208)
- Best online learner is still SAC: 1.01 % misses, 191.5 kJ
- Footnote: * PPO 1M: 10× the steps, not equal budget. Offline rows: 1M transitions. Mean of 3 seeds.

## 7. On- vs off-policy, online vs offline: measured — "Measured in Lab 13 · 100 unseen scenarios × 72 rows"
- Figure: paper/figs/lab13/fig2_gap_to_floor.png
- On-policy vs off-policy (online, 100k steps)
  - PPO: 24.34 % misses, 140.4 % above floor
  - SAC: 1.01 % misses, 44.5 % above floor
  - TD3: 25.86 % misses, 39.0 % above floor
  - PPO with 10× the steps: 7.42 %, 63.1 %
- Online vs offline (1M PID transitions)
  - BC: 0.00 % misses, 22.6 % above floor
  - IQL: 0.02 % misses, 23.8 % above floor
  - CQL: 10.94 % misses, 43.9 % above floor
- Bold line: Economical PID: 0.00 % misses, 20.0 % above floor. Offline is close to it because its data is PID data.
- Footnote: Not tested: DDPG [6], TQC [8], TD3+BC [11], MBPO [13], MPC.

## 8. Why the learners lose: oscillating commands — "Measured in Lab 13 · 100 unseen scenarios × 72 rows"
- Figure: paper/figs/lab13/fig4_smoothness_vs_oscillation.png
- Online learners move the valve 4 to 32 times more than the PID
- Mean |du/dt|: 2.23 to 15.96 against 0.50 per second
- 55 to 70 % of their energy above the floor is oscillation
- Not sensor noise: 5.5 to 14.2 cm error with no noise (PID 1.4 cm)
- BC and IQL are as smooth as the PID (0.55)
- CQL is smooth (0.76), but 56 % of its excess is overshoot

(The qualitative method table of the earlier slide 8 is removed. The methods that were not tested are
named on slide 7 and in its speaker notes.)

## 9. Benchmark choice: why our own plant
- (a) Own real plant: PID points and energy floor. Used in Lab 13
- (b) Spinelli 2024 [21]: PPO, real handler, slew 0.46° vs PI 12.38°
- (b) Kurinov 2020 [22]: PPO-CMA, simulation, 67–97 % bucket mass
- (b) Hu 2026 [23]: simulated 21 t, 20–22 % less energy, relative
- (c) DMC [15] / D4RL [14]: code sanity check, not task
- Bold line: Fair comparison only in (a), so Lab 13 uses it. (b) differs in machine, task and metric; (c) has no boom.

## 10. Benchmark protocol used in Lab 13
- Same plant, observation and reward (w = 0.03) for all
- Every learner: residual on the balanced PID, seeds 0/1/2
- Online: SAC, TD3, PPO, 100k steps, SB3 defaults
- Offline: BC, CQL, IQL on 1M PID transitions (d3rlpy)
- Metrics: misses at 10/20/30 cm, error, E_mech, gap to floor
- Paired Wilcoxon test against the economical PID
- Box: test rows (Lab 12e protocol): 100 unseen scenarios, 18 conditions × 4 variants (baseline · fill_half · fill · dump), noise 0.2° / 0.5° / 1.0° · J · friction. Not done: TQC, tuning per algorithm, IQM with intervals [16, 17]

## 11. Reward mismatch, and two questions
- Figure: paper/figs/lab13/fig5_reward_mismatch.png (hollow markers = return during training; no return exists for BC, CQL, IQL)
- Reward (w = 0.03) ranks the balanced PID above the economical PID
- Return −9.78 against −10.30, but 26.4 % against 20.0 % above the floor
- So no learner is pushed toward the economical PID
- Reward kept as in Lab 12, so the setup stays identical
- Box, questions for Dr. Sina:
  - (a) Is this framing right for the paper?
  - (b) Should I test a corrected reward (Lab 14) before submission?

(The Lab 12 gain-scheduler design of the earlier slide 11 is removed from the deck; it is in
`docs/12_sac_real_plant.md`.)

## 12. References
- [1]–[24], each with DOI or URL (see sources.md)
