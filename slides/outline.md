# DX300 boom: SAC, alternatives and a benchmark — outline

12 slides, 16:9. Slides 6–8 are expectations from the literature, not our results.
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
- Energy above the floor goes in the reward directly
- Continuous gain action: SAC is built for continuous control
- Entropy term: broad exploration, less seed sensitivity [1, 2]
- Replay buffer reuses every step: fewer simulator steps

## 6. Off-policy vs on-policy (expected, not our results)
- SAC (off-policy) learns from a replay buffer of old data
- PPO / TRPO / A2C (on-policy) need fresh rollouts per update [3–5]
- Expect PPO: more simulator steps for similar score [1, 4]
- Expect PPO: stable, but implementation details matter [19, 20]
- Final cm and kJ for this task: unknown until measured

## 7. Online vs offline RL (expected, not our results)
- Now: online; the agent acts in the simulator while learning
- Offline: learn only from logged machine data, no interaction
- Methods: CQL, IQL, TD3+BC; benchmark datasets D4RL [9–11, 14]
- Risk: policy leaves the data; values get overestimated [12]
- Relevant once DX300 logs exist (Stage 2 quarry data)

## 8. Other ways to solve it, and how they compare (expected, not our results)
| Method | Sample eff. | Stability | Constraints | Needs model | Fit here |
|---|---|---|---|---|---|
| SAC (ours) | High | Good across seeds | Via reward | No | Good, in use |
| TD3 / TQC | High | Good | Via reward | No | Good, cheap to test |
| DDPG | High | Brittle | Via reward | No | Fair |
| PPO / TRPO | Lower | Good, detail-sensitive | Via reward | No | Fair; domain standard |
| MBPO | Higher than model-free | Medium | Via reward | Learned | Fair |
| Offline: CQL / IQL / TD3+BC | No env steps | Bound by data | Via data | No | Later, with logs |
| PID + feed-forward | No training | Good | Saturation only | Gravity term | Baseline (Lab 11) |
| LQR / MPC | No training | Good | MPC: hard limits | Yes | Strong if model good |

(The former separate "Other ways" bullet slide is merged here; its content is in the speaker notes.)

## 9. Candidate benchmarks
- (a) Own real plant: Lab 11 PIDs, energy floor, filtered PID
- (b) Spinelli 2024 [21]: PPO, real handler, slew 0.46° vs PI 12.38°
- (b) Kurinov 2020 [22]: PPO-CMA, simulation, 67–97 % bucket mass
- (b) Hu 2026 [23]: simulated 21 t, 20–22 % less energy, relative
- (c) DMC [15] / D4RL [14]: code sanity check, not task

## 10. Proposed benchmark protocol
- Same real plant, Lab 12 training distribution, unseen set
- Same seeds 0/1/2 and step budget per algorithm
- Metrics: gap to floor, reach time, miss %, kJ
- Run: SAC, TD3, TQC, PPO; Lab 11 PID, filtered PID
- Report IQM with confidence intervals (rliable) [16, 17]
- Box: 100 unseen scenarios, 18 conditions × 4 variants; optional TD3+BC / IQL

## 11. Lab 12: SAC as a gain scheduler on the real plant
- Action scales PID gains: g = 4^a, range 0.25–4
- Zero-initialised actor: starts exactly as balanced PID
- 29 measured inputs, 0.4 s history; no true state
- Reward: error, time outside 20 cm, work above floor
- SAC, w_E 0.06 / 0.12, seeds 0–2, 300k steps
- Goal box (vs balanced PID + fix 2b, 100 unseen, 72 rows): gap to floor −25 % relative; move time ≤ +0.1 s; no row > 5 % miss; no row > 1 point worse; nominal row ≤ +1 kJ
- Status: training now · results pending
- Questions for you: PPO baseline? offline RL scope? energy metric?

## 12. References
- [1]–[24], each with DOI or URL (see sources.md)
