# Sources for slides/dx300_sac_methods.pptx

## Numbers and the repo file each one comes from

All real-plant numbers are model outputs (ESTIMATED), as their source files state.

| Slide | Number(s) | Meaning | Source file |
|---|---|---|---|
| 1, 2 | 6.245 m | Official DX300 boom length | `dx300sim/dx300_params.py` (`BOOM_LENGTH_OFFICIAL_M`) |
| 1 | 1° ≈ 10.9 cm | 6.245 m × π/180 | `README.md` (Stage 1 labs), `docs/acceptable_error.md` |
| 1 | 20 cm | Tip tolerance, design assumption | `docs/acceptable_error.md` |
| 2 notes | J = 8e4 kg·m², m = 5000 kg | Placeholder plant | `rl/boom_env.py` |
| 2 table | 5.552 vs 5.619 s, −1.19 % | Boom-up time, Python vs Simscape, no friction | `rl/10_real_plant.md` section 1 (from `rl/10_simscape_check.csv`) |
| 2 | −1.13 % | Boom-up time with friction | `rl/10_real_plant.md` section 1 |
| 2 table | 665.3 vs 672.3 kJ, −1.04 % | Boom-up signed work, no friction | `rl/10_real_plant.md` section 1 |
| 2 | −0.54 % | Boom-up signed work with friction | `rl/10_real_plant.md` section 1 |
| 2 table | within 1.90 % | Static holding torque, every 5° bin | `rl/10_real_plant.md` section 1 |
| 2 | ×2.21 (378.7 vs 171.7 kN·m) | Gravity torque at θ = 0, real vs placeholder | `rl/10_real_plant.md` (statement section) |
| 2 notes | 494 L/min, 330 bar | Pump flow, relief (OFFICIAL) | `rl/10_real_plant.md` (status of the inputs) |
| 3 table | Kp 5/Kd 2, 1.5/1, 0.8/0.8 | Three operating points | `rl/11_points.json`; `rl/11_real_plant_tuning.md` |
| 3 table | 1.18 / 1.45 / 1.97 s | Reach 20 cm, nominal, 100 unseen | `rl/11_real_plant_tuning.md`, "What is left", first table |
| 3 table | 171.0 / 162.8 / 155.5 kJ | E_mech, nominal, 100 unseen | same table |
| 3 | 154.9 kJ | Mean energy floor, nominal, 100 unseen | same table |
| 3 table | +10.4 / +5.1 / +0.3 % (16.1 / 7.8 / 0.5 kJ) | Gap to floor, nominal | same table and its "Reading" |
| 3 table | +15.4 / +13.7 / +10.7 % | Mean gap, rows without sensor noise, fix 2b | `rl/11_real_plant_tuning.md`, "What is left", Reading, bullet 2 |
| 3 table | +71.2 / +46.7 / +37.0 % | Mean gap, rows with 1.0° sensor noise, fix 2b | same bullet |
| 3 | 3406 → 1120 kg at 5 s | Dump variant | `rl/11_real_plant_tuning.md` section 1.3 |
| 3 | up to 80 % misses | Lab 10 tuned gains, all_n1.0_J0.38_b0.5 dump, 100 unseen | `rl/11_real_plant_tuning.md` section 3; `rl/10_real_plant.md` |
| 3 | 0 of 54 dump rows over 5 %; 0 of 162 other rows worse | Fix 2b (adaptive filter) on 100 unseen | `rl/11_real_plant_tuning.md` section 4.3 (verdict); `rl/11_dump_fix_choice.json` |
| 3 notes | T_f = 1 s, c = 1, n_hold = 2 | Fix 2b parameters | `rl/11_dump_fix_choice.json` |
| 3 notes | −18.8 / +0.4 / +2.7 kJ | Fix 2b energy change on other rows (fast/balanced/economical) | `rl/11_real_plant_tuning.md` section 4.3 |
| 4 table | 127.9 / 3.7 / 16.3 | SAC w=0.1 (placeholder): kJ, % > 20 cm, worst variant | `paper/table1.md` |
| 4 table | 100.8 / 1.3 / 66.7 | SAC w=0.3 (placeholder) | `paper/table1.md` |
| 4 table | 68.6 / 1.0 / 100.0 | PID + FF (placeholder) | `paper/table1.md` |
| 4 table | 97.7 / 1.0 / 99.0 | PID + I + FF (placeholder) | `paper/table1.md` |
| 4 table | 6.3, 3.0, 100.0, 37.0 | Payload ×1.2 % > 20 cm (SAC w=0.1, SAC w=0.3, PID, PID+I) | `rl/08b_eval_n10.csv`, rows `set=new100, variant=payload_x1.2`, mean of `share_over_20cm` over seeds (Table 1 aggregation); also `README.md` |
| 4 notes | 100 % → 1 %, 68.5 kJ | Payload-aware PID (Lab 09c) | `README.md` (Lab 09c row); `rl/09c_comparison.md` |
| 4 notes | 20.7–34.7 % vs 1–10 % | Residual SAC on sched vs sched alone, combined rows (09e) | `README.md` (Lab 09d/09e row); `rl/09e_comparison.md` |
| 4 figure | Fig 1 | Energy vs worst-variant miss share (placeholder plant) | `paper/figs/fig1_energy_vs_robustness.png` (reused, not regenerated) |
| 5, 6 notes | ~12 min per Stage 1 run | Training time | `README.md`; `rl/08b_train_log.csv` |
| 9, 11 notes | T_w = 0.025 s; gap 23.8 % vs 24.0 % | Lab 12a filter choice, tuning set | `rl/12_filter_choice.json` |
| 11 | g = 4^a, 0.25–4; Kp 0.375–6, Kd 0.25–4 | Lab 12 action range | `docs/12_sac_real_plant.md` section 2.1 |
| 11 | 29 inputs, 8 steps = 0.4 s | Observation | `docs/12_sac_real_plant.md` section 2.3 |
| 11 notes | 0.1, 0.05, 0.005, 5·min(d, 3)² | Reward terms | `docs/12_sac_real_plant.md` section 2.4 |
| 11 | w_E 0.06 / 0.12 | Energy weights | `docs/12_sac_real_plant.md` sections 2.4, 2.6 |
| 11 | seeds 0–2, 300k steps; 2 × 256; buffer 1,000,000 | SAC settings | `docs/12_sac_real_plant.md` section 2.6; `rl/12_sac_train.py` (`SEEDS`, `STEPS`) |
| 11 notes | J 0.38–1.5, friction 0.5–2, noise 0–1.0°, step 4–7 s | Training distribution | `docs/12_sac_real_plant.md` section 2.5 |
| 11 box | ≥ 25 % relative, +0.1 s, 5 %, 1 point, 1 kJ | Lab 12 goal | `docs/12_sac_real_plant.md` (goal, lines 5–7) and section 3 |
| 11 notes | dp_valve 20 bar, η_pump 0.85 (ASSUMED) | E_pump parameters | `rl/11_real_plant_tuning.md` (status of the values) |
| 11 | "Training now" | Status | stated by the user on 2026-11-12; `docs/12_sac_real_plant.md` line 3 still says "design, waiting for approval" |

Numbers from published papers (slide 9 and notes) are listed with their reference below.

## References (all opened and checked; arXiv DOIs are 10.48550/arXiv.<id>)

| # | Title | Authors | Year | DOI / URL |
|---|---|---|---|---|
| 1 | Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (ICML) | Haarnoja, Zhou, Abbeel, Levine | 2018 | https://arxiv.org/abs/1801.01290 |
| 2 | Soft Actor-Critic Algorithms and Applications | Haarnoja et al. | 2018 | https://arxiv.org/abs/1812.05905 |
| 3 | Trust Region Policy Optimization (ICML) | Schulman et al. | 2015 | https://arxiv.org/abs/1502.05477 |
| 4 | Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | https://arxiv.org/abs/1707.06347 |
| 5 | Asynchronous Methods for Deep Reinforcement Learning (ICML) | Mnih et al. | 2016 | https://arxiv.org/abs/1602.01783 |
| 6 | Continuous control with deep reinforcement learning (DDPG) | Lillicrap et al. | 2015 | https://arxiv.org/abs/1509.02971 |
| 7 | Addressing Function Approximation Error in Actor-Critic Methods (TD3, ICML) | Fujimoto, van Hoof, Meger | 2018 | https://arxiv.org/abs/1802.09477 |
| 8 | Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics (TQC) | Kuznetsov et al. | 2020 | https://arxiv.org/abs/2005.04269 |
| 9 | Conservative Q-Learning for Offline Reinforcement Learning | Kumar, Zhou, Tucker, Levine | 2020 | https://arxiv.org/abs/2006.04779 |
| 10 | Offline Reinforcement Learning with Implicit Q-Learning | Kostrikov, Nair, Levine | 2021 | https://arxiv.org/abs/2110.06169 |
| 11 | A Minimalist Approach to Offline Reinforcement Learning (TD3+BC, NeurIPS) | Fujimoto, Gu | 2021 | https://arxiv.org/abs/2106.06860 |
| 12 | Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems | Levine, Kumar, Tucker, Fu | 2020 | https://arxiv.org/abs/2005.01643 |
| 13 | When to Trust Your Model: Model-Based Policy Optimization (MBPO, NeurIPS) | Janner, Fu, Zhang, Levine | 2019 | https://arxiv.org/abs/1906.08253 |
| 14 | D4RL: Datasets for Deep Data-Driven Reinforcement Learning | Fu et al. | 2020 | https://arxiv.org/abs/2004.07219 |
| 15 | DeepMind Control Suite | Tassa et al. | 2018 | https://arxiv.org/abs/1801.00690 |
| 16 | Deep Reinforcement Learning that Matters (AAAI) | Henderson et al. | 2018 | https://arxiv.org/abs/1709.06560 |
| 17 | Deep Reinforcement Learning at the Edge of the Statistical Precipice (NeurIPS) | Agarwal et al. | 2021 | https://arxiv.org/abs/2108.13264 |
| 18 | Stable-Baselines3: Reliable Reinforcement Learning Implementations (JMLR 22(268)) | Raffin et al. | 2021 | https://jmlr.org/papers/v22/20-1364.html |
| 19 | Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO (ICLR) | Engstrom et al. | 2020 | https://arxiv.org/abs/2005.12729 |
| 20 | What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study | Andrychowicz et al. | 2020 | https://arxiv.org/abs/2006.05990 |
| 21 | Reinforcement Learning Control for Autonomous Hydraulic Material Handling Machines with Underactuated Tools (IROS, pp. 12694–12701) | Spinelli et al. | 2024 | doi:10.1109/IROS58592.2024.10802199; https://arxiv.org/abs/2410.05093 |
| 22 | Automated Excavator Based on Reinforcement Learning and Multibody System Dynamics (IEEE Access 8, 213998–214006) | Kurinov, Orzechowski, Hämäläinen, Mikkola | 2020 | doi:10.1109/ACCESS.2020.3040246; https://zenodo.org/record/5910322 |
| 23 | Bayesian reinforcement learning for adaptive control of energy recuperation in hydraulic excavator arms (Sci. Rep. 16:6195) | Hu, Wen, Zhang, Chen, Gong | 2026 | doi:10.1038/s41598-026-35391-y; https://pmc.ncbi.nlm.nih.gov/articles/PMC12905251/ |
| 24 | Autonomous Control of Redundant Hydraulic Manipulator Using Reinforcement Learning with Action Feedback (IROS 2022 per arXiv comment) | Dhakate et al. | 2022 (arXiv 2025) | https://arxiv.org/abs/2504.15714 |

### Published numbers quoted on slide 9

| Ref | Number | Where in the paper |
|---|---|---|
| 21 | PI 12.38°, RL 0.46° (no history), 1.43° (history), slew steady-state error | Table IV |
| 21 | RL 1.078 m vs driver 0.593 m, inexperienced driver 1.145 m (end-effector error) | Table V |
| 22 | 67–97 % of maximum bucket mass moved; about two days of training | Results and Conclusion sections |
| 23 | Energy per cycle about 20–22 % lower, precision about 10–20 % better, vs fixed-parameter baseline; simulated 21-ton excavator | Results and analysis |

## [uncertain] — not cited on any slide

- Egli, Hutter, "Towards RL-Based Hydraulic Excavator Automation", IROS 2020, and "A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning", IEEE RA-L 7(2):5679–5686, 2022. Found in search results only; the ETH, dblp and PDF pages could not be opened, so no DOI or numbers were verified. Mentioned by name only in the slide 9 and 12 speaker notes, marked [uncertain].
- LQR / MPC statements on slide 8 are standard control knowledge with no citation.
