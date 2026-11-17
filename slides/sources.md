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
| 9 notes | T_w = 0.025 s; gap 23.8 % vs 24.0 % | Lab 12a filter choice, tuning set | `rl/12_filter_choice.json` |
| 5, 10 | w = 0.03 | Energy weight of the training reward, same for every learner | `rl/13_benchmark.md`, header |
| 5 | 100k steps; 1.01 % and 24.34 % | Misses over 20 cm of SAC and PPO at the same budget | `rl/13_benchmark.md`, 13c table |
| 5 notes | 25.86 % | Misses over 20 cm of TD3 | `rl/13_benchmark.md`, 13c table |
| 5 notes | u = clip(u_balanced + a, −1, 1); [θ, ω, e, ∫e dt]; −(e² + 0.01 ω² + w P / P_REF) | Residual action, observation and reward as run in Lab 12 and Lab 13 | `rl/boom_env_real_robust.py` (docstring) |
| 5 notes | 100,000 steps | Training budget of the Lab 12 SAC models | `README.md` (Limitations, "Training budget"); `rl/12_robust_train.py` |
| 10 | seeds 0/1/2; 100k steps; 1M transitions; 10/20/30 cm | Seeds, online budget, offline dataset, miss thresholds | `rl/13_benchmark.md` ("Setup and changes", 13c table); `rl/13_common.py` (`SEEDS`) |
| 10 box | 100 unseen scenarios; 18 conditions × 4 variants; noise 0.2° / 0.5° / 1.0° | The 72 evaluation rows of the Lab 12e protocol | `rl/12_eval.py` (`ROWS72`), `rl/10_sched_real.py` (`GROUPS`, `VARIANTS`); noise levels also in `rl/13_benchmark.md`, 13d table |
| 10 notes | 100,352 steps; 1M steps; noise 0.1; 100k and 300k transitions; 7200 moves | PPO stop point, extra PPO row, dataset noise, dataset-size study, moves per controller and seed | `rl/13_benchmark.md` ("Setup and changes", "Offline dataset", header) |
| 10 notes | "no dataset episode starts from an unseen scenario" | Assertion in the dataset script | `rl/13_dataset.py` (`check_unseen`) |
| 10 notes | 300,000 steps, gain-scheduler action | The earlier proposal that Lab 13 did not follow | `docs/12_sac_real_plant.md` section 2.6 |
| 6–8 pill | 100 unseen scenarios × 72 rows | Lab 12e protocol, 7200 moves per controller and seed | `rl/13_benchmark.md`, header |
| 6 table, 7, 11 | 0.00 % / 2.1 cm / 159.1 kJ / 20.0 % | Economical PID: misses over 20 cm, mean error, E_mech, energy above the floor | `rl/13_benchmark.md`, 13c table |
| 6 table, 11 | 0.00 % / 1.4 cm / 167.5 kJ / 26.4 % | Balanced PID, same four columns | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 24.34 % / 14.5 cm / 318.6 kJ / 140.4 % | PPO 100k (stops at 100,352 steps) | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 7.42 % / 7.8 cm / 216.2 kJ / 63.1 % | PPO 1M, not equal budget | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 1.01 % / 5.7 cm / 191.5 kJ / 44.5 % | SAC 100k (Lab 12 models, row taken from `rl/12e_moves.csv.gz`) | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 25.86 % / 13.5 cm / 184.3 kJ / 39.0 % | TD3 100k | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 0.00 % / 2.2 cm / 162.6 kJ / 22.6 % | BC, 1M transitions | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 10.94 % / 8.0 cm / 190.8 kJ / 43.9 % | CQL, 1M transitions | `rl/13_benchmark.md`, 13c table |
| 6 table, 7 | 0.02 % / 2.4 cm / 164.1 kJ / 23.8 % | IQL, 1M transitions | `rl/13_benchmark.md`, 13c table |
| 6 | +1.13 kJ, p = 6.3e-208 | BC against the economical PID: paired Wilcoxon, median difference of E_mech over 7200 moves | `rl/13_benchmark.md`, 13c table |
| 6 notes | w = 0.03, 3 seeds, 7200 moves | Setup of every learner | `rl/13_benchmark.md`, header and "Setup and changes" |
| 6 notes | "beats" rule | No more misses over 20 cm, lower E_mech with p < 0.05, no significantly larger error (ASSUMED) | `rl/13_benchmark.md`, 13c text |
| 7 | 100k steps; 10× the steps; 1M PID transitions | Online budget, PPO 1M row, offline dataset | `rl/13_benchmark.md`, "Setup and changes", "Offline dataset" |
| 7 notes | ± 16.43 | TD3: spread of misses over 20 cm between the 3 seeds | `rl/13_benchmark.md`, 13c table |
| 7 notes | noise 0.1 | Gaussian noise on the PID command in the dataset | `rl/13_benchmark.md`, "Offline dataset" |
| 7 notes | 3.5 / 2.8 / 2.2 cm; 161.7 / 162.7 / 162.6 kJ | BC at 100k / 300k / 1M transitions | `rl/13_benchmark.md`, "Dataset size (offline)" |
| 7 notes | action noise 0.1; observation and reward normalisation | Changes from the library defaults (TD3; PPO) | `rl/13_benchmark.md`, "Setup and changes" |
| 8 | 4 to 32 times; 2.23 to 15.96 against 0.50 (1/s) | Mean abs(du/dt) of the online learners against the economical PID | `rl/13_benchmark.md`, 13d table and reading |
| 8 | 55 to 70 % | Oscillation share of the energy above the floor, online learners | `rl/13_benchmark.md`, 13d table and reading |
| 8 | 5.5 to 14.2 cm; 1.4 cm | Mean error without sensor noise: online learners; economical PID | `rl/13_benchmark.md`, 13d table and reading |
| 8 | 0.55 | Mean abs(du/dt) of BC and of IQL | `rl/13_benchmark.md`, 13d table |
| 8 | 0.76; 56 % | CQL: mean abs(du/dt); overshoot share of its energy above the floor | `rl/13_benchmark.md`, 13d table |
| 8 notes | 0.64; 2.81; 4.75 | Mean abs(du/dt): balanced PID, SAC, PPO 1M | `rl/13_benchmark.md`, 13d table |
| 8 notes | 51.7 to 186.1 kJ against 26.6 kJ | Energy above the floor: online learners, economical PID | `rl/13_benchmark.md`, 13d table and reading |
| 8 notes | 0.8 cm; 7.6 cm | Largest change of the online learners' error from 0 to 1° noise; CQL error without noise | `rl/13_benchmark.md`, 13d reading and table |
| 11 | w = 0.03 | Energy weight of the training reward | `rl/13_benchmark.md`, header |
| 11 | −9.78 against −10.30 | Return per episode, balanced against economical PID, no noise, 300 training-distribution episodes | `rl/13_benchmark.md`, 13d reading ("Reward"); measured in `docs/13_benchmark.md` section 3 |
| 11 notes | −10.22 against −10.41 | Mean return in the 1M dataset (with noise), balanced against economical behaviour | `rl/13_benchmark.md`, "Offline dataset" and 13d reading; `rl/13_dataset.csv` |
| 11 notes | 39.0 to 140.4 % | Energy above the floor of the online learners | `rl/13_benchmark.md`, 13c table |
| 11 notes | −(e² + 0.01 ω² + w P / P_REF) | Training reward | `rl/boom_env_real_robust.py` (docstring and `step`) |
| 11 notes | "next step after the paper" | Learned gain scheduler | `README.md`, Roadmap |
| 6 figure | Fig. 1 | Energy against tip error, all controllers | `paper/figs/lab13/fig1_energy_vs_error.png` (reused, not regenerated) |
| 7 figure | Fig. 2 | Energy above the floor, grouped bars | `paper/figs/lab13/fig2_gap_to_floor.png` (reused) |
| 8 figure | Fig. 4 | Command activity against oscillation energy | `paper/figs/lab13/fig4_smoothness_vs_oscillation.png` (reused) |
| 11 figure | Fig. 5 | Reward return against energy above the floor | `paper/figs/lab13/fig5_reward_mismatch.png` (reused) |
| 11 box, notes | "under either reward"; "SAC and TD3 exploited the corrected one" | Question (a), framing after Lab 14 | `rl/14_reward_fix.md`, Statement |
| 11 box, notes | hard accuracy constraint, "cannot be paid for by missing" | Question (b), Lab 15 constrained RL or future work: the case Lab 14 lists as not tested | `rl/14_reward_fix.md`, Statement, last bullet |
| 12 pill | 100 unseen scenarios × 72 rows | Lab 13 protocol used by Lab 14 | `rl/14_reward_fix.md`, header |
| 12 table | 0.00 / 2.1 / 159.1 / −4.699 | Economical PID: misses over 20 cm %, mean error cm, E_mech kJ, corrected return | `rl/14_reward_fix.md`, 14e table and "Return on the training objective" table |
| 12 table | 48.06 / 43.0 / 136.0 / −4.339 | SAC 100k, corrected reward | same two tables |
| 12 table | 48.75 / 40.6 / 134.8 / −4.213 | TD3 100k, corrected reward | same two tables |
| 12 table | 0.47 / 3.1 / 159.1 / −4.632 | IQL 1M, corrected reward | same two tables |
| 12 table | 10.62 / 8.8 / 150.5 / −4.584 | CQL 1M, corrected reward | same two tables |
| 12 table | 0.09 / 4.4 / 157.6 / −4.724 | BC, economical-PID data only (500k) | same two tables |
| 12 footnote | 20 tuning scenarios × 4 variants, no noise; 3 seeds | Where the return column is measured | `rl/14_reward_fix.md`, "Return on the training objective" |
| 12 bullet | 48 % misses | SAC 48.06 %, TD3 48.75 %, rounded | `rl/14_reward_fix.md`, 14e table |
| 12 box | 0.021 rad/s; 0.01 rad/s | Mean abs(omega) over the last 2 s of SAC and TD3; stall speed of the reward | `rl/14_reward_fix.md`, "Why SAC and TD3 score higher…" bullet 2 |
| 12 box | 79 % and 80 %; 37 %; 5 cm | Steps outside the tolerance while moving: SAC, TD3, economical PID; tolerance | same bullet (table: 78.9, 80.3, 36.9) |
| 12 box | +1.035; 0.271; 0.403 | SAC against the economical PID: gain on the energy term, loss on tracking and on the stall term | same section, bullet 1 |
| 12 notes | r = −(e² + 0.01 ω² + 20 P/P_REF + 17.5 h)/50; h; 0.7 %; 42 % | Corrected reward, stall term, margin and share of episodes in order | `rl/14_reward_fix.md`, 14a |
| 12 notes | 100,000 steps; 1M dataset; 500k transitions | Lab 14 budgets | `rl/14_reward_fix.md`, Setup |
| 12 notes | −1.5 kJ | BC on economical-PID data against the economical PID, E_mech | `rl/14_reward_fix.md`, Statement |
| 12 notes | 30.2 to 52.8 % | Misses of the learning-rate check (x0.3, x3, 1 seed) | `rl/14_reward_fix.md`, Statement |
| 13 pill | 7200 moves per run | Lab 12e protocol | `rl/13e_direction_split.md`, header |
| 13 chart labels | 45.06 / 46.70 / 47.64 / 47.14 / 45.42 / 45.52 / 46.74 / 48.28 / 47.76 / 48.62 % | Boom-down (lowering) share of E_mech: economical PID, balanced PID, SAC w=0.03 and w=0.1 (balanced base), BC 1M, IQL 1M, CQL 1M, TD3 100k, PPO 100k, PPO 1M | `rl/13e_direction_split.md`, "Per controller", boom-down % column |
| 13 chart bars | E_down and E_up per move, kJ | Mean over seeds of the total per run divided by 7200 moves (derived; equals the md totals / 7200) | `rl/13e_direction_split.csv` (`E_down_kJ`, `E_up_kJ`, `moves`) |
| 13 stat | 45.06–48.62 % | Lowering share, lowest (economical PID) to highest (PPO 1M), 16 controllers | `rl/13e_direction_split.md`, Spread |
| 13 stat | 42.67–45.41 % | W_neg (negative work) share, lowest (PPO 100k) to highest (TD3 100k) | `rl/13e_direction_split.md`, Spread |
| 13 stat | 3.56 pp | Spread of the lowering share between controllers | `rl/13e_direction_split.md`, Spread |
| 13 text, notes | 159.1 to 318.6 kJ | E_mech per move across controllers | `rl/13e_direction_split.md`, Spread |
| 13 notes | 2.74 pp; 1.78 pp; 352,800 moves, 0.0 kJ | W_neg spread; widest seed range (TD3 100k); validation against the published per-move files | `rl/13e_direction_split.md`, Spread and Checks |

The title "~45 % braking work" rounds the W_neg share (42.67–45.41 %). Lowering share and negative work are two separate
splits in `rl/13e_direction_split.md`; the negative work done while lowering (their overlap) was not logged, so the
slide quotes no number for it.

Slides 12 (Lab 14) and 13 (energy split) were added and the slide 11 questions updated on 2026-11-17; References moved
to slide 14.

Slides 6, 7, 8 and 11 were rewritten on 2026-11-15 with the Lab 13 results. Slides 5, 9 and 10 were brought in line
with what Lab 13 did on the same day. Every number in their text
and notes is in `rl/13_benchmark.md`, except the two rows above that name another file.

Numbers that appear only inside the four figures and are not in `rl/13_benchmark.md`
(full list per figure: `paper/figs/lab13/README.md`):

| Slide | Number(s) | Meaning | Source file |
|---|---|---|---|
| 6, 7 figures | fast PID point; 36.0 % | Fast PID: mean error, E_mech, energy above the floor | `rl/12_eval.csv` (72 rows of `new100`, `controller = fast`) |
| 6 figure | 132.5 kJ | Mean energy floor (dashed line) | `rl/12_eval.csv`, `mean_floor_kJ` |
| 8 figure | y axis | Oscillation energy = `excess_kJ` × `oscillation_share` / 100 | `rl/13_benchmark.csv` |
| 11 figure | hollow markers, x axis | Training return of SAC, TD3 and PPO (exploration on) | `paper/figs/lab13/train_returns.csv`, read from the TensorBoard logs in `rl/tb/` |

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

- Egli, Hutter, "Towards RL-Based Hydraulic Excavator Automation", IROS 2020, and "A General Approach for the Automation of Hydraulic Excavator Arms Using Reinforcement Learning", IEEE RA-L 7(2):5679–5686, 2022. Found in search results only; the ETH, dblp and PDF pages could not be opened, so no DOI or numbers were verified. Mentioned by name only in the slide 9 and 14 speaker notes, marked [uncertain].
- MPC is named on slide 7 only as a method that was not tested; no citation.
