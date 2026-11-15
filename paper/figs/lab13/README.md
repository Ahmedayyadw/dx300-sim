# Lab 13 figures

Made by `python rl/13_figures.py` (repo root, `.venv` active) from existing result files. No training,
no evaluation and no rollout was run for them. Each figure is a `.pdf` (fonts embedded, TrueType) and a
300 dpi `.png`, drawn 3.5 in wide for one IEEE column.

All plotted values are simulation results (**ESTIMATED**). Boom length 6.245 m is **OFFICIAL**, read from
`dx300sim/dx300_params.py`. Colours are the Okabe-Ito set; the marker shape repeats the group, so no
figure depends on colour alone.

Protocol behind every evaluation number: Lab 12e, 100 unseen scenarios x 72 rows = 7200 moves per
controller and seed. Learned controllers: residual on the balanced PID, w = 0.03, 3 seeds. "Seed spread"
is the standard deviation of the 3 per-seed results; PID points have no seeds and no bar.

## Source of every number

| Figure | What is plotted | Source file | Columns or lines |
|---|---|---|---|
| `fig1_energy_vs_error` | Mean tip error and mean E_mech with seed spread: balanced and economical PID, SAC, TD3, PPO 100k, PPO 1M, BC, CQL, IQL (offline at 1M transitions) | `rl/13_benchmark.csv` | `mean_cm_mean`, `mean_cm_std`, `E_mech_mean`, `E_mech_std` |
| | Fast PID point: mean over the 72 rows of `new100` | `rl/12_eval.csv` | `mean_deg` (x 6.245 m), `mean_E_mech_kJ`, rows `controller = fast` |
| | Energy floor line, 132.5 kJ: mean over the 72 rows | `rl/12_eval.csv` | `mean_floor_kJ` |
| `fig2_gap_to_floor` | Energy above the floor in % with seed spread, same nine controllers | `rl/13_benchmark.csv` | `gap_mean`, `gap_std`, `group` |
| | Fast PID, 36.0 %: 100 x (E_mech / floor - 1) of the two means above | `rl/12_eval.csv` | `mean_E_mech_kJ`, `mean_floor_kJ` |
| `fig3_dataset_size` | BC and IQL at 100k, 300k and 1M transitions: mean tip error and mean E_mech with seed spread | `rl/13_benchmark.csv` | rows `bc_*`, `iql_*`; `mean_cm_*`, `E_mech_*` |
| | Economical PID reference line | `rl/13_benchmark.csv` | row `economical` |
| `fig4_smoothness_vs_oscillation` | x: mean abs(du/dt) of the applied command | `rl/13_benchmark.csv` | `du_dt` |
| | y: oscillation energy per move = `excess_kJ` x `oscillation_share` / 100 | `rl/13_benchmark.csv` | `excess_kJ`, `oscillation_share` |
| `fig5_reward_mismatch` | y: energy above the floor with seed spread | `rl/13_benchmark.csv` | `gap_mean`, `gap_std` |
| | x, PID points: return per episode without noise, balanced -9.78, economical -10.30 | `docs/13_benchmark.md` | section 3, probe table, rows `bal 0.00` and `eco 0.00` |
| | x, SAC: training return | `rl/tb/12_residual_w0.03_s{0,1,2}_1` | TensorBoard scalar `rollout/ep_rew_mean` |
| | x, TD3, PPO 100k, PPO 1M: training return | `rl/tb/13_td3_100k_s*_2`, `rl/tb/13_ppo_100k_s*_1`, `rl/tb/13_ppo_1M_s*_2` | TensorBoard scalar `rollout/ep_rew_mean` |

`rl/13_benchmark.csv` takes the PID and SAC rows from `rl/12e_moves.csv.gz` and the other rows from
`rl/13_moves.csv.gz` (`python rl/13_eval.py`). The script asserts that the economical PID energy in
`rl/12_eval.csv` equals the one in `rl/13_benchmark.csv`, so the fast PID and the floor are on the same
protocol as the rest.

`rl/tb/` is gitignored. The training returns read from it are saved in `train_returns.csv` in this
folder (one row per seed, with the TensorBoard run used), and the script reads that file when `rl/tb/`
is missing. Training return of a seed = mean of `rollout/ep_rew_mean` over the last 10 % of the training
steps, from the run that reached the full budget.

## Limits of the figures

- **Fig. 4 has no fast PID.** The 13d columns (du/dt, energy split) were computed for the balanced and
  economical PID only. No seed spread: the 13d columns are pooled over seeds.
- **Fig. 5 mixes three kinds of return, and has no offline learners.**
  - PID points: deterministic controller, 300 training-distribution episodes, no noise. This pair is the
    evidence for the mismatch: the reward gives the balanced PID the higher return (-9.78 against -10.30)
    although it uses more energy (26.4 % against 20.0 % above the floor).
  - SAC, TD3, PPO (hollow markers): return logged during training, with exploration noise, on the training
    distribution. It is not the return of the deterministic policy that was evaluated, and it is noisy
    (seed spread 0.5 to 1.0). Read these points as a rough position only.
  - BC, CQL and IQL: no return exists in any result file. They are left out. So are the fast PID (no
    return measured) and the return of any controller on the 100 unseen scenarios.
  - The y axis is from the unseen evaluation scenarios (72 rows with disturbances); the x axis is from the
    training distribution. The two axes are not measured on the same episodes.
- **PPO 1M is not at equal budget** (10 x the steps of the other online learners).
