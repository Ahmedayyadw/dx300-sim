# Lab 13e: boom-up / boom-down split of E_mech

Source: `rl/12e_moves_dir.csv.gz`, `rl/13_moves_dir.csv.gz` (`--dir` runs of `rl/12e_stats_eval.py` and `rl/13_eval.py`, Lab 12e protocol: 100 unseen scenarios x 72 rows = 7,200 moves per controller and seed). Script: `rl/13e_direction_split.py`.

- E_mech = sum |tau_cyl omega| h, delivered mechanical energy only; no hydraulic correction.
- E_up / E_down: the same term where omega > 0 / omega < 0. W_pos / W_neg: where tau_cyl omega >= 0 / < 0 (motoring / braking). W_neg is not boom-down energy.
- Shares are sums over all moves (sum E_down / sum E_mech), per controller and seed; never the mean of per-move ratios. Per controller: mean over seeds, range = min-max over seeds.
- All values are simulation results: **ESTIMATED**.

## Checks

- Move set: 49 controller-seed runs (both files), each the same 7200 moves, no duplicates.
- E_mech per move vs the published `rl/12e_moves.csv.gz` and `rl/13_moves.csv.gz`: 352800 moves, largest difference 0.0 kJ.
- Controllers in both files (economical, balanced, res_bal_w0.03): 36000 moves, all five energy columns, largest difference 0.0 kJ.
- E_up + E_down = E_mech and W_pos + W_neg = E_mech for every controller and seed (rel 1e-9).

## Per controller

| controller | seeds x moves | E_mech kJ per move | E_mech kJ total | E_up kJ total | E_down kJ total | boom-down % | boom-down % range | W_neg % | W_neg % range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| economical PID | 1 x 7200 | 159.11 | 1,145,591 | 629,410 | 516,181 | 45.06 | - | 44.82 | - |
| balanced PID | 1 x 7200 | 167.54 | 1,206,267 | 642,975 | 563,292 | 46.70 | - | 44.95 | - |
| SAC w=0.03, balanced base | 3 x 7200 | 191.46 | 1,378,487 | 721,829 | 656,658 | 47.64 | 47.31-48.30 | 44.34 | 44.01-44.85 |
| SAC w=0.03, economical base | 3 x 7200 | 184.38 | 1,327,544 | 695,689 | 631,855 | 47.59 | 46.64-48.35 | 44.95 | 44.30-45.64 |
| SAC w=0.1, balanced base | 3 x 7200 | 186.08 | 1,339,756 | 708,241 | 631,515 | 47.14 | 46.91-47.57 | 44.08 | 43.79-44.49 |
| SAC w=0.1, economical base | 3 x 7200 | 181.19 | 1,304,567 | 704,456 | 600,111 | 46.01 | 45.72-46.55 | 44.09 | 43.54-44.71 |
| BC 100k | 3 x 7200 | 161.72 | 1,164,350 | 632,386 | 531,964 | 45.69 | 45.37-46.04 | 45.18 | 44.80-45.56 |
| BC 300k | 3 x 7200 | 162.73 | 1,171,672 | 638,227 | 533,445 | 45.53 | 45.33-45.92 | 44.90 | 44.57-45.35 |
| BC 1M | 3 x 7200 | 162.56 | 1,170,426 | 638,795 | 531,631 | 45.42 | 45.10-45.81 | 44.79 | 44.57-45.15 |
| IQL 100k | 3 x 7200 | 162.36 | 1,168,970 | 636,768 | 532,202 | 45.53 | 45.40-45.71 | 44.87 | 44.70-44.96 |
| IQL 300k | 3 x 7200 | 163.65 | 1,178,301 | 641,046 | 537,255 | 45.60 | 45.49-45.68 | 44.73 | 44.70-44.74 |
| IQL 1M | 3 x 7200 | 164.14 | 1,181,805 | 643,892 | 537,913 | 45.52 | 45.29-45.64 | 44.68 | 44.51-44.86 |
| CQL 1M | 3 x 7200 | 190.77 | 1,373,552 | 731,083 | 642,469 | 46.74 | 46.44-47.18 | 45.02 | 44.51-45.68 |
| TD3 100k | 3 x 7200 | 184.25 | 1,326,609 | 686,049 | 640,560 | 48.28 | 47.55-49.33 | 45.41 | 44.90-46.02 |
| PPO 100k | 3 x 7200 | 318.60 | 2,293,937 | 1,198,689 | 1,095,248 | 47.76 | 47.35-48.36 | 42.67 | 42.33-43.18 |
| PPO 1M | 3 x 7200 | 216.21 | 1,556,722 | 799,674 | 757,048 | 48.62 | 48.04-49.72 | 44.70 | 44.20-45.58 |

Totals are per run (mean over seeds). Per seed: `rl/13e_direction_split.csv`.

## Spread

- Boom-down share: highest PPO 1M 48.62 %, lowest economical PID 45.06 %, spread 3.56 percentage points.
- W_neg share: highest TD3 100k 45.41 %, lowest PPO 100k 42.67 %, spread 2.74 percentage points.
- Widest seed range of the boom-down share: TD3 100k, 1.78 percentage points.
- E_mech per move across controllers: 159.1 to 318.6 kJ.
