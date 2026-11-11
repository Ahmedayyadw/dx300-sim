# Lab 09f: Kp retune of the sched controller

No RL and no training. Controller: `pid_gain_sched` of Lab 09c3 (payload-aware PID with gain scheduling, T_f = 0.2 s), Ki = 0 and Kd = 0.25 as before, only Kp changed. All numbers are from `rl/09f_kp_retune.csv`. One Euler step per 0.05 s, 10 s per move. Miss % = share of moves with final tip error over 20 cm.
E_pump uses dp_valve = 20 bar and eta_pump = 0.85, both ASSUMED - not from manual.

## Goal

**The goal is met** with Kp = 0.55 (tuned on the 20 original scenarios, checked on the 100 unseen scenarios, against sched with Kp = 0.5, per row).

1. **Combined rows, miss rate <= 3 %: met.** 18 of 18 rows pass. Range 1-3% (Kp=0.5: 1-10%). Worst row: all_n1.0_J0.38_b0.5 baseline, 3% (Kp=0.5: 7%).
2. **Combined rows, E_mech <= 1.10 x sched and < 100 kJ: met.** 18 of 18 rows within the ratio, 18 of 18 under 100 kJ. E_mech 67.3-89.1 kJ (Kp=0.5: 66.2-85.4 kJ), ratio 1.013-1.046. Highest ratio: all_n1.0_J1.5_b2.0 baseline, 1.046. Highest E_mech: all_n1.0_J1.5_b2.0 payload_x1.2, 89.1 kJ.
3. **Single rows, no loss (miss <= sched, E_mech <= 1.10 x sched): met.** Miss above Kp=0.5 on 0 of 20 rows. Miss range 0-2% (Kp=0.5: 0-6%). E_mech above the ratio on 0 of 20 rows. Ratio 1.013-1.045.

## Tuning on the 20 original scenarios

Scenarios of RNG seed 7 (Labs 03-07), the 19 conditions of the 09c3 tables x 2 variants = 38 rows. Rule (Lab 07 criterion on these rows): lowest mean E_mech among the Kp with no move over 20 cm; if every Kp has a miss, fewest misses, then lowest mean E_mech. Kp without a miss: 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1. **Tuned Kp = 0.55.**

| Kp | moves over 20 cm (of 760) | worst error cm | mean E_mech kJ | |
|---:|---:|---:|---:|---|
| 0.50 | 26 | 24.5 | 76.3 | Lab 07 / 09c3 |
| 0.55 | 0 | 17.8 | 78.0 | **tuned** |
| 0.60 | 0 | 13.1 | 79.6 |  |
| 0.65 | 0 | 12.0 | 81.2 |  |
| 0.70 | 0 | 11.5 | 82.8 |  |
| 0.75 | 0 | 11.0 | 84.4 |  |
| 0.80 | 0 | 10.5 | 86.0 |  |
| 0.85 | 0 | 10.0 | 87.7 |  |
| 0.90 | 0 | 9.4 | 89.3 |  |
| 0.95 | 0 | 9.0 | 90.9 |  |
| 1.00 | 0 | 8.5 | 92.6 |  |

Limits of this tuning:
- Before this lab, Kp = 0.6, 0.7 and 0.9 were tried on the 100 unseen scenarios in a scratch check (Lab 09e diagnosis). The rule above does not read those scenarios, but the choice of grid and the decision to retune Kp came after seeing them.
- The tuning conditions are the evaluation conditions (same noise levels, J and b factors, tau_d, payload step); only the moves differ. This is not a test on unseen plant parameters.
- Kd and the filter time were not retuned.

## Kp sweep on the 100 unseen scenarios

Not used for the tuning. Goal columns as above, each Kp against Kp = 0.5.

| Kp | combined: miss % | rows <= 3 % | E_mech kJ | ratio to Kp=0.5 | rows within ratio and < 100 kJ | single: miss % | rows with miss above Kp=0.5 | ratio to Kp=0.5 | rows within ratio | goal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 1-10 | 6/18 | 66.2-85.4 | 1.000-1.000 | 18/18 | 0-6 | 0/20 | 1.000-1.000 | 20/20 | reference |
| 0.55 (tuned) | 1-3 | 18/18 | 67.3-89.1 | 1.013-1.046 | 18/18 | 0-2 | 0/20 | 1.013-1.045 | 20/20 | met |
| 0.60 | 0-2 | 18/18 | 68.3-92.6 | 1.025-1.089 | 18/18 | 0-1 | 0/20 | 1.026-1.086 | 20/20 | met |
| 0.65 | 0-2 | 18/18 | 69.1-95.8 | 1.036-1.129 | 12/18 | 0-1 | 0/20 | 1.037-1.125 | 18/20 | NOT met |
| 0.70 | 0-1 | 18/18 | 70.0-98.9 | 1.047-1.167 | 12/18 | 0-0 | 0/20 | 1.049-1.161 | 15/20 | NOT met |
| 0.75 | 0-0 | 18/18 | 70.8-101.9 | 1.057-1.203 | 9/18 | 0-0 | 0/20 | 1.061-1.195 | 4/20 | NOT met |
| 0.80 | 0-0 | 18/18 | 71.6-104.7 | 1.068-1.237 | 4/18 | 0-0 | 0/20 | 1.074-1.227 | 2/20 | NOT met |
| 0.85 | 0-0 | 18/18 | 72.4-107.4 | 1.079-1.269 | 2/18 | 0-0 | 0/20 | 1.087-1.257 | 1/20 | NOT met |
| 0.90 | 0-0 | 18/18 | 73.2-109.9 | 1.091-1.300 | 1/18 | 0-0 | 0/20 | 1.101-1.286 | 0/20 | NOT met |
| 0.95 | 0-0 | 18/18 | 74.0-112.4 | 1.104-1.330 | 0/18 | 0-0 | 0/20 | 1.115-1.314 | 0/20 | NOT met |
| 1.00 | 0-0 | 18/18 | 74.9-114.8 | 1.116-1.358 | 0/18 | 0-0 | 0/20 | 1.129-1.341 | 0/20 | NOT met |

## Comparison on every 09c3 row

100 unseen scenarios (RNG seed 123). The 09e columns are read from `rl/09e_comparison.csv` (mean ± std over three seeds).

| condition | variant | miss %: sched Kp=0.5 | sched Kp=0.55 | sched + residual (09e) | SAC 08b w=0.3 (09e) | E_mech kJ: sched Kp=0.5 | sched Kp=0.55 | sched + residual (09e) | SAC 08b w=0.3 (09e) | E_pump kJ: sched Kp=0.5 | sched Kp=0.55 | sched + residual (09e) | SAC 08b w=0.3 (09e) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | baseline | 1 | 0 | 24.0 ± 10.6 | 1.7 ± 1.2 | 68.5 | 70.1 | 73.4 ± 1.1 | 101.5 ± 0.5 | 80.3 | 83.0 | 93.3 ± 1.0 | 133.5 ± 0.7 |
| none | payload_x1.2 | 1 | 0 | 23.3 ± 18.9 | 3.0 ± 3.6 | 70.8 | 72.2 | 76.1 ± 2.3 | 103.5 ± 0.6 | 81.7 | 84.3 | 95.4 ± 1.7 | 133.7 ± 0.8 |
| noise_0.2 | baseline | 1 | 0 | 24.0 ± 12.7 | 1.3 ± 1.2 | 68.9 | 70.5 | 74.0 ± 1.1 | 107.6 ± 0.7 | 80.6 | 83.4 | 93.8 ± 1.0 | 139.3 ± 1.3 |
| noise_0.2 | payload_x1.2 | 1 | 0 | 25.3 ± 20.9 | 3.0 ± 3.6 | 71.1 | 72.6 | 76.6 ± 2.3 | 109.3 ± 0.6 | 82.0 | 84.6 | 95.9 ± 1.7 | 139.1 ± 1.2 |
| noise_0.5 | baseline | 3 | 1 | 24.3 ± 13.2 | 1.3 ± 1.2 | 70.2 | 71.9 | 76.1 ± 1.0 | 118.1 ± 1.1 | 81.9 | 84.8 | 95.8 ± 0.9 | 150.8 ± 2.0 |
| noise_0.5 | payload_x1.2 | 2 | 1 | 27.0 ± 19.6 | 3.0 ± 3.6 | 72.6 | 74.2 | 78.7 ± 2.0 | 120.1 ± 0.7 | 83.3 | 86.1 | 97.8 ± 1.5 | 150.5 ± 1.5 |
| noise_1.0 | baseline | 5 | 2 | 27.7 ± 15.1 | 1.7 ± 1.2 | 73.7 | 75.7 | 81.1 ± 0.7 | 135.1 ± 1.4 | 85.7 | 88.8 | 100.8 ± 1.0 | 172.2 ± 2.7 |
| noise_1.0 | payload_x1.2 | 6 | 2 | 27.7 ± 15.4 | 4.7 ± 4.5 | 76.5 | 78.4 | 83.9 ± 1.5 | 137.7 ± 0.6 | 87.3 | 90.4 | 102.6 ± 1.4 | 171.7 ± 1.8 |
| tau_noise_5pct | baseline | 1 | 0 | 24.0 ± 12.0 | 1.7 ± 1.2 | 68.5 | 70.1 | 73.6 ± 1.2 | 101.5 ± 0.5 | 80.3 | 83.0 | 93.5 ± 1.0 | 133.5 ± 0.7 |
| tau_noise_5pct | payload_x1.2 | 1 | 0 | 24.3 ± 19.5 | 3.0 ± 3.6 | 70.8 | 72.2 | 76.2 ± 2.5 | 103.5 ± 0.6 | 81.7 | 84.3 | 95.6 ± 1.8 | 133.7 ± 0.8 |
| J_x0.38 | baseline | 1 | 0 | 25.0 ± 10.0 | 18.0 ± 10.7 | 68.3 | 69.7 | 73.3 ± 0.8 | 907.6 ± 138.6 | 80.0 | 82.6 | 93.4 ± 0.5 | 1228.8 ± 179.8 |
| J_x0.38 | payload_x1.2 | 1 | 0 | 24.0 ± 20.0 | 19.0 ± 9.4 | 70.5 | 71.8 | 76.6 ± 2.6 | 864.2 ± 107.3 | 81.4 | 83.9 | 96.1 ± 1.7 | 1171.2 ± 141.1 |
| J_x1.5 | baseline | 1 | 0 | 25.0 ± 12.0 | 1.3 ± 1.2 | 69.1 | 70.8 | 74.4 ± 0.9 | 101.7 ± 0.5 | 80.9 | 83.8 | 94.1 ± 1.1 | 133.3 ± 0.9 |
| J_x1.5 | payload_x1.2 | 1 | 0 | 23.7 ± 19.4 | 2.7 ± 3.1 | 71.3 | 72.9 | 77.1 ± 2.0 | 103.7 ± 0.5 | 82.3 | 85.0 | 96.3 ± 1.3 | 133.6 ± 1.0 |
| b_x0.5 | baseline | 2 | 0 | 25.7 ± 12.3 | 1.7 ± 1.2 | 65.7 | 66.8 | 69.0 ± 0.8 | 93.9 ± 1.7 | 71.2 | 72.9 | 78.6 ± 1.0 | 120.9 ± 2.0 |
| b_x0.5 | payload_x1.2 | 4 | 0 | 26.0 ± 22.2 | 5.3 ± 6.8 | 68.3 | 69.2 | 71.8 ± 2.3 | 96.3 ± 1.6 | 72.8 | 74.5 | 80.8 ± 1.4 | 121.3 ± 1.9 |
| b_x2 | baseline | 0 | 0 | 29.0 ± 14.7 | 1.3 ± 1.2 | 80.3 | 83.9 | 85.2 ± 5.5 | 114.7 ± 1.1 | 101.8 | 106.9 | 112.5 ± 7.9 | 150.2 ± 1.4 |
| b_x2 | payload_x1.2 | 1 | 0 | 33.7 ± 21.7 | 1.7 ± 0.9 | 82.1 | 85.5 | 87.0 ± 6.0 | 116.0 ± 1.2 | 102.9 | 107.9 | 113.9 ± 8.6 | 149.8 ± 1.5 |
| tau_d | baseline | 2 | 0 | 19.3 ± 9.9 | 1.7 ± 1.7 | 68.3 | 69.8 | 72.1 ± 1.9 | 96.7 ± 0.6 | 79.9 | 82.6 | 91.5 ± 1.9 | 127.7 ± 1.0 |
| tau_d | payload_x1.2 | 4 | 0 | 27.3 ± 16.7 | 7.7 ± 5.7 | 70.5 | 71.9 | 74.2 ± 3.2 | 99.0 ± 0.6 | 81.3 | 83.9 | 92.8 ± 2.8 | 127.9 ± 1.0 |
| all_n0.2_J0.38_b0.5 | baseline | 5 | 1 | 21.3 ± 10.1 | 2.3 ± 2.1 | 66.2 | 67.3 | 69.8 ± 1.5 | 217.5 ± 25.8 | 71.6 | 73.3 | 79.1 ± 1.2 | 288.7 ± 39.1 |
| all_n0.2_J0.38_b0.5 | payload_x1.2 | 7 | 1 | 26.7 ± 15.0 | 15.3 ± 5.6 | 68.9 | 69.8 | 72.1 ± 2.6 | 227.4 ± 17.4 | 73.2 | 74.8 | 80.4 ± 2.0 | 292.0 ± 24.4 |
| all_n0.2_J1.5_b0.5 | baseline | 5 | 1 | 21.0 ± 11.0 | 1.7 ± 2.4 | 66.2 | 67.5 | 70.1 ± 1.8 | 96.4 ± 1.4 | 71.7 | 73.6 | 79.4 ± 1.7 | 120.9 ± 1.9 |
| all_n0.2_J1.5_b0.5 | payload_x1.2 | 7 | 1 | 27.0 ± 15.3 | 11.7 ± 6.0 | 68.9 | 69.9 | 72.4 ± 2.7 | 98.5 ± 1.2 | 73.3 | 75.1 | 80.7 ± 2.4 | 120.8 ± 1.9 |
| all_n0.2_J1.5_b2.0 | baseline | 1 | 1 | 24.3 ± 13.6 | 1.7 ± 1.2 | 80.4 | 84.0 | 82.6 ± 5.5 | 110.4 ± 0.8 | 101.8 | 106.9 | 109.1 ± 6.6 | 144.2 ± 1.0 |
| all_n0.2_J1.5_b2.0 | payload_x1.2 | 1 | 1 | 33.0 ± 19.8 | 8.0 ± 4.9 | 82.2 | 85.6 | 83.9 ± 6.2 | 111.3 ± 0.7 | 102.9 | 107.9 | 109.7 ± 7.6 | 143.3 ± 0.9 |
| all_n0.5_J0.38_b0.5 | baseline | 7 | 1 | 20.7 ± 10.8 | 2.7 ± 2.5 | 70.0 | 71.4 | 74.8 ± 0.9 | 227.1 ± 17.7 | 75.4 | 77.3 | 84.1 ± 1.0 | 298.8 ± 27.3 |
| all_n0.5_J0.38_b0.5 | payload_x1.2 | 5 | 2 | 27.3 ± 15.1 | 13.0 ± 6.4 | 73.1 | 74.3 | 77.8 ± 1.9 | 234.0 ± 10.6 | 77.2 | 79.2 | 85.9 ± 1.8 | 298.3 ± 15.6 |
| all_n0.5_J1.5_b0.5 | baseline | 6 | 1 | 21.7 ± 11.3 | 1.3 ± 1.9 | 67.6 | 69.0 | 72.1 ± 1.5 | 104.3 ± 1.7 | 73.0 | 75.0 | 81.3 ± 1.5 | 128.8 ± 2.3 |
| all_n0.5_J1.5_b0.5 | payload_x1.2 | 5 | 2 | 27.3 ± 15.1 | 12.3 ± 6.2 | 70.4 | 71.6 | 74.5 ± 2.3 | 106.3 ± 1.7 | 74.7 | 76.6 | 82.5 ± 2.1 | 128.4 ± 2.4 |
| all_n0.5_J1.5_b2.0 | baseline | 1 | 1 | 23.3 ± 13.9 | 2.0 ± 1.6 | 81.1 | 84.8 | 83.6 ± 5.6 | 115.7 ± 0.9 | 102.6 | 107.7 | 109.9 ± 6.8 | 149.9 ± 1.2 |
| all_n0.5_J1.5_b2.0 | payload_x1.2 | 2 | 1 | 33.3 ± 19.9 | 8.7 ± 5.7 | 83.0 | 86.5 | 84.8 ± 6.3 | 116.5 ± 0.8 | 103.7 | 108.7 | 110.3 ± 7.7 | 148.7 ± 1.0 |
| all_n1.0_J0.38_b0.5 | baseline | 7 | 3 | 23.0 ± 13.0 | 1.7 ± 1.2 | 79.4 | 81.2 | 86.4 ± 0.6 | 256.8 ± 7.6 | 85.5 | 87.9 | 96.2 ± 1.5 | 339.1 ± 11.7 |
| all_n1.0_J0.38_b0.5 | payload_x1.2 | 9 | 2 | 29.3 ± 14.0 | 15.0 ± 9.2 | 83.8 | 85.5 | 90.8 ± 1.7 | 261.3 ± 7.4 | 88.0 | 90.4 | 98.7 ± 2.4 | 335.4 ± 11.3 |
| all_n1.0_J1.5_b0.5 | baseline | 8 | 3 | 24.0 ± 11.6 | 2.3 ± 1.9 | 71.3 | 73.0 | 77.2 ± 0.9 | 116.8 ± 2.3 | 76.8 | 79.0 | 86.1 ± 1.3 | 142.8 ± 2.9 |
| all_n1.0_J1.5_b0.5 | payload_x1.2 | 10 | 2 | 29.0 ± 15.3 | 13.0 ± 7.9 | 74.5 | 76.0 | 79.8 ± 1.7 | 119.0 ± 2.4 | 78.6 | 80.8 | 87.4 ± 2.0 | 142.1 ± 3.2 |
| all_n1.0_J1.5_b2.0 | baseline | 3 | 1 | 28.0 ± 13.9 | 2.0 ± 1.6 | 83.2 | 87.0 | 85.9 ± 5.7 | 124.5 ± 1.0 | 104.9 | 110.1 | 112.0 ± 7.5 | 160.9 ± 1.5 |
| all_n1.0_J1.5_b2.0 | payload_x1.2 | 2 | 2 | 34.7 ± 19.4 | 10.3 ± 6.9 | 85.4 | 89.1 | 87.3 ± 6.5 | 125.5 ± 0.7 | 106.1 | 111.3 | 112.4 ± 8.4 | 159.4 ± 1.1 |

## J x0.38 rows with 10 substeps

Same rows with the dynamics integrated in 10 steps of 0.005 s per command (integrator check as README Lab 08c). Not part of the goal.

| condition | variant | miss %: Kp=0.5 | Kp=0.55 | E_mech kJ: Kp=0.5 | Kp=0.55 |
|---|---|---:|---:|---:|---:|
| J_x0.38_sub10 | baseline | 1 | 0 | 68.3 | 69.8 |
| J_x0.38_sub10 | payload_x1.2 | 1 | 0 | 70.6 | 71.8 |
| all_n0.2_J0.38_b0.5_sub10 | baseline | 4 | 1 | 66.1 | 67.1 |
| all_n0.2_J0.38_b0.5_sub10 | payload_x1.2 | 6 | 1 | 68.7 | 69.6 |
| all_n0.5_J0.38_b0.5_sub10 | baseline | 6 | 1 | 68.9 | 70.3 |
| all_n0.5_J0.38_b0.5_sub10 | payload_x1.2 | 5 | 2 | 71.9 | 73.1 |
| all_n1.0_J0.38_b0.5_sub10 | baseline | 8 | 2 | 76.3 | 78.0 |
| all_n1.0_J0.38_b0.5_sub10 | payload_x1.2 | 9 | 2 | 80.3 | 81.9 |
