# Lab 10: control study on the real plant

No RL and no training. Plant: `rl/boom_env_real.py` (Lab 10a), built from the Lab 08a Simscape export `matlab/lab08a/export/boom_params_lab08a.json`; design and the source of every value in `docs/10_real_plant.md`. Controller: `sched` (pid_gain_sched of Lab 09c3 / 09f) adapted to the plant (`rl/10_sched_real.py`). All result numbers are from `rl/10_real_plant.csv`, `rl/10_tuning_grid.csv` and `rl/10_simscape_check.csv`.

Status of the inputs: pump flow 494 L/min, relief 330 bar, bore, rod, boom mass and length are OFFICIAL. Pins, moment arm, torque limits, inertia, front point masses, friction and the payload density (1800 kg/m^3) are ESTIMATED (Lab 08a). The hard retraction speed limit, the substep count and the payload variants are ASSUMED. E_pump uses dp_valve = 20 bar and eta_pump = 0.85, both ASSUMED - not from manual; it is the energy of an ideal demand-following pump, a lower bound (Lab 09b2).

## Statement: placeholder plant against real plant

Placeholder numbers: `sched` with Kp = 0.55 on `rl/boom_env.py`, 100 unseen scenarios, from `rl/09f_kp_retune.csv`. Matched rows: the 18 conditions that exist on both plants, real `baseline` against placeholder `baseline` and real `fill_half` (tip mass 2263 -> 3406 kg, gravity torque x 1.23) against placeholder `payload_x1.2`. The placeholder combined cases include its `tau_d` flag; on the real plant the angle-dependent limit is always present.

- **Real plant, tuned gains**, against the placeholder plant on the 36 matched rows: miss rate 0-1 % (mean 0.1 %) against 0-3 % (mean 0.9 %); higher on 0 rows, lower on 22. E_mech 134.3-189.7 kJ (mean 151.4) against 66.8-89.1 kJ (mean 75.3), ratio 1.58-2.36 (mean 2.02). E_pump (lower bound, ASSUMED parameters) 106.5-146.6 kJ against 72.9-111.3 kJ, ratio 1.00-1.70 (mean 1.39).
- **Real plant, old gains**, against the placeholder plant on the 36 matched rows: miss rate 0-4 % (mean 0.8 %) against 0-3 % (mean 0.9 %); higher on 8 rows, lower on 12. E_mech 193.3-349.6 kJ (mean 254.4) against 66.8-89.1 kJ (mean 75.3), ratio 2.48-4.79 (mean 3.39). E_pump (lower bound, ASSUMED parameters) 152.5-260.6 kJ against 72.9-111.3 kJ, ratio 1.55-3.30 (mean 2.26).
- Without stress (`none`, `baseline`): E_mech 153.2 kJ tuned and 243.0 kJ old gains, against 70.1 kJ on the placeholder plant. Miss rate 0 % tuned, 0 % old gains, 0 % placeholder.
- Main cause of the higher energy: the real plant carries the front attachment. Gravity torque at theta = 0 is 378.7 kN m against 171.7 kN m on the placeholder plant (x 2.21), so the same move needs about that much more work.
- Tuned against old gains on the real plant: the Lab 07 criterion picks the lowest energy, and the tuned gains are slow. Mean time to come within 20 cm is 3.88-4.45 s tuned against 1.43-1.54 s old gains over the 72 rows. The old gains overshoot (the real plant has about 1/300 of the placeholder damping), which costs the extra energy.
- Rows over 5 % miss on the real plant (all 72 rows, including `fill` and `dump`, which have no placeholder counterpart). Tuned: noise_1.0 dump (11 %); all_n1.0_J0.38_b0.5 dump (80 %); all_n1.0_J1.5_b0.5 dump (10 %); all_n1.0_J1.5_b2.0 dump (10 %). Old gains: all_n1.0_J0.38_b0.5 dump (13 %).
- The energy figures of the two plants are not the same quantity in one respect: on the real plant E_mech uses the torque the cylinders deliver, which on the flow limit is below the commanded torque. The placeholder plant has no flow limit.

## 1. Check against Simscape (10b)

From `rl/10_simscape_check.py`. No parameter was tuned to make these match.

| check | quantity | Python | Simscape | diff | diff % |
|---|---|---:|---:|---:|---:|
| static | holding torque at -40 deg [kN m] | 212.0 | 208.3 | +3.7 | +1.77 |
| static | holding torque at -30 deg [kN m] | 267.9 | 272.6 | -4.6 | -1.69 |
| static | holding torque at -20 deg [kN m] | 312.4 | 314.1 | -1.7 | -0.54 |
| static | holding torque at -10 deg [kN m] | 351.9 | 345.3 | +6.6 | +1.90 |
| static | holding torque at +0 deg [kN m] | 379.2 | 382.5 | -3.3 | -0.86 |
| static | holding torque at +10 deg [kN m] | 394.0 | 398.1 | -4.1 | -1.03 |
| static | holding torque at +20 deg [kN m] | 397.8 | 401.7 | -3.9 | -0.98 |
| static | holding torque at +30 deg [kN m] | 388.9 | 391.9 | -3.0 | -0.76 |
| static | holding torque at +40 deg [kN m] | 368.8 | 365.0 | +3.8 | +1.04 |
| static | holding torque at +50 deg [kN m] | 338.0 | 339.9 | -1.8 | -0.54 |
| static | holding torque at theta_min (-47.2 deg), t = 0 [kN m] | 167.1 | 167.1 | +0.0 | +0.00 |
| static | holding torque at theta_max (+60.4 deg), t = 0 [kN m] | 293.9 | 293.9 | +0.0 | +0.00 |
| boom_up_time | time to theta_max - 0.2 deg, no friction [s] | 5.552 | 5.619 | -0.067 | -1.19 |
| boom_up_work | signed work, no friction [kJ] | 665.3 | 672.3 | -7.0 | -1.04 |
| boom_up_work | absolute work, no friction [kJ] | 665.3 | 679.8 | -14.5 | -2.13 |
| boom_up_time | time to theta_max - 0.2 deg, friction [s] | 5.553 | 5.617 | -0.064 | -1.13 |
| boom_up_work | signed work, friction [kJ] | 669.0 | 672.7 | -3.7 | -0.54 |
| boom_up_work | absolute work, friction [kJ] | 669.0 | 681.1 | -12.1 | -1.77 |

- Static holding torque: within 1.90 % of the Simscape value in every 5 deg bin. The Simscape figure is tau - J dw/dt from the logged boom-up of Lab 09b2, which oscillates (limit cycle), so the bins carry the noise of a differentiated signal. The two t = 0 rows are not an independent check: `build_stage4.m` sets the initial cap pressure from the same analytic formula.
- Boom-up time: 5.552 s against 5.619 s (-1.19 %) without friction, 5.553 s against 5.617 s (-1.13 %) with friction. The Python plant is rigid and sits on the flow limit; the Simscape oil column is compressible and the boom oscillates.
- Boom-up mechanical work, signed: 665.3 kJ against 672.3 kJ (-1.04 %) without friction, 669.0 against 672.7 kJ (-0.54 %) with friction. Absolute work is -2.13 % and -1.77 %: Simscape has negative-power phases in its oscillation, the Python plant has none on this move.

## 2. Tuning on the real plant

Kp x Kd over the Lab 07 grid, Ki = 0, nominal real plant (`none`, `baseline`), the 20 original scenarios (RNG seed 7). The unseen scenarios are not used. Cell: worst-case tip error / mean E_mech. Lab 07 criterion: lowest mean E_mech among the sets with worst-case error < 20 cm. 25 of 45 sets pass. **Tuned: Kp = 0.25, Ki = 0, Kd = 0.5** (worst case 5.6 cm, 162.6 kJ). Old gains for reference: Kp = 0.55, Ki = 0, Kd = 0.25 (Lab 09f). Gains are in command units per rad; one unit of command is 911.8 kN m at theta = 0 on the real plant and 500 kN m on the placeholder plant.

| Kp \ Kd | 0 | 0.25 | 0.5 | 1 | 2 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 252 cm / 579 kJ | 5 cm / 203 kJ | **6 cm / 163 kJ** | 58 cm / 150 kJ | 197 cm / 116 kJ |
| 0.5 | 144 cm / 615 kJ | 2 cm / 257 kJ | 2 cm / 174 kJ | 6 cm / 163 kJ | 56 cm / 151 kJ |
| 1 | 83 cm / 694 kJ | 1 cm / 311 kJ | 1 cm / 209 kJ | 1 cm / 165 kJ | 6 cm / 164 kJ |
| 2 | 75 cm / 803 kJ | 3 cm / 361 kJ | 0 cm / 251 kJ | 0 cm / 185 kJ | 1 cm / 170 kJ |
| 5 | 71 cm / 936 kJ | 5 cm / 450 kJ | 0 cm / 304 kJ | 0 cm / 224 kJ | 0 cm / 182 kJ |
| 10 | 65 cm / 1037 kJ | 30 cm / 744 kJ | 1 cm / 363 kJ | 0 cm / 255 kJ | 0 cm / 201 kJ |
| 20 | 93 cm / 1182 kJ | 61 cm / 1069 kJ | 21 cm / 752 kJ | 0 cm / 304 kJ | 0 cm / 230 kJ |
| 30 | 81 cm / 1260 kJ | 60 cm / 1183 kJ | 63 cm / 1104 kJ | 0 cm / 396 kJ | 0 cm / 252 kJ |
| 50 | 65 cm / 1322 kJ | 67 cm / 1280 kJ | 60 cm / 1232 kJ | 47 cm / 1135 kJ | 0 cm / 299 kJ |

## 3. Results on the real plant

100 unseen scenarios (RNG seed 123), 10 s per move, T_f = 0.2 s. **tuned** = Kp 0.25, Kd 0.5; **old** = Kp 0.55, Kd 0.25. Status: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok.

| rows | miss %: tuned | old | E_mech kJ: tuned | old | reach 20 cm, s: tuned | old | flow limit %: tuned | old | torque limit %: tuned | old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single conditions (36 rows) | 0-11 | 0-1 | 125.3-178.1 | 177.4-292.9 | 3.95-4.44 | 1.45-1.53 | 0.5-0.8 | 4.1-6.0 | 0.00 | 0.00-0.02 |
| combined cases (36 rows) | 0-80 | 0-13 | 127.0-193.1 | 179.8-349.6 | 3.88-4.45 | 1.43-1.54 | 0.4-1.1 | 4.2-6.8 | 0.00 | 0.00-0.03 |
| status over all 72 rows | 4 break, 0 degraded, 68 ok | 1 break, 10 degraded, 61 ok | | | | | | | | |

Columns: miss % = share of moves with final tip error over 20 cm. Reach 20 cm = mean time to first come within 20 cm, over the moves that do (a move that starts within 20 cm counts as 0 s); never reached = moves that never come within 20 cm. Flow limit % and torque limit % = share of the 0.005 s substeps with the rod speed on the pump flow limit, and with |u| = 1 while not on the flow limit. m_hat / m_tip = estimate over true tip mass in the last 2 s.
Payload variants (tip mass, step at t = 5 s): `baseline` 3406 kg; `fill_half` 2263 -> 3406 kg; `fill` 1120 -> 3406 kg; `dump` 3406 -> 1120 kg.
330 bar could not hold the flow limit in 0 substeps over all rows.

### 3.1 One condition at a time

| condition | variant | miss %: tuned | old | E_mech kJ: tuned | old | E_pump kJ (lower bound, ASSUMED parameters): tuned | old | reach 20 cm, s: tuned | old | never reached: tuned | old | flow limit %: tuned | old | torque limit %: tuned | old | m_hat / m_tip (tuned) | status: tuned / old |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| none | baseline | 0 | 0 | 153.2 | 243.0 | 120.2 | 185.1 | 4.11 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 1.000 ± 0.001 | ok / ok |
| none | fill_half | 0 | 0 | 135.6 | 213.4 | 107.4 | 167.5 | 4.24 | 1.47 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 1.000 ± 0.000 | ok / ok |
| none | fill | 0 | 0 | 128.4 | 197.9 | 103.1 | 159.5 | 4.16 | 1.47 | 0 | 0 | 0.7 | 4.4 | 0.0 | 0.0 | 1.000 ± 0.000 | ok / ok |
| none | dump | 0 | 0 | 166.5 | 257.0 | 132.9 | 197.7 | 4.40 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 0.994 ± 0.003 | ok / ok |
| noise_0.2 | baseline | 0 | 0 | 156.1 | 246.4 | 122.4 | 187.5 | 4.11 | 1.49 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 1.000 ± 0.035 | ok / ok |
| noise_0.2 | fill_half | 0 | 0 | 137.2 | 215.1 | 109.3 | 168.3 | 4.24 | 1.47 | 0 | 0 | 0.7 | 4.4 | 0.0 | 0.0 | 1.003 ± 0.035 | ok / ok |
| noise_0.2 | fill | 0 | 0 | 129.0 | 198.5 | 103.7 | 159.8 | 4.12 | 1.47 | 0 | 0 | 0.7 | 4.6 | 0.0 | 0.0 | 1.002 ± 0.036 | ok / ok |
| noise_0.2 | dump | 0 | 0 | 167.5 | 258.3 | 133.5 | 198.8 | 4.40 | 1.49 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 0.990 ± 0.114 | ok / ok |
| noise_0.5 | baseline | 0 | 0 | 161.9 | 268.9 | 126.5 | 203.4 | 4.12 | 1.49 | 0 | 0 | 0.7 | 5.2 | 0.0 | 0.0 | 1.001 ± 0.088 | ok / ok |
| noise_0.5 | fill_half | 0 | 0 | 141.3 | 230.6 | 112.5 | 179.4 | 4.20 | 1.46 | 0 | 0 | 0.7 | 5.3 | 0.0 | 0.0 | 1.003 ± 0.088 | ok / ok |
| noise_0.5 | fill | 0 | 0 | 131.5 | 208.0 | 106.0 | 166.5 | 4.11 | 1.45 | 0 | 0 | 0.7 | 5.4 | 0.0 | 0.0 | 1.004 ± 0.089 | ok / ok |
| noise_0.5 | dump | 0 | 0 | 170.7 | 273.4 | 135.7 | 210.0 | 4.44 | 1.49 | 0 | 0 | 0.7 | 5.2 | 0.0 | 0.0 | 0.987 ± 0.284 | ok / ok |
| noise_1.0 | baseline | 0 | 0 | 174.0 | 292.9 | 135.1 | 220.6 | 4.16 | 1.50 | 0 | 0 | 0.6 | 6.0 | 0.0 | 0.0 | 1.001 ± 0.177 | ok / ok |
| noise_1.0 | fill_half | 0 | 0 | 151.4 | 247.6 | 119.9 | 191.6 | 4.16 | 1.47 | 0 | 0 | 0.6 | 5.7 | 0.0 | 0.0 | 1.003 ± 0.177 | ok / ok |
| noise_1.0 | fill | 1 | 0 | 137.7 | 219.3 | 110.1 | 174.3 | 3.95 | 1.46 | 1 | 0 | 0.8 | 5.8 | 0.0 | 0.0 | 1.003 ± 0.178 | ok / ok |
| noise_1.0 | dump | 11 | 1 | 178.1 | 290.1 | 142.9 | 223.7 | 4.26 | 1.50 | 3 | 0 | 0.6 | 6.0 | 0.0 | 0.0 | 1.034 ± 0.559 | BREAKS / ok |
| tau_noise_5pct | baseline | 0 | 0 | 153.2 | 242.9 | 120.2 | 185.0 | 4.10 | 1.49 | 0 | 0 | 0.7 | 4.2 | 0.0 | 0.0 | 1.000 ± 0.029 | ok / ok |
| tau_noise_5pct | fill_half | 0 | 0 | 135.6 | 213.3 | 107.4 | 167.4 | 4.24 | 1.47 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 1.000 ± 0.029 | ok / ok |
| tau_noise_5pct | fill | 0 | 0 | 128.4 | 197.7 | 103.1 | 159.4 | 4.15 | 1.46 | 0 | 0 | 0.7 | 4.5 | 0.0 | 0.0 | 1.000 ± 0.029 | ok / ok |
| tau_noise_5pct | dump | 0 | 0 | 166.5 | 257.0 | 132.9 | 197.7 | 4.40 | 1.49 | 0 | 0 | 0.7 | 4.2 | 0.0 | 0.0 | 0.995 ± 0.054 | ok / ok |
| J_x0.38 | baseline | 0 | 0 | 153.3 | 216.6 | 120.3 | 165.9 | 4.11 | 1.47 | 0 | 0 | 0.5 | 4.1 | 0.0 | 0.0 | 1.000 ± 0.001 | ok / ok |
| J_x0.38 | fill_half | 0 | 0 | 134.3 | 193.3 | 106.5 | 152.5 | 4.25 | 1.45 | 0 | 0 | 0.5 | 4.1 | 0.0 | 0.0 | 1.001 ± 0.001 | ok / ok |
| J_x0.38 | fill | 0 | 0 | 125.3 | 177.4 | 100.6 | 144.8 | 4.13 | 1.45 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 1.003 ± 0.001 | ok / ok |
| J_x0.38 | dump | 0 | 0 | 165.6 | 236.0 | 132.2 | 182.1 | 4.41 | 1.47 | 0 | 0 | 0.5 | 4.1 | 0.0 | 0.0 | 0.990 ± 0.005 | ok / ok |
| J_x1.5 | baseline | 0 | 0 | 153.1 | 289.9 | 120.1 | 219.1 | 4.10 | 1.53 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 1.000 ± 0.001 | ok / ok |
| J_x1.5 | fill_half | 0 | 0 | 136.6 | 242.0 | 108.3 | 188.0 | 4.22 | 1.49 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 0.998 ± 0.001 | ok / ok |
| J_x1.5 | fill | 0 | 0 | 131.1 | 219.9 | 105.2 | 175.1 | 4.14 | 1.48 | 0 | 0 | 0.8 | 4.5 | 0.0 | 0.0 | 0.996 ± 0.001 | ok / ok |
| J_x1.5 | dump | 0 | 0 | 167.2 | 287.5 | 133.5 | 220.8 | 4.35 | 1.53 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 0.998 ± 0.001 | ok / ok |
| b_x0.5 | baseline | 0 | 0 | 153.1 | 243.3 | 120.0 | 184.9 | 4.10 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 1.000 ± 0.004 | ok / ok |
| b_x0.5 | fill_half | 0 | 0 | 135.8 | 214.2 | 107.4 | 167.5 | 4.22 | 1.46 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 0.996 ± 0.001 | ok / ok |
| b_x0.5 | fill | 0 | 0 | 128.7 | 198.8 | 103.0 | 159.6 | 4.15 | 1.46 | 0 | 0 | 0.7 | 4.4 | 0.0 | 0.0 | 0.996 ± 0.000 | ok / ok |
| b_x0.5 | dump | 0 | 0 | 166.6 | 257.5 | 132.8 | 197.7 | 4.39 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 1.005 ± 0.002 | ok / ok |
| b_x2 | baseline | 0 | 0 | 153.2 | 242.7 | 120.7 | 185.7 | 4.11 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 1.000 ± 0.008 | ok / ok |
| b_x2 | fill_half | 0 | 0 | 135.1 | 212.2 | 107.5 | 167.5 | 4.26 | 1.47 | 0 | 0 | 0.7 | 4.3 | 0.0 | 0.0 | 1.007 ± 0.001 | ok / ok |
| b_x2 | fill | 0 | 0 | 127.9 | 196.0 | 103.1 | 159.4 | 4.17 | 1.47 | 0 | 0 | 0.7 | 4.4 | 0.0 | 0.0 | 1.007 ± 0.001 | ok / ok |
| b_x2 | dump | 0 | 0 | 166.3 | 256.2 | 133.3 | 197.9 | 4.41 | 1.49 | 0 | 0 | 0.6 | 4.2 | 0.0 | 0.0 | 0.972 ± 0.006 | ok / ok |

### 3.2 Combined cases (sensor noise, 5 % torque noise, J and b scaled)

| condition | variant | miss %: tuned | old | E_mech kJ: tuned | old | E_pump kJ (lower bound, ASSUMED parameters): tuned | old | reach 20 cm, s: tuned | old | never reached: tuned | old | flow limit %: tuned | old | torque limit %: tuned | old | m_hat / m_tip (tuned) | status: tuned / old |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| all_n0.2_J0.38_b0.5 | baseline | 0 | 0 | 158.2 | 222.5 | 123.7 | 169.8 | 4.12 | 1.46 | 0 | 0 | 0.4 | 4.2 | 0.0 | 0.0 | 1.001 ± 0.050 | ok / ok |
| all_n0.2_J0.38_b0.5 | fill_half | 0 | 0 | 137.6 | 197.2 | 109.4 | 154.9 | 4.22 | 1.45 | 0 | 0 | 0.5 | 4.3 | 0.0 | 0.0 | 1.001 ± 0.050 | ok / ok |
| all_n0.2_J0.38_b0.5 | fill | 0 | 0 | 127.0 | 179.8 | 102.1 | 145.8 | 4.12 | 1.45 | 0 | 0 | 0.8 | 4.5 | 0.0 | 0.0 | 1.001 ± 0.050 | ok / ok |
| all_n0.2_J0.38_b0.5 | dump | 0 | 0 | 167.7 | 238.7 | 133.3 | 184.0 | 4.41 | 1.46 | 0 | 0 | 0.4 | 4.2 | 0.0 | 0.0 | 0.995 ± 0.169 | ok / ok |
| all_n0.2_J1.5_b0.5 | baseline | 0 | 1 | 155.8 | 294.1 | 121.9 | 221.5 | 4.10 | 1.53 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 1.000 ± 0.045 | ok / ok |
| all_n0.2_J1.5_b0.5 | fill_half | 0 | 0 | 138.3 | 244.1 | 109.9 | 188.7 | 4.23 | 1.49 | 0 | 0 | 0.8 | 4.5 | 0.0 | 0.0 | 0.999 ± 0.046 | ok / ok |
| all_n0.2_J1.5_b0.5 | fill | 0 | 0 | 131.8 | 222.0 | 105.6 | 175.9 | 4.10 | 1.48 | 0 | 0 | 0.8 | 4.6 | 0.0 | 0.0 | 0.995 ± 0.046 | ok / ok |
| all_n0.2_J1.5_b0.5 | dump | 0 | 0 | 168.3 | 289.7 | 133.9 | 222.2 | 4.38 | 1.53 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 1.005 ± 0.122 | ok / ok |
| all_n0.2_J1.5_b2.0 | baseline | 0 | 1 | 155.6 | 291.5 | 122.4 | 221.2 | 4.11 | 1.54 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 1.000 ± 0.046 | ok / ok |
| all_n0.2_J1.5_b2.0 | fill_half | 0 | 0 | 137.5 | 241.6 | 109.9 | 188.7 | 4.24 | 1.50 | 0 | 0 | 0.8 | 4.5 | 0.0 | 0.0 | 1.009 ± 0.046 | ok / ok |
| all_n0.2_J1.5_b2.0 | fill | 0 | 0 | 130.9 | 218.3 | 105.7 | 175.3 | 4.11 | 1.48 | 0 | 0 | 0.8 | 4.6 | 0.0 | 0.0 | 1.006 ± 0.046 | ok / ok |
| all_n0.2_J1.5_b2.0 | dump | 0 | 0 | 167.9 | 287.9 | 134.4 | 222.0 | 4.40 | 1.54 | 0 | 0 | 0.8 | 4.4 | 0.0 | 0.0 | 0.973 ± 0.121 | ok / ok |
| all_n0.5_J0.38_b0.5 | baseline | 0 | 0 | 168.6 | 241.9 | 131.1 | 183.7 | 4.13 | 1.45 | 0 | 0 | 0.4 | 5.0 | 0.0 | 0.0 | 1.001 ± 0.106 | ok / ok |
| all_n0.5_J0.38_b0.5 | fill_half | 0 | 0 | 146.3 | 211.6 | 116.1 | 165.2 | 4.21 | 1.44 | 0 | 0 | 0.4 | 5.1 | 0.0 | 0.0 | 1.002 ± 0.106 | ok / ok |
| all_n0.5_J0.38_b0.5 | fill | 0 | 0 | 133.3 | 189.5 | 107.2 | 152.9 | 4.09 | 1.43 | 0 | 0 | 0.9 | 5.4 | 0.0 | 0.0 | 1.003 ± 0.107 | ok / ok |
| all_n0.5_J0.38_b0.5 | dump | 0 | 0 | 174.9 | 251.4 | 138.9 | 193.6 | 4.45 | 1.45 | 0 | 0 | 0.4 | 5.0 | 0.0 | 0.0 | 0.997 ± 0.399 | ok / ok |
| all_n0.5_J1.5_b0.5 | baseline | 0 | 3 | 160.8 | 327.7 | 125.5 | 245.3 | 4.11 | 1.53 | 0 | 0 | 0.8 | 6.2 | 0.0 | 0.0 | 1.001 ± 0.092 | ok / degraded |
| all_n0.5_J1.5_b0.5 | fill_half | 0 | 3 | 141.5 | 268.2 | 112.5 | 206.0 | 4.19 | 1.49 | 0 | 0 | 0.8 | 5.8 | 0.0 | 0.0 | 1.000 ± 0.092 | ok / degraded |
| all_n0.5_J1.5_b0.5 | fill | 0 | 0 | 133.7 | 234.3 | 107.4 | 184.8 | 4.09 | 1.47 | 0 | 0 | 0.8 | 5.6 | 0.0 | 0.0 | 0.997 ± 0.092 | ok / ok |
| all_n0.5_J1.5_b0.5 | dump | 0 | 0 | 171.0 | 312.2 | 135.6 | 239.0 | 4.42 | 1.53 | 0 | 0 | 0.8 | 6.1 | 0.0 | 0.0 | 1.001 ± 0.277 | ok / ok |
| all_n0.5_J1.5_b2.0 | baseline | 0 | 3 | 160.1 | 325.0 | 125.6 | 245.0 | 4.12 | 1.53 | 0 | 0 | 0.8 | 6.2 | 0.0 | 0.0 | 1.001 ± 0.091 | ok / degraded |
| all_n0.5_J1.5_b2.0 | fill_half | 0 | 2 | 140.4 | 265.6 | 112.3 | 206.0 | 4.21 | 1.49 | 0 | 0 | 0.8 | 5.8 | 0.0 | 0.0 | 1.007 ± 0.091 | ok / degraded |
| all_n0.5_J1.5_b2.0 | fill | 0 | 0 | 132.7 | 230.9 | 107.4 | 184.4 | 4.11 | 1.47 | 0 | 0 | 0.8 | 5.6 | 0.0 | 0.0 | 1.008 ± 0.092 | ok / ok |
| all_n0.5_J1.5_b2.0 | dump | 0 | 1 | 170.4 | 310.3 | 136.0 | 239.0 | 4.44 | 1.53 | 0 | 0 | 0.8 | 6.1 | 0.0 | 0.0 | 0.973 ± 0.276 | ok / ok |
| all_n1.0_J0.38_b0.5 | baseline | 0 | 0 | 189.7 | 266.9 | 146.6 | 201.6 | 4.18 | 1.46 | 0 | 0 | 0.4 | 5.3 | 0.0 | 0.0 | 1.001 ± 0.207 | ok / ok |
| all_n1.0_J0.38_b0.5 | fill_half | 0 | 0 | 165.9 | 231.0 | 130.7 | 179.4 | 4.15 | 1.45 | 0 | 0 | 0.5 | 5.4 | 0.0 | 0.0 | 1.002 ± 0.207 | ok / ok |
| all_n1.0_J0.38_b0.5 | fill | 0 | 0 | 151.7 | 204.8 | 118.9 | 163.2 | 3.89 | 1.47 | 0 | 0 | 1.1 | 5.8 | 0.0 | 0.0 | 1.001 ± 0.207 | ok / ok |
| all_n1.0_J0.38_b0.5 | dump | 80 | 13 | 193.1 | 269.9 | 159.2 | 210.6 | 3.93 | 1.46 | 11 | 0 | 0.4 | 5.3 | 0.0 | 0.0 | 1.148 ± 0.748 | BREAKS / BREAKS |
| all_n1.0_J1.5_b0.5 | baseline | 0 | 4 | 171.0 | 349.6 | 132.7 | 260.6 | 4.17 | 1.53 | 0 | 0 | 0.7 | 6.8 | 0.0 | 0.0 | 1.002 ± 0.176 | ok / degraded |
| all_n1.0_J1.5_b0.5 | fill_half | 1 | 4 | 149.5 | 283.2 | 118.3 | 216.7 | 4.12 | 1.50 | 1 | 0 | 0.7 | 6.4 | 0.0 | 0.0 | 1.001 ± 0.177 | ok / degraded |
| all_n1.0_J1.5_b0.5 | fill | 1 | 4 | 138.2 | 246.2 | 110.4 | 193.2 | 3.88 | 1.48 | 1 | 0 | 0.8 | 6.2 | 0.0 | 0.0 | 0.999 ± 0.177 | ok / degraded |
| all_n1.0_J1.5_b0.5 | dump | 10 | 1 | 176.4 | 327.4 | 141.0 | 251.2 | 4.34 | 1.53 | 2 | 0 | 0.7 | 6.8 | 0.0 | 0.0 | 1.037 ± 0.538 | BREAKS / ok |
| all_n1.0_J1.5_b2.0 | baseline | 0 | 4 | 169.9 | 346.4 | 132.6 | 260.1 | 4.18 | 1.54 | 0 | 0 | 0.7 | 6.8 | 0.0 | 0.0 | 1.002 ± 0.175 | ok / degraded |
| all_n1.0_J1.5_b2.0 | fill_half | 1 | 4 | 148.2 | 280.5 | 118.1 | 216.7 | 4.14 | 1.50 | 1 | 0 | 0.7 | 6.4 | 0.0 | 0.0 | 1.004 ± 0.176 | ok / degraded |
| all_n1.0_J1.5_b2.0 | fill | 1 | 4 | 136.6 | 242.1 | 110.1 | 192.4 | 3.94 | 1.48 | 1 | 0 | 0.8 | 6.2 | 0.0 | 0.0 | 1.006 ± 0.176 | ok / degraded |
| all_n1.0_J1.5_b2.0 | dump | 10 | 1 | 175.5 | 325.0 | 141.2 | 251.0 | 4.35 | 1.54 | 3 | 0 | 0.7 | 6.7 | 0.0 | 0.0 | 1.023 ± 0.536 | BREAKS / ok |

## 4. Placeholder plant against real plant, row by row

| condition | variant (real / placeholder) | miss %: placeholder | real, old gains | real, tuned | E_mech kJ: placeholder | real, old gains | real, tuned | E_pump kJ (lower bound, ASSUMED parameters): placeholder | real, old gains | real, tuned |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | baseline / baseline | 0 | 0 | 0 | 70.1 | 243.0 | 153.2 | 83.0 | 185.1 | 120.2 |
| none | fill_half / payload_x1.2 | 0 | 0 | 0 | 72.2 | 213.4 | 135.6 | 84.3 | 167.5 | 107.4 |
| noise_0.2 | baseline / baseline | 0 | 0 | 0 | 70.5 | 246.4 | 156.1 | 83.4 | 187.5 | 122.4 |
| noise_0.2 | fill_half / payload_x1.2 | 0 | 0 | 0 | 72.6 | 215.1 | 137.2 | 84.6 | 168.3 | 109.3 |
| noise_0.5 | baseline / baseline | 1 | 0 | 0 | 71.9 | 268.9 | 161.9 | 84.8 | 203.4 | 126.5 |
| noise_0.5 | fill_half / payload_x1.2 | 1 | 0 | 0 | 74.2 | 230.6 | 141.3 | 86.1 | 179.4 | 112.5 |
| noise_1.0 | baseline / baseline | 2 | 0 | 0 | 75.7 | 292.9 | 174.0 | 88.8 | 220.6 | 135.1 |
| noise_1.0 | fill_half / payload_x1.2 | 2 | 0 | 0 | 78.4 | 247.6 | 151.4 | 90.4 | 191.6 | 119.9 |
| tau_noise_5pct | baseline / baseline | 0 | 0 | 0 | 70.1 | 242.9 | 153.2 | 83.0 | 185.0 | 120.2 |
| tau_noise_5pct | fill_half / payload_x1.2 | 0 | 0 | 0 | 72.2 | 213.3 | 135.6 | 84.3 | 167.4 | 107.4 |
| J_x0.38 | baseline / baseline | 0 | 0 | 0 | 69.7 | 216.6 | 153.3 | 82.6 | 165.9 | 120.3 |
| J_x0.38 | fill_half / payload_x1.2 | 0 | 0 | 0 | 71.8 | 193.3 | 134.3 | 83.9 | 152.5 | 106.5 |
| J_x1.5 | baseline / baseline | 0 | 0 | 0 | 70.8 | 289.9 | 153.1 | 83.8 | 219.1 | 120.1 |
| J_x1.5 | fill_half / payload_x1.2 | 0 | 0 | 0 | 72.9 | 242.0 | 136.6 | 85.0 | 188.0 | 108.3 |
| b_x0.5 | baseline / baseline | 0 | 0 | 0 | 66.8 | 243.3 | 153.1 | 72.9 | 184.9 | 120.0 |
| b_x0.5 | fill_half / payload_x1.2 | 0 | 0 | 0 | 69.2 | 214.2 | 135.8 | 74.5 | 167.5 | 107.4 |
| b_x2 | baseline / baseline | 0 | 0 | 0 | 83.9 | 242.7 | 153.2 | 106.9 | 185.7 | 120.7 |
| b_x2 | fill_half / payload_x1.2 | 0 | 0 | 0 | 85.5 | 212.2 | 135.1 | 107.9 | 167.5 | 107.5 |
| all_n0.2_J0.38_b0.5 | baseline / baseline | 1 | 0 | 0 | 67.3 | 222.5 | 158.2 | 73.3 | 169.8 | 123.7 |
| all_n0.2_J0.38_b0.5 | fill_half / payload_x1.2 | 1 | 0 | 0 | 69.8 | 197.2 | 137.6 | 74.8 | 154.9 | 109.4 |
| all_n0.2_J1.5_b0.5 | baseline / baseline | 1 | 1 | 0 | 67.5 | 294.1 | 155.8 | 73.6 | 221.5 | 121.9 |
| all_n0.2_J1.5_b0.5 | fill_half / payload_x1.2 | 1 | 0 | 0 | 69.9 | 244.1 | 138.3 | 75.1 | 188.7 | 109.9 |
| all_n0.2_J1.5_b2.0 | baseline / baseline | 1 | 1 | 0 | 84.0 | 291.5 | 155.6 | 106.9 | 221.2 | 122.4 |
| all_n0.2_J1.5_b2.0 | fill_half / payload_x1.2 | 1 | 0 | 0 | 85.6 | 241.6 | 137.5 | 107.9 | 188.7 | 109.9 |
| all_n0.5_J0.38_b0.5 | baseline / baseline | 1 | 0 | 0 | 71.4 | 241.9 | 168.6 | 77.3 | 183.7 | 131.1 |
| all_n0.5_J0.38_b0.5 | fill_half / payload_x1.2 | 2 | 0 | 0 | 74.3 | 211.6 | 146.3 | 79.2 | 165.2 | 116.1 |
| all_n0.5_J1.5_b0.5 | baseline / baseline | 1 | 3 | 0 | 69.0 | 327.7 | 160.8 | 75.0 | 245.3 | 125.5 |
| all_n0.5_J1.5_b0.5 | fill_half / payload_x1.2 | 2 | 3 | 0 | 71.6 | 268.2 | 141.5 | 76.6 | 206.0 | 112.5 |
| all_n0.5_J1.5_b2.0 | baseline / baseline | 1 | 3 | 0 | 84.8 | 325.0 | 160.1 | 107.7 | 245.0 | 125.6 |
| all_n0.5_J1.5_b2.0 | fill_half / payload_x1.2 | 1 | 2 | 0 | 86.5 | 265.6 | 140.4 | 108.7 | 206.0 | 112.3 |
| all_n1.0_J0.38_b0.5 | baseline / baseline | 3 | 0 | 0 | 81.2 | 266.9 | 189.7 | 87.9 | 201.6 | 146.6 |
| all_n1.0_J0.38_b0.5 | fill_half / payload_x1.2 | 2 | 0 | 0 | 85.5 | 231.0 | 165.9 | 90.4 | 179.4 | 130.7 |
| all_n1.0_J1.5_b0.5 | baseline / baseline | 3 | 4 | 0 | 73.0 | 349.6 | 171.0 | 79.0 | 260.6 | 132.7 |
| all_n1.0_J1.5_b0.5 | fill_half / payload_x1.2 | 2 | 4 | 1 | 76.0 | 283.2 | 149.5 | 80.8 | 216.7 | 118.3 |
| all_n1.0_J1.5_b2.0 | baseline / baseline | 1 | 4 | 0 | 87.0 | 346.4 | 169.9 | 110.1 | 260.1 | 132.6 |
| all_n1.0_J1.5_b2.0 | fill_half / payload_x1.2 | 2 | 4 | 1 | 89.1 | 280.5 | 148.2 | 111.3 | 216.7 | 118.1 |

## 5. Limits

- Conditions: `tau_d` of 09c3 is not a condition here, the angle-dependent torque limit is part of the plant. `b_x0.5` and `b_x2` scale the cylinder friction (1.9 kN, 2.5 kN s/m, ESTIMATED), which is about 1/300 of the placeholder damping.
- The plant keeps the torque-command abstraction of `BoomEnv`: no oil compressibility, no valve dynamics, no relief flow. Lab 09b2 showed that the lab08a Simscape model is not an energy reference either.
- The retraction speed limit is a hard limit here (ASSUMED). The lab08a Simscape model cavitates on boom-down instead.
- Tuning used the nominal plant only. Kd of the old gains and T_f were not retuned.
- All geometry is ESTIMATED from a drawing.
