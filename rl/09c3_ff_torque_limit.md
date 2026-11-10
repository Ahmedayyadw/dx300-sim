# Lab 09c3: feed-forward for the angle-dependent torque limit

Kp=0.5 Ki=0 Kd=0.25, T_f = 0.2 s, 100 unseen scenarios (RNG seed 123), 10 s each. All numbers are from `rl/09c3_ff_torque_limit.csv`.
Feed-forward command u_ff = m_hat g Lc cos(theta) / k: **09c** k = 5e5 (unchanged), **geom** k = 5e5 x |d(theta)| / |d(0)| (limit known from geometry), **gain** k = k_hat (torque per unit command, estimated online from tau / u), **sched** = gain, with the PID terms scaled by 5e5 / k_hat as well.
E_pump uses dp_valve = 20 bar and eta_pump = 0.85, both ASSUMED - not from manual. Status: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok.

## Result

Miss rate >20 cm as baseline / payload_x1.2.

| conditions | 09c | geom | gain | sched |
|---|---:|---:|---:|---:|
| `tau_d` alone | 44% / 47% | 12% / 13% | 14% / 15% | 2% / 4% |
| no stress (`none`) | 1% / 1% | 58% / 58% | 1% / 1% | 1% / 1% |
| the nine single conditions without `tau_d` | 0-6% | 52-59% | 0-6% | 0-6% |
| the nine combined cases (all with `tau_d`) | 43-51% | 5-17% | 8-20% | 1-10% |
| status over all 38 rows | 21 break, 7 degraded, 10 ok | 37 break, 1 degraded, 0 ok | 21 break, 5 degraded, 12 ok | 9 break, 14 degraded, 15 ok |

## 1. One condition at a time

| condition | variant | miss: 09c | geom | gain | sched | E_mech kJ: 09c | geom | gain | sched | E_pump kJ: 09c | geom | gain | sched | m_hat / m (sched) | status: 09c / geom / gain / sched |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| none | baseline | 1% | 58% | 1% | 1% | 68.5 | 78.0 | 68.5 | 68.5 | 80.3 | 93.8 | 80.3 | 80.3 | 1.000 ± 0.000 | ok / BREAKS / ok / ok |
| none | payload_x1.2 | 1% | 58% | 1% | 1% | 70.8 | 82.5 | 70.8 | 70.8 | 81.7 | 98.4 | 81.7 | 81.7 | 1.000 ± 0.000 | ok / BREAKS / ok / ok |
| noise_0.2 | baseline | 1% | 57% | 1% | 1% | 68.9 | 78.3 | 68.9 | 68.9 | 80.6 | 94.1 | 80.6 | 80.6 | 1.000 ± 0.015 | ok / BREAKS / ok / ok |
| noise_0.2 | payload_x1.2 | 1% | 58% | 1% | 1% | 71.1 | 82.8 | 71.1 | 71.1 | 82.0 | 98.7 | 82.0 | 82.0 | 1.000 ± 0.012 | ok / BREAKS / ok / ok |
| noise_0.5 | baseline | 3% | 58% | 3% | 3% | 70.2 | 79.4 | 70.2 | 70.2 | 81.9 | 95.3 | 81.9 | 81.9 | 1.000 ± 0.037 | degraded / BREAKS / degraded / degraded |
| noise_0.5 | payload_x1.2 | 2% | 57% | 2% | 2% | 72.6 | 84.0 | 72.6 | 72.6 | 83.3 | 99.9 | 83.3 | 83.3 | 1.000 ± 0.031 | degraded / BREAKS / degraded / degraded |
| noise_1.0 | baseline | 5% | 56% | 5% | 5% | 73.7 | 82.7 | 73.7 | 73.7 | 85.7 | 98.9 | 85.7 | 85.7 | 0.999 ± 0.075 | degraded / BREAKS / degraded / degraded |
| noise_1.0 | payload_x1.2 | 6% | 56% | 6% | 6% | 76.5 | 87.4 | 76.5 | 76.5 | 87.3 | 103.5 | 87.3 | 87.3 | 0.999 ± 0.062 | BREAKS / BREAKS / BREAKS / BREAKS |
| tau_noise_5pct | baseline | 2% | 56% | 1% | 1% | 68.9 | 78.3 | 68.6 | 68.5 | 80.7 | 94.1 | 80.3 | 80.3 | 1.000 ± 0.017 | degraded / BREAKS / ok / ok |
| tau_noise_5pct | payload_x1.2 | 2% | 56% | 1% | 1% | 71.3 | 82.9 | 70.8 | 70.8 | 82.1 | 98.8 | 81.7 | 81.7 | 1.000 ± 0.017 | degraded / BREAKS / ok / ok |
| J_x0.38 | baseline | 1% | 58% | 1% | 1% | 68.3 | 77.9 | 68.3 | 68.3 | 80.0 | 93.6 | 80.0 | 80.0 | 1.000 ± 0.001 | ok / BREAKS / ok / ok |
| J_x0.38 | payload_x1.2 | 1% | 58% | 1% | 1% | 70.5 | 82.5 | 70.5 | 70.5 | 81.4 | 98.2 | 81.4 | 81.4 | 1.000 ± 0.001 | ok / BREAKS / ok / ok |
| J_x1.5 | baseline | 1% | 58% | 1% | 1% | 69.1 | 78.3 | 69.1 | 69.1 | 80.9 | 94.2 | 80.9 | 80.9 | 1.000 ± 0.001 | ok / BREAKS / ok / ok |
| J_x1.5 | payload_x1.2 | 1% | 57% | 1% | 1% | 71.3 | 82.9 | 71.3 | 71.3 | 82.3 | 98.8 | 82.3 | 82.3 | 1.000 ± 0.001 | ok / BREAKS / ok / ok |
| b_x0.5 | baseline | 2% | 59% | 2% | 2% | 65.7 | 72.4 | 65.7 | 65.7 | 71.2 | 82.0 | 71.2 | 71.2 | 0.997 ± 0.018 | degraded / BREAKS / degraded / degraded |
| b_x0.5 | payload_x1.2 | 4% | 58% | 4% | 4% | 68.3 | 76.9 | 68.3 | 68.3 | 72.8 | 86.4 | 72.8 | 72.8 | 0.996 ± 0.015 | degraded / BREAKS / degraded / degraded |
| b_x2 | baseline | 0% | 52% | 0% | 0% | 80.3 | 96.2 | 80.3 | 80.3 | 101.8 | 123.6 | 101.8 | 101.8 | 1.005 ± 0.032 | ok / BREAKS / ok / ok |
| b_x2 | payload_x1.2 | 1% | 52% | 1% | 1% | 82.1 | 101.1 | 82.1 | 82.1 | 102.9 | 129.1 | 102.9 | 102.9 | 1.009 ± 0.026 | ok / BREAKS / ok / ok |
| tau_d | baseline | 44% | 12% | 14% | 2% | 63.1 | 68.2 | 67.9 | 68.3 | 71.7 | 78.8 | 78.7 | 79.9 | 1.000 ± 0.000 | BREAKS / BREAKS / BREAKS / degraded |
| tau_d | payload_x1.2 | 47% | 13% | 15% | 4% | 64.9 | 70.5 | 70.3 | 70.5 | 71.9 | 80.2 | 80.0 | 81.3 | 1.000 ± 0.000 | BREAKS / BREAKS / BREAKS / degraded |

## 2. All together (sensor noise, 5 % torque noise, J and b scaled, tau_d)

The three J x0.38, b x2 cases are left out (Euler step at its stability limit, 09c2 section 5).

| condition | variant | miss: 09c | geom | gain | sched | E_mech kJ: 09c | geom | gain | sched | E_pump kJ: 09c | geom | gain | sched | m_hat / m (sched) | status: 09c / geom / gain / sched |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| all_n0.2_J0.38_b0.5 | baseline | 45% | 15% | 18% | 5% | 62.9 | 66.6 | 66.0 | 66.2 | 65.1 | 71.2 | 70.7 | 71.6 | 0.998 ± 0.028 | BREAKS / BREAKS / BREAKS / degraded |
| all_n0.2_J0.38_b0.5 | payload_x1.2 | 49% | 14% | 15% | 7% | 65.3 | 69.6 | 68.7 | 68.9 | 65.7 | 73.0 | 72.3 | 73.2 | 0.996 ± 0.026 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n0.2_J1.5_b0.5 | baseline | 44% | 15% | 20% | 5% | 62.8 | 66.3 | 65.8 | 66.2 | 65.1 | 71.2 | 70.7 | 71.7 | 0.997 ± 0.030 | BREAKS / BREAKS / BREAKS / degraded |
| all_n0.2_J1.5_b0.5 | payload_x1.2 | 49% | 15% | 15% | 7% | 65.1 | 69.3 | 68.6 | 68.9 | 65.7 | 73.0 | 72.4 | 73.3 | 0.996 ± 0.027 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n0.2_J1.5_b2.0 | baseline | 43% | 5% | 9% | 1% | 72.4 | 78.7 | 78.2 | 80.4 | 90.1 | 99.2 | 98.6 | 101.8 | 1.006 ± 0.040 | BREAKS / degraded / BREAKS / ok |
| all_n0.2_J1.5_b2.0 | payload_x1.2 | 48% | 8% | 11% | 1% | 73.8 | 80.8 | 80.1 | 82.2 | 90.0 | 100.3 | 99.6 | 102.9 | 1.009 ± 0.035 | BREAKS / BREAKS / BREAKS / ok |
| all_n0.5_J0.38_b0.5 | baseline | 45% | 14% | 18% | 7% | 66.7 | 69.9 | 69.4 | 70.0 | 68.8 | 74.6 | 74.3 | 75.4 | 0.998 ± 0.045 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n0.5_J0.38_b0.5 | payload_x1.2 | 49% | 16% | 19% | 5% | 69.5 | 73.2 | 72.6 | 73.1 | 69.7 | 76.5 | 76.1 | 77.2 | 0.996 ± 0.039 | BREAKS / BREAKS / BREAKS / degraded |
| all_n0.5_J1.5_b0.5 | baseline | 45% | 14% | 19% | 6% | 64.2 | 67.6 | 67.1 | 67.6 | 66.4 | 72.4 | 72.0 | 73.0 | 0.998 ± 0.046 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n0.5_J1.5_b0.5 | payload_x1.2 | 49% | 17% | 19% | 5% | 66.6 | 70.6 | 70.0 | 70.4 | 67.1 | 74.2 | 73.7 | 74.7 | 0.996 ± 0.040 | BREAKS / BREAKS / BREAKS / degraded |
| all_n0.5_J1.5_b2.0 | baseline | 44% | 7% | 8% | 1% | 73.1 | 79.4 | 78.9 | 81.1 | 90.8 | 100.0 | 99.3 | 102.6 | 1.005 ± 0.055 | BREAKS / BREAKS / BREAKS / ok |
| all_n0.5_J1.5_b2.0 | payload_x1.2 | 47% | 8% | 10% | 2% | 74.5 | 81.5 | 80.9 | 83.0 | 90.7 | 101.1 | 100.3 | 103.7 | 1.008 ± 0.047 | BREAKS / BREAKS / BREAKS / degraded |
| all_n1.0_J0.38_b0.5 | baseline | 48% | 14% | 17% | 7% | 75.8 | 78.7 | 78.5 | 79.4 | 78.4 | 84.0 | 83.9 | 85.5 | 0.998 ± 0.080 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n1.0_J0.38_b0.5 | payload_x1.2 | 51% | 17% | 20% | 9% | 79.9 | 83.0 | 82.8 | 83.8 | 80.1 | 86.5 | 86.4 | 88.0 | 0.997 ± 0.067 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n1.0_J1.5_b0.5 | baseline | 48% | 14% | 18% | 8% | 68.0 | 71.1 | 70.7 | 71.3 | 70.1 | 76.0 | 75.6 | 76.8 | 0.998 ± 0.082 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n1.0_J1.5_b0.5 | payload_x1.2 | 51% | 16% | 20% | 10% | 70.8 | 74.4 | 73.9 | 74.5 | 71.1 | 77.9 | 77.4 | 78.6 | 0.997 ± 0.069 | BREAKS / BREAKS / BREAKS / BREAKS |
| all_n1.0_J1.5_b2.0 | baseline | 45% | 6% | 10% | 3% | 75.0 | 81.4 | 80.9 | 83.2 | 92.9 | 102.2 | 101.5 | 104.9 | 1.003 ± 0.091 | BREAKS / BREAKS / BREAKS / degraded |
| all_n1.0_J1.5_b2.0 | payload_x1.2 | 47% | 7% | 9% | 2% | 76.8 | 83.7 | 83.1 | 85.4 | 93.0 | 103.4 | 102.6 | 106.1 | 1.007 ± 0.076 | BREAKS / BREAKS / BREAKS / degraded |
