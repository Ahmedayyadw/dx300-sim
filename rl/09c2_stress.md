# Lab 09c2: stress test of the payload estimator

Payload-aware PID of Lab 09c, Kp=0.5 Ki=0 Kd=0.25, 100 unseen scenarios (RNG seed 123), 10 s each. All numbers are from `rl/09c2_stress.csv`.
`m_hat / m` is mean ± std over the last 2 s of all episodes. E_pump uses dp_valve = 20 bar and eta_pump = 0.85, both ASSUMED - not from manual.
Status: BREAKS if miss rate > 5 %, degraded if > 1 % (the 09c result), else ok. "old PID miss" is the Lab 07 PID in the same condition.

## Verdict: which conditions break the controller

Miss rate >20 cm as baseline / payload_x1.2, T_f = 0.2 s unless stated.

1. **`tau_d` breaks it: 44% / 47%.** The estimator is right there (m_hat / m = 1.000 ± 0.000) and the old PID misses the same 44% on baseline. The fault is in the feed-forward: it divides by the nominal torque limit 5e5 N·m (`PID.act` in `rl/lab09_common.py`) while the true limit follows the moment arm. Under payload it still beats the old PID (47% vs 100%).
2. **Sensor noise of 1.0° breaks it under payload (6%) and degrades baseline (5%, at the 5 % line)**; old PID 2% / 100%. 0.5° degrades it (3% / 2%). 0.2° holds (1% / 1%). The m_hat / m std on baseline grows from 0.015 at 0.2° to 0.075 at 1.0°.
3. **5 % torque noise degrades it: 2% / 2%.**
4. **A wrong J holds:** 1% / 1% at J x0.38 and 1% / 1% at J x1.5.
5. **b x0.5 degrades it: 2% / 4%.** On baseline the old PID is better there (0%). **b x2 holds: 0% / 1%**, where the old PID misses 67% / 80%.
6. **All together, as requested: every case breaks.** The nine cases with a valid integrator step miss 43-51%, the `tau_d` level of point 1.
7. **All together without `tau_d`:** noise 1.0° with b x0.5 breaks (8-9%). Noise 0.2° and 0.5° with b x0.5 are degraded (3-5%). J x1.5 with b x2 holds at 0.2° and 0.5° (1%) and is degraded at 1.0° (3-4%).
8. **J x0.38 with b x2 is a simulator failure, not a controller failure.** b·dt/J = 1.97 against a stability limit of 2 for the env's Euler step. The payload-aware PID misses 100%, the old PID 98-100%, and E_mech is 28566 kJ or more. With 10 substeps the same cases miss 1-4% without `tau_d` and 43-48% with it, in line with the other cases.
9. **Filter T_f:** without `tau_d`, mean miss over the nine valid cases is 4.1% / 4.4% at 0.1 s, 4.1% / 4.4% at 0.2 s, 6.0% / 7.3% at 0.5 s, 9.9% / 15.3% at 1.0 s. Above 0.2 s the miss rate rises; 0.1 s and 0.2 s miss equally (m_hat / m std per T_f in 4b). With `tau_d` no T_f helps: 45.8% / 48.9% at 0.1 s, 45.2% / 48.9% at 0.2 s, 45.7% / 50.2% at 0.5 s, 47.6% / 58.9% at 1.0 s.

## 1. One condition at a time (T_f = 0.2 s)

| condition | variant | m_hat / m | E_mech kJ | E_pump kJ | miss >20 cm | old PID miss | status |
|---|---|---|---:|---:|---:|---:|---|
| none | baseline | 1.000 ± 0.000 | 68.5 | 80.3 | 1% | 1% | ok |
| none | payload_x1.2 | 1.000 ± 0.000 | 70.8 | 81.7 | 1% | 100% | ok |
| noise_0.2 | baseline | 1.000 ± 0.015 | 68.9 | 80.6 | 1% | 1% | ok |
| noise_0.2 | payload_x1.2 | 1.000 ± 0.012 | 71.1 | 82.0 | 1% | 100% | ok |
| noise_0.5 | baseline | 1.000 ± 0.037 | 70.2 | 81.9 | 3% | 1% | degraded |
| noise_0.5 | payload_x1.2 | 1.000 ± 0.031 | 72.6 | 83.3 | 2% | 100% | degraded |
| noise_1.0 | baseline | 0.999 ± 0.075 | 73.7 | 85.7 | 5% | 2% | degraded |
| noise_1.0 | payload_x1.2 | 0.999 ± 0.062 | 76.5 | 87.3 | 6% | 100% | BREAKS |
| tau_noise_5pct | baseline | 1.000 ± 0.017 | 68.9 | 80.7 | 2% | 1% | degraded |
| tau_noise_5pct | payload_x1.2 | 1.000 ± 0.017 | 71.3 | 82.1 | 2% | 100% | degraded |
| J_x0.38 | baseline | 1.000 ± 0.001 | 68.3 | 80.0 | 1% | 1% | ok |
| J_x0.38 | payload_x1.2 | 1.000 ± 0.001 | 70.5 | 81.4 | 1% | 100% | ok |
| J_x1.5 | baseline | 1.000 ± 0.001 | 69.1 | 80.9 | 1% | 1% | ok |
| J_x1.5 | payload_x1.2 | 1.000 ± 0.001 | 71.3 | 82.3 | 1% | 100% | ok |
| b_x0.5 | baseline | 0.997 ± 0.018 | 65.7 | 71.2 | 2% | 0% | degraded |
| b_x0.5 | payload_x1.2 | 0.996 ± 0.015 | 68.3 | 72.8 | 4% | 100% | degraded |
| b_x2 | baseline | 1.005 ± 0.032 | 80.3 | 101.8 | 0% | 67% | ok |
| b_x2 | payload_x1.2 | 1.009 ± 0.026 | 82.1 | 102.9 | 1% | 80% | ok |
| tau_d | baseline | 1.000 ± 0.000 | 63.1 | 71.7 | 44% | 44% | BREAKS |
| tau_d | payload_x1.2 | 1.000 ± 0.000 | 64.9 | 71.9 | 47% | 100% | BREAKS |

Worst of the two variants per condition:

- **Breaks** (miss > 5 %): `noise_1.0` (6%, payload_x1.2), `tau_d` (47%, payload_x1.2)
- **Degraded** (miss > 1 %): `noise_0.5` (3%, baseline), `tau_noise_5pct` (2%, baseline), `b_x0.5` (4%, payload_x1.2)
- **Holds** (miss <= 1 %): `none` (1%, baseline), `noise_0.2` (1%, baseline), `J_x0.38` (1%, baseline), `J_x1.5` (1%, baseline), `b_x2` (1%, payload_x1.2)

## 2. All together (T_f = 0.2 s)

Each case: sensor noise at the given level, 5 % torque noise, J and b scaled, tau_d.

| condition | variant | m_hat / m | E_mech kJ | E_pump kJ | miss >20 cm | old PID miss | b·h/J | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| all_n0.2_J0.38_b0.5 | baseline | 1.000 ± 0.023 | 62.9 | 65.1 | 45% | 45% | 0.49 | BREAKS |
| all_n0.2_J0.38_b0.5 | payload_x1.2 | 1.000 ± 0.020 | 65.3 | 65.7 | 49% | 100% | 0.49 | BREAKS |
| all_n0.2_J0.38_b2.0 | baseline | 0.993 ± 0.890 | 29137.2 | 37224.3 | 100% | 100% | 1.97 | BREAKS |
| all_n0.2_J0.38_b2.0 | payload_x1.2 | 0.866 ± 0.722 | 28566.2 | 36476.6 | 100% | 98% | 1.97 | BREAKS |
| all_n0.2_J1.5_b0.5 | baseline | 1.000 ± 0.026 | 62.8 | 65.1 | 44% | 44% | 0.12 | BREAKS |
| all_n0.2_J1.5_b0.5 | payload_x1.2 | 0.999 ± 0.023 | 65.1 | 65.7 | 49% | 100% | 0.12 | BREAKS |
| all_n0.2_J1.5_b2.0 | baseline | 0.999 ± 0.036 | 72.4 | 90.1 | 43% | 77% | 0.50 | BREAKS |
| all_n0.2_J1.5_b2.0 | payload_x1.2 | 1.001 ± 0.031 | 73.8 | 90.0 | 48% | 85% | 0.50 | BREAKS |
| all_n0.5_J0.38_b0.5 | baseline | 1.000 ± 0.040 | 66.7 | 68.8 | 45% | 45% | 0.49 | BREAKS |
| all_n0.5_J0.38_b0.5 | payload_x1.2 | 1.000 ± 0.034 | 69.5 | 69.7 | 49% | 100% | 0.49 | BREAKS |
| all_n0.5_J0.38_b2.0 | baseline | 0.955 ± 0.892 | 29159.5 | 37254.4 | 100% | 99% | 1.97 | BREAKS |
| all_n0.5_J0.38_b2.0 | payload_x1.2 | 0.836 ± 0.719 | 28615.8 | 36544.7 | 100% | 99% | 1.97 | BREAKS |
| all_n0.5_J1.5_b0.5 | baseline | 1.000 ± 0.043 | 64.2 | 66.4 | 45% | 45% | 0.12 | BREAKS |
| all_n0.5_J1.5_b0.5 | payload_x1.2 | 1.000 ± 0.037 | 66.6 | 67.1 | 49% | 100% | 0.12 | BREAKS |
| all_n0.5_J1.5_b2.0 | baseline | 0.998 ± 0.051 | 73.1 | 90.8 | 44% | 77% | 0.50 | BREAKS |
| all_n0.5_J1.5_b2.0 | payload_x1.2 | 1.000 ± 0.044 | 74.5 | 90.7 | 47% | 85% | 0.50 | BREAKS |
| all_n1.0_J0.38_b0.5 | baseline | 1.000 ± 0.075 | 75.8 | 78.4 | 48% | 45% | 0.49 | BREAKS |
| all_n1.0_J0.38_b0.5 | payload_x1.2 | 1.000 ± 0.063 | 79.9 | 80.1 | 51% | 100% | 0.49 | BREAKS |
| all_n1.0_J0.38_b2.0 | baseline | 0.911 ± 0.894 | 28792.9 | 36786.1 | 100% | 100% | 1.97 | BREAKS |
| all_n1.0_J0.38_b2.0 | payload_x1.2 | 0.792 ± 0.716 | 28627.1 | 36560.4 | 100% | 99% | 1.97 | BREAKS |
| all_n1.0_J1.5_b0.5 | baseline | 1.001 ± 0.078 | 68.0 | 70.1 | 48% | 45% | 0.12 | BREAKS |
| all_n1.0_J1.5_b0.5 | payload_x1.2 | 1.000 ± 0.065 | 70.8 | 71.1 | 51% | 100% | 0.12 | BREAKS |
| all_n1.0_J1.5_b2.0 | baseline | 0.997 ± 0.086 | 75.0 | 92.9 | 45% | 77% | 0.50 | BREAKS |
| all_n1.0_J1.5_b2.0 | payload_x1.2 | 0.999 ± 0.072 | 76.8 | 93.0 | 47% | 85% | 0.50 | BREAKS |

Worst of the two variants per case:

- **Breaks** (miss > 5 %): `all_n0.2_J0.38_b0.5` (49%, payload_x1.2), `all_n0.2_J0.38_b2.0` (100%, baseline), `all_n0.2_J1.5_b0.5` (49%, payload_x1.2), `all_n0.2_J1.5_b2.0` (48%, payload_x1.2), `all_n0.5_J0.38_b0.5` (49%, payload_x1.2), `all_n0.5_J0.38_b2.0` (100%, baseline), `all_n0.5_J1.5_b0.5` (49%, payload_x1.2), `all_n0.5_J1.5_b2.0` (47%, payload_x1.2), `all_n1.0_J0.38_b0.5` (51%, payload_x1.2), `all_n1.0_J0.38_b2.0` (100%, baseline), `all_n1.0_J1.5_b0.5` (51%, payload_x1.2), `all_n1.0_J1.5_b2.0` (47%, payload_x1.2)
- **Degraded** (miss > 1 %): none
- **Holds** (miss <= 1 %): none

## 3. All together without tau_d (T_f = 0.2 s)

Not in the request. Added because tau_d alone already breaks the controller (section 1) and hides the rest.

| condition | variant | m_hat / m | E_mech kJ | E_pump kJ | miss >20 cm | old PID miss | b·h/J | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| noTauD_n0.2_J0.38_b0.5 | baseline | 0.997 ± 0.026 | 67.0 | 72.3 | 3% | 0% | 0.49 | degraded |
| noTauD_n0.2_J0.38_b0.5 | payload_x1.2 | 0.996 ± 0.023 | 69.8 | 74.1 | 3% | 100% | 0.49 | degraded |
| noTauD_n0.2_J0.38_b2.0 | baseline | 0.880 ± 0.922 | 31836.0 | 39977.6 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.2_J0.38_b2.0 | payload_x1.2 | 0.748 ± 0.757 | 30691.5 | 38521.1 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.2_J1.5_b0.5 | baseline | 0.997 ± 0.030 | 66.9 | 72.4 | 3% | 0% | 0.12 | degraded |
| noTauD_n0.2_J1.5_b0.5 | payload_x1.2 | 0.996 ± 0.027 | 69.7 | 74.2 | 4% | 100% | 0.12 | degraded |
| noTauD_n0.2_J1.5_b2.0 | baseline | 1.005 ± 0.041 | 81.2 | 102.8 | 1% | 67% | 0.50 | ok |
| noTauD_n0.2_J1.5_b2.0 | payload_x1.2 | 1.009 ± 0.037 | 83.1 | 103.9 | 1% | 80% | 0.50 | ok |
| noTauD_n0.5_J0.38_b0.5 | baseline | 0.998 ± 0.044 | 70.7 | 76.0 | 5% | 0% | 0.49 | degraded |
| noTauD_n0.5_J0.38_b0.5 | payload_x1.2 | 0.996 ± 0.037 | 73.9 | 78.0 | 4% | 100% | 0.49 | degraded |
| noTauD_n0.5_J0.38_b2.0 | baseline | 0.845 ± 0.917 | 31787.7 | 39919.8 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.5_J0.38_b2.0 | payload_x1.2 | 0.704 ± 0.753 | 30647.6 | 38469.6 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.5_J1.5_b0.5 | baseline | 0.997 ± 0.046 | 68.3 | 73.7 | 5% | 0% | 0.12 | degraded |
| noTauD_n0.5_J1.5_b0.5 | payload_x1.2 | 0.996 ± 0.040 | 71.2 | 75.5 | 5% | 100% | 0.12 | degraded |
| noTauD_n0.5_J1.5_b2.0 | baseline | 1.004 ± 0.056 | 82.0 | 103.6 | 1% | 67% | 0.50 | ok |
| noTauD_n0.5_J1.5_b2.0 | payload_x1.2 | 1.008 ± 0.048 | 83.9 | 104.7 | 1% | 80% | 0.50 | ok |
| noTauD_n1.0_J0.38_b0.5 | baseline | 0.998 ± 0.079 | 80.0 | 86.1 | 8% | 0% | 0.49 | BREAKS |
| noTauD_n1.0_J0.38_b0.5 | payload_x1.2 | 0.997 ± 0.066 | 84.4 | 88.7 | 9% | 100% | 0.49 | BREAKS |
| noTauD_n1.0_J0.38_b2.0 | baseline | 0.811 ± 0.913 | 31728.4 | 39849.9 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n1.0_J0.38_b2.0 | payload_x1.2 | 0.670 ± 0.745 | 30620.4 | 38441.4 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n1.0_J1.5_b0.5 | baseline | 0.998 ± 0.082 | 72.0 | 77.4 | 8% | 0% | 0.12 | BREAKS |
| noTauD_n1.0_J1.5_b0.5 | payload_x1.2 | 0.996 ± 0.069 | 75.2 | 79.4 | 9% | 100% | 0.12 | BREAKS |
| noTauD_n1.0_J1.5_b2.0 | baseline | 1.003 ± 0.092 | 84.0 | 105.9 | 3% | 66% | 0.50 | degraded |
| noTauD_n1.0_J1.5_b2.0 | payload_x1.2 | 1.007 ± 0.077 | 86.3 | 107.1 | 4% | 80% | 0.50 | degraded |

Worst of the two variants per case:

- **Breaks** (miss > 5 %): `noTauD_n0.2_J0.38_b2.0` (100%, baseline), `noTauD_n0.5_J0.38_b2.0` (100%, baseline), `noTauD_n1.0_J0.38_b0.5` (9%, payload_x1.2), `noTauD_n1.0_J0.38_b2.0` (100%, baseline), `noTauD_n1.0_J1.5_b0.5` (9%, payload_x1.2)
- **Degraded** (miss > 1 %): `noTauD_n0.2_J0.38_b0.5` (3%, baseline), `noTauD_n0.2_J1.5_b0.5` (4%, payload_x1.2), `noTauD_n0.5_J0.38_b0.5` (5%, baseline), `noTauD_n0.5_J1.5_b0.5` (5%, baseline), `noTauD_n1.0_J1.5_b2.0` (4%, payload_x1.2)
- **Holds** (miss <= 1 %): `noTauD_n0.2_J1.5_b2.0` (1%, baseline), `noTauD_n0.5_J1.5_b2.0` (1%, baseline)

## 4. Filter sweep

### 4a. Combined cases (with tau_d)

Mean over the 9 cases with b·dt/J < 1.5 (the J x0.38, b x2 cases are in section 5); "worst miss" is the worst of those cases.

| T_f s | variant | m_hat / m mean | m_hat / m std | E_mech kJ | E_pump kJ | mean miss | worst miss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | baseline | 1.000 | 0.079 | 71.6 | 79.2 | 45.8% | 48% |
| 0.1 | payload_x1.2 | 1.000 | 0.067 | 74.1 | 79.9 | 48.9% | 51% |
| 0.2 | baseline | 1.000 | 0.051 | 69.0 | 76.4 | 45.2% | 48% |
| 0.2 | payload_x1.2 | 1.000 | 0.043 | 71.4 | 77.0 | 48.9% | 51% |
| 0.5 | baseline | 0.999 | 0.032 | 66.8 | 74.3 | 45.7% | 48% |
| 0.5 | payload_x1.2 | 0.999 | 0.027 | 69.7 | 75.2 | 50.2% | 52% |
| 1.0 | baseline | 0.999 | 0.032 | 65.6 | 73.3 | 47.6% | 50% |
| 1.0 | payload_x1.2 | 0.992 | 0.027 | 69.5 | 74.4 | 58.9% | 66% |

Miss rate >20 cm per case and T_f (m_hat / m std in brackets):

| condition | variant | T_f 0.1 s | T_f 0.2 s | T_f 0.5 s | T_f 1.0 s |
|---|---|---:|---:|---:|---:|
| all_n0.2_J0.38_b0.5 | baseline | 45% (0.034) | 45% (0.023) | 45% (0.018) | 49% (0.020) |
| all_n0.2_J0.38_b0.5 | payload_x1.2 | 49% (0.030) | 49% (0.020) | 50% (0.015) | 54% (0.016) |
| all_n0.2_J0.38_b2.0 | baseline | 100% (0.918) | 100% (0.890) | 100% (0.844) | 100% (0.785) |
| all_n0.2_J0.38_b2.0 | payload_x1.2 | 100% (0.746) | 100% (0.722) | 100% (0.680) | 100% (0.626) |
| all_n0.2_J1.5_b0.5 | baseline | 45% (0.036) | 44% (0.026) | 46% (0.020) | 49% (0.021) |
| all_n0.2_J1.5_b0.5 | payload_x1.2 | 49% (0.033) | 49% (0.023) | 50% (0.017) | 55% (0.018) |
| all_n0.2_J1.5_b2.0 | baseline | 44% (0.044) | 43% (0.036) | 44% (0.030) | 45% (0.045) |
| all_n0.2_J1.5_b2.0 | payload_x1.2 | 48% (0.039) | 48% (0.031) | 49% (0.025) | 65% (0.037) |
| all_n0.5_J0.38_b0.5 | baseline | 45% (0.072) | 45% (0.040) | 46% (0.024) | 48% (0.022) |
| all_n0.5_J0.38_b0.5 | payload_x1.2 | 48% (0.060) | 49% (0.034) | 51% (0.020) | 56% (0.018) |
| all_n0.5_J0.38_b2.0 | baseline | 100% (0.915) | 100% (0.892) | 100% (0.847) | 100% (0.785) |
| all_n0.5_J0.38_b2.0 | payload_x1.2 | 100% (0.744) | 100% (0.719) | 100% (0.685) | 99% (0.629) |
| all_n0.5_J1.5_b0.5 | baseline | 45% (0.067) | 45% (0.043) | 46% (0.026) | 48% (0.024) |
| all_n0.5_J1.5_b0.5 | payload_x1.2 | 48% (0.057) | 49% (0.037) | 52% (0.022) | 56% (0.020) |
| all_n0.5_J1.5_b2.0 | baseline | 44% (0.071) | 44% (0.051) | 44% (0.037) | 45% (0.047) |
| all_n0.5_J1.5_b2.0 | payload_x1.2 | 48% (0.060) | 47% (0.044) | 48% (0.031) | 65% (0.039) |
| all_n1.0_J0.38_b0.5 | baseline | 48% (0.139) | 48% (0.075) | 48% (0.038) | 48% (0.029) |
| all_n1.0_J0.38_b0.5 | payload_x1.2 | 51% (0.116) | 51% (0.063) | 51% (0.032) | 56% (0.024) |
| all_n1.0_J0.38_b2.0 | baseline | 100% (0.917) | 100% (0.894) | 100% (0.845) | 100% (0.810) |
| all_n1.0_J0.38_b2.0 | payload_x1.2 | 100% (0.745) | 100% (0.716) | 100% (0.684) | 100% (0.650) |
| all_n1.0_J1.5_b0.5 | baseline | 48% (0.125) | 48% (0.078) | 48% (0.042) | 50% (0.031) |
| all_n1.0_J1.5_b0.5 | payload_x1.2 | 51% (0.104) | 51% (0.065) | 51% (0.035) | 57% (0.026) |
| all_n1.0_J1.5_b2.0 | baseline | 48% (0.127) | 45% (0.086) | 44% (0.055) | 46% (0.055) |
| all_n1.0_J1.5_b2.0 | payload_x1.2 | 48% (0.106) | 47% (0.072) | 50% (0.045) | 66% (0.045) |

### 4b. Combined cases without tau_d

Mean over the 9 cases with b·dt/J < 1.5 (the J x0.38, b x2 cases are in section 5); "worst miss" is the worst of those cases.

| T_f s | variant | m_hat / m mean | m_hat / m std | E_mech kJ | E_pump kJ | mean miss | worst miss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | baseline | 1.000 | 0.084 | 77.4 | 88.6 | 4.1% | 8% |
| 0.1 | payload_x1.2 | 1.000 | 0.072 | 80.5 | 90.5 | 4.4% | 9% |
| 0.2 | baseline | 1.000 | 0.055 | 74.7 | 85.6 | 4.1% | 8% |
| 0.2 | payload_x1.2 | 1.000 | 0.047 | 77.5 | 87.3 | 4.4% | 9% |
| 0.5 | baseline | 1.000 | 0.037 | 72.3 | 83.4 | 6.0% | 10% |
| 0.5 | payload_x1.2 | 1.000 | 0.032 | 75.5 | 85.1 | 7.3% | 13% |
| 1.0 | baseline | 1.001 | 0.040 | 71.1 | 82.3 | 9.9% | 16% |
| 1.0 | payload_x1.2 | 0.994 | 0.033 | 75.1 | 83.9 | 15.3% | 20% |

Miss rate >20 cm per case and T_f (m_hat / m std in brackets):

| condition | variant | T_f 0.1 s | T_f 0.2 s | T_f 0.5 s | T_f 1.0 s |
|---|---|---:|---:|---:|---:|
| noTauD_n0.2_J0.38_b0.5 | baseline | 3% (0.038) | 3% (0.026) | 9% (0.022) | 14% (0.025) |
| noTauD_n0.2_J0.38_b0.5 | payload_x1.2 | 3% (0.033) | 3% (0.023) | 9% (0.019) | 20% (0.021) |
| noTauD_n0.2_J0.38_b2.0 | baseline | 100% (0.950) | 100% (0.922) | 100% (0.849) | 100% (0.771) |
| noTauD_n0.2_J0.38_b2.0 | payload_x1.2 | 100% (0.779) | 100% (0.757) | 100% (0.699) | 100% (0.617) |
| noTauD_n0.2_J1.5_b0.5 | baseline | 3% (0.040) | 3% (0.030) | 8% (0.025) | 14% (0.027) |
| noTauD_n0.2_J1.5_b0.5 | payload_x1.2 | 3% (0.036) | 4% (0.027) | 9% (0.021) | 19% (0.023) |
| noTauD_n0.2_J1.5_b2.0 | baseline | 1% (0.049) | 1% (0.041) | 1% (0.037) | 0% (0.056) |
| noTauD_n0.2_J1.5_b2.0 | payload_x1.2 | 1% (0.044) | 1% (0.037) | 1% (0.032) | 1% (0.048) |
| noTauD_n0.5_J0.38_b0.5 | baseline | 5% (0.077) | 5% (0.044) | 7% (0.028) | 14% (0.027) |
| noTauD_n0.5_J0.38_b0.5 | payload_x1.2 | 4% (0.065) | 4% (0.037) | 10% (0.024) | 20% (0.023) |
| noTauD_n0.5_J0.38_b2.0 | baseline | 100% (0.942) | 100% (0.917) | 100% (0.856) | 100% (0.776) |
| noTauD_n0.5_J0.38_b2.0 | payload_x1.2 | 100% (0.771) | 100% (0.753) | 100% (0.697) | 100% (0.627) |
| noTauD_n0.5_J1.5_b0.5 | baseline | 4% (0.070) | 5% (0.046) | 8% (0.031) | 15% (0.029) |
| noTauD_n0.5_J1.5_b0.5 | payload_x1.2 | 4% (0.060) | 5% (0.040) | 9% (0.026) | 20% (0.024) |
| noTauD_n0.5_J1.5_b2.0 | baseline | 2% (0.075) | 1% (0.056) | 1% (0.044) | 0% (0.059) |
| noTauD_n0.5_J1.5_b2.0 | payload_x1.2 | 3% (0.065) | 1% (0.048) | 1% (0.037) | 8% (0.050) |
| noTauD_n1.0_J0.38_b0.5 | baseline | 8% (0.148) | 8% (0.079) | 9% (0.041) | 15% (0.033) |
| noTauD_n1.0_J0.38_b0.5 | payload_x1.2 | 9% (0.124) | 9% (0.066) | 12% (0.035) | 19% (0.027) |
| noTauD_n1.0_J0.38_b2.0 | baseline | 100% (0.934) | 100% (0.913) | 100% (0.856) | 100% (0.769) |
| noTauD_n1.0_J0.38_b2.0 | payload_x1.2 | 100% (0.764) | 100% (0.745) | 100% (0.701) | 100% (0.625) |
| noTauD_n1.0_J1.5_b0.5 | baseline | 7% (0.130) | 8% (0.082) | 10% (0.046) | 16% (0.035) |
| noTauD_n1.0_J1.5_b0.5 | payload_x1.2 | 8% (0.109) | 9% (0.069) | 13% (0.038) | 20% (0.030) |
| noTauD_n1.0_J1.5_b2.0 | baseline | 4% (0.133) | 3% (0.092) | 1% (0.061) | 1% (0.066) |
| noTauD_n1.0_J1.5_b2.0 | payload_x1.2 | 5% (0.112) | 4% (0.077) | 2% (0.052) | 11% (0.055) |

## 5. J x0.38 with b x2: integrator check

In these cases b·dt/J = 1.97, at the stability limit 2 of the env's explicit Euler step. Same cases with 10 substeps per control step (T_f = 0.2 s). Energies of the substep rows use the end-of-step omega and are approximate.

| condition | variant | m_hat / m | E_mech kJ | E_pump kJ | miss >20 cm | old PID miss | b·h/J | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| all_n0.2_J0.38_b2.0 | baseline | 0.993 ± 0.890 | 29137.2 | 37224.3 | 100% | 100% | 1.97 | BREAKS |
| all_n0.2_J0.38_b2.0 | payload_x1.2 | 0.866 ± 0.722 | 28566.2 | 36476.6 | 100% | 98% | 1.97 | BREAKS |
| all_n0.5_J0.38_b2.0 | baseline | 0.955 ± 0.892 | 29159.5 | 37254.4 | 100% | 99% | 1.97 | BREAKS |
| all_n0.5_J0.38_b2.0 | payload_x1.2 | 0.836 ± 0.719 | 28615.8 | 36544.7 | 100% | 99% | 1.97 | BREAKS |
| all_n1.0_J0.38_b2.0 | baseline | 0.911 ± 0.894 | 28792.9 | 36786.1 | 100% | 100% | 1.97 | BREAKS |
| all_n1.0_J0.38_b2.0 | payload_x1.2 | 0.792 ± 0.716 | 28627.1 | 36560.4 | 100% | 99% | 1.97 | BREAKS |
| noTauD_n0.2_J0.38_b2.0 | baseline | 0.880 ± 0.922 | 31836.0 | 39977.6 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.2_J0.38_b2.0 | payload_x1.2 | 0.748 ± 0.757 | 30691.5 | 38521.1 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.5_J0.38_b2.0 | baseline | 0.845 ± 0.917 | 31787.7 | 39919.8 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n0.5_J0.38_b2.0 | payload_x1.2 | 0.704 ± 0.753 | 30647.6 | 38469.6 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n1.0_J0.38_b2.0 | baseline | 0.811 ± 0.913 | 31728.4 | 39849.9 | 100% | 100% | 1.97 | BREAKS |
| noTauD_n1.0_J0.38_b2.0 | payload_x1.2 | 0.670 ± 0.745 | 30620.4 | 38441.4 | 100% | 100% | 1.97 | BREAKS |
| all_n0.2_J0.38_b2.0_sub10 | baseline | 0.999 ± 0.036 | 72.1 | 89.8 | 43% | 77% | 0.20 | BREAKS |
| all_n0.2_J0.38_b2.0_sub10 | payload_x1.2 | 1.001 ± 0.031 | 73.5 | 89.8 | 48% | 85% | 0.20 | BREAKS |
| all_n0.5_J0.38_b2.0_sub10 | baseline | 0.998 ± 0.051 | 73.2 | 91.0 | 44% | 77% | 0.20 | BREAKS |
| all_n0.5_J0.38_b2.0_sub10 | payload_x1.2 | 1.000 ± 0.043 | 74.8 | 91.0 | 47% | 85% | 0.20 | BREAKS |
| all_n1.0_J0.38_b2.0_sub10 | baseline | 0.997 ± 0.085 | 76.4 | 94.4 | 45% | 77% | 0.20 | BREAKS |
| all_n1.0_J0.38_b2.0_sub10 | payload_x1.2 | 0.999 ± 0.072 | 78.4 | 94.7 | 47% | 85% | 0.20 | BREAKS |
| noTauD_n0.2_J0.38_b2.0_sub10 | baseline | 1.005 ± 0.041 | 80.9 | 102.5 | 1% | 67% | 0.20 | ok |
| noTauD_n0.2_J0.38_b2.0_sub10 | payload_x1.2 | 1.009 ± 0.036 | 82.8 | 103.6 | 1% | 80% | 0.20 | ok |
| noTauD_n0.5_J0.38_b2.0_sub10 | baseline | 1.004 ± 0.056 | 82.1 | 103.8 | 1% | 67% | 0.20 | ok |
| noTauD_n0.5_J0.38_b2.0_sub10 | payload_x1.2 | 1.008 ± 0.048 | 84.2 | 104.9 | 1% | 80% | 0.20 | ok |
| noTauD_n1.0_J0.38_b2.0_sub10 | baseline | 1.003 ± 0.091 | 85.4 | 107.5 | 3% | 66% | 0.20 | degraded |
| noTauD_n1.0_J0.38_b2.0_sub10 | payload_x1.2 | 1.007 ± 0.077 | 87.9 | 108.8 | 4% | 80% | 0.20 | degraded |
