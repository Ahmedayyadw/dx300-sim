# Lab 09c: old PID vs payload-aware PID

Both controllers: Kp=0.5 Ki=0 Kd=0.25 (Lab 07 `pid`). Estimator filter T_f = 0.2 s. 100 unseen scenarios (RNG seed 123), 10 s each.
E_pump uses dp_valve = 20 bar and eta_pump = 0.85, both ASSUMED - not from manual (`rl/09b_hyd_params.py`).
Miss rate = share of moves with final tip error over 20 cm. Source: `rl/09c_payload_ff.csv`.

| variant | controller | mean deg (cm) | max deg (cm) | E_mech kJ | E_pump kJ | miss rate >20 cm | m_hat / m |
|---|---|---|---|---:|---:|---:|---:|
| baseline | old PID (Lab 07) | 0.68 (7.4) | 2.06 (22.5) | 68.5 | 80.3 | 1% | n/a |
| baseline | payload-aware PID | 0.68 (7.4) | 2.06 (22.5) | 68.5 | 80.3 | 1% | 1.000 |
| payload_x1.2 | old PID (Lab 07) | 6.18 (67.3) | 7.48 (81.6) | 78.2 | 75.9 | 100% | n/a |
| payload_x1.2 | payload-aware PID | 0.70 (7.6) | 2.17 (23.6) | 70.8 | 81.7 | 1% | 1.000 |
| m_x0.42 | old PID (Lab 07) | 18.05 (196.7) | 23.11 (251.9) | 70.8 | 108.3 | 100% | n/a |
| m_x0.42 | payload-aware PID | 0.67 (7.3) | 2.00 (21.8) | 39.9 | 61.6 | 1% | 1.000 |
