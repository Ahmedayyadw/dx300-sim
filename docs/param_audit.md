# Env vs DX300 parameter audit

Comparison of every physical parameter in `rl/boom_env.py` with `dx300sim/dx300_params.py`. `boom_env.py` imports nothing from `dx300_params.py`; the only params value in use is the boom length for the cm conversion in `rl/05_generalization.py`. Derived values come from the params pins and cylinder data via `dx300sim/boom_geometry.py`.

Flags: **env only** means `boom_env.py` has it and params has no counterpart; **params only** means the reverse. Angles are in degrees from horizontal.

| Parameter | boom_env value | dx300_params value | Units | Ratio (env / params) | Used in reward/dynamics? |
|---|---|---|---|---|---|
| Mass `m` / `BOOM_MASS_KG` | 5000 | 2330 (boom only, official) | kg | 2.15 | Dynamics (gravity torque) |
| CoM distance `Lc` | 3.5 | **env only** (L/2 would be 3.12) | m | n/a | Dynamics (gravity torque) |
| Inertia `J` | 8e4 | **env only** (a uniform rod, m·L²/3, would be 3.03e4) | kg·m² | n/a | Dynamics |
| Damping `b` | 6e5 | **env only** | N·m·s/rad | n/a | Dynamics |
| Gravity `g` | 9.81 (inline, `boom_env.py:28`) | **env only** | m/s² | n/a | Dynamics |
| Torque limit `tau_max` | 5e5, constant | **env only** (cylinder data exists, but no pressure, so no limit) | N·m | n/a | Dynamics and kJ (τ = u·tau_max). Not in reward: the reward uses the normalised u. |
| Time step `dt` | 0.05 | **env only** | s | n/a | Dynamics and kJ (Σ\|τω\|·dt) |
| Episode length `max_steps` | 200 (10 s) | **env only** | steps | n/a | Episode length |
| Lower stop `th_min` | -47.2 | -47.13 (derived from `CLOSED_LENGTH_M`, not stored) | deg | 1.00 | Dynamics (stop sets ω = 0) |
| Upper stop `th_max` | 60.4 | 60.44 (derived from `EXTENDED_LENGTH_M`, not stored) | deg | 1.00 | Dynamics (stop sets ω = 0) |
| Reset start angle range | -0.4 to 0.9 | **env only** | rad | n/a | Scenarios |
| Reset target range | -0.3 to 0.9 | **env only** | rad | n/a | Scenarios, reward (err) |
| `BOOM_LENGTH_OFFICIAL_M` | not used | 6.245 | m | n/a | Neither. Used only for the cm conversion in `05`. |
| `BOOM_LENGTH_M` (derived \|AD\|) | not used | 6.245 | m | n/a | Neither |
| `THETA_REF_RAD` | **params only** | -6.3 | deg | n/a | Neither |
| `B_XY_M`, `C_REF_XY_M`, `D_REF_XY_M` | **params only** | (0.571, -0.695), (2.629, 0.850), (6.208, -0.682) | m | n/a | Neither |
| `C_M`, `DELTA_RAD` | **params only** | 2.763, -24.2 | m, deg | n/a | Neither |
| `N_CYL` | **params only** | 2 | – | n/a | Neither |
| `BORE_M`, `ROD_M` | **params only** | 0.140, 0.095 | m | n/a | Neither |
| `STROKE_M`, `CLOSED_LENGTH_M`, `EXTENDED_LENGTH_M` | **params only** | 1.450, 2.010, 3.460 | m | n/a | Neither (they only set the derived stops above) |
| `CYL_MASS_KG` | **params only** | 260 each | kg | n/a | Neither |

The reward itself, `-(err² + 0.01·ω² + w·u²)`, uses no physical parameter. The physics only reach it through how θ and ω evolve.

## Mismatches that would most change the kJ and tip-error results

- **Mass and CoM (affect kJ most).** Gravity torque m·g·Lc is 171.7 kN·m in the env. With the boom-only mass at L/2 it would be 71.4 kN·m, 0.42× as much. Work to lift the boom would fall by about that factor, so all kJ values (127 vs 148) would shift. The 14% gap between weights could change too. Neither file has the arm, bucket or payload mass that the real boom carries.
- **Torque limit.** The env's 5e5 N·m is the same at every angle. The real cylinder moment arm changes from 0.90 m at θ = 0° to 0.51 m at the top stop, which means 56% of the torque at the top. Lowering also pushes on the smaller rod-side area. A real limit would saturate the command more often on high targets (the target range reaches 0.9 rad, 51.6°). That would likely raise worst-case tip error and the share of moves over 20 cm, and change kJ through τ = u·tau_max.
- **Inertia and damping.** The rod estimate for J is 0.38× the env value, so the boom would respond faster and could overshoot more. J/b would fall from 0.133 s to 0.050 s, the same as `dt`, so the explicit Euler step loses most of its stability margin. That mainly affects tip error (mean and p95). There's no DX300 value for b at all, so any real-value run first needs a damping and friction estimate.
