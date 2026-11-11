# Lab 10: real plant design (10a)

Status: approved 2026-11-11 (four payload variants, command mapping by sign of `u`).
Implemented in `rl/boom_env_real.py`, checked by `rl/10_simscape_check.py`, controller study in
`rl/10_sched_real.py`, results in `rl/10_real_plant.md`. Section 6 is kept as the record of
what was decided.

Goal: a training-free plant `rl/boom_env_real.py` that matches the Lab 08a Simscape model, with
the same observation and action interface as `rl/boom_env.py` (`BoomEnv`).

Status tags as in the repo: OFFICIAL (Doosan/Develon document), ESTIMATED (drawing or
literature, Lab 08a), ASSUMED (model choice of this lab, not DX300 data).

## 1. Equations

State: boom angle `θ` [rad] about foot pin A, angular speed `ω` [rad/s]. One command `u` in
[-1, 1], held for 0.05 s, integrated in 10 substeps of `h` = 0.005 s (semi-implicit Euler).

```
J(m_tip) ω' = τ_cyl − τ_g(θ, m_tip) − τ_f(θ, ω)
θ' = ω
```

**Inertia** (as `stage3_validate.m`):

```
J = J_boom + m_D |p_D|² + m_arm |p_arm|² + m_tip |p_tip|²
J_boom = m_boom (L² + 0.5²)/12 + m_boom Lc²
p_D = (L, 0),  p_arm = (L + ½La cos φ, ½La sin φ),  p_tip = (L + La cos φ, La sin φ)   boom frame
```

With the JSON values (φ = −90°) this gives 261 745.52 kg m², equal to `J_with_front`
(asserted at import).

**Gravity torque** (boom plus three front point masses, as `stage3_validate.m`):

```
τ_g = g [ m_boom Lc cos θ + Σ_i m_i (x_i cos θ − y_i sin θ) ]
```

**Command to torque.** `u` is the normalised pressure of the driven chamber:

```
τ_cmd = u τ_up(θ)     u ≥ 0   (cap side, 330 bar at u = 1)
τ_cmd = u τ_down(θ)   u < 0   (rod side, 330 bar at u = −1)
```

`τ_up`, `τ_down` are interpolated linearly in the 51-point JSON tables.

**Friction** at the rod (`friction_cylinder_eq`), `v = r(θ) ω`:

```
τ_f = r(θ) [ F_c sign(v) + c_v v ]
```

At `ω = 0` the boom sticks while `|τ_cmd − τ_g| ≤ r F_c`. A substep that would cross `ω = 0`
ends at `ω = 0`.

**Moment arm.** `r(θ) = τ_up(θ) / (p_relief A_cap)`, from the JSON table. It differs from
`dx300sim.boom_geometry.moment_arm` by at most 0.50 mm (0.09 %).

**Flow limit** (pump 494 L/min into one chamber):

```
−Q/A_ann ≤ r(θ) ω ≤ Q/A_cap        −0.4957 m/s ≤ v ≤ +0.2674 m/s
```

Per substep: integrate with `τ_cmd`. If the new rod speed is outside the band, set
`ω = v_lim / r(θ)` and compute the torque the cylinder really delivered:

```
τ_cyl = J (ω_new − ω_old)/h + τ_g + τ_f          (on the flow limit)
τ_cyl = τ_cmd                                    (otherwise)
```

`τ_cyl` is clipped to `[−τ_down, τ_up]` (330 bar). If the clip is active the speed limit is
exceeded for that substep and counted (`info["flow_violation"]`); with the numbers below this
should never happen, since `τ_up − τ_g ≥ 220 kN m` everywhere.

Meaning of the flow limit: extending, the cap pressure falls below the commanded pressure
because the pump cannot follow. Retracting under gravity, the cylinder holds the load back
(`τ_cyl > 0`, meter-out) so that the rod side stays filled.

**End stops**: as `BoomEnv`. `θ` is clipped to `[θ_min, θ_max]` and `ω` set to 0.

**Energy**, summed per substep with the delivered torque:

```
E_mech = Σ |τ_cyl ω| h          W_pos = Σ max(τ_cyl ω, 0) h          W_neg = Σ max(−τ_cyl ω, 0) h
E_pump = (W_pos + dp_valve V) / η_pump      V = Σ A |v| h   (A_cap extending, A_ann retracting)
```

`E_pump` is the Lab 09b formula. Label in every table: "lower bound, ASSUMED parameters"
(Lab 09b2: an ideal demand-following pump).

**Interface.** Observation `[θ, ω, target − θ]`, action `Box(−1, 1, (1,))`, `dt` = 0.05 s,
200 steps, reset ranges and reward formula copied from `BoomEnv`. `step()` also returns
`info = {tau_cmd, tau_cyl (mean over the step), flow_lim (substeps on the flow limit),
torque_lim (substeps with |u| = 1 and not on the flow limit), flow_violation}`.

## 2. Parameters

| Parameter | Value | Status | Source |
|---|---|---|---|
| `m_boom` | 2330 kg | OFFICIAL | JSON `m_boom` |
| `L` (boom length) | 6.245 m | OFFICIAL | JSON `L_boom`; asserted equal to `dx300_params.BOOM_LENGTH_OFFICIAL_M` |
| `Lc` | 3.1225 m | ESTIMATED | JSON `Lc` (CoM at mid-boom) |
| `J_boom` | 30 338.6 kg m² | ESTIMATED | JSON `J_boom_only` (brick about A) |
| `m_D` | 340 kg | ESTIMATED lumping, mass OFFICIAL | JSON `front_masses` (arm cylinder at D) |
| `m_arm` | 1270 kg | ESTIMATED lumping, mass OFFICIAL | JSON `front_masses` (arm + bucket cylinder at arm mid) |
| `m_tip` nominal | 3406 kg | ESTIMATED | JSON `front_masses`: bucket 1120 kg (OFFICIAL) + 1.27 m³ × 1800 kg/m³ (density ESTIMATED) |
| `La`, `φ` | 3.1 m, −90° | length OFFICIAL, pose ESTIMATED | JSON `front_masses` (arm fixed perpendicular to boom) |
| `J` nominal | 261 745.5 kg m² | ESTIMATED | computed from the masses; asserted equal to JSON `J_with_front` |
| `τ_up(θ)` | 514–912 kN m | ESTIMATED | JSON `tau_up` (330 bar × A_cap × r, no losses) |
| `τ_down(θ)` | 277–492 kN m | ESTIMATED | JSON `tau_down` (330 bar × A_ann × r); ratio to `τ_up` 0.5395 |
| `r(θ)` | 0.506–0.900 m | ESTIMATED | derived from JSON `tau_up` (pins from the drawing) |
| `F_c`, `c_v` | 1900 N, 2500 N s/m | ESTIMATED | JSON `friction_cylinder_eq` (Li 2020, scaled) |
| `θ_min`, `θ_max` | −47.2°, 60.4° | ESTIMATED | JSON `th_min`, `th_max` |
| `Q_pump` | 494 L/min | OFFICIAL | JSON `Q_pump` |
| `p_relief` | 330 bar | OFFICIAL | JSON `p_relief` |
| bore, rod, cylinders | 0.140 m, 0.095 m, 2 | OFFICIAL | `dx300sim/dx300_params.py` (not in the JSON) |
| `A_cap`, `A_ann` | 0.030788, 0.016611 m² | OFFICIAL-derived | both cylinders |
| `v` limits | +0.2674, −0.4957 m/s | OFFICIAL-derived | `Q_pump / A_cap`, `Q_pump / A_ann` |
| retraction limit as a hard limit | | ASSUMED | no cavitation, no regeneration. The lab08a Simscape model does cavitate on boom-down (peak 0.69 m/s, Lab 09b2) |
| `g` | 9.81 m/s² | | as `boom_env.py` |
| `dt`, substeps | 0.05 s, 10 | ASSUMED | task |
| `dp_valve`, `η_pump` | 20 bar, 0.85 | ASSUMED - not from manual | `rl/09b_hyd_params.py` |

Not used: `b_zeta_candidates` (artificial rod damper of the Stage 5 sensitivity study, not a
physical parameter), `b_friction_rot_mean` (the table `c_v r(θ)²` is used instead),
`J_range_vs_pose` (arm pose is fixed).

Numbers that follow from the table (output of the design script):

```
v_ext_max 0.2674 m/s  v_ret_max 0.4957 m/s  flow-only up time 5.422 s  down 2.925 s
omega limits: up 0.2973..0.5285 rad/s, down 0.5510..0.9795 rad/s
tau_g full: th_min 167.1, theta=0 378.7, max 398.1 at 17.3 deg, th_max 293.9 kN m
margin tau_up - tau_g: min 220.2 kN m at 60.4 deg; hold command u=tau_g/tau_up 0.287..0.572
dEp th_min->th_max full bucket 629.9 kJ
friction: Coulomb torque 961..1709 N m; b=c r^2 640..2023 N m s/rad; placeholder b 6e5
placeholder: tau_g(0) 171.7 kN m, J 8e4, tau_max 5e5; J ratio 3.27, tau_g ratio 2.21
```

Consequence to expect: the real plant has about 1/300 of the placeholder damping. The flow
limit, not friction, bounds the speed.

## 3. Payload step (proposal)

A payload step changes `m_tip` at step 100 (t = 5 s, as `PayloadBoomEnv`). Both `τ_g` and `J`
change, since both depend on `m_tip`.

| Variant | `m_tip` before → after | `τ_g(0)` factor | `J` factor | Meaning |
|---|---|---:|---:|---|
| `baseline` | 3406 kg, constant | 1 | 1 | full bucket, as Simscape |
| `fill_half` | 2263 → 3406 kg | × 1.227 | × 1.269 | half-full to full; closest to the placeholder `payload_x1.2` |
| `fill` | 1120 → 3406 kg | × 1.587 | × 1.738 | empty to full bucket |
| `dump` | 3406 → 1120 kg | × 0.630 | × 0.575 | full bucket emptied |

1120 kg is the empty bucket (OFFICIAL, model unverified). 2263 kg is the bucket with half of
the 2286 kg payload. All payload masses carry the ESTIMATED density of 1800 kg/m³. The
controller always starts from the nominal 3406 kg.

## 4. Check against Simscape (10b)

No parameter is tuned to make these match.

| Check | Python | Simscape reference |
|---|---|---|
| Static holding torque over θ | `τ_g(θ)` | (a) the logged boom-up of Lab 09b2 (`matlab/lab09b2/export/simscape_up.csv`): `τ − J dω/dt` over θ_min..θ_max; (b) its static point at t = 0: 167.115 kN m at θ_min |
| Boom-up time, `u` = 1 from θ_min to θ_max − 0.2° | rollout | 5.619 s (no friction), 5.617 s (friction model), JSON |
| Boom-up mechanical work, same window | `Σ τ_cyl ω h` | 672.3 kJ signed, 679.8 kJ absolute (Lab 09b2, no friction); 672.7 / 681.1 kJ (friction model) |

Known differences before running: Simscape has oil compressibility, a constant-flow pump and
a relief valve, so the boom oscillates and arrives with speed (peak 0.378 m/s). The Python
plant is rigid and flow-limited. Expected boom-up time about 5.4–5.5 s (flow-only 5.42 s),
expected work about ΔE_p + E_kin = 629.9 + 36 kJ.

## 5. Controller on the real plant (10c)

`sched` of Lab 09c3 / 09f with three changes forced by the plant:

1. **Feed-forward** from the real gravity model: `τ_ff = τ_g(θ, m̂_tip)`, `u_ff = τ_ff / k̂`.
2. **Payload estimator** for the tip mass, same filter as Lab 09c (`T_f` = 0.2 s, one step late,
   end-stop gate, clip to `[0, 2 × 3406 kg]`):
   `m_raw = (τ_cyl − J_nom α − τ_f − τ_g(θ, 0)) / (g x_tip(θ))`, with `x_tip = L cos θ + La sin θ`.
   It reads `τ_cyl`, the torque the cylinders deliver (on the machine: from the chamber
   pressures). `J_nom` is the nominal inertia; the bias from a wrong `J` vanishes at rest, as
   in Lab 09c.
3. **Gain scheduling**: `k̂ = τ_cyl / u` as in 09c3, PID terms scaled by `K_ref / k̂` with
   `K_ref = τ_up(0)` = 911.9 kN m. Two additions: one estimate per sign of `u` (`τ_up` and
   `τ_down` differ by 0.54), and no update on a step at the flow limit, where `τ_cyl / u` is
   not the gain. The controller detects the flow limit from the measured `r(θ) ω`.

Tuning: `Kp` × `Kd` over the Lab 07 grid (9 × 5 = 45 sets, `Ki` = 0 as before), nominal real
plant, 20 original scenarios, Lab 07 criterion (lowest mean `E_mech` among the sets with
worst-case error < 20 cm). If no set passes: fewest misses, then lowest `E_mech`, and the
report says so.

Reference run: the same controller with the old gains of Lab 09f (`Kp` 0.55, `Ki` 0, `Kd` 0.25).

Conditions (100 unseen scenarios, every variant of section 3):

| 09c3 condition | On the real plant |
|---|---|
| `none`, `noise_0.2/0.5/1.0`, `tau_noise_5pct` | applies unchanged |
| `J_x0.38`, `J_x1.5` | applies; scales `J(m_tip)` of the plant, controller keeps `J_nom` |
| `b_x0.5`, `b_x2` | applies; scales `F_c` and `c_v`. Friction is small, so little effect is expected |
| `tau_d` | does not apply as a condition: the angle-dependent limit is part of the plant |
| nine combined cases | apply, without the `tau_d` flag |

Metrics per row: miss rate > 20 cm, `E_mech`, `E_pump` (lower bound, ASSUMED parameters), time
to first reach within 20 cm, share of substeps on the flow limit, share on the torque limit.

## 6. Open points for approval

1. Payload variants of section 3 (three step variants plus baseline, or fewer).
2. Command mapping `u × τ_up` / `u × τ_down` by sign (section 1). Alternative: `u × τ_up` for
   both signs, clipped at `−τ_down`.
3. Lab 09b2 is not on `master` (commit 3e1be72, branch `lab09b2-energy-check`, PR #1 open).
   10b needs its Simscape log and its work value.
