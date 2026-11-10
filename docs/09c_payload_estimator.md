# Lab 09c: online payload estimator (derivation)

Status: approved 2026-11-11. Implemented in `rl/lab09_common.py` (`PID`, `payload_aware=True`),
evaluated by `rl/09c_payload_ff.py`, results in `rl/09c_comparison.md`.

## 1. Model

Boom about foot pin A, same model as `rl/boom_env.py`:

```
J·θ'' = F_cyl·r(θ) − m·g·Lc·cos θ − b·θ'
```

- `F_cyl` is the total force of both boom cylinders [N], `r(θ)` the moment arm [m]
  (`moment_arm` in `dx300sim/boom_geometry.py`).
- `m` is the lumped mass at distance `Lc` from A. A payload raises `m`
  (Lab 07 `PayloadBoomEnv`: `m *= 1.2` at step 100).
- In the env the joint torque is `τ = u·tau_max`, so `F_cyl = τ / r(θ)` and
  `F_cyl·r(θ) = τ`. On the machine `F_cyl` comes from the cylinder pressures.

## 2. Moment balance solved for the mass

Everything except `m` is measured or known:

```
m·g·Lc·cos θ = F_cyl·r(θ) − J·θ'' − b·θ'

m_raw = (F_cyl·r(θ) − J_nom·θ'' − b_nom·θ') / (g·Lc·cos θ)
```

`cos θ` never reaches zero: the end stops are −47.2° and 60.4°, so `cos θ ≥ 0.494`.

Payload mass (reported only, lumped at `Lc`): `m_payload = m_hat − m_nom`.
Only the product `m·Lc` is identifiable. The feed-forward uses that same product,
so an error in `Lc` does not hurt the controller.

## 3. Discrete form

The env integrates with semi-implicit Euler (`dt = 0.05 s`):

```
ω[k+1] = ω[k] + (τ[k] − m·g·Lc·cos θ[k] − b·ω[k]) / J · dt
```

So the acceleration over step k is `θ''[k] = (ω[k+1] − ω[k]) / dt`, and

```
m_raw[k] = (τ[k] − J_nom·(ω[k+1] − ω[k])/dt − b_nom·ω[k]) / (g·Lc·cos θ[k])
```

With exact `J`, `b` and no noise this returns the true `m` exactly.
The estimate is available at step k+1, one control step (0.05 s) late.
`ω` and `θ` are what the PID observes; `τ[k]` is the torque applied in step k.

## 4. Low-pass filter

First-order filter, started at the nominal mass:

```
m_hat[0]   = m_nom
m_hat[k+1] = m_hat[k] + α·(m_raw[k] − m_hat[k]),   α = dt / (T_f + dt)
```

Proposed `T_f = 0.2 s`, so `α = 0.2` and a mass step is 95 % tracked in about 0.7 s.
`m_hat` is clipped to `[0, 2·m_nom]`.

Gate: no update on a step that ends on an end stop (`θ[k+1]` equal to `th_min`
or `th_max`). The env sets `ω = 0` there, and the stop reaction is not in the
moment balance.

## 5. Feed-forward

```
Lab 07:   u_ff = m_nom·g·Lc·cos θ / tau_max
Lab 09c:  u_ff = m_hat·g·Lc·cos θ / tau_max
```

PID terms, gains and anti-windup stay as in Lab 07 (`pid`, Ki = 0).

## 6. Why it should remove the payload error

Lab 07 `pid` at rest (Ki = 0): `Kp·e + u_ff = m·g·Lc·cos θ / tau_max`, so

```
e_ss = (m − m_nom)·g·Lc·cos θ / (Kp·tau_max)
```

With `m_hat → m` the right side is zero. This is the expectation; Lab 09c measures it.

## 7. Error sources

| Source | Effect on `m_hat` | At rest (θ' = θ'' = 0) |
|---|---|---|
| `J` wrong by ΔJ | bias `−ΔJ·θ'' / (g·Lc·cos θ)` | zero |
| `b` wrong by Δb | bias `−Δb·θ' / (g·Lc·cos θ)` | zero |
| Noise on ω | amplified by `J_nom / dt` in the `θ''` term | filtered only |
| End stop | reaction force unmodelled | gated out |

At rest the estimate reduces to `m_hat = τ / (g·Lc·cos θ)`, independent of `J` and `b`.
The noise variants of Lab 06b are outside this lab's evaluation (nominal, payload ×1.2,
mass ×0.42).
