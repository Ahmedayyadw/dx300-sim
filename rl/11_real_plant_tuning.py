"""Lab 11: energy floor, time-limited tuning and the dump failure on the real plant.

No RL, no training. Plant: rl/boom_env_real.py (Lab 10a, unchanged). Controller: SchedReal of
rl/10_sched_real.py (loaded, not modified); Sched11 below only adds estimator options and is
bit-identical to SchedReal with the default options (--step check proves it).

11a  Energy floor of one move theta0 -> target with no overshoot (monotone, rest to rest):
       raise:  floor = dPE(m_tip) + F_c ds                       viscous friction -> 0 (slow limit)
       lower:  floor = |dPE(m_tip)| - (F_c + c_v v_ret_max) ds   friction helps the brake
     dPE = integral of |tau_g| over the angle (tau_g > 0 on the whole range, asserted),
     ds = integral of r(theta) dtheta = rod travel, m_tip = tip mass at the start of the move.
     On a lowering move the cylinder absorbs |dPE| minus what friction dissipates, so friction
     lowers E_mech there; v_ret_max (flow limit) gives the largest friction help, which keeps the
     floor a lower bound of a monotone move.
     Split of E_mech = sum |tau_cyl omega| h, per 0.05 s control step, by the angle the step covers:
       direct      first pass over the angle between start and target
       overshoot   first outward and first return pass over the angle beyond the target
       oscillation every repeat pass, and any travel behind the start
     E_mech = direct + overshoot + oscillation exactly, and
     E_mech - floor = overshoot + oscillation + direct_excess - shortfall
       direct_excess = direct - floor of the angle really covered (reverse-torque braking, viscous
                       friction, kinetic energy carried past the target)
       shortfall     = floor of the angle between the furthest point reached and the target
     Resolution 0.05 s: a reversal inside one control step is attributed by net travel (ESTIMATED).

11b  Kp x Kd grid (Ki = 0) on the 20 original scenarios, nominal plant. Pareto front of mean E_mech
     against mean time to reach 20 cm over the gain sets with worst-case error < 20 cm.

11c  Dump failure: diagnosis and estimator fixes (clamp, slower / adaptive filter, acceleration
     gate). Fix parameters are chosen on the 20 original scenarios only.

The 100 unseen scenarios are read only after rl/11_points.json holds "approved": true.
E_pump is the Lab 09b formula: lower bound, ASSUMED parameters.

Usage (repo root, .venv active): python rl/11_real_plant_tuning.py --step STEP\n  check     this file against Lab 10 (no file written)\n  a         11a floor on the 20 original scenarios\n  b         11b grid, Pareto front, proposed points, figure\n  diag      11c diagnosis (tuning set)\n  fixtune   11c fix sweep (tuning set; reuses finished settings)\n  fixcheck  11c spot checks (tuning set)\n  eval      unseen scenarios; needs the approved rl/11_points.json\nReport: python rl/11_report.py
"""
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L10 = _load("lab10_sched_real", "10_sched_real.py")     # imports only; its main is guarded
R = L10.R
HYD, cm = L10.HYD, L10.cm

FLOOR_CSV = "rl/11_floor_moves.csv"
GRID_CSV = "rl/11_gain_grid.csv"
FRONT_CSV = "rl/11_pareto_front.csv"
POINTS_JSON = "rl/11_points.json"
L10_GRID_CSV = "rl/10_tuning_grid.csv"                  # read only

M_FULL, M_EMPTY = L10.M_FULL, L10.M_EMPTY
REACH_M, TAIL_STEPS = L10.REACH_M, L10.TAIL_STEPS
KI = 0.0                                                # as Lab 10
L10_TUNED, L10_OLD = (0.25, 0.0, 0.5), L10.OLD_GAINS

# 11b grid (ASSUMED choice): the Lab 07 values plus 1-1.2-1.5-2-2.5-3-4-5-6-8 steps per decade
KP_FINE = [0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25,
           30, 40, 50]
KD_FINE = [0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]

for _m in (M_EMPTY, M_FULL):                            # tau_g is linear in m_tip
    assert min(R.gravity_torque(t, _m) for t in np.linspace(R.TH_MIN, R.TH_MAX, 2001)) > 0.0


# ----------------------------------------------------------------------------- 11a: floor
def rod_travel(th_a, th_b):
    """Integral of r(theta) dtheta between two angles [m], exact for the piecewise-linear table."""
    lo, hi = min(th_a, th_b), max(th_a, th_b)
    knots = [lo] + [t for t in R._TH if lo < t < hi] + [hi]
    return sum(0.5 * (R.moment_arm(a) + R.moment_arm(b)) * (b - a) for a, b in zip(knots, knots[1:]))


def floor_terms(th0, th1, m_tip, fric_scale=1.0):
    """(gravity term, friction term) [J] of the monotone move th0 -> th1. Floor = sum."""
    grav = abs(R.potential_energy(th1, m_tip) - R.potential_energy(th0, m_tip))
    ds = rod_travel(th0, th1)
    if th1 >= th0:
        return grav, fric_scale * R.F_COULOMB_N * ds
    return grav, -fric_scale * (R.F_COULOMB_N + R.C_VISC_NSM * R.V_RET_MAX) * ds


def floor_J(th0, th1, m_tip, fric_scale=1.0):
    return sum(floor_terms(th0, th1, m_tip, fric_scale))


class Split:
    """Attribution of E_mech to direct, overshoot and oscillation travel (module docstring)."""

    def __init__(self, th0, target):
        self.th0, self.sgn, self.D = th0, (1.0 if target >= th0 else -1.0), abs(target - th0)
        self.p, self.f, self.ret = 0.0, self.D, []      # near frontier, far frontier, returned set
        self.direct = self.over = self.osc = 0.0

    def _first_return(self, x0, x1):
        """Length of [x0, x1] not yet returned over; then mark it returned."""
        covered = sum(max(0.0, min(x1, b) - max(x0, a)) for a, b in self.ret)
        touch = [(a, b) for a, b in self.ret if not (b < x0 or a > x1)]
        self.ret = [(a, b) for a, b in self.ret if b < x0 or a > x1]
        self.ret.append((min([x0] + [a for a, _ in touch]), max([x1] + [b for _, b in touch])))
        return (x1 - x0) - covered

    def add(self, th_a, th_b, d_e):
        if d_e <= 0.0:
            return
        a, b = self.sgn * (th_a - self.th0), self.sgn * (th_b - self.th0)
        if a == b:
            self.osc += d_e                               # work without net travel
            return
        lo, hi = min(a, b), max(a, b)
        rate, d, o = d_e / (hi - lo), 0.0, 0.0
        if b > a:                                         # towards / past the target
            d = max(0.0, min(hi, self.D) - max(lo, self.p))
            o = max(0.0, hi - max(lo, self.f))
            self.p, self.f = max(self.p, min(hi, self.D)), max(self.f, hi)
        elif hi > self.D:                                 # coming back from beyond the target
            o = self._first_return(max(lo, self.D), hi)
        self.direct += rate * d
        self.over += rate * o
        self.osc += rate * (hi - lo - d - o)


# ----------------------------------------------------------------------------- controller
EST_BASE = {"name": "base", "clamp": None, "t_f": L10.T_F_S, "adapt": None, "acc_gate": None,
            "oracle": False, "clean": False}


class Sched11(L10.SchedReal):
    """SchedReal with estimator options. Defaults reproduce SchedReal bit for bit.

    clamp     (lo, hi) [kg] limits of m_hat instead of [0, 2 x 3406]
    t_f       time constant of the m_hat filter [s] (the gain-schedule filter keeps 0.2 s)
    adapt     (c, n_hold): decaying-gain filter restarted by a change detector. A fast filter
              (0.2 s) runs in parallel; when |m_fast - m_hat| > c x (estimated std of m_fast) for
              n_hold accepted samples, m_hat jumps to m_fast and the gain restarts at the fast
              value and decays as 1 / n down to the slow value dt / (t_f + dt).
    acc_gate  gamma: m_hat is updated only when |J_nom dw/dt| <= gamma g x_tip (3406 - 1120 kg)
    oracle    m_hat = true tip mass (diagnosis only);  clean: estimator reads the true state
              (diagnosis only)
    """

    def __init__(self, gains, est=None, tau_noise=0.0):
        self.est = {**EST_BASE, **(est or {})}
        self.m_lo, self.m_hi = self.est["clamp"] or (0.0, 2.0 * M_FULL)
        super().__init__(gains, t_f=L10.T_F_S, tau_noise=tau_noise)
        self.alpha_fast = self.alpha
        self.alpha_m = self.dt / (self.est["t_f"] + self.dt)

    def reset(self):
        super().reset()
        self.m_fast, self.var, self.n_acc, self.n_det = M_FULL, 0.0, 0, 0
        self.n_upd = self.n_lo = self.n_hi = self.n_jump = 0

    def update(self, obs, tau, th_min, th_max):
        if self.tau_noise > 0:                                     # one reading feeds both estimates
            tau = tau * (1.0 + self.tau_noise * self.tau_rng.normal())
        th1, om1 = float(obs[0]), float(obs[1])
        th0, om0 = self.prev
        r = R.moment_arm(th0)
        v = r * om1                                                # measured rod speed
        on_flow = v >= L10.FLOW_GATE * R.V_EXT_MAX or v <= -L10.FLOW_GATE * R.V_RET_MAX
        if abs(self.u_prev) >= L10.U_MIN and not on_flow:
            k_raw = tau / self.u_prev
            if self.u_prev > 0:
                self.k_up = float(np.clip(self.k_up + self.alpha * (k_raw - self.k_up),
                                          0.2 * L10.K_REF, 2.0 * L10.K_REF))
            else:
                self.k_down = float(np.clip(self.k_down + self.alpha * (k_raw - self.k_down),
                                            0.2 * L10.K_DOWN_REF, 2.0 * L10.K_DOWN_REF))
        if not (th1 <= th_min + 1e-6 or th1 >= th_max - 1e-6):     # end-stop gate
            tau_f = r * (R.F_COULOMB_N * np.sign(om0) + R.C_VISC_NSM * r * om0)   # nominal friction
            gx = R.G * R.x_tip(th0)
            m_raw = (tau - R.J_NOM * (om1 - om0) / self.dt - tau_f - R.gravity_torque(th0, 0.0)) / gx
            gate = self.est["acc_gate"]
            if gate is None or abs(R.J_NOM * (om1 - om0) / self.dt) <= gate * gx * (M_FULL - M_EMPTY):
                self.n_upd += 1
                if self.est["adapt"] is None:
                    new = self.m_hat + self.alpha_m * (m_raw - self.m_hat)
                else:
                    c, n_hold = self.est["adapt"]
                    innov = m_raw - self.m_fast
                    self.var += self.alpha_m * (innov * innov - self.var)
                    self.m_fast += self.alpha_fast * innov
                    thr = c * math.sqrt(self.var * self.alpha_fast / (2.0 - self.alpha_fast))
                    self.n_det = self.n_det + 1 if abs(self.m_fast - self.m_hat) > thr else 0
                    if self.n_det >= n_hold:
                        new, self.n_acc, self.n_det = self.m_fast, 0, 0
                        self.n_jump += 1
                    else:
                        a = max(self.alpha_m, 1.0 / (self.n_acc + 1.0 / self.alpha_fast))
                        new = self.m_hat + a * (m_raw - self.m_hat)
                        self.n_acc += 1
                self.n_lo += new <= self.m_lo
                self.n_hi += new >= self.m_hi
                self.m_hat = float(np.clip(new, self.m_lo, self.m_hi))
        self.hist.append(self.m_hat)


# ----------------------------------------------------------------------------- rollout, rows
def _approved():
    if not os.path.exists(POINTS_JSON):
        return False
    with open(POINTS_JSON) as f:
        return bool(json.load(f).get("approved"))


def scenarios(set_name):
    if set_name == "new100" and not _approved():
        raise RuntimeError(f"unseen scenarios are locked until {POINTS_JSON} is approved")
    return L10.SCEN[set_name]


def rollout(ctrl, env, th0, target):
    """One 10 s move; same call order as rollout of Lab 10, plus the 11a split."""
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs = env._obs()
    ctrl.reset()
    sp = Split(th0, target)
    oracle, clean = ctrl.est["oracle"], ctrl.est["clean"]
    t_reach = 0.0 if abs(target - th0) * R.L_BOOM_M <= REACH_M else float("nan")
    for k in range(env.max_steps):
        if oracle:
            ctrl.m_hat = env.payload[1] if env.payload and env.k >= env.payload[0] else env.m_tip0
        u = ctrl.act(obs)
        th_a, om_a, e_a = env.th, env.om, env.energy_J
        obs, _, _, trunc, info = env.step(np.array([u]))
        if clean:
            ctrl.prev = (th_a, om_a)
            ctrl.update(np.array([env.th, env.om, env.target - env.th]), info["tau_cyl"],
                        env.th_min, env.th_max)
        else:
            ctrl.update(obs, info["tau_cyl"], env.th_min, env.th_max)
        sp.add(th_a, env.th, env.energy_J - e_a)
        if math.isnan(t_reach) and abs(env.target - env.th) * R.L_BOOM_M <= REACH_M:
            t_reach = (k + 1) * env.dt
        if trunc:
            break
    assert abs(sp.direct + sp.over + sp.osc - env.energy_J) <= 1e-6 * max(1.0, env.energy_J)
    grav, fric = floor_terms(th0, target, env.m_tip0, env.fric_scale)
    th_p = th0 + sp.sgn * sp.p                                     # furthest point of the direct pass
    floor_p = floor_J(th0, th_p, env.m_tip0, env.fric_scale)
    tail = np.array(ctrl.hist[-TAIL_STEPS:])
    return {"th0_deg": math.degrees(th0), "target_deg": math.degrees(target),
            "err_deg": abs(np.degrees(env.target - env.th)),
            "err_signed_deg": float(np.degrees(env.target - env.th)),
            "E_mech_kJ": env.energy_J / 1e3,
            "W_pos_kJ": env.W_pos_J / 1e3,
            "E_pump_kJ": (env.W_pos_J + HYD.DP_VALVE_PA * env.vol_m3) / HYD.ETA_PUMP / 1e3,
            "t_reach_s": t_reach, "flow": env.n_flow / env.n_sub, "torque": env.n_torque / env.n_sub,
            "violations": env.n_violation,
            "floor_grav_kJ": grav / 1e3, "floor_fric_kJ": fric / 1e3, "floor_kJ": (grav + fric) / 1e3,
            "overshoot_kJ": sp.over / 1e3, "oscillation_kJ": sp.osc / 1e3,
            "direct_excess_kJ": (sp.direct - floor_p) / 1e3,
            "shortfall_kJ": (grav + fric - floor_p) / 1e3,
            "peak_overshoot_cm": (sp.f - sp.D) * R.L_BOOM_M * 100,
            "tail": tail / env.m_tip, "tail_kg": tail,
            "upd": ctrl.n_upd / max(1, len(ctrl.hist)), "lo": ctrl.n_lo / max(1, ctrl.n_upd),
            "hi": ctrl.n_hi / max(1, ctrl.n_upd)}


SUM_KEYS = ("floor_kJ", "overshoot_kJ", "oscillation_kJ", "direct_excess_kJ", "shortfall_kJ")


def run_row(job):
    """One row: a controller on one condition, variant, scenario set and noise seed. Worker."""
    label, gains, est, set_name, group, name, cond, variant, seed, keep_moves = job
    env = L10.make_env(variant, **cond)
    ctrl = Sched11(gains, est, tau_noise=cond.get("tau_noise", 0.0))
    if seed:                                                       # seed 0 = the Lab 10 streams
        if cond.get("sigma", 0.0) > 0:
            env.noise_rng = np.random.default_rng(2 * seed)
        ctrl.tau_rng = np.random.default_rng(2 * seed + 1)
    rs = [rollout(ctrl, env, *sc) for sc in scenarios(set_name)]
    errs = np.array([r["err_deg"] for r in rs])
    reach = np.array([r["t_reach_s"] for r in rs])
    tail = np.concatenate([r["tail"] for r in rs])
    tail_kg = np.concatenate([r["tail_kg"] for r in rs])
    m0, payload = L10.VARIANTS[variant]
    e_mean = float(np.mean([r["E_mech_kJ"] for r in rs]))
    row = {"set": set_name, "n_scen": len(rs), "noise_seed": seed, "group": group, "condition": name,
           "sigma_th_deg": cond.get("sigma", 0.0), "tau_noise": cond.get("tau_noise", 0.0),
           "J_factor": cond.get("j", 1.0), "b_factor": cond.get("b", 1.0), "variant": variant,
           "m_tip_start_kg": m0, "m_tip_end_kg": payload[1] if payload else m0,
           "controller": label, "kp": gains[0], "ki": gains[1], "kd": gains[2],
           "estimator": ctrl.est["name"], "t_f_s": ctrl.est["t_f"],
           "mean_deg": errs.mean(), "max_deg": errs.max(), "max_cm": float(cm(errs.max())),
           "mean_err_signed_deg": float(np.mean([r["err_signed_deg"] for r in rs])),
           "share_over_20cm": float((cm(errs) > 20).mean()),
           "mean_E_mech_kJ": e_mean,
           "mean_E_pump_kJ": float(np.mean([r["E_pump_kJ"] for r in rs])),
           "mean_t_reach_s": float(np.nanmean(reach)) if not np.all(np.isnan(reach)) else float("nan"),
           "max_t_reach_s": float(np.nanmax(reach)) if not np.all(np.isnan(reach)) else float("nan"),
           "n_never_reached": int(np.isnan(reach).sum()),
           "share_flow_limit": float(np.mean([r["flow"] for r in rs])),
           "share_torque_limit": float(np.mean([r["torque"] for r in rs])),
           "flow_violations": int(sum(r["violations"] for r in rs)),
           "m_hat_over_m_mean": float(tail.mean()), "m_hat_over_m_std": float(tail.std()),
           "m_hat_tail_mean_kg": float(tail_kg.mean()), "m_hat_tail_std_kg": float(tail_kg.std()),
           "share_est_updates": float(np.mean([r["upd"] for r in rs])),
           "share_at_lower_limit": float(np.mean([r["lo"] for r in rs])),
           "share_at_upper_limit": float(np.mean([r["hi"] for r in rs])),
           "max_peak_overshoot_cm": float(max(r["peak_overshoot_cm"] for r in rs))}
    for key in SUM_KEYS:
        row["mean_" + key] = float(np.mean([r[key] for r in rs]))
    row["pct_above_floor"] = 100.0 * (e_mean / row["mean_floor_kJ"] - 1.0)
    if keep_moves:
        drop = ("tail", "tail_kg", "upd", "lo", "hi")
        row["moves"] = [{"move": i + 1, **{k: v for k, v in r.items() if k not in drop}}
                        for i, r in enumerate(rs)]
    return row


def job(label, gains, variant="baseline", set_name="orig20", group="tuning", name="none", cond=None,
        est=None, seed=0, keep_moves=False):
    return (label, gains, est, set_name, group, name, cond or {}, variant, seed, keep_moves)


def pool_map(jobs):
    with mp.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
        return pool.map(run_row, jobs, chunksize=1)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


read_csv = L10.read_csv


# ----------------------------------------------------------------------------- check
def step_check():
    """Sched11 + rollout of this file against Lab 10: the tuning grid CSV and SchedReal itself."""
    ref = read_csv(L10_GRID_CSV)
    rows = pool_map([job("grid", (r["kp"], r["ki"], r["kd"])) for r in ref])
    worst = {k: max(abs(a[k] - b[k]) for a, b in zip(rows, ref))
             for k in ("mean_E_mech_kJ", "max_deg", "mean_t_reach_s", "mean_E_pump_kJ", "m_hat_over_m_std")}
    print(f"check 1: {len(ref)} gain sets of {L10_GRID_CSV}, largest absolute difference: {worst}")
    assert max(worst.values()) < 1e-9
    n = 0
    for variant in L10.VARIANTS:                     # noisy rows: Sched11 against SchedReal, orig20
        for name, cond in L10.SINGLES + L10.COMBINED:
            mine = run_row(job("x", L10_TUNED, variant, name=name, cond=cond))
            env = L10.make_env(variant, **cond)
            ctrl = L10.SchedReal(L10_TUNED, tau_noise=cond.get("tau_noise", 0.0))
            rs = [L10.rollout(ctrl, env, *sc) for sc in L10.SCEN["orig20"]]
            e = float(np.mean([r["E_mech_kJ"] for r in rs]))
            d = float(np.max([r["err_deg"] for r in rs]))
            assert mine["mean_E_mech_kJ"] == e and mine["max_deg"] == d, (variant, name)
            n += 1
    print(f"check 2: Sched11 (default options) equals SchedReal of Lab 10 exactly on {n} rows"
          f" (18 conditions x 4 variants, 20 original scenarios, Kp 0.25 Kd 0.5)")


# ----------------------------------------------------------------------------- 11a
A_CTRL = [("L10 tuned Kp0.25 Kd0.5", L10_TUNED), ("L10 old Kp0.55 Kd0.25", L10_OLD),
          ("Kp1 Kd1", (1.0, KI, 1.0)), ("Kp5 Kd2", (5.0, KI, 2.0))]


def a_controllers():
    """Before approval: the gain sets named in the task. After: Lab 10 pair plus the approved points."""
    if _approved():
        return [("L10_tuned", L10_TUNED), ("L10_old", L10_OLD)] + load_points()
    return A_CTRL


def step_a():
    global A_CTRL
    A_CTRL = a_controllers()
    jobs = [job(label, gains, variant, keep_moves=True) for label, gains in A_CTRL for variant in L10.VARIANTS]
    rows = pool_map(jobs)
    idx = {(r["controller"], r["variant"]): r for r in rows}
    moves = [{"set": r["set"], "variant": r["variant"], "controller": r["controller"], "kp": r["kp"],
              "kd": r["kd"], **m} for r in rows for m in r["moves"]]
    write_csv(FLOOR_CSV, moves)
    base = [idx[(label, "baseline")] for label, _ in A_CTRL]
    print("\n11a floor, 20 original scenarios, nominal plant, variant baseline (tip mass 3406 kg)")
    print("floor = gravity + friction [kJ]; then E_mech [kJ] (percent above floor) per controller")
    print("| move | start deg | target deg | gravity | friction | floor | "
          + " | ".join(label for label, _ in A_CTRL) + " |")
    print("|---:|---:|---:|---:|---:|---:|" + "---:|" * len(A_CTRL))
    for i in range(len(base[0]["moves"])):
        m = base[0]["moves"][i]
        cells = " | ".join(f"{r['moves'][i]['E_mech_kJ']:.1f} ({100 * (r['moves'][i]['E_mech_kJ'] / m['floor_kJ'] - 1):+.0f} %)"
                           for r in base)
        print(f"| {m['move']} | {m['th0_deg']:.1f} | {m['target_deg']:.1f} | {m['floor_grav_kJ']:.1f} |"
              f" {m['floor_fric_kJ']:+.2f} | {m['floor_kJ']:.1f} | {cells} |")
    print("| mean | | | {:.1f} | {:+.2f} | {:.1f} | ".format(
        np.mean([m["floor_grav_kJ"] for m in base[0]["moves"]]),
        np.mean([m["floor_fric_kJ"] for m in base[0]["moves"]]), base[0]["mean_floor_kJ"])
        + " | ".join(f"{r['mean_E_mech_kJ']:.1f} ({r['pct_above_floor']:+.1f} %)" for r in base) + " |")
    print("\n11a split of the excess, mean over the 20 moves [kJ]")
    print("| controller | variant | floor | E_mech | above floor % | excess | overshoot | oscillation |"
          " direct excess | shortfall | reach 20 cm s | worst error cm |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _ in A_CTRL:
        for variant in L10.VARIANTS:
            r = idx[(label, variant)]
            ex = r["mean_E_mech_kJ"] - r["mean_floor_kJ"]
            closure = r["mean_overshoot_kJ"] + r["mean_oscillation_kJ"] + r["mean_direct_excess_kJ"] - r["mean_shortfall_kJ"]
            assert abs(ex - closure) < 1e-6
            print(f"| {label} | {variant} | {r['mean_floor_kJ']:.1f} | {r['mean_E_mech_kJ']:.1f} |"
                  f" {r['pct_above_floor']:+.1f} | {ex:+.1f} | {r['mean_overshoot_kJ']:.1f} |"
                  f" {r['mean_oscillation_kJ']:.1f} | {r['mean_direct_excess_kJ']:+.1f} |"
                  f" {-r['mean_shortfall_kJ']:+.1f} | {r['mean_t_reach_s']:.2f} | {r['max_cm']:.1f} |")
    print("excess = overshoot + oscillation + direct excess - shortfall holds on every row (asserted)")


# ----------------------------------------------------------------------------- 11b
def pareto(rows):
    """Non-dominated rows in (mean_t_reach_s, mean_E_mech_kJ), sorted by time."""
    front = []
    for r in sorted(rows, key=lambda r: (r["mean_t_reach_s"], r["mean_E_mech_kJ"])):
        if not front or r["mean_E_mech_kJ"] < front[-1]["mean_E_mech_kJ"] - 1e-12:
            front.append(r)
    return front


def step_b():
    t0 = time.time()
    grid = pool_map([job("grid", (kp, KI, kd)) for kp in KP_FINE for kd in KD_FINE])
    print(f"11b grid: {len(grid)} gain sets x {grid[0]['n_scen']} original scenarios in {time.time() - t0:.0f} s")
    write_csv(GRID_CSV, grid)
    ok = [g for g in grid if g["max_cm"] < 20]
    front = pareto(ok)
    print(f"{len(ok)} of {len(grid)} gain sets have worst-case error < 20 cm; {len(front)} on the Pareto front")
    edge = [g for g in front if g["kp"] in (KP_FINE[0], KP_FINE[-1]) or g["kd"] in (KD_FINE[0], KD_FINE[-1])]
    print("front points on the edge of the grid: "
          + (", ".join(f"Kp {g['kp']:g} Kd {g['kd']:g}" for g in edge) or "none"))
    # robustness of the front points on the tuning set (orig20): sensor noise 1.0 deg, 4 variants
    noisy = pool_map([job("front", (g["kp"], KI, g["kd"]), variant, name="noise_1.0", cond={"sigma": 1.0})
                      for g in front for variant in L10.VARIANTS])
    nidx = {(r["kp"], r["kd"], r["variant"]): r for r in noisy}
    for g in front:
        rs = [nidx[(g["kp"], g["kd"], v)] for v in L10.VARIANTS]
        g["n1_baseline_E_mech_kJ"] = nidx[(g["kp"], g["kd"], "baseline")]["mean_E_mech_kJ"]
        g["n1_worst_miss_share"] = max(r["share_over_20cm"] for r in rs)
        g["n1_dump_miss_share"] = nidx[(g["kp"], g["kd"], "dump")]["share_over_20cm"]
    write_csv(FRONT_CSV, front)
    return grid, front


def print_front(front):
    print("\n11b Pareto front (20 original scenarios, nominal plant; noise columns: orig20, sensor noise 1.0 deg)")
    print("| # | Kp | Kd | reach 20 cm s (mean) | (max) | E_mech kJ | above floor % | E_pump kJ (lower bound, ASSUMED parameters) |"
          " worst error cm | peak overshoot cm | flow limit % | noise: E_mech kJ | noise: worst miss % (4 variants) | noise: dump miss % |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, g in enumerate(front):
        print(f"| {i + 1} | {g['kp']:g} | {g['kd']:g} | {g['mean_t_reach_s']:.2f} | {g['max_t_reach_s']:.2f} |"
              f" {g['mean_E_mech_kJ']:.1f} | {g['pct_above_floor']:+.1f} | {g['mean_E_pump_kJ']:.1f} |"
              f" {g['max_cm']:.1f} | {g['max_peak_overshoot_cm']:.1f} | {g['share_flow_limit'] * 100:.1f} |"
              f" {g['n1_baseline_E_mech_kJ']:.1f} | {g['n1_worst_miss_share'] * 100:.0f} | {g['n1_dump_miss_share'] * 100:.0f} |")


FAST_BAND_S = 0.10          # two control steps (ASSUMED): front points this close count as equally fast
OVERSHOOT_MAX_CM = 20.0     # the accuracy bound, applied to the transient (ASSUMED)
ROLES = ("fast", "balanced", "economical")


def pick_points(front):
    """Rule of 11b, tuning set only. Returns ({role: front row}, admissible rows)."""
    adm = [g for g in front if g["pct_above_floor"] >= 0.0 and g["max_peak_overshoot_cm"] < OVERSHOOT_MAX_CM]
    eco = min(adm, key=lambda g: g["mean_E_mech_kJ"])
    t_min = min(g["mean_t_reach_s"] for g in adm)
    fast = min((g for g in adm if g["mean_t_reach_s"] <= t_min + FAST_BAND_S), key=lambda g: g["mean_E_mech_kJ"])
    dt, de = eco["mean_t_reach_s"] - fast["mean_t_reach_s"], fast["mean_E_mech_kJ"] - eco["mean_E_mech_kJ"]
    mid = [g for g in adm if fast["mean_t_reach_s"] < g["mean_t_reach_s"] < eco["mean_t_reach_s"]]
    bal = min(mid, key=lambda g: math.hypot((g["mean_t_reach_s"] - fast["mean_t_reach_s"]) / dt,
                                            (g["mean_E_mech_kJ"] - eco["mean_E_mech_kJ"]) / de))
    return {"fast": fast, "balanced": bal, "economical": eco}, adm


FIG_PATH = "rl/11_pareto.png"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ROLE_STYLE = {"fast": ("#2a78d6", "o"), "balanced": ("#eb6834", "s"), "economical": ("#1baf7a", "D")}


def make_figure(grid, front, points):
    """Pareto figure: mean E_mech against mean time to reach 20 cm, 20 original scenarios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [g for g in grid if g["max_cm"] < 20]
    floor = front[0]["mean_floor_kJ"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y_lo, y_hi, x_lo, x_hi = 157.0, 206.0, 0.9, 6.6
    ax.axhspan(y_lo, floor, color="#efeeea", zorder=0)
    ax.axhline(floor, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(x_hi - 0.08, floor + 0.7, f"energy floor {floor:.1f} kJ", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.text(x_lo + 0.08, y_lo + 0.8, "below the floor: the move stops short of the target", ha="left",
            va="bottom", color=INK2, fontsize=9)
    ax.scatter([g["mean_t_reach_s"] for g in ok], [g["mean_E_mech_kJ"] for g in ok], s=10, color="#b9b8b2",
               linewidths=0, zorder=2, label=f"gain sets with worst-case error < 20 cm ({len(ok)})")
    ax.plot([g["mean_t_reach_s"] for g in front], [g["mean_E_mech_kJ"] for g in front], color=INK, lw=2.0,
            marker="o", ms=3.5, zorder=3, label=f"Pareto front ({len(front)} gain sets)")
    offsets = {"fast": (0.22, 4.0), "balanced": (0.30, 5.5), "economical": (0.20, 9.0)}
    for role in ROLES:
        g, (color, marker) = points[role], ROLE_STYLE[role]
        ax.scatter([g["mean_t_reach_s"]], [g["mean_E_mech_kJ"]], s=110, color=color, marker=marker,
                   edgecolors=SURFACE, linewidths=2.0, zorder=5, label=role)
        dx, dy = offsets[role]
        ax.annotate(f"{role}: Kp {g['kp']:g}, Kd {g['kd']:g}\n{g['mean_t_reach_s']:.2f} s, {g['mean_E_mech_kJ']:.1f} kJ"
                    f" ({g['pct_above_floor']:+.1f} %)",
                    (g["mean_t_reach_s"], g["mean_E_mech_kJ"]), (g["mean_t_reach_s"] + dx, g["mean_E_mech_kJ"] + dy),
                    color=INK, fontsize=9, va="bottom", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
                    bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": "none", "alpha": 0.9}, zorder=6)
    hidden = sum(g["mean_E_mech_kJ"] > y_hi or g["mean_t_reach_s"] > x_hi for g in ok)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("mean time to come within 20 cm of the target [s]", color=INK2)
    ax.set_ylabel("mean E_mech per move [kJ]", color=INK2)
    ax.set_title("Lab 11b: energy against move time, sched controller on the real plant\n"
                 "20 original scenarios, nominal plant, Ki = 0 (model values ESTIMATED)", loc="left",
                 color=INK, fontsize=11)
    ax.grid(color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, f"{hidden} passing gain sets lie outside the axes (above {y_hi:.0f} kJ or slower than {x_hi:.1f} s).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_PATH, facecolor=SURFACE)
    print(f"wrote {FIG_PATH}")


def propose(grid, front):
    points, adm = pick_points(front)
    approved = _approved()
    if approved:
        print(f"{POINTS_JSON} is already approved; not overwritten")
    else:
        with open(POINTS_JSON, "w") as f:
            json.dump({"approved": False, "tuning_set": "orig20", "rule": pick_points.__doc__,
                       "fast_band_s": FAST_BAND_S, "overshoot_max_cm": OVERSHOOT_MAX_CM,
                       "points": {role: {"kp": g["kp"], "ki": KI, "kd": g["kd"]} for role, g in points.items()}},
                      f, indent=2)
        print(f"wrote {POINTS_JSON} (approved: false)")
    print(f"\nadmissible front points (E_mech >= floor, peak overshoot < {OVERSHOOT_MAX_CM:.0f} cm): "
          + ", ".join(f"Kp {g['kp']:g} Kd {g['kd']:g}" for g in adm))
    for role in ROLES:
        g = points[role]
        print(f"proposed {role}: Kp {g['kp']:g} Kd {g['kd']:g} | {g['mean_t_reach_s']:.2f} s | {g['mean_E_mech_kJ']:.1f} kJ"
              f" ({g['pct_above_floor']:+.1f} % above floor) | worst error {g['max_cm']:.1f} cm |"
              f" peak overshoot {g['max_peak_overshoot_cm']:.1f} cm | noise 1.0 deg: {g['n1_baseline_E_mech_kJ']:.1f} kJ,"
              f" worst miss {g['n1_worst_miss_share'] * 100:.0f} %")
    make_figure(grid, front, points)


# ----------------------------------------------------------------------------- 11c
DIAG_CSV = "rl/11_dump_diagnosis.csv"
ALL_CASES = L10.SINGLES + L10.COMBINED
N1_CASES = [(n, c) for n, c in ALL_CASES if c.get("sigma") == 1.0]     # the four 1.0 deg conditions
N_SEEDS = 10                                                           # noise seeds on the tuning set (ASSUMED)
NO_CLIP = (-1e12, 1e12)
DIAG_EST = [{"name": "base"}, {"name": "oracle", "oracle": True}, {"name": "clean_state", "clean": True},
            {"name": "no_limit", "clamp": NO_CLIP}]


def load_points():
    with open(POINTS_JSON) as f:
        d = json.load(f)
    return [(role, (float(d["points"][role]["kp"]), float(d["points"][role]["ki"]), float(d["points"][role]["kd"])))
            for role in ROLES]


def pooled(rows, keys):
    """Mean over noise seeds of every numeric field, grouped by keys."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for rs in groups.values():
        row = dict(rs[0])
        for k, v in rs[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("kp", "ki", "kd"):
                row[k] = float(np.mean([r[k] for r in rs]))
        row["n_seeds"], row["n_moves"] = len(rs), int(sum(r["n_scen"] for r in rs))
        row["max_cm"] = float(max(r["max_cm"] for r in rs))
        out.append(row)
    return out


def kg_per_20cm(kp):
    """Constant m_hat error that gives a 20 cm static error with P action alone, at theta = 0 [kg]."""
    return L10.K_REF * kp * (REACH_M / R.L_BOOM_M) / (R.G * R.x_tip(0.0))


def step_diag():
    ctrls = [("L10_tuned", L10_TUNED)] + load_points()
    jobs = [job(label, gains, "dump", name=name, cond=cond, est=est, seed=s)
            for label, gains in ctrls for name, cond in N1_CASES for est in DIAG_EST for s in range(N_SEEDS)]
    t0 = time.time()
    rows = pooled(pool_map(jobs), ("controller", "condition", "estimator"))
    print(f"11c diagnosis: {len(jobs)} rows x 20 original scenarios in {time.time() - t0:.0f} s")
    write_csv(DIAG_CSV, rows)
    idx = {(r["controller"], r["condition"], r["estimator"]): r for r in rows}
    print(f"\n11c diagnosis, dump variant (3406 -> 1120 kg at 5 s), 20 original scenarios x {N_SEEDS} noise seeds")
    print("| controller | Kp | m_hat error for 20 cm, kg | condition | miss %: base | true m_tip | estimator on true state |"
          " no limit on m_hat | base: m_hat / m_tip | m_hat std kg | steps at lower limit % | mean signed error deg |"
          " no limit: m_hat / m_tip |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for label, gains in ctrls:
        for name, _ in N1_CASES:
            b, o, c, n = (idx[(label, name, e["name"])] for e in DIAG_EST)
            print(f"| {label} | {gains[0]:g} | {kg_per_20cm(gains[0]):.0f} | {name} | {b['share_over_20cm'] * 100:.1f} |"
                  f" {o['share_over_20cm'] * 100:.1f} | {c['share_over_20cm'] * 100:.1f} | {n['share_over_20cm'] * 100:.1f} |"
                  f" {b['m_hat_over_m_mean']:.3f} ± {b['m_hat_over_m_std']:.3f} | {b['m_hat_tail_std_kg']:.0f} |"
                  f" {b['share_at_lower_limit'] * 100:.1f} | {b['mean_err_signed_deg']:+.2f} |"
                  f" {n['m_hat_over_m_mean']:.3f} ± {n['m_hat_over_m_std']:.3f} |")


FIXTUNE_CSV = "rl/11_dump_fix_tuning.csv"
FIX_JSON = "rl/11_dump_fix_choice.json"
PHYS = (M_EMPTY, M_FULL)                                # clamp of fix 1 [kg]
TUNE_SEEDS = 5                                          # noise seeds per noisy row in the sweep (ASSUMED)
T_SLOW = (0.5, 1.0, 2.0, 4.0)                           # candidates, all ASSUMED
# pass 1: c in (2, 3, 4), gates up to 0.5. Pass 2 (after pass 1 put the best c and the best gate
# on the edge of the grid): c in (0.5, 1, 1.5), gates 1 and 2. Pass 3 (best adaptive T_f on the edge):
# adaptive T_f = 0.5 s. All passes: tuning set only.
ADAPT = [(t, c, n) for t in (0.5, 1.0, 2.0, 4.0) for c in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0) for n in (2, 4)]
GATES = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)


def est_slow(t):
    return {"name": f"slow_Tf{t:g}", "t_f": t}


def est_adapt(t, c, n):
    return {"name": f"adapt_Tf{t:g}_c{c:g}_n{n}", "t_f": t, "adapt": (c, n)}


def est_gate(g):
    return {"name": f"gate_{g:g}", "acc_gate": g}


def est_all(filt, g):
    """Clamp + filter + gate. filt without a name = the 0.2 s filter of Lab 10."""
    return {**filt, "name": f"all_{filt.get('name', 'Tf0.2')}_gate{g:g}", "clamp": PHYS, "acc_gate": g}


def est_clamp(filt):
    return {**filt, "name": f"clamp+{filt['name']}", "clamp": PHYS}


def singles():
    return ([{"name": "base"}, {"name": "clamp", "clamp": PHYS}] + [est_slow(t) for t in T_SLOW]
            + [est_adapt(*a) for a in ADAPT] + [est_gate(g) for g in GATES])


def joint_pass1():
    out = []
    for filt in [{"t_f": L10.T_F_S}] + [est_slow(t) for t in T_SLOW] + [est_adapt(*a) for a in ADAPT
                                                                       if a[0] >= 1.0 and a[1] >= 2.0 and a[2] == 4]:
        out += [est_all(filt, g) for g in GATES[:6]]
        if "name" in filt:
            out.append(est_clamp(filt))
    return out


def joint_pass2(filters):
    """Clamp + the chosen slow / adaptive filter + every gate, and the clamp + filter pair."""
    return [e for filt in filters for e in [est_all(filt, g) for g in GATES] + [est_clamp(filt)]]


def is_together(name):
    """All three fixes active: clamp, a slower or adaptive filter, and the gate."""
    return name.startswith("all_") and not name.startswith("all_Tf0.2")


def sweep_jobs(ests, ctrls, set_name="orig20", seeds=TUNE_SEEDS):
    jobs = []
    for est in ests:
        for label, gains in ctrls:
            for group, cases in L10.GROUPS:
                for name, cond in cases:
                    noisy = cond.get("sigma", 0.0) > 0 or cond.get("tau_noise", 0.0) > 0
                    for variant in L10.VARIANTS:
                        for s in range(seeds if noisy else 1):
                            jobs.append(job(label, gains, variant, set_name, group, name, cond, est, s))
    return jobs


def summarise(cells):
    """One line per estimator from pooled (controller, condition, variant) cells."""
    out = {}
    for name in dict.fromkeys(c["estimator"] for c in cells):
        cs = [c for c in cells if c["estimator"] == name]
        dump = [c for c in cs if c["variant"] == "dump" and c["sigma_th_deg"] == 1.0]
        other = [c for c in cs if not (c["variant"] == "dump" and c["sigma_th_deg"] == 1.0)]
        out[name] = {"estimator": name, "cells": len(cs),
                     "worst_miss_pct": 100 * max(c["share_over_20cm"] for c in cs),
                     "cells_over_5pct": sum(c["share_over_20cm"] > 0.05 for c in cs),
                     "cells_over_1pct": sum(c["share_over_20cm"] > 0.01 for c in cs),
                     "mean_miss_pct": 100 * float(np.mean([c["share_over_20cm"] for c in cs])),
                     "dump_n1_worst_miss_pct": 100 * max(c["share_over_20cm"] for c in dump),
                     "dump_n1_mean_miss_pct": 100 * float(np.mean([c["share_over_20cm"] for c in dump])),
                     "dump_n1_m_hat_ratio_std": float(np.mean([c["m_hat_over_m_std"] for c in dump])),
                     "dump_n1_m_hat_ratio_mean": float(np.mean([c["m_hat_over_m_mean"] for c in dump])),
                     "other_worst_miss_pct": 100 * max(c["share_over_20cm"] for c in other),
                     "other_mean_miss_pct": 100 * float(np.mean([c["share_over_20cm"] for c in other])),
                     "mean_E_mech_kJ": float(np.mean([c["mean_E_mech_kJ"] for c in cs])),
                     "mean_est_update_share": float(np.mean([c["share_est_updates"] for c in cs]))}
    return out


def rank_key(s):
    """Choice rule of 11c on the tuning set: worst miss rate, then mean miss rate, then mean E_mech."""
    return (round(s["worst_miss_pct"], 6), round(s["mean_miss_pct"], 6), s["mean_E_mech_kJ"])


def run_settings(ests, ctrls, summ):
    """Simulate the settings that are not in summ yet (tuning set) and add their summary lines."""
    new = [e for e in ests if e["name"] not in summ]
    if not new:
        return
    t0 = time.time()
    jobs = sweep_jobs(new, ctrls)
    cells = pooled(pool_map(jobs), ("estimator", "controller", "condition", "variant"))
    summ.update(summarise(cells))
    print(f"11c sweep: {len(new)} new estimator settings, {len(jobs)} rows x 20 original scenarios"
          f" in {time.time() - t0:.0f} s", flush=True)
    write_csv(FIXTUNE_CSV, list(summ.values()))


def best_of(summ, names):
    return min(names, key=lambda n: rank_key(summ[n]))


def step_fixtune():
    ctrls = load_points()
    summ = {r["estimator"]: r for r in read_csv(FIXTUNE_CSV)} if os.path.exists(FIXTUNE_CSV) else {}
    single, joint1 = singles(), joint_pass1()
    by_name = {e["name"]: e for e in single + joint1}
    run_settings(single + joint1, ctrls, summ)
    names = [e["name"] for e in single]
    best = {k: best_of(summ, [n for n in names if n.startswith(k + "_")]) for k in ("slow", "adapt", "gate")}
    joint2 = joint_pass2([by_name[best["slow"]], by_name[best["adapt"]]])
    by_name.update({e["name"]: e for e in joint2})
    run_settings(joint2, ctrls, summ)
    best["together"] = best_of(summ, [n for n in by_name if is_together(n)])
    best["clamp_filter"] = best_of(summ, [n for n in by_name if n.startswith("clamp+")])
    best["recommended"] = best_of(summ, list(by_name))
    order = ("slow", "adapt", "gate", "together")
    chosen = [by_name["base"], by_name["clamp"]] + [by_name[best[k]] for k in order]
    with open(FIX_JSON, "w") as f:
        json.dump({"tuning_set": "orig20", "noise_seeds": TUNE_SEEDS, "rule": rank_key.__doc__,
                   "n_settings": len(by_name), "roles": {"clamp": "clamp", **best}, "estimators": chosen}, f, indent=2)
    print(f"wrote {FIX_JSON}")
    print(f"\n11c fix sweep, 20 original scenarios, three approved points x 72 rows = {int(summ['base']['cells'])} cells"
          f" per setting, {TUNE_SEEDS} noise seeds on noisy rows. * = chosen by the rule.")
    picked = set(best.values())
    print(SWEEP_HEAD)
    for n in names:
        print(sweep_line(summ[n], " *" if n in picked else ""))
    print("\njoint settings, fifteen best by the rule (all_ = clamp + filter + gate)")
    print(SWEEP_HEAD)
    for n in sorted((n for n in by_name if n not in names), key=lambda n: rank_key(summ[n]))[:15]:
        print(sweep_line(summ[n], " *" if n in picked else ""))
    print("\nroles: " + json.dumps(best))


SWEEP_HEAD = ("| estimator | worst miss % | cells > 5 % | mean miss % | dump 1.0 deg: worst miss % | mean miss % |"
              " m_hat / m_tip | other rows: worst miss % | mean miss % | mean E_mech kJ | steps with update % |\n"
              "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")


def sweep_line(s, mark=""):
    return (f"| {s['estimator']}{mark} | {s['worst_miss_pct']:.1f} | {s['cells_over_5pct']:.0f} | {s['mean_miss_pct']:.2f} |"
            f" {s['dump_n1_worst_miss_pct']:.1f} | {s['dump_n1_mean_miss_pct']:.2f} |"
            f" {s['dump_n1_m_hat_ratio_mean']:.3f} ± {s['dump_n1_m_hat_ratio_std']:.3f} |"
            f" {s['other_worst_miss_pct']:.1f} | {s['other_mean_miss_pct']:.2f} | {s['mean_E_mech_kJ']:.1f} |"
            f" {s['mean_est_update_share'] * 100:.0f} |")


CHECK_CSV = "rl/11_dump_fix_checks.csv"


def step_fixcheck():
    """Two spot checks behind the explanations of 11c, `economical`, tuning set. Written to CHECK_CSV."""
    eco = dict(load_points())["economical"]
    jobs = [job("economical", eco, v, est=est_slow(t)) for t in (2.0, 4.0) for v in L10.VARIANTS]
    for est in ({"name": "base"}, est_gate(0.5), est_gate(2.0)):
        for name, cond in (("none", {}), ("noise_1.0", {"sigma": 1.0})):
            jobs += [job("economical", eco, v, name=name, cond=cond, est=est, seed=s)
                     for v in ("baseline", "dump") for s in range(TUNE_SEEDS if cond else 1)]
    rows = pooled(pool_map(jobs), ("estimator", "condition", "variant"))
    write_csv(CHECK_CSV, rows)
    for r in rows:
        print(f"{r['estimator']:10s} {r['condition']:10s} {r['variant']:10s} miss {r['share_over_20cm'] * 100:5.1f} %"
              f"  updates {r['share_est_updates'] * 100:5.1f} %  m_hat/m {r['m_hat_over_m_mean']:.3f} ± {r['m_hat_over_m_std']:.3f}"
              f"  m_hat std {r['m_hat_tail_std_kg']:.0f} kg  signed err {r['mean_err_signed_deg']:+.2f} deg")


# ----------------------------------------------------------------------------- unseen evaluation
OUT_CSV = "rl/11_real_plant_tuning.csv"
MD_PATH = "rl/11_real_plant_tuning.md"
L10_CSV = "rl/10_real_plant.csv"                        # read only
L10_REF = [("L10_tuned", L10_TUNED, "sched_tuned"), ("L10_old", L10_OLD, "sched_old_gains")]


def step_eval():
    """Three approved points x chosen estimators x 72 rows, and the two Lab 10 controllers, unseen set."""
    if not _approved():
        raise RuntimeError(f"{POINTS_JSON} is not approved")
    with open(FIX_JSON) as f:
        ests = json.load(f)["estimators"]
    rows72 = [(group, name, cond, variant) for group, cases in L10.GROUPS for name, cond in cases
              for variant in L10.VARIANTS]
    jobs = [job(role, gains, variant, "new100", group, name, cond, est)
            for est in ests for role, gains in load_points() for group, name, cond, variant in rows72]
    jobs += [job(label, gains, variant, "new100", group, name, cond)
             for label, gains, _ in L10_REF for group, name, cond, variant in rows72]
    t0 = time.time()
    rows = pool_map(jobs)
    print(f"evaluation: {len(rows)} rows x {rows[0]['n_scen']} unseen moves in {time.time() - t0:.0f} s")
    write_csv(OUT_CSV, rows)
    ref = {(r["controller"], r["condition"], r["variant"]): r for r in read_csv(L10_CSV)}
    keys = ("mean_E_mech_kJ", "share_over_20cm", "mean_t_reach_s", "mean_E_pump_kJ", "m_hat_over_m_std")
    worst, n = {k: 0.0 for k in keys}, 0
    for label, _, old in L10_REF:
        for r in rows:
            if r["controller"] == label:
                n += 1
                for k in keys:
                    worst[k] = max(worst[k], abs(r[k] - ref[(old, r["condition"], r["variant"])][k]))
    print(f"cross-check with {L10_CSV}: {n} rows, largest absolute difference {worst}")
    assert max(worst.values()) < 1e-9


if __name__ == "__main__":
    step = sys.argv[sys.argv.index("--step") + 1] if "--step" in sys.argv else ""
    if step == "eval":
        step_eval()
        sys.exit(0)
    if step == "fixcheck":
        step_fixcheck()
        sys.exit(0)
    if step == "fixtune":
        step_fixtune()
        sys.exit(0)
    if step == "diag":
        step_diag()
        sys.exit(0)
    if step == "check":
        step_check()
    elif step == "a":
        step_a()
    elif step == "b":
        grid_rows, front_rows = step_b()
        print_front(front_rows)
        propose(grid_rows, front_rows)
    else:
        print(__doc__)
