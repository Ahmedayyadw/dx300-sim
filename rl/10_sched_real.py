"""Lab 10c: the sched controller on the real plant (rl/boom_env_real.py). No RL, no training.

Design: docs/10_real_plant.md, section 5. Controller = pid_gain_sched of Lab 09c3 / 09f with
the three changes the plant forces:
  feed-forward   tau_ff = tau_g(theta, m_hat), the real gravity model with the estimated tip mass
  estimator      m_raw = (tau_cyl - J_nom dw/dt - tau_f - tau_g(theta, 0)) / (g x_tip(theta))
                 filter, end-stop gate and T_f = 0.2 s as Lab 09c, clipped to [0, 2 x 3406 kg].
                 tau_cyl is the torque the cylinders deliver (plant info, mean over the step).
  gain sched     k_hat = tau_cyl / u as 09c3, one estimate per sign of u (tau_up and tau_down
                 differ), no update on a step at the flow limit (detected from the measured
                 rod speed r(theta) omega). Command:
                   u = (K_REF (Kp e + Ki int(e) - Kd omega) + tau_ff) / k_hat
                 K_REF = tau_up(0), k_hat of the side given by the sign of the bracket.

Tuning: Kp x Kd over the Lab 07 grid, Ki = 0, nominal real plant, the 20 original scenarios
(RNG seed 7). Lab 07 criterion: lowest mean E_mech among the sets with worst-case error < 20 cm.
If no set passes: fewest moves over 20 cm, then lowest mean E_mech.
Evaluation: the 100 unseen scenarios (RNG seed 123), the 09c3 conditions that still apply
(tau_d is part of the plant now), four payload variants, the tuned gains and the old gains of
Lab 09f for reference.
E_pump is the Lab 09b formula: lower bound, ASSUMED parameters (dp_valve, eta_pump in
rl/09b_hyd_params.py, ASSUMED - not from manual).

Usage: python rl/10_sched_real.py [--dry-run] [--report-only]
  --dry-run       3 x 2 gain sets, 3 original and 5 unseen scenarios; prints the report, writes
                  no file
  --report-only   rebuild rl/10_real_plant.md from the existing CSV files, no simulation
"""
import csv
import math
import multiprocessing as mp
import os
import sys
import time
import numpy as np
import boom_env_real as R
from lab09_common import G, HYD, KD, KP, LC, M_NOM, NEW, ORIG, cm

CSV_PATH = "rl/10_real_plant.csv"
GRID_CSV = "rl/10_tuning_grid.csv"
MD_PATH = "rl/10_real_plant.md"
CHECK_CSV = "rl/10_simscape_check.csv"      # read only, written by rl/10_simscape_check.py
REF_09F = "rl/09f_kp_retune.csv"            # read only: sched on the placeholder plant
DRY_RUN = "--dry-run" in sys.argv
REPORT_ONLY = "--report-only" in sys.argv

T_F_S = 0.2                                 # estimator filter time, as Lab 09c
TAIL_STEPS = 40                             # last 2 s, as 09c2/09c3
U_MIN = 0.05                                # no gain update below this command, as 09c3
FLOW_GATE = 0.95                            # no gain update above this share of the rod speed limit
K_REF = R.tau_up(0.0)                       # N m per unit command at theta = 0 (ESTIMATED table)
K_DOWN_REF = R.tau_down(0.0)
KI = 0.0                                    # Lab 07 / 09c3 / 09f value
OLD_GAINS = (0.55, 0.0, 0.25)               # Lab 09f (rl/09f_kp_retune.md)
REACH_M = 0.20                              # "within 20 cm"
KP_GRID = KP[:3] if DRY_RUN else KP
KD_GRID = KD[1:3] if DRY_RUN else KD
SCEN = {"orig20": ORIG[:3] if DRY_RUN else ORIG, "new100": NEW[:5] if DRY_RUN else NEW}

# payload variants (docs/10_real_plant.md, section 3): start mass, (step, mass after) [kg]
M_FULL = R.M_TIP_NOM_KG                     # 3406 kg, ESTIMATED (payload density 1800 kg/m^3)
M_EMPTY = 1120.0                            # bucket alone, OFFICIAL (S5 p.21, model unverified)
M_HALF = M_EMPTY + 0.5 * (M_FULL - M_EMPTY)  # 2263 kg, ESTIMATED
VARIANTS = {"baseline": (M_FULL, None), "fill_half": (M_HALF, (100, M_FULL)),
            "fill": (M_EMPTY, (100, M_FULL)), "dump": (M_FULL, (100, M_EMPTY))}
PLACEHOLDER_VARIANT = {"baseline": "baseline", "fill_half": "payload_x1.2"}   # for the comparison

# conditions of 09c3 without tau_d (the angle-dependent limit is part of the plant)
SINGLES = [("none", {})] + [(f"noise_{s}", {"sigma": s}) for s in (0.2, 0.5, 1.0)] + [
    ("tau_noise_5pct", {"tau_noise": 0.05}),
    ("J_x0.38", {"j": 0.38}), ("J_x1.5", {"j": 1.5}),
    ("b_x0.5", {"b": 0.5}), ("b_x2", {"b": 2.0}),
]
COMBINED = [(f"all_n{s}_J{j}_b{b}", {"sigma": s, "tau_noise": 0.05, "j": j, "b": b})
            for s in (0.2, 0.5, 1.0) for j in (0.38, 1.5) for b in (0.5, 2.0)
            if not (j == 0.38 and b == 2.0)]           # same nine cases as 09c3
GROUPS = [("single", SINGLES), ("combined", COMBINED)]


class NoisyBoomEnvReal(R.BoomEnvReal):
    """Gaussian noise on observed theta and omega, as NoisyMixin of 09c3 (from 06b)."""

    def _obs(self):
        th = self.th + self.noise_rng.normal(0.0, self.sigma_th)
        om = self.om + self.noise_rng.normal(0.0, self.sigma_om)
        return np.array([th, om, self.target - th], dtype=np.float32)


def make_env(variant, sigma=0.0, j=1.0, b=1.0, **_):
    m_tip, payload = VARIANTS[variant]
    cls = NoisyBoomEnvReal if sigma > 0 else R.BoomEnvReal
    env = cls(m_tip=m_tip, payload=payload, j_scale=j, fric_scale=b)
    if sigma > 0:
        env.sigma_th = np.radians(sigma)
        env.sigma_om = env.sigma_th / env.dt * 0.1
        env.noise_rng = np.random.default_rng(0)
    return env


class SchedReal:
    """sched (pid_gain_sched of 09c3) adapted to the real plant. See the module docstring."""

    def __init__(self, gains, t_f=T_F_S, tau_noise=0.0, dt=0.05):
        self.kp, self.ki, self.kd = gains
        self.dt, self.alpha = dt, dt / (t_f + dt)
        self.tau_noise, self.tau_rng = tau_noise, np.random.default_rng(1)
        self.reset()

    def reset(self):
        self.integ, self.m_hat, self.prev = 0.0, M_FULL, None
        self.k_up, self.k_down, self.u_prev, self.hist = K_REF, K_DOWN_REF, 0.0, []

    def act(self, obs):
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])   # what the controller observes
        tau_des = K_REF * (self.kp * e + self.ki * self.integ + self.kd * (-om)) \
            + R.gravity_torque(th, self.m_hat)
        u_raw = tau_des / (self.k_up if tau_des >= 0.0 else self.k_down)
        if abs(u_raw) < 1.0:                                       # anti-windup
            self.integ += e * self.dt
        self.prev = (th, om)
        self.u_prev = float(np.clip(u_raw, -1.0, 1.0))
        return self.u_prev

    def update(self, obs, tau, th_min, th_max):
        """Estimator step. obs is the observation after the step that delivered torque tau."""
        if self.tau_noise > 0:                                     # one reading feeds both estimates
            tau = tau * (1.0 + self.tau_noise * self.tau_rng.normal())
        th1, om1 = float(obs[0]), float(obs[1])
        th0, om0 = self.prev
        r = R.moment_arm(th0)
        v = r * om1                                                # measured rod speed
        on_flow = v >= FLOW_GATE * R.V_EXT_MAX or v <= -FLOW_GATE * R.V_RET_MAX
        if abs(self.u_prev) >= U_MIN and not on_flow:
            k_raw = tau / self.u_prev
            if self.u_prev > 0:
                self.k_up = float(np.clip(self.k_up + self.alpha * (k_raw - self.k_up),
                                          0.2 * K_REF, 2.0 * K_REF))
            else:
                self.k_down = float(np.clip(self.k_down + self.alpha * (k_raw - self.k_down),
                                            0.2 * K_DOWN_REF, 2.0 * K_DOWN_REF))
        if not (th1 <= th_min + 1e-6 or th1 >= th_max - 1e-6):     # end-stop gate
            tau_f = r * (R.F_COULOMB_N * np.sign(om0) + R.C_VISC_NSM * r * om0)   # nominal friction
            m_raw = (tau - R.J_NOM * (om1 - om0) / self.dt - tau_f - R.gravity_torque(th0, 0.0)) \
                / (R.G * R.x_tip(th0))
            self.m_hat = float(np.clip(self.m_hat + self.alpha * (m_raw - self.m_hat), 0.0, 2.0 * M_FULL))
        self.hist.append(self.m_hat)


def rollout(ctrl, env, th0, target):
    """One 10 s move, set up as in 06/07/09. Metrics from the true state."""
    env.reset(seed=0)
    env.th, env.om, env.target = th0, 0.0, target
    obs = env._obs()
    ctrl.reset()
    reach_m = abs(target - th0) * R.L_BOOM_M
    t_reach = 0.0 if reach_m <= REACH_M else float("nan")
    for k in range(env.max_steps):
        u = ctrl.act(obs)
        obs, _, _, trunc, info = env.step(np.array([u]))
        ctrl.update(obs, info["tau_cyl"], env.th_min, env.th_max)
        if math.isnan(t_reach) and abs(env.target - env.th) * R.L_BOOM_M <= REACH_M:
            t_reach = (k + 1) * env.dt
        if trunc:
            break
    return {"err_deg": abs(np.degrees(env.target - env.th)), "E_mech_kJ": env.energy_J / 1e3,
            "E_pump_kJ": (env.W_pos_J + HYD.DP_VALVE_PA * env.vol_m3) / HYD.ETA_PUMP / 1e3,
            "t_reach_s": t_reach, "flow": env.n_flow / env.n_sub, "torque": env.n_torque / env.n_sub,
            "violations": env.n_violation,
            "tail": [m / env.m_tip for m in ctrl.hist[-TAIL_STEPS:]]}


def run_row(job):
    """One table row: a controller on one condition, variant and scenario set. Worker process."""
    label, gains, set_name, group, name, cond, variant = job
    env = make_env(variant, **cond)
    ctrl = SchedReal(gains, tau_noise=cond.get("tau_noise", 0.0))
    rs = [rollout(ctrl, env, *sc) for sc in SCEN[set_name]]
    errs = np.array([r["err_deg"] for r in rs])
    reach = np.array([r["t_reach_s"] for r in rs])
    tail = np.concatenate([r["tail"] for r in rs])
    m0, payload = VARIANTS[variant]
    return {"set": set_name, "n_scen": len(rs), "group": group, "condition": name,
            "sigma_th_deg": cond.get("sigma", 0.0), "tau_noise": cond.get("tau_noise", 0.0),
            "J_factor": cond.get("j", 1.0), "b_factor": cond.get("b", 1.0), "variant": variant,
            "m_tip_start_kg": m0, "m_tip_end_kg": payload[1] if payload else m0,
            "controller": label, "kp": gains[0], "ki": gains[1], "kd": gains[2], "t_f_s": T_F_S,
            "mean_deg": errs.mean(), "max_deg": errs.max(),
            "share_over_20cm": float((cm(errs) > 20).mean()),
            "mean_E_mech_kJ": float(np.mean([r["E_mech_kJ"] for r in rs])),
            "mean_E_pump_kJ": float(np.mean([r["E_pump_kJ"] for r in rs])),
            "mean_t_reach_s": float(np.nanmean(reach)) if not np.all(np.isnan(reach)) else float("nan"),
            "n_never_reached": int(np.isnan(reach).sum()),
            "share_flow_limit": float(np.mean([r["flow"] for r in rs])),
            "share_torque_limit": float(np.mean([r["torque"] for r in rs])),
            "flow_violations": int(sum(r["violations"] for r in rs)),
            "m_hat_over_m_mean": float(tail.mean()), "m_hat_over_m_std": float(tail.std())}


def pick(grid):
    """Lab 07 criterion on the tuning rows. Returns (best row, True if a set had no miss)."""
    ok = [g for g in grid if cm(g["max_deg"]) < 20]
    if ok:
        return min(ok, key=lambda g: g["mean_E_mech_kJ"]), True
    return min(grid, key=lambda g: (g["share_over_20cm"], g["mean_E_mech_kJ"])), False


def simulate():
    t0 = time.time()
    with mp.get_context("spawn").Pool(os.cpu_count() or 1) as pool:
        jobs = [("grid", (kp, KI, kd), "orig20", "tuning", "none", {}, "baseline")
                for kp in KP_GRID for kd in KD_GRID]
        grid = pool.map(run_row, jobs)
        best, _ = pick(grid)
        tuned = (best["kp"], best["ki"], best["kd"])
        print(f"tuning: {len(grid)} gain sets x {len(SCEN['orig20'])} scenarios in {time.time() - t0:.0f} s"
              f" -> Kp={tuned[0]:g} Ki={tuned[1]:g} Kd={tuned[2]:g}")
        t1 = time.time()
        jobs = [(label, gains, "new100", group, name, cond, variant)
                for label, gains in (("sched_tuned", tuned), ("sched_old_gains", OLD_GAINS))
                for group, cases in GROUPS for name, cond in cases for variant in VARIANTS]
        rows = pool.map(run_row, jobs, chunksize=1)
    print(f"evaluation: {len(rows)} rows x {len(SCEN['new100'])} moves in {time.time() - t1:.0f} s")
    return grid, rows


def parse(r):
    out = {}
    for k, v in r.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def read_csv(path):
    with open(path) as f:
        return [parse(r) for r in csv.DictReader(f)]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def status(share_pct):
    return "BREAKS" if share_pct > 5 else "degraded" if share_pct > 1 else "ok"


def span(vals, digits=1, unit=""):
    lo, hi = min(vals), max(vals)
    return f"{lo:.{digits}f}{unit}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f}-{hi:.{digits}f}{unit}"


def report(grid, rows, write=True):
    best, clean = pick(grid)
    tuned = (best["kp"], best["ki"], best["kd"])
    idx = {(r["controller"], r["condition"], r["variant"]): r for r in rows}
    labels = ["sched_tuned", "sched_old_gains"]
    short = {"sched_tuned": "tuned", "sched_old_gains": "old"}
    cases = SINGLES + COMBINED
    n_new, n_orig = int(rows[0]["n_scen"]), int(grid[0]["n_scen"])

    # --- Simscape check (10b) ---
    check = read_csv(CHECK_CSV) if os.path.exists(CHECK_CSV) else []
    check_table = ["| check | quantity | Python | Simscape | diff | diff % |", "|---|---|---:|---:|---:|---:|"]
    for r in check:
        dg = 3 if r["unit"] == "s" else 1
        check_table.append(f"| {r['check']} | {r['quantity']} [{r['unit']}] | {r['python']:.{dg}f} |"
                           f" {r['simscape']:.{dg}f} | {r['diff']:+.{dg}f} | {r['diff_pct']:+.2f} |")
    bins = [abs(r["diff_pct"]) for r in check if r["check"] == "static" and "t = 0" not in r["quantity"]]
    times = [r for r in check if r["check"] == "boom_up_time"]
    works = [r for r in check if r["check"] == "boom_up_work"]

    # --- tuning ---
    kps, kds = sorted({g["kp"] for g in grid}), sorted({g["kd"] for g in grid})
    gidx = {(g["kp"], g["kd"]): g for g in grid}
    tune_table = ["| Kp \\ Kd | " + " | ".join(f"{kd:g}" for kd in kds) + " |", "|---:|" + "---:|" * len(kds)]
    for kp in kps:
        grid_cells = []
        for kd in kds:
            g = gidx[(kp, kd)]
            cell = f"{cm(g['max_deg']):.0f} cm / {g['mean_E_mech_kJ']:.0f} kJ"
            grid_cells.append(f"**{cell}**" if (kp, kd) == (best["kp"], best["kd"]) else cell)
        tune_table.append(f"| {kp:g} | " + " | ".join(grid_cells) + " |")
    n_ok = sum(cm(g["max_deg"]) < 20 for g in grid)

    # --- results table ---
    def cells(name, variant, fmt):
        return " | ".join(fmt(idx[(label, name, variant)]) for label in labels)

    heads = " | ".join(short[label] for label in labels)

    def table(group_cases):
        out = [f"| condition | variant | miss %: {heads} | E_mech kJ: {heads} |"
               f" E_pump kJ (lower bound, ASSUMED parameters): {heads} | reach 20 cm, s: {heads} |"
               f" never reached: {heads} | flow limit %: {heads} | torque limit %: {heads} |"
               f" m_hat / m_tip (tuned) | status: {heads.replace(' | ', ' / ')} |",
               "|---|---|" + "---:|" * (7 * len(labels)) + "---|---|"]
        for name, _ in group_cases:
            for variant in VARIANTS:
                t = idx[("sched_tuned", name, variant)]
                out.append(
                    f"| {name} | {variant} | {cells(name, variant, lambda r: format(r['share_over_20cm'] * 100, '.0f'))} |"
                    f" {cells(name, variant, lambda r: format(r['mean_E_mech_kJ'], '.1f'))} |"
                    f" {cells(name, variant, lambda r: format(r['mean_E_pump_kJ'], '.1f'))} |"
                    f" {cells(name, variant, lambda r: format(r['mean_t_reach_s'], '.2f'))} |"
                    f" {cells(name, variant, lambda r: format(r['n_never_reached'], '.0f'))} |"
                    f" {cells(name, variant, lambda r: format(r['share_flow_limit'] * 100, '.1f'))} |"
                    f" {cells(name, variant, lambda r: format(r['share_torque_limit'] * 100, '.1f'))} |"
                    f" {t['m_hat_over_m_mean']:.3f} ± {t['m_hat_over_m_std']:.3f} |"
                    f" {' / '.join(status(idx[(label, name, variant)]['share_over_20cm'] * 100) for label in labels)} |")
        return out

    def tally(label):
        c = {"BREAKS": 0, "degraded": 0, "ok": 0}
        for name, _ in cases:
            for variant in VARIANTS:
                c[status(idx[(label, name, variant)]["share_over_20cm"] * 100)] += 1
        return f"{c['BREAKS']} break, {c['degraded']} degraded, {c['ok']} ok"

    def col(label, group_cases, key, scale=1.0, variants=VARIANTS):
        return [idx[(label, name, v)][key] * scale for name, _ in group_cases for v in variants]

    summary = ["| rows | miss %: " + heads + " | E_mech kJ: " + heads + " | reach 20 cm, s: " + heads
               + " | flow limit %: " + heads + " | torque limit %: " + heads + " |",
               "|---|" + "---:|" * (5 * len(labels))]
    for title, group_cases in (("single conditions", SINGLES), ("combined cases", COMBINED)):
        summary.append(
            f"| {title} ({len(group_cases) * len(VARIANTS)} rows) | "
            + " | ".join(span(col(label, group_cases, "share_over_20cm", 100), 0) for label in labels) + " | "
            + " | ".join(span(col(label, group_cases, "mean_E_mech_kJ")) for label in labels) + " | "
            + " | ".join(span(col(label, group_cases, "mean_t_reach_s", 1, VARIANTS), 2) for label in labels) + " | "
            + " | ".join(span(col(label, group_cases, "share_flow_limit", 100)) for label in labels) + " | "
            + " | ".join(span(col(label, group_cases, "share_torque_limit", 100), 2) for label in labels) + " |")
    summary.append(f"| status over all {len(cases) * len(VARIANTS)} rows | "
                   + " | ".join(tally(label) for label in labels) + " |" + " |" * (4 * len(labels)))

    def breaks(label):
        hit = [f"{name} {v} ({idx[(label, name, v)]['share_over_20cm'] * 100:.0f} %)" for name, _ in cases
               for v in VARIANTS if status(idx[(label, name, v)]["share_over_20cm"] * 100) == "BREAKS"]
        return "; ".join(hit) if hit else "none"

    reach_t = col("sched_tuned", cases, "mean_t_reach_s")
    reach_o = col("sched_old_gains", cases, "mean_t_reach_s")
    tau_g_real, tau_g_ph = R.gravity_torque(0.0), M_NOM * G * LC

    # --- comparison with the placeholder plant (Lab 09f, sched Kp = 0.55, 100 unseen scenarios) ---
    ref = {}
    if os.path.exists(REF_09F):
        for r in read_csv(REF_09F):
            if r["set"] == "new100" and abs(r["kp"] - OLD_GAINS[0]) < 1e-9 and r["group"] in ("single", "combined"):
                ref[(r["condition"], r["variant"])] = r
    comp, pairs = [], []
    if ref:
        comp = ["| condition | variant (real / placeholder) | miss %: placeholder | real, old gains | real, tuned |"
                " E_mech kJ: placeholder | real, old gains | real, tuned |"
                " E_pump kJ (lower bound, ASSUMED parameters): placeholder | real, old gains | real, tuned |",
                "|---|---|" + "---:|" * 9]
        for name, _ in cases:
            for variant, pv in PLACEHOLDER_VARIANT.items():
                p, o, t = ref[(name, pv)], idx[("sched_old_gains", name, variant)], idx[("sched_tuned", name, variant)]
                pairs.append((p, o, t))
                comp.append(f"| {name} | {variant} / {pv} | {p['share_over_20cm'] * 100:.0f} |"
                            f" {o['share_over_20cm'] * 100:.0f} | {t['share_over_20cm'] * 100:.0f} |"
                            f" {p['mean_kJ']:.1f} | {o['mean_E_mech_kJ']:.1f} | {t['mean_E_mech_kJ']:.1f} |"
                            f" {p['mean_pump_kJ']:.1f} | {o['mean_E_pump_kJ']:.1f} | {t['mean_E_pump_kJ']:.1f} |")

    def stat(i, key_p, key_r, scale=1.0):
        """(placeholder, real) lists of one metric over the matched rows; i = 1 old, 2 tuned."""
        return [p[0][key_p] * scale for p in pairs], [p[i][key_r] * scale for p in pairs]

    statement = []
    if pairs:
        for i, who in ((2, "tuned gains"), (1, "old gains")):
            mp_, mr = stat(i, "share_over_20cm", "share_over_20cm", 100)
            ep, er = stat(i, "mean_kJ", "mean_E_mech_kJ")
            pp, pr = stat(i, "mean_pump_kJ", "mean_E_pump_kJ")
            ratio_e = [b / a for a, b in zip(ep, er)]
            ratio_p = [b / a for a, b in zip(pp, pr)]
            statement.append(
                f"- **Real plant, {who}**, against the placeholder plant on the {len(pairs)} matched rows: "
                f"miss rate {span(mr, 0, ' %')} (mean {np.mean(mr):.1f} %) against {span(mp_, 0, ' %')} "
                f"(mean {np.mean(mp_):.1f} %); higher on {sum(b > a for a, b in zip(mp_, mr))} rows, lower on "
                f"{sum(b < a for a, b in zip(mp_, mr))}. E_mech {span(er)} kJ (mean {np.mean(er):.1f}) against "
                f"{span(ep)} kJ (mean {np.mean(ep):.1f}), ratio {span(ratio_e, 2)} (mean {np.mean(ratio_e):.2f}). "
                f"E_pump (lower bound, ASSUMED parameters) {span(pr)} kJ against {span(pp)} kJ, "
                f"ratio {span(ratio_p, 2)} (mean {np.mean(ratio_p):.2f}).")

    none_t, none_o = idx[("sched_tuned", "none", "baseline")], idx[("sched_old_gains", "none", "baseline")]
    viol = int(sum(r["flow_violations"] for r in rows))
    lines = [
        "# Lab 10: control study on the real plant",
        "",
        "No RL and no training. Plant: `rl/boom_env_real.py` (Lab 10a), built from the Lab 08a Simscape "
        "export `matlab/lab08a/export/boom_params_lab08a.json`; design and the source of every value in "
        "`docs/10_real_plant.md`. Controller: `sched` (pid_gain_sched of Lab 09c3 / 09f) adapted to the "
        f"plant (`rl/10_sched_real.py`). All result numbers are from `{CSV_PATH}`, `{GRID_CSV}` and "
        f"`{CHECK_CSV}`.",
        "",
        "Status of the inputs: pump flow 494 L/min, relief 330 bar, bore, rod, boom mass and length are "
        "OFFICIAL. Pins, moment arm, torque limits, inertia, front point masses, friction and the payload "
        "density (1800 kg/m^3) are ESTIMATED (Lab 08a). The hard retraction speed limit, the substep count "
        f"and the payload variants are ASSUMED. E_pump uses dp_valve = {HYD.DP_VALVE_PA / 1e5:.0f} bar and "
        f"eta_pump = {HYD.ETA_PUMP}, both ASSUMED - not from manual; it is the energy of an ideal "
        "demand-following pump, a lower bound (Lab 09b2).",
        "",
        "## Statement: placeholder plant against real plant",
        "",
        f"Placeholder numbers: `sched` with Kp = {OLD_GAINS[0]:g} on `rl/boom_env.py`, {n_new} unseen "
        f"scenarios, from `{REF_09F}`. Matched rows: the {len(cases)} conditions that exist on both plants, "
        "real `baseline` against placeholder `baseline` and real `fill_half` (tip mass 2263 -> 3406 kg, "
        "gravity torque x 1.23) against placeholder `payload_x1.2`. The placeholder combined cases include "
        "its `tau_d` flag; on the real plant the angle-dependent limit is always present.",
        "",
        *(statement or ["Not available: " + REF_09F + " not found."]),
        f"- Without stress (`none`, `baseline`): E_mech {none_t['mean_E_mech_kJ']:.1f} kJ tuned and "
        f"{none_o['mean_E_mech_kJ']:.1f} kJ old gains"
        + (f", against {ref[('none', 'baseline')]['mean_kJ']:.1f} kJ on the placeholder plant" if ref else "")
        + f". Miss rate {none_t['share_over_20cm'] * 100:.0f} % tuned, {none_o['share_over_20cm'] * 100:.0f} % old gains"
        + (f", {ref[('none', 'baseline')]['share_over_20cm'] * 100:.0f} % placeholder." if ref else "."),
        f"- Main cause of the higher energy: the real plant carries the front attachment. Gravity torque at "
        f"theta = 0 is {tau_g_real / 1e3:.1f} kN m against {tau_g_ph / 1e3:.1f} kN m on the placeholder plant "
        f"(x {tau_g_real / tau_g_ph:.2f}), so the same move needs about that much more work.",
        f"- Tuned against old gains on the real plant: the Lab 07 criterion picks the lowest energy, and the "
        f"tuned gains are slow. Mean time to come within 20 cm is {span(reach_t, 2)} s tuned against "
        f"{span(reach_o, 2)} s old gains over the {len(cases) * len(VARIANTS)} rows. The old gains overshoot "
        "(the real plant has about 1/300 of the placeholder damping), which costs the extra energy.",
        f"- Rows over 5 % miss on the real plant (all {len(cases) * len(VARIANTS)} rows, including `fill` and "
        f"`dump`, which have no placeholder counterpart). Tuned: {breaks('sched_tuned')}. Old gains: "
        f"{breaks('sched_old_gains')}.",
        "- The energy figures of the two plants are not the same quantity in one respect: on the real plant "
        "E_mech uses the torque the cylinders deliver, which on the flow limit is below the commanded "
        "torque. The placeholder plant has no flow limit.",
        "",
        "## 1. Check against Simscape (10b)",
        "",
        "From `rl/10_simscape_check.py`. No parameter was tuned to make these match.",
        "",
        *check_table,
        "",
        (f"- Static holding torque: within {max(bins):.2f} % of the Simscape value in every 5 deg bin. The "
         "Simscape figure is tau - J dw/dt from the logged boom-up of Lab 09b2, which oscillates (limit "
         "cycle), so the bins carry the noise of a differentiated signal. The two t = 0 rows are not an "
         "independent check: `build_stage4.m` sets the initial cap pressure from the same analytic formula."
         if bins else "- Static check not available."),
        *([f"- Boom-up time: {times[0]['python']:.3f} s against {times[0]['simscape']:.3f} s "
           f"({times[0]['diff_pct']:+.2f} %) without friction, {times[1]['python']:.3f} s against "
           f"{times[1]['simscape']:.3f} s ({times[1]['diff_pct']:+.2f} %) with friction. The Python plant is "
           "rigid and sits on the flow limit; the Simscape oil column is compressible and the boom "
           "oscillates."] if len(times) == 2 else []),
        *([f"- Boom-up mechanical work, signed: {works[0]['python']:.1f} kJ against {works[0]['simscape']:.1f} kJ "
           f"({works[0]['diff_pct']:+.2f} %) without friction, {works[2]['python']:.1f} against "
           f"{works[2]['simscape']:.1f} kJ ({works[2]['diff_pct']:+.2f} %) with friction. Absolute work is "
           f"{works[1]['diff_pct']:+.2f} % and {works[3]['diff_pct']:+.2f} %: Simscape has negative-power phases "
           "in its oscillation, the Python plant has none on this move."] if len(works) == 4 else []),
        "",
        "## 2. Tuning on the real plant",
        "",
        f"Kp x Kd over the Lab 07 grid, Ki = {KI:g}, nominal real plant (`none`, `baseline`), the {n_orig} "
        "original scenarios (RNG seed 7). The unseen scenarios are not used. Cell: worst-case tip error / "
        "mean E_mech. Lab 07 criterion: lowest mean E_mech among the sets with worst-case error < 20 cm. "
        + (f"{n_ok} of {len(grid)} sets pass. " if clean else
           "**No set passes**, so the fallback applies: fewest moves over 20 cm, then lowest mean E_mech. ")
        + f"**Tuned: Kp = {tuned[0]:g}, Ki = {tuned[1]:g}, Kd = {tuned[2]:g}** (worst case "
        f"{cm(best['max_deg']):.1f} cm, {best['mean_E_mech_kJ']:.1f} kJ). Old gains for reference: "
        f"Kp = {OLD_GAINS[0]:g}, Ki = {OLD_GAINS[1]:g}, Kd = {OLD_GAINS[2]:g} (Lab 09f). Gains are in command "
        f"units per rad; one unit of command is {K_REF / 1e3:.1f} kN m at theta = 0 on the real plant and "
        "500 kN m on the placeholder plant.",
        "",
        *tune_table,
        "",
        "## 3. Results on the real plant",
        "",
        f"{n_new} unseen scenarios (RNG seed 123), 10 s per move, T_f = {T_F_S} s. **tuned** = "
        f"Kp {tuned[0]:g}, Kd {tuned[2]:g}; **old** = Kp {OLD_GAINS[0]:g}, Kd {OLD_GAINS[2]:g}. "
        "Status: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok.",
        "",
        *summary,
        "",
        "Columns: miss % = share of moves with final tip error over 20 cm. Reach 20 cm = mean time to "
        "first come within 20 cm, over the moves that do (a move that starts within 20 cm counts as 0 s); "
        "never reached = moves that never come within 20 cm. Flow limit % and torque limit % = share of "
        "the 0.005 s substeps with the rod speed on the pump flow limit, and with |u| = 1 while not on the "
        "flow limit. m_hat / m_tip = estimate over true tip mass in the last 2 s.",
        "Payload variants (tip mass, step at t = 5 s): `baseline` 3406 kg; `fill_half` 2263 -> 3406 kg; "
        "`fill` 1120 -> 3406 kg; `dump` 3406 -> 1120 kg.",
        f"330 bar could not hold the flow limit in {viol} substeps over all rows.",
        "",
        "### 3.1 One condition at a time",
        "",
        *table(SINGLES),
        "",
        "### 3.2 Combined cases (sensor noise, 5 % torque noise, J and b scaled)",
        "",
        *table(COMBINED),
        "",
        "## 4. Placeholder plant against real plant, row by row",
        "",
        *(comp or ["Not available."]),
        "",
        "## 5. Limits",
        "",
        "- Conditions: `tau_d` of 09c3 is not a condition here, the angle-dependent torque limit is part of "
        "the plant. `b_x0.5` and `b_x2` scale the cylinder friction (1.9 kN, 2.5 kN s/m, ESTIMATED), which "
        "is about 1/300 of the placeholder damping.",
        "- The plant keeps the torque-command abstraction of `BoomEnv`: no oil compressibility, no valve "
        "dynamics, no relief flow. Lab 09b2 showed that the lab08a Simscape model is not an energy "
        "reference either.",
        "- The retraction speed limit is a hard limit here (ASSUMED). The lab08a Simscape model cavitates "
        "on boom-down instead.",
        "- Tuning used the nominal plant only. Kd of the old gains and T_f were not retuned.",
        "- All geometry is ESTIMATED from a drawing.",
    ]
    print("\n" + "\n".join(lines[2:]))
    if write:
        with open(MD_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nCSV: {CSV_PATH}, {GRID_CSV}\nReport: {MD_PATH}")


if __name__ == "__main__":
    if REPORT_ONLY:
        report(read_csv(GRID_CSV), read_csv(CSV_PATH))
    else:
        grid, rows = simulate()
        if DRY_RUN:
            report(grid, rows, write=False)
            print(f"\nDRY RUN: {len(grid)} gain sets, {len(SCEN['orig20'])} original and"
                  f" {len(SCEN['new100'])} unseen scenarios. No file written.")
        else:
            write_csv(GRID_CSV, grid)
            write_csv(CSV_PATH, rows)
            report(read_csv(GRID_CSV), read_csv(CSV_PATH))
