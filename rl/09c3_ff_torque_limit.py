"""Lab 09c3: feed-forward that copes with the angle-dependent torque limit (tau_d).

No RL, no training. Lab 09c2 found that tau_d breaks the payload-aware PID although m_hat is
correct: the feed-forward command m_hat g Lc cos(theta) / 5e5 assumes the nominal, constant
torque per unit command. With tau_d one unit of command gives 5e5 x |d(theta)| / |d(0)| N m.

The feed-forward torque stays tau_ff = m_hat g Lc cos(theta). Only its conversion to a command
changes, u_ff = tau_ff / k:
  pid_payload_ff   k = 5e5                              Lab 09c, unchanged (reference)
  pid_ff_geom      k = 5e5 x |d(theta)| / |d(0)|        torque limit known from the geometry
  pid_ff_gain      k = k_hat                            torque per unit command, estimated online
  pid_gain_sched   as pid_ff_gain, and the PID terms are scaled by 5e5 / k_hat too, so the whole
                   command is converted (added after the first run: the feed-forward fix alone
                   left 12-15 % miss under tau_d)
k_hat: the controller knows its own command u[k] and reads the applied torque tau[k] (the same
signal the mass estimator reads), so k_raw = tau[k] / u[k]. Low-pass filtered with the same
alpha as m_hat, started at 5e5, clipped to [0.2, 2] x 5e5, no update while |u[k]| < 0.05.
pid_ff_geom needs to know that the plant has the tau_d limit; the k_hat controllers do not.

Evaluation as in 09c2 (100 unseen scenarios, RNG seed 123, baseline and payload_x1.2,
T_f = 0.2 s): the ten single conditions and the nine combined cases with a valid integrator
step. The three J x0.38, b x2 cases are left out: b dt / J = 1.97 is at the stability limit of
the env's Euler step (09c2, section 5). Env mixins and conditions are copied from 09c2_stress.py.
The pid_payload_ff rows must equal rl/09c2_stress.csv (asserted).
Status per row: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok.
E_pump as in 09b (dp_valve and eta_pump ASSUMED - not from manual, rl/09b_hyd_params.py).

Usage: python rl/09c3_ff_torque_limit.py [--report-only]
  --report-only   rebuild rl/09c3_ff_torque_limit.md from the existing CSV, no simulation
"""
import csv
import sys
import time
import numpy as np
from boom_env import BoomEnv
from lab09_common import (G, HYD, LC, NEW, PID, TAU_MAX, PayloadBoomEnv, cm, d, e_pump_J, rollout,
                          tune_pid)

CSV_PATH = "rl/09c3_ff_torque_limit.csv"
MD_PATH = "rl/09c3_ff_torque_limit.md"
REF_CSV = "rl/09c2_stress.csv"              # read only, to check the 09c rows match 09c2
VARIANT_NAMES = ["baseline", "payload_x1.2"]
MODES = {"pid_payload_ff": "nominal", "pid_ff_geom": "geom", "pid_ff_gain": "gain",
         "pid_gain_sched": "gain_all"}
SHORT = {"pid_payload_ff": "09c", "pid_ff_geom": "geom", "pid_ff_gain": "gain",
         "pid_gain_sched": "sched"}
T_F_S = 0.2
TAIL_STEPS = 40                             # last 2 s
U_MIN = 0.05                                # no gain update below this command
REPORT_ONLY = "--report-only" in sys.argv
D0_M = abs(d(0.0))                          # about 0.898 m, as in 06


# --- env, copied from 09c2_stress.py ---
class NoisyMixin:
    """Gaussian noise on observed theta and omega. Copied from NoisyBoomEnv in 06b."""

    def _obs(self):
        th = self.th + self.noise_rng.normal(0.0, self.sigma_th)
        om = self.om + self.noise_rng.normal(0.0, self.sigma_om)
        return np.array([th, om, self.target - th], dtype=np.float32)


class TauDMixin:
    """Angle-dependent torque limit. Copied from TauDBoomEnv in 06."""

    def step(self, action):
        self.tau_max = 5e5 * abs(d(self.th)) / D0_M   # limit at the pre-step angle
        return super().step(action)


def make_env(variant, sigma=0.0, j=1.0, b=1.0, tau_d=False, **_):
    bases = (([TauDMixin] if tau_d else []) + ([NoisyMixin] if sigma > 0 else [])
             + ([PayloadBoomEnv] if variant == "payload_x1.2" else []) + [BoomEnv])
    env = type("StressEnv", tuple(bases), {})()
    env.J *= j
    env.b *= b
    if sigma > 0:
        env.sigma_th = np.radians(sigma)
        env.sigma_om = env.sigma_th / env.dt * 0.1
        env.noise_rng = np.random.default_rng(0)
    return env


SINGLES = [("none", {})] + [(f"noise_{s}", {"sigma": s}) for s in (0.2, 0.5, 1.0)] + [
    ("tau_noise_5pct", {"tau_noise": 0.05}),
    ("J_x0.38", {"j": 0.38}), ("J_x1.5", {"j": 1.5}),
    ("b_x0.5", {"b": 0.5}), ("b_x2", {"b": 2.0}),
    ("tau_d", {"tau_d": True}),
]
COMBINED = [(f"all_n{s}_J{j}_b{b}", {"sigma": s, "tau_noise": 0.05, "j": j, "b": b, "tau_d": True})
            for s in (0.2, 0.5, 1.0) for j in (0.38, 1.5) for b in (0.5, 2.0)
            if not (j == 0.38 and b == 2.0)]
CASES = SINGLES + COMBINED


# --- controller ---
class FFPID(PID):
    """Payload-aware PID of 09c with a selectable command conversion for the feed-forward."""

    def __init__(self, gains, mode, t_f=T_F_S, tau_noise=0.0):
        self.mode, self.tau_noise, self.tau_rng = mode, tau_noise, np.random.default_rng(1)
        super().__init__(gains, payload_aware=True, t_f=t_f)

    def reset(self):
        super().reset()
        self.k_hat, self.u_prev, self.hist = TAU_MAX, 0.0, []

    def act(self, obs):
        th, om, e = float(obs[0]), float(obs[1]), float(obs[2])   # what the PID observes
        if self.mode == "geom":
            k = TAU_MAX * abs(d(th)) / D0_M
        elif self.mode in ("gain", "gain_all"):
            k = self.k_hat
        else:
            k = TAU_MAX
        fb = self.kp * e + self.ki * self.integ + self.kd * (-om)
        if self.mode == "gain_all":
            fb *= TAU_MAX / k
        u_raw = fb + self.m_hat * G * LC / k * np.cos(th)
        if abs(u_raw) < 1.0:                                       # anti-windup
            self.integ += e * self.dt
        self.prev = (th, om)
        self.u_prev = float(np.clip(u_raw, -1.0, 1.0))
        return self.u_prev

    def update(self, obs, tau, th_min, th_max):
        if self.tau_noise > 0:                                     # one reading feeds both estimates
            tau = tau * (1.0 + self.tau_noise * self.tau_rng.normal())
        if self.mode in ("gain", "gain_all") and abs(self.u_prev) >= U_MIN:
            k_raw = tau / self.u_prev
            self.k_hat = float(np.clip(self.k_hat + self.alpha * (k_raw - self.k_hat),
                                       0.2 * TAU_MAX, 2.0 * TAU_MAX))
        super().update(obs, tau, th_min, th_max)
        self.hist.append(self.m_hat)


def run_case(group, name, cond, variant, label):
    env = make_env(variant, **cond)
    ctrl = FFPID(gains, MODES[label], tau_noise=cond.get("tau_noise", 0.0))
    errs, kj, pump, tail = [], [], [], []
    for sc in NEW:
        r = rollout(ctrl, env, *sc)
        errs.append(r["err_deg"]); kj.append(r["kJ"]); pump.append(e_pump_J(r) / 1000)
        tail.extend(np.array(ctrl.hist[-TAIL_STEPS:]) / env.m)
    errs = np.array(errs)
    return {"group": group, "condition": name, "sigma_th_deg": cond.get("sigma", 0.0),
            "tau_noise": cond.get("tau_noise", 0.0), "J_factor": cond.get("j", 1.0),
            "b_factor": cond.get("b", 1.0), "tau_d": int(cond.get("tau_d", False)),
            "t_f_s": T_F_S, "kp": gains[0], "ki": gains[1], "kd": gains[2],
            "variant": variant, "w": label, "seed": 0,
            "mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": np.mean(kj),
            "mean_pump_kJ": np.mean(pump), "share_over_20cm": float((cm(errs) > 20).mean()),
            "m_hat_over_m_mean": np.mean(tail), "m_hat_over_m_std": np.std(tail)}


if not REPORT_ONLY:
    gains = tune_pid()
    t0 = time.time()
    rows = [run_case(group, name, cond, variant, label)
            for group, cases in [("single", SINGLES), ("combined", COMBINED)]
            for name, cond in cases for variant in VARIANT_NAMES for label in MODES]
    print(f"{len(rows)} rows x {len(NEW)} episodes in {time.time() - t0:.1f} s")

    # the unchanged 09c controller must equal its 09c2 rows
    with open(REF_CSV) as f:
        ref = {(r["condition"], r["variant"]): r for r in csv.DictReader(f)
               if r["w"] == "pid_payload_ff" and float(r["t_f_s"]) == T_F_S}
    keys = ("mean_deg", "max_deg", "mean_kJ", "mean_pump_kJ", "share_over_20cm",
            "m_hat_over_m_mean", "m_hat_over_m_std")
    n = 0
    for r in rows:
        if r["w"] == "pid_payload_ff":
            for key in keys:
                want = float(ref[(r["condition"], r["variant"])][key])
                assert abs(r[key] - want) <= 1e-9 * max(1.0, abs(want)), f"{r['condition']} {key}"
            n += 1
    print(f"pid_payload_ff: all {n} rows match {REF_CSV}")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


# --- markdown report, built from the CSV ---
def parse(r):
    out = {}
    for k, v in r.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


with open(CSV_PATH) as f:
    rows = [parse(r) for r in csv.DictReader(f)]


def find(condition, variant, label):
    return next(r for r in rows if r["condition"] == condition and r["variant"] == variant
                and r["w"] == label)


def miss(r):
    return f"{r['share_over_20cm'] * 100:.0f}%"


def status(r):
    share = r["share_over_20cm"] * 100
    return "BREAKS" if share > 5 else "degraded" if share > 1 else "ok"


def trio(condition, variant, fmt):
    return " | ".join(fmt(find(condition, variant, label)) for label in MODES)


def table(cases):
    names = " | ".join(SHORT.values())
    out = [f"| condition | variant | miss: {names} | E_mech kJ: {names} | E_pump kJ: {names} |"
           " m_hat / m (sched) | status: " + " / ".join(SHORT.values()) + " |",
           "|---|---|" + "---:|" * (3 * len(MODES)) + "---|---|"]
    for name, _ in cases:
        for variant in VARIANT_NAMES:
            g = find(name, variant, "pid_gain_sched")
            out.append(f"| {name} | {variant} | {trio(name, variant, miss)} |"
                       f" {trio(name, variant, lambda r: format(r['mean_kJ'], '.1f'))} |"
                       f" {trio(name, variant, lambda r: format(r['mean_pump_kJ'], '.1f'))} |"
                       f" {g['m_hat_over_m_mean']:.3f} ± {g['m_hat_over_m_std']:.3f} |"
                       f" {' / '.join(status(find(name, variant, label)) for label in MODES)} |")
    return out


def count(cases, label):
    tally = {"BREAKS": 0, "degraded": 0, "ok": 0}
    for name, _ in cases:
        for variant in VARIANT_NAMES:
            tally[status(find(name, variant, label))] += 1
    return f"{tally['BREAKS']} break, {tally['degraded']} degraded, {tally['ok']} ok"


def span(cases, label):
    vals = [find(n, v, label)["share_over_20cm"] * 100 for n, _ in cases for v in VARIANT_NAMES]
    return f"{min(vals):.0f}%" if min(vals) == max(vals) else f"{min(vals):.0f}-{max(vals):.0f}%"


def pair(name, label):
    return " / ".join(miss(find(name, v, label)) for v in VARIANT_NAMES)


NO_TAUD_SINGLES = [c for c in SINGLES if not c[1].get("tau_d")]
first = rows[0]
lines = [
    "# Lab 09c3: feed-forward for the angle-dependent torque limit",
    "",
    f"Kp={first['kp']:g} Ki={first['ki']:g} Kd={first['kd']:g}, T_f = {T_F_S} s, {len(NEW)} unseen "
    "scenarios (RNG seed 123), 10 s each. All numbers are from `rl/09c3_ff_torque_limit.csv`.",
    "Feed-forward command u_ff = m_hat g Lc cos(theta) / k: **09c** k = 5e5 (unchanged), "
    "**geom** k = 5e5 x |d(theta)| / |d(0)| (limit known from geometry), "
    "**gain** k = k_hat (torque per unit command, estimated online from tau / u), "
    "**sched** = gain, with the PID terms scaled by 5e5 / k_hat as well.",
    f"E_pump uses dp_valve = {HYD.DP_VALVE_PA / 1e5:.0f} bar and eta_pump = {HYD.ETA_PUMP}, both "
    "ASSUMED - not from manual. Status: BREAKS if miss rate > 5 %, degraded if > 1 %, else ok.",
    "",
    "## Result",
    "",
    "Miss rate >20 cm as baseline / payload_x1.2.",
    "",
    "| conditions | " + " | ".join(SHORT.values()) + " |",
    "|---|" + "---:|" * len(MODES),
    "| `tau_d` alone | " + " | ".join(pair("tau_d", label) for label in MODES) + " |",
    "| no stress (`none`) | " + " | ".join(pair("none", label) for label in MODES) + " |",
    "| the nine single conditions without `tau_d` | "
    + " | ".join(span(NO_TAUD_SINGLES, label) for label in MODES) + " |",
    "| the nine combined cases (all with `tau_d`) | "
    + " | ".join(span(COMBINED, label) for label in MODES) + " |",
    f"| status over all {len(CASES) * 2} rows | " + " | ".join(count(CASES, label) for label in MODES) + " |",
    "",
    "## 1. One condition at a time",
    "",
    *table(SINGLES),
    "",
    "## 2. All together (sensor noise, 5 % torque noise, J and b scaled, tau_d)",
    "",
    "The three J x0.38, b x2 cases are left out (Euler step at its stability limit, 09c2 section 5).",
    "",
    *table(COMBINED),
]
with open(MD_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines[2:]))
print(f"\nCSV: {CSV_PATH}\nReport: {MD_PATH}")
