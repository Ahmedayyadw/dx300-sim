"""Lab 09c2: stress test of the Lab 09c payload estimator.

No RL, no training. Reruns the 09c evaluation (payload-aware PID, same Lab 07 "pid" gains,
100 unseen scenarios, RNG seed 123, variants baseline and payload_x1.2) under:
  noise_0.2/0.5/1.0  Gaussian noise on observed theta and omega, as Lab 06b
                     (sigma_om = sigma_th / dt x 0.1, noise RNG seed 0)
  tau_noise_5pct     the torque the estimator reads is tau x (1 + 0.05 N(0, 1)), RNG seed 1;
                     the applied torque is unchanged
  J_x0.38, J_x1.5    true J scaled, estimator keeps J_nom
  b_x0.5, b_x2       true b scaled, estimator keeps b_nom
  tau_d              Lab 06 torque limit tau_max = 5e5 x |d(theta)| / |d(0)|; the estimator
                     reads the applied torque, the feed-forward still divides by the nominal 5e5
  all_n{s}_J{j}_b{b} everything together: noise level s, 5 % torque noise, J x j, b x b, tau_d
                     (12 cases: 3 noise levels x 2 J x 2 b)
  noTauD_n{s}_J{j}_b{b}  the same 12 cases without tau_d. Added because tau_d alone already
                     breaks the controller and hides the effect of everything else.
  ..._sub10          the J x0.38, b x2 cases again with the dynamics integrated in 10 substeps
                     per control step (command held). In those cases b dt / J = 1.97, at the
                     stability limit 2 of the env's explicit Euler step; the substep rerun tells
                     an integrator failure from a controller failure. Its energies use the
                     end-of-step omega and are approximate.
Filter sweep: T_f = 0.1, 0.2, 0.5, 1.0 s on both sets of 12 combined cases.
The old Lab 07 PID runs in every condition as the reference (it has no estimator).

m_hat / m: m_hat over the true mass, pooled over the last 2 s (40 steps) of all 100 episodes;
mean and std of those 4000 samples. Other metrics as in 09c; E_pump as in 09b
(dp_valve and eta_pump ASSUMED - not from manual, rl/09b_hyd_params.py).
Status per row: BREAKS if miss rate > 5 %, degraded if > 1 % (the 09c result), else ok.
b_dt_over_J is the env's explicit Euler factor b h / J (h = dt / substeps); the step is
unstable above 2.

Usage: python rl/09c2_stress.py [--report-only]
  --report-only   rebuild rl/09c2_stress.md from the existing CSV, no simulation
The report is always built from the CSV, so every number in it is a CSV value.
"""
import csv
import sys
import time
import numpy as np
from boom_env import BoomEnv
from lab09_common import HYD, NEW, PID, PayloadBoomEnv, cm, d, e_pump_J, rollout, tune_pid

CSV_PATH = "rl/09c2_stress.csv"
MD_PATH = "rl/09c2_stress.md"
REF_CSV = "rl/09c_payload_ff.csv"           # read only, to check the unstressed rerun matches 09c
VARIANT_NAMES = ["baseline", "payload_x1.2"]
T_F_DEFAULT = 0.2
T_F_SWEEP = [0.1, 0.2, 0.5, 1.0]
REPORT_ONLY = "--report-only" in sys.argv
TAIL_STEPS = 40                             # last 2 s
D0_M = abs(d(0.0))                          # about 0.898 m, as in 06


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


class SubstepBoomEnv(BoomEnv):
    """BoomEnv.step with the dynamics integrated in SUBSTEPS steps of dt / SUBSTEPS, command held
    (loop as in boom_env_robust.py). energy_J uses the end-of-step omega, as rollout() does."""
    SUBSTEPS = 10

    def step(self, action):
        u = float(np.clip(action[0], -1, 1))
        tau = u * (self.tau_max if self.tau_max_fn is None else self.tau_max_fn(self.th))
        h = self.dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            tau_g = self.m * 9.81 * self.Lc * np.cos(self.th)
            self.om += (tau - tau_g - self.b * self.om) / self.J * h
            self.th += self.om * h
            if self.th <= self.th_min or self.th >= self.th_max:      # hit end stop
                self.th, self.om = float(np.clip(self.th, self.th_min, self.th_max)), 0.0
        self.energy_J += abs(tau * self.om) * self.dt
        err = self.target - self.th
        reward = -(err ** 2 + 0.01 * self.om ** 2 + self.w_energy * u ** 2)
        self.k += 1
        return self._obs(), reward, False, self.k >= self.max_steps, {}


def make_env(variant, sigma=0.0, j=1.0, b=1.0, tau_d=False, substeps=1, **_):
    core = SubstepBoomEnv if substeps > 1 else BoomEnv
    bases = (([TauDMixin] if tau_d else []) + ([NoisyMixin] if sigma > 0 else [])
             + ([PayloadBoomEnv] if variant == "payload_x1.2" else []) + [core])
    env = type("StressEnv", tuple(bases), {})()
    env.J *= j
    env.b *= b
    if sigma > 0:
        env.sigma_th = np.radians(sigma)
        env.sigma_om = env.sigma_th / env.dt * 0.1
        env.noise_rng = np.random.default_rng(0)
    return env


class StressPID(PID):
    """Payload-aware PID of 09c; optional noise on the torque the estimator reads; logs m_hat."""

    def __init__(self, gains, t_f, tau_noise=0.0):
        self.tau_noise, self.tau_rng = tau_noise, np.random.default_rng(1)
        super().__init__(gains, payload_aware=True, t_f=t_f)

    def reset(self):
        super().reset()
        self.hist = []

    def update(self, obs, tau, th_min, th_max):
        if self.tau_noise > 0:
            tau = tau * (1.0 + self.tau_noise * self.tau_rng.normal())
        super().update(obs, tau, th_min, th_max)
        self.hist.append(self.m_hat)


SINGLES = [("none", {})] + [(f"noise_{s}", {"sigma": s}) for s in (0.2, 0.5, 1.0)] + [
    ("tau_noise_5pct", {"tau_noise": 0.05}),
    ("J_x0.38", {"j": 0.38}), ("J_x1.5", {"j": 1.5}),
    ("b_x0.5", {"b": 0.5}), ("b_x2", {"b": 2.0}),
    ("tau_d", {"tau_d": True}),
]
COMBINED = [(f"all_n{s}_J{j}_b{b}", {"sigma": s, "tau_noise": 0.05, "j": j, "b": b, "tau_d": True})
            for s in (0.2, 0.5, 1.0) for j in (0.38, 1.5) for b in (0.5, 2.0)]
NO_TAUD = [(name.replace("all_", "noTauD_"), {**cond, "tau_d": False}) for name, cond in COMBINED]
SUBSTEP = [(name + "_sub10", {**cond, "substeps": SubstepBoomEnv.SUBSTEPS})
           for name, cond in COMBINED + NO_TAUD if cond["j"] == 0.38 and cond["b"] == 2.0]


def run_case(group, name, cond, variant, label, t_f):
    env = make_env(variant, **cond)
    aware = label == "pid_payload_ff"
    ctrl = StressPID(gains, t_f, cond.get("tau_noise", 0.0)) if aware else PID(gains)
    errs, kj, pump, tail = [], [], [], []
    for sc in NEW:
        r = rollout(ctrl, env, *sc)
        errs.append(r["err_deg"]); kj.append(r["kJ"]); pump.append(e_pump_J(r) / 1000)
        if aware:
            tail.extend(np.array(ctrl.hist[-TAIL_STEPS:]) / env.m)
    errs = np.array(errs)
    return {"group": group, "condition": name, "sigma_th_deg": cond.get("sigma", 0.0),
            "tau_noise": cond.get("tau_noise", 0.0), "J_factor": cond.get("j", 1.0),
            "b_factor": cond.get("b", 1.0), "tau_d": int(cond.get("tau_d", False)),
            "substeps": cond.get("substeps", 1),
            "b_dt_over_J": env.b * env.dt / env.J / cond.get("substeps", 1),
            "t_f_s": t_f if aware else "",
            "kp": gains[0], "ki": gains[1], "kd": gains[2],
            "variant": variant, "w": label, "seed": 0,
            "mean_deg": errs.mean(), "max_deg": errs.max(), "mean_kJ": np.mean(kj),
            "mean_pump_kJ": np.mean(pump), "share_over_20cm": float((cm(errs) > 20).mean()),
            "m_hat_over_m_mean": np.mean(tail) if aware else "",
            "m_hat_over_m_std": np.std(tail) if aware else ""}


if not REPORT_ONLY:
    gains = tune_pid()
    t0 = time.time()
    rows = []
    for group, cases in [("single", SINGLES), ("combined", COMBINED),
                         ("combined_no_tau_d", NO_TAUD), ("substep_check", SUBSTEP)]:
        for name, cond in cases:
            for variant in VARIANT_NAMES:
                rows.append(run_case(group, name, cond, variant, "pid", T_F_DEFAULT))
                rows.append(run_case(group, name, cond, variant, "pid_payload_ff", T_F_DEFAULT))
    for t_f in T_F_SWEEP:
        if t_f == T_F_DEFAULT:
            continue                            # already in the combined rows
        for name, cond in COMBINED + NO_TAUD:
            for variant in VARIANT_NAMES:
                rows.append(run_case("tf_sweep", name, cond, variant, "pid_payload_ff", t_f))
    print(f"{len(rows)} rows x {len(NEW)} episodes in {time.time() - t0:.1f} s")

    # the unstressed rerun must equal 09c
    with open(REF_CSV) as f:
        ref = {(r["variant"], r["w"]): r for r in csv.DictReader(f)}
    for r in rows:
        if r["condition"] == "none":
            for key in ("mean_deg", "max_deg", "mean_kJ", "mean_pump_kJ", "share_over_20cm"):
                want = float(ref[(r["variant"], r["w"])][key])
                assert abs(r[key] - want) <= 1e-9 * max(1.0, abs(want)), f"{r['variant']} {r['w']} {key}"
    print(f"condition none: all four rows match {REF_CSV}")

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


def find(condition, variant, label="pid_payload_ff", t_f=T_F_DEFAULT):
    return next(r for r in rows if r["condition"] == condition and r["variant"] == variant
                and r["w"] == label and (label == "pid" or r["t_f_s"] == t_f))


def miss(r):
    return f"{r['share_over_20cm'] * 100:.0f}%"


def status(r):
    share = r["share_over_20cm"] * 100
    return "BREAKS" if share > 5 else "degraded" if share > 1 else "ok"


def mhat(r):
    return f"{r['m_hat_over_m_mean']:.3f} ± {r['m_hat_over_m_std']:.3f}"


def table(cases, with_euler):
    head = "| condition | variant | m_hat / m | E_mech kJ | E_pump kJ | miss >20 cm | old PID miss |"
    rule = "|---|---|---|---:|---:|---:|---:|"
    if with_euler:
        head, rule = head + " b·h/J |", rule + "---:|"
    out = [head + " status |", rule + "---|"]
    for name, _ in cases:
        for variant in VARIANT_NAMES:
            r, old = find(name, variant), find(name, variant, "pid")
            line = (f"| {name} | {variant} | {mhat(r)} | {r['mean_kJ']:.1f} | {r['mean_pump_kJ']:.1f} |"
                    f" {miss(r)} | {miss(old)} |")
            if with_euler:
                line += f" {r['b_dt_over_J']:.2f} |"
            out.append(line + f" {status(r)} |")
    return out


def verdict(cases):
    groups = {"BREAKS": [], "degraded": [], "ok": []}
    for name, _ in cases:
        worst = max((find(name, v) for v in VARIANT_NAMES), key=lambda r: r["share_over_20cm"])
        groups[status(worst)].append(f"`{name}` ({miss(worst)}, {worst['variant']})")
    return [f"- **Breaks** (miss > 5 %): {', '.join(groups['BREAKS']) or 'none'}",
            f"- **Degraded** (miss > 1 %): {', '.join(groups['degraded']) or 'none'}",
            f"- **Holds** (miss <= 1 %): {', '.join(groups['ok']) or 'none'}"]


def tf_tables(cases):
    valid = [name for name, _ in cases if find(name, "baseline")["b_dt_over_J"] < 1.5]
    out = [f"Mean over the {len(valid)} cases with b·dt/J < 1.5 (the J x0.38, b x2 cases are in "
           "section 5); \"worst miss\" is the worst of those cases.", "",
           "| T_f s | variant | m_hat / m mean | m_hat / m std | E_mech kJ | E_pump kJ | mean miss | worst miss |",
           "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for t_f in T_F_SWEEP:
        for variant in VARIANT_NAMES:
            rs = [find(name, variant, t_f=t_f) for name in valid]
            out.append(
                f"| {t_f} | {variant} | {np.mean([r['m_hat_over_m_mean'] for r in rs]):.3f} |"
                f" {np.mean([r['m_hat_over_m_std'] for r in rs]):.3f} |"
                f" {np.mean([r['mean_kJ'] for r in rs]):.1f} | {np.mean([r['mean_pump_kJ'] for r in rs]):.1f} |"
                f" {np.mean([r['share_over_20cm'] for r in rs]) * 100:.1f}% |"
                f" {max(r['share_over_20cm'] for r in rs) * 100:.0f}% |")
    out += ["", "Miss rate >20 cm per case and T_f (m_hat / m std in brackets):", "",
            "| condition | variant | " + " | ".join(f"T_f {t} s" for t in T_F_SWEEP) + " |",
            "|---|---|" + "---:|" * len(T_F_SWEEP)]
    for name, _ in cases:
        for variant in VARIANT_NAMES:
            cells = [find(name, variant, t_f=t) for t in T_F_SWEEP]
            out.append(f"| {name} | {variant} | " + " | ".join(
                f"{miss(c)} ({c['m_hat_over_m_std']:.3f})" for c in cells) + " |")
    return out


# --- verdict, every number read from the CSV rows ---
B, PL = "baseline", "payload_x1.2"


def m(name, variant, label="pid_payload_ff"):
    return miss(find(name, variant, label))


def pair(name, label="pid_payload_ff"):
    return f"{m(name, B, label)} / {m(name, PL, label)}"


def span(names, label="pid_payload_ff"):
    vals = [find(n, v, label)["share_over_20cm"] * 100 for n in names for v in VARIANT_NAMES]
    return f"{min(vals):.0f}%" if min(vals) == max(vals) else f"{min(vals):.0f}-{max(vals):.0f}%"


def pick(cases, **want):
    return [name for name, cond in cases if all(cond.get(k) == v for k, v in want.items())]


def tf_mean(names, variant, t_f):
    return f"{np.mean([find(n, variant, t_f=t_f)['share_over_20cm'] for n in names]) * 100:.1f}%"


def tf_pairs(names):
    return ", ".join(f"{tf_mean(names, B, t)} / {tf_mean(names, PL, t)} at {t} s" for t in T_F_SWEEP)


euler_all, euler_no = pick(COMBINED, j=0.38, b=2.0), pick(NO_TAUD, j=0.38, b=2.0)
valid_all = [n for n, _ in COMBINED if n not in euler_all]
valid_no = [n for n, _ in NO_TAUD if n not in euler_no]
sub_all = [n for n, _ in SUBSTEP if n.startswith("all_")]
sub_no = [n for n, _ in SUBSTEP if n.startswith("noTauD_")]
euler_rows = [find(n, v) for n in euler_all + euler_no for v in VARIANT_NAMES]
summary = [
    "## Verdict: which conditions break the controller",
    "",
    f"Miss rate >20 cm as baseline / payload_x1.2, T_f = {T_F_DEFAULT} s unless stated.",
    "",
    f"1. **`tau_d` breaks it: {pair('tau_d')}.** The estimator is right there (m_hat / m = "
    f"{mhat(find('tau_d', B))}) and the old PID misses the same {m('tau_d', B, 'pid')} on baseline. "
    "The fault is in the feed-forward: it divides by the nominal torque limit 5e5 N·m (`PID.act` in "
    "`rl/lab09_common.py`) while the true limit follows the moment arm. Under payload it still beats "
    f"the old PID ({m('tau_d', PL)} vs {m('tau_d', PL, 'pid')}).",
    f"2. **Sensor noise of 1.0° breaks it under payload ({m('noise_1.0', PL)}) and degrades baseline "
    f"({m('noise_1.0', B)}, at the 5 % line)**; old PID {pair('noise_1.0', 'pid')}. "
    f"0.5° degrades it ({pair('noise_0.5')}). 0.2° holds ({pair('noise_0.2')}). The m_hat / m std on "
    f"baseline grows from {find('noise_0.2', B)['m_hat_over_m_std']:.3f} at 0.2° to "
    f"{find('noise_1.0', B)['m_hat_over_m_std']:.3f} at 1.0°.",
    f"3. **5 % torque noise degrades it: {pair('tau_noise_5pct')}.**",
    f"4. **A wrong J holds:** {pair('J_x0.38')} at J x0.38 and {pair('J_x1.5')} at J x1.5.",
    f"5. **b x0.5 degrades it: {pair('b_x0.5')}.** On baseline the old PID is better there "
    f"({m('b_x0.5', B, 'pid')}). **b x2 holds: {pair('b_x2')}**, where the old PID misses "
    f"{pair('b_x2', 'pid')}.",
    f"6. **All together, as requested: every case breaks.** The nine cases with a valid integrator "
    f"step miss {span(valid_all)}, the `tau_d` level of point 1.",
    f"7. **All together without `tau_d`:** noise 1.0° with b x0.5 breaks "
    f"({span(pick(NO_TAUD, sigma=1.0, b=0.5))}). Noise 0.2° and 0.5° with b x0.5 are degraded "
    f"({span(pick(NO_TAUD, sigma=0.2, b=0.5) + pick(NO_TAUD, sigma=0.5, b=0.5))}). J x1.5 with b x2 "
    f"holds at 0.2° and 0.5° ({span(pick(NO_TAUD, sigma=0.2, j=1.5, b=2.0) + pick(NO_TAUD, sigma=0.5, j=1.5, b=2.0))}) "
    f"and is degraded at 1.0° ({span(pick(NO_TAUD, sigma=1.0, j=1.5, b=2.0))}).",
    f"8. **J x0.38 with b x2 is a simulator failure, not a controller failure.** b·dt/J = "
    f"{euler_rows[0]['b_dt_over_J']:.2f} against a stability limit of 2 for the env's Euler step. "
    f"The payload-aware PID misses {span(euler_all + euler_no)}, the old PID "
    f"{span(euler_all + euler_no, 'pid')}, and E_mech is {min(r['mean_kJ'] for r in euler_rows):.0f} kJ "
    f"or more. With {SubstepBoomEnv.SUBSTEPS} substeps the same cases miss {span(sub_no)} without "
    f"`tau_d` and {span(sub_all)} with it, in line with the other cases.",
    f"9. **Filter T_f:** without `tau_d`, mean miss over the nine valid cases is {tf_pairs(valid_no)}. "
    "Above 0.2 s the miss rate rises; 0.1 s and 0.2 s miss equally (m_hat / m std per T_f in 4b). "
    f"With `tau_d` no T_f helps: {tf_pairs(valid_all)}.",
    "",
]

first = rows[0]
lines = [
    "# Lab 09c2: stress test of the payload estimator",
    "",
    f"Payload-aware PID of Lab 09c, Kp={first['kp']:g} Ki={first['ki']:g} Kd={first['kd']:g}, "
    f"{len(NEW)} unseen scenarios (RNG seed 123), 10 s each. All numbers are from `rl/09c2_stress.csv`.",
    f"`m_hat / m` is mean ± std over the last 2 s of all episodes. E_pump uses dp_valve = "
    f"{HYD.DP_VALVE_PA / 1e5:.0f} bar and eta_pump = {HYD.ETA_PUMP}, both ASSUMED - not from manual.",
    "Status: BREAKS if miss rate > 5 %, degraded if > 1 % (the 09c result), else ok. "
    "\"old PID miss\" is the Lab 07 PID in the same condition.",
    "",
    *summary,
    f"## 1. One condition at a time (T_f = {T_F_DEFAULT} s)",
    "",
    *table(SINGLES, False),
    "",
    "Worst of the two variants per condition:",
    "",
    *verdict(SINGLES),
    "",
    f"## 2. All together (T_f = {T_F_DEFAULT} s)",
    "",
    "Each case: sensor noise at the given level, 5 % torque noise, J and b scaled, tau_d.",
    "",
    *table(COMBINED, True),
    "",
    "Worst of the two variants per case:",
    "",
    *verdict(COMBINED),
    "",
    f"## 3. All together without tau_d (T_f = {T_F_DEFAULT} s)",
    "",
    "Not in the request. Added because tau_d alone already breaks the controller (section 1) "
    "and hides the rest.",
    "",
    *table(NO_TAUD, True),
    "",
    "Worst of the two variants per case:",
    "",
    *verdict(NO_TAUD),
    "",
    "## 4. Filter sweep",
    "",
    "### 4a. Combined cases (with tau_d)",
    "",
    *tf_tables(COMBINED),
    "",
    "### 4b. Combined cases without tau_d",
    "",
    *tf_tables(NO_TAUD),
    "",
    "## 5. J x0.38 with b x2: integrator check",
    "",
    "In these cases b·dt/J = 1.97, at the stability limit 2 of the env's explicit Euler step. "
    f"Same cases with {SubstepBoomEnv.SUBSTEPS} substeps per control step (T_f = {T_F_DEFAULT} s). "
    "Energies of the substep rows use the end-of-step omega and are approximate.",
    "",
    *table([c for c in COMBINED + NO_TAUD if c[1]["j"] == 0.38 and c[1]["b"] == 2.0] + SUBSTEP, True),
]

with open(MD_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines[2:]))
print(f"\nCSV: {CSV_PATH}\nReport: {MD_PATH}")
