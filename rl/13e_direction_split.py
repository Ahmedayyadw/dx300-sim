"""Lab 13e: boom-up / boom-down split of E_mech per controller. No simulation, no training.

Input (written by `python rl/12e_stats_eval.py --dir` and `python rl/13_eval.py --dir`):
  rl/12e_moves_dir.csv.gz   economical, balanced PID; SAC residual w = 0.03 / 0.1 on both bases
  rl/13_moves_dir.csv.gz    economical, balanced PID; SAC res_bal_w0.03; PPO, TD3, BC, IQL, CQL
Per move: E_mech = sum |tau_cyl om| h; E_up / E_down = the same term where om > 0 / om < 0;
W_pos / W_neg = the same term where tau_cyl om >= 0 / < 0 (motoring / braking).

Method
  share = sum over moves of E_down (or W_neg) / sum over moves of E_mech, per controller and seed;
  never the mean of per-move ratios. Per controller: mean and min-max over seeds.
  Controllers in both files (economical, balanced, res_bal_w0.03) are taken from rl/13_moves_dir.csv.gz;
  the two files must agree on them (asserted).
Checks (asserted)
  1  every controller and seed ran the same (condition, variant, scen) move set, no duplicates
  2  E_mech per move equals the published rl/12e_moves.csv.gz and rl/13_moves.csv.gz exactly
  3  E_up + E_down = E_mech and W_pos + W_neg = E_mech (rel 1e-9) per controller and seed
Mechanical energy only: no hydraulic correction.

Usage (repo root, .venv active):
  python rl/13e_direction_split.py
Writes rl/13e_direction_split.csv (one row per controller and seed) and rl/13e_direction_split.md.
"""

import collections
import csv
import gzip
import math

NEW = {"12e": "rl/12e_moves_dir.csv.gz", "13": "rl/13_moves_dir.csv.gz"}
PUB = {"12e": "rl/12e_moves.csv.gz", "13": "rl/13_moves.csv.gz"}
CSV_PATH = "rl/13e_direction_split.csv"
MD_PATH = "rl/13e_direction_split.md"
F = ("E_mech_kJ", "E_up_kJ", "E_down_kJ", "W_pos_kJ", "W_neg_kJ")
BOTH = ("economical", "balanced", "res_bal_w0.03")
ORDER = ("economical", "balanced", "res_bal_w0.03", "res_eco_w0.03", "res_bal_w0.1", "res_eco_w0.1",
         "bc_100k", "bc_300k", "bc_1M", "iql_100k", "iql_300k", "iql_1M", "cql_1M", "td3_100k",
         "ppo_100k", "ppo_1M")
NAME = {"economical": "economical PID", "balanced": "balanced PID",
        "res_bal_w0.03": "SAC w=0.03, balanced base", "res_eco_w0.03": "SAC w=0.03, economical base",
        "res_bal_w0.1": "SAC w=0.1, balanced base", "res_eco_w0.1": "SAC w=0.1, economical base",
        "bc_100k": "BC 100k", "bc_300k": "BC 300k", "bc_1M": "BC 1M", "iql_100k": "IQL 100k",
        "iql_300k": "IQL 300k", "iql_1M": "IQL 1M", "cql_1M": "CQL 1M", "td3_100k": "TD3 100k",
        "ppo_100k": "PPO 100k", "ppo_1M": "PPO 1M"}


def load(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def key(m):
    return (m["condition"], m["variant"], m["scen"])


def check_move_sets(new):
    sets, counts = {}, collections.Counter()
    for src, rows in new.items():
        for m in rows:
            u = (src, m["controller"], m["seed"])
            counts[u] += 1
            sets.setdefault(u, set()).add(key(m))
    ref = sets[("13", "economical", "")]
    assert all(s == ref for s in sets.values()), [u for u, s in sets.items() if s != ref]
    assert all(counts[u] == len(sets[u]) for u in counts)
    return len(counts), len(ref)


def check_published(new, pub):
    worst, n = 0.0, 0
    for src in new:
        p = {(m["controller"], m["seed"]) + key(m): float(m["E_mech_kJ"]) for m in pub[src]}
        assert len(p) == len(new[src]), (src, len(p), len(new[src]))
        for m in new[src]:
            worst = max(worst, abs(float(m["E_mech_kJ"]) - p[(m["controller"], m["seed"]) + key(m)]))
            n += 1
    assert worst == 0.0, worst
    return n, worst


def check_overlap(new):
    ov = collections.defaultdict(dict)
    for src, rows in new.items():
        for m in rows:
            if m["controller"] in BOTH:
                ov[(m["controller"], m["seed"]) + key(m)][src] = tuple(float(m[f]) for f in F)
    worst = max(abs(a - b) for v in ov.values() for a, b in zip(v["12e"], v["13"]))
    assert worst == 0.0, worst
    return len(ov), worst


def sums(new):
    units = {}
    for src in ("13", "12e"):
        for m in new[src]:
            if src == "12e" and m["controller"] in BOTH:
                continue
            u = units.setdefault((m["controller"], m["seed"]), {"file": NEW[src], "moves": 0, **{f: 0.0 for f in F}})
            for f in F:
                u[f] += float(m[f])
            u["moves"] += 1
    for u in units.values():
        assert math.isclose(u["E_up_kJ"] + u["E_down_kJ"], u["E_mech_kJ"], rel_tol=1e-9)
        assert math.isclose(u["W_pos_kJ"] + u["W_neg_kJ"], u["E_mech_kJ"], rel_tol=1e-9)
        u["down_pct"] = 100 * u["E_down_kJ"] / u["E_mech_kJ"]
        u["W_neg_pct"] = 100 * u["W_neg_kJ"] / u["E_mech_kJ"]
    assert {c for c, _ in units} == set(ORDER), {c for c, _ in units} ^ set(ORDER)
    return units


def summarise(units):
    out = {}
    for c in ORDER:
        us = [u for (cc, _), u in sorted(units.items()) if cc == c]
        mean = {k: sum(u[k] for u in us) / len(us) for k in F + ("down_pct", "W_neg_pct")}
        out[c] = {"seeds": len(us), "moves": us[0]["moves"], **mean,
                  "per_move": mean["E_mech_kJ"] / us[0]["moves"],
                  "down_rng": (min(u["down_pct"] for u in us), max(u["down_pct"] for u in us)),
                  "W_neg_rng": (min(u["W_neg_pct"] for u in us), max(u["W_neg_pct"] for u in us))}
    return out


def write_csv(units):
    cols = ("controller", "seed", "file", "moves") + F + ("down_pct", "W_neg_pct")
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in ORDER:
            for (cc, s), u in sorted(units.items()):
                if cc == c:
                    w.writerow({"controller": c, "seed": s, **{k: u[k] for k in cols[2:]}})


def spread(summ, k):
    hi = max(summ, key=lambda c: summ[c][k])
    lo = min(summ, key=lambda c: summ[c][k])
    return hi, lo, summ[hi][k] - summ[lo][k]


def report(summ, checks):
    rng = lambda s, k: "-" if s["seeds"] == 1 else f"{s[k][0]:.2f}-{s[k][1]:.2f}"
    w = ["# Lab 13e: boom-up / boom-down split of E_mech", "",
         "Source: `rl/12e_moves_dir.csv.gz`, `rl/13_moves_dir.csv.gz` (`--dir` runs of `rl/12e_stats_eval.py`"
         " and `rl/13_eval.py`, Lab 12e protocol: 100 unseen scenarios x 72 rows = 7,200 moves per controller"
         " and seed). Script: `rl/13e_direction_split.py`.", "",
         "- E_mech = sum |tau_cyl omega| h, delivered mechanical energy only; no hydraulic correction.",
         "- E_up / E_down: the same term where omega > 0 / omega < 0. W_pos / W_neg: where tau_cyl omega"
         " >= 0 / < 0 (motoring / braking). W_neg is not boom-down energy.",
         "- Shares are sums over all moves (sum E_down / sum E_mech), per controller and seed; never the"
         " mean of per-move ratios. Per controller: mean over seeds, range = min-max over seeds.",
         "- All values are simulation results: **ESTIMATED**.", "", "## Checks", ""]
    w += [f"- {c}" for c in checks]
    w += ["", "## Per controller", "",
          "| controller | seeds x moves | E_mech kJ per move | E_mech kJ total | E_up kJ total | E_down kJ total"
          " | boom-down % | boom-down % range | W_neg % | W_neg % range |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        s = summ[c]
        w.append(f"| {NAME[c]} | {s['seeds']} x {s['moves']} | {s['per_move']:.2f} | {s['E_mech_kJ']:,.0f} |"
                 f" {s['E_up_kJ']:,.0f} | {s['E_down_kJ']:,.0f} | {s['down_pct']:.2f} | {rng(s, 'down_rng')} |"
                 f" {s['W_neg_pct']:.2f} | {rng(s, 'W_neg_rng')} |")
    w += ["", "Totals are per run (mean over seeds). Per seed: `rl/13e_direction_split.csv`.", "", "## Spread", ""]
    lines = {}
    for k, name in (("down_pct", "Boom-down share"), ("W_neg_pct", "W_neg share")):
        hi, lo, d = spread(summ, k)
        lines[k] = (hi, lo, d)
        w.append(f"- {name}: highest {NAME[hi]} {summ[hi][k]:.2f} %, lowest {NAME[lo]} {summ[lo][k]:.2f} %,"
                 f" spread {d:.2f} percentage points.")
    seed_w = max((s["down_rng"][1] - s["down_rng"][0], c) for c, s in summ.items() if s["seeds"] > 1)
    pm = [s["per_move"] for s in summ.values()]
    w.append(f"- Widest seed range of the boom-down share: {NAME[seed_w[1]]}, {seed_w[0]:.2f} percentage points.")
    w.append(f"- E_mech per move across controllers: {min(pm):.1f} to {max(pm):.1f} kJ.")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(w) + "\n")
    print("\n".join(w))
    print(f"\nwrote {MD_PATH}, {CSV_PATH}")


def main():
    new = {s: load(p) for s, p in NEW.items()}
    pub = {s: load(p) for s, p in PUB.items()}
    n_units, n_set = check_move_sets(new)
    n_pub, d_pub = check_published(new, pub)
    n_ov, d_ov = check_overlap(new)
    units = sums(new)
    checks = [f"Move set: {n_units} controller-seed runs (both files), each the same {n_set} moves, no duplicates.",
              f"E_mech per move vs the published `rl/12e_moves.csv.gz` and `rl/13_moves.csv.gz`: {n_pub} moves,"
              f" largest difference {d_pub} kJ.",
              f"Controllers in both files ({', '.join(BOTH)}): {n_ov} moves, all five energy columns, largest"
              f" difference {d_ov} kJ.",
              "E_up + E_down = E_mech and W_pos + W_neg = E_mech for every controller and seed (rel 1e-9)."]
    write_csv(units)
    report(summarise(units), checks)


if __name__ == "__main__":
    main()
