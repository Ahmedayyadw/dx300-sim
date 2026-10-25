"""Step 2: detect hydraulic faults in synthetic DX300 boom pressure data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

rng = np.random.default_rng(42)
FS, CYCLE_S, N_CYCLES = 100, 20, 200        # 100 Hz, 20 s cycle, 200 cycles
CONTAMINATION = 0.15                         # expected fault share (try 0.10 / 0.15 / 0.20)

def dig_cycle(fault):
    t = np.arange(0, CYCLE_S, 1 / FS)
    # boom head-side pressure [bar]: lift loaded bucket, then hold during swing
    p = 40 + 180 * np.exp(-((t - 5) / 2.0) ** 2) + 120 * np.exp(-((t - 11) / 1.5) ** 2)
    if fault == "leak":                      # internal leakage -> low pressure
        p *= rng.uniform(0.6, 0.8)
    elif fault == "overload":                # relief valve hit
        k = rng.integers(3 * FS, 15 * FS)
        p[k:k + FS // 2] = 350
    return p + rng.normal(0, 4, t.size)      # sensor noise

def features(p):
    return {"mean": p.mean(), "std": p.std(), "max": p.max(),
            "p95": np.percentile(p, 95), "frac_above_300": (p > 300).mean()}

rows, labels, examples = [], [], {}
for _ in range(N_CYCLES):
    fault = rng.choice(["normal", "leak", "overload"], p=[0.9, 0.05, 0.05])
    p = dig_cycle(fault)
    rows.append(features(p)); labels.append(fault)
    examples.setdefault(fault, p)

X, y = pd.DataFrame(rows), np.array(labels)

model = IsolationForest(contamination=CONTAMINATION, random_state=0).fit(X)
pred = np.where(model.predict(X) == -1, "fault", "normal")
truth = np.where(y == "normal", "normal", "fault")

print(f"contamination = {CONTAMINATION}")
print(classification_report(truth, pred))
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")))

t = np.arange(0, CYCLE_S, 1 / FS)
for name, p in examples.items():
    plt.plot(t, p, label=name)
plt.xlabel("time [s]"); plt.ylabel("boom head pressure [bar]")
plt.title("Example dig cycles"); plt.legend(); plt.grid(alpha=0.3)
plt.savefig("ml/01_cycles.png", dpi=150)
print("Plot saved: ml/01_cycles.png")
