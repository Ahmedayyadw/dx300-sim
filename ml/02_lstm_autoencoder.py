"""Step 3: LSTM autoencoder on raw boom pressure, trained on normal cycles only."""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

torch.manual_seed(0)
rng = np.random.default_rng(42)
FS, CYCLE_S, DS = 100, 20, 10              # 100 Hz, 20 s; downsample x10 -> 200 steps
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

def dig_cycle(fault):
    t = np.arange(0, CYCLE_S, 1 / FS)
    p = 40 + 180 * np.exp(-((t - 5) / 2.0) ** 2) + 120 * np.exp(-((t - 11) / 1.5) ** 2)
    if fault == "leak":
        p *= rng.uniform(0.6, 0.8)
    elif fault == "overload":
        k = rng.integers(3 * FS, 15 * FS)
        p[k:k + FS // 2] = 350
    return p + rng.normal(0, 4, t.size)

def make(n, probs):
    X, y = [], []
    for _ in range(n):
        f = rng.choice(["normal", "leak", "overload"], p=probs)
        X.append(dig_cycle(f)[::DS]); y.append(f)
    return np.array(X, dtype=np.float32), np.array(y)

X_train, _ = make(400, [1, 0, 0])          # healthy only
X_val, _ = make(100, [1, 0, 0])            # healthy only, sets the threshold
X_test, y_test = make(200, [0.9, 0.05, 0.05])

MU, SD = X_train.mean(), X_train.std()
def to_t(a):                                # (N, T) -> (N, T, 1) normalised tensor
    return torch.tensor((a - MU) / SD).unsqueeze(-1).to(dev)

class LSTMAE(nn.Module):
    def __init__(self, h=32, z=8):
        super().__init__()
        self.enc = nn.LSTM(1, h, batch_first=True)
        self.to_z = nn.Linear(h, z)         # bottleneck: whole cycle -> 8 numbers
        self.from_z = nn.Linear(z, h)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, 1)
    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = self.to_z(hn[-1])
        rep = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)

model = LSTMAE().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xtr = to_t(X_train)

t0 = time.time()
for epoch in range(1, 401):
    model.train()
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(0, len(Xtr), 32):
        b = Xtr[perm[i:i + 32]]
        loss = ((model(b) - b) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
print(f"training time: {time.time() - t0:.1f} s")

@torch.no_grad()
def score(a):
    model.eval(); x = to_t(a)
    return ((model(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy()

thr = np.percentile(score(X_val), 99)       # 99th percentile of healthy error
pred = np.where(score(X_test) > thr, "fault", "normal")
truth = np.where(y_test == "normal", "normal", "fault")

print(f"threshold = {thr:.4f}")
print(classification_report(truth, pred))
print(pd.crosstab(pd.Series(y_test, name="true"), pd.Series(pred, name="predicted")))
