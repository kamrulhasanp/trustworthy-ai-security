import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tool1_drift"))
from train import load

warnings.filterwarnings("ignore")

# Tool 5, part 1: attacks on the Tool 1 network detector (CIC-IDS2017).
#  A. Evasion: an attacker changes the traffic's features so that attack flows look different.
#  B. Poisoning: wrong labels are slipped into the training data (attack flows labeled as normal).
SAMPLE = 50000          # same as pipeline.py
SLICE = 0.2
SEED = 42
N_PROBE = 2000          # attack flows used for the single-feature evasion probe
HEALTH = 0.8            # Tool 1's health level: an update is triggered if recall on a labeled slice falls below this
rng = np.random.default_rng(0)

# 1. Rebuild the training and test data exactly as the pipeline did
X1, y1 = load("data/*Monday*.csv", sample=SAMPLE)
X2, y2 = load("data/*Tuesday*.csv", sample=SAMPLE)
tr_X, tr_y, te_X, te_y, te_day = [X1, X2], [y1, y2], [], [], []
for day in ["Wednesday", "Thursday", "Friday"]:
    X, y = load(f"data/*{day}*.csv", sample=SAMPLE)
    X_rest, X_new, y_rest, y_new = train_test_split(X, y, test_size=SLICE, random_state=SEED, stratify=y)
    Xv, Xt, yv, yt = train_test_split(X_rest, y_rest, test_size=0.75, random_state=SEED, stratify=y_rest)
    tr_X.append(X_new); tr_y.append(y_new)
    te_X.append(Xt); te_y.append(yt); te_day += [day] * len(Xt)
X_train, y_train = pd.concat(tr_X), pd.concat(tr_y).values
X_test, y_test, day_test = pd.concat(te_X), pd.concat(te_y).values, np.array(te_day)
cols = list(X_test.columns)
model = joblib.load("models/pipeline_v4.joblib")

A = X_test[(y_test == 1)]
A_day = day_test[(y_test == 1)]
caught = model.predict(A) == 1
A, A_day = A[caught], A_day[caught]          # only attack flows the clean model catches can be "evaded"
base = A.values.astype(float)
print(f"Attack flows in the test data: {int((y_test == 1).sum())}; caught by the clean model: {len(A)} "
      f"({len(A) / (y_test == 1).sum():.1%}). Every result below is the share of these that are STILL caught.")


def names(idx):
    return [cols[i] for i in idx]


def pick(match, exclude=()):
    return [i for i, c in enumerate(cols) if any(m in c.lower() for m in match) and not any(e in c.lower() for e in exclude)]


time_cols = pick(["iat", "duration"])
rate_cols = pick(["/s"])
len_cols = pick(["length", "packet size", "segment size"], exclude=["std", "variance"])
print(f"Features an attacker can slow down: {len(time_cols)} timing, {len(rate_cols)} rate. Features padding changes: {len(len_cols)} length or size.")


def still_caught(Xp):
    return float((model.predict(pd.DataFrame(Xp, columns=cols)) == 1).mean())


rows = []
for eps in [0.05, 0.1, 0.2, 0.5]:
    rows.append(("Random noise on every feature", f"{eps:.0%}", still_caught(base * (1 + eps * rng.standard_normal(base.shape)))))
for k in [2, 5, 10]:
    Xp = base.copy()
    Xp[:, time_cols] *= k
    Xp[:, rate_cols] /= k
    rows.append(("Slow-down (timing x k, rates / k)", f"x{k}", still_caught(Xp)))
for f in [0.1, 0.5, 1.0]:
    Xp = base.copy()
    Xp[:, len_cols] *= (1 + f)
    rows.append(("Padding (lengths and sizes grow)", f"+{f:.0%}", still_caught(Xp)))

# single-feature probe: can changing ONE of the 10 most important features hide the flow?
imp = model.feature_importances_
top = np.argsort(imp)[::-1][:10]
probe_idx = rng.choice(len(base), min(N_PROBE, len(base)), replace=False)
P, P_day = base[probe_idx], A_day[probe_idx]
evaded = np.zeros(len(P), dtype=bool)
by_feature = {}
for i in top:
    hit = np.zeros(len(P), dtype=bool)
    for fct in [0.0, 0.5, 2.0, 10.0]:
        Xp = P.copy()
        Xp[:, i] *= fct
        hit |= model.predict(pd.DataFrame(Xp, columns=cols)) == 0
    by_feature[cols[i]] = float(hit.mean())
    evaded |= hit
rows.append(("Change ONE top-10 feature (x0, x0.5, x2, x10)", "best of 40 tries", 1 - float(evaded.mean())))
res = pd.DataFrame(rows, columns=["attack", "strength", "still_caught"])
res["still_caught"] = res["still_caught"].round(3)
print("\nA. Evasion: share of attack flows still caught")
print(res.to_string(index=False))
print("\nSingle-feature evasion by day (share of flows hidden by changing one feature):")
for d in ["Wednesday", "Thursday", "Friday"]:
    m = P_day == d
    print(f"  {d}: {evaded[m].mean():.1%} of {int(m.sum())} flows")
print("Features that hide the most flows when changed alone:")
for n, v in sorted(by_feature.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {n}: {v:.1%}")

# 2. Poisoning: label some attack flows in the training data as normal, then retrain
def fit(X, y):
    return XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1).fit(X, y)


def score(m):
    pred = m.predict(X_test)
    return float(pred[y_test == 1].mean()), int(((pred == 1) & (y_test == 0)).sum())


prow = []
rec, fa = score(fit(X_train, y_train))
prow.append(dict(flipped="0% (clean)", recall=round(rec, 3), false_alarms=fa, health_check_fires=rec < HEALTH))
atk = np.flatnonzero(y_train == 1)
for p in [0.05, 0.1, 0.2, 0.4]:
    yp = y_train.copy()
    yp[rng.choice(atk, int(p * len(atk)), replace=False)] = 0
    rec, fa = score(fit(X_train, yp))
    prow.append(dict(flipped=f"{p:.0%}", recall=round(rec, 3), false_alarms=fa, health_check_fires=rec < HEALTH))
pres = pd.DataFrame(prow)
print("\nB. Poisoning: share of attack labels flipped to normal in the training data")
print(pres.to_string(index=False))

worst = min(res["still_caught"].iloc[4:].min(), res["still_caught"].iloc[-1])
print(f"\nRobustness score (lowest share still caught under attacker-controlled changes): {worst:.1%}")
res.to_csv("docs/tool5_network_evasion.csv", index=False)
pres.to_csv("docs/tool5_network_poisoning.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lab = [f"{a.split(' (')[0][:18]}\n{s}" for a, s in zip(res["attack"], res["strength"])]
axes[0].bar(range(len(res)), res["still_caught"], color="#2a9d8f")
axes[0].set_xticks(range(len(res)))
axes[0].set_xticklabels(lab, rotation=75, fontsize=6)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel("Attack flows still caught")
axes[0].set_title("Evasion", fontsize=10)
axes[1].plot(pres["flipped"], pres["recall"], marker="o", color="#e63946")
axes[1].axhline(HEALTH, ls="--", color="gray")
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("Attack labels flipped to normal in training")
axes[1].set_ylabel("Recall on test attacks")
axes[1].set_title("Poisoning (dashed line: Tool 1 health level)", fontsize=10)
plt.tight_layout()
plt.savefig("docs/tool5_network_chart.png", dpi=150)
print("Saved docs/tool5_network_evasion.csv, tool5_network_poisoning.csv, tool5_network_chart.png")