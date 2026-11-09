import glob
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Second dataset test: CSE-CIC-IDS2018.
# Same method and SAME settings as the CIC-IDS2017 run. Do not change them after seeing results.
# Optional: python tool1_drift/pipeline_2018.py 0.05   (share of each day that is labeled)
SLICE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
TAG = "" if SLICE == 0.2 else f"_slice{int(round(SLICE * 100))}"

DATA = "data2018"
INIT_SAMPLE = 100000   # rows from the first day, used to train the first model
SAMPLE = 300000        # rows loaded per later day (lower if your Mac is slow)
DRIFT_SAMPLE = 20000
SHIFT = 0.1            # same as 2017
ALERT = 0.20           # same as 2017
MIN_GAIN = 0.02        # same as 2017
MIN_RECALL = 0.8       # same as 2017

DAYS = [
    ("Thursday-15-02", "Thu 15-02"),
    ("Friday-16-02", "Fri 16-02"),
    ("Wednesday-21-02", "Wed 21-02"),
    ("Thursday-22-02", "Thu 22-02"),
    ("Friday-02-03", "Fri 02-03"),
]
DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
REF_COLS = None


def load(prefix, sample):
    """Load one day of CSE-CIC-IDS2018, clean it, and take a random sample."""
    global REF_COLS
    files = sorted(glob.glob(f"{DATA}/{prefix}*.csv"))
    if not files:
        sys.exit(f"No file starting with '{prefix}' found in {DATA}/")
    df = pd.read_csv(files[0], low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]                      # repeated header rows inside the file
    labels = df["Label"]
    X = df.drop(columns=[c for c in DROP + ["Label"] if c in df.columns])
    del df
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    X, labels = X[ok], labels[ok]
    if REF_COLS is None:
        REF_COLS = list(X.columns)
    else:
        missing = [c for c in REF_COLS if c not in X.columns]
        if missing:
            print(f"Warning: {prefix} is missing columns {missing}; filled with 0")
        X = X.reindex(columns=REF_COLS, fill_value=0)
    if len(X) > sample:
        X = X.sample(sample, random_state=42)
        labels = labels.loc[X.index]
    y = (labels != "Benign").astype(int)
    return X, y, labels.values


def drift_score(ref, new):
    changed = 0
    for col in ref.columns:
        stat, _ = ks_2samp(ref[col], new[col])
        if stat > SHIFT:
            changed += 1
    return changed / len(ref.columns)


def fit(X, y):
    m = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
    m.fit(X, y)
    return m


def score(model, X, y):
    pred = model.predict(X)
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_alarms": int(((pred == 1) & (y == 0)).sum()),
    }


print(f"Labeled slice per day: {SLICE:.0%}")

# 1. First model: Wednesday 14-02 (normal traffic plus FTP and SSH brute force)
X0, y0, n0 = load("Wednesday-14-02", INIT_SAMPLE)
print("First day attack types:", sorted(set(n0[y0.values == 1])))
X_tr, X_te, y_tr, y_te = train_test_split(X0, y0, test_size=0.2, random_state=42, stratify=y0)
static_model = fit(X_tr, y_tr)
print(classification_report(y_te, static_model.predict(X_te)))
joblib.dump(static_model, "models/model2018_v1.joblib")

X_hist, y_hist = X0, y0
ref = X_hist.sample(DRIFT_SAMPLE, random_state=1)
active = static_model
version = 1
rows = []

# 2. Later days, one at a time
for prefix, label in DAYS:
    X, y, names = load(prefix, SAMPLE)
    types = sorted(set(names[y.values == 1]))

    d = drift_score(ref, X.sample(min(len(X), DRIFT_SAMPLE), random_state=1))
    drift_alert = d > ALERT

    X_rest, X_new, y_rest, y_new = train_test_split(
        X, y, test_size=SLICE, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.75, random_state=42, stratify=y_rest)

    health = score(active, X_new, y_new)["recall"] if y_new.sum() > 0 else 1.0
    perf_alert = health < MIN_RECALL
    alert = drift_alert or perf_alert

    never = score(static_model, X_test, y_test)
    before = score(active, X_test, y_test)

    promoted = False
    if alert:
        X_hist = pd.concat([X_hist, X_new])
        y_hist = pd.concat([y_hist, y_new])
        challenger = fit(X_hist, y_hist)
        champ_f1 = score(active, X_val, y_val)["f1"]
        chall_f1 = score(challenger, X_val, y_val)["f1"]
        if chall_f1 > champ_f1 + MIN_GAIN:
            active = challenger
            version += 1
            joblib.dump(active, f"models/pipeline2018{TAG}_v{version}.joblib")
            promoted = True
            ref = X_hist.sample(DRIFT_SAMPLE, random_state=1)

    after = score(active, X_test, y_test)
    rows.append({
        "day": label,
        "attack_types": "; ".join(types),
        "drift_score": round(d, 3),
        "drift_alert": drift_alert,
        "perf_alert": perf_alert,
        "health_recall": round(health, 3),
        "promoted": promoted,
        "recall_never_updated": round(never["recall"], 3),
        "recall_pipeline_before": round(before["recall"], 3),
        "recall_pipeline": round(after["recall"], 3),
        "false_alarms_never_updated": never["false_alarms"],
        "false_alarms_pipeline": after["false_alarms"],
        "attacks_in_test": int(y_test.sum()),
        "attacks_in_labeled_slice": int(y_new.sum()),
    })
    print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(f"docs/pipeline2018_results{TAG}.csv", index=False)

x = np.arange(len(res))
w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.8))
b1 = ax.bar(x - w / 2, res["recall_never_updated"], w, label="Never updated", color="#e63946")
b2 = ax.bar(x + w / 2, res["recall_pipeline"], w, label="Self-updating pipeline", color="#2a9d8f")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(res["day"])
ax.set_ylim(0, 1.3)
ax.set_ylabel("Recall (share of attacks caught)")
ax.set_title(f"CSE-CIC-IDS2018: pipeline vs never updated ({SLICE:.0%} labeled per day)")
ax.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig(f"docs/pipeline2018_chart{TAG}.png", dpi=150)
print(f"Saved docs/pipeline2018_results{TAG}.csv and docs/pipeline2018_chart{TAG}.png")