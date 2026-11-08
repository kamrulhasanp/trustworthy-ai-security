import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tool1_drift"))
from train import load

SAMPLE = 50000      # must match pipeline.py
MIN_ATTACKS = 30    # a group needs this many attacks in the test data to count
FA_CAP = 0.005      # allowed false-alarm rate when picking a threshold (0.5%)

GROUPS = {
    "Web": {80, 443, 8080},
    "DNS": {53},
    "Remote login": {22, 23, 3389},
    "File and mail": {21, 25, 110, 139, 143, 445},
}


def group_of(port):
    for name, ports in GROUPS.items():
        if port in ports:
            return name
    return "Other"


def metrics(y, pred):
    attacks = int((y == 1).sum())
    normal = int((y == 0).sum())
    caught = int(((pred == 1) & (y == 1)).sum())
    false_alarms = int(((pred == 1) & (y == 0)).sum())
    return {
        "attacks": attacks,
        "normal": normal,
        "recall": caught / attacks if attacks else np.nan,
        "false_alarm_rate": false_alarms / normal if normal else np.nan,
    }


def pick_threshold(p, y):
    """Best alert threshold for one group: most attacks caught
    while keeping false alarms under FA_CAP."""
    if (y == 1).sum() < 10 or (y == 0).sum() < 10:
        return 0.5
    best_t, best_recall = 0.5, -1.0
    for t in np.arange(0.95, 0.04, -0.05):   # high to low, so ties keep the higher threshold
        pred = p >= t
        fa = (pred & (y == 0)).sum() / (y == 0).sum()
        rec = (pred & (y == 1)).sum() / (y == 1).sum()
        if fa <= FA_CAP and rec > best_recall + 1e-9:
            best_t, best_recall = float(t), rec
    return round(best_t, 2)


# 1. Rebuild the same held-out data that pipeline.py never trained on
val_X, val_y, test_X, test_y = [], [], [], []
for day in ["Wednesday", "Thursday", "Friday"]:
    X, y = load(f"data/*{day}*.csv", sample=SAMPLE)
    X_rest, X_new, y_rest, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    Xv, Xt, yv, yt = train_test_split(
        X_rest, y_rest, test_size=0.75, random_state=42, stratify=y_rest)
    val_X.append(Xv); val_y.append(yv)
    test_X.append(Xt); test_y.append(yt)

X_val = pd.concat(val_X); y_val = pd.concat(val_y).values
X_test = pd.concat(test_X); y_test = pd.concat(test_y).values

# 2. Predict and assign each row to a service-type group
model = joblib.load("models/pipeline_v4.joblib")
p_val = model.predict_proba(X_val)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]
g_val = X_val["Destination Port"].astype(int).map(group_of).values
g_test = X_test["Destination Port"].astype(int).map(group_of).values

# 3. Before: one threshold (0.5) for everyone
pred_before = (p_test >= 0.5).astype(int)

# 4. Fix: one threshold per group, picked on the validation data
thresholds = {g: pick_threshold(p_val[g_val == g], y_val[g_val == g])
              for g in sorted(set(g_val))}
pred_after = np.array(
    [int(p >= thresholds.get(g, 0.5)) for p, g in zip(p_test, g_test)])

# 5. Results table
rows = []
for g in sorted(set(g_test)):
    m = g_test == g
    b = metrics(y_test[m], pred_before[m])
    a = metrics(y_test[m], pred_after[m])
    rows.append({
        "group": g, "attacks": b["attacks"], "normal": b["normal"],
        "recall_before": b["recall"], "recall_after": a["recall"],
        "false_alarm_before": b["false_alarm_rate"],
        "false_alarm_after": a["false_alarm_rate"],
        "threshold": thresholds.get(g, 0.5),
        "counted": b["attacks"] >= MIN_ATTACKS,
    })
b = metrics(y_test, pred_before)
a = metrics(y_test, pred_after)
rows.append({
    "group": "ALL TRAFFIC", "attacks": b["attacks"], "normal": b["normal"],
    "recall_before": b["recall"], "recall_after": a["recall"],
    "false_alarm_before": b["false_alarm_rate"],
    "false_alarm_after": a["false_alarm_rate"],
    "threshold": np.nan, "counted": True,
})
res = pd.DataFrame(rows)
res.to_csv("docs/fairness_results.csv", index=False)
print(res.round(4).to_string(index=False))

# 6. The gap between the best and worst group
c = res[res["counted"] & (res["group"] != "ALL TRAFFIC")]
if len(c) >= 2:
    gap_b = c["recall_before"].max() - c["recall_before"].min()
    gap_a = c["recall_after"].max() - c["recall_after"].min()
    print(f"\nGap between best and worst group: before {gap_b:.3f}, after {gap_a:.3f}")
    print(f"Groups counted: {', '.join(c['group'])}")
else:
    print("\nFewer than 2 groups have enough attacks to compare.")

small = res[~res["counted"]]["group"].tolist()
if small:
    print("Too few attacks to count:", ", ".join(small))

# 7. Chart
if len(c) >= 1:
    x = np.arange(len(c))
    w = 0.38
    fig, ax = plt.subplots()
    b1 = ax.bar(x - w / 2, c["recall_before"], w, label="One threshold for all", color="#e63946")
    b2 = ax.bar(x + w / 2, c["recall_after"], w, label="Threshold per group", color="#2a9d8f")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(c["group"])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Recall (share of attacks caught)")
    ax.set_title("Attack detection by service type")
    ax.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig("docs/fairness_chart.png", dpi=150)
    print("Saved docs/fairness_results.csv and docs/fairness_chart.png")