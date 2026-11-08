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

SAMPLE = 50000          # must match pipeline.py
MIN_ATTACKS = 30        # a group needs this many attacks in the TEST data to be compared
MIN_VAL_ATTACKS = 50    # a group needs this many attacks in the TUNING data to get its own threshold
FA_CAP = 0.005          # allowed false-alarm rate when picking a threshold (0.5%)

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


def wilson(k, n, z=1.96):
    """95% range for a share (caught k out of n). Small groups give wide ranges."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


def metrics(y, pred):
    attacks = int((y == 1).sum())
    normal = int((y == 0).sum())
    caught = int(((pred == 1) & (y == 1)).sum())
    false_alarms = int(((pred == 1) & (y == 0)).sum())
    lo, hi = wilson(caught, attacks)
    return {
        "attacks": attacks, "normal": normal, "caught": caught,
        "recall": caught / attacks if attacks else np.nan,
        "low": lo, "high": hi,
        "false_alarm_rate": false_alarms / normal if normal else np.nan,
    }


def pick_threshold(p, y):
    """Best alert level for one group: most attacks caught while keeping
    false alarms under FA_CAP. Groups with too little tuning data keep 0.5."""
    if (y == 1).sum() < MIN_VAL_ATTACKS or (y == 0).sum() < 10:
        return 0.5, False
    best_t, best_recall = 0.5, -1.0
    for t in np.arange(0.95, 0.04, -0.05):
        pred = p >= t
        fa = (pred & (y == 0)).sum() / (y == 0).sum()
        rec = (pred & (y == 1)).sum() / (y == 1).sum()
        if fa <= FA_CAP and rec > best_recall + 1e-9:
            best_t, best_recall = float(t), rec
    return round(best_t, 2), True


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

# 4. Fix: own threshold only for groups with enough tuning data
thresholds, tuned = {}, {}
for g in sorted(set(g_val)):
    thresholds[g], tuned[g] = pick_threshold(p_val[g_val == g], y_val[g_val == g])
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
        "recall_before": b["recall"], "before_low": b["low"], "before_high": b["high"],
        "recall_after": a["recall"], "after_low": a["low"], "after_high": a["high"],
        "false_alarm_before": b["false_alarm_rate"], "false_alarm_after": a["false_alarm_rate"],
        "threshold": thresholds.get(g, 0.5), "own_threshold": tuned.get(g, False),
        "counted": b["attacks"] >= MIN_ATTACKS,
    })
b = metrics(y_test, pred_before)
a = metrics(y_test, pred_after)
rows.append({
    "group": "ALL TRAFFIC", "attacks": b["attacks"], "normal": b["normal"],
    "recall_before": b["recall"], "before_low": b["low"], "before_high": b["high"],
    "recall_after": a["recall"], "after_low": a["low"], "after_high": a["high"],
    "false_alarm_before": b["false_alarm_rate"], "false_alarm_after": a["false_alarm_rate"],
    "threshold": np.nan, "own_threshold": False, "counted": True,
})
res = pd.DataFrame(rows)
res.to_csv("docs/fairness_results.csv", index=False)
show = ["group", "attacks", "recall_before", "before_low", "before_high",
        "recall_after", "false_alarm_before", "false_alarm_after", "threshold", "own_threshold"]
print(res[show].round(4).to_string(index=False))

# 6. The gap between the best and worst group, and whether it could be noise
c = res[res["counted"] & (res["group"] != "ALL TRAFFIC")]
if len(c) >= 2:
    gap_b = c["recall_before"].max() - c["recall_before"].min()
    gap_a = c["recall_after"].max() - c["recall_after"].min()
    best = c.loc[c["recall_before"].idxmax()]
    worst = c.loc[c["recall_before"].idxmin()]
    print(f"\nGap between best and worst group: before {gap_b:.3f}, after {gap_a:.3f}")
    print(f"Best: {best['group']} ({best['recall_before']:.3f}). "
          f"Worst: {worst['group']} ({worst['recall_before']:.3f}, "
          f"95% range {worst['before_low']:.3f} to {worst['before_high']:.3f})")
    if best["before_low"] > worst["before_high"]:
        print("The gap is larger than the noise: it is probably real.")
    else:
        print("The ranges overlap: the gap could be noise from small groups.")
    print(f"Groups compared: {', '.join(c['group'])}")
else:
    print("\nFewer than 2 groups have enough attacks to compare.")

small = res[~res["counted"]]["group"].tolist()
if small:
    print("Too few attacks to compare:", ", ".join(small))
own = [g for g, v in tuned.items() if v]
print("Groups with their own threshold:", ", ".join(own) if own else "none")

# 7. Chart with 95% ranges
if len(c) >= 1:
    x = np.arange(len(c))
    w = 0.38
    err_b = [c["recall_before"] - c["before_low"], c["before_high"] - c["recall_before"]]
    err_a = [c["recall_after"] - c["after_low"], c["after_high"] - c["recall_after"]]
    fig, ax = plt.subplots()
    b1 = ax.bar(x - w / 2, c["recall_before"], w, yerr=err_b, capsize=4,
                label="One threshold for all", color="#e63946")
    b2 = ax.bar(x + w / 2, c["recall_after"], w, yerr=err_a, capsize=4,
                label="Own threshold where enough data", color="#2a9d8f")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.04,
                    f"{bar.get_height():.2f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({n} attacks)" for g, n in zip(c["group"], c["attacks"])])
    ax.set_ylim(0.5, 1.2)
    ax.set_ylabel("Recall (share of attacks caught)")
    ax.set_title("Attack detection by service type (with 95% ranges)")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("docs/fairness_chart.png", dpi=150)
    print("Saved docs/fairness_results.csv and docs/fairness_chart.png")