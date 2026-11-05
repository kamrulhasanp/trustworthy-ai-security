import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from train import load

SAMPLE = 50000         # rows loaded per day (lower if your Mac is slow)
DRIFT_SAMPLE = 20000   # rows used for the drift check
SHIFT = 0.1            # a feature "changed" if its KS distance is above this
ALERT = 0.20           # alert if more than 20% of features changed
MIN_GAIN = 0.02        # new model must beat the old one by this much (F1)
MIN_RECALL = 0.8       # alert if the model catches fewer than 80% of attacks in the small labeled slice


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


# Starting point: Monday + Tuesday data and the first model
X1, y1 = load("data/*Monday*.csv", sample=SAMPLE)
X2, y2 = load("data/*Tuesday*.csv", sample=SAMPLE)
X_hist = pd.concat([X1, X2])
y_hist = pd.concat([y1, y2])
ref = X_hist.sample(DRIFT_SAMPLE, random_state=1)

static_model = joblib.load("models/model_v1.joblib")  # never updated
active = static_model                                  # the self-updating one
version = 1
rows = []

for day in ["Wednesday", "Thursday", "Friday"]:
    X, y = load(f"data/*{day}*.csv", sample=SAMPLE)

        # Step 1: label-free drift check
    d = drift_score(ref, X.sample(min(len(X), DRIFT_SAMPLE), random_state=1))
    drift_alert = d > ALERT

    # Split the day: 20% recent labeled data, 20% for the shadow test, 60% final test
    X_rest, X_new, y_rest, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.75, random_state=42, stratify=y_rest)

        # Step 1b: quick health check on the small labeled slice
    health = score(active, X_new, y_new)["recall"] if y_new.sum() > 0 else 1.0
    perf_alert = health < MIN_RECALL
    alert = drift_alert or perf_alert

    never = score(static_model, X_test, y_test)
    before = score(active, X_test, y_test)

    promoted = False
    if alert:
        # Step 2: train a challenger on old data + the recent labeled slice
        X_hist = pd.concat([X_hist, X_new])
        y_hist = pd.concat([y_hist, y_new])
        challenger = fit(X_hist, y_hist)

        # Step 3: shadow test, both models on the same validation data
        champ_f1 = score(active, X_val, y_val)["f1"]
        chall_f1 = score(challenger, X_val, y_val)["f1"]

        # Step 4: switch only if clearly better
        if chall_f1 > champ_f1 + MIN_GAIN:
            active = challenger
            version += 1
            joblib.dump(active, f"models/pipeline_v{version}.joblib")
            promoted = True
            ref = X_hist.sample(DRIFT_SAMPLE, random_state=1)  # new normal

    after = score(active, X_test, y_test)

    # Step 5: log the decision
    rows.append({
        "day": day,
        "drift_score": round(d, 3),
        "alert": alert,
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
    })
    print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv("docs/pipeline_results.csv", index=False)

# Chart: never updated vs self-updating
x = range(len(res))
w = 0.35
fig, ax = plt.subplots()
b1 = ax.bar([i - w / 2 for i in x], res["recall_never_updated"], w,
            label="Never updated", color="#e63946")
b2 = ax.bar([i + w / 2 for i in x], res["recall_pipeline"], w,
            label="Self-updating pipeline", color="#2a9d8f")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center")
ax.set_xticks(list(x))
ax.set_xticklabels(res["day"])
ax.set_ylim(0, 1.3)
ax.set_ylabel("Recall (share of attacks caught)")
ax.set_title("Self-updating pipeline vs a model that is never updated")
ax.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig("docs/pipeline_chart.png", dpi=150)
print("Saved docs/pipeline_results.csv and docs/pipeline_chart.png")