import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from train import load

SAMPLE = 100000  # lower to 50000 if your Mac is slow


def fit(X, y):
    m = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
    m.fit(X, y)
    return m


# Start with the first model and its training data (Monday + Tuesday)
X1, y1 = load("data/*Monday*.csv", sample=SAMPLE)
X2, y2 = load("data/*Tuesday*.csv", sample=SAMPLE)
X_hist = pd.concat([X1, X2])
y_hist = pd.concat([y1, y2])
model = joblib.load("models/model_v1.joblib")

rows = []
for i, day in enumerate(["Wednesday", "Thursday", "Friday"], start=2):
    X, y = load(f"data/*{day}*.csv", sample=SAMPLE)

    # 30% = "recent labeled data" used to update, 70% = fresh test data
    X_new, X_eval, y_new, y_eval = train_test_split(
        X, y, test_size=0.7, random_state=42, stratify=y
    )

    # Before update: current model on the fresh test data
    old_pred = model.predict(X_eval)

    # After update: retrain with old data + the recent slice
    X_hist = pd.concat([X_hist, X_new])
    y_hist = pd.concat([y_hist, y_new])
    new_model = fit(X_hist, y_hist)
    new_pred = new_model.predict(X_eval)

    rows.append({
        "day": day,
        "attacks_in_test": int(y_eval.sum()),
        "recall_before": recall_score(y_eval, old_pred, zero_division=0),
        "recall_after": recall_score(y_eval, new_pred, zero_division=0),
        "precision_after": precision_score(y_eval, new_pred, zero_division=0),
        "false_alarms_before": int(((old_pred == 1) & (y_eval == 0)).sum()),
        "false_alarms_after": int(((new_pred == 1) & (y_eval == 0)).sum()),
    })
    print(rows[-1])

    joblib.dump(new_model, f"models/model_v{i}.joblib")
    model = new_model

results = pd.DataFrame(rows)
results.to_csv("docs/retrain_results.csv", index=False)

# Chart: before vs after each update
x = np.arange(len(results))
w = 0.35
fig, ax = plt.subplots()
b1 = ax.bar(x - w / 2, results["recall_before"], w, label="Before update", color="#e63946")
b2 = ax.bar(x + w / 2, results["recall_after"], w, label="After update", color="#2a9d8f")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{b.get_height():.2f}", ha="center")
ax.set_xticks(x)
ax.set_xticklabels(results["day"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Recall (share of attacks caught)")
ax.set_title("Retraining on a small slice of new data")
ax.legend()
plt.tight_layout()
plt.savefig("docs/retrain_chart.png", dpi=150)
print("Saved docs/retrain_results.csv and docs/retrain_chart.png")