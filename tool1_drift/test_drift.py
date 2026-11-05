import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score
from train import load

model = joblib.load("models/model_v1.joblib")

days = {
    "Tuesday\n(trained on)": "data/*Tuesday*.csv",
    "Wednesday": "data/*Wednesday*.csv",
    "Thursday": "data/*Thursday*.csv",
    "Friday": "data/*Friday*.csv",
}

names, recalls = [], []

for name, pattern in days.items():
    X, y = load(pattern, sample=100000)
    pred = model.predict(X)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    false_alarms = int(((pred == 1) & (y == 0)).sum())
    label = name.replace("\n", " ")
    print(f"{label}: attacks={int(y.sum())} F1={f1:.3f} Recall={rec:.3f} FalseAlarms={false_alarms}")
    names.append(name)
    recalls.append(rec)

colors = ["#2a9d8f"] + ["#e63946"] * (len(names) - 1)
bars = plt.bar(names, recalls, color=colors)
for bar, r in zip(bars, recalls):
    plt.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f"{r:.3f}", ha="center")
plt.title("Attack detection (recall): old attacks vs new attacks")
plt.ylabel("Recall (share of attacks caught)")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("docs/drift_chart.png", dpi=150)
print("Saved docs/drift_chart.png")