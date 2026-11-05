import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from train import load

SAMPLE = 20000
SHIFT = 0.1    # a feature counts as "changed" if its KS distance is above this
ALERT = 0.30   # alert if more than 30% of features changed


def drift_score(ref, new):
    changed = 0
    for col in ref.columns:
        stat, _ = ks_2samp(ref[col], new[col])
        if stat > SHIFT:
            changed += 1
    return changed / len(ref.columns)


X1, _ = load("data/*Monday*.csv", sample=SAMPLE)
X2, _ = load("data/*Tuesday*.csv", sample=SAMPLE)
ref_all = pd.concat([X1, X2]).sample(frac=1, random_state=42).reset_index(drop=True)
half = len(ref_all) // 2
ref, same = ref_all.iloc[:half], ref_all.iloc[half:]

scores = {"Held-out\nMon+Tue": drift_score(ref, same)}
for day in ["Wednesday", "Thursday", "Friday"]:
    X, _ = load(f"data/*{day}*.csv", sample=SAMPLE)
    scores[day] = drift_score(ref, X)

for name, s in scores.items():
    status = "ALERT" if s > ALERT else "ok"
    print(f"{name.replace(chr(10), ' ')}: {s:.0%} of features changed -> {status}")

colors = ["#2a9d8f" if s <= ALERT else "#e63946" for s in scores.values()]
bars = plt.bar(list(scores.keys()), list(scores.values()), color=colors)
for b, s in zip(bars, scores.values()):
    plt.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.0%}", ha="center")
plt.axhline(ALERT, linestyle="--", color="gray")
plt.ylim(0, 1.1)
plt.ylabel("Share of features that changed")
plt.title("Label-free drift check")
plt.tight_layout()
plt.savefig("docs/drift_score_chart.png", dpi=150)
print("Saved docs/drift_score_chart.png")