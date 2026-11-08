import glob

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest

SAMPLE = 50000     # must match pipeline.py
SLICE = 0.2        # labeled share of each day, same as the main results
MAX_WEIGHT = 10    # rare attack types count up to 10 times more


def load_with_labels(pattern, sample):
    # Same steps as train.load, but also keeps the attack names
    files = glob.glob(pattern)
    df = pd.concat([pd.read_csv(f) for f in files])
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    names = df["Label"].values
    y = (df["Label"] != "BENIGN").astype(int)
    X = df.drop(columns=["Label"])
    return X, y, names


# 1. Rebuild the data the pipeline ended up with (Mon + Tue + labeled slices of Wed, Thu, Fri)
hist_X, hist_y, hist_n = [], [], []
for day in ["Monday", "Tuesday"]:
    X, y, n = load_with_labels(f"data/*{day}*.csv", SAMPLE)
    hist_X.append(X); hist_y.append(y); hist_n.append(n)

test_X, test_y, test_n = [], [], []
for day in ["Wednesday", "Thursday", "Friday"]:
    X, y, n = load_with_labels(f"data/*{day}*.csv", SAMPLE)
    X_rest, X_new, y_rest, y_new, n_rest, n_new = train_test_split(
        X, y, n, test_size=SLICE, random_state=42, stratify=y)
    Xv, Xt, yv, yt, nv, nt = train_test_split(
        X_rest, y_rest, n_rest, test_size=0.75, random_state=42, stratify=y_rest)
    hist_X.append(X_new); hist_y.append(y_new); hist_n.append(n_new)
    test_X.append(Xt); test_y.append(yt); test_n.append(nt)

X_train = pd.concat(hist_X)
y_train = pd.concat(hist_y).values
n_train = np.concatenate(hist_n)
X_test = pd.concat(test_X)
y_test = pd.concat(test_y).values
n_test = np.concatenate(test_n)

# 2. Weights: the rarer the attack type, the more each example counts
counts = pd.Series(n_train[y_train == 1]).value_counts()
biggest = counts.max()
weight_of = {t: min(MAX_WEIGHT, float(np.sqrt(biggest / c))) for t, c in counts.items()}
w = np.array([weight_of.get(t, 1.0) if yy == 1 else 1.0 for t, yy in zip(n_train, y_train)])
print("Weights given to attack types in training:")
for t, c in counts.sort_values().items():
    print(f"  {t}: {c} examples, weight {weight_of[t]:.1f}")

# 3. Train one model without weights and one with weights
def fit(weights=None):
    m = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
    m.fit(X_train, y_train, sample_weight=weights)
    return m

pred_plain = fit().predict(X_test)
pred_weighted = fit(w).predict(X_test)

# 4. Compare, attack type by attack type
df = pd.DataFrame({"type": n_test, "y": y_test, "plain": pred_plain, "weighted": pred_weighted})
att = df[df["y"] == 1]
out = (att.groupby("type")
          .agg(attacks=("y", "size"),
               caught_plain=("plain", "sum"),
               caught_weighted=("weighted", "sum"))
          .reset_index())
out["recall_plain"] = (out["caught_plain"] / out["attacks"]).round(3)
out["recall_weighted"] = (out["caught_weighted"] / out["attacks"]).round(3)
out = out.sort_values("attacks")
print("\n" + out[["type", "attacks", "recall_plain", "recall_weighted"]].to_string(index=False))

ben = df[df["y"] == 0]
print(f"\nOverall recall: plain {att['plain'].mean():.4f}, weighted {att['weighted'].mean():.4f}")
print(f"False alarms on normal traffic: plain {int(ben['plain'].sum())}, "
      f"weighted {int(ben['weighted'].sum())} (out of {len(ben)} normal rows)")
out.to_csv("docs/rare_attack_results.csv", index=False)
print("Saved docs/rare_attack_results.csv")


# 5. Is the difference more than luck? Compare the two models on the same attacks

def paired(sub, label):
    only_w = int(((sub["plain"] == 0) & (sub["weighted"] == 1)).sum())
    only_p = int(((sub["plain"] == 1) & (sub["weighted"] == 0)).sum())
    n = only_w + only_p
    p = binomtest(only_w, n, 0.5).pvalue if n else float("nan")
    print(f"{label}: caught only by weighted = {only_w}, only by plain = {only_p}, p-value = {p:.4f}")


paired(att, "All attacks")
bot_name = [t for t in att["type"].unique() if str(t).strip().lower() == "bot"]
if bot_name:
    paired(att[att["type"] == bot_name[0]], "Bot only")