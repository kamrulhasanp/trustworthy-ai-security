import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SAMPLE = 50000   # must match pipeline.py

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


test_X, test_y, test_n = [], [], []
for day in ["Wednesday", "Thursday", "Friday"]:
    X, y, n = load_with_labels(f"data/*{day}*.csv", SAMPLE)
    X_rest, X_new, y_rest, y_new, n_rest, n_new = train_test_split(
        X, y, n, test_size=0.2, random_state=42, stratify=y)
    Xv, Xt, yv, yt, nv, nt = train_test_split(
        X_rest, y_rest, n_rest, test_size=0.75, random_state=42, stratify=y_rest)
    test_X.append(Xt); test_y.append(yt); test_n.append(nt)

X_test = pd.concat(test_X)
y_test = pd.concat(test_y).values
n_test = np.concatenate(test_n)

model = joblib.load("models/pipeline_v4.joblib")
pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
groups = X_test["Destination Port"].astype(int).map(group_of).values

df = pd.DataFrame({"group": groups, "attack_type": n_test, "y": y_test, "pred": pred})
att = df[df["y"] == 1]
out = (att.groupby(["group", "attack_type"])
          .agg(attacks=("y", "size"), missed=("pred", lambda s: int((s == 0).sum())))
          .reset_index())
out["missed_share"] = (out["missed"] / out["attacks"]).round(3)
print(out.to_string(index=False))
print("\nTotal attacks:", len(att), "| missed:", int((att["pred"] == 0).sum()))
out.to_csv("docs/attack_types_by_group.csv", index=False)
print("Saved docs/attack_types_by_group.csv")