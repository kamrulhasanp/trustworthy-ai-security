import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Tool 4 baseline: a simple harmful-content detector on the MLMA dataset (English, French, Arabic).
# Label rule from the dataset authors: a tweet labeled only "normal" is not toxic. Everything else is toxic.
DATA = "data_text"
LANGS = ["en", "fr", "ar"]
NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}
MIN_GROUP = 20          # a target group needs at least this many toxic test tweets to be reported
SEED = 42


def load(lang):
    df = pd.read_csv(f"{DATA}/{lang}_dataset.csv")
    df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    df["toxic"] = (df["sentiment"].astype(str).str.strip().str.lower() != "normal").astype(int)
    df["group"] = df["group"].fillna("none").astype(str).str.strip().str.lower()
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def boot_range(y, pred, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        vals.append(fn(y[s], pred[s]))
    return np.percentile(vals, [2.5, 97.5])


def fit(texts, y):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return lambda t: clf.predict(hstack([word.transform(t), char.transform(t)]).tocsr())


data = {l: load(l) for l in LANGS}
splits = {}
for l, df in data.items():
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["toxic"])
    splits[l] = (tr.reset_index(drop=True), te.reset_index(drop=True))
    print(f"{NAMES[l]}: {len(df)} tweets, {df['toxic'].mean():.0%} toxic (train {len(tr)}, test {len(te)})")

# 1. Transfer matrix: train on one language (or all), test on each language's held-out test split
rows, matrix = [], {}
trainsets = {NAMES[l]: [l] for l in LANGS}
trainsets["All three"] = LANGS
for tname, tl in trainsets.items():
    train = pd.concat([splits[l][0] for l in tl], ignore_index=True)
    predict = fit(train["tweet"].astype(str), train["toxic"].values)
    for l in LANGS:
        te = splits[l][1]
        y = te["toxic"].values
        pred = predict(te["tweet"].astype(str))
        macro = f1_score(y, pred, average="macro")
        lo, hi = boot_range(y, pred, lambda a, b: f1_score(a, b, average="macro"))
        rows.append(dict(trained_on=tname, tested_on=NAMES[l], n_test=len(y),
                         precision_toxic=round(precision_score(y, pred, zero_division=0), 3),
                         recall_toxic=round(recall_score(y, pred, zero_division=0), 3),
                         macro_f1=round(macro, 3), macro_f1_low=round(lo, 3), macro_f1_high=round(hi, 3)))
        matrix[(tname, NAMES[l])] = macro
res = pd.DataFrame(rows)
res.to_csv("docs/tool4_results.csv", index=False)
mat = pd.DataFrame({NAMES[l]: [matrix[(t, NAMES[l])] for t in trainsets] for l in LANGS}, index=list(trainsets)).round(3)
print("\nMacro F1 (rows: trained on, columns: tested on)")
print(mat.to_string())
print("\nDetails with 95% ranges")
print(res.to_string(index=False))

# 2. Which target groups does the model miss most? (same-language model, toxic tweets only)
grp_rows = []
for l in LANGS:
    tr, te = splits[l]
    predict = fit(tr["tweet"].astype(str), tr["toxic"].values)
    te = te.assign(pred=predict(te["tweet"].astype(str)))
    tox = te[te["toxic"] == 1]
    for g, sub in tox.groupby("group"):
        if len(sub) >= MIN_GROUP and g != "none":
            k = int(sub["pred"].sum())
            lo, hi = wilson(k, len(sub))
            grp_rows.append(dict(language=NAMES[l], group=g, toxic_tweets=len(sub),
                                 recall=round(k / len(sub), 3), low=round(lo, 3), high=round(hi, 3)))
grp = pd.DataFrame(grp_rows)
grp.to_csv("docs/tool4_group_recall.csv", index=False)
print("\nShare of toxic tweets caught, by target group (same-language model)")
print(grp.to_string(index=False) if len(grp) else "No group had enough toxic test tweets.")

# 3. Chart: the transfer matrix
fig, ax = plt.subplots(figsize=(6.5, 4))
im = ax.imshow(mat.values, vmin=0.4, vmax=1.0, cmap="Greens")
ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels(mat.columns)
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(mat.index)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_xlabel("Tested on")
ax.set_ylabel("Trained on")
ax.set_title("Harmful content detector: macro F1 by language")
plt.tight_layout()
plt.savefig("docs/tool4_chart.png", dpi=150)
print("\nSaved docs/tool4_results.csv, docs/tool4_group_recall.csv, docs/tool4_chart.png")