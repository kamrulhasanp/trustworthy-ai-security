import warnings

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Tool 4 baseline, version 2: honest scoring on the MLMA dataset (English, French, Arabic).
# Two tasks:
#   any_toxic: anything not purely "normal" (the dataset authors' rule)
#   severe:    the tweet is labeled hateful or abusive (my own stricter rule, not the authors')
# Every score is shown next to what a model that always answers "yes" would get.
DATA = "data_text"
LANGS = ["en", "fr", "ar"]
NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}
TARGETS = ["any_toxic", "severe"]
MIN_GROUP = 20
SEED = 42


def load(lang):
    df = pd.read_csv(f"{DATA}/{lang}_dataset.csv")
    df = df.dropna(subset=["tweet"]).drop_duplicates(subset=["tweet"]).reset_index(drop=True)
    s = df["sentiment"].astype(str).str.strip().str.lower()
    df["any_toxic"] = (s != "normal").astype(int)
    parts = s.str.split("_")
    df["severe"] = parts.apply(lambda p: int("hateful" in p or "abusive" in p))
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


def boot(y, score, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) < 2:
            continue
        vals.append(fn(y[s], score[s]))
    return np.percentile(vals, [2.5, 97.5])


def fit(texts, y):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
    return lambda t: clf.predict_proba(hstack([word.transform(t), char.transform(t)]).tocsr())[:, 1]


def trivial_macro_f1(y):
    p = y.mean()
    return (2 * p / (1 + p)) / 2        # always answering "yes": positive F1 is 2p/(1+p), negative F1 is 0


data = {l: load(l) for l in LANGS}
splits = {}
for l, df in data.items():
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["any_toxic"])
    splits[l] = (tr.reset_index(drop=True), te.reset_index(drop=True))
    print(f"{NAMES[l]}: {len(df)} tweets | any_toxic {df['any_toxic'].mean():.0%} | severe {df['severe'].mean():.0%}")

same_rows, xfer_rows, grp_rows = [], [], []
trainsets = {NAMES[l]: [l] for l in LANGS}
trainsets["All three"] = LANGS
auc_mats = {}
for target in TARGETS:
    auc_mat = pd.DataFrame(index=list(trainsets), columns=[NAMES[l] for l in LANGS], dtype=float)
    for tname, tl in trainsets.items():
        train = pd.concat([splits[l][0] for l in tl], ignore_index=True)
        score_fn = fit(train["tweet"].astype(str), train[target].values)
        for l in LANGS:
            te = splits[l][1]
            y = te[target].values
            sc = score_fn(te["tweet"].astype(str))
            pred = (sc >= 0.5).astype(int)
            auc = roc_auc_score(y, sc)
            auc_mat.loc[tname, NAMES[l]] = auc
            row = dict(target=target, trained_on=tname, tested_on=NAMES[l], n_test=len(y),
                       positive_share=round(y.mean(), 3),
                       macro_f1=round(f1_score(y, pred, average="macro"), 3),
                       always_yes_macro_f1=round(trivial_macro_f1(y), 3),
                       auc=round(auc, 3),
                       positives_caught=round(float(pred[y == 1].mean()), 3),
                       negatives_left_alone=round(float(1 - pred[y == 0].mean()), 3))
            lo, hi = boot(y, sc, roc_auc_score)
            row["auc_low"], row["auc_high"] = round(lo, 3), round(hi, 3)
            (same_rows if (len(tl) == 1 and tl[0] == l) else xfer_rows).append(row)
    auc_mats[target] = auc_mat

    # which target groups does the same-language model miss most?
    for l in LANGS:
        tr, te = splits[l]
        score_fn = fit(tr["tweet"].astype(str), tr[target].values)
        te = te.assign(pred=(score_fn(te["tweet"].astype(str)) >= 0.5).astype(int))
        for g, sub in te[te[target] == 1].groupby("group"):
            if len(sub) >= MIN_GROUP and g != "none":
                k = int(sub["pred"].sum())
                lo, hi = wilson(k, len(sub))
                grp_rows.append(dict(target=target, language=NAMES[l], group=g, positive_tweets=len(sub),
                                     caught=round(k / len(sub), 3), low=round(lo, 3), high=round(hi, 3)))

same, xfer, grp = pd.DataFrame(same_rows), pd.DataFrame(xfer_rows), pd.DataFrame(grp_rows)
pd.concat([same, xfer]).to_csv("docs/tool4_v2_results.csv", index=False)
grp.to_csv("docs/tool4_v2_group_recall.csv", index=False)

for target in TARGETS:
    print(f"\n=== Task: {target} ===")
    print("Same language (compare macro F1 with the 'always yes' column; AUC 0.5 means no skill):")
    print(same[same["target"] == target].drop(columns=["target", "trained_on"]).to_string(index=False))
    print("\nAUC when trained on one language (or all) and tested on another:")
    print(auc_mats[target].round(3).to_string())
    g = grp[grp["target"] == target]
    print("\nShare of positive tweets caught, by target group (same-language model):")
    print(g.drop(columns=["target"]).to_string(index=False) if len(g) else "No group had enough tweets.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, target in zip(axes, TARGETS):
    m = auc_mats[target]
    ax.imshow(m.values.astype(float), vmin=0.4, vmax=1.0, cmap="Greens")
    ax.set_xticks(range(3)); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title(f"AUC, task: {target}", fontsize=10)
    ax.set_xlabel("Tested on")
axes[0].set_ylabel("Trained on")
plt.tight_layout()
plt.savefig("docs/tool4_v2_chart.png", dpi=150)
print("\nSaved docs/tool4_v2_results.csv, docs/tool4_v2_group_recall.csv, docs/tool4_v2_chart.png")